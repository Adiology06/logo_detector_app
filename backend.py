# backend.py

import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

import cv2
import easyocr
from rapidfuzz import fuzz


# ------------------------------
# 1. Paths + model + embeddings
# ------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGO_DIR = os.path.join(BASE_DIR, "Logos_Images")

EMBED_PATH = os.path.join(DATA_DIR, "logo_embeddings_efficientnet_b0.npy")
META_PATH = os.path.join(DATA_DIR, "logo_metadata.csv")

IMG_SIZE = 224

# EfficientNet feature extractor
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
base_model.trainable = False

# Load precomputed embeddings + metadata
embeddings = np.load(EMBED_PATH)
metadata_df = pd.read_csv(META_PATH)

# Ensure file_path points to the local "Logos Images" folder
if "file_path" in metadata_df.columns:
    metadata_df["file_path"] = metadata_df["file_path"].apply(
        lambda p: os.path.join(LOGO_DIR, os.path.basename(str(p)))
    )

# OCR reader
ocr_reader = easyocr.Reader(["en"], gpu=False)


# ------------------------------
# 2. Helper functions
# ------------------------------

def load_and_preprocess_img(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def embed_logo_image(img_path: str) -> np.ndarray:
    x = load_and_preprocess_img(img_path)
    features = base_model.predict(x, verbose=0)[0]
    # L2-normalise
    features = features / (np.linalg.norm(features) + 1e-8)
    return features


def get_average_hsv(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mean = img.reshape(-1, 3).mean(axis=0)
    return hsv_mean


def color_similarity(hsv1, hsv2) -> float:
    """Return 0–100 color similarity."""
    if hsv1 is None or hsv2 is None:
        return 0.0
    dist = np.linalg.norm(hsv1 - hsv2)
    max_dist = np.linalg.norm([180, 255, 255])
    sim = 100.0 * (1.0 - dist / max_dist)
    return float(max(sim, 0.0))


def extract_logo_text(img_path: str) -> str:
    """Run OCR and clean text."""
    results = ocr_reader.readtext(img_path, detail=0)
    if not results:
        return ""
    raw_text = " ".join(results)
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", raw_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def text_similarity_score(text: str, brand_name: str) -> float:
    """Return 0–1 similarity between detected text and brand name."""
    if not text or not brand_name:
        return 0.0
    brand_name = re.sub(r"[^A-Za-z0-9 ]+", " ", brand_name.lower()).strip()
    score = fuzz.token_set_ratio(text, brand_name) / 100.0
    return float(score)


# ------------------------------
# 3. Main logic function
# ------------------------------

def find_similar_logos(
    query_img_path,
    top_k=3,
    known_threshold=0.60,
    strong_match_threshold=0.95,
    text_strong_threshold=0.70,
):
    """
    Improved logic:

    1. Compute shape similarities (EfficientNet embeddings).
    2. Run OCR and compute text similarity with *every* brand name.
    3. If any brand's text similarity is very high, give it priority.
    4. Otherwise, use a combined score (shape + text) to choose best brand.
    5. NO_MATCH only if both shape and text are low for ALL brands.
    """

    # ---------- 1) Shape similarity ----------
    query_emb = embed_logo_image(query_img_path).reshape(1, -1)
    shape_sims = cosine_similarity(query_emb, embeddings)[0]  # (N,) in [0, 1]

    shape_best_idx = int(np.argmax(shape_sims))
    shape_best_sim = float(shape_sims[shape_best_idx])

    # ---------- 2) OCR text + text similarity ----------
    detected_text = extract_logo_text(query_img_path)

    if detected_text:
        text_sims = np.array([
            text_similarity_score(detected_text, brand)
            for brand in metadata_df["brand_name_raw"]
        ])  # (N,) in [0, 1]
    else:
        text_sims = np.zeros(len(metadata_df), dtype=float)

    text_best_idx = int(np.argmax(text_sims))
    text_best_sim = float(text_sims[text_best_idx])

    # ---------- 3) Choose best index ----------
    # 3a. If OCR text is extremely confident (≈ exact match), force that brand
    if text_best_sim >= 0.99:
        best_idx = text_best_idx

    else:
        # 3b. Otherwise use combined shape + text score
        w_shape = 0.65
        w_text = 0.35

        combined_scores = w_shape * shape_sims + w_text * text_sims
        best_idx = int(np.argmax(combined_scores))

    # Now read values for the chosen best_idx
    best_shape_sim = float(shape_sims[best_idx])
    best_text_sim = float(text_sims[best_idx])
    best_brand = metadata_df.loc[best_idx, "brand_name_raw"]

    # Recompute combined_scores for top-k listing (if not already defined)
    if "combined_scores" not in locals():
        w_shape = 0.65
        w_text = 0.35
        combined_scores = w_shape * shape_sims + w_text * text_sims

    # ---------- 4) Color similarity for best brand ----------
    query_hsv = get_average_hsv(query_img_path)
    best_logo_path = metadata_df.loc[best_idx, "file_path"]
    best_logo_hsv = get_average_hsv(best_logo_path)
    color_sim = color_similarity(query_hsv, best_logo_hsv)  # 0–100

    # ---------- 5) Overall confidence (0–1) ----------
    color_norm = color_sim / 100.0
    w_color = 0.15  # small weight for color

    overall_conf = (
        w_shape * best_shape_sim +
        w_text * best_text_sim +
        w_color * color_norm
    )

    # ---------- 6) Decide match category ----------
    # Use *best over all brands* for text when checking NO_MATCH
    if (shape_best_sim < known_threshold) and (text_best_sim < text_strong_threshold):
        match_status = "NO_MATCH"
        final_brand = None
        confidence_label = "LOW"
    elif (best_shape_sim >= strong_match_threshold) and (best_text_sim >= text_strong_threshold):
        match_status = "STRONG_MATCH"
        final_brand = best_brand
        confidence_label = "HIGH"
    else:
        match_status = "WEAK_MATCH"
        final_brand = best_brand
        confidence_label = "MEDIUM"

    # ---------- 7) Build top-k list (by combined score) ----------
    top_idx = np.argsort(combined_scores)[::-1][:top_k]
    top_matches = []
    for rank, idx in enumerate(top_idx, start=1):
        top_matches.append({
            "rank": rank,
            "brand_name": metadata_df.loc[idx, "brand_name_raw"],
            "file_path": metadata_df.loc[idx, "file_path"],
            "shape_similarity": float(shape_sims[idx]),
            "text_similarity": float(text_sims[idx]),
            "combined_score": float(combined_scores[idx]),
        })

    # ---------- 8) Return for Streamlit ----------
    return {
        "match_status": match_status,
        "final_brand": final_brand,
        "confidence": confidence_label,
        "shape_similarity_best": best_shape_sim,
        "color_similarity_best": float(color_sim),
        "text_detected": detected_text,          # always returned, even for NO_MATCH
        "text_similarity_best": best_text_sim,
        "overall_confidence": float(overall_conf),
        "top_k_matches": top_matches,
    }
