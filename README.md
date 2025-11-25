Logo Similarity & Authenticity Checker

Compare uploaded logos against a verified brand database using a lightweight image-embedding model (EfficientNetB0).
Features:

Predict best-matching brand.

Compute shape similarity, color similarity, and text similarity (OCR + fuzzy text matching).

Estimate whether the uploaded logo is real (likely from verified brand) or fake (low similarity / suspicious).

Simple Streamlit UI for upload, thresholds, and results.

Shape Similarity
Compute cosine similarity between the query embedding and all stored embeddings.
Use the top-k nearest neighbours (sorted by cosine) as candidate logos.

OCR Text Extraction
Library: EasyOCR for detecting text in arbitrary images (handles scene text).
Clean OCR output (regex), normalize (lowercase).
Compare detected text to brand names using RapidFuzz (token_set_ratio) to get a 0–1 text similarity score.
OCR priority rule: If OCR text matches a brand extremely well (e.g., ≥ 0.90), consider text a strong indicator and prioritize it.

Color Similarity
Convert images to HSV and compute average H, S, V.
Color similarity computed as 1 − (euclidean_distance / max_possible_distance) → percentage (0–100).
Use color as a weaker signal (helps disambiguate visually-similar shapes).
