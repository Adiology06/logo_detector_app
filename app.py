# app.py

import streamlit as st
from PIL import Image
import tempfile
import os

from backend import find_similar_logos  # backend logic


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Logo Similarity & Authenticity Checker",
    page_icon="🔍",
    layout="wide"
)


# ---------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1F4E79;
    }
    .sub-title {
        font-size: 16px;
        color: #555;
    }
    .match-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        background: #F5F7FB;
        border: 1px solid #D0D7E2;
        color: #2C3E50 !important;      /* make ALL text on the card dark */
    }
    .no-match-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        background: #FFF5F5;
        border: 1px solid #F5B7B1;
        color: #7B241C !important;
    }
    .strong-match {
        border-left: 6px solid #2ECC71;
    }
    .weak-match {
        border-left: 6px solid #F4D03F;
    }
    .info-label {
        font-weight: 600;
        color: #1B2631;
    }
    .value-text {
        font-weight: 500;
        color: #2C3E50;
    }
    .small-text {
        font-size: 12px;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")

upload_option = st.sidebar.radio(
    "Choose input method:",
    ("Upload image", "Scan using camera")
)

uploaded_file = None
captured_image = None

if upload_option == "Upload image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload logo image (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"]
    )
else:
    captured_image = st.sidebar.camera_input("Capture logo image")

st.sidebar.markdown("---")
st.sidebar.markdown("### Threshold Controls")

known_threshold = st.sidebar.slider(
    "Known match threshold (shape)",
    0.30, 0.90, 0.60, 0.01
)

strong_match_threshold = st.sidebar.slider(
    "Strong match (shape)",
    0.80, 1.00, 0.95, 0.01
)

text_strong_threshold = st.sidebar.slider(
    "Strong OCR text match",
    0.50, 1.00, 0.70, 0.01
)

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("🔍 Analyze Logo")


# ---------- PAGE HEADER ----------
st.markdown(
    '<div class="main-title">Logo Similarity &amp; Authenticity Checker</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Compare logos with your verified brand database using AI.</div><br>',
    unsafe_allow_html=True
)


# ---------- TWO COLUMN LAYOUT ----------
col_left, col_right = st.columns([1, 1])


# ---------- LEFT COLUMN – INPUT ----------
with col_left:
    st.header("1️⃣ Input Logo")

    final_input_image = None

    if uploaded_file:
        final_input_image = Image.open(uploaded_file).convert("RGB")
        # smaller, fixed width for better layout
        st.image(final_input_image, caption="Uploaded Logo", width=320)

    elif captured_image:
        final_input_image = Image.open(captured_image).convert("RGB")
        st.image(final_input_image, caption="Captured Logo", width=320)

    else:
        st.info("Upload an image or use camera from the sidebar.")


# ---------- RIGHT COLUMN – RESULTS ----------
with col_right:
    st.header("2️⃣ Analysis Result")

    if analyze_btn:

        if final_input_image is None:
            st.warning("Please upload or scan a logo first.")
        else:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                temp_path = tmp.name
                final_input_image.save(temp_path)

            # Backend similarity check
            try:
                result = find_similar_logos(
                    query_img_path=temp_path,
                    top_k=3,
                    known_threshold=known_threshold,
                    strong_match_threshold=strong_match_threshold,
                    text_strong_threshold=text_strong_threshold
                )
            except Exception as e:
                st.error(f"Processing error: {e}")
                result = None

            # Delete temp file on disk
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # ---- SHOW RESULTS ----
            if result:
                match_status = result["match_status"]          # STRONG_MATCH / WEAK_MATCH / NO_MATCH
                final_brand = result["final_brand"]            # brand name or None
                confidence = result["confidence"]              # HIGH / MEDIUM / LOW
                shape_sim = result["shape_similarity_best"] * 100.0
                color_sim = result["color_similarity_best"]
                text_detected = result["text_detected"]
                text_sim = result["text_similarity_best"] * 100.0
                top_matches = result["top_k_matches"]

                # ---------- COMBINED SCORE ----------
                shape_norm = shape_sim / 100.0
                color_norm = color_sim / 100.0
                text_norm = text_sim / 100.0

                # You can tweak these weights
                combined_score = 0.6 * shape_norm + 0.25 * color_norm + 0.15 * text_norm
                combined_pct = combined_score * 100.0

                # ---------- NO MATCH ----------
                if match_status == "NO_MATCH":
                    st.markdown(
                        """
                        <div class="no-match-card">
                            <span class="info-label">❌ No Data Match Found</span><br>
                            The uploaded logo does not match any brand in your database.<br>
                            <span class="small-text">Nearest suggestions are listed below.</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ---------- MATCH FOUND ----------
                else:
                    css_class = "strong-match" if match_status == "STRONG_MATCH" else "weak-match"

                    st.markdown(
                        f"""
                        <div class="match-card {css_class}">
                            <span class="info-label">Predicted Brand:</span> {final_brand}<br>
                            <span class="info-label">Match Status:</span> {match_status} ({confidence})<br>
                            <span class="info-label">Overall Confidence (combined):</span> {combined_pct:.2f}%<br><br>
                            <span class="info-label">Shape Similarity:</span> {shape_sim:.2f}%<br>
                            <span class="info-label">Color Similarity:</span> {color_sim:.2f}%<br>
                            <span class="info-label">OCR Text Detected:</span> {text_detected if text_detected else "—"}<br>
                            <span class="info-label">Text Similarity:</span> {text_sim:.2f}%<br>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ---------- BRAND CONFIDENCE METER ----------
                st.markdown("**Brand Confidence Meter**")
                st.progress(min(max(combined_score, 0.0), 1.0))

                # ---------- SIDE-BY-SIDE BRAND COMPARISON ----------
                if result and match_status != "NO_MATCH" and top_matches:
                    st.subheader("3️⃣ Brand Comparison")

                    best_match_path = top_matches[0]["file_path"]

                    comp_cols = st.columns(2)
                    with comp_cols[0]:
                        st.markdown("**Input Logo**")
                        st.image(final_input_image, width=220)

                    with comp_cols[1]:
                        st.markdown(f"**Best Match – {final_brand}**")
                        if os.path.exists(best_match_path):
                            st.image(best_match_path, width=220)
                        else:
                            st.error("Best-match logo image not found on disk.")

                # ---------- TOP 3 NEAREST LOGOS ----------
                if top_matches:
                    st.subheader("4️⃣ Nearest Logos in Database")

                    for m in top_matches:
                        brand = m["brand_name"]
                        path = m["file_path"]
                        sim_percent = m["shape_similarity"] * 100.0

                        row_cols = st.columns([1, 3])
                        with row_cols[0]:
                            if os.path.exists(path):
                                st.image(path, width=130)
                            else:
                                st.text("Image missing")

                        with row_cols[1]:
                            st.markdown(f"**Brand:** {brand}")
                            st.markdown(f"**Shape Similarity:** {sim_percent:.2f}%")
                            st.markdown(
                                f"<span class='small-text'>File: {path}</span>",
                                unsafe_allow_html=True
                            )
                        st.markdown("---")

    else:
        st.info("Click **Analyze Logo** after selecting an image.")

