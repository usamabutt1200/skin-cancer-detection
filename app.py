
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="DermaAI | Skin Image Lab",
    page_icon="🔬",
    layout="wide"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_cnn.keras")


model = load_model()


# ---------- PREMIUM DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #081720, #102b36);
    color: #eef6fa;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(120deg, #173b49, #102630);
    border: 1px solid #34535e;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff;
    font-size: 44px;
    margin: 0;
}

.hero p {
    color: #b9ced7;
    font-size: 16px;
}

.panel {
    background: #102630;
    border: 1px solid #294650;
    border-radius: 18px;
    padding: 22px;
}

.result-card {
    background: linear-gradient(135deg, #163540, #102630);
    border: 1px solid #34535e;
    border-radius: 16px;
    padding: 22px;
    margin-top: 14px;
}

.result-title {
    color: #b9d0d9;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}

.result-value {
    color: #ffffff;
    font-size: 28px;
    font-weight: 700;
    margin: 10px 0;
}

.metric {
    background: #15343f;
    border: 1px solid #294650;
    border-radius: 14px;
    padding: 17px;
    min-height: 100px;
}

.metric-label {
    color: #a8c1cc;
    font-size: 12px;
    letter-spacing: 1px;
}

.metric-value {
    color: #50ddc1;
    font-size: 22px;
    font-weight: 700;
    margin-top: 8px;
}

.medical-notice {
    background: linear-gradient(135deg, #142d38, #0d202a);
    border: 1px solid #8d7742;
    border-left: 4px solid #d7b663;
    border-radius: 14px;
    padding: 20px;
    margin-top: 26px;
}

.notice-heading {
    color: #e5c778;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.notice-text {
    color: #e5edf1;
    font-size: 14px;
    line-height: 1.8;
}

div.stButton > button {
    background: linear-gradient(90deg, #50ddc1, #a1e8d8);
    color: #081720;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 48px;
    width: 100%;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background: #c0f5e9;
    color: #081720;
    border: none;
}

div[data-testid="stFileUploader"] {
    background: #132d38;
    border-radius: 12px;
    padding: 10px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <p>✦ INTELLIGENT IMAGE CLASSIFICATION</p>
    <h1>🔬 DermaAI</h1>
    <p>
        AI-powered skin image classification
        <br>
        Explore deep learning through an educational demo.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- IMAGE UPLOAD ----------
left, right = st.columns([0.9, 1.1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("📤 Upload Image")
    st.write("Select a JPG, JPEG or PNG image to begin.")

    uploaded_file = st.file_uploader(
        "Choose your image",
        type=["jpg", "jpeg", "png"]
    )

    st.caption("Image input is resized to 128 × 128 pixels.")

    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True
            )
        except Exception:
            st.error("Unable to open this image. Please try another.")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- RESULTS ----------
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("📊 Analysis Dashboard")

    if uploaded_file is None:
        st.info("Your analysis result will appear here.")

        st.markdown("""
        <div class="result-card">
            <div class="result-title">SYSTEM STATUS</div>
            <div class="result-value">Ready to Analyze</div>
            <p style="color:#b9ced7;">
                Upload an image to view the model's prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif image is not None:
        if st.button("✦ Analyze Image", key="analyze_image"):
            try:
                resized = image.resize((128, 128))

                image_array = np.array(
                    resized, dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array, axis=0
                )

                with st.spinner("Analyzing image..."):
                    probability = float(
                        model.predict(
                            image_array, verbose=0
                        )[0][0]
                    )

                confidence = max(
                    probability, 1 - probability
                )

                # Brief warning notification
                st.toast(
                    "Educational result — not a diagnosis",
                    icon="⚠️"
                )

                st.markdown("""
                <div class="result-card">
                    <div class="result-title">
                        ANALYSIS STATUS
                    </div>
                    <div class="result-value">
                        Analysis Complete
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Low confidence is not a reliable
                # detector of unsupported image classes.
                if confidence < 0.80:
                    st.warning(
                        "⚠️ Uncertain prediction. The model "
                        "cannot confidently classify this image. "
                        "This does not confirm that it belongs "
                        "to another class."
                    )

                elif probability >= 0.5:
                    st.error("Model prediction: Malignant")

                else:
                    st.success("Model prediction: Benign")

                # Confidence display
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">
                        MODEL CONFIDENCE
                    </div>

                    <div class="result-value">
                        {confidence:.1%}
                    </div>

                    <div style="
                        background:#294650;
                        border-radius:20px;
                        height:10px;
                        margin-top:12px;
                    ">
                        <div style="
                            background:linear-gradient(
                                90deg, #36bfa8, #a1e8d8
                            );
                            width:{confidence * 100:.2f}%;
                            height:10px;
                            border-radius:20px;
                        "></div>
                    </div>

                    <p style="
                        color:#b9ced7;
                        font-size:13px;
                        margin-top:14px;
                        line-height:1.7;
                    ">
                        This score reflects the model's output.
                        It is not the probability that a person
                        has or does not have skin cancer.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Model details
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("""
                    <div class="metric">
                        <div class="metric-label">
                            MODEL ARCHITECTURE
                        </div>
                        <div class="metric-value">CNN</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    st.markdown("""
                    <div class="metric">
                        <div class="metric-label">
                            INPUT RESOLUTION
                        </div>
                        <div class="metric-value">
                            128 × 128
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception:
                st.error(
                    "Unable to analyze this image. "
                    "Please upload another image."
                )

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- PREMIUM MEDICAL NOTICE ----------
st.markdown("""
<div class="medical-notice">
    <div class="notice-heading">
        ⚕ &nbsp; IMPORTANT MEDICAL NOTICE
    </div>

    <div class="notice-text">
        <b>For Educational Use Only</b><br>
        DermaAI is an experimental image-classification project,
        not a medical diagnostic tool. It may misclassify skin
        images, including conditions outside its two trained
        categories. Please consult a qualified healthcare
        professional for medical concerns.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
    text-align:center;
    color:#7895a1;
    font-size:12px;
    margin-top:22px;
">
    DERMAAI · EDUCATIONAL AI PROJECT
</p>
""", unsafe_allow_html=True)
