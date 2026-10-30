
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

st.markdown("""
<style>
.stApp {
    background: #081720;
    color: #eef6fa;
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(120deg, #153b49, #102631);
    border: 1px solid #31515e;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 25px;
}
.hero h1 {
    color: #ffffff;
    font-size: 42px;
    margin: 0;
}
.hero p {
    color: #b4cbd5;
    font-size: 16px;
}
.panel {
    background: #102630;
    border: 1px solid #294650;
    border-radius: 18px;
    padding: 22px;
}
.result-card {
    background: #15343f;
    border: 1px solid #31535e;
    border-radius: 16px;
    padding: 22px;
    margin-top: 12px;
}
.result-title {
    color: #b9d0d9;
    font-size: 13px;
    letter-spacing: 1.5px;
}
.result-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 700;
    margin: 8px 0;
}
.metric {
    background: #15343f;
    border: 1px solid #294650;
    border-radius: 14px;
    padding: 17px;
    min-height: 110px;
}
.metric-label {
    color: #a8c1cc;
    font-size: 13px;
}
.metric-value {
    color: #50ddc1;
    font-size: 22px;
    font-weight: 700;
    margin-top: 8px;
}
div.stButton > button {
    background: #50ddc1;
    color: #081720;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 46px;
    width: 100%;
}
div.stButton > button:hover {
    background: #80f0da;
    color: #081720;
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

st.markdown("""
<div class="hero">
    <p>✦ INTELLIGENT IMAGE CLASSIFICATION</p>
    <h1>🔬 DermaAI</h1>
    <p>Your AI-powered skin image classification demo.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([0.9, 1.1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📤 Upload Image")
    st.write("Choose a JPG, JPEG or PNG image to begin.")

    uploaded_file = st.file_uploader(
        "Select an image",
        type=["jpg", "jpeg", "png"]
    )

    st.caption("Supported input: RGB image, resized to 128 × 128.")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📊 Analysis Dashboard")

    if uploaded_file is None:
        st.info("Upload an image to see the model's result here.")
        st.markdown("""
        <div class="result-card">
            <div class="result-title">SYSTEM STATUS</div>
            <div class="result-value">Ready to Analyze</div>
            <p style="color:#b4cbd5">
                Your result and model metrics will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif st.button("✦ Analyze Image"):
        try:
            with st.spinner("Running model analysis..."):
                resized = image.resize((128, 128))
                array = np.array(resized, dtype=np.float32)
                array = np.expand_dims(array, axis=0)

                probability = float(
                    model.predict(array, verbose=0)[0][0]
                )

            confidence = max(probability, 1 - probability)

            st.markdown("""
            <div class="result-card">
                <div class="result-title">ANALYSIS STATUS</div>
                <div class="result-value">Analysis Complete</div>
            </div>
            """, unsafe_allow_html=True)

            # This is a confidence threshold, not an "Other"
            # class detector. It cannot reliably detect unsupported images.
            if confidence < 0.80:
                st.warning(
                    "Uncertain prediction. The model cannot confidently "
                    "classify this image as Benign or Malignant."
                )
            elif probability >= 0.5:
                st.error("Model prediction: Malignant")
            else:
                st.success("Model prediction: Benign")

            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">MODEL CONFIDENCE</div>
                <div class="result-value">{confidence:.1%}</div>
                <div style="background:#294650;border-radius:20px;
                            height:10px;margin-top:12px">
                    <div style="background:#50ddc1;
                                width:{confidence * 100:.2f}%;
                                height:10px;border-radius:20px">
                    </div>
                </div>
                <p style="color:#b4cbd5;font-size:13px;margin-top:12px">
                    Confidence reflects the model's output, not the
                    probability that a person has or does not have cancer.
                </p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class="metric">
                    <div class="metric-label">MODEL TYPE</div>
                    <div class="metric-value">CNN</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown("""
                <div class="metric">
                    <div class="metric-label">INPUT SIZE</div>
                    <div class="metric-value">128 × 128</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as error:
            st.error("Unable to analyze this image. Please try another.")
            st.caption(str(error))

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.markdown("""
<div style="
    background: linear-gradient(135deg, #102630, #183844);
    border: 1px solid #c9a85c;
    border-left: 4px solid #d8b765;
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 20px;
">
    <div style="
        color: #e5c778;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 10px;
    ">
        ⚕ IMPORTANT MEDICAL NOTICE
    </div>
    <div style="
        color: #edf4f7;
        font-size: 14px;
        line-height: 1.8;
    ">
        <b>For Educational Use Only</b><br>
        
Based on the dataset provided for training, this AI model
classifies images into Benign and Malignant categories and
may misclassify other skin conditions. It is not a medical
diagnostic tool. Please consult a qualified healthcare
professional for medical concerns.
    </div>
</div>
""", unsafe_allow_html=True)

