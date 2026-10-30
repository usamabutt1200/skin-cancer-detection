
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
    background: linear-gradient(135deg, #071923, #102d3a);
    color: #f2f7fa;
}
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
}
.hero {
    padding: 30px;
    border: 1px solid #31515d;
    border-radius: 22px;
    background: linear-gradient(120deg, #153743, #10232e);
    margin-bottom: 25px;
}
.hero h1 {
    color: #ffffff;
    font-size: 42px;
    margin-bottom: 8px;
}
.hero p {
    color: #bdd0d8;
    font-size: 17px;
}
.panel {
    background: #122b37;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #31515d;
}
div.stButton > button {
    background: #36c6ad;
    color: #071923;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}
div.stButton > button:hover {
    background: #63e0ca;
    color: #071923;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🔬 DermaAI</h1>
    <p>AI Skin Image Lab · Educational Classification Demo</p>
    <p>Explore how a machine-learning model classifies skin images.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📤 Upload an Image")
    st.write("Choose a JPG, JPEG or PNG image to test the model.")

    uploaded_file = st.file_uploader(
        "Select skin image",
        type=["jpg", "jpeg", "png"]
    )

    st.caption(
        "This demo is trained only on Benign and Malignant "
        "dataset categories."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📊 Model Result")

    if uploaded_file is None:
        st.info("Your result will appear here after you upload an image.")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)

            if st.button("Analyze Image"):
                resized = image.resize((128, 128))
                array = np.array(resized, dtype=np.float32)
                array = np.expand_dims(array, axis=0)

                with st.spinner("Analyzing image..."):
                    probability = float(
                        model.predict(array, verbose=0)[0][0]
                    )

                confidence = max(probability, 1 - probability)

                if confidence < 0.80:
                    st.warning(
                        "Unable to classify confidently. "
                        "This image may be outside the model's "
                        "supported categories."
                    )
                elif probability >= 0.5:
                    st.error("Model prediction: Malignant")
                else:
                    st.success("Model prediction: Benign")

                st.write(f"Model confidence: {confidence:.1%}")
                st.caption(
                    "Confidence is not proof of correctness. "
                    "The model may confidently misclassify pimples, "
                    "normal skin or other conditions."
                )

        except Exception:
            st.error("Could not read this image. Please upload another.")

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.warning(
    "Medical safety notice: This is an educational AI project, "
    "not a medical device. It cannot diagnose or rule out skin cancer. "
    "For any concerning skin change, consult a qualified doctor."
)
