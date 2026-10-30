
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🔬")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_cnn.keras")

model = load_model()

st.title("Skin Cancer Detection")
st.write("Educational AI demo — not a medical diagnosis tool.")

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    image = image.resize((128, 128))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Predict"):
        prediction = float(model.predict(image_array, verbose=0)[0][0])

        if prediction >= 0.5:
            st.write("Model prediction: Malignant")
        else:
            st.write("Model prediction: Benign")

        st.warning(
            "This educational model can make mistakes. "
            "It cannot diagnose skin cancer. Please consult a qualified doctor."
        )
