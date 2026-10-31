
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#101827,#203653);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.result {
    background: #172b43;
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #426184;
}
</style>
""", unsafe_allow_html=True)

st.title("✍️ Handwritten Character Recognition")
st.caption("An AI-powered character recognition application using CNN")

MODEL_PATH = "model/character_model.keras"
CLASS_PATH = "model/classes.json"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r") as file:
        return json.load(file)

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
    st.warning("Model not found. Please run train_model.py first.")
    st.stop()

model = load_model()
classes = load_classes()

st.markdown("### Upload your handwritten image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Uploaded handwritten image",
        width=250
    )

    if st.button("Recognize Character", use_container_width=True):

        if np.array(image).mean() > 127:
            image = ImageOps.invert(image)

        image = image.resize((28, 28))

        image_array = np.array(image).astype("float32") / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        prediction = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_character = classes[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

        st.markdown(f"""
        <div class="result">
            <h3>Recognized Character</h3>
            <h1 style="font-size:65px;">{predicted_character}</h1>
            <p>Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(prediction[predicted_index]))

        st.success("Character recognition completed!")

st.markdown("---")
st.markdown("### About This Project")
st.write("""
This project uses a Convolutional Neural Network (CNN)
to recognize handwritten digits and characters.

Dataset: EMNIST Balanced
Framework: TensorFlow and Keras
Interface: Streamlit
""")

st.caption("Developed as an AI/ML project.")

