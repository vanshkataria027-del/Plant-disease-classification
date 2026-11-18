import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "plant_disease_model.h5"

CLASS_NAMES = [
    'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust',
    'Apple__healthy', 'Blueberry__healthy', 'Cherry__healthy',
    'Cherry__Powdery_mildew', 'Corn__Cercospora_leaf_spot',
    'Corn__Common_rust', 'Corn__healthy', 'Corn__Northern_Leaf_Blight',
    'Grape__Black_rot', 'Grape__Esca', 'Grape__healthy',
    'Grape__Leaf_blight', 'Orange__Haunglongbing',
    'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper__Bacterial_spot', 'Pepper__healthy',
    'Potato__Early_blight', 'Potato__healthy', 'Potato__Late_blight',
    'Raspberry__healthy', 'Soybean__healthy', 'Squash__Powdery_mildew',
    'Strawberry__healthy', 'Strawberry__Leaf_scorch',
    'Tomato__Bacterial_spot', 'Tomato__Early_blight', 'Tomato__healthy',
    'Tomato__Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites',
    'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
    'Tomato__Yellow_Leaf_Curl_Virus'
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")
st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image to detect disease")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' repo mein nahi mili. GitHub par upload karo.")
    st.stop()

model = load_model()

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Image", width=300)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[int(np.argmax(prediction))]
    confidence = float(np.max(prediction)) * 100

    st.success(f"**Disease:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
