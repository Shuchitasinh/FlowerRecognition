# Flower_Recognition_App.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# 🌸 Load the trained model
model = tf.keras.models.load_model("flower_model.h5")

# 📋 Load class names saved during training
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# 🖼️ Set image size (same as used during training)
img_height = 180
img_width = 180

# 🧹 Image preprocessing function
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((img_width, img_height))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 180, 180, 3)

# 🧠 Prediction function
def predict_flower(image_array):
    predictions = model.predict(image_array)
    probabilities = tf.nn.softmax(predictions[0])
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].numpy() * 100
    return predicted_class, confidence

# 🌺 Streamlit UI
st.title("🌸 Flower Recognition App")

uploaded_file = st.file_uploader("Upload an image of a flower", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_file = Image.open(uploaded_file)
    st.image(image_file, caption="Uploaded Image", use_container_width=True)

    image_array = preprocess_image(uploaded_file)

    if st.button("Predict"):
        predicted_class, confidence = predict_flower(image_array)
        st.success(f"🌼 Prediction: {predicted_class} ({confidence:.2f}% confidence)")
