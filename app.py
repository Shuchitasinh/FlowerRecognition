import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("flower_model.keras")

# Load class names from saved JSON file
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Set image size (must match training)
img_height = 180
img_width = 180

# Streamlit UI
st.title("🌸 Flower Recognition App")
uploaded_file = st.file_uploader("Upload an image of a flower", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_file = Image.open(uploaded_file).convert("RGB")
    st.image(image_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image_file.resize((img_width, img_height))
    img_array = np.array(img) / 255.0  # Normalize to 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    if st.button("Predict"):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = tf.nn.softmax(predictions[0])[predicted_index] * 100

        st.success(f"🌼 Prediction: {predicted_class} ({confidence:.2f}% confidence)")
