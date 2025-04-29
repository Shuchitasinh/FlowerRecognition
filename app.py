# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("flower_model.h5")

# Set image size (same as used in training)
img_height = 180
img_width = 180

# Define class names (same order as training)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Streamlit UI
st.title("🌸 Flower Recognition App")
uploaded_file = st.file_uploader("Upload an image of a flower", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_file = Image.open(uploaded_file)
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    img = image_file.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    if st.button("Predict"):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_index = np.argmax(score)
        predicted_class = class_names[predicted_index]
        confidence = 100 * np.max(score)

        st.success(f"🌼 Prediction: {predicted_class} ({confidence:.2f}% confidence)")