#!/usr/bin/env python
# coding: utf-8

# In[1]:

# 📚 Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib
import tarfile
import os

# 📁 Download and manually extract the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
tgz_path = tf.keras.utils.get_file('flower_photos.tgz', origin=dataset_url, extract=False)

# Extract the tgz file manually
extract_path = os.path.splitext(tgz_path)[0]  # remove '.tgz'
if not os.path.exists(extract_path):
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=os.path.dirname(tgz_path))

data_dir = pathlib.Path(extract_path)

# 📊 Split dataset
batch_size = 32
img_height = 180
img_width = 180

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print("Classes:", class_names)

# 📈 Prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 🧠 Build the CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# 🏋️ Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 📈 Plot training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# 📷 Upload and predict image
import streamlit as st
from PIL import Image
import numpy as np
# 📚 Import libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import os

# 🧠 Load your trained model
model = tf.keras.models.load_model("flower_model.h5")

# 🖼️ Set image size (same size you used in training)
img_height = 180
img_width = 180

# 🌸 List your flower classes
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # replace with actual classes if different

st.title("🌸 Flower Recognition App")

uploaded_file = st.file_uploader("Upload an image of a flower", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_file = Image.open(uploaded_file)
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    # Resize image to match model input
    img = image_file.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    if st.button("Predict"):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_index = np.argmax(score)
        predicted_class = class_names[predicted_index]
        confidence = 100 * np.max(score)

        st.success(f"🌼 Prediction: {predicted_class} ({confidence:.2f}% confidence)")


    img_path = fn
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Predict flower
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Get predicted class and confidence
    predicted_index = np.argmax(score)
    predicted_class = class_names[predicted_index]
    confidence = 100 * np.max(score)

    # Display the image with flower name and confidence
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"🌸 {predicted_class} ({confidence:.2f}% confidence)", fontsize=16)
    plt.show()