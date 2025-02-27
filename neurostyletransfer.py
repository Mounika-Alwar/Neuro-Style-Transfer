import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Load the TensorFlow Hub model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to load and preprocess an image
def load_image(img):
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

# Streamlit app
st.title("Neural Style Transfer")
st.write("Upload a content image and a style image to generate a stylized image.")

# Upload content image
content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
if content_file is not None:
    content_image = load_image(content_file.read())
    st.image(content_image.numpy().squeeze(), caption="Content Image", use_column_width=True)

# Upload style image
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
if style_file is not None:
    style_image = load_image(style_file.read())
    st.image(style_image.numpy().squeeze(), caption="Style Image", use_column_width=True)

# Generate stylized image
if content_file is not None and style_file is not None:
    if st.button("Generate Stylized Image"):
        with st.spinner("Generating stylized image..."):
            # Perform style transfer
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
            stylized_image = stylized_image.numpy().squeeze()
        
        # Display the stylized image
        st.image(stylized_image, caption="Stylized Image", use_column_width=True)