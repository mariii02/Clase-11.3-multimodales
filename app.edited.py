import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Function to predict the digit
def predict_digit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    
    plt.imshow(img[0].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()
    
    pred = model.predict(img)
    result = np.argmax(pred[0])
    
    return result

# Streamlit application
st.set_page_config(page_title='Reconocimiento de Dígitos', layout='wide')
st.title('🔍 Reconocimiento de Dígitos escritos a mano')
st.subheader("✏️ Dibuja el dígito en el panel y presiona 'Predecir'")

# Canvas parameters
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Predict button
if st.button('Predecir'):
    if canvas_result.image_data is

