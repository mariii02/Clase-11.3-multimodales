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
    
    # Preprocess the image
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    
    # Display the image
    plt.imshow(img[0].reshape(28, 28), cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()
    
    # Predict the digit
    pred = model.predict(img)
    result = np.argmax(pred[0])
    
    return result

# Streamlit application
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
st.title('Reconocimiento de Dígitos escritos a mano')
st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# Canvas parameters
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'  # Color del trazo
bg_color = '#000000'      # Color de fondo

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno con opacidad
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Predict button
if st.button('Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        
        # Load the image for prediction
        img = Image.open("prediction/img.png")
        res = predict_digit(img)
        
        st.header(f'El Dígito es: {res}')
    else:
        st.header('Por favor, dibuja en el canvas el dígito.')

# Sidebar information
st.sidebar.title("Acerca de:")
st.sidebar.text("En esta aplicación se evalúa")
st.sidebar.text("la capacidad de una RNA de reconocer")
st.sidebar.text("dígitos escritos a mano.")
st.sidebar.text("Desarrollado por Vinay Uniyal")
# st.sidebar.text("GitHub Repository")
# st.sidebar.write("[GitHub Repo Link](https://github.com/Vinay2022/Handwritten-Digit-Recognition)")
