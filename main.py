# mejor modelo árabe
import os
import numpy as np
import cv2
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas


logging.basicConfig(level=logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


model = tf.keras.models.load_model('mejor_modelo_arabe.h5', compile=True)

st.title('Reconocimiento de dígitos árabes')
st.markdown('''
Dibuje un dígito en el lienzo de abajo o suba una imagen. El modelo predecirá el dígito. 
Además, se presenta la distribución de las puntuaciones de confianza en todas las clases.
''')

SIZE = 280
mode = st.checkbox("Dibujar o borrar", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Resultado')
    st.image(rescaled)

if st.button('Predecir'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1, 32, 32))
    st.write(f'Resultado: {np.argmax(val[0])}')
    st.bar_chart(val[0])