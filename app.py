import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# -------------------------
# CONFIGURACIÓN DE LA APP
# -------------------------
st.set_page_config(page_title="IA Reconocimiento de Dígitos", layout="centered")

st.title("🧠 IA para Identificación de Números")
st.subheader(
    "Dibuja un número del 0 al 9. "
    "La red neuronal lo procesará y predecirá qué dígito escribiste."
)

# -------------------------
# CARGAR MODELO
# -------------------------
@st.cache_resource
def cargar_modelo():
    ruta = r"./modelo_entrenado.keras"
    modelo = tf.keras.models.load_model(ruta)
    return modelo

modelo = cargar_modelo()

# -------------------------
# CANVAS
# -------------------------
st.write("### ✏️ Escribe aquí tu número")

canvas_result = st_canvas(
    fill_color="black",      # fondo
    stroke_width=15,          # grosor lápiz
    stroke_color="white",    # lápiz blanco
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -------------------------
# PROCESAMIENTO DE IMAGEN
# -------------------------
def procesar_imagen(img_array):
    # convertir a escala de grises
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # redimensionar a 28x28
    small = cv2.resize(gray, (28, 28))

    # normalizar
    small = small / 255.0

    # reshape para CNN
    small = small.reshape(1, 28, 28, 1)

    return small

# -------------------------
# BOTÓN DE PREDICCIÓN
# -------------------------
if st.button("📸 Capturar y Predecir"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data.astype(np.uint8)
        img_proc = procesar_imagen(img)

        pred = modelo.predict(img_proc)

        numero = np.argmax(pred)
        prob = np.max(pred) * 100

        st.success(f"🔢 Número predicho: **{numero}**")
        st.info(f"📊 Probabilidad: **{prob:.2f}%**")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("©️ UNAB 2025 • Realizado por Mateo Sandoval 😊")