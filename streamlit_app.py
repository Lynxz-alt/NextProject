import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Prediksi Gambar Berikutnya (MNIST) - Lite")

uploaded_files = st.file_uploader("Unggah 4 Gambar MNIST", type=["png", "jpg"], accept_multiple_files=True)

if len(uploaded_files) == 4:
    images = [np.array(Image.open(f).convert('L').resize((28, 28))) / 255.0 for f in uploaded_files]
    input_seq = np.stack(images, axis=0).reshape(1, 4, 784).astype(np.float32)

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_seq)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    st.image(output_data[0], caption="Prediksi Gambar", clamp=True)
else:
    st.warning("Tolong upload tepat 4 gambar untuk memulai prediksi.")
