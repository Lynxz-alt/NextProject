import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

st.title("Prediksi Gambar Berikutnya (MNIST)")

uploaded_files = st.file_uploader("Unggah 4 Gambar MNIST", type=["png", "jpg"], accept_multiple_files=True)

if len(uploaded_files) == 4:
    images = [np.array(Image.open(f).convert('L').resize((28, 28))) / 255.0 for f in uploaded_files]
    input_seq = np.stack(images, axis=0).reshape(1, 4, 784)

    model = load_model("final_model_adam.h5")
    pred = model.predict(input_seq)

    st.image(pred[0], caption='Hasil Prediksi', clamp=True, width=150)