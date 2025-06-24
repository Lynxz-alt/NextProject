# MNIST Next Image Prediction using LSTM

Proyek ini memanfaatkan model LSTM untuk memprediksi gambar digit berikutnya berdasarkan urutan digit sebelumnya menggunakan dataset MNIST.

## Fitur
- Pelatihan model LSTM untuk data sekuensial gambar
- Evaluasi dan visualisasi loss
- Aplikasi Streamlit untuk prediksi interaktif

## Struktur
- `model_training.py` — Melatih model dan menyimpan hasil
- `streamlit_app.py` — UI berbasis Streamlit
- `utils/` — Folder modul pembantu

## Cara Menjalankan
1. Jalankan `model_training.py` untuk melatih dan menyimpan model.
2. Jalankan UI dengan:
```
streamlit run streamlit_app.py
```