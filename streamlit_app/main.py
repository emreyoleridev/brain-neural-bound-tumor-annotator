import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Proje kök dizinini ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.inference.predictor import BrainTumorPredictor

# Sayfa ayarları
st.set_page_config(page_title="MedVision AI", layout="wide")

st.title("🧠 MedVision AI: Brain Tumor Segmentation")
st.markdown("Upload an MRI scan to automatically segment potential tumors.")

# Ağırlık yolu
weights_path = os.path.join("app/inference/weights", "best_unet_model.pth")
predictor = BrainTumorPredictor(model_path=weights_path)

# Dosya yükleme
uploaded_file = st.file_uploader("Choose an MRI image...", type=["tif", "png", "jpg"])

if uploaded_file is not None:
    # Geçici dosyaya kaydet (predictor path beklediği için)
    temp_path = "temp_input.tif"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Tahmin
    mask = predictor.predict(temp_path)
    img = cv2.imread(temp_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Overlay oluşturma
    masked_img = img.copy()
    masked_img[mask == 1] = [255, 0, 0] # Tümörlü bölgeyi kırmızı yap
    
    # UI'da göster
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original MRI", width=600)
    with col2:
        st.image(masked_img, caption="AI Prediction Overlay", width=600)

    # İstatistik
    tumor_area = np.sum(mask)
    st.info(f"Detected Tumor Pixels: {tumor_area}")