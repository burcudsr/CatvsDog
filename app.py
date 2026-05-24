import tensorflow as tf
import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image

@st.cache_resource
def load_my_model():
    # 1. MİMARİYİ BİRLEŞTİR (3 parça .keras)
    combined_keras = 'full_model.keras'
    with open(combined_keras, 'wb') as outfile:
        for i in range(3): 
            part_name = f'catdog_part{i}.keras'
            if os.path.exists(part_name):
                with open(part_name, 'rb') as infile:
                    outfile.write(infile.read())
    
    # 2. AĞIRLIKLARI BİRLEŞTİR (9 parça .h5)
    combined_weights = 'full_weights.h5'
    with open(combined_weights, 'wb') as outfile:
        for i in range(9):
            part_name = f'catdog_part{i}.h5'
            if os.path.exists(part_name):
                with open(part_name, 'rb') as infile:
                    outfile.write(infile.read())
    
    # 3. HİBRİT YÜKLEME: Önce mimariyi, sonra ağırlıkları
    # compile=False kullanarak modelin eğitim ayarlarını yüklemeden sadece yapıyı alıyoruz
    model = tf.keras.models.load_model(combined_keras, compile=False)
    
    # Ağırlıkları mimarinin üzerine güvenle yüklüyoruz
    model.load_weights(combined_weights)
    
    return model

model = load_my_model()

# ARAYÜZ (Tahmin kodların aynı kalabilir)
st.title("🐱 Cat vs Dog Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    img = cv2.resize(np.array(image), (120, 120)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    if st.button("Predict"):
        prediction = model.predict(img)
        if prediction[0][0] > 0.5:
            st.success("Result: This is a DOG! 🐶")
        else:
            st.success("Result: This is a CAT! 🐱")
