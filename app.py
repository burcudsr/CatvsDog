import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image

# 1. Model Mimarini Tanımla (Eğitim kodundaki ile BİREBİR aynı olmalı!)
def get_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# 2. Modeli ve Ağırlıkları Yükle
@st.cache_resource
def load_my_model():
    weights_filename = 'catdog_final_weights.h5'
    
    # 9 parçayı birleştir
    with open(weights_filename, 'wb') as outfile:
        for i in range(9): 
            part_name = f'catdog_part{i}.h5'
            if os.path.exists(part_name):
                with open(part_name, 'rb') as infile:
                    outfile.write(infile.read())
    
    # Boş mimariyi oluştur ve ağırlıkları yükle
    model = get_model()
    model.load_weights(weights_filename)
    return model

# Modeli başlangıçta yükle
model = load_my_model()

# 3. Streamlit Arayüzü
st.title("🐱 Cat vs Dog Classifier")
st.write("Upload an image, and the model will predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Resmi göster
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Resim ön işleme (Eğitim mantığınla aynı olmalı)
    img_array = np.array(image)
    img = cv2.resize(img_array, (120, 120))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Tahmin etme butonu
    if st.button("Predict"):
        prediction = model.predict(img)
        
        # 1 = Dog, 0 = Cat
        if prediction[0][0] > 0.5:
            st.success("Result: This is a DOG! 🐶")
        else:
            st.success("Result: This is a CAT! 🐱")
