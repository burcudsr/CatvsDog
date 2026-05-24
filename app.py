import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image

# 1. MİMARİ: summary() çıktın ile BİREBİR aynı olacak şekilde güncellendi
def get_model():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(120, 120, 3)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(128, (3, 3)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Flatten(),
        Dense(128),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# 2. YÜKLEME: Parçaları birleştir ve ağırlıkları yükle
@st.cache_resource
def load_my_model():
    weights_filename = 'catdog_final_weights.h5'
    
    # Parçaları birleştir
    with open(weights_filename, 'wb') as outfile:
        for i in range(9): 
            part_name = f'catdog_part{i}.h5'
            if os.path.exists(part_name):
                with open(part_name, 'rb') as infile:
                    outfile.write(infile.read())
    
    model = get_model()
    # ÖNEMLİ: Ağırlıkları yükle
    model.load_weights(weights_filename)
    return model

# Modeli yükle
model = load_my_model()

# 3. ARAYÜZ
st.title("🐱 Cat vs Dog Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    img_array = np.array(image)
    img = cv2.resize(img_array, (120, 120))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    if st.button("Predict"):
        prediction = model.predict(img)
        if prediction[0][0] > 0.5:
            st.success("Result: This is a DOG! 🐶")
        else:
            st.success("Result: This is a CAT! 🐱")
