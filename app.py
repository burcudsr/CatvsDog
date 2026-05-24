import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 1. Mimariyi kodla oluştur
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

@st.cache_resource
def load_my_model():
    model = get_model()
    
    # 2. .npz dosyasını yükle
    data = np.load('catdog_weights.npz')
    
    # .npz içindeki ağırlıkları listeye al
    weights = [data[key] for key in sorted(data.files, key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)]
    
    # 3. Ağırlıkları modele set et
    model.set_weights(weights)
    return model

model = load_my_model()

# ARAYÜZ (Tahmin kodların aynı)
st.title("🐱 Cat vs Dog Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    img = cv2.resize(np.array(image), (120, 120)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    if st.button("Predict"):
        prediction = model.predict(img)
        st.success(f"Result: {'DOG 🐶' if prediction[0][0] > 0.5 else 'CAT 🐱'}")
