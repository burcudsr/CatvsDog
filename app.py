import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, Input
from tensorflow.keras import regularizers
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 1. MİMARİ: Eğitim kodundaki yapı ile %100 UYUMLU
def get_model():
    model = Sequential([
        Input(shape=(120, 120, 3)),
        
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),
        
        Flatten(),
        Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

@st.cache_resource
def load_my_model():
    model = get_model()
    
    # NPZ dosyasını doğrudan oku
    # Dosya zaten GitHub'da olduğu için doğrudan dosya yolunu veriyoruz
    data = np.load('catdog_weights.npz')
    
    # weights key'lerini eğitiminle aynı sıraya göre al
    # Eğer key'lerin "arr_0", "arr_1" şeklindeyse sıralı yükler
    weights = [data[key] for key in sorted(data.files)]
    model.set_weights(weights)
    
    return model

model = load_my_model()

# ARAYÜZ (Aynı kalabilir)
st.title("🐱 Cat vs Dog Classifier")
# ... (Tahmin kodun)
