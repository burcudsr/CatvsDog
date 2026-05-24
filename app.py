import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, Input
from tensorflow.keras import regularizers
import streamlit as st
import numpy as np
import cv2
from PIL import Image

def get_model():
    # Eğitim kodundaki mimarin
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
    # 1. HATA ALIRSAN: by_name=True ve skip_mismatch=True parametreleri hayat kurtarır.
    # 2. Eğer npz kullanıyorsan ve hata alıyorsan, ağırlık dosyanın .h5 olması önerilir.
    # .npz dosyaları "by_name" desteklemez, bu yüzden .h5 tercih et.
    try:
        model.load_weights('catdog_weights.h5', by_name=True, skip_mismatch=True)
    except:
        # Eğer yine olmazsa, en son çare model ağırlıklarını manuel çek
        pass
    return model

model = load_my_model()

# ... (Streamlit Arayüzü)
