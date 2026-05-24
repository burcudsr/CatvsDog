import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, Input
from tensorflow.keras import regularizers
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 1. MİMARİ (Eğitimdeki yapıyla %100 uyumlu)
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

# 2. MODEL YÜKLEME (NPZ üzerinden garantili yükleme)
@st.cache_resource
def load_my_model():
    model = get_model()
    # NPZ dosyasını oku ve ağırlıkları modele zorla enjekte et
    data = np.load('catdog_weights.npz')
    # npz içindeki dosyaların listesini al ve sıralı olarak model ağırlıklarına ata
    weights = [data[key] for key in sorted(data.files)]
    model.set_weights(weights)
    return model

model = load_my_model()

# 3. STREAMLIT ARAYÜZÜ
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱")
st.title("🐱 Cat vs Dog Classifier")
st.write("Upload an image of a cat or a dog to see the prediction!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Resmi göster
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Resim ön işleme (Eğitimdeki boyut ve normalizasyon ile aynı)
    img_array = np.array(image)
    img = cv2.resize(img_array, (120, 120))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    if st.button("Predict"):
        with st.spinner('Analyzing...'):
            prediction = model.predict(img)
            score = prediction[0][0]
            
            if score > 0.5:
                st.success(f"Result: This is a DOG! 🐶 (Confidence: {score:.2%})")
            else:
                st.success(f"Result: This is a CAT! 🐱 (Confidence: {1-score:.2%})")

st.sidebar.info("Model Info: CNN architecture with L2 Regularization.")
