import tensorflow as tf
import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image

# 1. YÜKLEME: Mimariyi .keras, ağırlıkları .h5 dosyalarından yükle
@st.cache_resource
def load_my_model():
    # --- A. Mimariyi Birleştir ve Yükle ---
    keras_filename = 'full_model_structure.keras'
    # .keras parçalarının sayısını buraya yaz (örneğin 3 ise range(3))
    with open(keras_filename, 'wb') as outfile:
        for i in range(3): 
            part_name = f'model_part{i}.keras'
            if os.path.exists(part_name):
                with open(part_name, 'rb') as infile:
                    outfile.write(infile.read())
    
    # --- B. Ağırlıkları Birleştir ---
    weights_filename = 'final_weights.h5'
    with open(weights_filename, 'wb') as outfile:
        for i in range(9):
            part_name = f'catdog_part{i}.h5'
            if os.path.exists(part_name):
                with open(part_name, 'rb') as infile:
                    outfile.write(infile.read())
    
    # --- C. Hibrit Yükleme ---
    # Mimariyi dosyadan oku (compile=False yaparak hatayı engelliyoruz)
    model = tf.keras.models.load_model(keras_filename, compile=False)
    # Ağırlıkları mimarinin üzerine enjekte et
    model.load_weights(weights_filename)
    
    return model

# Modeli yükle
model = load_my_model()

# 2. ARAYÜZ (Aynı kalabilir)
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
