import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image

# 1. Model Mimarini Tanımla (EĞİTİM KODUNDAKİYLE AYNI OLMALI!)
def get_model():
    model = Sequential([
        # ÖRNEK: Eğitirken kullandığın katmanları buraya yaz
        # Örneğin:
        Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# 2. Ağırlıkları Yükle
# 1. Önce fonksiyonu tanımla
@st.cache_resource
def load_my_model():
    weights_filename = 'catdog_final_weights.h5'
    with open(weights_filename, 'wb') as outfile:
        for i in range(9): 
            part_name = f'catdog_part{i}.h5' 
            if os.path.exists(part_name):
                with open(part_name, 'rb') as infile:
                    outfile.write(infile.read())
    
    # Model mimarisini oluştur
    model = get_model() 
    # Birleşen ağırlıkları yükle
    model.load_weights(weights_filename)
    
    # Fonksiyonun işi bitince modeli dışarı ver
    return model

# 2. FONKSİYON DIŞINDA: Modeli bir değişkene ata
model = load_my_model()

# 3. Artık aşağıda gönül rahatlığıyla 'model' değişkenini kullanabilirsin
# (Örn: prediction = model.predict(img))

# ... (Streamlit arayüzü kodların aynı kalabilir)
# 2. Streamlit Interface
st.title("🐱 Cat vs Dog Classifier")
st.write("Upload an image, and the model will predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 3. Preprocessing steps (must match your training logic)
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to 120x120 as per training configuration
    img = cv2.resize(img_array, (120, 120))
    
    # Normalize pixel values
    img = img / 255.0
    
    # Expand dimensions to (1, 120, 120, 3) for the model
    img = np.expand_dims(img, axis=0)
    
    # 4. Prediction logic
    if st.button("Predict"):
        prediction = model.predict(img)
        
        # Classification logic: 1 = Dog, 0 = Cat
        if prediction[0][0] > 0.5:
            st.success("Result: This is a DOG! 🐶")
        else:
            st.success("Result: This is a CAT! 🐱")
