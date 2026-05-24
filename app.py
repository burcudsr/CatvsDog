import io
import tensorflow as tf
import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import h5py 

@st.cache_resource
def load_my_model():
    # Parçaları birleştirip geçici bir dosya oluşturmak bazen daha garantidir
    temp_filename = 'temp_model.h5'
    with open(temp_filename, 'wb') as outfile:
        for i in range(3):
            part_name = f'catdog_part{i}.h5'
            if os.path.exists(part_name):
                with open(part_name, 'rb') as infile:
                    outfile.write(infile.read())
    
    # h5py ile dosyayı oku ve Keras'a ver
    model = tf.keras.models.load_model(temp_filename, compile=False)
    return model

# Modeli yükle
model = load_my_model()

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
