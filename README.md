
# 🐱 vs 🐶 Classifier

This project is a deep learning-based image classification application trained on **1000 verified cat and dog images**. It utilizes a custom **CNN (Convolutional Neural Network)** architecture, optimized for accuracy and performance.

### 🌐 Live Application
You can test the classifier here: [CatvsDog](https://catvsdog-bdsr.streamlit.app/)

## 🛠 Model Architecture
The model's design focuses on robustness and feature extraction through carefully tuned layers:

* **Input:** 120x120 pixels, 3-channel (RGB) images.
* **Core Layers:** 3x Convolutional layers (Conv2D), integrated with **Batch Normalization** and **L2 Regularization** to prevent overfitting.
* **Refinement:** Dropout layers and LeakyReLU activations ensure smooth gradient flow and model generalization.
* **Classification:** Binary output via Sigmoid activation.

* 
## 📋 Technical Specs
* **Dataset:** 1000 images processed and normalized.
* **Preprocessing:** Automatic image resizing and pixel normalization (0-1 range) via OpenCV and Pillow.
* **Tech Stack:** `TensorFlow/Keras` (Core Engine), `NumPy` (Weight Handling), `Streamlit` (Frontend).

## 🚀 Scaling & Future Improvements
While the current custom CNN performs on the 1000-image dataset, future iterations will focus on further increasing accuracy:

* **Transfer Learning:** Integrating pre-trained models (e.g., MobileNetV2, ResNet50) to leverage features learned from millions of images (ImageNet). This will significantly boost performance on complex, real-world data.
