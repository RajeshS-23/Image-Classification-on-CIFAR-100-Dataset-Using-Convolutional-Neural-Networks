# CIFAR-100 Image Classification Using Deep Learning

This project implements image classification on the CIFAR-100 dataset using
Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN)
with TensorFlow and Keras.

The goal of the project is to train a deep learning model that can classify
32×32 RGB images into one of 100 different object categories.

---

## Dataset

- Dataset Name: CIFAR-100
- Number of Classes: 100
- Image Size: 32 × 32 pixels
- Color Channels: RGB (3 channels)
- Training Samples: 50,000
- Testing Samples: 10,000

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## Model Architecture

### ANN Model
- Flatten layer
- Dense layer with ReLU activation
- Output Dense layer with Softmax activation

### CNN Model
- Convolutional layers with ReLU activation
- Max Pooling layers
- Fully connected Dense layers
- Softmax output layer for multi-class classification

---

## Loss Function and Optimizer

- Loss Function: Sparse Categorical Crossentropy
- Optimizer: Adam
- Evaluation Metric: Accuracy

---

## Training Details

- Epochs: 10
- Validation: Test dataset used for validation
- Data Normalization: Pixel values scaled between 0 and 1

---

## Results

The CNN model achieves better accuracy compared to the ANN model.
Due to the complexity of the CIFAR-100 dataset, accuracy is limited
when using a basic CNN architecture.

---

## How to Run the Project

1. Install required libraries:
   ```bash
   pip install tensorflow matplotlib numpy
