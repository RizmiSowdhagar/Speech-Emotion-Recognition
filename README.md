# Speech Emotion Recognition using CNN-LSTM and MFCC

This project implements a deep learning pipeline to detect and classify human emotions from speech audio clips using MFCC features, Convolutional Neural Networks (CNN), and LSTM layers. It supports multi-class emotion recognition with robust preprocessing and visualization.

---

## Overview

Emotion recognition from voice is a key task in human-computer interaction. This system analyzes vocal tone and frequency patterns to classify emotions such as happy, sad, angry, neutral, and more using supervised learning techniques.

---

## Features

- Audio preprocessing using Librosa and extraction of MFCC features  
- Deep learning model combining CNN and LSTM layers for sequential classification  
- Visualizations of waveform, spectrogram, and training curves  
- Confusion matrix and accuracy tracking for model evaluation  
- Structured dataset loading and label encoding for scalable training

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- Librosa  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## Workflow

1. Load and preprocess audio files using Librosa  
2. Extract MFCC features for each audio signal  
3. Build CNN-LSTM model architecture  
4. Train and evaluate the model on labeled data  
5. Visualize training loss/accuracy and confusion matrix  

---

## Dataset

- The dataset used is a labeled set of `.wav` files categorized by emotion.
- Audio samples are normalized, padded/truncated, and converted to feature arrays.
- Example emotions: happy, angry, fear, calm, neutral, etc.

Note: You can use datasets like RAVDESS, CREMA-D, or custom labeled sets.

---

## Evaluation Metrics

- Accuracy  
- Loss curves  
- Confusion matrix  
- Optional: F1-score, precision, recall via `sklearn.metrics`

---

## Future Enhancements

- Add support for real-time audio emotion detection via microphone input  
- Convert model to ONNX or TensorFlow Lite for deployment  
- Explore transfer learning or transformer-based audio models

---

## Contribution

Feel free to fork the repo, improve the model, or test it on new datasets. Pull requests are welcome.

---

## License

This project is licensed under the MIT License.
