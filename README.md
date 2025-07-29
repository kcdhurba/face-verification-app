# 🧠 Face Detection & Verification App

A real-time face detection and face verification web application built using **Python**, **OpenCV**, and deep learning models. This app can not only detect faces in real-time from webcam or images, but also **verify whether two faces belong to the same person**.

## 🚀 Features

- 🔍 Real-time face detection using webcam  
- 🖼️ Detect faces in uploaded images  
- 🎯 Face verification: compare two faces to verify identity  
- 💡 User-friendly interface
- 📦 Clean and modular codebase  

## 🛠️ Tech Stack

- Python 3.10
- OpenCV  
- Tensorflow(for verification)  
- Haar Cascade(for detection)  
- Streamlit  
- NumPy(for image handling)

## 🧩 How It Works

### Face Detection:
1. Load face detection model (e.g., Haar Cascade)
2. Capture frame from webcam or image
3. Detect and draw bounding boxes around all faces

### Face Verification:
1. Extract facial embeddings using deep learning model
2. Compare embeddings using cosine or Euclidean distance
3. Return match result and confidence score

