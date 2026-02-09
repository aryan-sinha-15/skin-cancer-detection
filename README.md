# 🩺 Skin Cancer Detection AI

## 📋 Project Overview
AI-powered system for detecting skin cancer from dermoscopic images using machine learning.

## 🚀 Features
- Upload skin lesion images
- AI analysis with confidence scores
- Benign vs Melanoma classification
- Web interface for easy use

## 📁 Project Structure
skin-cancer-detection/
├── app_fixed.py # Web application
├── train_simple.py # AI model training
├── skin_cancer_model.joblib # Trained model
├── data_loader.py # Data processing
└── HAM10000_metadata.csv # Dataset labels


## 🛠️ Installation
```bash
pip install streamlit scikit-learn opencv-python joblib


How to Run
Train model: python train_simple.py

Run app: streamlit run app_fixed.py
