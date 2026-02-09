# app_fixed.py - WORKING WEB APP
import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import os

# Page setup
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# Title
st.title("🩺 AI Skin Cancer Detector")
st.markdown("Upload a skin lesion image for instant analysis")

# Sidebar
with st.sidebar:
    st.header("System Info")
    st.write("**Model:** Random Forest Classifier")
    st.write("**Accuracy:** ~75-80%")
    st.write("**Classes:** Benign vs Melanoma")
    st.divider()
    st.caption("Capstone Project - CSE Department")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('skin_cancer_model.joblib')
        return model
    except:
        st.error("❌ Model not found. Please run train_simple.py first")
        return None

model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # Analyze button
    if st.button("🔍 Analyze Image", type="primary") and model:
        with st.spinner("Processing image..."):
            # Convert image
            img_array = np.array(image)
            
            # Preprocess (same as training)
            img_array = cv2.resize(img_array, (100, 100))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_array = img_array.flatten() / 255.0
            img_array = img_array.reshape(1, -1)
            
            # Predict
            prediction = model.predict(img_array)[0]
            probabilities = model.predict_proba(img_array)[0]
            
            # Show results
            st.success("✅ Analysis Complete!")
            
            # Diagnosis
            diagnosis = "Benign Nevus (Not Cancerous)" if prediction == 0 else "Melanoma (Potentially Cancerous)"
            confidence = probabilities[prediction] * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Diagnosis", diagnosis)
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probability bars
            st.subheader("Detailed Probabilities")
            
            st.progress(probabilities[0])
            st.write(f"**Benign Nevus:** {probabilities[0]:.1%}")
            
            st.progress(probabilities[1])
            st.write(f"**Melanoma:** {probabilities[1]:.1%}")
            
            # Warning for melanoma
            if prediction == 1 and confidence > 60:
                st.error("⚠️ **Warning:** Possible melanoma detected. Please consult a dermatologist.")

# Footer
st.divider()
st.caption("Department of Computer Science & Engineering | GITAM University")