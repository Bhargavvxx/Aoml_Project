import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Skin Cancer Classifier", page_icon="🎗️", layout="centered")

# Load trained models
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("skin_cancer_cnn_model.h5")
    xgb_model = joblib.load('skin_cancer.pkl')
    return model, xgb_model

cnn_model, xgb_model = load_models()

# Preprocessing function
def preprocess_image(image, img_size=(224, 224)):
    image = image.resize(img_size)
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

# Skin condition descriptions
condition_info = {
    "Actinic keratoses": "Precancerous skin patches caused by sun damage, which can develop into skin cancer.",
    "Basal cell carcinoma": "A common type of skin cancer that grows slowly and rarely spreads, often appearing as a shiny bump.",
    "Benign keratosis": "A non-cancerous growth on the skin that may appear warty or scaly, usually harmless.",
    "Dermatofibroma": "A benign skin growth that feels like a small, firm bump, often appearing on the lower legs.",
    "Melanoma": "A serious type of skin cancer that develops in pigment-producing cells and can spread rapidly.",
    "Nevus": "Commonly known as a mole, a nevus is a benign skin growth made of pigment-producing cells.",
    "Vascular lesion": "An abnormal growth of blood vessels in the skin, often appearing red or purple in color."
}

# Streamlit UI
st.title("🎗️ Skin Cancer Classification")
st.write("Upload an image to classify it using our AI-powered model.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with st.spinner("Processing image..."):
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)

        # Preprocess image
        processed_image = preprocess_image(image)

        # Feature extraction using trained CNN
        features = cnn_model.predict(processed_image, verbose=0)
        features = features.reshape(1, -1)  # Ensure correct shape

        # Prediction using XGBoost
        prediction = xgb_model.predict(features)

        # Load class labels
        class_labels = [
            "Actinic keratoses", "Basal cell carcinoma", "Benign keratosis",
            "Dermatofibroma", "Melanoma", "Nevus", "Vascular lesion"
        ]
        result = class_labels[prediction[0]] if prediction[0] < len(class_labels) else "Unknown"

        st.success(f"🎯 **Prediction:** {result}")
        
        # Display additional information about the predicted condition
        if result in condition_info:
            st.info(f"📝 **About {result}:** {condition_info[result]}")
        
        # Additional info
        st.markdown(
            "💡 *Note: This model is an AI-based classifier and should not be used as a replacement for medical diagnosis. Please consult a dermatologist for professional advice.*"
        )