import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import gdown  # For downloading the model from Google Drive

# Set the page configuration
st.set_page_config(
    page_title='Plant Disease Classifier',
    page_icon='🌿',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Google Drive model file ID
GDRIVE_FILE_ID = '1oY_2ICONDEokPjzoNcguP4E4WKmJmN05'

# Function to download the model from Google Drive if not already present
@st.cache_resource
def download_model():
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        with st.spinner('Downloading model...'):
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, model_path, quiet=False)
    return model_path

# Function to load the model
@st.cache_resource
def load_model():
    model_path = download_model()
    return tf.keras.models.load_model(model_path)

# Load the model
model = load_model()

# Class labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Blights'}

# Function to load and preprocess an image
def load_and_preprocess_image(image, target_size=(225, 225)):
    img = Image.open(image).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.  # Normalize pixel values
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = labels[predicted_class_index]
    return predicted_class_name, predictions

# Apply custom CSS styling
def apply_custom_styles():
    css_styles = """
        body {
            font-family: 'Arial', sans-serif;
        }
        .stSidebar > div:first-child {
            background-color: #f0f9e8;
            color: #346751;
        }
        .css-1tv6tad {
            background-color: #aacc96;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        button {
            border: 2px solid #346751;
            color: white;
            background-color: #346751;
            border-radius: 5px;
            font-size: 16px;
        }
        footer {
            visibility: hidden;
        }
    """
    st.markdown(f'<style>{css_styles}</style>', unsafe_allow_html=True)

# Apply styles
apply_custom_styles()

# Sidebar content
st.sidebar.header('🌿 Plant Disease Classifier')
st.sidebar.write("Upload an image of a plant leaf and click **Classify** to predict the disease.")

# Main content
st.header('Plant Disease Classification')
uploaded_image = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# Display uploaded image and classify
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', width=300)

    classify_button = st.button('🌱 Classify Image')
    if classify_button:
        with st.spinner('Classifying...'):
            prediction, predictions = predict_image_class(model, uploaded_image)
        st.success(f'**Prediction:** {prediction}')
        st.write("Additional information on the disease if needed.")

else:
    st.info('Please upload an image to classify.')
