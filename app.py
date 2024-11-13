import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Set the page configuration
st.set_page_config(page_title='Plant Disease Classifier', page_icon='🌿', layout='wide', initial_sidebar_state='expanded')

@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def load_and_preprocess_image(image, target_size=(225, 225)):
    img = Image.open(image).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = labels[predicted_class_index]
    return predicted_class_name, predictions

# Sidebar content
st.sidebar.header('🌿 Plant Disease Classifier')
st.sidebar.write("Upload an image of a plant leaf, and click **Classify** to predict the disease.")

# Main content
st.header('Plant Disease Classification')
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', width=300)  # Adjust width as needed

    classify_button = st.button('🌱 Classify Image')
    if classify_button:
        with st.spinner('Classifying...'):
            prediction, predictions = predict_image_class(model, uploaded_image)
        st.success(f'**Prediction:** {prediction}')
        # Display additional information if needed
        st.write("Additional information on the disease...")

else:
    st.info('Please upload an image to classify.')
