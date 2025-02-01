import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
import pickle

# Load models
@st.cache_resource
def load_model(model_name):
    if model_name == "InceptionV3":
        return tf.keras.models.load_model("InceptionNet.h5")
    elif model_name == "EfficientNet":
        return tf.keras.models.load_model("EfficientNet.h5")
    elif model_name == "NASNetMobile":
        return tf.keras.models.load_model("NASNetMobile.h5")
    elif model_name == "Xception":
        return tf.keras.models.load_model("XceptionModel.h5")
    elif model_name == "ResNet50":
        return tf.keras.models.load_model("ResNet50.h5")
    elif model_name == "VGG16":
        return tf.keras.models.load_model("Vgg16.h5")
    elif model_name == "MobileNet":
        return tf.keras.models.load_model("MobileNet.h5")
    elif model_name == "CustomNet (DO NOT TRY)":
        return tf.keras.models.load_model("CustomNet.h5")
    else:
        return None
# Streamlit UI
st.title("POKEDEX")
st.write("A classifier for 1000 Pokemon found across the world!")

# Dropdown for model selection
model_name = st.selectbox("Choose a Model:", ["InceptionV3", "EfficientNet", "NAS", "Xception", "ResNet50", "VGG16", "MobileNet", "CustomNet (DO NOT TRY)"])
model = load_model(model_name)

# Function to preprocess the image based on selected model
# def preprocess_image(image, model_name):
#     img = load_img(r'C:\Users\Aryan\Documents\CodesAndDatasets\SEM 6\CVA\CVA_assgn_pokemon\CompVision_Architechtures\Pokemon_Pikachu_art.png', target_size=(128, 128))
#     img = img_to_array(img)
#     img = img / 255.0
#     return image

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(128, 128))

    # Preprocess image with correct size
    processed_image = img_to_array(image)
    processed_image = processed_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)


    # Predict
    prediction = model.predict(processed_image)

    # Class labels (update if needed)
    with open('class_labels.pkl', 'rb') as f:
        class_labels = pickle.load(f)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image")
    
    with col2:
        st.write(f"### **Prediction: {predicted_class}**")
        st.write(f"**Confidence: {confidence:.2f}%**")



