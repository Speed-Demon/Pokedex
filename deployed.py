import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import plot_model
import pickle
import visualkeras
from io import BytesIO

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Inference", "Architectures", "Benchmarks"])

# Function to load model
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "InceptionV3": "InceptionNet.h5",
        "EfficientNet": "EfficientNet.h5",
        "NASNetMobile": "NASNetMobile.h5",
        "Xception": "XceptionModel.h5",
        "ResNet50": "ResNet50.h5",
        "VGG16": "Vgg16.h5",
        "MobileNet": "MobileNet.h5",
        "CustomCNN": "CustomNet.h5",
    }
    return tf.keras.models.load_model(model_paths.get(model_name))

with open('class_labels.pkl', 'rb') as f:
    class_labels = pickle.load(f)

# Page: Inference
if page == "Inference":
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image("pokedex_icon.png", width=100)  # Adjust path if needed
    with col2:
        st.title("Pokédex")
    st.write("A classifier for 1000 Pokémon found across the world!")

    # Model Selection
    model_name = st.selectbox(
        "Choose a Model:", 
        ["InceptionV3", "EfficientNet", "NASNetMobile", "Xception", "ResNet50", "VGG16", "MobileNet", "CustomCNN"]
    )
    model = load_model(model_name)

    # Make sure pokemon exists in class labels
    pokemon_name = st.text_input("Enter Pokémon Name:")
    if str.strip(str.lower(pokemon_name)) in class_labels and pokemon_name != "":
        st.write("✅ Good to go!")
    if str.strip(str.lower(pokemon_name)) not in class_labels and pokemon_name != "":
        st.error("❌ Pokémon not found in database. Please enter a valid Pokémon name.")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = load_img(uploaded_file, target_size=(128, 128))
        processed_image = img_to_array(image) / 255.0
        processed_image = np.expand_dims(processed_image, axis=0)

        prediction = model.predict(processed_image)

        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image")
        with col2:
            st.write(f"### **Prediction: {predicted_class}**")
            st.write(f"**Confidence: {confidence:.2f}%**")
            st.link_button(label=f"View {str.capitalize(predicted_class)}'s Pokedex Entry on Bulbapedia", url=f"https://bulbapedia.bulbagarden.net/wiki/{predicted_class}")

# Page: Architectures
elif page == "Architectures":
    st.title("Model Architectures")
    model_name = st.selectbox(
        "Choose a Model to Visualize:", 
        ["InceptionV3", "EfficientNet", "NASNetMobile", "Xception", "ResNet50", "VGG16", "MobileNet", "CustomCNN"]
    )
    model = load_model(model_name)

    # Model Architecture Image
    try:
        st.image(f"{model_name}_architecture.png", caption=f"{model_name} Architecture", use_column_width=True)
    except:
        pass

    # Visual Keras Plot
    st.subheader("Layer Visualization")
    vis_image = visualkeras.layered_view(model, legend=True)  # Returns a PIL image
    st.image(vis_image, caption=f"{model_name} Layers")

    # Model Summary
    st.subheader("Model Summary")
    buffer = BytesIO()
    model.summary(print_fn=lambda x: buffer.write(f"{x}\n".encode()))
    st.code(buffer.getvalue().decode(), language="text")

# Page: Benchmarks
elif page == "Benchmarks":
    st.title("Model Benchmarks")
    st.write("Benchmark results for various models.")

    import pandas as pd

    benchmark_data = pd.DataFrame({
        "Model": ["InceptionV3", "EfficientNet", "NASNetMobile", "Xception", "ResNet50", "VGG16", "MobileNet", "CustomCNN"],
        "Accuracy":          [0.7142, 0.9678, 0.7830, 0.8308, 0.0260, 0.7516, 0.9247, 5.1908e-04],
        "Weighted F1 Score": [0.7040, 0.9667, 0.7773, 0.8270, 0.0137, 0.7453, 0.9239, 0.0000],
        "Inference Time (s)":[5.89, 5.52, 12.20, 2.63, 3.46, 1.00, 1.68, 0.82],
        "Training Time (s)": [173.30, 512.98, 229.45, 294.69, 202.72, 272.40, 93.55, 186.97]
    })

    # Display as a table
    st.subheader("Performance Table")
    st.dataframe(benchmark_data, use_container_width=True)

    # Model selection for detailed plots
    st.subheader("Select a Model for Accuracy & Loss Plots")
    selected_model = st.selectbox("Choose a Model:", benchmark_data["Model"])

    # Placeholder for plots (Replace with actual image paths)
    accuracy_plot_path = f"{selected_model}_accuracy.png"
    loss_plot_path = f"{selected_model}_loss.png"

    col1, col2 = st.columns(2)

    with col1:
        st.image(accuracy_plot_path, caption=f"{selected_model} Accuracy Plot", use_column_width=True)

    with col2:
        st.image(loss_plot_path, caption=f"{selected_model} Loss Plot", use_column_width=True)

