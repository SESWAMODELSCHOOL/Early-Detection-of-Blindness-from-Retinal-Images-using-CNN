import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the binary classification model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_fusion_model.keras")

model = load_model()

# Page 1: Project Details
def project_details():
    st.title("Early Detection of Severe Diabetic Retinopathy")
    
    st.header("Introduction")
    st.write(
        "Diabetic Retinopathy (DR) is one of the leading causes of blindness globally. "
        "Early detection can significantly reduce the risk of severe vision loss. This project uses a fusion-based CNN model "
        "to classify retinal images as either 'No DR' or 'Severe DR' with high accuracy."
    )
    
    st.header("Dataset Used")
    st.write(
        "The dataset used in this project is the APTOS 2019 Blindness Detection dataset. "
        "It contains fundus images of retinas categorized into five severity levels. For this binary model, "
        "we used images labeled as 'No DR' (0) and 'Severe DR' (4)."
    )

    st.header("Model Performance")
    performance_data = {
    "Model Name": [
        "Multiclass",
        "Binary",
        "No DR vs Severe DR",
        "Balanced Data"
    ],
    "Number of Classes": [3, 2, 2, 3],
    "Accuracy": ["84.4%", "95.8%", "97.6%", "78.3%"],
    "Precision": ["83.7%", "95.8%", "97.6%", "77.1%"],
    "Recall": ["84.4%", "95.8%", "97.6%", "78.3%"],
    "F1 Score": ["82.2%", "95.8%", "97.6%", "76.7%"],
}

    performance_table = pd.DataFrame(performance_data)
    st.table(performance_table)

    st.header("Best Model")
    st.write(
        "The best model for this task is a fusion-based CNN combining EfficientNetB0 and InceptionV3 trained with NO DR and Severe DR Classes. "
        "It achieves exceptional performance on the binary classification task of distinguishing 'No DR' from 'Severe DR'."
    )

# Page 2: Model Inference
def model_inference():
    st.title("Diabetic Retinopathy Detection")

    uploaded_file = st.file_uploader("Upload a fundus image (PNG format)", type=["png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to array and ensure 3 channels
        image_array = np.array(image)
        if image_array.shape[-1] == 4:  # Convert RGBA to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Preprocess the image
        image_resized = cv2.resize(image_array, (224, 224))
        image_normalized = image_resized / 255.0  # Normalize pixel values
        image_input = np.expand_dims(image_normalized, axis=0)  # Add batch dimension

        # Predict
        if st.button("Predict"):
            prediction = model.predict(image_input)
            class_label = "No DR" if prediction[0] < 0.5 else "Severe DR"

            # Display the prediction
            st.header("Prediction Result")
            st.write(f"The model predicts: **{class_label}**")

# Streamlit App Layout
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Project Details", "Model Inference"])

    if page == "Project Details":
        project_details()
    elif page == "Model Inference":
        model_inference()

if __name__ == "__main__":
    main()
