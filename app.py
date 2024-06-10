import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Image dimensions
IMG_W = 100
IMG_H = 200
CHANNELS = 3  # Assuming RGB images

# Load the trained model
model = load_model('UASPCD.keras')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Read the image in color (ensure color mode is compatible with model)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply Canny Edge Detection
    edges = cv2.Canny(image, 40, 80)

    # Convert single-channel image to three-channel image
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)



    return edges_rgb

# Function to make predictions using the loaded model
def predict_image(image):
    edges_resized = cv2.resize(image, (IMG_W, IMG_H))

    # Normalize the image (assuming model expects normalized values)
    edges_resized = edges_resized.astype('float32') / 255.0

    # Expand dimensions for batch processing (even for single image)
    edges_resized = np.expand_dims(edges_resized, axis=0)
    prediction = model.predict(edges_resized)

    return prediction

# Streamlit app
st.title("Image Prediction App")

# Define the file uploader within the main function
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels='BGR', caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for prediction
    img_array = preprocess_image(image)
        # Resize the edge-detected image
    st.image(img_array, channels='BGR', caption='Preprocessed Image', use_column_width=True)

    # Make prediction
    prediction = predict_image(img_array)

    # You can interpret the prediction based on your model's output layer
    if prediction[0][0] > 0.5:
        st.write(f'Prediction: Pneumonia ({prediction[0][0]:.4f})')
    else:
        st.write(f'Prediction: Normal ({prediction[0][0]:.4f})')
