import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MODEL")))

import streamlit as st
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms , datasets
from Model_architecture import CNNModel


# load the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# apply transforms
Transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Class names
datasets_train= datasets.ImageFolder("DATA/archive/training_set/training_set")
class_names = datasets_train.classes

# Load the model
model = CNNModel().to(device)
model.load_state_dict(torch.load("APP/cnn_model.pth", map_location=device))
model.eval()

# Function to predict the image
def predict_image(image):
    image = Transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        predicted_class = predicted_class.item()
        confidence = confidence.item() * 100
    return class_names[predicted_class], confidence

# Streamlit app UI
st.title("Cat vs Dog Image Classification")
st.write("Upload an image of a cat or a dog to classify it.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        label, confidence = predict_image(image)
        st.write(f"Prediction: {label} with confidence {confidence:.2f}%")
        if confidence < 50:
            st.warning("The model is not confident about this prediction. Please try another image.")
        else:
            st.success(f"The model is confident about this prediction: {label} with confidence {confidence:.2f}%")
else:
    st.write("Please upload an image.")

# Display class names
st.sidebar.header("Class Names")
for idx, class_name in enumerate(class_names):
    st.sidebar.write(f"{idx}: {class_name}")

# Display model architecture
st.sidebar.header("Model Architecture")
st.sidebar.text(str(model))
st.sidebar.text("This model is trained to classify images of cats and dogs.")

# Display device information
st.sidebar.header("Device Information")
st.sidebar.text(f"Using device: {device}")
st.sidebar.text("If you have a GPU, the model will run on it for faster predictions.")

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Developed by YamenRM")
st.sidebar.text("For more information, visit: [GitHub Repository](https://github.com/YamenRM/Cat-VS-Dog-DL-Classification)")
st.sidebar.text("This app is built using Streamlit and PyTorch.")

