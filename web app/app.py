import streamlit as st
import torch
from PIL import Image
import numpy as np
import model_utils # Importing the file we just made
from pathlib import Path

# --- Configuration ---
MODEL_PATH = 'data/working/best_efficientnet_model.pth' # UPDATE THIS PATH if needed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(page_title="AI Image Detector", layout="wide")

# --- 1. Load Model (Cached so it doesn't reload every time) ---
@st.cache_resource
def load_model():
    model = model_utils.XceptionClassifier(num_classes=1, pretrained=False)
    # Load weights
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Could not find model at {MODEL_PATH}. Please check the path.")
        return None

model = load_model()
transform = model_utils.get_transform()

# --- 2. UI Layout ---
st.title("Ai Image Detector with Explainability")
st.markdown("Upload an image below.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # 1. Load and Display Original
    image = Image.open(uploaded_file).convert('RGB')
    
    # 2. Process
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    # 3. Inference & Grad-CAM
    grad_cam = model_utils.GradCAM(model)
    
    with st.spinner('Analyzing pixels...'):
        heatmap_raw, probability = grad_cam.generate_cam(input_tensor)
        
        # Determine Label
        label = "FAKE" if probability > 0.5 else "REAL"
        confidence = probability if probability > 0.5 else 1 - probability
        color = "red" if label == "FAKE" else "green"

    # 4. Display Results
    st.divider()
    
    # Header Result
    st.markdown(f"<h2 style='text-align: center; color: {color};'>Result: {label} ({confidence:.2%})</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("AI Attention (Heatmap)")
        # Generate overlay
        heatmap_img, overlay_img = model_utils.get_overlay(image, heatmap_raw)
        st.image(heatmap_img, use_container_width=True, caption="Warmer colors = High Attention")
        
    with col3:
        st.subheader("Explanation Overlay")
        st.image(overlay_img, use_container_width=True, caption="Where the model looked to decide")