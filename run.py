import os

# Install required system packages
os.system('apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev')

# Rest of your imports
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def apply_lipstick(image, landmarks, hex_color, intensity=0.5):
    color = hex_to_bgr(hex_color)
    h, w, _ = image.shape

    lip_landmarks = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 62
    ]

    lip_points = np.array([(int(landmarks[point].x * w), int(landmarks[point].y * h)) for point in lip_landmarks])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, lip_points, 255)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    color_overlay = np.full((h, w, 3), color, dtype=np.uint8)
    lip_region = cv2.bitwise_and(image, image, mask=mask)
    blended = cv2.addWeighted(lip_region, 1 - intensity, cv2.bitwise_and(color_overlay, color_overlay, mask=mask), intensity, 0)
    mask_norm = mask.astype(float) / 255.0
    mask_norm = np.expand_dims(mask_norm, axis=2)
    result = image.copy()
    result = (1 - mask_norm) * result + mask_norm * blended
    return result.astype(np.uint8)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Streamlit UI
st.set_page_config(page_title="Virtual Lipstick Try-On", layout="wide")

st.title("Virtual Lipstick Try-On")
st.write("Upload a photo and try different lipstick colors!")

# Predefined color gradients
color_gradients = {
    "Red Passion": ["#FF0000", "#8B0000", "#DC143C"],
    "Pink Love": ["#FF69B4", "#FF1493", "#C71585"],
    "Nude": ["#DEB887", "#D2691E", "#8B4513"],
    "Berry": ["#8B008B", "#800080", "#4B0082"],
    "Custom": []  # For custom color selection
}

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Color gradient selection
    selected_gradient = st.selectbox("Choose Color Gradient", list(color_gradients.keys()))
    
    if selected_gradient == "Custom":
        custom_color = st.color_picker("Pick a custom color", "#FF0000")
        selected_color = custom_color
    else:
        selected_color = st.selectbox("Select Shade", color_gradients[selected_gradient])
    
    intensity = st.slider("Color Intensity", 0.0, 1.0, 0.5, 0.1)
    
    st.markdown("---")
    st.markdown("### Current Color")
    st.markdown(
        f'<div style="background-color: {selected_color}; height: 50px; border-radius: 5px;"></div>',
        unsafe_allow_html=True
    )
    if st.button("Copy Color Code"):
        st.code(selected_color)

# Main content area
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and process the image
    image_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = face_mesh.process(image_rgb)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb)
    
    with col2:
        st.subheader("With Lipstick")
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                result = apply_lipstick(image, face_landmarks.landmark, selected_color, intensity)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb)
        else:
            st.error("No face detected in the image. Please try another photo.")

# Footer
st.markdown("---")
st.markdown("""
    ### How to use:
    1. Upload a clear front-facing photo
    2. Choose a color gradient from the sidebar
    3. Adjust the intensity if needed
    4. Copy the color code if you want to save it for later
""")