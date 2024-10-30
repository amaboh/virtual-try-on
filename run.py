import streamlit as st
import numpy as np
from PIL import Image
import face_recognition
from io import BytesIO

# Configure Streamlit page
st.set_page_config(page_title="Virtual Lip Depigmentation Try-On", layout="wide")

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def apply_color_to_lips(image_array, face_landmarks, color, intensity=0.5):
    """Apply color to lips using PIL and numpy"""
    if not face_landmarks:
        return image_array
    
    # Create mask for lips
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get lip landmarks
    top_lip = face_landmarks['top_lip']
    bottom_lip = face_landmarks['bottom_lip']
    
    # Convert landmarks to numpy arrays
    top_lip = np.array(top_lip, dtype=np.int32)
    bottom_lip = np.array(bottom_lip, dtype=np.int32)
    
    # Fill lip regions
    cv2.fillPoly(mask, [top_lip], 255)
    cv2.fillPoly(mask, [bottom_lip], 255)
    
    # Create color overlay
    color_layer = np.full_like(image_array, color)
    
    # Blend
    mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
    blended = (1 - intensity) * image_array + intensity * color_layer
    result = np.where(mask_3d > 0, blended, image_array)
    
    return result.astype(np.uint8)

# Predefined color gradients
color_gradients = {
    "Red Passion": ["#FF0000", "#8B0000", "#DC143C"],
    "Pink Love": ["#FF69B4", "#FF1493", "#C71585"],
    "Nude": ["#DEB887", "#D2691E", "#8B4513"],
    "Berry": ["#8B008B", "#800080", "#4B0082"],
    "Custom": []
}

st.title("Virtual Lip Depigmentation Try-On")
st.write("Upload a photo and try different lipstick colors!")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
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
    # Load image
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    # Detect face landmarks
    face_locations = face_recognition.face_locations(image_array)
    face_landmarks_list = face_recognition.face_landmarks(image_array, face_locations)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image)
    
    with col2:
        st.subheader("With Lip Depigmentation")
        if face_landmarks_list:
            # Apply color to first detected face
            color_rgb = hex_to_rgb(selected_color)
            result = apply_color_to_lips(image_array, face_landmarks_list[0], color_rgb, intensity)
            st.image(result)
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