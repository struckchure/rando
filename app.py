import streamlit as st
import torch
from PIL import Image
import numpy as np
import pathlib

# Ensure compatibility with file paths on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 🔹 Load the YOLOv5 model trained on cassava diseases
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# 🔹 Define class names manually if not present in the .pt file
model.names = [
    'Cassava Mosaic Disease', 
    'Cassava Brown Streak Disease', 
    'Cassava Green Mite', 
    'Healthy'
]

# 🔹 Streamlit page configuration
st.set_page_config(
    page_title="Cassava Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🔹 Page title and instructions
st.title("Cassava Disease Detection App")
st.write("Upload a cassava leaf image to detect whether it's healthy or infected.")

# 🔹 Sidebar - confidence threshold slider
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

# 🔹 Image uploader
uploaded_file = st.file_uploader("Upload a cassava leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')

    # Set model confidence threshold
    model.conf = confidence_threshold

    # Run inference
    results = model(image)

    # Render image with bounding boxes
    img_with_boxes = np.array(results.render()[0])
    img_with_boxes = Image.fromarray(img_with_boxes)

    # Display the processed image
    st.image(
        img_with_boxes, 
        caption=f'Detected: {len(results.xyxy[0])} region(s)', 
        use_container_width=True
    )

    # 🔹 Display detection results
    st.subheader("Detection Summary")
    if len(results.xyxy[0]) == 0:
        st.info("No disease or object detected above the confidence threshold.")
    else:
        for *box, conf, cls in results.xyxy[0]:
            class_id = int(cls)
            class_name = model.names[class_id]
            st.write(f"🔍 **{class_name}** — Confidence: `{conf:.2f}`")
else:
    st.warning("Please upload a cassava leaf image to start detection.")
