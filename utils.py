from ultralytics import YOLO
import streamlit as st
from PIL import Image
import tempfile
import cv2
import numpy as np
import time

@st.cache_resource
def load_model(model_path: str):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def process_video_frame(frame: np.ndarray, model, conf: float):
    results = model.predict(frame, conf=conf)
    return results[0].plot(), results[0].boxes

def infer_uploaded_image(conf: float, model):
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Detection"):
            results = model.predict(image, conf=conf)
            st.image(results[0].plot(), caption="Detected Image")

def infer_uploaded_video(conf: float, model):
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        vid_cap = cv2.VideoCapture(tfile.name)
        if st.button("Run Detection"):
            while vid_cap.isOpened():
                success, frame = vid_cap.read()
                if not success:
                    break
                frame, _ = process_video_frame(frame, model, conf)
                st.image(frame, channels="BGR")
