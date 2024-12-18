import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image
import cv2
from ultralytics import YOLO
import tempfile
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from pathlib import Path  # Import Path to handle file paths dynamically

# Setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Interactive Interface for YOLOv8")

# Sidebar
st.sidebar.header("DL Model Config")

# Model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

if task_type != "Detection":
    st.error("Currently only 'Detection' is implemented")

model_type = st.sidebar.selectbox(
    "Select Model",
    config.DETECTION_MODEL_LIST
)

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = Path(config.DETECTION_MODEL_DIR, model_type)
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# Image/Video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

if source_selectbox == "Image":
    infer_uploaded_image(confidence, model)
elif source_selectbox == "Video":
    infer_uploaded_video(confidence, model)
elif source_selectbox == "Webcam":
    st.header("Live Detection (Webcam)")
    class YOLOProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = confidence
            self.model = model

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(img, conf=self.conf)
            annotated_frame = results[0].plot()
            return annotated_frame

    webrtc_streamer(
        key="live-detection",
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
    )
else:
    st.error("Invalid Source Selected")
