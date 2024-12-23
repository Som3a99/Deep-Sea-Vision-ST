# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
from pathlib import Path
import logging
import av
from utils import load_model, infer_uploaded_image, infer_uploaded_video

# Must be the first Streamlit command
st.set_page_config(
    page_title="Underwater Object Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOProcessor(VideoProcessorBase):
    def __init__(self, confidence: float, model):
        self._confidence = confidence
        self._model = model
        self._last_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        try:
            results = self._model.predict(img, conf=self._confidence)
            annotated_frame = results[0].plot()
            self._last_frame = annotated_frame
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            if self._last_frame is not None:
                annotated_frame = self._last_frame
            else:
                annotated_frame = img
                
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

@st.cache_resource
def get_yolo_model(model_path):
    """Cache the YOLO model loading"""
    try:
        return load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def main():
    # Main page heading
    st.title("Underwater Object Detection using YOLOv8")
    st.markdown("""
        This application performs real-time object detection on underwater imagery 
        using YOLOv8. Select your input source and model parameters from the sidebar.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Model Configuration")
        
        # Model selection with error handling
        try:
            model_type = st.selectbox(
                "Select Model",
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                help="Smaller models (n, s) are faster but less accurate"
            )
            
            confidence = st.slider(
                "Detection Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher values reduce false positives"
            )

            model_path = Path('weights/detection') / model_type
            model = get_yolo_model(model_path)
            
            if model is None:
                st.error("Failed to load model. Please check model path and weights.")
                st.stop()

        except Exception as e:
            st.error(f"Error in model configuration: {e}")
            st.stop()

        # Source selection
        st.header("Input Source")
        source_type = st.selectbox(
            "Select Source",
            ["Image", "Video", "Webcam"],
            help="Choose your input source type"
        )

    # Main content area
    try:
        if source_type == "Image":
            infer_uploaded_image(confidence, model)
        elif source_type == "Video":
            infer_uploaded_video(confidence, model)
        elif source_type == "Webcam":
            st.header("Live Detection")
            st.info("Note: Webcam performance may vary based on your connection speed.")
            
            webrtc_ctx = webrtc_streamer(
                key="underwater-detection",
                video_processor_factory=lambda: YOLOProcessor(confidence, model),
                media_stream_constraints={
                    "video": {"frameRate": {"ideal": 10}},
                    "audio": False
                },
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()