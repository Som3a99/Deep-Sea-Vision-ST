# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import logging
import av
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from config import YOLO_WEIGHTS, SOURCES_LIST
from typing import Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance configurations
FRAME_RATE = 8
PROCESS_EVERY_N_FRAMES = 5

def initialize_session_state():
    """Initialize session state variables."""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_source' not in st.session_state:
        st.session_state.current_source = None
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

class YOLOProcessor(VideoProcessorBase):
    def __init__(self, confidence: float, model):
        self._confidence = confidence
        self._model = model
        self._last_frame = None
        self._frame_count = 0
        self._error_count = 0
        self._max_errors = 3
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            self._frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Process only every Nth frame to reduce compute load
            if self._frame_count % PROCESS_EVERY_N_FRAMES != 0:
                return av.VideoFrame.from_ndarray(
                    self._last_frame if self._last_frame is not None else img,
                    format="bgr24"
                )
            
            img = cv2.resize(img, (320, 240))  # Optimize for performance
            
            results = self._model.predict(img, conf=self._confidence)
            annotated_frame = results[0].plot()
            self._last_frame = annotated_frame
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            self._error_count += 1
            if self._error_count > self._max_errors:
                logger.warning("Too many errors, returning original frame.")
            return av.VideoFrame.from_ndarray(
                self._last_frame if self._last_frame is not None else img,
                format="bgr24"
            )

@st.cache_resource
def get_yolo_model(model_path: str) -> Optional[object]:
    """Cache the YOLO model loading with error handling."""
    try:
        model = load_model(model_path)
        st.session_state.model_loaded = True
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.session_state.model_loaded = False
        return None

def setup_webcam_interface(confidence: float, model):
    st.header("Live Detection")

    st.warning("""
        If the stream doesn't start:
        - Ensure camera permissions are granted.
        - Check network restrictions.
        - Use a different browser (Chrome/Firefox).
    """)

    rtc_configuration = {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},  # Google STUN server
            {"urls": "stun:stun1.l.google.com:19302"},  # Backup STUN server
            {"urls": "stun:stun2.l.google.com:19302"},  # Additional STUN server
        ]
    }

    try:
        webrtc_ctx = webrtc_streamer(
            key="detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=lambda: YOLOProcessor(confidence, model),
        )
        if webrtc_ctx.state.playing:
            st.success("Webcam is live!")
    except Exception as e:
        st.error("Error initializing webcam. Please check your setup.")
        logger.error(f"WebRTC setup error: {e}")


def main():
    initialize_session_state()

    st.set_page_config(
        page_title="Underwater Object Detection",
        page_icon="🌊",
        layout="wide"
    )

    st.title("Underwater Object Detection using YOLOv8")
    st.markdown("""
        Detect underwater objects in real time with YOLOv8.
        Select your model and input source to get started.
    """)

    with st.sidebar:
        st.header("Model Configuration")
        model_type = st.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
        confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)
        
        model = get_yolo_model(str(YOLO_WEIGHTS[model_type]))
        if not model:
            st.error("Model loading failed. Check the model path.")
            st.stop()
        
        source_type = st.selectbox("Select Input Source", SOURCES_LIST)

    if source_type == "Image":
        infer_uploaded_image(confidence, model)
    elif source_type == "Video":
        infer_uploaded_video(confidence, model)
    elif source_type == "Webcam":
        setup_webcam_interface(confidence, model)

if __name__ == "__main__":
    main()
