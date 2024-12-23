# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import logging
import av
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from typing import Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance configurations
FRAME_RATE = 15
PROCESS_EVERY_N_FRAMES = 2

class YOLOProcessor(VideoProcessorBase):
    def __init__(self, confidence: float, model):
        self._confidence = confidence
        self._model = model
        self._last_frame = None
        self._frame_count = 0
        self._last_process_time = time.time()
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Process every Nth frame to improve performance
        if self._frame_count % PROCESS_EVERY_N_FRAMES != 0:
            return av.VideoFrame.from_ndarray(
                self._last_frame if self._last_frame is not None else img, 
                format="bgr24"
            )
        
        current_time = time.time()
        # Skip processing if less than 1/FRAME_RATE seconds have passed
        if current_time - self._last_process_time < 1.0/FRAME_RATE:
            return av.VideoFrame.from_ndarray(
                self._last_frame if self._last_frame is not None else img, 
                format="bgr24"
            )
            
        try:
            results = self._model.predict(img, conf=self._confidence)
            annotated_frame = results[0].plot()
            self._last_frame = annotated_frame
            self._last_process_time = current_time
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            if self._last_frame is not None:
                annotated_frame = self._last_frame
            else:
                annotated_frame = img
                
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

@st.cache_resource
def get_yolo_model(model_path: str) -> Optional[object]:
    """Cache the YOLO model loading with error handling"""
    try:
        return load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def main():
    initialize_session_state()
    
    st.set_page_config(
        page_title="Underwater Object Detection",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Underwater Object Detection using YOLOv8")
    st.markdown("""
        This application performs real-time object detection on underwater imagery 
        using YOLOv8. Select your input source and model parameters from the sidebar.
    """)

    with st.sidebar:
        st.header("Model Configuration")
        
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

            model = get_yolo_model(f'weights/detection/{model_type}')
            
            if model is None:
                st.error("Failed to load model. Please check model path and weights.")
                st.stop()

        except Exception as e:
            st.error(f"Error in model configuration: {e}")
            st.stop()

        st.header("Input Source")
        source_type = st.selectbox(
            "Select Source",
            ["Image", "Video", "Webcam"],
            help="Choose your input source type"
        )

    try:
        if source_type == "Image":
            infer_uploaded_image(confidence, model)
        elif source_type == "Video":
            infer_uploaded_video(confidence, model)
        elif source_type == "Webcam":
            st.header("Live Detection")
            st.info("Note: Processing every 2nd frame to improve performance.")
            
            webrtc_ctx = webrtc_streamer(
                key="underwater-detection",
                video_processor_factory=lambda: YOLOProcessor(confidence, model),
                media_stream_constraints={
                    "video": {
                        "frameRate": {"ideal": FRAME_RATE},
                        "width": {"ideal": 640},
                        "height": {"ideal": 480}
                    },
                    "audio": False
                },
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]}
                    ]
                },
                async_processing=True
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()