# app.py
import streamlit as st
import cv2
import logging
import av
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from config import YOLO_WEIGHTS, SOURCES_LIST
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Optional
from dataclasses import dataclass
from ultralytics import YOLO
from time import perf_counter

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Improved performance configurations
@dataclass
class ProcessingConfig:
    FRAME_RATE: int = 10  # Reduced from 15
    PROCESS_EVERY_N_FRAMES: int = 3  # Increased from 2
    MAX_RESOLUTION: tuple = (480, 360)  # Reduced from 640x480
    THREAD_POOL_SIZE: int = 1  # Reduced from 2
    MAX_QUEUE_SIZE: int = 5  # Reduced from 10
    BATCH_SIZE: int = 1
    MAX_RETRIES: int = 3
    TIMEOUT: float = 0.5

# Model loading function with caching
@st.cache_resource
def get_yolo_model(model_path: str) -> Optional[YOLO]:
    """Load and cache YOLO model"""
    try:
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
def get_ice_servers():
    """Get ICE servers configuration with improved fallback options"""
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]
    
    try:
        # Add Twilio servers if available
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        
        if account_sid and auth_token:
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            ice_servers.extend(token.ice_servers)
    except Exception as e:
        logger.warning(f"Twilio configuration not available: {e}")
    
    return RTCConfiguration(iceServers=ice_servers)

class YOLOProcessor:
    def __init__(self, confidence: float, model):
        self.config = ProcessingConfig()
        self._confidence = confidence
        self._model = model
        self._frame_queue = queue.Queue(maxsize=self.config.MAX_QUEUE_SIZE)
        self._result_queue = queue.Queue(maxsize=self.config.MAX_QUEUE_SIZE)
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.THREAD_POOL_SIZE)
        self._processing = False
        self._lock = threading.Lock()
        self._last_result = None
        self._frame_count = 0
        self._error_count = 0
        self._last_process_time = perf_counter()
        
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with error handling and performance monitoring"""
        try:
            # Skip processing if too soon
            current_time = perf_counter()
            if current_time - self._last_process_time < self.config.TIMEOUT:
                return self._last_result if self._last_result is not None else frame

            # Resize frame for better performance
            if frame.shape[1] > self.config.MAX_RESOLUTION[0] or frame.shape[0] > self.config.MAX_RESOLUTION[1]:
                frame = cv2.resize(frame, self.config.MAX_RESOLUTION, interpolation=cv2.INTER_AREA)

            # Process frame with batching
            results = self._model.predict(
                frame,
                conf=self._confidence,
                batch=self.config.BATCH_SIZE,
            )
            
            processed_frame = results[0].plot()
            self._last_process_time = current_time
            return processed_frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self._error_count += 1
            return self._last_result if self._last_result is not None else frame
            
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Receive and process frames with improved error handling"""
        if self._error_count >= self.config.MAX_RETRIES:
            logger.error("Too many errors, stopping processing")
            return frame

        try:
            self._frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Skip frames based on counter
            if self._frame_count % self.config.PROCESS_EVERY_N_FRAMES != 0:
                return av.VideoFrame.from_ndarray(
                    self._last_result if self._last_result is not None else img,
                    format="bgr24"
                )

            with self._lock:
                processed = self._process_frame(img)
                self._last_result = processed

            return av.VideoFrame.from_ndarray(processed, format="bgr24")

        except Exception as e:
            logger.error(f"Frame receive error: {e}")
            return frame

    def __del__(self):
        """Cleanup resources properly"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)

def setup_webcam_interface(confidence: float, model):
    """Setup WebRTC interface with improved error handling and memory management"""
    st.header("Live Detection")
    
    status_placeholder = st.empty()
    webrtc_ctx = None
    
    try:
        webrtc_ctx = webrtc_streamer(
            key="underwater-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=get_ice_servers(),
            video_processor_factory=lambda: YOLOProcessor(confidence, model),
            async_processing=True,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": ProcessingConfig.MAX_RESOLUTION[0]},
                    "height": {"ideal": ProcessingConfig.MAX_RESOLUTION[1]},
                    "frameRate": {"ideal": ProcessingConfig.FRAME_RATE}
                },
            }
        )

        if webrtc_ctx.state.playing:
            status_placeholder.success("✅ Stream active")
        else:
            status_placeholder.warning("⚠️ Stream inactive")

    except Exception as e:
        logger.error(f"WebRTC setup error: {e}")
        st.error("Failed to initialize webcam. Please check browser compatibility and permissions.")
    
    return webrtc_ctx

def main():
    # Memory management
    if not hasattr(st, 'processor'):
        st.processor = None

    st.set_page_config(
        page_title="Underwater Object Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        model_type = st.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
        confidence = st.slider("Confidence", 0.0, 1.0, 0.5)
        
        model = get_yolo_model(str(YOLO_WEIGHTS[model_type]))
        if not model:
            st.error("Model loading failed")
            return

        source_type = st.selectbox("Input Source", SOURCES_LIST)

    if source_type == "Webcam":
        if st.processor:
            st.processor.__del__()
        st.processor = setup_webcam_interface(confidence, model)
    elif source_type == "Image":
        infer_uploaded_image(confidence, model)
    elif source_type == "Video":
        infer_uploaded_video(confidence, model)

    # Performance monitoring
    if st.processor:
        st.sidebar.markdown("### Performance Stats")
        st.sidebar.text(f"Frame Rate: {ProcessingConfig.FRAME_RATE}")
        st.sidebar.text(f"Resolution: {ProcessingConfig.MAX_RESOLUTION}")

if __name__ == "__main__":
    main()