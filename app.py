import streamlit as st
import cv2
import logging
import av
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
from utils import load_model, infer_uploaded_image, infer_uploaded_video, cleanup_memory
from config import YOLO_WEIGHTS, SOURCES_LIST, APP_CONFIG
import threading
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO
from time import perf_counter
import psutil
import gc
import torch
from collections import defaultdict
import time

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

@dataclass
class ProcessingConfig:
    """Dynamic configuration with automatic resource management"""
    frame_rate: int = 10
    process_interval: int = 2
    resolution: tuple = (640, 480)
    batch_size: int = 1
    timeout: float = 0.1
    max_retries: int = 3
    
    @classmethod
    def create(cls):
        try:
            mem_available = psutil.virtual_memory().available / (1024 * 1024)
            cpu_count = psutil.cpu_count(logical=False) or 1
            
            return cls(
                frame_rate=min(15, max(5, int(mem_available / 1024))),
                process_interval=max(1, 4 - cpu_count),
                resolution=(640, 480),
                batch_size=1 if mem_available < 2048 else 2,
                timeout=0.1,
                max_retries=3
            )
        except Exception as e:
            logger.warning(f"Using fallback configuration due to: {e}")
            return cls()

class VideoProcessor(VideoProcessorBase):
    """Enhanced video processor with better error handling and recovery"""
    def __init__(self, confidence: float, model: YOLO):
        super().__init__()
        self.config = ProcessingConfig.create()
        self.confidence = confidence
        self.model = model
        self.frame_count = 0
        self.last_result = None
        self.last_process_time = perf_counter()
        self.lock = threading.Lock()
        self.error_count = defaultdict(int)
        self.reconnection_attempts = 0

    def _handle_error(self, error_type: str, error: Exception) -> bool:
        """Handle errors with exponential backoff"""
        self.error_count[error_type] += 1
        wait_time = min(2 ** self.error_count[error_type], 30)
        
        logger.error(f"{error_type} error: {error}. Waiting {wait_time}s before retry.")
        time.sleep(wait_time)
        
        return self.error_count[error_type] < self.config.max_retries

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming frames with improved error recovery"""
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            if (self.frame_count % self.config.process_interval != 0 or
                perf_counter() - self.last_process_time < self.config.timeout):
                return av.VideoFrame.from_ndarray(
                    self.last_result if self.last_result is not None else img,
                    format="bgr24"
                )

            with self.lock:
                try:
                    results = self.model.predict(
                        img,
                        conf=self.confidence,
                        batch=self.config.batch_size
                    )
                    processed = results[0].plot()
                    self.last_result = processed
                    self.last_process_time = perf_counter()
                    self.error_count.clear()
                    
                except Exception as e:
                    if self._handle_error("prediction", e):
                        processed = self.last_result if self.last_result is not None else img
                    else:
                        raise RuntimeError("Maximum prediction retries exceeded")

            return av.VideoFrame.from_ndarray(processed, format="bgr24")

        except Exception as e:
            if self._handle_error("frame_processing", e):
                return frame
            else:
                raise RuntimeError("Maximum frame processing retries exceeded")

def get_ice_servers():
    """Get ICE servers with fallback and retry logic"""
    ice_servers = [
        {"urls": [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
            "stun:stun3.l.google.com:19302",
            "stun:stun4.l.google.com:19302",
        ]},
    ]
    
    turn_servers = os.environ.get("TURN_SERVERS")
    if turn_servers:
        try:
            import json
            additional_servers = json.loads(turn_servers)
            ice_servers.extend(additional_servers)
        except Exception as e:
            logger.error(f"Failed to load TURN servers: {e}")
    
    return ice_servers

def initialize_webrtc(confidence: float, model: YOLO):
    """Initialize WebRTC with connection monitoring"""
    config = ProcessingConfig.create()
    
    try:
        webrtc_ctx = webrtc_streamer(
            key="underwater-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": get_ice_servers()}
            ),
            video_processor_factory=lambda: VideoProcessor(confidence, model),
            async_processing=True,
            media_stream_constraints={
                "video": {
                    "width": {"min": 320, "ideal": config.resolution[0], "max": 1280},
                    "height": {"min": 240, "ideal": config.resolution[1], "max": 720},
                    "frameRate": {"min": 5, "ideal": config.frame_rate, "max": 30}
                },
                "audio": False
            },
            video_html_attrs={
                "style": {"width": "100%", "height": "auto"},
                "controls": False,
                "autoPlay": True,
            }
        )
        
        if webrtc_ctx.state.playing:
            st.success("WebRTC stream started successfully")
            
            if webrtc_ctx.state.playing:
                st.info("Connection established successfully")
            elif webrtc_ctx.state.failed:
                st.error("Connection failed. Please check your network connection and try again.")
            
        return webrtc_ctx
        
    except Exception as e:
        logger.error(f"WebRTC initialization error: {e}")
        st.error("""
            Failed to initialize webcam stream. Please try:
            1. Checking your camera permissions
            2. Using a different browser
            3. Using image/video upload instead
        """)
        return None

def initialize_app():
    """Initialize Streamlit application with enhanced error handling"""
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
        }
        .stProgress > div > div {
            background-color: #2ea043;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title(APP_CONFIG["title"])
    
    # Sidebar configuration
    with st.sidebar:
        model_type = st.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
        confidence = st.slider("Confidence", 0.0, 1.0, 0.5)
        
        # System information
        st.markdown("### System Info")
        current_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        available_mem = psutil.virtual_memory().available / (1024 * 1024)
        
        st.text(f"Memory Usage: {current_mem:.0f}MB")
        st.text(f"Available Memory: {available_mem:.0f}MB")
        st.text(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        if available_mem < 1000:  # Less than 1GB available
            st.warning("Low memory available. Performance may be affected.")

    try:
        with st.spinner("Loading model..."):
            model = load_model(str(YOLO_WEIGHTS[model_type]))
            if model is None:
                st.error("Failed to load model. Please check the model path.")
                st.stop()
            
        source_type = st.sidebar.selectbox("Input Source", SOURCES_LIST)
        return model, confidence, source_type

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        st.error("Failed to initialize application. Please refresh the page.")
        cleanup_memory()
        st.stop()

def main():
    try:
        model, confidence, source_type = initialize_app()
        
        if source_type == "Webcam":
            st.info("Initializing webcam stream... This may take a few moments.")
            webrtc_ctx = initialize_webrtc(confidence, model)
            
            if not webrtc_ctx:
                st.warning("""
                    Webcam initialization failed. Please try:
                    1. Checking browser permissions
                    2. Using a different browser
                    3. Using Image or Video input instead
                """)
                
        elif source_type == "Image":
            infer_uploaded_image(confidence, model)
        elif source_type == "Video":
            infer_uploaded_video(confidence, model)

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An error occurred. Please refresh the page.")
        cleanup_memory()

if __name__ == "__main__":
    main()