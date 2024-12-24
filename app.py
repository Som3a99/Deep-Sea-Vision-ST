# app.py
import streamlit as st
import cv2
import logging
import av
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from config import YOLO_WEIGHTS, SOURCES_LIST
import threading
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO
from time import perf_counter
import psutil
import gc
import torch
from collections import defaultdict

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
    """Dynamic configuration based on system resources"""
    frame_rate: int = 10
    process_interval: int = 2
    resolution: tuple = (640, 480)
    batch_size: int = 1
    timeout: float = 0.1

    @classmethod
    def create(cls):
        try:
            mem_available = psutil.virtual_memory().available / (1024 * 1024)
            return cls(
                frame_rate=min(15, max(5, int(mem_available / 512))),
                process_interval=2,
                resolution=(640, 480),
                batch_size=1,
                timeout=0.1
            )
        except Exception as e:
            logger.warning(f"Using fallback configuration: {e}")
            return cls()

class VideoProcessor(VideoProcessorBase):
    """Handles video processing with YOLO model"""
    def __init__(self, confidence: float, model: YOLO):
        self.config = ProcessingConfig.create()
        self.confidence = confidence
        self.model = model
        self.frame_count = 0
        self.last_result = None
        self.last_process_time = perf_counter()
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Handle incoming video frames with improved error handling"""
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            if self.frame_count % self.config.process_interval != 0:
                return av.VideoFrame.from_ndarray(
                    self.last_result if self.last_result is not None else img,
                    format="bgr24"
                )

            current_time = perf_counter()
            if current_time - self.last_process_time < self.config.timeout:
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
                    self.last_process_time = current_time
                except Exception as e:
                    logger.error(f"Model prediction error: {e}")
                    processed = self.last_result if self.last_result is not None else img

            return av.VideoFrame.from_ndarray(processed, format="bgr24")

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame

def get_ice_servers():
    """Get ICE servers configuration with fallback options"""
    ice_servers = [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
                "stun:stun2.l.google.com:19302",
            ]
        }
    ]
    
    # Add Twilio servers if configured
    if 'TWILIO_ACCOUNT_SID' in os.environ:
        ice_servers.extend([
            {
                "urls": f"turn:global.turn.twilio.com:3478?transport=udp",
                "username": os.environ.get("TWILIO_USERNAME"),
                "credential": os.environ.get("TWILIO_CREDENTIAL")
            },
            {
                "urls": f"turn:global.turn.twilio.com:3478?transport=tcp",
                "username": os.environ.get("TWILIO_USERNAME"),
                "credential": os.environ.get("TWILIO_CREDENTIAL")
            }
        ])
    
    return ice_servers

def initialize_webrtc(confidence: float, model: YOLO):
    """Initialize WebRTC with improved error handling"""
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
                    "width": {"ideal": config.resolution[0]},
                    "height": {"ideal": config.resolution[1]},
                    "frameRate": {"ideal": config.frame_rate}
                },
                "audio": False
            }
        )
        
        if webrtc_ctx.state.playing:
            st.success("WebRTC stream started successfully")
        
        return webrtc_ctx
    except Exception as e:
        logger.error(f"WebRTC initialization error: {e}")
        st.error("Failed to initialize webcam stream. Please try again or use a different input source.")
        return None

def initialize_app():
    """Initialize Streamlit application with proper error handling"""
    st.set_page_config(
        page_title="Underwater Object Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Underwater Object Detection")
    
    model_type = st.sidebar.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)
    
    # System information display
    current_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    st.sidebar.markdown("### System Info")
    st.sidebar.text(f"Memory Usage: {current_mem:.0f}MB")
    st.sidebar.text(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

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
        st.stop()

def main():
    try:
        model, confidence, source_type = initialize_app()
        
        if source_type == "Webcam":
            st.info("Initializing webcam stream... This may take a few moments.")
            webrtc_ctx = initialize_webrtc(confidence, model)
            
            if not webrtc_ctx:
                st.warning("""
                    Webcam initialization failed. This could be due to:
                    1. Browser permissions not granted
                    2. No webcam detected
                    3. Connection issues
                    
                    Consider using Image or Video input if issues persist.
                """)
                
        elif source_type == "Image":
            infer_uploaded_image(confidence, model)
        elif source_type == "Video":
            infer_uploaded_video(confidence, model)

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An error occurred. Please refresh the page.")
        cleanup_memory()

def cleanup_memory():
    """Cleanup memory resources"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
