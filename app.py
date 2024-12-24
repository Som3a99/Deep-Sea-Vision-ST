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

class ResourceManager:
    """Manages system resources and configurations"""
    def __init__(self, max_mem_mb: int = 4096):
        self.max_mem_mb = max_mem_mb
        self.error_count = defaultdict(int)
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB"""
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    def cleanup_memory(self):
        """Force garbage collection and clear CUDA cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def check_memory(self):
        """Monitor and manage memory usage"""
        current_mem = self.get_memory_usage()
        if current_mem > self.max_mem_mb:
            self.cleanup_memory()
            return False
        return True
    
    def log_error(self, message: str):
        """Log errors with count limiting"""
        if self.error_count[message] < 3:  # Log only first 3 occurrences
            logger.error(message)
        self.error_count[message] += 1

@dataclass
class ProcessingConfig:
    """Dynamic configuration based on system resources"""
    frame_rate: int
    process_interval: int
    resolution: tuple
    thread_pool_size: int
    queue_size: int
    batch_size: int
    timeout: float

    @classmethod
    def create(cls):
        try:
            cpu_count = psutil.cpu_count(logical=False) or 1
            mem_available = psutil.virtual_memory().available / (1024 * 1024)
            
            return cls(
                frame_rate=min(15, max(5, int(mem_available / 512))),
                process_interval=2,
                resolution=(640, 480),  # More reasonable default resolution
                thread_pool_size=min(cpu_count, 4),
                queue_size=3,
                batch_size=1,
                timeout=0.1
            )
        except Exception as e:
            logger.warning(f"Using fallback configuration: {e}")
            return cls(
                frame_rate=5,
                process_interval=3,
                resolution=(320, 240),
                thread_pool_size=1,
                queue_size=2,
                batch_size=1,
                timeout=0.2
            )

class VideoProcessor:
    """Handles video processing with YOLO model"""
    def __init__(self, confidence: float, model: YOLO):
        self.config = ProcessingConfig.create()
        self.confidence = confidence
        self.model = model
        self.resource_manager = ResourceManager()
        self.frame_count = 0
        self.last_result = None
        self.last_process_time = perf_counter()
        self.lock = threading.Lock()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with error handling and resource management"""
        try:
            if not self.resource_manager.check_memory():
                return self.last_result if self.last_result is not None else frame

            current_time = perf_counter()
            if current_time - self.last_process_time < self.config.timeout:
                return self.last_result if self.last_result is not None else frame

            results = self.model.predict(
                frame,
                conf=self.confidence,
                batch=self.config.batch_size
            )
            
            processed_frame = results[0].plot()
            self.last_process_time = current_time
            self.last_result = processed_frame
            return processed_frame

        except Exception as e:
            self.resource_manager.log_error(f"Frame processing error: {e}")
            return self.last_result if self.last_result is not None else frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Handle incoming video frames"""
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            if self.frame_count % self.config.process_interval != 0:
                return av.VideoFrame.from_ndarray(
                    self.last_result if self.last_result is not None else img,
                    format="bgr24"
                )

            with self.lock:
                processed = self.process_frame(img)

            return av.VideoFrame.from_ndarray(processed, format="bgr24")

        except Exception as e:
            self.resource_manager.log_error(f"Frame receive error: {e}")
            return frame

def initialize_app():
    """Initialize Streamlit application with error handling"""
    st.set_page_config(
        page_title="Underwater Object Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    model_type = st.sidebar.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)
    
    # System information display
    resource_manager = ResourceManager()
    current_mem = resource_manager.get_memory_usage()
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
            config = ProcessingConfig.create()
            # Remove the problematic on_error parameter
            webrtc_ctx = webrtc_streamer(
                key="underwater-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                video_processor_factory=lambda: VideoProcessor(confidence, model),
                async_processing=True,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": config.resolution[0]},
                        "height": {"ideal": config.resolution[1]},
                        "frameRate": {"ideal": config.frame_rate}
                    }
                }
            )
        elif source_type == "Image":
            infer_uploaded_image(confidence, model)
        elif source_type == "Video":
            infer_uploaded_video(confidence, model)

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An error occurred. Please refresh the page.")

if __name__ == "__main__":
    main()