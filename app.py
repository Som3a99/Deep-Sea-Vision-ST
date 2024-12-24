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
from typing import Optional, Dict
from dataclasses import dataclass
from ultralytics import YOLO
from time import perf_counter
import psutil
import gc
import torch
from collections import defaultdict
import tempfile

# Memory management utilities
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def cleanup_memory():
    """Force garbage collection and release unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning(f"CUDA cleanup failed: {e}")

def manage_memory(max_mem_mb: int = 4096):
    """Manage memory usage with fallbacks"""
    try:
        current_mem = get_memory_usage()
        if current_mem > max_mem_mb:
            cleanup_memory()
            current_mem = get_memory_usage()
            if current_mem > max_mem_mb:
                logger.warning(f"High memory usage: {current_mem:.0f}MB")
    except Exception as e:
        logger.error(f"Memory management error: {e}")

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

error_count = defaultdict(int)
def log_error_once(message: str):
    """Log an error message only once"""
    if error_count[message] < 1:
        logger.error(message)
    error_count[message] += 1

@dataclass
class ProcessingConfig:
    """Configuration class for video processing parameters"""
    def __init__(self):
        self.update_config()
    
    def update_config(self):
        """Update configuration based on system resources"""
        try:
            cpu_count = psutil.cpu_count(logical=False) or 1
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            
            self.FRAME_RATE = min(10, max(5, int(available_memory / 1024)))
            self.PROCESS_EVERY_N_FRAMES = max(2, int(10 / self.FRAME_RATE))
            self.MAX_RESOLUTION = (
                min(480, int(available_memory / 10)),
                min(360, int(available_memory / 15))
            )
            self.THREAD_POOL_SIZE = min(cpu_count, 2)
            self.MAX_QUEUE_SIZE = max(3, min(5, int(available_memory / 1024)))
            self.BATCH_SIZE = 1
            self.MAX_RETRIES = 3
            self.TIMEOUT = max(0.1, 1.0 / self.FRAME_RATE)
        except Exception as e:
            log_error_once(f"Error updating config: {e}")
            # Fallback to conservative defaults
            self.FRAME_RATE = 5
            self.PROCESS_EVERY_N_FRAMES = 3
            self.MAX_RESOLUTION = (320, 240)
            self.THREAD_POOL_SIZE = 1
            self.MAX_QUEUE_SIZE = 3
            self.BATCH_SIZE = 1
            self.MAX_RETRIES = 3
            self.TIMEOUT = 0.2

class YOLOProcessor:
    def __init__(self, confidence: float, model):
        self.config = ProcessingConfig()
        self._confidence = confidence
        self._model = model
        self._frame_count = 0
        self._error_count = 0
        self._last_result = None
        self._last_process_time = perf_counter()
        self._lock = threading.Lock()
        
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with error handling"""
        try:
            current_time = perf_counter()
            if current_time - self._last_process_time < self.config.TIMEOUT:
                return self._last_result if self._last_result is not None else frame

            results = self._model.predict(
                frame,
                conf=self._confidence,
                batch=self.config.BATCH_SIZE,
            )
            
            processed_frame = results[0].plot()
            self._last_process_time = current_time
            return processed_frame

        except Exception as e:
            log_error_once(f"Frame processing error: {e}")
            self._error_count += 1
            return self._last_result if self._last_result is not None else frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Receive and process frames"""
        if self._error_count >= self.config.MAX_RETRIES:
            log_error_once("Too many errors, stopping processing")
            return frame

        try:
            self._frame_count += 1
            img = frame.to_ndarray(format="bgr24")

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
            log_error_once(f"Frame receive error: {e}")
            return frame

def initialize_sidebar():
    """Initialize Streamlit sidebar for user settings"""
    model_type = st.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
    confidence = st.slider("Confidence", 0.0, 1.0, 0.5)
    
    current_mem = get_memory_usage()
    st.sidebar.markdown("### System Info")
    st.sidebar.text(f"Memory Usage: {current_mem:.0f}MB")
    st.sidebar.text(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    with st.spinner("Loading model..."):
        model = load_model(str(YOLO_WEIGHTS[model_type]))
    
    if model is None:
        st.error("Failed to load model. Please check the model path and try again.")
        st.stop()

    source_type = st.selectbox("Input Source", SOURCES_LIST)
    return model, confidence, source_type

def main():
    manage_memory()
    st.set_page_config(
        page_title="Underwater Object Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    model, confidence, source_type = initialize_sidebar()

    try:
        if source_type == "Webcam":
            config = ProcessingConfig()
            webrtc_ctx = webrtc_streamer(
                key="underwater-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    iceServers=[{"urls": ["stun:stun.l.google.com:19302"]}]
                ),
                video_processor_factory=lambda: YOLOProcessor(confidence, model),
                async_processing=True,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": config.MAX_RESOLUTION[0]},
                        "height": {"ideal": config.MAX_RESOLUTION[1]},
                        "frameRate": {"ideal": min(config.FRAME_RATE, 30)}
                    },
                },
                on_error=lambda e: st.error(f"Webcam initialization error: {e}")
            )
        elif source_type == "Image":
            infer_uploaded_image(confidence, model)
        elif source_type == "Video":
            infer_uploaded_video(confidence, model)

    except Exception as e:
        log_error_once(f"Main application error: {e}")
        st.error("An error occurred. Please try refreshing the page or selecting a different input source.")
        cleanup_memory()

if __name__ == "__main__":
    main()
