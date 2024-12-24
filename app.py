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
from typing import Optional, Dict
from dataclasses import dataclass
from ultralytics import YOLO
from time import perf_counter
import psutil
import gc
import torch
# Memory management utilities
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def cleanup_memory():
    """Force garbage collection and release unused memory"""
    gc.collect()
    if hasattr(torch, 'cuda'):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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

# Enhanced logging with rotation
from logging.handlers import RotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            'app.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for processing with adaptive parameters"""
    def __init__(self):
        self.update_config()
    
    def update_config(self):
        """Update configuration based on system resources"""
        try:
            cpu_count = psutil.cpu_count(logical=False) or 1
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            
            # Adjust parameters based on available memory
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
            self.TIMEOUT = 1.0 / self.FRAME_RATE
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            # Fallback to conservative defaults
            self.FRAME_RATE = 5
            self.PROCESS_EVERY_N_FRAMES = 3
            self.MAX_RESOLUTION = (320, 240)
            self.THREAD_POOL_SIZE = 1
            self.MAX_QUEUE_SIZE = 3
            self.BATCH_SIZE = 1
            self.MAX_RETRIES = 3
            self.TIMEOUT = 0.2

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def get_yolo_model(model_path: str) -> Optional[YOLO]:
    """Load and cache YOLO model with memory management"""
    try:
        cleanup_memory()  # Clean up before loading model
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        model.fuse()  # Fuse layers for better inference speed
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

class WebRTCStats:
    """Track WebRTC performance metrics"""
    def __init__(self):
        self.fps = 0
        self.latency = 0
        self.dropped_frames = 0
        self.start_time = perf_counter()
        self.frame_times = queue.Queue(maxsize=30)  # Rolling average of last 30 frames
    
    def update(self, frame_time: float):
        """Update performance metrics"""
        if self.frame_times.full():
            self.frame_times.get()
        self.frame_times.put(frame_time)
        
        # Calculate FPS using rolling average
        avg_frame_time = sum(list(self.frame_times.queue)) / self.frame_times.qsize()
        self.fps = 1 / avg_frame_time if avg_frame_time > 0 else 0

class YOLOProcessor:
    def __init__(self, confidence: float, model):
        self.config = ProcessingConfig()
        self._confidence = confidence
        self._model = model
        self._stats = WebRTCStats()
        self._frame_queue = queue.Queue(maxsize=self.config.MAX_QUEUE_SIZE)
        self._result_queue = queue.Queue(maxsize=self.config.MAX_QUEUE_SIZE)
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.THREAD_POOL_SIZE)
        self._lock = threading.Lock()
        self._last_result = None
        self._frame_count = 0
        self._error_count = 0
        self._last_process_time = perf_counter()
        
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with adaptive processing"""
        try:
            current_time = perf_counter()
            if current_time - self._last_process_time < self.config.TIMEOUT:
                return self._last_result if self._last_result is not None else frame

            # Adaptive resolution scaling
            scale_factor = 1.0
            if self._stats.fps < 5:  # If FPS is too low, reduce resolution
                scale_factor = 0.75
            elif self._stats.fps > 15:  # If FPS is high, we can increase resolution
                scale_factor = 1.25

            target_width = int(self.config.MAX_RESOLUTION[0] * scale_factor)
            target_height = int(self.config.MAX_RESOLUTION[1] * scale_factor)
            
            if frame.shape[1] > target_width or frame.shape[0] > target_height:
                frame = cv2.resize(frame, (target_width, target_height), 
                                 interpolation=cv2.INTER_AREA)

            results = self._model.predict(
                frame,
                conf=self._confidence,
                batch=self.config.BATCH_SIZE,
            )
            
            processed_frame = results[0].plot()
            self._last_process_time = current_time
            
            # Update performance stats
            process_time = perf_counter() - current_time
            self._stats.update(process_time)
            
            return processed_frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self._error_count += 1
            return self._last_result if self._last_result is not None else frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Receive and process frames with adaptive frame skipping"""
        if self._error_count >= self.config.MAX_RETRIES:
            logger.error("Too many errors, stopping processing")
            return frame

        try:
            self._frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Adaptive frame skipping based on performance
            skip_frames = max(1, int(30 / self._stats.fps)) if self._stats.fps > 0 else 1
            
            if self._frame_count % skip_frames != 0:
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

    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        return {
            "FPS": round(self._stats.fps, 2),
            "Dropped Frames": self._stats.dropped_frames,
            "Resolution": self.config.MAX_RESOLUTION,
            "Frame Skip": self.config.PROCESS_EVERY_N_FRAMES
        }

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)

def main():
    # Initialize memory management
    manage_memory()
    
    st.set_page_config(
        page_title="Underwater Object Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Cache cleanup
    if not hasattr(st.session_state, 'processor'):
        st.session_state.processor = None

    with st.sidebar:
        model_type = st.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
        confidence = st.slider("Confidence", 0.0, 1.0, 0.5)
        
        # Show memory usage
        current_mem = get_memory_usage()
        st.sidebar.markdown(f"Memory Usage: {current_mem:.0f}MB")
        
        model = get_yolo_model(str(YOLO_WEIGHTS[model_type]))
        if not model:
            st.error("Model loading failed")
            return

        source_type = st.selectbox("Input Source", SOURCES_LIST)

    # Main content area with error handling and memory management
    try:
        if source_type == "Webcam":
            if st.session_state.processor:
                st.session_state.processor.__del__()
            
            # Clean up before starting new processor
            cleanup_memory()
            
            st.session_state.processor = webrtc_streamer(
                key="underwater-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    iceServers=[{"urls": ["stun:stun.l.google.com:19302"]}]
                ),
                video_processor_factory=lambda: YOLOProcessor(confidence, model),
                async_processing=True,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": ProcessingConfig().MAX_RESOLUTION[0]},
                        "height": {"ideal": ProcessingConfig().MAX_RESOLUTION[1]},
                        "frameRate": {"ideal": ProcessingConfig().FRAME_RATE}
                    },
                }
            )
            
            # Display performance metrics
            if st.session_state.processor and hasattr(st.session_state.processor, 'get_stats'):
                stats = st.session_state.processor.get_stats()
                st.sidebar.markdown("### Performance Stats")
                for key, value in stats.items():
                    st.sidebar.text(f"{key}: {value}")
                
        elif source_type == "Image":
            infer_uploaded_image(confidence, model)
        elif source_type == "Video":
            infer_uploaded_video(confidence, model)

        # Periodic memory management
        manage_memory()

    except Exception as e:
        logger.error(f"Main application error: {e}")
        st.error("An error occurred. Please try refreshing the page or selecting a different input source.")
        cleanup_memory()  # Clean up after error

if __name__ == "__main__":
    main()