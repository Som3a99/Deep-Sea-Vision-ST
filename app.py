# app.py
import streamlit as st
import cv2
import logging
import av
import os
import warnings
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
from utils import ModelManager, ImageProcessor, MemoryManager, cleanup_memory, infer_uploaded_image, infer_uploaded_video
from config import YOLO_WEIGHTS, SOURCES_LIST, APP_CONFIG, WEBRTC_CONFIG
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
from torch.serialization import SourceChangeWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=SourceChangeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*'torch.load' with 'weights_only=False'.*")

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
            gpu_available = torch.cuda.is_available()
            
            return cls(
                frame_rate=min(20 if gpu_available else 15, max(5, int(mem_available / 1024))),
                process_interval=max(1, 3 - cpu_count),
                resolution=(1280, 720) if gpu_available else (640, 480),
                batch_size=2 if gpu_available and mem_available > 4096 else 1,
                timeout=0.05 if gpu_available else 0.1,
                max_retries=5 if gpu_available else 3
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
        self.frames_buffer = []
        self.results_cache = {}
        self.performance_metrics = {
            'fps': 0,
            'latency': 0,
            'memory_usage': 0
        }

    def _update_metrics(self, start_time: float):
        """Update performance metrics"""
        current_time = perf_counter()
        self.performance_metrics['latency'] = (current_time - start_time) * 1000
        self.performance_metrics['fps'] = 1 / (current_time - self.last_process_time)
        self.performance_metrics['memory_usage'] = psutil.Process().memory_info().rss / 1024 / 1024

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
                start_time = perf_counter()
                try:
                    results = self.model.predict(
                        img,
                        conf=self.confidence,
                        batch=self.config.batch_size
                    )
                    processed = results[0].plot()
                    self.last_result = processed
                    self.last_process_time = perf_counter()
                    self._update_metrics(start_time)
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

    def get_metrics(self) -> dict:
        """Return current performance metrics"""
        return self.performance_metrics

def initialize_webrtc(confidence: float, model: YOLO):
    """Initialize WebRTC with improved error handling and timeout management"""
    try:
        rtc_configuration = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": ["turn:numb.viagenie.ca"],
                    "username": "webrtc@live.com",
                    "credential": "muazkh"
                }
            ]}
        )

        webrtc_ctx = webrtc_streamer(
            key="underwater-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=lambda: VideoProcessor(confidence, model),
            async_processing=True,
            media_stream_constraints={
                "video": {
                    "width": {"min": 320, "ideal": 640, "max": 1280},
                    "height": {"min": 240, "ideal": 480, "max": 720},
                    "frameRate": {"ideal": 15, "max": 30}
                },
                "audio": False
            },
            video_html_attrs={
                "style": {"width": "100%", "height": "auto"},
                "controls": False,
                "autoPlay": True,
            },
            timeout=20.0,
        )

        if webrtc_ctx.video_processor:
            if webrtc_ctx.state.playing:
                st.success("WebRTC connection established successfully")
                
                # Display metrics if available
                if hasattr(webrtc_ctx.video_processor, 'get_metrics'):
                    metrics = webrtc_ctx.video_processor.get_metrics()
                    st.sidebar.markdown("### Performance Metrics")
                    st.sidebar.text(f"FPS: {metrics['fps']:.1f}")
                    st.sidebar.text(f"Latency: {metrics['latency']:.0f}ms")
                    st.sidebar.text(f"Memory: {metrics['memory_usage']:.0f}MB")
            
            elif webrtc_ctx.state.failed:
                return use_fallback_camera(confidence, model)

        return webrtc_ctx

    except Exception as e:
        logger.error(f"WebRTC initialization error: {e}")
        return use_fallback_camera(confidence, model)

def use_fallback_camera(confidence: float, model: YOLO):
    """Fallback to OpenCV camera capture"""
    st.warning("Switching to fallback camera mode...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera")
            return None

        stframe = st.empty()
        stop_button = st.button("Stop Camera")

        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=confidence)
            processed_frame = results[0].plot()
            
            stframe.image(
                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                caption="Live Feed",
                use_container_width=True
            )

        cap.release()
        return None

    except Exception as e:
        logger.error(f"Fallback camera error: {e}")
        st.error("Camera access failed. Please try using image or video upload.")
        return None

def initialize_app():
    """Initialize Streamlit application with enhanced error handling"""
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title(APP_CONFIG["title"])
    
    # Initialize session state
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Sidebar configuration
    with st.sidebar:
        model_type = st.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
        confidence = st.slider("Confidence", 0.0, 1.0, 0.5)
        
        # System information
        st.markdown("### System Info")
        col1, col2 = st.columns(2)
        
        with col1:
            current_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            available_mem = psutil.virtual_memory().available / (1024 * 1024)
            st.metric("Memory Usage", f"{current_mem:.0f}MB")
            st.metric("Available Memory", f"{available_mem:.0f}MB")
            
        with col2:
            device = 'CUDA' if torch.cuda.is_available() else 'CPU'
            st.metric("Device", device)
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                st.metric("GPU Memory", f"{gpu_mem:.0f}MB")
        
        if available_mem < APP_CONFIG["memory_threshold"]:
            st.warning("Low memory available. Performance may be affected.")
            
        # Advanced settings
        with st.expander("Advanced Settings"):
            st.slider("Batch Size", 1, 4, 1, key="batch_size")
            st.slider("Frame Skip", 1, 5, 2, key="frame_skip")
            st.checkbox("Enable Debug Mode", key="debug_mode")

    try:
        with st.spinner("Loading model..."):
            model = ModelManager.load_model(str(YOLO_WEIGHTS[model_type]))
            if model is None:
                st.error("Failed to load model. Please check the model path.")
                st.stop()
            st.session_state.model_loaded = True
        
        source_type = st.sidebar.selectbox("Input Source", SOURCES_LIST)
        return model, confidence, source_type

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        st.error("Failed to initialize application. Please refresh the page.")
        cleanup_memory()
        st.stop()

def main():
    """Main application function"""
    try:
        start_time = time.time()
        
        model, confidence, source_type = initialize_app()
        
        if st.session_state.get('debug_mode', False):
            st.sidebar.markdown("### Application Metrics")
            st.sidebar.text(f"Uptime: {time.time() - start_time:.1f}s")
            st.sidebar.text(f"Error Count: {st.session_state.error_count}")
        
        if source_type == "Webcam":
            st.info("Initializing webcam stream... This may take a few moments.")
            webrtc_ctx = initialize_webrtc(confidence, model)
            
            if not webrtc_ctx:
                st.warning("""
                    Webcam initialization failed. Please try:
                    1. Using a different browser (Chrome recommended)
                    2. Checking camera permissions
                    3. Using Image or Video upload instead
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