import streamlit as st
import cv2
import logging
import av
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
from utils import load_model, infer_uploaded_image, infer_uploaded_video, cleanup_memory, MemoryManager
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
import json

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
        
        if self.error_count[error_type] >= self.config.max_retries:
            logger.critical(f"Maximum retries exceeded for {error_type}")
            return False
        return True

    def _process_frame(self, img: np.ndarray) -> np.ndarray:
        """Process a single frame with caching"""
        frame_hash = hash(img.tobytes())
        if frame_hash in self.results_cache:
            return self.results_cache[frame_hash]

        start_time = perf_counter()
        try:
            results = self.model.predict(
                img,
                conf=self.confidence,
                batch=self.config.batch_size
            )
            processed = results[0].plot()
            self.results_cache[frame_hash] = processed
            
            # Limit cache size
            if len(self.results_cache) > 100:
                self.results_cache.pop(next(iter(self.results_cache)))
                
            self._update_metrics(start_time)
            return processed

        except Exception as e:
            if not self._handle_error("prediction", e):
                return img
            return self.last_result if self.last_result is not None else img

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming frames with improved error recovery"""
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            # Skip frames based on interval
            if (self.frame_count % self.config.process_interval != 0 or
                perf_counter() - self.last_process_time < self.config.timeout):
                return av.VideoFrame.from_ndarray(
                    self.last_result if self.last_result is not None else img,
                    format="bgr24"
                )

            with self.lock:
                processed = self._process_frame(img)
                self.last_result = processed
                self.last_process_time = perf_counter()
                self.error_count.clear()

            return av.VideoFrame.from_ndarray(processed, format="bgr24")

        except Exception as e:
            if self._handle_error("frame_processing", e):
                return frame
            else:
                raise RuntimeError("Maximum frame processing retries exceeded")

    def get_metrics(self) -> dict:
        """Return current performance metrics"""
        return self.performance_metrics

def get_ice_servers():
    """Get ICE servers with fallback and retry logic"""
    ice_servers = [
        {"urls": WEBRTC_CONFIG["STUN_SERVERS"]},
    ]
    
    # Add TURN servers if configured
    turn_servers = os.environ.get("TURN_SERVERS")
    if turn_servers:
        try:
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
                    "width": {"min": 320, "ideal": config.resolution[0], "max": 1920},
                    "height": {"min": 240, "ideal": config.resolution[1], "max": 1080},
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
            
            # Display connection status
            if webrtc_ctx.state.playing:
                st.info("Connection established successfully")
                
                # Display performance metrics
                if hasattr(webrtc_ctx.video_processor, 'get_metrics'):
                    metrics = webrtc_ctx.video_processor.get_metrics()
                    st.sidebar.markdown("### Performance Metrics")
                    st.sidebar.text(f"FPS: {metrics['fps']:.1f}")
                    st.sidebar.text(f"Latency: {metrics['latency']:.0f}ms")
                    st.sidebar.text(f"Memory: {metrics['memory_usage']:.0f}MB")
                    
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
    try:
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
            .metrics-container {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
            }
            </style>
            """, unsafe_allow_html=True)

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
                batch_size = st.slider("Batch Size", 1, 4, 1)
                frame_skip = st.slider("Frame Skip", 1, 5, 2)
                st.checkbox("Enable Debug Mode", key="debug_mode")

        try:
            with st.spinner("Loading model..."):
                model = load_model(str(YOLO_WEIGHTS[model_type]))
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
            
    except Exception as e:
        logger.error(f"Application setup error: {e}")
        st.error("Critical error during setup. Please contact support.")
        st.stop()

def main():
    """Main application function with enhanced error handling and monitoring"""
    try:
        # Initialize performance monitoring
        start_time = time.time()
        
        model, confidence, source_type = initialize_app()
        
        # Display application metrics
        if st.session_state.get('debug_mode', False):
            st.sidebar.markdown("### Application Metrics")
            st.sidebar.text(f"Uptime: {time.time() - start_time:.1f}s")
            st.sidebar.text(f"Error Count: {st.session_state.error_count}")
        
        if source_type == "Webcam":
            st.info("Initializing webcam stream... This may take a few moments.")
            
            # Try WebRTC first
            webrtc_ctx = initialize_webrtc(confidence, model)
            
            if not webrtc_ctx and st.session_state.error_count < 3:
                st.warning("Attempting fallback to standard webcam...")
                try:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        raise RuntimeError("Failed to open webcam")
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        results = model.predict(frame, conf=confidence)
                        st.image(
                            cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB),
                            caption="Webcam Feed",
                            use_column_width=True
                        )
                        
                        if st.button("Stop"):
                            break
                            
                    cap.release()
                    
                except Exception as e:
                    logger.error(f"Fallback webcam error: {e}")
                    st.error("Webcam capture failed. Please try using image or video upload.")
                    st.session_state.error_count += 1
                    
        elif source_type == "Image":
            infer_uploaded_image(confidence, model)
        elif source_type == "Video":
            infer_uploaded_video(confidence, model)

        # Memory management
        if MemoryManager.check_memory() < APP_CONFIG["memory_threshold"]:
            cleanup_memory()

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An error occurred. Please refresh the page.")
        cleanup_memory()
        
    finally:
        # Cleanup and metrics logging
        if st.session_state.get('debug_mode', False):
            st.sidebar.text(f"Total Runtime: {time.time() - start_time:.1f}s")
        cleanup_memory()

if __name__ == "__main__":
    main()