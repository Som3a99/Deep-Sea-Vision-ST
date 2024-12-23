# app.py
import streamlit as st
import cv2
import logging
import av
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from config import YOLO_WEIGHTS, SOURCES_LIST
from twilio.rest import Client
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Performance configurations
FRAME_RATE = 15
PROCESS_EVERY_N_FRAMES = 2
MAX_RESOLUTION = (640, 480)
THREAD_POOL_SIZE = 2
MAX_QUEUE_SIZE = 10

def get_ice_servers():
    """Get ICE servers configuration with fallback options"""
    try:
        # Attempt to use Twilio TURN servers if credentials are available
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        
        if account_sid and auth_token:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return RTCConfiguration(
                iceServers=token.ice_servers
            )
    except Exception as e:
        logger.warning(f"Failed to get Twilio ICE servers: {e}")
    
    # Fallback to multiple public STUN/TURN servers
    return RTCConfiguration(
        iceServers=[
            {'url': 'stun:global.stun.twilio.com:3478', 
             'urls': 'stun:global.stun.twilio.com:3478'}, 
            {'url': 'turn:global.turn.twilio.com:3478?transport=udp', 
             'username': '9d4724e20cf48b4a108d2699ca374b2ab6c79760abe466520c7069ac557dbf4f', 
            'urls': 'turn:global.turn.twilio.com:3478?transport=udp', 
            'credential': 'n5EXqt0225m7tEY9x23FqnG2lLLnohyjw2zY/G7ILyo='}, 
            {'url': 'turn:global.turn.twilio.com:3478?transport=tcp', 
             'username': '9d4724e20cf48b4a108d2699ca374b2ab6c79760abe466520c7069ac557dbf4f', 
             'urls': 'turn:global.turn.twilio.com:3478?transport=tcp', 
             'credential': 'n5EXqt0225m7tEY9x23FqnG2lLLnohyjw2zY/G7ILyo='}, 
            {'url': 'turn:global.turn.twilio.com:443?transport=tcp', 
             'username': '9d4724e20cf48b4a108d2699ca374b2ab6c79760abe466520c7069ac557dbf4f', 
             'urls': 'turn:global.turn.twilio.com:443?transport=tcp', 
             'credential': 'n5EXqt0225m7tEY9x23FqnG2lLLnohyjw2zY/G7ILyo='}],
        iceTransportPolicy="all"
    )

class YOLOProcessor:
    def __init__(self, confidence: float, model):
        self._confidence = confidence
        self._model = model
        self._frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._result_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
        self._processing = False
        self._lock = threading.Lock()
        self._last_result = None
        self._frame_count = 0
        self._error_count = 0
        self._max_errors = 3
        
    def _process_frame(self, frame):
        """Process a single frame with the YOLO model."""
        try:
            results = self._model.predict(frame, conf=self._confidence)
            return results[0].plot()
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame
            
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self._processing:
            return av.VideoFrame.from_ndarray(
                self._last_result if self._last_result is not None else frame.to_ndarray(format="bgr24"),
                format="bgr24"
            )

        try:
            self._processing = True
            self._frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Skip frames based on counter
            if self._frame_count % PROCESS_EVERY_N_FRAMES != 0:
                return av.VideoFrame.from_ndarray(
                    self._last_result if self._last_result is not None else img,
                    format="bgr24"
                )

            # Resize for performance
            if img.shape[1] > MAX_RESOLUTION[0] or img.shape[0] > MAX_RESOLUTION[1]:
                img = cv2.resize(img, MAX_RESOLUTION)

            # Process frame in thread pool
            future = self._thread_pool.submit(self._process_frame, img)
            processed = future.result(timeout=1.0)
            
            self._last_result = processed
            return av.VideoFrame.from_ndarray(processed, format="bgr24")

        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            self._error_count += 1
            if self._error_count > self._max_errors:
                logger.warning("Too many errors, returning original frame")
            return av.VideoFrame.from_ndarray(
                self._last_result if self._last_result is not None else img,
                format="bgr24"
            )
        finally:
            self._processing = False

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)

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
    """Setup WebRTC interface with improved error handling and user feedback."""
    st.header("Live Detection")
    
    status_placeholder = st.empty()
    error_placeholder = st.empty()
    stats_placeholder = st.empty()

    try:
        webrtc_ctx = webrtc_streamer(
            key="underwater-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=get_ice_servers(),
            video_processor_factory=lambda: YOLOProcessor(confidence, model),
            async_processing=True,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": MAX_RESOLUTION[0]},
                    "height": {"ideal": MAX_RESOLUTION[1]},
                    "frameRate": {"ideal": FRAME_RATE}
                },
                "audio": False
            }
        )

        # Show connection status
        if webrtc_ctx.state.playing:
            status_placeholder.success("✅ Webcam is active and processing!")
            
            # Display performance stats
            stats = {
                "Resolution": f"{MAX_RESOLUTION[0]}x{MAX_RESOLUTION[1]}",
                "Frame Rate": f"{FRAME_RATE} FPS",
                "Processing": f"Every {PROCESS_EVERY_N_FRAMES} frames",
                "Model Confidence": f"{confidence:.2f}"
            }
            stats_placeholder.info("Performance Settings:\n" + "\n".join([f"- {k}: {v}" for k, v in stats.items()]))
        else:
            status_placeholder.warning("""
                Waiting for webcam connection...
                - Ensure camera permissions are granted
                - Check if camera is being used by another application
                - Try refreshing the page
            """)

    except Exception as e:
        logger.error(f"WebRTC setup error: {e}")
        error_placeholder.error(f"""
            Failed to initialize webcam. Please try:
            1. Refreshing the page
            2. Using a different browser (Chrome/Firefox recommended)
            3. Checking your firewall settings
            4. Ensuring your camera is not in use by another application
            
            Technical details: {str(e)}
        """)

def main():
    st.set_page_config(
        page_title="Underwater Object Detection",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    st.title("Underwater Object Detection using YOLOv8")
    st.markdown("""
        Real-time underwater object detection powered by YOLOv8.
        Select your preferred model and input source below to begin.
    """)

    with st.sidebar:
        st.header("Model Configuration")
        
        # Model selection with info
        model_type = st.selectbox(
            "Select Model",
            list(YOLO_WEIGHTS.keys()),
            help="YOLOv8n is fastest, YOLOv8m is most accurate"
        )
        
        # Confidence slider with visual feedback
        confidence = st.slider(
            "Detection Confidence",
            0.0, 1.0, 0.5, 0.05,
            help="Higher values = fewer but more confident detections"
        )
        
        # Load model with progress indicator
        with st.spinner("Loading model..."):
            model = get_yolo_model(str(YOLO_WEIGHTS[model_type]))
        
        if not model:
            st.error("Failed to load model. Please check model path and try again.")
            st.stop()
        else:
            st.success("Model loaded successfully!")
        
        # Source selection
        source_type = st.selectbox(
            "Select Input Source",
            SOURCES_LIST,
            help="Choose how you want to input images for detection"
        )

    # Process based on source type
    if source_type == "Image":
        infer_uploaded_image(confidence, model)
    elif source_type == "Video":
        infer_uploaded_video(confidence, model)
    elif source_type == "Webcam":
        setup_webcam_interface(confidence, model)

    # Add footer with performance tips
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        ### Performance Tips:
        - Use YOLOv8n for faster processing
        - Lower confidence for more detections
        - Close other browser tabs
        - Use Chrome or Firefox
    """)

if __name__ == "__main__":
    main()