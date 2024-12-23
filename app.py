# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import cv2
import logging
import av
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from config import YOLO_WEIGHTS, SOURCES_LIST
from typing import Optional, Dict
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Performance configurations
FRAME_RATE = 15  # Increased from 8
PROCESS_EVERY_N_FRAMES = 3  # Reduced from 5
MAX_RESOLUTION = (640, 480)  # Optimal resolution for web streaming
THREAD_POOL_SIZE = 2

class YOLOProcessor(VideoProcessorBase):
    def __init__(self, confidence: float, model):
        self._confidence = confidence
        self._model = model
        self._last_frame = None
        self._frame_count = 0
        self._error_count = 0
        self._max_errors = 3
        self._lock = threading.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
        self._processing = False
        
    def _process_frame(self, img):
        """Process a single frame with the YOLO model."""
        try:
            results = self._model.predict(img, conf=self._confidence)
            return results[0].plot()
        except Exception as e:
            logger.error(f"Error in YOLO processing: {e}")
            return img

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self._processing:
            return av.VideoFrame.from_ndarray(
                self._last_frame if self._last_frame is not None else frame.to_ndarray(format="bgr24"),
                format="bgr24"
            )

        try:
            self._processing = True
            self._frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Skip frames based on counter
            if self._frame_count % PROCESS_EVERY_N_FRAMES != 0:
                return av.VideoFrame.from_ndarray(
                    self._last_frame if self._last_frame is not None else img,
                    format="bgr24"
                )

            # Resize for performance
            height, width = img.shape[:2]
            if width > MAX_RESOLUTION[0] or height > MAX_RESOLUTION[1]:
                img = cv2.resize(img, MAX_RESOLUTION)

            # Process frame in thread pool
            future = self._thread_pool.submit(self._process_frame, img)
            processed_frame = future.result(timeout=1.0)  # 1 second timeout
            
            self._last_frame = processed_frame
            return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            self._error_count += 1
            if self._error_count > self._max_errors:
                logger.warning("Too many errors, returning original frame")
            return av.VideoFrame.from_ndarray(
                self._last_frame if self._last_frame is not None else img,
                format="bgr24"
            )
        finally:
            self._processing = False

def get_webrtc_config() -> Dict:
    """Get WebRTC configuration with multiple STUN servers."""
    return RTCConfiguration(
        iceServers=[
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun.ekiga.net:3478"]},
            {"urls": ["stun:stun.ideasip.com:3478"]},
            {"urls": ["stun:stun.schlund.de:3478"]},
            {"urls": ["stun:stun.voiparound.com:3478"]},
            {"urls": ["stun:stun.voipbuster.com:3478"]},
            {"urls": ["stun:stun.voipstunt.com:3478"]},
            {"urls": ["stun:stun.counterpath.com:3478"]}
        ]
    )

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

    try:
        webrtc_ctx = webrtc_streamer(
            key="detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=get_webrtc_config(),
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

        if webrtc_ctx.state.playing:
            status_placeholder.success("Webcam is active and processing!")
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
        model_type = st.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
        confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)
        
        model = get_yolo_model(str(YOLO_WEIGHTS[model_type]))
        if not model:
            st.error("Failed to load model. Please check model path and try again.")
            st.stop()
        
        source_type = st.selectbox("Select Input Source", SOURCES_LIST)

    if source_type == "Image":
        infer_uploaded_image(confidence, model)
    elif source_type == "Video":
        infer_uploaded_video(confidence, model)
    elif source_type == "Webcam":
        setup_webcam_interface(confidence, model)

if __name__ == "__main__":
    main()