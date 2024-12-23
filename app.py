# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import logging
import av
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from typing import Optional
import time
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance configurations
FRAME_RATE = 10  # Reduced from 15 to improve stability
PROCESS_EVERY_N_FRAMES = 3  # Increased from 2 to reduce processing load

class YOLOProcessor(VideoProcessorBase):
    def __init__(self, confidence: float, model):
        self._confidence = confidence
        self._model = model
        self._last_frame = None
        self._frame_count = 0
        self._last_process_time = time.time()
        self._error_count = 0
        self._max_errors = 3
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            self._frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            # Process every Nth frame to improve performance
            if self._frame_count % PROCESS_EVERY_N_FRAMES != 0:
                return av.VideoFrame.from_ndarray(
                    self._last_frame if self._last_frame is not None else img, 
                    format="bgr24"
                )
            
            current_time = time.time()
            # Skip processing if less than 1/FRAME_RATE seconds have passed
            if current_time - self._last_process_time < 1.0/FRAME_RATE:
                return av.VideoFrame.from_ndarray(
                    self._last_frame if self._last_frame is not None else img, 
                    format="bgr24"
                )
                
            # Resize frame to improve performance
            img = cv2.resize(img, (640, 480))
            
            results = self._model.predict(img, conf=self._confidence)
            annotated_frame = results[0].plot()
            self._last_frame = annotated_frame
            self._last_process_time = current_time
            self._error_count = 0  # Reset error count on successful processing
            
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            self._error_count += 1
            
            # If too many errors occur, return the original frame
            if self._error_count > self._max_errors:
                logger.warning("Too many errors, returning original frame")
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Return last successful frame if available, otherwise return original frame
            return av.VideoFrame.from_ndarray(
                self._last_frame if self._last_frame is not None else img,
                format="bgr24"
            )

def get_webrtc_config():
    """Get WebRTC configuration based on environment"""
    # Basic STUN servers
    rtc_config = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]}
        ]
    }
    
    return rtc_config

def setup_webcam_interface(confidence: float, model):
    """Setup WebRTC interface with proper error handling"""
    st.header("Live Detection")
    
    # Add warning about browser compatibility
    st.warning("""
        Note: For best results:
        1. Use Chrome or Firefox browser
        2. Allow camera access when prompted
        3. If the stream doesn't start, try refreshing the page
        4. Processing every 3rd frame to improve performance
    """)
    
    try:
        webrtc_ctx = webrtc_streamer(
            key="underwater-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: YOLOProcessor(confidence, model),
            media_stream_constraints={
                "video": {
                    "frameRate": {"ideal": FRAME_RATE, "max": 15},
                    "width": {"ideal": 640},
                    "height": {"ideal": 480}
                },
                "audio": False
            },
            rtc_configuration=get_webrtc_config(),
            async_processing=True,
            video_html_attrs={
                "style": {"width": "100%", "margin": "0 auto", "border": "1px solid black"},
                "controls": False,
                "autoPlay": True,
            },
        )
        
        if webrtc_ctx.state.playing:
            st.success("Stream started successfully!")
        
        # Add status indicators
        col1, col2 = st.columns(2)
        with col1:
            st.write("WebRTC State:", webrtc_ctx.state)
        with col2:
            if webrtc_ctx.video_transformer:
                st.write("Processing Status: Active")
            else:
                st.write("Processing Status: Inactive")
                
    except Exception as e:
        st.error(f"""
            Error initializing webcam stream: {str(e)}
            
            Troubleshooting steps:
            1. Check if your camera is properly connected
            2. Make sure no other application is using your camera
            3. Try using a different browser (Chrome or Firefox recommended)
            4. Clear your browser cache and refresh the page
        """)
        logger.error(f"WebRTC initialization error: {e}", exc_info=True)


def main():
    initialize_session_state()
    
    st.set_page_config(
        page_title="Underwater Object Detection",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Underwater Object Detection using YOLOv8")
    st.markdown("""
        This application performs real-time object detection on underwater imagery 
        using YOLOv8. Select your input source and model parameters from the sidebar.
    """)

    with st.sidebar:
        st.header("Model Configuration")
        
        try:
            model_type = st.selectbox(
                "Select Model",
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                help="Smaller models (n, s) are faster but less accurate"
            )
            
            confidence = st.slider(
                "Detection Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher values reduce false positives"
            )

            model = get_yolo_model(f'weights/detection/{model_type}')
            
            if model is None:
                st.error("Failed to load model. Please check model path and weights.")
                st.stop()

        except Exception as e:
            st.error(f"Error in model configuration: {e}")
            st.stop()

        st.header("Input Source")
        source_type = st.selectbox(
            "Select Source",
            ["Image", "Video", "Webcam"],
            help="Choose your input source type"
        )

   try:
        if source_type == "Image":
            infer_uploaded_image(confidence, model)
        elif source_type == "Video":
            infer_uploaded_video(confidence, model)
        elif source_type == "Webcam":
            setup_webcam_interface(confidence, model)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()