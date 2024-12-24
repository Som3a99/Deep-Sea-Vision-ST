import streamlit as st
import cv2
import logging
import av
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import numpy as np
import torch
from pathlib import Path
import tempfile
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, confidence: float, model: YOLO):
        self.confidence = confidence
        self.model = model
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Process every other frame for better performance
            self.frame_count += 1
            if self.frame_count % 2 == 0:
                results = self.model.predict(img, conf=self.confidence)
                img = results[0].plot()
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

def get_webrtc_config():
    return RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": "turn:global.turn.twilio.com: 3478?transport=udp",
                "username": os.getenv("TURN_USERNAME", ""),
                "credential": os.getenv("TURN_PASSWORD", ""),
            }
        ]
    })

def initialize_webrtc(confidence: float, model: YOLO):
    webrtc_ctx = webrtc_streamer(
        key="underwater-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=get_webrtc_config(),
        video_processor_factory=lambda: VideoProcessor(confidence, model),
        media_stream_constraints={
            "video": {
                "width": 640,
                "height": 480,
                "frameRate": {"ideal": 15, "max": 30}
            },
            "audio": False,
        },
        async_processing=True,
    )
    return webrtc_ctx

def process_uploaded_image(uploaded_file, model, confidence):
    if uploaded_file is not None:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Read image
            image = cv2.imread(tmp_path)
            if image is None:
                st.error("Failed to load image")
                return

            # Create columns for display
            col1, col2 = st.columns(2)
            
            # Display original image
            with col1:
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Original Image",
                    use_column_width=True
                )
            
            # Process image
            results = model.predict(image, conf=confidence)
            processed_image = results[0].plot()
            
            # Display processed image
            with col2:
                st.image(
                    cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                    caption="Detected Objects",
                    use_column_width=True
                )

            # Cleanup
            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def process_uploaded_video(uploaded_file, model, confidence):
    if uploaded_file is not None:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            cap = cv2.VideoCapture(tmp_path)
            stframe = st.empty()
            
            # Progress bar
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = model.predict(frame, conf=confidence)
                processed_frame = results[0].plot()
                
                # Display frame
                stframe.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    caption="Video Processing",
                    use_column_width=True
                )

                # Update progress
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)

            # Cleanup
            cap.release()
            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

def main():
    # Page config
    st.set_page_config(
        page_title="Underwater Object Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Underwater Object Detection")

    # Sidebar
    with st.sidebar:
        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        model_type = st.selectbox(
            "Select Model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
        )

        # System info
        st.markdown("### System Info")
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        st.info(f"Using device: {device}")

    try:
        # Load model
        model_path = Path(__file__).parent / "weights" / "detection" / model_type
        with st.spinner("Loading model..."):
            model = YOLO(str(model_path))
            model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Input source selection
        source_type = st.radio("Select Input Source", ["Webcam", "Image", "Video"])

        if source_type == "Webcam":
            st.info("Initializing webcam stream...")
            webrtc_ctx = initialize_webrtc(confidence, model)
            
            if webrtc_ctx.state.playing:
                st.success("Stream started successfully")
            elif webrtc_ctx.state.failed:
                st.error("Failed to start stream")
                
        elif source_type == "Image":
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['jpg', 'jpeg', 'png']
            )
            process_uploaded_image(uploaded_file, model, confidence)
            
        else:  # Video
            uploaded_file = st.file_uploader(
                "Upload Video",
                type=['mp4', 'avi', 'mov']
            )
            process_uploaded_video(uploaded_file, model, confidence)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()