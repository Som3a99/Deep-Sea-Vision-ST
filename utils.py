# utils.py
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import tempfile
import cv2
import numpy as np
import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

@st.cache_resource
def load_model(model_path: str) -> Optional[YOLO]:
    """Load YOLO model with error handling and caching."""
    try:
        model = YOLO(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None

def process_video_frame(frame: np.ndarray, model, conf: float) -> Tuple[np.ndarray, Optional[object]]:
    """Process a single video frame with error handling."""
    try:
        results = model.predict(frame, conf=conf)
        return results[0].plot(), results[0].boxes
    except Exception as e:
        logger.error(f"Error processing video frame: {e}")
        return frame, None

def infer_uploaded_image(conf: float, model):
    """Handle image upload and inference with progress tracking."""
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        try:
            # Show upload status
            with st.spinner("Processing image..."):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Run Detection", help="Click to start object detection"):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)
                    
                    # Run inference
                    start_time = time.time()
                    results = model.predict(image, conf=conf)
                    progress_bar.progress(75)
                    
                    # Show results
                    st.image(results[0].plot(), caption="Detected Objects")
                    
                    # Show performance metrics
                    end_time = time.time()
                    process_time = end_time - start_time
                    progress_bar.progress(100)
                    
                    # Display detection stats
                    st.success(f"""
                        Detection completed in {process_time:.2f} seconds
                        - Detected objects: {len(results[0].boxes)}
                        - Confidence threshold: {conf}
                    """)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            logger.error(f"Image processing error: {e}")

def infer_uploaded_video(conf: float, model):
    """Handle video upload and inference with progress tracking."""
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"],
        help="Supported formats: MP4, AVI, MOV"
    )
    
    if uploaded_file:
        try:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            # Open video capture
            vid_cap = cv2.VideoCapture(tfile.name)
            total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if st.button("Run Detection", help="Click to start object detection"):
                # Initialize progress tracking
                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                stats_placeholder = st.empty()
                current_frame = 0
                
                start_time = time.time()
                
                while vid_cap.isOpened():
                    success, frame = vid_cap.read()
                    if not success:
                        break
                    
                    # Update progress
                    current_frame += 1
                    progress = int((current_frame / total_frames) * 100)
                    progress_bar.progress(progress)
                    
                    # Process frame
                    frame, boxes = process_video_frame(frame, model, conf)
                    frame_placeholder.image(frame, channels="BGR", caption="Detection in Progress")
                    
                    # Update stats
                    fps = current_frame / (time.time() - start_time)
                    stats_placeholder.info(f"""
                        Processing Stats:
                        - Frame: {current_frame}/{total_frames}
                        - FPS: {fps:.2f}
                        - Progress: {progress}%
                    """)
                
                # Cleanup
                vid_cap.release()
                os.unlink(tfile.name)
                
                # Final status
                st.success(f"Video processing completed! Processed {current_frame} frames.")
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Video processing error: {e}")