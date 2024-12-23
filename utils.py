# utils.py
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import numpy as np
from typing import Optional, Tuple
import time

@st.cache_resource
def load_model(model_path: str) -> Optional[YOLO]:
    """Load a YOLO object detection model from the specified path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_video_frame(frame: np.ndarray, model: YOLO, conf: float) -> Tuple[np.ndarray, list]:
    """Process a single video frame and return the annotated frame and detections."""
    results = model.predict(frame, conf=conf)
    return results[0].plot(), results[0].boxes

def infer_uploaded_image(conf: float, model: YOLO):
    """Perform inference on uploaded images with improved UI."""
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )
    
    if source_img:
        col1, col2 = st.columns(2)
        
        try:
            uploaded_image = Image.open(source_img)
            with col1:
                st.image(
                    uploaded_image, 
                    caption="Uploaded Image",
                    use_container_width=True
                )

            if st.button("Execute"):
                with st.spinner("Processing image..."):
                    start_time = time.time()
                    results = model.predict(uploaded_image, conf=conf)
                    process_time = time.time() - start_time
                    
                    res_plotted = results[0].plot()
                    boxes = results[0].boxes

                    with col2:
                        st.image(
                            res_plotted,
                            caption=f"Detected Image (Process Time: {process_time:.2f}s)",
                            use_container_width=True
                        )

                    with st.expander("Detection Results"):
                        if len(boxes) == 0:
                            st.info("No objects detected in the image.")
                        else:
                            for idx, box in enumerate(boxes, 1):
                                st.json({
                                    "detection": idx,
                                    "position": {
                                        "x": float(box.xywh[0][0]),
                                        "y": float(box.xywh[0][1]),
                                        "width": float(box.xywh[0][2]),
                                        "height": float(box.xywh[0][3]),
                                    },
                                    "confidence": float(box.conf[0]),
                                    "class": model.names[int(box.cls[0])]
                                })
                                
        except Exception as e:
            st.error(f"Error processing image: {e}")

def infer_uploaded_video(conf: float, model: YOLO):
    """Perform inference on uploaded videos with improved performance."""
    source_video = st.sidebar.file_uploader(
        label="Choose a video...",
        type=("mp4", "avi", "mkv", "mov")
    )
    
    if source_video:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(source_video.read())
            
            vid_cap = cv2.VideoCapture(tfile.name)
            total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
            
            st.info(f"Video Info: {total_frames} frames, {fps} FPS")
            progress_bar = st.progress(0)
            frame_placeholder = st.empty()
            
            if st.button("Execute"):
                with st.spinner("Processing video..."):
                    frame_count = 0
                    start_time = time.time()
                    
                    while vid_cap.isOpened():
                        success, frame = vid_cap.read()
                        if not success:
                            break
                            
                        frame_count += 1
                        if frame_count % 3 != 0:  # Process every 3rd frame
                            continue
                            
                        annotated_frame, _ = process_video_frame(frame, model, conf)
                        frame_placeholder.image(
                            annotated_frame,
                            channels="BGR",
                            use_container_width=True,
                            caption=f"Frame {frame_count}/{total_frames}"
                        )
                        progress_bar.progress(frame_count / total_frames)
                        
                    process_time = time.time() - start_time
                    st.success(f"Video processing completed in {process_time:.2f} seconds")
                    
        except Exception as e:
            st.error(f"Error processing video: {e}")
        finally:
            if 'vid_cap' in locals():
                vid_cap.release()