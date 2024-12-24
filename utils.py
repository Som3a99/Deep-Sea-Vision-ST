# utils.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any
import torch
import psutil
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manage memory usage and cleanup"""
    @staticmethod
    def check_memory():
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            cleanup_memory()
            logger.warning("High memory usage detected - cleanup performed")
        return memory.available / (1024 * 1024)  # Available memory in MB

    @staticmethod
    @contextmanager
    def monitor_memory():
        try:
            initial_mem = MemoryManager.check_memory()
            yield
        finally:
            final_mem = MemoryManager.check_memory()
            if final_mem < initial_mem * 0.8:  # More than 20% memory consumed
                cleanup_memory()

@st.cache_resource
def load_model(model_path: str) -> Optional[YOLO]:
    """Load YOLO model with error handling and caching"""
    try:
        with MemoryManager.monitor_memory():
            model = YOLO(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None

def process_video_frame(
    frame: np.ndarray,
    model: YOLO,
    conf: float
) -> Tuple[np.ndarray, Optional[Any]]:
    """Process a single video frame with error handling"""
    try:
        with MemoryManager.monitor_memory():
            results = model.predict(frame, conf=conf)
            return results[0].plot(), results[0].boxes
    except Exception as e:
        logger.error(f"Error processing video frame: {e}")
        return frame, None

def infer_uploaded_image(conf: float, model: YOLO):
    """Handle image upload and inference with progress tracking"""
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=APP_CONFIG["supported_image_types"],
        help=f"Supported formats: {', '.join(APP_CONFIG['supported_image_types']).upper()}"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.size > APP_CONFIG["max_upload_size"] * 1024 * 1024:
                st.error(f"File too large. Maximum size is {APP_CONFIG['max_upload_size']}MB")
                return

            with st.spinner("Processing image..."):
                col1, col2 = st.columns(2)
                
                image = cv2.imdecode(
                    np.frombuffer(uploaded_file.read(), np.uint8),
                    cv2.IMREAD_COLOR
                )
                
                with col1:
                    st.image(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        caption="Original Image",
                        use_column_width=True
                    )
                
                if st.button("Detect Objects"):
                    with MemoryManager.monitor_memory():
                        progress_bar = st.progress(0)
                        start_time = time.time()
                        
                        results = model.predict(image, conf=conf)
                        progress_bar.progress(50)
                        
                        with col2:
                            st.image(
                                cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB),
                                caption="Detected Objects",
                                use_column_width=True
                            )
                        
                        end_time = time.time()
                        process_time = end_time - start_time
                        progress_bar.progress(100)
                        
                        # Display metrics
                        metrics = {
                            "Process Time": f"{process_time:.2f}s",
                            "Objects Detected": len(results[0].names),
                            "Memory Usage": f"{MemoryManager.check_memory():.2f}MB"
                            }
                        st.write(metrics)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
def infer_uploaded_video(conf: float, model: YOLO):
    
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=APP_CONFIG["supported_video_types"],
        help=f"Supported formats: {', '.join(APP_CONFIG['supported_video_types']).upper()}"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.size > APP_CONFIG["max_upload_size"] * 1024 * 1024:
                st.error(f"File too large. Maximum size is {APP_CONFIG['max_upload_size']}MB")
                return

            with st.spinner("Processing video..."):
                video_path = save_uploaded_file(uploaded_file)
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                progress_bar = st.progress(0)
                frame_num = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    with MemoryManager.monitor_memory():
                        frame, _ = process_video_frame(frame, model, conf)
                        frame_num += 1
                        progress = (frame_num / frame_count) * 100
                        progress_bar.progress(progress)
                        
                    st.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        caption="Processed Frame",
                        use_column_width=True
                    )
                    
                cap.release()
                os.remove(video_path)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            