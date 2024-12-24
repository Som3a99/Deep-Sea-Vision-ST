# utils.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
import time
import logging
import torch
import psutil
import gc
from config import APP_CONFIG, PERFORMANCE_CONFIG
from PIL import Image
import io
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryManager:
    """Enhanced memory management system"""
    
    @staticmethod
    def check_memory() -> float:
        """Check memory and perform cleanup if needed"""
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            cleanup_memory()
            logger.warning("High memory usage detected - cleanup performed")
        return memory.available / (1024 * 1024)

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total / (1024 ** 2),
                'available': memory.available / (1024 ** 2),
                'percent': memory.percent,
                'used': memory.used / (1024 ** 2)
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

    @staticmethod
    def monitor_resources() -> Dict[str, Any]:
        """Monitor system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = MemoryManager.get_memory_usage()
            gpu_info = {}
            
            if torch.cuda.is_available():
                gpu_info = {
                    'gpu_utilization': torch.cuda.utilization(),
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024**2),
                    'gpu_memory_cached': torch.cuda.memory_reserved() / (1024**2)
                }
            
            return {
                'cpu_percent': cpu_percent,
                'memory_info': memory_info,
                'gpu_info': gpu_info
            }
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            return {}

class ModelManager:
    """Model management system"""
    
    @staticmethod
    @st.cache_resource
    def load_model(model_path: str) -> Optional[YOLO]:
        """Load YOLO model with improved error handling"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                st.error(f"Model file not found: {model_path}")
                return None

            # Set torch configurations
            torch.backends.cudnn.benchmark = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model
            model = YOLO(str(model_path))
            
            # Move model to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            logger.info(f"Successfully loaded model from {model_path} on {device}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            return None

    @staticmethod
    def get_optimal_batch_size() -> int:
        """Determine optimal batch size based on available resources"""
        if torch.cuda.is_available():
            return PERFORMANCE_CONFIG['batch_size']['gpu']
        return PERFORMANCE_CONFIG['batch_size']['cpu']

class ImageProcessor:
    """Image processing operations"""
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        max_width, max_height = max_size
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        return image

    @staticmethod
    def process_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
        """Process uploaded image with enhanced error handling"""
        try:
            # Verify file size
            if uploaded_file.size > APP_CONFIG["max_upload_size"] * 1024 * 1024:
                raise ValueError(f"File too large. Maximum size is {APP_CONFIG['max_upload_size']}MB")
            
            # Read image using PIL first
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Convert from RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Resize if necessary
            image_bgr = ImageProcessor.resize_image(image_bgr, APP_CONFIG["max_resolution"])
            
            return image_bgr
            
        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}")
            return None

def cleanup_memory():
    """Comprehensive memory cleanup"""
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Collect garbage
        gc.collect()
        
        # Clear Streamlit cache if needed
        if st.session_state.get('clear_cache', False):
            st.cache_resource.clear()
            st.session_state.clear_cache = False
        
        logger.info("Memory cleanup performed successfully")
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")

def infer_uploaded_image(conf: float, model: YOLO):
    """Handle image upload and inference with improved error handling"""
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=APP_CONFIG["supported_image_types"]
    )
    
    if uploaded_file:
        try:
            # Create a temporary directory to save the image
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Read image using OpenCV
                image = cv2.imread(temp_path)
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
                        use_container_width=True
                    )
                
                if st.button("Detect Objects"):
                    with st.spinner("Processing..."):
                        # Model inference
                        results = model.predict(image, conf=conf)
                        
                        # Display results
                        with col2:
                            st.image(
                                cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB),
                                caption="Detected Objects",
                                use_container_width=True
                            )
                        
                        # Display metrics
                        st.success(f"Found {len(results[0].boxes)} objects")

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            st.error("Error processing image")
            cleanup_memory()

def infer_uploaded_video(conf: float, model: YOLO):
    """Handle video upload and inference with enhanced error handling"""
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=APP_CONFIG["supported_video_types"],
        help=f"Supported formats: {', '.join(APP_CONFIG['supported_video_types']).upper()}"
    )
    
    if uploaded_file:
        try:
            # Verify file size
            if uploaded_file.size > APP_CONFIG["max_upload_size"] * 1024 * 1024:
                st.error(f"File too large. Maximum size is {APP_CONFIG['max_upload_size']}MB")
                return

            with st.spinner("Processing video..."):
                # Save uploaded file temporarily
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process video
                cap = cv2.VideoCapture(temp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Check video duration
                duration = total_frames / fps
                if duration > APP_CONFIG["max_video_duration"]:
                    st.error(f"Video too long. Maximum duration is {APP_CONFIG['max_video_duration']} seconds")
                    cleanup_video(temp_path, cap)
                    return

                # Setup progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                
                frame_count = 0
                start_time = time.time()
                batch_frames = []
                batch_size = ModelManager.get_optimal_batch_size()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    
                    # Resize frame if necessary
                    frame = ImageProcessor.resize_image(frame, APP_CONFIG["max_resolution"])
                    batch_frames.append(frame)
                    
                    # Process batch
                    if len(batch_frames) >= batch_size or frame_count == total_frames:
                        results = model.predict(
                            batch_frames,
                            conf=conf,
                            verbose=False
                        )
                        
                        for result in results:
                            processed_frame = result.plot()
                            frame_placeholder.image(
                                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                caption="Processed Frame",
                                use_container_width=True
                            )
                        
                        batch_frames = []
                    
                    # Update progress
                    progress = (frame_count / total_frames) * 100
                    progress_bar.progress(int(progress))
                    
                    # Update status
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    status_text.text(
                        f"Processing frame {frame_count}/{total_frames} "
                        f"({current_fps:.1f} FPS)"
                    )

                # Cleanup
                cleanup_video(temp_path, cap)
                
                # Display final metrics
                st.success("Video processing complete!")
                st.markdown("### Processing Metrics")
                cols = st.columns(3)
                cols[0].metric("Total Frames", total_frames)
                cols[1].metric("Average FPS", f"{frame_count / elapsed_time:.1f}")
                cols[2].metric("Process Time", f"{elapsed_time:.1f}s")

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            st.error("An error occurred while processing the video")
            cleanup_memory()

def cleanup_video(video_path: str, cap: cv2.VideoCapture):
    """Clean up video resources"""
    try:
        cap.release()
        os.remove(video_path)
    except Exception as e:
        logger.error(f"Error cleaning up video resources: {e}")