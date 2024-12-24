import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
import torch
import psutil
from contextlib import contextmanager
import gc
from config import APP_CONFIG, PERFORMANCE_CONFIG
from PIL import Image
import io

logger = logging.getLogger(__name__)

class MemoryManager:
    """Enhanced memory management system"""
    
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
    def check_memory() -> float:
        """Check memory and perform cleanup if needed"""
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            cleanup_memory()
            logger.warning("High memory usage detected - cleanup performed")
        return memory.available / (1024 * 1024)

    @staticmethod
    @contextmanager
    def monitor_memory():
        """Context manager for memory monitoring"""
        try:
            initial_mem = MemoryManager.check_memory()
            yield
        finally:
            final_mem = MemoryManager.check_memory()
            if final_mem < initial_mem * 0.8:
                cleanup_memory()
                logger.info(f"Memory cleaned up: {initial_mem - final_mem:.2f}MB released")

class ModelManager:
    """Manage model loading and inference"""
    
    @staticmethod
    @st.cache_resource
    def load_model(model_path: str) -> Optional[YOLO]:
        """Load YOLO model with caching and error handling"""
        try:
            with MemoryManager.monitor_memory():
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = YOLO(model_path)
                model.to(device)
                
                # Optimize model for inference
                if device == 'cuda':
                    model.model.half()  # FP16 for faster inference
                    torch.cuda.empty_cache()
                
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
    """Handle image processing operations"""
    
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
        """Process uploaded image with error handling"""
        try:
            # Read image using PIL first
            image = Image.open(uploaded_file)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to numpy array
            image_np = np.array(image)
            # Convert from RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            return image_bgr
        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}")
            return None

def infer_uploaded_image(conf: float, model: YOLO):
    """Enhanced image upload and inference with progress tracking"""
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=APP_CONFIG["supported_image_types"],
        help=f"Supported formats: {', '.join(APP_CONFIG['supported_image_types']).upper()}"
    )
    
    if uploaded_file:
        try:
            # Validate file size
            if uploaded_file.size > APP_CONFIG["max_upload_size"] * 1024 * 1024:
                st.error(f"File too large. Maximum size is {APP_CONFIG['max_upload_size']}MB")
                return

            with st.spinner("Processing image..."):
                # Create columns for display
                col1, col2 = st.columns(2)
                
                # Process and display original image
                image = ImageProcessor.process_uploaded_image(uploaded_file)
                if image is None:
                    st.error("Failed to process uploaded image")
                    return
                
                # Resize if necessary
                image = ImageProcessor.resize_image(image, APP_CONFIG["max_resolution"])
                
                with col1:
                    st.image(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        caption="Original Image",
                        use_column_width=True
                    )
                
                if st.button("Detect Objects"):
                    with MemoryManager.monitor_memory():
                        # Create progress bar
                        progress_bar = st.progress(0)
                        start_time = time.time()
                        
                        # Model inference
                        results = model.predict(
                            image, 
                            conf=conf,
                            verbose=False
                        )
                        progress_bar.progress(50)
                        
                        # Process results
                        processed_image = results[0].plot()
                        
                        with col2:
                            st.image(
                                cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                                caption="Detected Objects",
                                use_column_width=True
                            )
                        
                        # Calculate and display metrics
                        end_time = time.time()
                        process_time = end_time - start_time
                        progress_bar.progress(100)
                        
                        # Display detailed metrics
                        metrics = {
                            "Process Time": f"{process_time:.2f}s",
                            "Objects Detected": len(results[0].boxes),
                            "Confidence Threshold": f"{conf:.2f}",
                            "Memory Usage": f"{MemoryManager.get_memory_usage()['used']:.0f}MB"
                        }
                        
                        # Display metrics in an organized way
                        st.markdown("### Detection Metrics")
                        cols = st.columns(len(metrics))
                        for col, (metric, value) in zip(cols, metrics.items()):
                            col.metric(metric, value)
                        
                        # Display detailed object information
                        if len(results[0].boxes) > 0:
                            st.markdown("### Detected Objects")
                            for box in results[0].boxes:
                                conf = box.conf.item()
                                cls = results[0].names[box.cls.item()]
                                st.text(f"{cls}: {conf:.2%} confidence")

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            st.error("An error occurred while processing the image")
            cleanup_memory()

def infer_uploaded_video(conf: float, model: YOLO):
    """Enhanced video processing with progress tracking and error handling"""
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=APP_CONFIG["supported_video_types"],
        help=f"Supported formats: {', '.join(APP_CONFIG['supported_video_types']).upper()}"
    )
    
    if uploaded_file:
        try:
            # Validate file size
            if uploaded_file.size > APP_CONFIG["max_upload_size"] * 1024 * 1024:
                st.error(f"File too large. Maximum size is {APP_CONFIG['max_upload_size']}MB")
                return

            with st.spinner("Processing video..."):
                # Save uploaded file temporarily
                video_path = save_uploaded_file(uploaded_file)
                cap = cv2.VideoCapture(video_path)
                
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                duration = total_frames / fps
                
                # Check video duration
                if duration > APP_CONFIG["max_video_duration"]:
                    st.error(f"Video too long. Maximum duration is {APP_CONFIG['max_video_duration']} seconds")
                    cleanup_video(video_path, cap)
                    return

                # Create progress bar and status
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
                        with MemoryManager.monitor_memory():
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
                                    use_column_width=True
                                )
                            
                            batch_frames = []
                    
                    # Update progress
                    progress = (frame_count / total_frames) * 100
                    progress_bar.progress(int(progress))
                    
                    # Update status
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    status_text.text(
                        f"Processing frame {frame_count}/{total_frames} "
                        f"({fps:.1f} FPS)"
                    )

                # Cleanup
                cleanup_video(video_path, cap)
                
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

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file with proper error handling"""
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
            
        return file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise

def cleanup_video(video_path: str, cap: cv2.VideoCapture):
    """Clean up video resources"""
    try:
        cap.release()
        os.remove(video_path)
    except Exception as e:
        logger.error(f"Error cleaning up video resources: {e}")

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