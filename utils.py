# utils.py
import streamlit as st
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import io

logger = logging.getLogger(__name__)

class ModelManager:
    """Simplified model management system"""
    
    @staticmethod
    @st.cache_resource
    def load_model(model_path: Union[str, Path]) -> Optional[YOLO]:
        """Load and cache YOLO model"""
        try:
            # Verify model path
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Configure device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model
            model = YOLO(str(model_path))
            model.to(device)
            
            # Optional: Warm up the model
            if torch.cuda.is_available():
                dummy_input = torch.zeros((1, 3, 640, 640)).to(device)
                model(dummy_input)
                torch.cuda.empty_cache()
            
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

class ImageProcessor:
    """Simplified image processing operations"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width/width, target_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(image, (new_width, new_height))

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """Prepare image for model inference"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image_norm = image_rgb.astype(np.float32) / 255.0
        
        return image_norm

    @staticmethod
    def process_uploaded_file(uploaded_file) -> Optional[np.ndarray]:
        """Process uploaded image file"""
        try:
            # Read image using PIL
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            return image_bgr
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return None

def create_temp_file(uploaded_file, suffix: str) -> Optional[str]:
    """Create temporary file from uploaded file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error creating temporary file: {e}")
        return None

def cleanup_temp_file(file_path: str):
    """Safely remove temporary file"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up temporary file: {e}")

def get_device_info() -> dict:
    """Get system device information"""
    device_info = {
        "device": "cpu",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "cuda_memory": None
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "device": "cuda",
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_memory": torch.cuda.get_device_properties(0).total_memory
        })
    
    return device_info