# config.py
from pathlib import Path

# Base paths
ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / 'weights' / 'detection'

# Input sources
SOURCES_LIST = ["Webcam", "Image", "Video"]

# Model configurations
YOLO_WEIGHTS = {
    "yolov8n.pt": WEIGHTS_DIR / "yolov8n.pt",  # Nano model
    "yolov8s.pt": WEIGHTS_DIR / "yolov8s.pt",  # Small model
    "yolov8m.pt": WEIGHTS_DIR / "yolov8m.pt",  # Medium model
}

# Application settings
APP_CONFIG = {
    "title": "Underwater Object Detection",
    "description": "Real-time underwater object detection using YOLOv8",
    "version": "1.0.0",
    
    # Resource limits
    "memory_threshold": 500,  # MB
    "max_upload_size": 100,   # MB
    "max_video_duration": 180,  # seconds
    
    # Input constraints
    "supported_image_types": ["jpg", "jpeg", "png"],
    "supported_video_types": ["mp4", "mov"],
    "max_resolution": (1280, 720),
    
    # Model defaults
    "default_confidence": 0.5,
    "default_model": "yolov8s.pt",
}

# WebRTC Configuration
WEBRTC_CONFIG = {
    "video_config": {
        "width": {"min": 320, "ideal": 640, "max": 1280},
        "height": {"min": 240, "ideal": 480, "max": 720},
        "frameRate": {"ideal": 15, "max": 30},
    },
    "rtc_config": {
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ],
    },
    "media_stream_constraints": {
        "video": True,
        "audio": False,
    },
}

# Performance settings
PERFORMANCE_CONFIG = {
    "batch_size": {
        "cpu": 1,
        "gpu": 2,
    },
    "frame_skip": {
        "cpu": 2,
        "gpu": 1,
    },
    "resolution": {
        "cpu": (480, 360),
        "gpu": (640, 480),
    },
    "processing_fps": {
        "cpu": 15,
        "gpu": 30,
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "app.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        },
    }
}