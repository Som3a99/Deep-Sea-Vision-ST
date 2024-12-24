# config.py
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

SOURCES_LIST = ["Image", "Video", "Webcam"]

DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
DETECTION_MODEL_LIST = [
    "yolov8n.pt",  # Nano model (fastest)
    "yolov8s.pt",  # Small model (balanced)
    "yolov8m.pt",  # Medium model (most accurate)
]

YOLO_WEIGHTS = {model: DETECTION_MODEL_DIR / model for model in DETECTION_MODEL_LIST}

APP_CONFIG = {
    "title": "Underwater Object Detection",
    "description": "Detect objects in underwater imagery using YOLOv8",
    "version": "1.0.0",
    "memory_threshold": 1000,  # MB
    "max_upload_size": 200,    # MB
    "supported_image_types": ["jpg", "jpeg", "png", "bmp"],
    "supported_video_types": ["mp4", "avi", "mov", "mkv"],
    "max_video_duration": 300,  # seconds
    "max_resolution": (1920, 1080),
    "default_confidence": 0.5,
    "default_model": "yolov8s.pt",
}

# config.py
WEBRTC_CONFIG = {
    "RTCConfiguration": {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                    "stun:stun2.l.google.com:19302",
                ],
            },
            {
                "urls": [
                    "turn:openrelay.metered.ca:80",
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": [
                    "turn:openrelay.metered.ca:443",
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ],
        "iceTransportPolicy": "all",
                    },
    "MEDIA_STREAM_CONSTRAINTS": {
        "video": {
            "width": {"min": 320, "ideal": 640, "max": 1920},
            "height": {"min": 240, "ideal": 480, "max": 1080},
            "frameRate": {"ideal": 30, "max": 60},
        },
        "audio": False
    },
    "VIDEO_HTML_ATTRS": {
        "style": {"width": "100%", "height": "auto"},
        "controls": False,
        "autoPlay": True,
    },
    "TIMEOUT": 30,
    "MAX_RETRIES": 3
}

PERFORMANCE_CONFIG = {
    "batch_size": {
        "cpu": 1,
        "gpu": 4,
    },
    "frame_skip": {
        "cpu": 3,
        "gpu": 1,
    },
    "resolution": {
        "cpu": (640, 480),
        "gpu": (1280, 720),
    },
    "max_memory_usage": 0.8,  # 80% of available memory
    "gpu_memory_fraction": 0.7,  # 70% of GPU memory
}

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