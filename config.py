# config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SOURCES_LIST = ["Image", "Video", "Webcam"]

DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
DETECTION_MODEL_LIST = [
    "yolov8n.pt",  # Nano model (fastest)
    "yolov8s.pt",  # Small model (balanced)
    "yolov8m.pt"   # Medium model (most accurate)
]

YOLO_WEIGHTS = {model: DETECTION_MODEL_DIR / model for model in DETECTION_MODEL_LIST}

APP_CONFIG = {
    "title": "Underwater Object Detection",
    "description": "Detect objects in underwater imagery using YOLOv8",
    "version": "1.0.0",
    "memory_threshold": 1000,  # MB
    "max_upload_size": 200,    # MB
    "supported_image_types": ["jpg", "jpeg", "png"],
    "supported_video_types": ["mp4", "avi", "mov"],
}