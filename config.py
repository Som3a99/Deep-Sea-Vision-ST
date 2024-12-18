from pathlib import Path

# Get the absolute path of the current file
ROOT = Path(__file__).resolve().parent

# Source types
SOURCES_LIST = ["Image", "Video", "Webcam", "WebRTC"]

# Model directory and available YOLO models
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

# Add weights paths dynamically for clarity (optional)
YOLO_WEIGHTS = {model: DETECTION_MODEL_DIR / model for model in DETECTION_MODEL_LIST}
