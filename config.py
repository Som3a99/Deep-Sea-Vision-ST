from pathlib import Path

ROOT = Path(__file__).resolve().parent

SOURCES_LIST = ["Image", "Video", "Webcam"]

DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
DETECTION_MODEL_LIST = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"
]

YOLO_WEIGHTS = {model: DETECTION_MODEL_DIR / model for model in DETECTION_MODEL_LIST}
