#!/bin/bash

# Create necessary directories
mkdir -p weights/detection
mkdir -p .streamlit

# Install system dependencies
if [ "$(uname)" == "Linux" ]; then
    sudo apt-get update
    sudo apt-get install -y $(cat packages.txt)
fi

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download YOLOv8 weights
python -c "
from ultralytics import YOLO
YOLO('yolov8n.pt').download()
YOLO('yolov8s.pt').download()
YOLO('yolov8m.pt').download()
"

# Move weights to correct directory
mv yolov8*.pt weights/detection/

echo "Setup complete!"