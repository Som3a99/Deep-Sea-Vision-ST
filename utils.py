#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import cv2

# Error handling decorator for OpenCV operations
def safe_cv2_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"OpenCV Error: {str(e)}")
            return None
    return wrapper

@safe_cv2_operation
def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    """
    try:
        # Convert PIL Image to numpy array if necessary
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Ensure image is in correct format
        if image is None:
            st.error("Error: Could not process image")
            return

        # Resize the image to a standard size
        image = cv2.resize(image, (720, int(720 * (9 / 16))))

        # Predict the objects in the image using YOLOv8 model
        res = model.predict(image, conf=conf)

        # Plot the detected objects on the video frame
        res_plotted = res[0].plot()
        
        st_frame.image(res_plotted,
                      caption='Detected Video',
                      channels="BGR",
                      use_column_width=True
                      )
    except Exception as e:
        st.error(f"Error in display_detected_frames: {str(e)}")

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            try:
                uploaded_image = Image.open(source_img)
                st.image(
                    image=source_img,
                    caption="Uploaded Image",
                    use_column_width=True
                )
            except Exception as e:
                st.error(f"Error opening image: {str(e)}")
                return

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    res = model.predict(uploaded_image, conf=conf)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]

                    with col2:
                        st.image(res_plotted,
                                caption="Detected Image",
                                use_column_width=True)
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")

def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    with tempfile.NamedTemporaryFile(suffix='.mp4') as tfile:
                        tfile.write(source_video.read())
                        vid_cap = cv2.VideoCapture(tfile.name)
                        
                        if not vid_cap.isOpened():
                            st.error("Error: Could not open video file")
                            return

                        st_frame = st.empty()
                        while vid_cap.isOpened():
                            success, image = vid_cap.read()
                            if success:
                                _display_detected_frames(conf, model, st_frame, image)
                            else:
                                break
                        vid_cap.release()
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")

def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    """
    # Check if running on Streamlit Cloud
    if st.runtime.exists():  # Check if running on Streamlit
        st.warning("⚠️ Webcam capture is not supported on Streamlit Cloud. Please run this app locally for webcam support.")
        st.info("💡 You can still use the Image and Video upload features!")
        return

    try:
        st.info("ℹ️ Starting webcam... Please allow access if prompted by your browser.")
        vid_cap = cv2.VideoCapture(0)
        
        if not vid_cap.isOpened():
            st.error("❌ Error: Could not access webcam! Please check if:")
            st.error("1. Your webcam is properly connected")
            st.error("2. You've granted browser permission to access the webcam")
            st.error("3. No other application is using the webcam")
            return

        st.success("✅ Webcam successfully accessed!")
        
        # Create a stop button
        stop = st.button("Stop")
        st_frame = st.empty()
        
        while not stop:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_frame, image)
            else:
                st.error("❌ Failed to read from webcam")
                break
            
        vid_cap.release()
        st.success("✅ Webcam released successfully")
        
    except Exception as e:
        st.error(f"❌ Error accessing webcam: {str(e)}")
        st.error("Please make sure you have the necessary permissions and drivers installed")