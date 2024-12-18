from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

@st.cache_resource
def load_model(model_path):
    """
    Load a YOLO object detection model from the specified path.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def infer_uploaded_image(conf, model):
    """
    Perform inference on uploaded images.
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )
    col1, col2 = st.columns(2)

    if source_img:
        uploaded_image = Image.open(source_img)
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Execute"):
            with st.spinner("Processing image..."):
                res = model.predict(uploaded_image, conf=conf)
                res_plotted = res[0].plot()
                boxes = res[0].boxes

                with col2:
                    st.image(res_plotted, caption="Detected Image", use_column_width=True)

                with st.expander("Detection Results"):
                    for box in boxes:
                        st.json({
                            "x": float(box.xywh[0][0]),
                            "y": float(box.xywh[0][1]),
                            "width": float(box.xywh[0][2]),
                            "height": float(box.xywh[0][3]),
                            "confidence": float(box.conf[0])
                        })

def infer_uploaded_video(conf, model):
    """
    Perform inference on uploaded videos.
    """
    source_video = st.sidebar.file_uploader(label="Choose a video...", type=("mp4", "avi", "mkv", "mov"))
    if source_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_video.read())
        vid_cap = cv2.VideoCapture(tfile.name)

        st_frame = st.empty()
        if st.button("Execute"):
            with st.spinner("Processing video..."):
                while vid_cap.isOpened():
                    success, frame = vid_cap.read()
                    if not success:
                        break
                    res = model.predict(frame, conf=conf)
                    res_plotted = res[0].plot()
                    st_frame.image(res_plotted, channels="BGR", use_column_width=True)
                vid_cap.release()

def infer_uploaded_webcam(conf, model):
    """
    Perform inference on webcam input using OpenCV.
    """
    vid_cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    if st.button("Stop Webcam"):
        vid_cap.release()
    else:
        while vid_cap.isOpened():
            success, frame = vid_cap.read()
            if not success:
                break
            res = model.predict(frame, conf=conf)
            res_plotted = res[0].plot()
            st_frame.image(res_plotted, channels="BGR", use_column_width=True)
        vid_cap.release()

class YOLOVideoTransformer(VideoTransformerBase):
    """
    Streamlit WebRTC Video Transformer for real-time YOLO inference.
    """
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        res = self.model.predict(img, conf=self.conf)
        res_plotted = res[0].plot()
        return res_plotted

def infer_uploaded_webrtc(conf, model):
    """
    Perform inference using WebRTC for real-time video.
    """
    webrtc_streamer(
        key="live-detection",
        video_transformer_factory=lambda: YOLOVideoTransformer(conf, model),
        media_stream_constraints={"video": True, "audio": False},
    )
