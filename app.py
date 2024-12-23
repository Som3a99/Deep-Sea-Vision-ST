# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import logging
import av
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from config import YOLO_WEIGHTS, SOURCES_LIST
from typing import Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance configurations
FRAME_RATE = 8
PROCESS_EVERY_N_FRAMES = 5

def initialize_session_state():
    """Initialize session state variables."""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_source' not in st.session_state:
        st.session_state.current_source = None
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

class YOLOProcessor(VideoProcessorBase):
    def __init__(self, confidence: float, model):
        self._confidence = confidence
        self._model = model
        self._last_frame = None
        self._frame_count = 0
        self._error_count = 0
        self._max_errors = 3
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            self._frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Process only every Nth frame to reduce compute load
            if self._frame_count % PROCESS_EVERY_N_FRAMES != 0:
                return av.VideoFrame.from_ndarray(
                    self._last_frame if self._last_frame is not None else img,
                    format="bgr24"
                )
            
            img = cv2.resize(img, (320, 240))  # Optimize for performance
            
            results = self._model.predict(img, conf=self._confidence)
            annotated_frame = results[0].plot()
            self._last_frame = annotated_frame
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            self._error_count += 1
            if self._error_count > self._max_errors:
                logger.warning("Too many errors, returning original frame.")
            return av.VideoFrame.from_ndarray(
                self._last_frame if self._last_frame is not None else img,
                format="bgr24"
            )

@st.cache_resource
def get_yolo_model(model_path: str) -> Optional[object]:
    """Cache the YOLO model loading with error handling."""
    try:
        model = load_model(model_path)
        st.session_state.model_loaded = True
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.session_state.model_loaded = False
        return None

def setup_webcam_interface(confidence: float, model):
    st.header("Live Detection")

    st.warning("""
        If the stream doesn't start:
        - Ensure camera permissions are granted.
        - Check network restrictions.
        - Use a different browser (Chrome/Firefox).
    """)

    rtc_configuration = {
"iceServers": [
    {"urls": "stun:stun.1und1.de:3478"},
    {"urls": "stun:stun.gmx.net:3478"},
    {"urls": "stun:stun.l.google.com:19302"},
    {"urls": "stun:stun1.l.google.com:19302"},
    {"urls": "stun:stun2.l.google.com:19302"},
    {"urls": "stun:stun3.l.google.com:19302"},
    {"urls": "stun:stun4.l.google.com:19302"},
    {"urls": "stun:23.21.150.121:3478"},
    {"urls": "stun:iphone-stun.strato-iphone.de:3478"},
    {"urls": "stun:numb.viagenie.ca:3478"},
    {"urls": "stun:stun.12connect.com:3478"},
    {"urls": "stun:stun.12voip.com:3478"},
    {"urls": "stun:stun.1und1.de:3478"},
    {"urls": "stun:stun.2talk.co.nz:3478"},
    {"urls": "stun:stun.2talk.com:3478"},
    {"urls": "stun:stun.3clogic.com:3478"},
    {"urls": "stun:stun.3cx.com:3478"},
    {"urls": "stun:stun.a-mm.tv:3478"},
    {"urls": "stun:stun.aa.net.uk:3478"},
    {"urls": "stun:stun.acrobits.cz:3478"},
    {"urls": "stun:stun.actionvoip.com:3478"},
    {"urls": "stun:stun.advfn.com:3478"},
    {"urls": "stun:stun.aeta-audio.com:3478"},
    {"urls": "stun:stun.aeta.com:3478"},
    {"urls": "stun:stun.altar.com.pl:3478"},
    {"urls": "stun:stun.annatel.net:3478"},
    {"urls": "stun:stun.antisip.com:3478"},
    {"urls": "stun:stun.arbuz.ru:3478"},
    {"urls": "stun:stun.avigora.fr:3478"},
    {"urls": "stun:stun.awa-shima.com:3478"},
    {"urls": "stun:stun.b2b2c.ca:3478"},
    {"urls": "stun:stun.bahnhof.net:3478"},
    {"urls": "stun:stun.barracuda.com:3478"},
    {"urls": "stun:stun.bluesip.net:3478"},
    {"urls": "stun:stun.bmwgs.cz:3478"},
    {"urls": "stun:stun.botonakis.com:3478"},
    {"urls": "stun:stun.budgetsip.com:3478"},
    {"urls": "stun:stun.cablenet-as.net:3478"},
    {"urls": "stun:stun.callromania.ro:3478"},
    {"urls": "stun:stun.callwithus.com:3478"},
    {"urls": "stun:stun.chathelp.ru:3478"},
    {"urls": "stun:stun.cheapvoip.com:3478"},
    {"urls": "stun:stun.ciktel.com:3478"},
    {"urls": "stun:stun.cloopen.com:3478"},
    {"urls": "stun:stun.comfi.com:3478"},
    {"urls": "stun:stun.commpeak.com:3478"},
    {"urls": "stun:stun.comtube.com:3478"},
    {"urls": "stun:stun.comtube.ru:3478"},
    {"urls": "stun:stun.cope.es:3478"},
    {"urls": "stun:stun.counterpath.com:3478"},
    {"urls": "stun:stun.counterpath.net:3478"},
    {"urls": "stun:stun.datamanagement.it:3478"},
    {"urls": "stun:stun.dcalling.de:3478"},
    {"urls": "stun:stun.demos.ru:3478"},
    {"urls": "stun:stun.develz.org:3478"},
    {"urls": "stun:stun.dingaling.ca:3478"},
    {"urls": "stun:stun.doublerobotics.com:3478"},
    {"urls": "stun:stun.dus.net:3478"},
    {"urls": "stun:stun.easycall.pl:3478"},
    {"urls": "stun:stun.easyvoip.com:3478"},
    {"urls": "stun:stun.ekiga.net:3478"},
    {"urls": "stun:stun.epygi.com:3478"},
    {"urls": "stun:stun.etoilediese.fr:3478"},
    {"urls": "stun:stun.faktortel.com.au:3478"},
    {"urls": "stun:stun.freecall.com:3478"},
    {"urls": "stun:stun.freeswitch.org:3478"},
    {"urls": "stun:stun.freevoipdeal.com:3478"},
    {"urls": "stun:stun.gmx.de:3478"},
    {"urls": "stun:stun.gmx.net:3478"},
    {"urls": "stun:stun.gradwell.com:3478"},
    {"urls": "stun:stun.halonet.pl:3478"},
    {"urls": "stun:stun.hellonanu.com:3478"},
    {"urls": "stun:stun.hoiio.com:3478"},
    {"urls": "stun:stun.hosteurope.de:3478"},
    {"urls": "stun:stun.ideasip.com:3478"},
    {"urls": "stun:stun.infra.net:3478"},
    {"urls": "stun:stun.internetcalls.com:3478"},
    {"urls": "stun:stun.intervoip.com:3478"},
    {"urls": "stun:stun.ipcomms.net:3478"},
    {"urls": "stun:stun.ipfire.org:3478"},
    {"urls": "stun:stun.ippi.fr:3478"},
    {"urls": "stun:stun.ipshka.com:3478"},
    {"urls": "stun:stun.irian.at:3478"},
    {"urls": "stun:stun.it1.hr:3478"},
    {"urls": "stun:stun.ivao.aero:3478"},
    {"urls": "stun:stun.jumblo.com:3478"},
    {"urls": "stun:stun.justvoip.com:3478"},
    {"urls": "stun:stun.kanet.ru:3478"},
    {"urls": "stun:stun.kiwilink.co.nz:3478"},
    {"urls": "stun:stun.l.google.com:19302"},
    {"urls": "stun:stun.linea7.net:3478"},
    {"urls": "stun:stun.linphone.org:3478"},
    {"urls": "stun:stun.liveo.fr:3478"},
    {"urls": "stun:stun.lowratevoip.com:3478"},
    {"urls": "stun:stun.lugosoft.com:3478"},
    {"urls": "stun:stun.lundimatin.fr:3478"},
    {"urls": "stun:stun.magnet.ie:3478"},
    {"urls": "stun:stun.mgn.ru:3478"},
    {"urls": "stun:stun.mit.de:3478"},
    {"urls": "stun:stun.mitake.com.tw:3478"},
    {"urls": "stun:stun.miwifi.com:3478"},
    {"urls": "stun:stun.modulus.gr:3478"},
    {"urls": "stun:stun.myvoiptraffic.com:3478"},
    {"urls": "stun:stun.mywatson.it:3478"},
    {"urls": "stun:stun.nas.net:3478"},
    {"urls": "stun:stun.neotel.co.za:3478"},
    {"urls": "stun:stun.netappel.com:3478"},
    {"urls": "stun:stun.netgsm.com.tr:3478"},
    {"urls": "stun:stun.nfon.net:3478"},
    {"urls": "stun:stun.noblogs.org:3478"},
    {"urls": "stun:stun.noc.ams-ix.net:3478"},
    {"urls": "stun:stun.nonoh.net:3478"},
    {"urls": "stun:stun.nottingham.ac.uk:3478"},
    {"urls": "stun:stun.novopayment.com:3478"},
    {"urls": "stun:stun.okaycall.com:3478"},
    {"urls": "stun:stun.ono.com:3478"},
    {"urls": "stun:stun.optimum.net:3478"},
    {"urls": "stun:stun.orangesip.fr:3478"},
    {"urls": "stun:stun.ovh.net:3478"},
    {"urls": "stun:stun.pccw.com:3478"},
    {"urls": "stun:stun.pjsip.org:3478"},
    {"urls": "stun:stun.planet.com:3478"},
    {"urls": "stun:stun.plivo.com:3478"},
    {"urls": "stun:stun.powwownow.com:3478"},
    {"urls": "stun:stun.proofpoint.com:3478"},
    {"urls": "stun:stun.qc.tel:3478"},
    {"urls": "stun:stun.rapidrecon.com:3478"},
    {"urls": "stun:stun.russianvoip.com:3478"},
    {"urls": "stun:stun.sipcall.ch:3478"},
    {"urls": "stun:stun.sipgate.net:3478"},
    {"urls": "stun:stun.sipnet.ru:3478"},
    {"urls": "stun:stun.sipsimple.org:3478"},
    {"urls": "stun:stun.siptraffic.com:3478"},
    {"urls": "stun:stun.siptrunk.net:3478"},
    {"urls": "stun:stun.sivip.com:3478"},
    {"urls": "stun:stun.solidhost.com:3478"},
    {"urls": "stun:stun.southcoastnetworks.com:3478"},
    {"urls": "stun:stun.spd.ray.com:3478"},
    {"urls": "stun:stun.spectra.raycomtech.com:3478"},
    {"urls": "stun:stun.spro.net:3478"},
    {"urls": "stun:stun.starphone.de:3478"},
    {"urls": "stun:stun.sweetvoip.com:3478"},
    {"urls": "stun:stun.telia.com:3478"},
    {"urls": "stun:stun.terrapin.io:3478"},
    {"urls": "stun:stun.three.com:3478"},
    {"urls": "stun:stun.tmn.pl:3478"},
    {"urls": "stun:stun.tohn.de:3478"},
    {"urls": "stun:stun.trendnet.com:3478"},
    {"urls": "stun:stun.twilio.com:3478"},
    {"urls": "stun:stun.ubiquiti.com:3478"},
    {"urls": "stun:stun.ubiquiti.com:3478"},
    {"urls": "stun:stun.voicera.com:3478"},
    {"urls": "stun:stun.voip.com:3478"},
    {"urls": "stun:stun.voip.ms:3478"},
    {"urls": "stun:stun.voip.in:3478"},
    {"urls": "stun:stun.voiptelecom.de:3478"},
    {"urls": "stun:stun.vonage.net:3478"},
    {"urls": "stun:stun.wl.fi:3478"},
    {"urls": "stun:stun.xs4all.nl:3478"},
    {"urls": "stun:stun.your.ia:3478"}
]
    }

    try:
        webrtc_ctx = webrtc_streamer(
            key="detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=lambda: YOLOProcessor(confidence, model),
        )
        if webrtc_ctx.state.playing:
            st.success("Webcam is live!")
    except Exception as e:
        st.error("Error initializing webcam. Please check your setup.")
        logger.error(f"WebRTC setup error: {e}")


def main():
    initialize_session_state()

    st.set_page_config(
        page_title="Underwater Object Detection",
        page_icon="🌊",
        layout="wide"
    )

    st.title("Underwater Object Detection using YOLOv8")
    st.markdown("""
        Detect underwater objects in real time with YOLOv8.
        Select your model and input source to get started.
    """)

    with st.sidebar:
        st.header("Model Configuration")
        model_type = st.selectbox("Select Model", list(YOLO_WEIGHTS.keys()))
        confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)
        
        model = get_yolo_model(str(YOLO_WEIGHTS[model_type]))
        if not model:
            st.error("Model loading failed. Check the model path.")
            st.stop()
        
        source_type = st.selectbox("Select Input Source", SOURCES_LIST)

    if source_type == "Image":
        infer_uploaded_image(confidence, model)
    elif source_type == "Video":
        infer_uploaded_video(confidence, model)
    elif source_type == "Webcam":
        setup_webcam_interface(confidence, model)

if __name__ == "__main__":
    main()
