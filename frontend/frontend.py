import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import requests
import numpy as np


st.title("Live Webcam Tracking")


def log(message):
    st.info(message)

def video_frame_callback(frame):
    if backend_url == "":
        return frame
    img = frame.to_ndarray(format="bgr24")
    _, img_encoded = cv2.imencode('.jpg', img)

    files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg', {'Expires': '0'})}

    response = requests.post(backend_url, files=files)

    if response.status_code == 200:

        img_color = np.frombuffer(response.content, dtype=np.uint8)
        img_color = cv2.imdecode(img_color, cv2.IMREAD_COLOR)

        return av.VideoFrame.from_ndarray(img_color, format="bgr24")
    else:

        return None

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

modele = st.radio(
    "Sélection du modèle",
    ["***Aucun modèle***", "***Bounding box et mask***", "***Mask***", "***UNet***"])

if modele == '***Aucun modèle***':
    backend_url = ""
elif modele == '***Bounding box et mask***':
    backend_url = "http://127.0.0.1:5000/upload_mask_42m"
elif modele == '***Mask***':
    backend_url = "http://127.0.0.1:5000/upload_mask_21m"
elif modele == '***UNet***':
    backend_url = "http://127.0.0.1:5000/upload_mask_unet"