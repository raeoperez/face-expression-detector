
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

LM = {
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_inner": 13,
    "lower_inner": 14,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "nose_tip": 1,
}


def distance(a, b):
    return np.linalg.norm(np.array([a.x - b.x, a.y - b.y]))


def classify_expression(landmarks):

    L = landmarks[LM["mouth_left"]]
    R = landmarks[LM["mouth_right"]]
    U = landmarks[LM["upper_inner"]]
    D = landmarks[LM["lower_inner"]]

    LE_in = landmarks[LM["left_eye_inner"]]
    LE_out = landmarks[LM["left_eye_outer"]]
    RE_in = landmarks[LM["right_eye_inner"]]
    RE_out = landmarks[LM["right_eye_outer"]]

    nose = landmarks[LM["nose_tip"]]

    # Mouth geometry
    mouth_width = distance(L, R)
    mouth_height = distance(U, D)
    mar = mouth_height / (mouth_width + 1e-6)

    # Smile curvature 
    dx = R.x - L.x
    dy = R.y - L.y
    angle = np.degrees(np.arctan2(dy, dx))
    curvature = -angle / 15

    # Corner height
    mouth_center_y = (U.y + D.y) / 2
    corner_height = ((mouth_center_y - L.y) + (mouth_center_y - R.y)) / 2
    corner_norm = corner_height / (mouth_width + 1e-6)

    # Eye squeeze
    left_eye = distance(LE_in, LE_out)
    right_eye = distance(RE_in, RE_out)
    eye_avg = (left_eye + right_eye) / 2
    eye_squint = max(0, 0.14 - eye_avg)

    smile_score = (
        0.65 * corner_norm +
        0.25 * eye_squint +
        0.10 * mar
    )

    sad_score = (
        -0.70 * corner_norm +
        0.10 * (0.22 - mar)
    )

    if smile_score > 0.045:
        return "smiling"
    elif sad_score > 0.12:
        return "sad"
    else:
        return "no reaction"


class EmotionProcessor(VideoProcessorBase):

    def __init__(self):
        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        label = "no face"

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0].landmark
            label = classify_expression(face)

        color = (0, 255, 0) if label == "smiling" else \
                (0, 165, 255) if label == "no reaction" else (0, 0, 255)

        cv2.putText(img, label, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("😊 Real-Time Web-based Face Expression Detector")

webrtc_streamer(
    key="emotions",
    video_processor_factory=EmotionProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    mode=WebRtcMode.SENDRECV,
)
