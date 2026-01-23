
# file: app_web.py
import math
from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# Realtime video (WebRTC)
try:
    import av  # required by streamlit-webrtc
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

st.set_page_config(page_title="Face Expression Detector (Web)", page_icon="😊")

st.title("😊 Real‑Time Expression Detector (Web Version)")
st.write("Allow camera access to see live predictions. If WebRTC is blocked, use the Snapshot mode below.")

# -----------------------------
# MediaPipe setup (0.10.x API)
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmark indices we use (MediaPipe FaceMesh - 468 points)
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


def _dist_norm(a, b):
    """Distance in normalized [0..1] coordinate space."""
    dx, dy = (a.x - b.x), (a.y - b.y)
    return math.hypot(dx, dy)


def classify_expression(landmarks):
    """
    Heuristic scoring similar to your original logic.
    Uses normalized FaceMesh points so it's resolution-independent.
    """
    L = landmarks[LM["mouth_left"]]
    R = landmarks[LM["mouth_right"]]
    U = landmarks[LM["upper_inner"]]
    D = landmarks[LM["lower_inner"]]
    LE_in = landmarks[LM["left_eye_inner"]]
    LE_out = landmarks[LM["left_eye_outer"]]
    RE_in = landmarks[LM["right_eye_inner"]]
    RE_out = landmarks[LM["right_eye_outer"]]

    mouth_width = _dist_norm(L, R) + 1e-6
    mouth_height = _dist_norm(U, D)
    mar = mouth_height / mouth_width  # mouth aspect ratio

    # Corner height / curvature proxies
    dx = R.x - L.x
    dy = R.y - L.y
    angle_deg = math.degrees(math.atan2(dy, dx))
    _curvature = -angle_deg / 15.0  # kept for parity but not used directly

    mouth_center_y = (U.y + D.y) / 2.0
    corner_height = ((mouth_center_y - L.y) + (mouth_center_y - R.y)) / 2.0
    corner_norm = corner_height / mouth_width

    # Eye squint proxy
    left_eye = _dist_norm(LE_in, LE_out)
    right_eye = _dist_norm(RE_in, RE_out)
    eye_avg = (left_eye + right_eye) / 2.0
    eye_squint = max(0.0, 0.14 - eye_avg)

    # Heuristic scores (from your original weighting)
    smile_score = (0.65 * corner_norm) + (0.25 * eye_squint) + (0.10 * mar)
    sad_score = (-0.70 * corner_norm) + (0.10 * (0.22 - mar))

    if smile_score > 0.045:
        return "smiling"
    elif sad_score > 0.12:
        return "sad"
    else:
        return "no reaction"


# -----------------------------
# Cache the FaceMesh instance
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_mesh(refine: bool, det_conf: float, track_conf: float):
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=refine,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    refine = st.toggle("Refine landmarks (eyes/irises)", value=True)
    det_conf = st.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
    track_conf = st.slider("Min tracking confidence", 0.0, 1.0, 0.5, 0.05)
    draw_mesh = st.toggle("Draw face mesh", value=False)

mesh = get_mesh(refine, det_conf, track_conf)

# -----------------------------
# Realtime Video Processor (WebRTC)
# -----------------------------
@dataclass
class _StreamState:
    label: str = "no face"


if WEBRTC_AVAILABLE:

    class VideoProcessor:
        def __init__(self):
            self.state = _StreamState()

        def recv(self, frame):
            # Convert to numpy BGR image
            img_bgr = frame.to_ndarray(format="bgr24")
            # MediaPipe expects RGB
