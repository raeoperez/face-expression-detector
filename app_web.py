
# file: web_app.py
import math
from dataclasses import dataclass

import av  # from PyAV, required by streamlit-webrtc
import cv2
import numpy as np
import streamlit as st

try:
    # streamlit-webrtc is the recommended way to do realtime video on Streamlit Cloud
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
except Exception as e:
    webrtc_streamer = None

import mediapipe as mp

st.set_page_config(page_title="Face Expression Detector (Web)", page_icon="😊")

st.title("😊 Real‑Time Expression Detector (Web Version)")
st.write("Allow camera access to see live predictions.")

# -----------------------------
# MediaPipe setup (0.10.x API)
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmark indices we use (MediaPipe FaceMesh, 468 pts)
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
    # Works on normalized [0..1] coordinates
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
    mar = mouth_height / mouth_width  # mouth aspect ratio (roughly)

    # Corner height/curvature proxies
    dx = R.x - L.x
    dy = R.y - L.y
    angle_deg = math.degrees(math.atan2(dy, dx))
    curvature = -angle_deg / 15.0  # unused but kept for parity

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
# Video Processor (webrtc)
# -----------------------------
@dataclass
class State:
    label: str = "no face"


class VideoProcessor:
    def __init__(self):
        self.state = State()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert to numpy BGR image
        img_bgr = frame.to_ndarray(format="bgr24")
        h, w = img_bgr.shape[:2]

        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = mesh.process(img_rgb)

        label = "no face"
        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0].landmark
            label = classify_expression(face)

            if draw_mesh:
                mp_drawing.draw_landmarks(
                    image=img_bgr,
                    landmark_list=result.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=img_bgr,
                    landmark_list=result.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                )

        # Overlay label
        text = label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img_bgr, (10, 10), (10 + tw + 16, 10 + th + 16), (0, 0, 0), -1)
        cv2.putText(img_bgr, text, (18, 10 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        self.state.label = label
        # Return frame
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


# -----------------------------
# Start WebRTC streamer
# -----------------------------
if webrtc_streamer is None:
    st.error(
        "streamlit-webrtc is not installed. Please add `streamlit-webrtc` to requirements.txt "
        "or use the fallback snapshot mode below."
    )
else:
    rtc_config = RTCConfiguration(
        {
            # Use Google STUN server for NAT traversal; Streamlit Cloud supports this setup.
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    ctx = webrtc_streamer(
        key="expr-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    if ctx.video_processor:
        st.caption("Status: streaming…  If you see a blank video, check that your browser allows camera access.")

st.markdown(
    """
---
**Tip:** If your network blocks WebRTC, you can still try snapshot mode below (one-shot analysis).
"""
)

# -----------------------------
# Fallback: snapshot mode
# -----------------------------
img_file = st.camera_input("Snapshot mode (click 'Take photo')", disabled=False)
if img_file is not None:
    file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)

    if res.multi_face_landmarks:
        face = res.multi_face_landmarks[0].landmark
        label = classify_expression(face)
        if draw_mesh:
            mp_drawing.draw_landmarks(
                image=bgr,
                landmark_list=res.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=bgr,
                landmark_list=res.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
            )
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption=f"Detected: {label}")
    else:
        st.warning("No face detected in the snapshot.")
``
