
# file: app_web.py
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# Try to enable realtime video via WebRTC.
# If packages aren't available, we fall back to Snapshot Mode only.
WEBRTC_AVAILABLE = False
try:
    import av  # required by streamlit-webrtc
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

st.set_page_config(page_title="Face Expression Detector (Web)", page_icon="😊")

st.title("😊 Real‑Time Expression Detector (Web Version)")
st.write(
    "Allow camera access to see live predictions. If your network blocks WebRTC, use Snapshot Mode at the bottom."
)

# -------------------------------------------------------------------
# MediaPipe FaceMesh setup (0.10.x API)
# -------------------------------------------------------------------
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


def _dist_norm(a, b) -> float:
    """Distance in normalized [0..1] coordinate space."""
    dx, dy = (a.x - b.x), (a.y - b.y)
    return math.hypot(dx, dy)


def classify_expression(landmarks) -> str:
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

    # Corner height proxy
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


# -------------------------------------------------------------------
# Cache the FaceMesh instance (created once per settings combination)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_mesh(refine: bool, det_conf: float, track_conf: float):
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=refine,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    refine = st.toggle("Refine landmarks (eyes/irises)", value=True)
    det_conf = st.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
    track_conf = st.slider("Min tracking confidence", 0.0, 1.0, 0.5, 0.05)
    draw_mesh = st.toggle("Draw face mesh overlay", value=False)

    st.subheader("Camera Constraints (WebRTC)")
    facing = st.selectbox("Facing mode", ["user (front)", "environment (back)"], index=0)
    width = st.selectbox("Width", [640, 1280], index=0)
    height = st.selectbox("Height", [480, 720], index=0)
    fps = st.selectbox("FPS", [15, 30], index=0)

mesh = get_mesh(refine, det_conf, track_conf)

# -------------------------------------------------------------------
# Realtime Video Processor (WebRTC)
# -------------------------------------------------------------------
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

            # Overlay label background + text
            text = label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(img_bgr, (10, 10), (10 + tw + 16, 10 + th + 16), (0, 0, 0), -1)
            cv2.putText(img_bgr, text, (18, 10 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            self.state.label = label

            # Return frame
            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    # ----------------- ICE / STUN / TURN configuration -----------------
    # Default STUN servers (public)
    ice_servers: List[Dict[str, Any]] = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478?transport=udp"]},
    ]

    # Optional TURN config from st.secrets to traverse strict networks.
    # Example in .streamlit/secrets.toml:
    # [turn]
    # urls = ["turn:your.turn.server:3478?transport=udp"]
    # username = "turn_username"
    # credential = "turn_password"
    try:
        turn = st.secrets.get("turn", {})
        turn_urls = turn.get("urls")
        turn_username = turn.get("username")
        turn_credential = turn.get("credential")
        if turn_urls:
            if isinstance(turn_urls, (list, tuple)):
                turn_entry: Dict[str, Any] = {"urls": list(turn_urls)}
            else:
                turn_entry = {"urls": [str(turn_urls)]}
            if turn_username and turn_credential:
                turn_entry["username"] = str(turn_username)
                turn_entry["credential"] = str(turn_credential)
            ice_servers.append(turn_entry)
    except Exception:
        # If secrets aren't configured, we just skip TURN.
        pass

    rtc_config = RTCConfiguration({"iceServers": ice_servers})

    # Gentler constraints help avoid OverconstrainedError and permission issues
    media_stream_constraints = {
        "video": {
            "facingMode": "user" if facing.startswith("user") else "environment",
            "width": {"ideal": int(width)},
            "height": {"ideal": int(height)},
            "frameRate": {"ideal": int(fps)},
        },
        "audio": False,
    }

    st.subheader("Live Webcam (WebRTC)")

    ctx = webrtc_streamer(
        key="expr-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints=media_stream_constraints,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    with st.expander("Connection / Permission Status", expanded=False):
        if ctx is None:
            st.write("WebRTC context is not available.")
        else:
            # These attributes help you see what the frontend is doing
            st.write(f"WebRTC state: **{getattr(ctx, 'state', None)}**")
            st.write(f"Playing: **{getattr(ctx, 'playing', None)}**")
            if ctx.video_processor:
                st.write(f"Last label: **{ctx.video_processor.state.label}**")
            st.info(
                "If you don't get a camera prompt: "
                "1) Click the lock icon (🔒) → Allow Camera, then reload. "
                "2) Close other apps using the camera (Zoom/Teams/Meet). "
                "3) Try a different network or mobile hotspot (WebRTC may be blocked)."
            )
else:
    st.warning(
        "Realtime mode requires `streamlit-webrtc` and `av`. "
        "They were not imported, so only Snapshot Mode is available."
    )

st.markdown("---")
st.subheader("Snapshot Mode (no WebRTC)")

# -------------------------------------------------------------------
# Fallback: snapshot mode using st.camera_input (works without WebRTC)
# -------------------------------------------------------------------
img_file = st.camera_input("Take a photo")
if img_file is not None:
    file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode image from camera.")
    else:
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
