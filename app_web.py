
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import base64

st.set_page_config(page_title="Face Expression Detector", page_icon="😊")

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

    mouth_width = distance(L, R)
    mouth_height = distance(U, D)
    mar = mouth_height / (mouth_width + 1e-6)

    dx = R.x - L.x
    dy = R.y - L.y
    angle = np.degrees(np.arctan2(dy, dx))
    curvature = -angle / 15

    mouth_center_y = (U.y + D.y) / 2
    corner_height = ((mouth_center_y - L.y) + (mouth_center_y - R.y)) / 2
    corner_norm = corner_height / (mouth_width + 1e-6)

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


# ------------------ STREAMLIT UI -------------------

st.title("😊 Real-Time Expression Detector (Web Version)")
st.write("Please allow camera access.")

# JS camera widget
camera_js = """
<script>
let video = document.createElement('video');
video.setAttribute('autoplay', '');
video.setAttribute('playsinline', '');
video.style.display = 'none';
document.body.appendChild(video);

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    function sendFrame() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        let dataURL = canvas.toDataURL('image/jpeg');
        window.parent.postMessage({type: 'frame', data: dataURL}, "*");
        requestAnimationFrame(sendFrame);
    }
    sendFrame();
})
.catch(err => console.log(err));
</script>
"""

st.components.v1.html(camera_js, height=0)

if "frame" not in st.session_state:
    st.session_state.frame = None

frame_container = st.empty()
label_container = st.empty()

def process_frame(data_url):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(BytesIO(img_bytes))
    img = img.convert("RGB")
    img_np = np.array(img)

    h, w = img_np.shape[:2]
    rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:
        res = mesh.process(rgb)

    if res.multi_face_landmarks:
        face = res.multi_face_landmarks[0].landmark
        return img_np, classify_expression(face)
    else:
        return img_np, "no face"


st.markdown("""
<script>
window.addEventListener("message", (event) => {
    if (event.data.type === "frame") {
        window.parent.streamlitSend({"frame": event.data.data});
    }
});
</script>
""", unsafe_allow_html=True)

frame = st.session_state.get("frame")

if frame:
    img_np, label = process_frame(frame)
    frame_container.image(img_np)
    label_container.subheader(f"**{label}**")
