import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import streamlit.components.v1 as components
import av
import cv2
import mediapipe as mp
import time
import base64
from pathlib import Path
from math import hypot
from mediapipe.python.solutions import face_mesh
from mediapipe.python.solutions import hands
from mediapipe.python.solutions import drawing_utils

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("Driver Drowsiness Detection")
st.write("Close eyes for more than 10 seconds → alarm starts automatically. Show thumbs up → alarm stops automatically.")

mp_face_mesh = face_mesh
mp_hands = hands
mp_draw = drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EYE_CLOSED_THRESHOLD = 0.21
DROWSY_SECONDS = 10
ALARM_FILE = "alarm.mp3"


def euclidean(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye_points):
    vertical_1 = euclidean(eye_points[1], eye_points[5])
    vertical_2 = euclidean(eye_points[2], eye_points[4])
    horizontal = euclidean(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_eye_points(landmarks, eye_indices, w, h):
    return [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indices]


def is_thumbs_up(hand_landmarks):
    lm = hand_landmarks.landmark

    thumb_tip = lm[4]
    thumb_ip = lm[3]

    index_tip = lm[8]
    middle_tip = lm[12]
    ring_tip = lm[16]
    pinky_tip = lm[20]

    index_pip = lm[6]
    middle_pip = lm[10]
    ring_pip = lm[14]
    pinky_pip = lm[18]

    thumb_up = thumb_tip.y < thumb_ip.y
    index_folded = index_tip.y > index_pip.y
    middle_folded = middle_tip.y > middle_pip.y
    ring_folded = ring_tip.y > ring_pip.y
    pinky_folded = pinky_tip.y > pinky_pip.y

    return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded


def render_alarm_audio():
    path = Path(ALARM_FILE)
    if not path.exists():
        st.error("alarm.mp3 not found in the project folder.")
        return

    audio_bytes = path.read_bytes()
    audio_base64 = base64.b64encode(audio_bytes).decode()

    html = f"""
    <audio id="alarm-audio" autoplay loop>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    <script>
        const existing = window.parent.document.getElementById("global-alarm-audio");
        if (existing) {{
            existing.play().catch(() => {{}});
        }} else {{
            const parser = new DOMParser();
            const doc = parser.parseFromString(`
                <audio id="global-alarm-audio" autoplay loop>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
            `, "text/html");
            const audio = doc.body.firstChild;
            window.parent.document.body.appendChild(audio);
            audio.play().catch(() => {{}});
        }}
    </script>
    """
    components.html(html, height=0)


def stop_alarm_audio():
    html = """
    <script>
        const audio = window.parent.document.getElementById("global-alarm-audio");
        if (audio) {
            audio.pause();
            audio.currentTime = 0;
            audio.remove();
        }
    </script>
    """
    components.html(html, height=0)


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.eyes_closed_start_time = None
        self.alarm_active = False
        self.status = "Monitoring..."
        self.ear = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb)
        hand_results = self.hands.process(rgb)

        eyes_closed = False
        thumbs_up_detected = False
        closed_duration = 0.0

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark

            left_eye_points = get_eye_points(landmarks, LEFT_EYE, w, h)
            right_eye_points = get_eye_points(landmarks, RIGHT_EYE, w, h)

            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            self.ear = (left_ear + right_ear) / 2.0

            for point in left_eye_points + right_eye_points:
                cv2.circle(img, point, 2, (0, 255, 255), -1)

            if self.ear < EYE_CLOSED_THRESHOLD:
                eyes_closed = True

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if is_thumbs_up(hand_landmarks):
                    thumbs_up_detected = True
                    break

        now = time.time()

        # Start countdown when eyes are closed
        if eyes_closed and not self.alarm_active:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = now

            closed_duration = now - self.eyes_closed_start_time

            if closed_duration >= DROWSY_SECONDS:
                self.alarm_active = True

        elif not eyes_closed and not self.alarm_active:
            self.eyes_closed_start_time = None

        # Alarm stops ONLY by thumbs up
        if thumbs_up_detected and self.alarm_active:
            self.alarm_active = False
            self.eyes_closed_start_time = None

        if self.alarm_active:
            self.status = "DROWSY ALERT! Alarm ON"
            color = (0, 0, 255)
        elif thumbs_up_detected:
            self.status = "Thumbs Up Detected"
            color = (0, 255, 0)
        elif eyes_closed:
            if self.eyes_closed_start_time is not None:
                closed_duration = now - self.eyes_closed_start_time
            self.status = f"Eyes Closed: {closed_duration:.1f}s"
            color = (0, 255, 255)
        else:
            self.status = "Eyes Open / Monitoring"
            color = (0, 255, 0)

        cv2.putText(img, self.status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(img, f"EAR: {self.ear:.3f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(
            img,
            "Close eyes > 10 sec => alarm starts",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "Thumbs up => alarm stops",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

status_placeholder = st.empty()
audio_placeholder = st.empty()

if ctx.state.playing and ctx.video_processor:
    processor = ctx.video_processor

    # Continuous sync loop: reads alarm state and controls browser audio
    last_alarm_state = None
    last_status = None

    while ctx.state.playing:
        current_alarm = processor.alarm_active
        current_status = processor.status

        if current_status != last_status:
            status_placeholder.info(f"Live Status: {current_status}")
            last_status = current_status

        if current_alarm != last_alarm_state:
            if current_alarm:
                with audio_placeholder:
                    render_alarm_audio()
            else:
                with audio_placeholder:
                    stop_alarm_audio()
            last_alarm_state = current_alarm

        time.sleep(0.2)
else:
    stop_alarm_audio()

st.info("Click START and allow camera access. After that, close your eyes for 10 seconds to trigger the alarm. Show thumbs up to stop it.")