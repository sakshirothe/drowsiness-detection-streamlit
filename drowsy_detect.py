import cv2
import mediapipe as mp
import pygame
import numpy as np
import os
import time
from math import hypot

# -----------------------------
# INITIAL SETUP
# -----------------------------
pygame.mixer.init()

ALARM_FILE = "alarm.mp3"
if not os.path.exists(ALARM_FILE):
    print("Error: alarm.mp3 not found in the project folder.")
    exit()

pygame.mixer.music.load(ALARM_FILE)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# CONSTANTS
# -----------------------------
EYE_CLOSED_THRESHOLD = 0.21    # EAR threshold; may need slight tuning
DROWSY_SECONDS = 10            # eyes closed for 10 sec => alarm
alarm_active = False
eyes_closed_start_time = None

# FaceMesh eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def euclidean(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye_points):
    """
    EAR formula:
    (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    vertical_1 = euclidean(eye_points[1], eye_points[5])
    vertical_2 = euclidean(eye_points[2], eye_points[4])
    horizontal = euclidean(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def get_eye_points(landmarks, eye_indices, w, h):
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))
    return points


def start_alarm():
    global alarm_active
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)
    alarm_active = True


def stop_alarm():
    global alarm_active
    pygame.mixer.music.stop()
    alarm_active = False


def is_thumbs_up(hand_landmarks):
    """
    Simple thumbs-up detection:
    - thumb tip above thumb IP joint
    - other fingers folded down
    """
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


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read webcam frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    eyes_closed = False
    thumbs_up_detected = False
    ear_text = "EAR: N/A"

    # -----------------------------
    # FACE / EYE DETECTION
    # -----------------------------
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        left_eye_points = get_eye_points(landmarks, LEFT_EYE, w, h)
        right_eye_points = get_eye_points(landmarks, RIGHT_EYE, w, h)

        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0

        ear_text = f"EAR: {avg_ear:.3f}"

        # draw eye points
        for point in left_eye_points + right_eye_points:
            cv2.circle(frame, point, 2, (0, 255, 255), -1)

        if avg_ear < EYE_CLOSED_THRESHOLD:
            eyes_closed = True
        else:
            eyes_closed = False

    # -----------------------------
    # HAND / THUMBS-UP DETECTION
    # -----------------------------
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_thumbs_up(hand_landmarks):
                thumbs_up_detected = True
                break

    # -----------------------------
    # DROWSINESS TIMER LOGIC
    # -----------------------------
    current_time = time.time()

    if eyes_closed:
        if eyes_closed_start_time is None:
            eyes_closed_start_time = current_time

        closed_duration = current_time - eyes_closed_start_time

        if closed_duration >= DROWSY_SECONDS and not alarm_active:
            start_alarm()
    else:
        eyes_closed_start_time = None

        # auto-stop when eyes reopen
        if alarm_active:
            stop_alarm()

    # -----------------------------
    # STOP ALARM WITH THUMBS-UP
    # -----------------------------
    if alarm_active and thumbs_up_detected:
        stop_alarm()
        eyes_closed_start_time = None

    # -----------------------------
    # STATUS TEXT
    # -----------------------------
    if eyes_closed_start_time is not None:
        closed_duration = current_time - eyes_closed_start_time
    else:
        closed_duration = 0

    if alarm_active:
        status = "DROWSY ALERT! Alarm ON"
        color = (0, 0, 255)
    elif thumbs_up_detected:
        status = "Thumbs Up Detected"
        color = (0, 255, 0)
    elif eyes_closed:
        status = f"Eyes Closed: {closed_duration:.1f}s"
        color = (0, 255, 255)
    else:
        status = "Eyes Open / Monitoring"
        color = (0, 255, 0)

    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, ear_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Close eyes > 10 sec => alarm", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Thumbs up => stop alarm | S => stop | Q => quit", (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        stop_alarm()
        eyes_closed_start_time = None
        print("Alarm stopped manually.")

    if key == ord("q"):
        stop_alarm()
        break

# -----------------------------
# CLEANUP
# -----------------------------
stop_alarm()
cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()