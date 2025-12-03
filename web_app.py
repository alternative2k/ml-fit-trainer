import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Load model
clf = joblib.load("model.pkl")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ------------ Angle calc -------------
def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


# ------------ Streamlit UI -------------
st.set_page_config(page_title="AI Fitness Trainer", layout="wide")

st.title("Fitness Trainer")
st.write("Turn in the camera and start doing exercise!")

run = st.checkbox("▶️ Start Camera")
FRAME_WINDOW = st.image([])

# ------------ State variables -------------
if "push_count" not in st.session_state:
    st.session_state.push_count = 0

if "squat_count" not in st.session_state:
    st.session_state.squat_count = 0

if "push_stage" not in st.session_state:
    st.session_state.push_stage = "up"

if "squat_stage" not in st.session_state:
    st.session_state.squat_stage = "up"


# ------------ MAIN CAMERA LOOP -------------
if run:
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while run:

        ret, frame = cap.read()
        if not ret:
            st.write("❌ Camera not found")
            break

        frame = cv2.resize(frame, (0, 0), fx=1.3, fy=1.3)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        label = "no_pose"

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            landmarks = [[p.x * w, p.y * h, p.z] for p in lm]

            # Prediction input
            ml_input = np.array(
                [[p.x, p.y, p.z] for p in lm]
            ).flatten().reshape(1, -1)

            label = clf.predict(ml_input)[0]

            # PUSHUPS — elbow angle
            elbow_angle = angle_between(
                landmarks[11][:2],
                landmarks[13][:2],
                landmarks[15][:2]
            )

            if elbow_angle > 150:
                st.session_state.push_stage = "up"

            if elbow_angle < 90 and st.session_state.push_stage == "up":
                st.session_state.push_stage = "down"
                st.session_state.push_count += 1

            # SQUATS — knee angle
            knee_angle = angle_between(
                landmarks[23][:2],
                landmarks[25][:2],
                landmarks[27][:2]
            )

            if knee_angle > 160:
                st.session_state.squat_stage = "up"

            if knee_angle < 100 and st.session_state.squat_stage == "up":
                st.session_state.squat_stage = "down"
                st.session_state.squat_count += 1

            # Draw pose on frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # UI text
        cv2.putText(frame, f"Exercise: {label}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(frame, f"Pushups: {st.session_state.push_count}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        cv2.putText(frame, f"Squats: {st.session_state.squat_count}", (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 255), 3)

        FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True)

        time.sleep(0.01)

    cap.release()
