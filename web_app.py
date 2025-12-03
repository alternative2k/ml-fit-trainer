import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os


SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
# Load model
clf = joblib.load("model.pkl")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


# STREAMLIT UI

st.set_page_config(page_title="AI Fitness Trainer", layout="centered") 

st.title("Fitness Trainer")
st.write("Turn on the camera to start exercise detection.")

run = st.checkbox("Start Camera")

hide_video = st.checkbox("Hide Camera Preview")

FRAME_WINDOW = st.image([])


#STATE VARs
if "push_count" not in st.session_state:
    st.session_state.push_count = 0

if "squat_count" not in st.session_state:
    st.session_state.squat_count = 0

if "push_stage" not in st.session_state:
    st.session_state.push_stage = "up"

if "squat_stage" not in st.session_state:
    st.session_state.squat_stage = "up"

if "recording" not in st.session_state:
    st.session_state.recording = False


#FILE NAME 
def get_next_session_filename():
    i = 1
    while True:
        name = f"session_{i:03d}.mp4"
        if not os.path.exists(os.path.join("sessions", name)):
            return name
        i += 1



# CAMERA

if run:

    # Start camera
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FPS, 60) 

    # Get FPS safely
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 5 or fps > 60:
        fps = 30

    width = int(cap.get(3))
    height = int(cap.get(4))

    # Start recording 
    if not st.session_state.recording:
        filename = os.path.join(SESSION_DIR, get_next_session_filename())
        st.session_state.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
        st.session_state.recording = True

    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera not found")
            break

        # Write frame to file 
        st.session_state.video_writer.write(frame)

        display_frame = frame

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        label = "no_pose"

        # looking for visibility of all landmarks
        landmarks_present = False
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5 and 
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5 and
                lm[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.5 and
                lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility > 0.5):
                landmarks_present = True

        #excercise only when the pose in recognized
        if landmarks_present:
            lm = results.pose_landmarks.landmark
            h, w, _ = display_frame.shape
            landmarks = [[p.x*w, p.y*h, p.z] for p in lm]

            ml_input = np.array([[p.x, p.y, p.z] for p in lm]).flatten().reshape(1, -1)
            label = clf.predict(ml_input)[0]

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

            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        else:
            cv2.putText(display_frame, "Pose not fully visible", (15, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.putText(display_frame, f"Exercise: {label}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(display_frame, f"Pushups: {st.session_state.push_count}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        cv2.putText(display_frame, f"Squats: {st.session_state.squat_count}", (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 255), 3)

        if not hide_video:
            FRAME_WINDOW.image(display_frame, channels="BGR", width=600) 

    cap.release()
    st.session_state.video_writer.release()
    st.session_state.recording = False
    st.write("Video saved.")

