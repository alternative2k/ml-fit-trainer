import streamlit as st
import mediapipe as mp
import numpy as np
import joblib
import time
import os
import importlib  # NEW: for lazy cv2 import

# NEW: import streamlit-webrtc for browser-based webcam on Streamlit Cloud
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# Load model
clf = joblib.load("model.pkl")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


# STREAMLIT UI
st.set_page_config(page_title="AI Fitness Trainer", layout="centered")

st.title("Fitness Trainer")
st.write("Turn on the camera to start exercise detection.")

run = st.checkbox("Start Camera")
hide_video = st.checkbox("Hide Camera Preview")
FRAME_WINDOW = st.empty()

# STATE VARs
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
if "exercise_label" not in st.session_state:
    st.session_state.exercise_label = "no_pose"


# FILE NAME
def get_next_session_filename():
    i = 1
    while True:
        name = f"session_{i:03d}.mp4"
        if not os.path.exists(os.path.join(SESSION_DIR, name)):
            return name
        i += 1


# MODIFIED: Video transformer with LAZY cv2 import
class FitnessVideoTransformer(VideoTransformerBase):
    """
    Processes each webcam frame with lazy cv2 import to fix Streamlit Cloud.
    Keeps ALL features: pose detection, exercise classification, rep counting, recording.
    """

    def __init__(self):
        # Create MediaPipe pose instance
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        # NEW: cv2 is lazily loaded only when first frame arrives
        self.cv2 = None
        
        # Video writer setup
        self.video_writer = None
        self.record_filename = None
        if not st.session_state.recording:
            self.record_filename = os.path.join(SESSION_DIR, get_next_session_filename())
            st.session_state.recording = True

    def _lazy_import_cv2(self):
        """NEW: Import cv2 only when first frame needs it (fixes Streamlit Cloud)"""
        if self.cv2 is None:
            self.cv2 = importlib.import_module('cv2')
        return self.cv2

    def _init_writer_if_needed(self, frame_array):
        """Initialize VideoWriter once frame size is known"""
        if self.video_writer is None and self.record_filename:
            cv2 = self._lazy_import_cv2()
            height, width, _ = frame_array.shape
            self.video_writer = cv2.VideoWriter(
                self.record_filename,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,  # Fixed FPS for WebRTC
                (width, height),
            )

    def recv(self, frame):
        """Process each incoming video frame"""
        # Convert WebRTC frame to numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        
        # NEW: Get cv2 reference safely
        cv2 = self._lazy_import_cv2()
        
        # Initialize recording with correct frame size
        self._init_writer_if_needed(img)
        
        # Write frame to video file if recording
        if self.video_writer is not None:
            self.video_writer.write(img)

        # Process frame (copy for drawing)
        display_frame = img.copy()
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        label = "no_pose"
        landmarks_present = False

        # Check landmark visibility (same logic as original)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if (
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5
                and lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5
                and lm[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.5
                and lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility > 0.5
            ):
                landmarks_present = True

        # Exercise detection and counting (EXACT SAME LOGIC AS ORIGINAL)
        if landmarks_present:
            lm = results.pose_landmarks.landmark
            h, w, _ = display_frame.shape
            landmarks = [[p.x * w, p.y * h, p.z] for p in lm]

            # ML classification
            ml_input = (
                np.array([[p.x, p.y, p.z] for p in lm])
                .flatten()
                .reshape(1, -1)
            )
            label = clf.predict(ml_input)[0]

            # Pushup counting (elbow angle)
            elbow_angle = angle_between(
                landmarks[11][:2],
                landmarks[13][:2],
                landmarks[15][:2],
            )
            if elbow_angle > 150:
                st.session_state.push_stage = "up"
            if elbow_angle < 90 and st.session_state.push_stage == "up":
                st.session_state.push_stage = "down"
                st.session_state.push_count += 1

            # Squat counting (knee angle)
            knee_angle = angle_between(
                landmarks[23][:2],
                landmarks[25][:2],
                landmarks[27][:2],
            )
            if knee_angle > 160:
                st.session_state.squat_stage = "up"
            if knee_angle < 100 and st.session_state.squat_stage == "up":
                st.session_state.squat_stage = "down"
                st.session_state.squat_count += 1

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )
        else:
            # Warning when pose not fully visible
            cv2.putText(
                display_frame,
                "Pose not fully visible",
                (15, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Store current exercise label
        st.session_state.exercise_label = str(label)

        # Draw overlay text (same as original)
        cv2.putText(
            display_frame,
            f"Exercise: {label}",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            display_frame,
            f"Pushups: {st.session_state.push_count}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3,
        )
        cv2.putText(
            display_frame,
            f"Squats: {st.session_state.squat_count}",
            (15, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 128, 255),
            3,
        )

        # Hide preview option (returns black frame but keeps counting)
        if hide_video:
            black_frame = np.zeros_like(display_frame)
            return black_frame
        return display_frame

    def __del__(self):
        """Clean up video writer"""
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            cv2 = self._lazy_import_cv2()
            self.video_writer.release()


# MAIN UI
if run:
    # WebRTC streamer (browser webcam)
    webrtc_ctx = webrtc_streamer(
        key="fitness-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=FitnessVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    # Live stats display
    st.write(f"**Current exercise**: {st.session_state.exercise_label}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pushups", st.session_state.push_count)
    with col2:
        st.metric("Squats", st.session_state.squat_count)

    # Recording info
    st.info(
        f"✅ Recording to: `{os.path.basename(st.session_state.get('record_filename', 'session.mp4'))}` "
        f"in `/sessions/` folder\n"
        f"*(Streamlit Cloud: files temporary per session)*"
    )
else:
    # Show stats when camera off
    st.write(f"**Last exercise**: {st.session_state.exercise_label}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Pushups", st.session_state.push_count)
    with col2:
        st.metric("Total Squats", st.session_state.squat_count)
