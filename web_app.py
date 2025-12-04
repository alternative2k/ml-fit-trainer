import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os

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

# CHANGED: we don't use a manual while-loop checkbox any more
run = st.checkbox("Start Camera")

hide_video = st.checkbox("Hide Camera Preview")

# CHANGED: this is now only used for showing a static frame if needed
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

# CHANGED: recording flag now used with WebRTC recording
if "recording" not in st.session_state:
    st.session_state.recording = False

# NEW: store latest exercise label in state
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


# NEW: Video transformer for streamlit-webrtc
class FitnessVideoTransformer(VideoTransformerBase):
    """
    This class processes each incoming video frame from the browser webcam.
    It replaces the manual OpenCV while-loop and keeps all logic:
    - pose detection
    - exercise classification
    - rep counting
    - drawing overlays
    """

    def __init__(self):
        # Create a single MediaPipe pose instance per transformer
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Prepare recording file on first use
        if not st.session_state.recording:
            filename = os.path.join(SESSION_DIR, get_next_session_filename())
            # Use a default fps (WebRTC does not give us exact fps easily)
            fps = 30
            # Width and height will be updated when first frame arrives
            self.video_writer = None
            self.record_filename = filename
            st.session_state.recording = True
        else:
            self.video_writer = None
            self.record_filename = None

    def _init_writer_if_needed(self, frame_array):
        """
        Lazily initialize the VideoWriter once we know frame size.
        """
        if self.video_writer is None and st.session_state.recording:
            height, width, _ = frame_array.shape
            self.video_writer = cv2.VideoWriter(
                self.record_filename,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (width, height),
            )

    def recv(self, frame):
        """
        This method is called for each video frame.
        `frame` is a VideoFrame from streamlit-webrtc.
        """
        # Convert from WebRTC frame to OpenCV BGR image
        img = frame.to_ndarray(format="bgr24")

        # Initialize writer with correct size
        self._init_writer_if_needed(img)

        # Write raw frame to file if recording
        if self.video_writer is not None:
            self.video_writer.write(img)

        display_frame = img.copy()
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        label = "no_pose"

        # looking for visibility of all landmarks
        landmarks_present = False
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if (
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5
                and lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5
                and lm[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.5
                and lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility > 0.5
            ):
                landmarks_present = True

        # exercise only when the pose is recognized
        if landmarks_present:
            lm = results.pose_landmarks.landmark
            h, w, _ = display_frame.shape
            landmarks = [[p.x * w, p.y * h, p.z] for p in lm]

            ml_input = (
                np.array([[p.x, p.y, p.z] for p in lm])
                .flatten()
                .reshape(1, -1)
            )
            label = clf.predict(ml_input)[0]

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

            knee_angle = angle_between(
                landmarks[23][:2],
                landmarks[25][:2],
                landmarks[27][:2],
            )

            if knee_angle > 160:
                st.session_state.squat_stage = "up"
            if (
                knee_angle < 100
                and st.session_state.squat_stage == "up"
            ):
                st.session_state.squat_stage = "down"
                st.session_state.squat_count += 1

            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )
        else:
            cv2.putText(
                display_frame,
                "Pose not fully visible",
                (15, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Store label in session so UI can display it
        st.session_state.exercise_label = str(label)

        # Draw overlay texts (exercise + counters)
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

        # If user wants to hide preview, return a black frame
        if hide_video:
            black = np.zeros_like(display_frame)
            return black
        else:
            return display_frame

    def __del__(self):
        """
        Ensure video writer is released when transformer is destroyed.
        """
        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.release()


# CAMERA / WEBRTC SECTION

if run:
    # NEW: create a WebRTC-based video streamer instead of cv2.VideoCapture
    webrtc_ctx = webrtc_streamer(
        key="fitness-webrtc",
        mode=WebRtcMode.SENDRECV,  # send and receive video
        video_transformer_factory=FitnessVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,  # process frames asynchronously for better performance
    )

    # Show current counts and label under the video
    st.write(f"Current exercise: {st.session_state.exercise_label}")
    st.write(f"Pushups counted: {st.session_state.push_count}")
    st.write(f"Squats counted: {st.session_state.squat_count}")

    # Info text about recording location
    st.info(
        "Session video is being saved in the 'sessions' folder in the app "
        "filesystem (note: on Streamlit Cloud, files are temporary per session)."
    )
else:
    # When camera is off, show last known counts
    st.write(f"Last exercise: {st.session_state.exercise_label}")
    st.write(f"Total pushups: {st.session_state.push_count}")
    st.write(f"Total squats: {st.session_state.squat_count}")
