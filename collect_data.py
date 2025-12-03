import cv2
import mediapipe as mp
import pandas as pd
import os

mp_pose = mp.solutions.pose

# соответствие имён файлов → метки классов
LABELS = {
    "pushups": "pushups",
    "pushups_2": "pushups",
    "squat": "squat",
    "squat_2": "squat"
}

def extract_keypoints(video_path, label):
    """Video, keypoints through MediaPipe and return DataFrame."""
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            row = []
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row.append(label)
            data.append(row)

    cap.release()
    df = pd.DataFrame(data)
    return df


def main():
    videos_path = "dataset"
    output_path = os.path.join("dataset", "keypoints.csv")

    all_data = []

    print(f"look for video: {os.path.abspath(videos_path)}")

    for file in os.listdir(videos_path):
        if file.endswith(".mp4"):
            name = os.path.splitext(file)[0]  # pushups.mp4 -> pushups
            label = LABELS.get(name)

            if label is None:
                print(f"UNKNOWN FILE {file}, skip")
                continue

            path = os.path.join(videos_path, file)
            print(f"processing {file} → label = {label}")

            df = extract_keypoints(path, label)
            if len(df) == 0:
                print(f"In file {file} no poses were found")
            all_data.append(df)

    if not all_data:
        print("couldn't find any video")
        return

    final = pd.concat(all_data, axis=0)
    final.to_csv(output_path, index=False)
    print(f"saved dataset to {output_path}, rows: {len(final)}")


if __name__ == "__main__":
    main()
