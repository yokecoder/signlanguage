import numpy as np
import cv2
import os
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

DATA_DIR = "gesture_data"  # your data folder
LABEL = "Hello"    # change to the gesture folder or prefix
# Find one sample file to test
files = [f for f in os.listdir(DATA_DIR) if f.startswith(LABEL)]
if not files:
    print(f"No files found for label {LABEL}")
    exit()

file_path = os.path.join(DATA_DIR, files[0])
sequence = np.load(file_path)
print(f"Loaded sequence shape: {sequence.shape}")  # Should be (SEQ_LEN, 63)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

for frame_landmarks in sequence:
    img = 255 * np.ones((480, 640, 3), dtype=np.uint8)

    landmarks = []
    for i in range(0, len(frame_landmarks), 3):
        landmarks.append(
            landmark_pb2.NormalizedLandmark(
                x=frame_landmarks[i],
                y=frame_landmarks[i + 1],
                z=frame_landmarks[i + 2],
            )
        )

    hand_landmarks = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)
    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Playback", img)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
