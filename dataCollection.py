import cv2
import mediapipe as mp
import numpy as np
import os
import time

# === SETTINGS ===
SAVE_DURATION = 2  # seconds to record each s
GESTURES = {
    '1': 'Hello',
    '2': 'See you later',
    '3': 'You good'
    
}
DATA_DIR = "collected_data"
FPS = 30  # Video FPS

# === Mediapipe Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# === Create directories ===
for label in GESTURES.values():
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

cap = cv2.VideoCapture(0)

print("Press 1, 2, or 3 to start recording gesture. Press q to quit.")

sample_counter = {label: 0 for label in GESTURES.values()}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if chr(key) in GESTURES:
        label = GESTURES[chr(key)]
        sample_counter[label] += 1
        npy_path = os.path.join(DATA_DIR, label, f"sample_{sample_counter[label]}.npy")
        mp4_path = os.path.join(DATA_DIR, label, f"sample_{sample_counter[label]}.mp4")

        print(f"Recording {label} -> sample {sample_counter[label]}")

        # === Setup video writer ===
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(mp4_path, fourcc, FPS, (width, height))

        landmarks_list = []
        start_time = time.time()

        while time.time() - start_time < SAVE_DURATION:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks_list.append(landmarks)
            else:
                landmarks_list.append([0] * 63)  # pad if no hand

            out.write(frame)
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        np.save(npy_path, np.array(landmarks_list))
        print(f"Saved {npy_path} and {mp4_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
