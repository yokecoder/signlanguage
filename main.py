import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from trainGestureModel import SEQ_LEN,labels
from collections import deque
import mediapipe as mp

MODEL_PATH = "gesture_model.keras"
LABELS_PATH = "labels.txt"
CONFIG_PATH = "config.npy"
CONF_THRESH = 0.8        # display only when confidence is high
SMOOTH_WINDOW = 5        # majority vote over last N predictions

# ---- Load model, labels, config ----
model = load_model(MODEL_PATH, compile=False)

# ---- Mediapipe setup ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
buffer = deque(maxlen=SEQ_LEN)
pred_hist = deque(maxlen=SMOOTH_WINDOW)

def extract63(results):
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        out = []
        for p in lm:
            out.extend([p.x, p.y, p.z])
        if len(out) == 63:
            return out
    # fallback (no hand)
    return [0.0]*63

print("Live detection started. Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # draw
    if results.multi_hand_landmarks:
        for hlm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

    feats = extract63(results)
    buffer.append(feats)

    display_text = "..."
    if len(buffer) == SEQ_LEN:
        inp = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)  # (1, SEQ_LEN, 63)
        probs = model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        pred_hist.append(idx)

        # smoothing: require majority over last SMOOTH_WINDOW and conf threshold
        maj_idx = max(set(pred_hist), key=pred_hist.count)
        if conf >= CONF_THRESH and pred_hist.count(maj_idx) >= (SMOOTH_WINDOW//2 + 1):
            display_text = f"{labels[idx]}  ({conf:.2f})"

    cv2.rectangle(frame, (10, 5), (400, 50), (0, 0, 0), -1)
    cv2.putText(frame, display_text, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Gesture Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
