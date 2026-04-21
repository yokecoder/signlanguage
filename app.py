"""
ISL Sign Language Recognition - Web Server  (v4 - high FPS)

Architecture for minimum latency
─────────────────────────────────
Browser  ──frame──►  handle_frame()  (SocketIO thread)
                          │ puts frame in latest_frame slot (non-blocking)
                          │ immediately returns ACK so browser sends next frame
                          ▼
                    inference_worker()  (dedicated background thread)
                          │ grabs latest_frame, drops stale ones
                          │ runs MediaPipe → Keras
                          ▼
                    pushes result via socketio.emit()

Key improvements over v3
─────────────────────────
• Dedicated inference thread — SocketIO threads never block on ML
• "Latest frame wins" slot — stale frames are discarded, not queued
• static_image_mode=False on a single thread — no lock, uses tracking
  (faster per frame because MediaPipe reuses previous detections)
• Keras warmup at startup — no cold-start on first frame
• Frame downscaled to 320×240 for MediaPipe (sufficient for hand detection)
• Browser sends at requestAnimationFrame speed (~60fps attempted),
  server always processes the most recent one
"""

import os, base64, threading, time
import numpy as np
import cv2
from collections import deque

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import mediapipe as mp

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "gesture_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
CONFIG_PATH = os.path.join(BASE_DIR, "config.npy")

# ── Config ─────────────────────────────────────────────────────────────────
CONF_THRESH   = 0.75   # slightly lower for faster feel
SMOOTH_WINDOW = 3      # reduced from 5 — snappier sign transitions
MP_WIDTH      = 320    # MediaPipe input resolution (enough for hand detection)
MP_HEIGHT     = 240

# ── Load Keras model & metadata ────────────────────────────────────────────
print("Loading Keras model…")
keras_model = load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH) as f:
    labels = [l.strip() for l in f.readlines() if l.strip()]

cfg     = np.load(CONFIG_PATH, allow_pickle=True).item()
SEQ_LEN = cfg.get("SEQ_LEN", 60)

# Warmup — eliminates TF cold-start on the first real frame
print("Warming up Keras model…")
_dummy = np.zeros((1, SEQ_LEN, 63), dtype=np.float32)
for _ in range(3):
    keras_model(_dummy, training=False)
print(f"Ready. Labels: {labels}  SEQ_LEN={SEQ_LEN}")

# ── MediaPipe (single dedicated thread — no lock needed) ───────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def _build_hands():
    return mp_hands.Hands(
        static_image_mode=False,       # tracking mode — reuses prev detections
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,            # fastest model (0=lite, 1=full)
    )

# ── Shared state between SocketIO threads and inference thread ─────────────
_latest_frame  = None          # only the most recent decoded frame
_latest_sid    = None          # which client sent it
_frame_lock    = threading.Lock()
_frame_event   = threading.Event()  # signals inference thread new frame ready

_latest_result = {}            # inference thread writes, SocketIO reads
_result_lock   = threading.Lock()

# ── Per-session ML state ───────────────────────────────────────────────────
buffer    = deque(maxlen=SEQ_LEN)
pred_hist = deque(maxlen=SMOOTH_WINDOW)

# ── FPS tracking ───────────────────────────────────────────────────────────
_inf_frames = 0
_inf_fps    = 0
_inf_last   = time.time()

# ── Flask / SocketIO app ───────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "isl_secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    ping_timeout=10, ping_interval=5)


# ── Landmark extraction ────────────────────────────────────────────────────
def extract_landmarks(results):
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        out = []
        for p in lm:
            out.extend([p.x, p.y, p.z])
        if len(out) == 63:
            return out
    return [0.0] * 63


class _FallbackResult:
    multi_hand_landmarks = None
    multi_handedness     = None


# ── Inference worker thread ────────────────────────────────────────────────
def inference_worker():
    global buffer, pred_hist, _inf_frames, _inf_fps, _inf_last

    hands = _build_hands()

    while True:
        # Block until a new frame arrives
        _frame_event.wait()
        _frame_event.clear()

        # Grab the latest frame (drop any that piled up)
        with _frame_lock:
            frame = _latest_frame
            sid   = _latest_sid

        if frame is None or sid is None:
            continue

        # ── Downscale for MediaPipe only ───────────────────────────────────
        small = cv2.resize(frame, (MP_WIDTH, MP_HEIGHT),
                           interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        # ── MediaPipe ─────────────────────────────────────────────────────
        try:
            results = hands.process(rgb)
        except Exception as exc:
            err = str(exc)
            if "timestamp" in err.lower() or "Graph has errors" in err:
                try:    hands.close()
                except: pass
                hands   = _build_hands()
                results = _FallbackResult()
            else:
                results = _FallbackResult()

        # ── Landmark data for browser drawing ─────────────────────────────
        hand_detected = bool(results.multi_hand_landmarks)
        lm_data = []
        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                lm_data.append([[p.x, p.y] for p in hlm.landmark])

        # ── Feature extraction + buffer ────────────────────────────────────
        feats = extract_landmarks(results)
        buffer.append(feats)

        # ── Keras inference ────────────────────────────────────────────────
        prediction = None
        confidence = 0.0
        all_probs  = {}

        if len(buffer) == SEQ_LEN:
            inp   = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
            probs = keras_model(inp, training=False).numpy()[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])
            pred_hist.append(idx)
            all_probs = {labels[i]: float(probs[i]) for i in range(len(labels))}

            maj_idx = max(set(pred_hist), key=pred_hist.count)
            if conf >= CONF_THRESH and pred_hist.count(maj_idx) >= (SMOOTH_WINDOW // 2 + 1):
                prediction = labels[idx]
                confidence = conf

        # ── FPS tracking ───────────────────────────────────────────────────
        _inf_frames += 1
        now = time.time()
        if now - _inf_last >= 1.0:
            _inf_fps   = _inf_frames
            _inf_frames = 0
            _inf_last  = now

        # ── Push result to client ──────────────────────────────────────────
        payload = {
            "landmarks":     lm_data,
            "prediction":    prediction,
            "confidence":    round(confidence * 100, 1),
            "all_probs":     all_probs,
            "hand_detected": hand_detected,
            "buffer_fill":   int((len(buffer) / SEQ_LEN) * 100),
            "server_fps":    _inf_fps,
        }
        socketio.emit("result", payload, to=sid)


# Start inference thread as daemon
_inf_thread = threading.Thread(target=inference_worker, daemon=True)
_inf_thread.start()


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── SocketIO frame handler — MUST be non-blocking ──────────────────────────
@socketio.on("frame")
def handle_frame(data):
    global _latest_frame, _latest_sid

    try:
        img_bytes = base64.b64decode(data.split(",")[1])
    except Exception:
        return

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    # Overwrite latest frame slot (inference thread takes the newest one)
    with _frame_lock:
        _latest_frame = frame
        _latest_sid   = request.sid  # noqa — imported below

    _frame_event.set()   # wake inference thread


# Need request for sid
from flask import request   # noqa: E402 (import after app creation is fine)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🤙  ISL Sign Language Recognition — Web Server (high-FPS mode)")
    print(f"    Labels ({len(labels)}): {', '.join(labels)}")
    print(f"    SEQ_LEN={SEQ_LEN}  CONF_THRESH={CONF_THRESH}  SMOOTH_WINDOW={SMOOTH_WINDOW}")
    print(f"\n    Open → http://localhost:5000\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)