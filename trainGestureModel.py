# train_gesture_model.py
import os, glob, json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "collected_data"
OUTPUT_MODEL = "gesture_model.keras"
OUTPUT_LABELS = "labels.txt"
OUTPUT_CONFIG = "config.npy"   # stores sequence_length etc.

# -------- Load data --------
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
label_to_idx = {lab: i for i, lab in enumerate(labels)}

X, y, lengths = [], [], []
for lab in labels:
    for fp in glob.glob(os.path.join(DATA_DIR, lab, "*.npy")):
        arr = np.load(fp)  # shape: (frames, 63)
        if arr.ndim != 2 or arr.shape[1] != 63:
            continue
        X.append(arr.astype(np.float32))
        y.append(label_to_idx[lab])
        lengths.append(arr.shape[0])

if not X:
    raise RuntimeError("No .npy samples found. Check collected_data/*/*.npy")

X = np.array(X, dtype=object)  # ragged
y = np.array(y, dtype=np.int64)

# Decide sequence length (default 60, or use min(95th percentile, 120))
default_len = 60
p95 = int(np.percentile(lengths, 95))
SEQ_LEN = max(30, min(120, p95 if p95 > 0 else default_len))
if SEQ_LEN < 45:  # your recordings were ~2s @ 30fps => ~60; keep reasonable
    SEQ_LEN = default_len

# -------- Pad / truncate --------
def pad_trunc(seq, target_len):
    if len(seq) == target_len:
        return seq
    if len(seq) > target_len:
        return seq[:target_len]
    pad = np.zeros((target_len - len(seq), seq.shape[1]), dtype=np.float32)
    return np.vstack([seq, pad])

X_pad = np.stack([pad_trunc(s, SEQ_LEN) for s in X], axis=0)  # (N, SEQ_LEN, 63)

# -------- Split --------
X_train, X_val, y_train, y_val = train_test_split(
    X_pad, y, test_size=0.2, stratify=y, random_state=42
)

# -------- Build model --------
num_classes = len(labels)
model = models.Sequential([
    layers.Masking(mask_value=0.0, input_shape=(SEQ_LEN, 63)),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------- Train --------
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, verbose=1),
]
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# -------- Save --------
model.save(OUTPUT_MODEL)
with open(OUTPUT_LABELS, "w") as f:
    f.write("\n".join(labels))
np.save(OUTPUT_CONFIG, {"SEQ_LEN": SEQ_LEN})
print(f"Saved model to {OUTPUT_MODEL}, labels to {OUTPUT_LABELS}, config to {OUTPUT_CONFIG}")
