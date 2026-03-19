# Real-Time Hand Gesture Recognition Prototype

A prototype system for **real-time hand gesture recognition** using **MediaPipe** for hand landmark detection and a deep learning model for classification. This framework works on **continuous video gestures** rather than static images, making it a step closer to natural sign language interpretation.

## Overview
Traditional sign language datasets often rely on isolated static images or pre-segmented poses. This project demonstrates a **live, skeleton-based approach** where gestures are recognized dynamically from camera input. The focus is on building a **foundation** for a more comprehensive sign language recognition system.

Currently, the prototype recognizes three example gestures:
- **Hello**
- **See you later**
- **You good?**

These serve as proof-of-concept classes, but the system can be scaled to cover complete sign language vocabularies.

## Features
- **Real-time** gesture detection from webcam feed.
- **Skeleton-based** feature extraction using MediaPipe hand landmarks.
- **Deep learning classifier** built with TensorFlow/Keras.
- **Video-assisted data collection** for verification and debugging.

## How It Works
1. **MediaPipe** detects and tracks hand landmarks in real time.
2. The landmark coordinates are processed and stored as training data.
3. A **deep learning model** learns to classify the gestures based on the sequence of landmark positions.
4. During inference, the webcam feed is processed live, and recognized gestures are displayed instantly.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/gesture-recognition-prototype.git
cd gesture-recognition-prototype
pip install -r requirements.txt
