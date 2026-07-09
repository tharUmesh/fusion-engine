# HRI Gesture Recognition Pipeline — A to Z Guide

This guide explains the entire workflow of the Human-Robot Interaction (HRI) gesture recognition pipeline. The system is designed to run efficiently on resource-constrained hardware such as the **NVIDIA Jetson Orin Nano** by combining static landmark-based neural classifiers (via MediaPipe + TensorFlow Lite) with a temporal motion resolver.

---

## Table of Contents
1. [Pipeline Architecture Overview](#1-pipeline-architecture-overview)
2. [Step A: Data Collection & Preparation](#2-step-a-data-collection--preparation)
3. [Step B: Feature Extraction (`extract_dataset.py`)](#3-step-b-feature-extraction-extract_datasetpy)
4. [Step C: Neural Network Training (`train_keypoint.py`)](#4-step-c-neural-network-training-train_keypointpy)
5. [Step D: Quantization & Deployment (TFLite)](#5-step-d-quantization--deployment-tflite)
6. [Step E: Real-time Inference & Dynamic Resolution](#6-step-e-real-time-inference--dynamic-resolution)
7. [Jetson Orin Nano Deployment Instructions](#7-jetson-orin-nano-deployment-instructions)

---

## 1. Pipeline Architecture Overview

The system uses a two-stage classification hierarchy:
1. **Frame-by-Frame Pose Classifier**: MediaPipe Hands detects the hand skeleton (21 joints, 3D coordinates). The raw coordinates are normalized and fed into a tiny Multi-Layer Perceptron (MLP) model, which predicts one of the **6 static hand poses** (Open Palm, Close, Pointer, Thumbs Up, Thumbs Down, Beckoning).
2. **Temporal Motion Resolver**: A queue tracks the history of coordinates and states over a rolling window (e.g. 16 frames) to detect motion patterns, resolving complex HRI scenario actions like waving or beckoning.

```
[Camera Input]
      │
      ▼
[MediaPipe Hands] (Extracts 21 2D/3D landmarks)
      │
      ▼
[Landmark Preprocessing] (Translation, Flattening, & Scale Normalization)
      │
      ├───────────────────────────────────┐
      ▼                                   ▼
[Static MLP Classifier (TFLite)]    [Point History Tracker] (16 frames)
      │                                   │
      │ (Predicted Pose Index)            │ (Skeletal Tip Trajectory)
      ▼                                   ▼
      └─────────────────┬─────────────────┘
                        ▼
            [Global Scenario Resolver]
                        │
                        ▼
        [Final Gesture / Intent Label] 
```

---

## 2. Step A: Data Collection & Preparation

The model is trained on the **HaGRID (HAnd Gesture Recognition Image Dataset)**. The folders containing the raw images are structured as:
- `train_val_palm` / `train_val_stop`: Open palm gestures (Class `0`).
- `train_val_fist`: Closed hand gestures (Class `1`).
- `train_val_one`: Pointing gesture (Class `2`).
- `train_val_like`: Thumbs up gesture (Class `3`).
- `train_val_dislike`: Thumbs down gesture (Class `4`).
- `train_val_call`: Beckoning/Telephone gesture (Class `5`).

---

## 3. Step B: Feature Extraction (`extract_dataset.py`)

Raw images are processed to extract landmark coordinates and save them into a compact CSV file:
1. Reads each gesture folder.
2. Runs MediaPipe Hands in static image mode.
3. Converts the 21 hand landmarks to relative coordinates by subtracting the wrist coordinates $(x_0, y_0)$ from all points.
4. Flattens the points to a 42-dimensional list: $[x_0, y_0, x_1, y_1, ..., x_{20}, y_{20}]$.
5. Normalizes the coordinates by dividing by the maximum absolute coordinate value, scaling all coordinates to the range $[-1, 1]$ (this ensures scale invariance).
6. Appends the label and preprocessed coordinates to `model/keypoint_classifier/keypoint.csv`.

**Run Command:**
```bash
python extract_dataset.py --max_per_class 1000
```

---

## 4. Step C: Neural Network Training (`train_keypoint.py`)

A feed-forward Multi-Layer Perceptron (MLP) is trained using TensorFlow and Keras:
1. Loads the 42-dimensional inputs and labels from `keypoint.csv`.
2. Splits data into 75% training and 25% validation.
3. The network architecture is designed to prevent overfitting while keeping the parameter count low:
   - **Input Layer**: 42 dimensions
   - **Dropout (20%)**: For regularization
   - **Dense Layer**: 32 units, ReLU activation
   - **Dropout (30%)**: For regularization
   - **Dense Layer**: 16 units, ReLU activation
   - **Output Layer**: 6 units, Softmax activation (predicts probabilities for each class)
4. Trains using the Adam optimizer and Sparse Categorical Crossentropy loss.
5. Employs Early Stopping to halt training once validation loss stabilizes.

**Run Command:**
```bash
python train/train_keypoint.py
```

---

## 5. Step D: Quantization & Deployment (TFLite)

To run efficiently on Jetson Orin Nano, the trained Keras model (`.hdf5`) is converted to a TensorFlow Lite (`.tflite`) model using post-training quantization (`tf.lite.Optimize.DEFAULT`). This compresses the model to a size of **~6-8 KB** and accelerates execution on the Jetson CPU/GPU.

---

## 6. Step E: Real-time Inference & Dynamic Resolution

When running real-time inference (via `app.py` or `test_video.py`):
1. **Frames** are read from a video file or live webcam stream.
2. **MediaPipe Hands** processes the frame and outputs the landmarks.
3. **KeyPointClassifier** loads the TFLite model and classifies the hand shape.
4. **Dynamic Actions** are computed using motion-tracking algorithms:
   - **Waving**: Checks if the horizontal excursion of the hand wrist/index tip exceeds a threshold and undergoes at least 2 reversals in direction.
   - **Beckoning**: Curling index finger patterns coupled with vertical/depth movement toward the body.
5. **Global Scenario Resolver** combines these cues to match the HRI scenarios in the [HRI_Dataset_Table](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/HRI_Dataset_Table..pdf) (e.g. mapping a Thumbs Up or Thumbs Down to task success/failures).

---

## 7. Jetson Orin Nano Deployment Instructions

1. **Prerequisites**: Ensure JetPack is installed on the Jetson Orin Nano.
2. **Dependencies**: Install the required libraries inside the Jetson virtual environment:
   ```bash
   pip install opencv-python numpy mediapipe-silicon tensorflow-lite
   ```
   *(Note: Use `mediapipe-silicon` or specific Jetson builds if standard `mediapipe` isn't available).*
3. **Running the Live Application**:
   Run the webcam detector using:
   ```bash
   python app.py --device 0
   ```
4. **Running Video Testing**:
   Run the seqential video tester:
   ```bash
   python test_video.py --video testVideo/1.mp4
   ```
