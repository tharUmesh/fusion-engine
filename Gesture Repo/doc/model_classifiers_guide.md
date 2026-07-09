# Gesture Classifiers & Model Files Guide

This document describes the two neural network models in the `model/` folder, their component files, and how they work.

---

## 📂 1. Keypoint Classifier (`model/keypoint_classifier/`)

The **Keypoint Classifier** is a static hand pose classifier. It determines the hand shape (e.g. Pointer, Open Palm, Thumbs Up) based on a single frame's 21 hand joints.

### 📄 Included Files:
*   **`keypoint_classifier.tflite`**: The quantized TensorFlow Lite model. It is optimized for edge-device CPUs (such as the Jetson Orin Nano) with a footprint of **~6-8 KB**. It runs inference in **<0.1 ms**.
*   **`keypoint_classifier.hdf5`**: The legacy Keras HDF5 model containing the raw float32 weights, saved after training.
*   **`keypoint_classifier_label.csv`**: The 6 target gesture labels:
    1.  `Open Palm`
    2.  `Close`
    3.  `Pointer`
    4.  `Thumbs Up`
    5.  `Thumbs Down`
    6.  `Beckoning`
*   **`keypoint.csv`**: The extracted dataset containing 7,000 normalized landmark coordinates (1,000 per target category).

---

## 📂 2. Point History Classifier (`model/point_history_classifier/`)

The **Point History Classifier** is a temporal motion classifier. It tracks the movement of the hand's index finger tip over a sliding window of **16 frames** to detect dynamic gestures (like circular motions or waving).

### 📄 Included Files:
*   **`point_history_classifier.tflite`**: The quantized TensorFlow Lite model. It processes a 32-dimensional input vector (16 frames $\times$ 2 coordinates) and outputs class probabilities.
*   **`point_history_classifier.hdf5`**: Keras weights.
*   **`point_history_classifier_label.csv`**: The dynamic motion labels:
    1.  `Stop`
    2.  `Clockwise`
    3.  `Counter Clockwise`
    4.  `Move`
    5.  `Wave`
    6.  `Come Here`
*   **`point_history.csv`**: The trajectory coordinates dataset used to train the temporal classifier.

---

## 🧠 Model Architectures

### Keypoint Classifier (MLP)
A feed-forward Multi-Layer Perceptron (MLP) mapping 42 inputs (21 landmarks $\times$ 2 coordinates) to 6 output classes:
```
[Input: 42] -> [Dropout 20%] -> [Dense 64 (ReLU)] -> [Dropout 30%] -> [Dense 32 (ReLU)] -> [Dropout 30%] -> [Dense 16 (ReLU)] -> [Output: 6 (Softmax)]
```

### Point History Classifier (MLP)
An MLP mapping a history vector of 32 inputs (16 frames $\times$ 2 relative coordinates) to 6 dynamic classes:
```
[Input: 32] -> [Dropout 20%] -> [Dense 24 (ReLU)] -> [Dropout 50%] -> [Dense 10 (ReLU)] -> [Output: 6 (Softmax)]
```
*(Note: Because dynamic gestures can also be resolved using horizontal/vertical variance metrics, we combine this neural model with deterministic movement rules for maximum classification robustness).*
