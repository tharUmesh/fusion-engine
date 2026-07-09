# Project Dependencies Guide

This document explains the purpose of each package listed in [requirements.txt](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/requirements.txt) and why it is required to run the HRI Gesture Recognition pipeline.

---

## 📦 Package-by-Package Breakdown

### 1. `numpy` (v1.26.4)
*   **What it does**: A library for scientific computing and multi-dimensional array operations.
*   **Why it's needed**:
    *   Used to perform matrix calculations, distance metrics, and vector geometry on landmark coordinates.
    *   Tracks horizontal and vertical coordinate histories (waving and raising movements) inside the dynamic motion filters.
    *   Converts training keypoints into arrays for the MLP model.

### 2. `opencv-python` (v4.9.0.80)
*   **What it does**: Open Source Computer Vision library used for image and video processing.
*   **Why it's needed**:
    *   Loads and streams raw video frames from both files and live webcams.
    *   Handles image preprocessing, including horizontal mirroring, EXIF metadata-based auto-rotation, and aspect ratio-preserving resizing.
    *   Draws visual overlays on the output window, including bounding boxes, FPS markers, controls, and scenario text.
    *   Powers the interactive video player GUI (`cv.imshow`, `cv.waitKey`).

### 3. `mediapipe` (v0.10.11)
*   **What it does**: Google's open-source framework for cross-platform body tracking and ML pipelines.
*   **Why it's needed**:
    *   Runs the core hand detection and joint extraction model.
    *   Locates and outputs the spatial coordinates of **21 skeletal hand landmarks** per hand in real-time.
    *   Provides standard skeleton drawing specification templates (`HAND_CONNECTIONS`).

### 4. `tensorflow` (v2.15.1)
*   **What it does**: Google's end-to-end open-source machine learning platform.
*   **Why it's needed**:
    *   Runs inference on the quantized static MLP classifier (`keypoint_classifier.tflite`) with sub-millisecond execution times.
    *   Provides the training backend (Keras API) used to build and train the MLP gesture classifier.

### 5. `protobuf` (v3.20.3)
*   **What it does**: Google's Protocol Buffers, a language-neutral serialization mechanism.
*   **Why it's needed**:
    *   Used internally by MediaPipe and TensorFlow to serialize and deserialize structured data packets (like landmark coordinate objects) passed between the native C++ backend and the Python scripts.

### 6. `matplotlib` (v3.10.9)
*   **What it does**: A comprehensive library for creating static, animated, and interactive visualizations in Python.
*   **Why it's needed**:
    *   Used inside training notebooks (like `keypoint_classification.ipynb`) to plot training history, loss curves, validation accuracy, and print confusion matrices to evaluate model performance.
