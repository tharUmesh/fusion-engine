# 🤖 High-Accuracy HRI Gesture Recognition Pipeline

An ultra-robust, real-time hand gesture recognition system designed for **Human-Robot Interaction (HRI)** scenarios, optimized for deployment on resource-constrained platforms like the **NVIDIA Jetson Orin Nano**.

🔗 **GitHub Repository**: [https://github.com/IshanDilhan/hand-gesture-recognition-mediapipe](https://github.com/IshanDilhan/hand-gesture-recognition-mediapipe)

---

## 🌟 Key Features

*   **Jetson Orin Nano Ready**: Highly optimized, quantized TFLite inference takes <0.1 ms.
*   **6 Static Hand Poses**: Model-based detection of Open Palm, Close, Pointer, Thumbs Up, Thumbs Down, and Beckoning.
*   **Landmark Smoothing Filter**: Built-in Exponential Moving Average (EMA) filter ($\alpha = 0.45$) stops coordinates from jittering ("dancing points"), stabilizing both display skeletal lines and model inputs.
*   **Rotation-Invariant Pointing**: Trained using 360-degree rotation data augmentation. Pointing works in all directions (sideways, downwards, diagonally).
*   **EXIF Orientation Metadata Auto-Rotation**: Automatically rotates phone-recorded videos (90°, 180°, 270°) to an upright orientation before MediaPipe processing.
*   **Aspect Ratio Resizing**: Scales high-resolution frames (e.g. 4K) to a maximum dimension of 960px to prevent window clipping and increase execution speed (FPS) significantly.
*   **Flex-Pose Hand Raising**: Identifies "One Hand Raised" or "Arms Up" with open palms, fists, or unknown shapes using dynamic vertical coordinate history tracking (monitoring start and end positions).
*   **Scale-Invariant Waving & Beckoning**: Normalizes coordinate excursions by the hand's own scale, enabling movement triggers (circular, waving, curling) at any distance from the camera.
*   **Menu-Driven Video Playlist Player**: Automatically plays test videos sequentially, loops back, and runs HRI scenario checks.

---

## 🛠️ How It Works & Techniques

The system utilizes a hybrid approach combining computer vision, deep learning, and temporal kinematics:

### 1. Preprocessing & Auto-Orientation
*   **Exif orientation correction**: High-resolution videos filmed on mobile phones are often rotated. The pipeline checks the EXIF orientation tag (`cv.CAP_PROP_ORIENTATION_META` or index 48) and applies appropriate rotation matrices to orient frames vertically before processing.
*   **Aspect-Ratio Scaling**: To ensure low-latency performance and prevent off-screen rendering on Jetson/PC screens, 4K frames are resized to a maximum dimension of 960px using area interpolation (`cv.INTER_AREA`).

### 2. Skeletal Joint Landmark Extraction (MediaPipe)
*   **Joint Detection**: MediaPipe Hands extracts 21 skeletal joints in 3D coordinates.
*   **Translation & Scale Normalization**:
    *   To make the model translation-invariant, the wrist landmark is subtracted from all other coordinates: $\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_{\text{wrist}}$
    *   To make it scale-invariant, the landmarks are divided by the hand scale (e.g., wrist-to-middle finger MCP distance or bounding box diagonals), flattening coordinates into a normalized 1D vector of 42 spatial dimensions ($x, y$).

### 3. Quantized MLP Neural Network Classifier
*   **Architecture**: A lightweight multi-layer perceptron (MLP) trained on a custom mapped subset of the HaGRID dataset.
*   **Rotation Augmentation**: The dataset is augmented by applying random rotation matrices ($0^{\circ}$ to $360^{\circ}$), making pointing gesture classification rotation-invariant.
*   **FP16/INT8 TFLite Quantization**: The model weights are quantized down to a compact size of ~6-8 KB, allowing sub-millisecond inference speeds on edge platforms like the Jetson Orin Nano.

### 4. Exponential Moving Average (EMA) Smoothing
*   To eliminate landmark jitter ("dancing points") caused by camera noise and lighting variations, an EMA filter is applied to the coordinates over time:
    $$\mathbf{x}_{\text{smooth}, t} = \alpha \mathbf{x}_t + (1 - \alpha) \mathbf{x}_{\text{smooth}, t-1}$$
    Where $\alpha = 0.45$. This creates a silky-smooth rendering overlay and improves classifier stability.

### 5. Temporal Path History Tracking for Hand Raising
*   **Flex-Pose detection**: Traditional palm-based checks fail when the user raises their hand as a fist or unknown gesture.
*   **Trajectory analysis**: The pipeline records a rolling history buffer of vertical coordinates ($y\_hist$). It calculates the difference between the starting position and the ending position of the gesture trajectory to verify that a hand-raising motion occurred, supporting Open Palm, Fist, or Unknown shapes.

### 6. Scale-Invariant Motion Metrics (Waving & Beckoning)
*   Dynamic gesture tracking calculates the bounding-box-normalized excursion of hand landmarks. Waving and beckoning are evaluated against this relative hand size metric rather than absolute pixel distances, ensuring stable detection regardless of how close or far the hand is from the camera.

---

## 📊 Mapped HRI Gestures

The system classifies hand landmarks into 6 static categories:

| ID | Pose Name | HaGRID Folders Mapped | HRI Scenario Intent |
|---|---|---|---|
| 0 | **Open Palm** | `train_val_palm`, `train_val_stop` | `raise hand`, `wave` (static), `both hands up` |
| 1 | **Close (Fist)** | `train_val_fist` | Neutral resting hand state / raising |
| 2 | **Pointer** | `train_val_one` | `point` (pointing in all directions) |
| 3 | **Thumbs Up** | `train_val_like` | `thumbs up` (confirm, task success) |
| 4 | **Thumbs Down** | `train_val_dislike` | `thumbs down` (help request, failure) |
| 5 | **Beckoning** | `train_val_call` | `beckoning` (static pose shape) |

---

## 👥 Message for my Friends (Setup & Installation)

Hey there! If you are cloning this repository to run the project, welcome! This project contains a complete list of required packages in the [requirements.txt](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/requirements.txt) file. 

Here is a breakdown of the dependencies we need, what each is used for, and how to set everything up.

### 📦 Package-by-Package Breakdown

1.  **`numpy` (v1.26.4)**
    *   *What it does*: Used for matrix math operations.
    *   *Why we need it*: Calculates coordinate distances, normalizes landmark skeletons, and stores movement history for waving and beckoning.
    *   *Install individually*: `pip install numpy==1.26.4`
2.  **`opencv-python` (v4.9.0.80)**
    *   *What it does*: Main camera and video manipulation framework.
    *   *Why we need it*: Captures webcams, reads video files, resizes 4K images to 960px, auto-rotates phone videos, and draws the visual skeleton overlay.
    *   *Install individually*: `pip install opencv-python==4.9.0.80`
3.  **`mediapipe` (v0.10.11)**
    *   *What it does*: Google's real-time ML tracking library.
    *   *Why we need it*: Finds the hand and returns the 21 3D coordinates of joints.
    *   *Install individually*: `pip install mediapipe==0.10.11`
4.  **`tensorflow` (v2.15.1)**
    *   *What it does*: Machine learning framework.
    *   *Why we need it*: Loads the lightweight quantized model (`keypoint_classifier.tflite`) and runs inference on our hand coordinates in under 1ms.
    *   *Install individually*: `pip install tensorflow==2.15.1`
5.  **`protobuf` (v3.20.3)**
    *   *What it does*: Data serialization library.
    *   *Why we need it*: Internally passes data structures back and forth between MediaPipe's C++ core and our Python code.
    *   *Install individually*: `pip install protobuf==3.20.3`
6.  **`matplotlib` (v3.10.9)**
    *   *What it does*: Graph and visualization generator.
    *   *Why we need it*: Used to plot validation curves and evaluation confusion matrices during the model retraining phase.
    *   *Install individually*: `pip install matplotlib==3.10.9`

---

### 🚀 Getting Started

Follow these steps to clone, set up, and run the pipeline:

#### Step 1: Clone the Repository
```bash
git clone https://github.com/IshanDilhan/hand-gesture-recognition-mediapipe.git
cd hand-gesture-recognition-mediapipe
```

#### Step 2: Create a Virtual Environment
It's recommended to use a clean Python environment (Python 3.10 or 3.11 is best):
```bash
python -m venv env
```
Activate the environment:
*   **Windows**:
    ```powershell
    .\env\Scripts\activate
    ```
*   **Linux/macOS**:
    ```bash
    source env/bin/activate
    ```

#### Step 3: Install the Packages
You can install them **one-by-one** using the individual commands listed in the breakdown above, or install all at once using:
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

### 1. Test video sequences:
We've included dynamic scenario testing. To play sample HRI videos:
```bash
python play_video.py
```
This script will sequentially load testing videos, perform auto-rotation, display the bounding boxes, and verify gesture categories like waving, beckoning, pointing, and hand raising.

### 2. Live Webcam Demo:
To run live classification using your PC's webcam:
```bash
python app.py
```

### 3. Extracting and Retraining (Optional):
*   To extract landmarks from new HaGRID images: `python extract_dataset.py`
*   To train the MLP model: `python train/train_keypoint.py`

---

## 📂 Step-by-Step Documentation (A to Z)

We have created individual guides detailing every step of the pipeline under the `doc/` directory:

1.  [doc/step_1_data_mapping.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_1_data_mapping.md): Details the HaGRID folder-to-intent mappings.
2.  [doc/step_2_extraction.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_2_extraction.md): Explains raw landmark calculation, translation, and scale-normalization.
3.  [doc/step_3_training.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_3_training.md): Covers MLP model training, parameters, and rotation augmentation.
4.  [doc/step_4_quantization.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_4_quantization.md): Explains Keras weights-to-TFLite quantization and optimization benefits.
5.  [doc/step_5_realtime_inference.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_5_realtime_inference.md): Explains Exponential Moving Average (EMA) landmark smoothing and temporal resolutions.
6.  [doc/step_6_jetson_deployment.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_6_jetson_deployment.md): Contains instructions, python environment commands, and tips for Jetson Orin Nano deployment.
7.  [doc/dependencies_guide.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/dependencies_guide.md): Comprehensive package-by-package installation guide.
8.  [doc/model_classifiers_guide.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/model_classifiers_guide.md): Describes the Keypoint and Point History models and their files.
