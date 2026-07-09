# Step 6: Jetson Orin Nano Deployment

This step provides instructions on deploying and running the finalized gesture recognition pipeline on the NVIDIA Jetson Orin Nano.

---

## 🛠️ Prerequisites

1.  **JetPack OS**: Ensure your Jetson Orin Nano is running JetPack 5.x or JetPack 6.x.
2.  **USB Webcam**: Connect a compatible USB webcam or CSI camera.

---

## 📦 Dependency Installation

We recommend creating an isolated virtual environment on the Jetson to avoid dependency conflicts with other system-level packages:

```bash
# 1. Create a virtual environment
python3 -m venv jetson_env

# 2. Activate the virtual environment
source jetson_env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install optimized dependencies
pip install numpy opencv-python

# 5. Install MediaPipe silicon/Jetson wheels
# Standard mediapipe packages do not support Jetson out-of-the-box.
# Install community-built aarch64 wheels (or install mediapipe-silicon):
pip install mediapipe-silicon
```

*(Note: If `mediapipe-silicon` is not available, install a precompiled wheel corresponding to your Python and JetPack versions).*

---

## 🏃 Running the Application on Jetson

Once setup is complete, run the core scripts using:

### 1. Webcam Real-time Detection
```bash
python app.py --device 0
```
*   **Performance Tip**: If the camera framerate is low, check CPU governors and set performance mode:
    ```bash
    sudo nvpmodel -m 0
    sudo jetson_clocks
    ```

### 2. Video Test Suite
Run the playlist selector to analyze the sequence video files:
```bash
python play_video.py
```
