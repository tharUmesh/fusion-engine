# HRI Motion Classifier — Inference Package

Self-contained package for running the HRI motion-classification model in
another project. Contains the best-performing checkpoint (fine-tuned on
real-world MediaPipe data) and everything needed to run inference —
no training code, no datasets.

## Contents

| File | Purpose |
|---|---|
| `checkpoints/best_model_finetuned.pt` | Model weights (~5.7 MB). LSTM + attention, ~1.8 M params. |
| `model.py` | `MotionLSTM` architecture — required to load the checkpoint. |
| `inference.py` | `MotionInference` — stateful engine: preprocessing, sliding window, prediction. |
| `skeleton_utils.py` | Converts MediaPipe Pose world landmarks → the (25, 3) NTU joint array the model expects. **Mandatory** if your skeletons come from MediaPipe. |
| `example_webcam.py` | Minimal runnable end-to-end example (webcam → prediction). |
| `requirements.txt` | Dependencies. Only `torch` + `numpy` for inference itself; `mediapipe` + `opencv-python` for skeleton extraction. |

## Setup

```bash
pip install -r requirements.txt
python example_webcam.py      # quick smoke test with a webcam
```

## Usage in your own pipeline

```python
import numpy as np
from inference import MotionInference
from skeleton_utils import mediapipe_to_ntu25

engine = MotionInference("checkpoints/best_model_finetuned.pt")  # device auto: CUDA if available

# In your per-frame loop (~30 fps):
joints_25 = mediapipe_to_ntu25(results.pose_world_landmarks.landmark)
result = engine.update(joints_25)

if result.label != "buffering":
    print(result.label, result.confidence)   # e.g. "walking" 0.91
    print(result.probs)                      # full 4-class distribution
    # result.to_dict() gives a JSON-friendly dict
```

Call `engine.reset()` whenever the person leaves the frame or the scene
changes — otherwise stale frames pollute the sliding window.

## Input contract (must be respected exactly)

- **Input per frame:** `(25, 3)` float32 array — NTU-layout 3D joints in
  metres. If your source is MediaPipe Pose, `mediapipe_to_ntu25()`
  produces exactly this (index remapping + X/Y axis flip + spine-joint
  approximation). If you feed raw MediaPipe coordinates without this
  conversion, the model will silently produce garbage.
- **Frame rate:** ~30 fps. The model classifies a sliding window of 30
  frames (≈1 s). Feeding e.g. 15 fps stretches the effective window to
  2 s and degrades accuracy.
- **Warm-up:** the first 29 frames return `label="buffering"`
  (`label_idx=-1`, confidence 0). Predictions start on frame 30 and then
  update every frame.
- **No person detected:** pass `np.zeros((25, 3), dtype=np.float32)` for
  brief gaps; call `engine.reset()` for longer absences.
- All preprocessing (hip-centering, shoulder-width normalisation,
  velocity features, windowing) happens inside `MotionInference` — do
  not normalise the joints yourself.

## Output classes

| index | label |
|---|---|
| 0 | sitting |
| 1 | standing |
| 2 | walking |
| 3 | stepping_back |

## Model provenance & performance

- Architecture: LSTM (hidden 256, 3 layers) with temporal attention
  pooling over a 30-frame window of 84-D features (14 joints × xyz
  position + velocity).
- Trained on NTU RGB+D–derived skeletons (96.7% val accuracy on NTU),
  then **fine-tuned on real-world MediaPipe-extracted HRI data** to close
  the Kinect→MediaPipe domain gap. This fine-tuned checkpoint is the
  best-performing model for live MediaPipe input — do not substitute the
  NTU-only checkpoint.
- Inference cost: a few ms per window on CPU; designed for Jetson-class
  edge deployment.

## Notes

- `model.py` and `inference.py` must stay in the same directory
  (`inference.py` does `from model import MotionLSTM`). If you vendor
  them into a package, adjust that import.
- The checkpoint is loaded with `weights_only=True` (safe loading);
  requires PyTorch ≥ 2.0.
- Keep `mediapipe==0.10.14` — other versions have produced incompatible
  landmark output.
