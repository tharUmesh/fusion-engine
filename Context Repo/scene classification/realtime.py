"""
Standalone real-time scene (environment) classification from a camera.
Uses an Intel RealSense camera if connected, else the default laptop webcam.

SELF-CONTAINED: this file needs only the trained weights (.pth) and these pip
packages — nothing else from the project:
    pip install torch torchvision opencv-python numpy
    # pyrealsense2 is optional (only for a RealSense camera)

Put the weights file next to this script (or pass --checkpoint), then:
    python realtime.py
    python realtime.py --checkpoint best_EfficientNet_B0.pth

Method: camera frame -> 224x224 + ImageNet normalize -> model -> scene label +
confidence, smoothed over a short rolling window.
"""
import argparse
import os
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

# ── Configuration (edit these to ship a different model) ──────────────────
SCENE_LABELS = ["classroom", "kitchen"]
DEFAULT_WEIGHTS = "best_EfficientNet_B0.pth"
IMAGE_SIZE = 224
SMOOTH_WINDOW = 15
CONF_THRESHOLD = 0.5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model(num_classes=len(SCENE_LABELS)):
    model = tvm.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def resolve_weights(name):
    candidates = [
        name,
        os.path.join(SCRIPT_DIR, name),
        os.path.join(SCRIPT_DIR, "checkpoints", name),
        os.path.join(SCRIPT_DIR, "..", "checkpoints", name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise SystemExit(
        f"Weights file '{name}' not found. Put it next to this script or pass "
        f"--checkpoint <path>.\nLooked in:\n  " + "\n  ".join(candidates))


def _try_start_realsense():
    if not _RS_AVAILABLE:
        return None, None
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(cfg)
        return pipeline, rs.align(rs.stream.color)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_WEIGHTS, help="Path to the .pth weights.")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (fallback).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_weights(args.checkpoint)
    model = build_model()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    transform = get_transform()
    print("=" * 60)
    print(f" weights: {ckpt}")
    print(f" device:  {device}")
    print(f" labels:  {SCENE_LABELS}")
    print("=" * 60)

    pipeline, align = _try_start_realsense()
    if pipeline is not None:
        print("RealSense camera connected.")
        window_title, cap = "Scene Classification (RealSense)", None
    else:
        print(f"RealSense not available — using webcam (index {args.camera}).")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Error: no camera found.")
            return
        window_title = "Scene Classification (Webcam)"

    prob_history = deque(maxlen=SMOOTH_WINDOW)
    try:
        while True:
            if pipeline is not None:
                frames = align.process(pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from webcam.")
                    break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
            prob_history.append(probs)
            avg = np.mean(prob_history, axis=0)
            idx = int(avg.argmax())
            conf = float(avg[idx])
            label = SCENE_LABELS[idx] if conf >= CONF_THRESHOLD else "uncertain"
            color = (0, 255, 0) if label != "uncertain" else (0, 165, 255)
            cv2.putText(frame, f"{label}: {conf * 100:.1f}%", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if pipeline is not None:
            pipeline.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
