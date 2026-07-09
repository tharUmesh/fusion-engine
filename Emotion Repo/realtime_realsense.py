"""
Standalone real-time emotion recognition from a camera.
Uses an Intel RealSense camera if connected, else the default laptop webcam.

SELF-CONTAINED: this file needs only the trained weights (.pth) and these pip
packages — nothing else from the project:
    pip install torch torchvision opencv-python mediapipe pillow numpy
    # pyrealsense2 is optional (only for a RealSense camera)

Put the weights file next to this script (or pass --checkpoint), then:
    python realtime_realsense.py
    python realtime_realsense.py --checkpoint best_MobileNetV2.pth

Method: camera frame -> MediaPipe close-range face detection -> tight crop ->
224x224 + ImageNet normalize -> model -> emotion label + confidence.
"""
import argparse
import os

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from torchvision import transforms

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

# ── Configuration (edit these to ship a different model) ──────────────────
EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
DEFAULT_WEIGHTS = "best_MobileNetV2.pth"
IMAGE_SIZE = 224
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model(num_classes=len(EMOTION_LABELS)):
    """MobileNetV2 with a `num_classes` head. Weights come from the checkpoint."""
    model = tvm.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_transform():
    return transforms.Compose([
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
    print(f" labels:  {EMOTION_LABELS}")
    print("=" * 60)

    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)

    pipeline, align = _try_start_realsense()
    if pipeline is not None:
        print("RealSense camera connected.")
        window_title, cap = "Emotion Recognition (RealSense)", None
    else:
        print(f"RealSense not available — using webcam (index {args.camera}).")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Error: no camera found.")
            return
        window_title = "Emotion Recognition (Webcam)"

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

            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                h, w, _ = frame.shape
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    x, y = max(0, int(box.xmin * w)), max(0, int(box.ymin * h))
                    bw, bh = int(box.width * w), int(box.height * h)
                    face = frame[y:y + bh, x:x + bw]
                    if face.size == 0:
                        continue
                    pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    tensor = transform(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        conf, pred = torch.max(F.softmax(model(tensor), dim=1), 1)
                    label = f"{EMOTION_LABELS[pred.item()]}: {conf.item() * 100:.1f}%"
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

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
