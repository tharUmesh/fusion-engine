"""
Standalone emotion recognition on a video file (or a folder of videos).

SELF-CONTAINED: this file needs only the trained weights (.pth) and these pip
packages — nothing else from the project:
    pip install torch torchvision opencv-python mediapipe pillow

Put the weights file next to this script (or pass --checkpoint), then:
    python video.py --video myclip.mp4
    python video.py --videos-dir ./my_videos          # batch a folder
    python video.py --video myclip.mp4 --checkpoint best_MobileNetV2.pth

Method (the plain pipeline that matches how the model was trained on RAF-DB):
    frame -> MediaPipe close-range face detection -> tight crop ->
    224x224 + ImageNet normalize -> model -> emotion label + confidence.
"""
import argparse
import os

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from torchvision import transforms

# ── Configuration (edit these to ship a different model) ──────────────────
EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
DEFAULT_WEIGHTS = "best_MobileNetV2.pth"   # filename of the trained weights
IMAGE_SIZE = 224
MAX_FRAME_WIDTH = 640                       # downscale wide frames before detection
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
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
    """Find the weights file next to the script, in ./checkpoints, or as given."""
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


def collect_videos(videos_dir):
    found = []
    for dirpath, _, filenames in os.walk(videos_dir):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def build_output_path(video_path, videos_dir, out_root):
    rel = os.path.relpath(video_path, videos_dir)
    out_path = os.path.join(out_root, os.path.splitext(rel)[0] + "_emotion.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


_HINT = "[SPACE] Pause   [N] Next   [P] Prev   [Q] Quit"


def _draw_overlay(frame, paused):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, _HINT, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220), 1, cv2.LINE_AA)
    if paused:
        text, font, scale, thick = "PAUSED", cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cx, cy = (w - tw) // 2, (h + th) // 2
        cv2.putText(frame, text, (cx + 2, cy + 2), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (cx, cy), font, scale, (0, 220, 255), thick, cv2.LINE_AA)


def process_video(video_path, out_path, model, transform, face_detection, device, show):
    """Process one video. Returns: "done" | "next" | "prev" | "quit"."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] Cannot open: {video_path}")
        return "next"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    action, paused, display_frame = "done", False, None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            small = (cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * MAX_FRAME_WIDTH / w)))
                     if w > MAX_FRAME_WIDTH else frame)
            results = face_detection.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if results.detections:
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
                    lbl = f"{EMOTION_LABELS[pred.item()]}: {conf.item() * 100:.1f}%"
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(frame, lbl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
            writer.write(frame)
            display_frame = frame.copy()

        if show and display_frame is not None:
            view = display_frame.copy()
            _draw_overlay(view, paused)
            cv2.imshow("Emotion Recognition", view)
            key = cv2.waitKey(100 if paused else 30) & 0xFF
            if key == ord("q"):
                action = "quit"; break
            elif key == ord("n"):
                action = "next"; break
            elif key == ord("p"):
                action = "prev"; break
            elif key == ord(" "):
                paused = not paused

    cap.release()
    writer.release()
    return action


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video", default=None, help="Path to a single input video file.")
    src.add_argument("--videos-dir", default=None,
                     help="Folder to scan recursively (default: current directory).")
    ap.add_argument("--checkpoint", default=DEFAULT_WEIGHTS, help="Path to the .pth weights.")
    ap.add_argument("--output", default=None, help="Output path (single-file mode only).")
    ap.add_argument("--out-dir", default="outputs", help="Output root for batch mode.")
    ap.add_argument("--no-show", action="store_true", help="Do not open a preview window.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip videos whose output file already exists.")
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
    out_root = os.path.abspath(args.out_dir)

    # ── Single-file mode ─────────────────────────────────────────────────
    if args.video is not None:
        os.makedirs(out_root, exist_ok=True)
        out_path = args.output or os.path.join(
            out_root, os.path.splitext(os.path.basename(args.video))[0] + "_emotion.mp4")
        print(f"Writing to {out_path}")
        process_video(args.video, out_path, model, transform, face_detection, device,
                      show=not args.no_show)
        cv2.destroyAllWindows()
        print(f"Saved: {out_path}")
        return

    # ── Batch mode ───────────────────────────────────────────────────────
    videos_dir = os.path.abspath(args.videos_dir or ".")
    videos = collect_videos(videos_dir)
    if not videos:
        print(f"No video files found under: {videos_dir}")
        return

    print(f"Found {len(videos)} video(s) under {videos_dir}")
    done = skipped = idx = 0
    while idx < len(videos):
        video_path = videos[idx]
        out_path = build_output_path(video_path, videos_dir, out_root)
        label = os.path.relpath(video_path, videos_dir)
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{idx+1}/{len(videos)}] skip (exists): {label}")
            skipped += 1; idx += 1
            continue
        print(f"[{idx+1}/{len(videos)}] {label}")
        action = process_video(video_path, out_path, model, transform, face_detection, device,
                               show=not args.no_show)
        cv2.destroyAllWindows()
        if action == "quit":
            print("  Quit by user."); break
        elif action == "prev":
            idx = max(0, idx - 1)
        else:
            done += 1; idx += 1

    cv2.destroyAllWindows()
    print(f"\nDone. {done} processed, {skipped} skipped.")


if __name__ == "__main__":
    main()
