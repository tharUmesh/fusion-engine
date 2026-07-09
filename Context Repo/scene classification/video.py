"""
Standalone scene (environment) classification on a video file (or a folder of
videos). Defaults to scanning the repo's videos/ folder.

SELF-CONTAINED: this file needs only the trained weights (.pth) and these pip
packages — nothing else from the project:
    pip install torch torchvision opencv-python pillow numpy

Put the weights file next to this script (or pass --checkpoint), then:
    python video.py                                    # batch the repo videos/ folder
    python video.py --video myclip.mp4
    python video.py --videos-dir ./my_videos            # batch a different folder
    python video.py --video myclip.mp4 --checkpoint best_EfficientNet_B0.pth

Method: frame -> 224x224 + ImageNet normalize -> model -> scene label +
confidence, smoothed over a short rolling window to reduce flicker.
"""
import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms

# ── Configuration (edit these to ship a different model) ──────────────────
SCENE_LABELS = ["classroom", "kitchen"]     # alphabetical == training order
DEFAULT_WEIGHTS = "best_EfficientNet_B0.pth"
IMAGE_SIZE = 224
SMOOTH_WINDOW = 15
CONF_THRESHOLD = 0.5
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# repo_root/modalities/context/scene_classification/inference -> repo_root
DEFAULT_VIDEOS_DIR = str(Path(SCRIPT_DIR).parents[3] / "videos")


def build_model(num_classes=len(SCENE_LABELS)):
    """EfficientNet-B0 with a `num_classes` head. Weights come from the checkpoint."""
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
    out_path = os.path.join(out_root, os.path.splitext(rel)[0] + "_scene.mp4")
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


def process_video(video_path, out_path, model, transform, device, show):
    """Process one video. Returns: "done" | "next" | "prev" | "quit"."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] Cannot open: {video_path}")
        return "next"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prob_history = deque(maxlen=SMOOTH_WINDOW)
    action, paused, display_frame = "done", False, None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
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
            writer.write(frame)
            display_frame = frame.copy()

        if show and display_frame is not None:
            view = display_frame.copy()
            _draw_overlay(view, paused)
            cv2.imshow("Scene Classification", view)
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
                     help=f"Folder to scan recursively (default: repo videos/ = {DEFAULT_VIDEOS_DIR}).")
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
    print(f" labels:  {SCENE_LABELS}")
    print("=" * 60)

    out_root = os.path.abspath(args.out_dir)

    # ── Single-file mode ─────────────────────────────────────────────────
    if args.video is not None:
        os.makedirs(out_root, exist_ok=True)
        out_path = args.output or os.path.join(
            out_root, os.path.splitext(os.path.basename(args.video))[0] + "_scene.mp4")
        print(f"Writing to {out_path}")
        process_video(args.video, out_path, model, transform, device, show=not args.no_show)
        cv2.destroyAllWindows()
        print(f"Saved: {out_path}")
        return

    # ── Batch mode (default: repo videos/ folder) ───────────────────────
    videos_dir = os.path.abspath(args.videos_dir or DEFAULT_VIDEOS_DIR)
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
        action = process_video(video_path, out_path, model, transform, device, show=not args.no_show)
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
