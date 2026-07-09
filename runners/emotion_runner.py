"""
Standalone Emotion runner. Imports the Emotion Repo's own video.py (unmodified)
for model construction/loading/preprocessing, then runs its own headless
per-frame loop (video.py's own loop is GUI-coupled and has no return value —
see Integration_API.md #3.1) and emits NormalisedFrameCue records.

Correctness fix applied here (see MODEL_ANALYSIS.md #5.1 / Integration_API.md
#2.1): uses model_selection=1 (full-range face detector) instead of the
native code's model_selection=0 (short-range), which was measured to miss
34-100% of frames on real HRI-distance footage.

Run inside .venvs/emotion (torch, torchvision, opencv-python, mediapipe,
pillow — see Integration_API.md #4).

Usage:
    # single clip
    .venvs/emotion/Scripts/python.exe runners/emotion_runner.py --clip <path> --out <out.jsonl>

    # batch mode: loads the model ONCE, loops every clip in clips.csv
    # (subprocess-per-clip is impractical at 1200+ clips -- see run_cue_models.py)
    .venvs/emotion/Scripts/python.exe runners/emotion_runner.py \
        --manifest Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/clips.csv \
        --clips-root Data/Dataset/hri-multimodal-intent-v1.0.0 \
        --out data/measured/emotion_frame_cues.jsonl
"""
import argparse
import os
import sys
import time

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(RUNNERS_DIR), "Emotion Repo"))

from common.schema import NormalisedFrameCue, write_jsonl, append_batch, read_manifest  # noqa: E402
from common.constants import CONFIDENCE_FLOOR  # noqa: E402

import cv2  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
import video as emotion_video  # noqa: E402  (Emotion Repo's own module, unmodified)

CUE = "emotion"
FLOOR = CONFIDENCE_FLOOR[CUE]


def pick_face(detections, w, h):
    """Multi-face policy (undefined in native code): largest bbox by area.
    See Integration_API.md #2.1."""
    best, best_area = None, -1
    for det in detections:
        bbox = det.location_data.relative_bounding_box
        area = bbox.width * bbox.height
        if area > best_area:
            best_area = area
            best = det
    return best


def load_model(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = emotion_video.resolve_weights(emotion_video.DEFAULT_WEIGHTS)
    model = emotion_video.build_model()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    transform = emotion_video.get_transform()
    return model, transform, device


def process_clip(clip_path: str, model, transform, device, mp_face):
    """Pure per-clip logic, no I/O beyond reading the clip itself.
    Reused by both single-clip and batch modes."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")

    records = []
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]
        small = frame
        if w > emotion_video.MAX_FRAME_WIDTH:
            small = cv2.resize(frame, (emotion_video.MAX_FRAME_WIDTH, int(h * emotion_video.MAX_FRAME_WIDTH / w)))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb_small)

        if not results.detections:
            records.append(NormalisedFrameCue(
                cue=CUE, frame_idx=frame_idx, label="Unknown", confidence=0.0,
                probs={}, valid=False, extra={"bbox": None}))
            continue

        det = pick_face(results.detections, w, h)
        bbox = det.location_data.relative_bounding_box
        x, y = max(0, int(bbox.xmin * w)), max(0, int(bbox.ymin * h))
        bw, bh = int(bbox.width * w), int(bbox.height * h)
        face = frame[y:y + bh, x:x + bw]
        if face.size == 0:
            records.append(NormalisedFrameCue(
                cue=CUE, frame_idx=frame_idx, label="Unknown", confidence=0.0,
                probs={}, valid=False, extra={"bbox": None}))
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(face_rgb)
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs_vec = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
        idx = int(probs_vec.argmax())
        conf = float(probs_vec[idx])
        label = emotion_video.EMOTION_LABELS[idx]
        probs = {lbl: float(p) for lbl, p in zip(emotion_video.EMOTION_LABELS, probs_vec)}

        records.append(NormalisedFrameCue(
            cue=CUE, frame_idx=frame_idx, label=label, confidence=conf,
            probs=probs, valid=(conf >= FLOOR),
            extra={"bbox": [x, y, bw, bh]}))

    cap.release()
    return records


def run_single(clip_path: str, out_path: str):
    model, transform, device = load_model()
    mp_face = emotion_video.mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    records = process_clip(clip_path, model, transform, device, mp_face)
    mp_face.close()
    write_jsonl(records, out_path)
    print(f"[emotion_runner] {len(records)} frames -> {out_path}")


def run_batch(manifest_csv: str, clips_root: str, out_path: str, limit=None, resume=False):
    rows = read_manifest(manifest_csv)
    if limit:
        rows = rows[:limit]

    done_ids = set()
    mode = "a"
    if resume and os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(line.split('"clip_id": "', 1)[1].split('"', 1)[0])
                except IndexError:
                    pass
        print(f"[emotion_runner] resuming: {len(done_ids)} clips already done")
    else:
        mode = "w"

    model, transform, device = load_model()
    mp_face = emotion_video.mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    t0 = time.time()
    n_done = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i, row in enumerate(rows):
            clip_id = row["clip_id"]
            if clip_id in done_ids:
                continue
            clip_path = os.path.join(clips_root, row["filepath"])
            try:
                records = process_clip(clip_path, model, transform, device, mp_face)
            except Exception as e:
                print(f"[emotion_runner] ERROR on {clip_id} ({clip_path}): {e}")
                continue
            append_batch(f, clip_id, records)
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(rows) - len(done_ids) - n_done) / rate if rate > 0 else float("inf")
                print(f"[emotion_runner] {i+1}/{len(rows)} clips ({n_done} this run, "
                      f"{rate:.2f} clips/s, ~{remaining/60:.1f} min remaining)")

    mp_face.close()
    print(f"[emotion_runner] batch done: {n_done} clips processed -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="single-clip mode: path to one clip")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--manifest", help="batch mode: path to clips.csv")
    ap.add_argument("--clips-root", help="batch mode: dataset root (filepath column is relative to this)")
    ap.add_argument("--limit", type=int, default=None, help="batch mode: only process first N rows (testing)")
    ap.add_argument("--resume", action="store_true", help="batch mode: skip clip_ids already present in --out")
    args = ap.parse_args()

    if args.manifest:
        if not args.clips_root:
            raise SystemExit("--clips-root is required with --manifest")
        run_batch(args.manifest, args.clips_root, args.out, limit=args.limit, resume=args.resume)
    else:
        if not args.clip:
            raise SystemExit("either --clip or --manifest is required")
        run_single(args.clip, args.out)
