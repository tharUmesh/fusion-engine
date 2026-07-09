"""
Standalone Context runner. Imports the Context Repo's own (rewritten,
self-contained) video.py for model construction/loading/preprocessing, then
runs its own headless per-frame loop and emits NormalisedFrameCue records.

Correctness fixes applied here (see Integration_API.md #2.4):
  - native "uncertain" label -> canonical "Unknown"
  - activity/engaged/n_objects are structurally absent from this model
    (no object detection, activity recognition, or engagement logic exists
    anywhere in the repo) -> hardcoded documented placeholders, not fabricated
    values.

Run inside .venvs/context (torch, torchvision, opencv-python, pillow, numpy
-- see Integration_API.md #4).

Usage:
    # single clip
    .venvs/context/Scripts/python.exe runners/context_runner.py --clip <path> --out <out.jsonl>

    # batch mode: loads the model ONCE, loops every clip in clips.csv
    .venvs/context/Scripts/python.exe runners/context_runner.py \
        --manifest Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/clips.csv \
        --clips-root Data/Dataset/hri-multimodal-intent-v1.0.0 \
        --out data/measured/context_frame_cues.jsonl
"""
import argparse
import os
import sys
import time
from collections import deque

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(RUNNERS_DIR), "Context Repo", "scene classification"))

from common.schema import NormalisedFrameCue, write_jsonl, append_batch, read_manifest  # noqa: E402
from common.constants import CONFIDENCE_FLOOR  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import video as ctx_video  # noqa: E402  (Context Repo's own module, unmodified)

CUE = "context"
FLOOR = CONFIDENCE_FLOOR[CUE]

# Structurally absent from this model -- see Integration_API.md #2.4.
NOT_MEASURED_EXTRA = {"activity": None, "engaged": None, "n_objects": 0}


def load_model(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ctx_video.resolve_weights(ctx_video.DEFAULT_WEIGHTS)
    model = ctx_video.build_model()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    transform = ctx_video.get_transform()
    return model, transform, device


def process_clip(clip_path: str, model, transform, device):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")

    prob_history = deque(maxlen=ctx_video.SMOOTH_WINDOW)
    records = []
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            probs_vec = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        prob_history.append(probs_vec)
        avg = np.mean(prob_history, axis=0)
        idx = int(avg.argmax())
        conf = float(avg[idx])

        native_label = ctx_video.SCENE_LABELS[idx] if conf >= ctx_video.CONF_THRESHOLD else "uncertain"
        label = "Unknown" if native_label == "uncertain" else native_label
        probs = {lbl: float(p) for lbl, p in zip(ctx_video.SCENE_LABELS, avg)}

        records.append(NormalisedFrameCue(
            cue=CUE, frame_idx=frame_idx, label=label, confidence=conf,
            probs=probs, valid=(conf >= FLOOR and label != "Unknown"),
            extra=dict(NOT_MEASURED_EXTRA)))

    cap.release()
    return records


def run_single(clip_path: str, out_path: str):
    model, transform, device = load_model()
    records = process_clip(clip_path, model, transform, device)
    write_jsonl(records, out_path)
    print(f"[context_runner] {len(records)} frames -> {out_path}")


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
        print(f"[context_runner] resuming: {len(done_ids)} clips already done")
    else:
        mode = "w"

    model, transform, device = load_model()

    t0 = time.time()
    n_done = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i, row in enumerate(rows):
            clip_id = row["clip_id"]
            if clip_id in done_ids:
                continue
            clip_path = os.path.join(clips_root, row["filepath"])
            try:
                records = process_clip(clip_path, model, transform, device)
            except Exception as e:
                print(f"[context_runner] ERROR on {clip_id} ({clip_path}): {e}")
                continue
            append_batch(f, clip_id, records)
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(rows) - len(done_ids) - n_done) / rate if rate > 0 else float("inf")
                print(f"[context_runner] {i+1}/{len(rows)} clips ({n_done} this run, "
                      f"{rate:.2f} clips/s, ~{remaining/60:.1f} min remaining)")

    print(f"[context_runner] batch done: {n_done} clips processed -> {out_path}")


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
