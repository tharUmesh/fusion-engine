"""
Investigation (report-only): extracts raw pose_world_landmarks per frame for
a fixed set of clips, so they can be diffed across two mediapipe versions
(0.10.11, currently used in .venvs/motion, vs 0.10.14, the Motion Repo's own
pin -- see Motion Repo/README.md's warning that other versions "have
produced incompatible landmark output in the past").

Run once per venv (mediapipe can't be imported twice in one process), each
writing to a version-tagged .npz file. A separate stdlib-only comparison
script (compare_landmarks.py) diffs the two outputs.

Same MediaPipe Pose settings as runners/motion_runner.py (model_complexity=1,
min_detection_confidence=0.55, min_tracking_confidence=0.55) -- this is
purely an extraction/comparison tool, not a decision-logic change.
"""
import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp
import numpy as np


def resize_with_aspect_ratio(image, max_dim=960):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_clip(clip_path):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")

    orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else cap.get(48)
    rotation_code = None
    if orientation in (90, 270):
        _ret0, _f0 = cap.read()
        if _ret0:
            _fh, _fw = _f0.shape[:2]
            if not (_fh > _fw):
                rotation_code = cv2.ROTATE_90_CLOCKWISE if orientation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif orientation == 180:
        rotation_code = cv2.ROTATE_180

    frames_out = []  # (33,3) per frame, NaN-filled if no detection
    with mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                       min_detection_confidence=0.55, min_tracking_confidence=0.55,
                       static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
            frame = resize_with_aspect_ratio(frame, max_dim=960)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            wl = results.pose_world_landmarks
            if wl is not None:
                arr = np.array([[lm.x, lm.y, lm.z] for lm in wl.landmark], dtype=np.float64)
            else:
                arr = np.full((33, 3), np.nan, dtype=np.float64)
            frames_out.append(arr)
    cap.release()
    return np.stack(frames_out) if frames_out else np.zeros((0, 33, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clips-root", required=True)
    ap.add_argument("--clip-ids", required=True, help="comma-separated clip_ids to extract")
    ap.add_argument("--version-tag", required=True, help="e.g. mp0.10.11 or mp0.10.14")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    target_ids = set(args.clip_ids.split(","))

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["clip_id"] in target_ids]
    print(f"[extract_world_landmarks] {len(rows)} clips, mediapipe={mp.__version__}, tag={args.version_tag}")

    for row in rows:
        clip_id = row["clip_id"]
        clip_path = os.path.join(args.clips_root, row["filepath"])
        arr = extract_clip(clip_path)
        out_path = os.path.join(args.out_dir, f"{clip_id}__{args.version_tag}.npy")
        np.save(out_path, arr)
        print(f"[extract_world_landmarks] {clip_id}: {arr.shape} -> {out_path}")


if __name__ == "__main__":
    main()
