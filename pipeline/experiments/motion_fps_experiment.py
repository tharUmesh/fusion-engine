"""
Investigation (report-only, per explicit instruction -- does NOT change
runners/motion_runner.py or the model's window-size assumption):

Tests whether fps-normalizing input to ~30fps (nearest-neighbour frame
duplication, since most clips here are natively BELOW 30fps -- no clip in
scope needs downsampling) changes motion agreement on the 3 remaining
cue_corrupted motion scenarios (S08_F06, S23_F08, S26_F02) plus the two
still-correct stepping_back scenarios (S06_F08, S27_F06), for full context.

Method: mediapipe Pose still runs once per REAL captured frame (unchanged --
duplicating frames wouldn't invent new pose information, only resample the
model's input cadence). For each real frame, the already-computed joints_25
array is fed into MotionInference.update() a variable number of times
(nearest-neighbour upsampling schedule -- see `dup_schedule`), so the
30-frame sliding window advances at a simulated ~30fps pace instead of the
clip's native fps. Only the LAST update() call per real frame is recorded as
that frame's NormalisedFrameCue, so the "after" side has exactly the same
per-clip frame count as the "before" (native-fps) side -- an apples-to-apples
comparison, not an inflated one from counting duplicated virtual frames.

Output written to pipeline/experiments/ -- never pipeline/measured/.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNERS_DIR = os.path.join(REPO_ROOT, "runners")
sys.path.insert(0, RUNNERS_DIR)
MOTION_REPO = os.path.join(REPO_ROOT, "Motion Repo")
sys.path.insert(0, MOTION_REPO)

from common.schema import NormalisedFrameCue, append_batch, read_manifest  # noqa: E402
from common.constants import CONFIDENCE_FLOOR  # noqa: E402
import motion_runner as mr  # noqa: E402 -- reused unmodified: resize_with_aspect_ratio, CHECKPOINT

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from inference import MotionInference, MOTION_LABELS  # noqa: E402
from skeleton_utils import mediapipe_to_ntu25  # noqa: E402

CUE = "motion"
FLOOR = CONFIDENCE_FLOOR[CUE]
TARGET_FPS = 30.0


def dup_schedule(native_fps: float, n_frames: int):
    """Nearest-neighbour frame-rate conversion schedule: for each real frame
    index i, how many times to feed it to the model to simulate TARGET_FPS.
    dup(i) = round((i+1)*T/F) - round(i*T/F), clamped to >=1 (this dataset's
    clips are all <=30fps in the scope tested here, so this is pure
    upsampling; the clamp just guards the theoretical >30fps case)."""
    ratio = TARGET_FPS / native_fps
    sched = []
    prev_cum = 0
    for i in range(n_frames):
        cum = round((i + 1) * ratio)
        sched.append(max(1, cum - prev_cum))
        prev_cum = cum
    return sched


def process_clip_fps_normalized(clip_path: str, engine: MotionInference, native_fps: float):
    mp_pose = mp.solutions.pose
    engine.reset()

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

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

    # Schedule needs a frame count up front; VideoCapture's count can be
    # unreliable for some codecs, so fall back to a generous guess and
    # extend on the fly if the real stream runs longer.
    sched = dup_schedule(native_fps, n_frames_total or 100000)

    records = []
    frame_idx = -1

    with mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                       min_detection_confidence=0.55, min_tracking_confidence=0.55,
                       static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
            frame = mr.resize_with_aspect_ratio(frame, max_dim=960)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            world_landmarks = results.pose_world_landmarks

            has_landmarks = world_landmarks is not None
            if has_landmarks:
                joints_25 = mediapipe_to_ntu25(world_landmarks.landmark)
            else:
                joints_25 = np.zeros((25, 3), dtype=np.float32)

            n_dup = sched[frame_idx] if frame_idx < len(sched) else round(TARGET_FPS / native_fps)
            result = None
            for _ in range(n_dup):
                result = engine.update(joints_25)

            is_buffering = (result.label == "buffering")
            label = result.label if not is_buffering else "Unknown"
            confidence = float(result.confidence)
            probs_dict = {} if is_buffering else {
                MOTION_LABELS[i]: float(p) for i, p in enumerate(result.probs)
            }
            valid = (not is_buffering) and (confidence >= FLOOR) and has_landmarks

            records.append(NormalisedFrameCue(
                cue=CUE, frame_idx=frame_idx, label=label, confidence=confidence,
                probs=probs_dict, valid=valid,
                extra={"buffering": is_buffering, "has_landmarks": has_landmarks}))

    cap.release()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clips-root", required=True)
    ap.add_argument("--scenarios", required=True, help="comma-separated scenario_ids to restrict to")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target_scenarios = set(args.scenarios.split(","))
    rows = read_manifest(args.manifest)
    rows = [r for r in rows if r["scenario_id"] in target_scenarios]
    print(f"[motion_fps_experiment] {len(rows)} clips across {len(target_scenarios)} scenarios")

    engine = MotionInference(mr.CHECKPOINT)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            clip_id = row["clip_id"]
            native_fps = float(row["fps"])
            clip_path = os.path.join(args.clips_root, row["filepath"])
            try:
                records = process_clip_fps_normalized(clip_path, engine, native_fps)
            except Exception as e:
                print(f"[motion_fps_experiment] ERROR on {clip_id} ({clip_path}): {e}")
                continue
            append_batch(f, clip_id, records)
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"[motion_fps_experiment] {i+1}/{len(rows)} clips done")

    print(f"[motion_fps_experiment] batch done: {len(rows)} clips -> {args.out}")


if __name__ == "__main__":
    main()
