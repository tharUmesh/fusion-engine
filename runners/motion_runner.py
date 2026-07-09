"""
Standalone Motion runner. Reimplements action_recognizer.py's run() per-frame
physics-engine logic headlessly (that function is GUI-coupled with no return
value -- see Integration_API.md #3.3), reusing the repo's own PoseClassifier
and MOTION_LABELS via direct import (module-level, safe to import without
triggering run()/main()).

Correctness fixes applied here (see MODEL_ANALYSIS.md #3.3/#3.9,
Integration_API.md #2.3):
  - The LSTM (motion_lstm_v2_best.pth) is loaded by the native script but
    never actually called anywhere -- classification is a fully deterministic
    physics/rules engine. This runner does not load it at all.
  - Cold-start guard: the native code fabricates label="Standing Still",
    confidence=0.90 for the first <4 buffered frames of a clip (and again
    after any single frame with no detected landmarks, since one bad frame
    clears the whole 30-frame window) -- a value that would pass the 0.50
    confidence floor despite not being a real detection. This runner marks
    such frames valid=False regardless of the fabricated confidence, and
    does so EVERY time the buffer drops below 4 frames -- not just at clip
    start -- because `is_cold_start` is a per-frame local re-derived from
    the live buffer length, not a one-shot flag. This has been verified to
    also catch mid-clip re-warm-up after an occlusion event (see
    MODEL_ANALYSIS.md's Phase 0 report).

In batch mode, MediaPipe Pose is recreated fresh per clip (it carries
internal cross-frame tracking state that must not leak between unrelated
clips -- matches the native script always starting one process per video).
PoseClassifier is stateless and reused across clips.

Run inside .venvs/motion (mediapipe==0.10.11, opencv-python, numpy, torch --
torch is still an import-time dependency of action_recognizer.py itself
(MotionLSTM's class definition needs torch.nn to exist at import), even
though this runner never instantiates or calls the LSTM -- see
Integration_API.md #4).

Usage:
    # single clip
    .venvs/motion/Scripts/python.exe runners/motion_runner.py --clip <path> --out <out.jsonl>

    # batch mode: loops every clip in clips.csv (Pose tracker recreated per clip)
    .venvs/motion/Scripts/python.exe runners/motion_runner.py \
        --manifest Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/clips.csv \
        --clips-root Data/Dataset/hri-multimodal-intent-v1.0.0 \
        --out data/measured/motion_frame_cues.jsonl
"""
import argparse
import os
import sys
import time
from collections import deque

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS_DIR)
MOTION_REPO = os.path.join(os.path.dirname(RUNNERS_DIR), "Motion Repo")
sys.path.insert(0, MOTION_REPO)

from common.schema import NormalisedFrameCue, write_jsonl, append_batch, read_manifest  # noqa: E402
from common.constants import CONFIDENCE_FLOOR  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import mediapipe as mp  # noqa: E402
# Module-level import only (PoseClassifier, MOTION_LABELS) -- does NOT load
# the LSTM or run any GUI code, both of which live inside run()/main().
from action_recognizer import PoseClassifier, MOTION_LABELS  # noqa: E402

CUE = "motion"
FLOOR = CONFIDENCE_FLOOR[CUE]

HIP_IDX = [23, 24]
BODY_IDX = HIP_IDX
RUN_THRESH = 1.30
SMOOTH_ALPHA = 0.25


def resize_with_aspect_ratio(image, max_dim=960):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def process_clip(clip_path: str, pose_classifier):
    """Pure per-clip logic. Creates a fresh MediaPipe Pose tracker for this
    clip (see module docstring)."""
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

    keypoints_queue = deque(maxlen=30)
    smooth_probs = np.ones(8) / 8

    records = []
    frame_idx = -1
    # Diagnostics for the occlusion/re-warm-up question -- not part of the
    # NormalisedFrameCue schema, returned separately per clip.
    n_occlusion_resets = 0
    was_tracking = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
            frame = resize_with_aspect_ratio(frame, max_dim=960)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            landmarks = results.pose_landmarks

            pose_label = pose_classifier.classify(landmarks)

            if landmarks:
                if not was_tracking and frame_idx > 0:
                    n_occlusion_resets += 1  # re-acquired tracking mid-clip after a drop
                was_tracking = True
                pts = np.zeros((33, 3), dtype=np.float32)
                for i, lm in enumerate(landmarks.landmark):
                    pts[i] = [lm.x, lm.y, lm.z]
                keypoints_queue.append(pts)
            else:
                was_tracking = False
                keypoints_queue.clear()

            # Defaults (native code's cold-start fabrication)
            motion_label = "Standing Still"
            confidence = 0.90
            probabilities = np.zeros(8)
            probabilities[1] = 0.90
            is_cold_start = True  # fix: re-derived every frame, see module docstring

            if len(keypoints_queue) >= 4 and landmarks:
                is_cold_start = False
                arr = np.array(keypoints_queue)
                vel = np.diff(arr, axis=0)

                body_vel = vel[:, BODY_IDX, :2]
                body_speed = float(np.mean(np.abs(body_vel)) * 100.0)

                look_back = min(15, len(arr) - 1)
                hip_w = np.linalg.norm(arr[-look_back - 1:, 23, :2] - arr[-look_back - 1:, 24, :2], axis=1)
                dh = float(hip_w[-1] - hip_w[0])
                path_h = float(np.sum(np.abs(np.diff(hip_w))))
                eff_h = abs(dh) / (path_h + 1e-5)

                hips_xy = arr[-look_back - 1:, HIP_IDX, :2].mean(axis=1)
                dx_hip = float(hips_xy[-1, 0] - hips_xy[0, 0])
                dy_hip = float(hips_xy[-1, 1] - hips_xy[0, 1])
                path_x_hip = float(np.sum(np.abs(np.diff(hips_xy[:, 0]))))
                path_y_hip = float(np.sum(np.abs(np.diff(hips_xy[:, 1]))))
                eff_x_hip = abs(dx_hip) / (path_x_hip + 1e-5)
                eff_y_hip = abs(dy_hip) / (path_y_hip + 1e-5)

                sh_xy = arr[-look_back - 1:, [11, 12], :2].mean(axis=1)
                dx_sh = float(sh_xy[-1, 0] - sh_xy[0, 0])
                dy_sh = float(sh_xy[-1, 1] - sh_xy[0, 1])
                path_x_sh = float(np.sum(np.abs(np.diff(sh_xy[:, 0]))))
                path_y_sh = float(np.sum(np.abs(np.diff(sh_xy[:, 1]))))
                eff_x_sh = abs(dx_sh) / (path_x_sh + 1e-5)
                eff_y_sh = abs(dy_sh) / (path_y_sh + 1e-5)

                is_translating_across = False
                if abs(dx_hip) > 0.024 and eff_x_hip > 0.70:
                    if abs(dx_sh) > 0.020 and eff_x_sh > 0.70:
                        if np.sign(dx_hip) == np.sign(dx_sh):
                            is_translating_across = True

                is_translating_vert = False
                if abs(dy_hip) > 0.020 and eff_y_hip > 0.70:
                    if abs(dy_sh) > 0.016 and eff_y_sh > 0.70:
                        if np.sign(dy_hip) == np.sign(dy_sh):
                            is_translating_vert = True

                is_directed_walk = False
                walk_type = "Walking"
                if is_translating_across and abs(dx_hip) > abs(dy_hip) * 1.5:
                    is_directed_walk = True
                    walk_type = "Walk Across"
                elif is_translating_vert or (abs(dh) > 0.012 and eff_h > 0.70):
                    is_directed_walk = True
                    walk_type = "Walking"

                probs = np.zeros(8)
                if body_speed >= RUN_THRESH:
                    if dx_hip < -0.01:
                        motion_label, probs[4] = "Run Backward", 0.85
                    else:
                        motion_label, probs[5] = "Run (Fast Movement)", 0.85
                elif is_directed_walk:
                    if walk_type == "Walk Across":
                        motion_label, probs[3] = "Walk Across", 0.80
                    else:
                        motion_label, probs[2] = "Walking", 0.80
                else:
                    if pose_label == "Sitting":
                        motion_label, probs[0] = "Sitting Still", 0.95
                    elif pose_label == "Crouching":
                        motion_label, probs[6] = "Leaning Forward", 0.90
                    elif body_speed < 0.08:
                        motion_label, probs[7] = "Frozen/Rigid Stand", 0.90
                    else:
                        motion_label, probs[1] = "Standing Still", 0.90

                smooth_probs = SMOOTH_ALPHA * probs + (1 - SMOOTH_ALPHA) * smooth_probs
                voted_idx = int(np.argmax(smooth_probs))
                motion_label = MOTION_LABELS[voted_idx]
                confidence = float(smooth_probs[voted_idx])
                probabilities = smooth_probs

            elif not landmarks:
                smooth_probs = np.ones(8) / 8
                motion_label = "Standing Still"
                confidence = 0.0
                probabilities = np.zeros(8)
                # not a cold-start fabrication -- genuinely no detection this frame

            probs_dict = {lbl: float(p) for lbl, p in zip(MOTION_LABELS, probabilities)}

            # Fix: cold-start frames (native code fabricates "Standing Still"
            # @ 0.90 before the 30-frame buffer has >=4 entries -- including
            # after a mid-clip occlusion reset) are never valid, regardless
            # of the fabricated confidence passing FLOOR.
            valid = (confidence >= FLOOR) and (not is_cold_start) and (landmarks is not None)

            records.append(NormalisedFrameCue(
                cue=CUE, frame_idx=frame_idx, label=motion_label, confidence=confidence,
                probs=probs_dict, valid=valid,
                extra={"pose": pose_label, "cold_start": is_cold_start}))

    cap.release()
    return records, n_occlusion_resets


def run_single(clip_path: str, out_path: str):
    pose_classifier = PoseClassifier()
    records, n_resets = process_clip(clip_path, pose_classifier)
    write_jsonl(records, out_path)
    n_valid = sum(1 for r in records if r.valid)
    print(f"[motion_runner] {len(records)} frames -> {out_path} "
          f"({n_valid} valid, {len(records)-n_valid} invalid, {n_resets} mid-clip occlusion resets)")


def run_batch(manifest_csv: str, clips_root: str, out_path: str, limit=None, resume=False,
              stats_path=None):
    rows = read_manifest(manifest_csv)
    if limit:
        rows = rows[:limit]

    done_ids = set()
    mode = "a"
    stats_mode = "a"
    if resume and os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(line.split('"clip_id": "', 1)[1].split('"', 1)[0])
                except IndexError:
                    pass
        print(f"[motion_runner] resuming: {len(done_ids)} clips already done")
    else:
        mode = "w"
        stats_mode = "w"

    pose_classifier = PoseClassifier()

    stats_f = None
    if stats_path:
        stats_f = open(stats_path, stats_mode, encoding="utf-8", newline="")
        if stats_mode == "w":
            stats_f.write("clip_id,total_frames,valid_frames,invalid_frames,cold_start_frames,"
                           "no_landmark_frames,mid_clip_occlusion_resets\n")

    t0 = time.time()
    n_done = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i, row in enumerate(rows):
            clip_id = row["clip_id"]
            if clip_id in done_ids:
                continue
            clip_path = os.path.join(clips_root, row["filepath"])
            try:
                records, n_resets = process_clip(clip_path, pose_classifier)
            except Exception as e:
                print(f"[motion_runner] ERROR on {clip_id} ({clip_path}): {e}")
                continue
            append_batch(f, clip_id, records)
            f.flush()

            if stats_f:
                n_valid = sum(1 for r in records if r.valid)
                n_cold = sum(1 for r in records if r.extra.get("cold_start"))
                n_no_landmark = sum(1 for r in records if r.confidence == 0.0 and not r.extra.get("cold_start"))
                stats_f.write(f"{clip_id},{len(records)},{n_valid},{len(records)-n_valid},"
                               f"{n_cold},{n_no_landmark},{n_resets}\n")
                stats_f.flush()

            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(rows) - len(done_ids) - n_done) / rate if rate > 0 else float("inf")
                print(f"[motion_runner] {i+1}/{len(rows)} clips ({n_done} this run, "
                      f"{rate:.2f} clips/s, ~{remaining/60:.1f} min remaining)")

    if stats_f:
        stats_f.close()
    print(f"[motion_runner] batch done: {n_done} clips processed -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="single-clip mode: path to one clip")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--manifest", help="batch mode: path to clips.csv")
    ap.add_argument("--clips-root", help="batch mode: dataset root (filepath column is relative to this)")
    ap.add_argument("--limit", type=int, default=None, help="batch mode: only process first N rows (testing)")
    ap.add_argument("--resume", action="store_true", help="batch mode: skip clip_ids already present in --out")
    ap.add_argument("--stats-out", help="batch mode: per-clip valid/invalid/occlusion CSV diagnostic")
    args = ap.parse_args()

    if args.manifest:
        if not args.clips_root:
            raise SystemExit("--clips-root is required with --manifest")
        run_batch(args.manifest, args.clips_root, args.out, limit=args.limit, resume=args.resume,
                   stats_path=args.stats_out)
    else:
        if not args.clip:
            raise SystemExit("either --clip or --manifest is required")
        run_single(args.clip, args.out)
