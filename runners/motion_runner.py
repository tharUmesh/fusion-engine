"""
Standalone Motion runner. Wraps the Motion Repo's new fine-tuned model
(MotionLSTM + temporal attention, 4-class taxonomy: sitting / standing /
walking / stepping_back -- see "Motion Repo/README.md") behind the same
NormalisedFrameCue interface as the other three cue runners.

This replaces the previous rule-based 8-class physics engine
(action_recognizer.py) that the old Motion Repo shipped. That repo has been
removed; this runner now imports the new repo's own MotionInference /
mediapipe_to_ntu25 unmodified.

Integration notes (see Motion Repo/README.md's input contract):
  - The model needs `results.pose_world_landmarks` (metric 3D), NOT
    `results.pose_landmarks` (normalised image-space) -- mediapipe_to_ntu25()
    is written specifically against the world-landmark format.
  - MotionInference is stateful (30-frame sliding window + previous-frame
    buffer for velocity features). The model is loaded ONCE per batch run
    (matches the other runners' "load once, loop every clip" convention);
    engine.reset() is called at the start of every clip instead of
    reconstructing MotionInference, exactly the use documented in the
    repo's own README ("call engine.reset() whenever the person leaves the
    frame or the scene changes") -- a new clip is that case.
  - MediaPipe Pose itself IS recreated fresh per clip (its own internal
    tracking state must not leak across unrelated clips, same reasoning as
    the previous runner and the other three).
  - The first WINDOW_SIZE-1 (29) frames of every clip return
    label="buffering", confidence=0.0 -- this is the new model's own honest
    "not enough context yet" state. Unlike the old model's cold-start
    fabrication (label="Standing Still", confidence=0.90 -- a silent trap,
    see MODEL_ANALYSIS.md #3.9), there is nothing to guard against here:
    buffering frames already report confidence=0.0 and are marked
    valid=False below without any special-casing.
  - No separate pose classifier (Sitting/Standing/Crouching/Lying/Unknown)
    exists in the new repo -- the single 4-class model already folds
    static-pose state into its primary label. `extra["pose"]` from the old
    runner is dropped; see HRI_Fusion_Engine_Handover.md §3's note on this.
  - Frame-rate mismatch (found during Phase 0 rerun, not fixed here per
    explicit instruction): the new model assumes ~30fps input (30-frame
    window ~= 1s). This dataset's clips.csv shows a wide fps spread
    (827/1270 clips at 15fps, only 226 at 30fps, remainder 23-31fps) -- the
    model's own README states 15fps "stretches the effective window to 2s
    and degrades accuracy". No resampling/frame-duplication is done here;
    the model sees exactly the source clip's native frame sequence,
    faithfully, so this is captured as a Phase 0 finding, not corrected.

Run inside .venvs/motion (torch, numpy, mediapipe==0.10.14, opencv-python --
see "Motion Repo/requirements.txt"; no tensorflow/protobuf pin needed any
more, the new repo has no TFLite dependency).

Usage:
    # single clip
    .venvs/motion/bin/python runners/motion_runner.py --clip <path> --out <out.jsonl>

    # batch mode: loads the model ONCE, loops every clip in clips.csv
    .venvs/motion/bin/python runners/motion_runner.py \
        --manifest Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/clips.csv \
        --clips-root Data/Dataset/hri-multimodal-intent-v1.0.0 \
        --out pipeline/measured/motion_frame_cues.jsonl
"""
import argparse
import os
import sys
import time
import numpy as np

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS_DIR)
MOTION_REPO = os.path.join(os.path.dirname(RUNNERS_DIR), "Motion Repo")
sys.path.insert(0, MOTION_REPO)

from common.schema import NormalisedFrameCue, write_jsonl, append_batch, read_manifest  # noqa: E402
from common.constants import CONFIDENCE_FLOOR  # noqa: E402

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
# Module-level imports only -- MotionInference's __init__ does the (one-time,
# batch-wide) checkpoint load; nothing here triggers webcam/GUI code.
from inference import MotionInference, MOTION_LABELS  # noqa: E402
from skeleton_utils import mediapipe_to_ntu25  # noqa: E402

CUE = "motion"
FLOOR = CONFIDENCE_FLOOR[CUE]
CHECKPOINT = os.path.join(MOTION_REPO, "checkpoints", "best_model_finetuned.pt")


def resize_with_aspect_ratio(image, max_dim=960):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def process_clip(clip_path: str, engine: MotionInference):
    """Pure per-clip logic. Creates a fresh MediaPipe Pose tracker for this
    clip and resets the (reused, already-loaded) MotionInference engine's
    sliding window -- see module docstring."""
    mp_pose = mp.solutions.pose
    engine.reset()

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

    records = []
    frame_idx = -1
    n_occlusion_resets = 0
    was_tracking = False

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
            frame = resize_with_aspect_ratio(frame, max_dim=960)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            world_landmarks = results.pose_world_landmarks

            has_landmarks = world_landmarks is not None
            if has_landmarks:
                if not was_tracking and frame_idx > 0:
                    n_occlusion_resets += 1  # re-acquired tracking mid-clip after a drop
                was_tracking = True
                joints_25 = mediapipe_to_ntu25(world_landmarks.landmark)
            else:
                was_tracking = False
                # README: "pass zeros for brief gaps" -- keeps the sliding
                # window advancing instead of fabricating a detection.
                joints_25 = np.zeros((25, 3), dtype=np.float32)

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
    return records, n_occlusion_resets


def run_single(clip_path: str, out_path: str):
    engine = MotionInference(CHECKPOINT)
    records, n_resets = process_clip(clip_path, engine)
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

    engine = MotionInference(CHECKPOINT)

    stats_f = None
    if stats_path:
        stats_f = open(stats_path, stats_mode, encoding="utf-8", newline="")
        if stats_mode == "w":
            stats_f.write("clip_id,total_frames,valid_frames,invalid_frames,buffering_frames,"
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
                records, n_resets = process_clip(clip_path, engine)
            except Exception as e:
                print(f"[motion_runner] ERROR on {clip_id} ({clip_path}): {e}")
                continue
            append_batch(f, clip_id, records)
            f.flush()

            if stats_f:
                n_valid = sum(1 for r in records if r.valid)
                n_buffering = sum(1 for r in records if r.extra.get("buffering"))
                n_no_landmark = sum(1 for r in records if not r.extra.get("has_landmarks"))
                stats_f.write(f"{clip_id},{len(records)},{n_valid},{len(records)-n_valid},"
                               f"{n_buffering},{n_no_landmark},{n_resets}\n")
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
    ap.add_argument("--stats-out", help="batch mode: per-clip valid/invalid/buffering CSV diagnostic")
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
