"""
Investigation (report-only): compares pose_world_landmarks extracted with
mediapipe==0.10.11 (currently used in .venvs/motion) vs mediapipe==0.10.14
(the Motion Repo's own pin) on the same clips/frames, to check whether the
version deviation documented in HRI_Fusion_Engine_Handover.md (Phase 0a) is
still safe now that a different model (mediapipe_to_ntu25 + MotionLSTM)
consumes these exact world landmarks.

Stdlib + numpy only -- reads the .npy files written by
extract_world_landmarks.py under both version tags.
"""
import glob
import os
import re
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPARE_DIR = os.path.join(REPO_ROOT, "pipeline", "experiments", "mp_compare")


def main():
    files_a = sorted(glob.glob(os.path.join(COMPARE_DIR, "*__mp0.10.11.npy")))
    clip_ids = [re.match(r"(.+)__mp0\.10\.11\.npy", os.path.basename(f)).group(1) for f in files_a]

    all_diffs = []
    detection_mismatches = 0
    total_frames_compared = 0
    per_clip_summary = []

    for clip_id in clip_ids:
        path_a = os.path.join(COMPARE_DIR, f"{clip_id}__mp0.10.11.npy")
        path_b = os.path.join(COMPARE_DIR, f"{clip_id}__mp0.10.14.npy")
        if not os.path.isfile(path_b):
            print(f"WARNING: missing 0.10.14 extraction for {clip_id}, skipping")
            continue
        a = np.load(path_a)
        b = np.load(path_b)
        n = min(len(a), len(b))
        if len(a) != len(b):
            print(f"WARNING: {clip_id} frame count differs: 0.10.11={len(a)} 0.10.14={len(b)} (comparing first {n})")
        a, b = a[:n], b[:n]

        nan_a = np.isnan(a).any(axis=(1, 2))
        nan_b = np.isnan(b).any(axis=(1, 2))
        mismatch = (nan_a != nan_b)
        detection_mismatches += int(mismatch.sum())
        total_frames_compared += n

        both_detected = (~nan_a) & (~nan_b)
        if both_detected.any():
            diff = np.linalg.norm(a[both_detected] - b[both_detected], axis=-1)  # (n_both, 33) per-joint L2 in metres
            all_diffs.append(diff)
            per_clip_summary.append((clip_id, n, int(both_detected.sum()), int(mismatch.sum()),
                                      float(diff.mean()), float(diff.max())))
        else:
            per_clip_summary.append((clip_id, n, 0, int(mismatch.sum()), float("nan"), float("nan")))

    print(f"{'clip_id':<16} {'frames':>7} {'both_det':>9} {'det_mismatch':>13} {'mean_diff_m':>12} {'max_diff_m':>11}")
    for row in per_clip_summary:
        print(f"{row[0]:<16} {row[1]:>7} {row[2]:>9} {row[3]:>13} {row[4]:>12.5f} {row[5]:>11.5f}")

    if all_diffs:
        combined = np.concatenate(all_diffs, axis=0)
        print()
        print(f"=== Overall ({len(clip_ids)} clips, {total_frames_compared} frames compared) ===")
        print(f"Detection presence mismatches (one version saw a person, other didn't): "
              f"{detection_mismatches} / {total_frames_compared} frames "
              f"({100*detection_mismatches/total_frames_compared:.2f}%)")
        print(f"Per-joint L2 distance (metres) over all jointly-detected frame/joint pairs:")
        print(f"  mean = {combined.mean():.5f}")
        print(f"  median = {np.median(combined):.5f}")
        print(f"  p95 = {np.percentile(combined, 95):.5f}")
        print(f"  max = {combined.max():.5f}")
    else:
        print("No jointly-detected frames found across any clip -- cannot compare numerically.")


if __name__ == "__main__":
    main()
