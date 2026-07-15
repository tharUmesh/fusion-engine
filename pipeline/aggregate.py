"""
Phase 2 feature-vector builder: turns per-frame measured cues (Phase 0's
pipeline/measured/<cue>_frame_cues.jsonl) into ONE fixed-length feature
vector per clip (Option A from HRI_Fusion_Engine_Handover.md §Phase 2).

Distinct from pipeline/aggregate_clip_cues.py (Phase 0's lightweight
majority-vote-label diagnostic for the agreement report) -- this module
builds the actual model-input feature vector.

PROVISIONAL, per explicit instruction: emotion and gesture cue models are
expected to be replaced (as motion already was), so this feature layout is a
moving baseline, not a frozen contract yet.

Deviations from the handover doc's original ~30-40 dim layout (kept
consistent with what the cue models actually emit, not the aspirational
plan):
  - Motion is 4-class (sitting/standing/walking/stepping_back), not the
    original 8-class taxonomy -- the Motion Repo was replaced 2026-07 (see
    runners/motion_runner.py). 4 mean-probs, not 8.
  - No `pose one-hot(5)` -- the new Motion Repo has no separate pose
    classifier (MODEL_ANALYSIS.md's Motion section, HRI_Fusion_Engine_Handover.md
    §3 note). Dropped, not zero-filled -- there is nothing to encode.
  - No `activity one-hot` / `engaged` bit -- Context can never deliver these
    (MODEL_ANALYSIS.md #0b finding 4, `NOT_MEASURED_EXTRA`); a constant
    "always missing" feature carries zero information and would just be
    dead weight in the vector. Acknowledged v1 scope cut, not an oversight.
  - No `motion_direction` / `point_target` one-hot -- gesture_runner.py's
    `extra` for these is hardcoded ("none" / "unknown") for every frame of
    every clip (MODEL_ANALYSIS.md #2.5); same reasoning as above.

Final feature vector (33 dims):
  emotion  (9):  7 mean-probs (Surprise,Fear,Disgust,Happy,Sad,Anger,Neutral)
                 + max_confidence + valid_fraction
  gesture  (10): 8 one-hot majority label (wave,point,thumbs_up,thumbs_down,
                 raise_hand,both_hands_up,beckoning,Unknown)
                 + mean_confidence (of the winning label) + valid_fraction
  motion   (6):  4 mean-probs (sitting,standing,walking,stepping_back)
                 + max_confidence + valid_fraction
  context  (4):  2 one-hot scene (classroom,kitchen)
                 + scene_confidence (mean) + valid_fraction
  missing  (4):  missing_emotion, missing_gesture, missing_motion, missing_context
                 (valid_fraction < CLIP_MISSING_THRESHOLD)

Always pairs zeroed cue blocks with their missing-bit (never a real zero
indistinguishable from an absent cue) -- see the handover doc's Phase 2
failure-point warning.

Stdlib + numpy only.
"""
import json
import os
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEASURED_DIR = os.path.join(REPO_ROOT, "pipeline", "measured")

CLIP_MISSING_THRESHOLD = 0.40  # matches handover schema.yaml's clip_missing_threshold

EMOTION_CLASSES = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
GESTURE_CLASSES = ["wave", "point", "thumbs_up", "thumbs_down", "raise_hand",
                    "both_hands_up", "beckoning", "Unknown"]
MOTION_CLASSES = ["sitting", "standing", "walking", "stepping_back"]
CONTEXT_CLASSES = ["classroom", "kitchen"]

FEATURE_NAMES = (
    [f"emotion_{c}" for c in EMOTION_CLASSES] + ["emotion_max_confidence", "emotion_valid_fraction"]
    + [f"gesture_{c}" for c in GESTURE_CLASSES] + ["gesture_mean_confidence", "gesture_valid_fraction"]
    + [f"motion_{c}" for c in MOTION_CLASSES] + ["motion_max_confidence", "motion_valid_fraction"]
    + [f"context_{c}" for c in CONTEXT_CLASSES] + ["context_mean_confidence", "context_valid_fraction"]
    + ["missing_emotion", "missing_gesture", "missing_motion", "missing_context"]
)


def load_frame_cues_by_clip(cue: str):
    """Returns {clip_id: [frame_record, ...]} for one cue's JSONL."""
    path = os.path.join(MEASURED_DIR, f"{cue}_frame_cues.jsonl")
    by_clip = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_clip[d["clip_id"]].append(d)
    return by_clip


def _prob_mean_features(records, classes):
    """Mean of per-frame probability vectors over valid frames -> len(classes)
    floats, plus max_confidence and valid_fraction. All-zero (+ missing bit
    set by the caller) if no valid frames."""
    valid = [r for r in records if r["valid"]]
    n_total = len(records)
    valid_fraction = len(valid) / n_total if n_total else 0.0
    if not valid:
        return [0.0] * len(classes), 0.0, valid_fraction
    mat = np.array([[r["probs"].get(c, 0.0) for c in classes] for r in valid], dtype=np.float64)
    means = mat.mean(axis=0).tolist()
    max_confidence = max(r["confidence"] for r in valid)
    return means, max_confidence, valid_fraction


def _majority_onehot_features(records, classes):
    """Majority-vote label over valid frames -> one-hot(len(classes)), plus
    mean_confidence of the winning label's frames and valid_fraction."""
    valid = [r for r in records if r["valid"]]
    n_total = len(records)
    valid_fraction = len(valid) / n_total if n_total else 0.0
    onehot = [0.0] * len(classes)
    if not valid:
        return onehot, 0.0, valid_fraction
    from collections import Counter
    votes = Counter(r["label"] for r in valid)
    winner, _ = votes.most_common(1)[0]
    if winner in classes:
        onehot[classes.index(winner)] = 1.0
    winner_confidences = [r["confidence"] for r in valid if r["label"] == winner]
    mean_confidence = sum(winner_confidences) / len(winner_confidences)
    return onehot, mean_confidence, valid_fraction


def build_clip_feature_row(clip_id, emotion_records, gesture_records, motion_records, context_records):
    row = {"clip_id": clip_id}

    e_means, e_max_conf, e_valid_frac = _prob_mean_features(emotion_records, EMOTION_CLASSES)
    for c, v in zip(EMOTION_CLASSES, e_means):
        row[f"emotion_{c}"] = v
    row["emotion_max_confidence"] = e_max_conf
    row["emotion_valid_fraction"] = e_valid_frac

    g_onehot, g_mean_conf, g_valid_frac = _majority_onehot_features(gesture_records, GESTURE_CLASSES)
    for c, v in zip(GESTURE_CLASSES, g_onehot):
        row[f"gesture_{c}"] = v
    row["gesture_mean_confidence"] = g_mean_conf
    row["gesture_valid_fraction"] = g_valid_frac

    m_means, m_max_conf, m_valid_frac = _prob_mean_features(motion_records, MOTION_CLASSES)
    for c, v in zip(MOTION_CLASSES, m_means):
        row[f"motion_{c}"] = v
    row["motion_max_confidence"] = m_max_conf
    row["motion_valid_fraction"] = m_valid_frac

    c_means, c_max_conf, c_valid_frac = _prob_mean_features(context_records, CONTEXT_CLASSES)
    # Context: mean scene-confidence (not max) -- matches "scene_confidence"
    # in the handover spec, which is the smoothed per-frame value, not a peak.
    valid_ctx = [r for r in context_records if r["valid"]]
    c_mean_conf = (sum(r["confidence"] for r in valid_ctx) / len(valid_ctx)) if valid_ctx else 0.0
    for c, v in zip(CONTEXT_CLASSES, c_means):
        row[f"context_{c}"] = v
    row["context_mean_confidence"] = c_mean_conf
    row["context_valid_fraction"] = c_valid_frac

    row["missing_emotion"] = float(e_valid_frac < CLIP_MISSING_THRESHOLD)
    row["missing_gesture"] = float(g_valid_frac < CLIP_MISSING_THRESHOLD)
    row["missing_motion"] = float(m_valid_frac < CLIP_MISSING_THRESHOLD)
    row["missing_context"] = float(c_valid_frac < CLIP_MISSING_THRESHOLD)

    return row
