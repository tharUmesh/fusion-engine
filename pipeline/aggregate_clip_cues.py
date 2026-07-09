"""
Lightweight clip-level cue aggregation -- majority vote over VALID frames
only, per (clip_id, cue). This is NOT the Phase 2 feature-vector builder
(handover doc's ~30-40 dim aggregation with probability means, one-hot
encodings, etc.) -- that is explicitly out of scope for this step. This
script produces just enough to drive the Phase 0 agreement report: one
dominant measured label per clip per cue, plus enough frame-validity
bookkeeping to know when a clip's measured cue is unreliable.

Hard contract: this script (and everything downstream of it) reads ONLY the
NormalisedFrameCue schema fields (cue, frame_idx, label, confidence, probs,
valid, extra) plus the clip_id envelope added by the runners' batch mode. It
contains no cue-model-specific assumptions -- see Integration_API.md's
requirement that aggregation/fusion code depend only on the schema, so that
swapping a cue model later only requires changing its runner, never this
script.

missing_fraction threshold (0.40) matches the handover document's
clip_missing_threshold (schema.yaml plan, section 3) -- a clip's cue is
flagged "insufficient_valid_frames" if fewer than 60% of its frames are
valid, independent of which label is voted for.

Stdlib-only. Reads pipeline/measured/<cue>_frame_cues.jsonl (one per cue,
written by the batch runners), writes pipeline/measured/clip_cues.csv.
"""
import csv
import json
import os
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEASURED_DIR = os.path.join(REPO_ROOT, "pipeline", "measured")
CUES = ["emotion", "gesture", "motion", "context"]
MISSING_FRACTION_THRESHOLD = 0.40  # matches handover schema.yaml's clip_missing_threshold


def load_frame_cues(cue: str):
    path = os.path.join(MEASURED_DIR, f"{cue}_frame_cues.jsonl")
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate_cue(records):
    """Groups by clip_id, majority-votes the label among valid frames only.
    Returns {clip_id: {n_frames, n_valid, valid_fraction, dominant_label,
    insufficient_valid_frames, vote_distribution}}."""
    by_clip = defaultdict(list)
    for r in records:
        by_clip[r["clip_id"]].append(r)

    out = {}
    for clip_id, frames in by_clip.items():
        n_frames = len(frames)
        valid_frames = [f for f in frames if f["valid"]]
        n_valid = len(valid_frames)
        valid_fraction = n_valid / n_frames if n_frames else 0.0

        votes = Counter(f["label"] for f in valid_frames)
        dominant_label = votes.most_common(1)[0][0] if votes else None
        insufficient = valid_fraction < MISSING_FRACTION_THRESHOLD

        out[clip_id] = {
            "n_frames": n_frames,
            "n_valid": n_valid,
            "valid_fraction": round(valid_fraction, 4),
            "dominant_label": dominant_label,
            "insufficient_valid_frames": insufficient,
            "vote_distribution": dict(votes),
        }
    return out


def main():
    out_path = os.path.join(MEASURED_DIR, "clip_cues.csv")
    rows = []
    for cue in CUES:
        records = load_frame_cues(cue)
        if not records:
            print(f"[aggregate_clip_cues] WARNING: no data for cue={cue} yet (skipping)")
            continue
        agg = aggregate_cue(records)
        for clip_id, stats in agg.items():
            rows.append({
                "clip_id": clip_id,
                "cue": cue,
                "n_frames": stats["n_frames"],
                "n_valid": stats["n_valid"],
                "valid_fraction": stats["valid_fraction"],
                "dominant_label": stats["dominant_label"],
                "insufficient_valid_frames": stats["insufficient_valid_frames"],
                "vote_distribution": json.dumps(stats["vote_distribution"]),
            })
        print(f"[aggregate_clip_cues] {cue}: {len(agg)} clips aggregated "
              f"({sum(1 for r in agg.values() if r['insufficient_valid_frames'])} "
              f"with insufficient valid frames)")

    if not rows:
        print("[aggregate_clip_cues] no data available yet -- nothing written")
        return

    fieldnames = ["clip_id", "cue", "n_frames", "n_valid", "valid_fraction",
                  "dominant_label", "insufficient_valid_frames", "vote_distribution"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[aggregate_clip_cues] wrote {len(rows)} (clip, cue) rows -> {out_path}")


if __name__ == "__main__":
    main()
