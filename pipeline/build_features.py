"""
Phase 2 orchestration: runs pipeline/aggregate.py's per-clip feature builder
over every clip in clips.csv, joins in the target label (scenarios.csv's
`Intent`, e.g. F01) and the train/val/test split assignment (splits.csv),
and writes Data/Dataset/hri-multimodal-intent-v1.0.0/../features/clip_features.parquet
(kept next to the dataset's own annotations, mirroring where clips.csv/
scenarios.csv/splits.csv already live).

PROVISIONAL, per explicit instruction -- emotion and gesture cue models are
expected to be replaced next, same as motion already was; this feature
layout and the resulting parquet are a moving baseline, not yet frozen.

Needs pandas + pyarrow (see .venvs/pipeline) -- unlike Phase 0's aggregation
scripts, Phase 2+ has no stdlib-only constraint (that hard contract was
specific to pipeline/aggregate_clip_cues.py / agreement_report.py).
"""
import csv
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from aggregate import build_clip_feature_row, load_frame_cues_by_clip  # noqa: E402

DATASET_ROOT = os.path.join(REPO_ROOT, "Data", "Dataset", "hri-multimodal-intent-v1.0.0")
CLIPS_CSV = os.path.join(DATASET_ROOT, "annotations", "clips.csv")
SCENARIOS_CSV = os.path.join(DATASET_ROOT, "annotations", "scenarios.csv")
SPLITS_CSV = os.path.join(DATASET_ROOT, "annotations", "splits.csv")
FEATURES_DIR = os.path.join(REPO_ROOT, "data", "features")
OUT_PATH = os.path.join(FEATURES_DIR, "clip_features.parquet")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    clips = read_csv(CLIPS_CSV)
    scenarios = {r["Scenario ID"]: r for r in read_csv(SCENARIOS_CSV)}
    splits = {r["clip_id"]: r for r in read_csv(SPLITS_CSV)}

    print("[build_features] loading per-frame cues (this reads all 4 *_frame_cues.jsonl files)...")
    frames_by_cue = {cue: load_frame_cues_by_clip(cue) for cue in ["emotion", "gesture", "motion", "context"]}

    rows = []
    skipped_no_scenario = 0
    skipped_no_split = 0
    for clip in clips:
        clip_id = clip["clip_id"]
        scenario_id = clip["scenario_id"]
        base_scenario = scenario_id.split("_")[0]
        scen_row = scenarios.get(base_scenario)
        if scen_row is None:
            skipped_no_scenario += 1
            continue
        split_row = splits.get(clip_id)
        if split_row is None:
            skipped_no_split += 1
            continue

        feat_row = build_clip_feature_row(
            clip_id,
            frames_by_cue["emotion"].get(clip_id, []),
            frames_by_cue["gesture"].get(clip_id, []),
            frames_by_cue["motion"].get(clip_id, []),
            frames_by_cue["context"].get(clip_id, []),
        )
        feat_row["scenario_id"] = scenario_id
        feat_row["subject_id"] = clip["subject_id"]
        feat_row["intent"] = scen_row["Intent"]
        feat_row["split_scenario"] = split_row["split_scenario"]
        feat_row["split_subject"] = split_row["split_subject"]
        feat_row["split_random_leaky_DO_NOT_USE_FOR_EVAL"] = split_row["split_random_leaky_DO_NOT_USE_FOR_EVAL"]
        rows.append(feat_row)

    if skipped_no_scenario:
        print(f"[build_features] WARNING: {skipped_no_scenario} clips skipped (no scenarios.csv match)")
    if skipped_no_split:
        print(f"[build_features] WARNING: {skipped_no_split} clips skipped (no splits.csv row -- run build_splits.py first)")

    df = pd.DataFrame(rows)
    assert df.isna().sum().sum() == 0, "NaNs present in feature matrix -- aggregation bug, see handover Phase 2 failure points"

    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[build_features] wrote {len(df)} clips x {len(df.columns)} columns -> {OUT_PATH}")
    print(f"[build_features] intent label distribution:\n{df['intent'].value_counts()}")


if __name__ == "__main__":
    main()
