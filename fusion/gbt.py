"""
Phase 4 GBT fusion -- PROVISIONAL (emotion + gesture cue models are expected
to be replaced next, same as motion already was; this model and its
accuracy number are a moving baseline, not the locked deliverable yet).

LightGBM multiclass classifier over the Phase-2 feature vector
(pipeline/aggregate.py's 33 cue-derived columns), trained/evaluated on
splits.csv's split_scenario partition (grouped by scenario, so variations of
one scenario never straddle train/test).

Implements, per the handover doc's Phase 4 spec:
  - Class weighting (`class_weight="balanced"`) -- F02 is never down-weighted
    by construction (balanced weighting up-weights rarer classes, and F02 is
    one of the more common ones here, so this does not suppress it either).
  - Modality-dropout augmentation during training: for each training row,
    with probability DROPOUT_P, zero one cue's block and set its missing bit
    -- teaches the model to redistribute weight onto the remaining cues
    instead of only ever seeing this dataset's near-total absence of real
    missingness.
  - Safety override: if the model's predicted F02 probability exceeds
    F02_SAFETY_THRESHOLD, classify as F02 regardless of argmax.

NOT yet implemented in this provisional pass (explicitly deferred, not
silently skipped): isotonic/Platt calibration, SHAP per-prediction
attribution. Both are meaningful follow-up work once the emotion/gesture
cue models stop moving.
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from aggregate import FEATURE_NAMES  # noqa: E402
sys.path.insert(0, os.path.join(REPO_ROOT, "fusion"))
from rule_based import predict_all, fit_fallback  # noqa: E402

FEATURES_PATH = os.path.join(REPO_ROOT, "data", "features", "clip_features.parquet")

CUE_BLOCKS = {
    "emotion": [c for c in FEATURE_NAMES if c.startswith("emotion_")],
    "gesture": [c for c in FEATURE_NAMES if c.startswith("gesture_")],
    "motion": [c for c in FEATURE_NAMES if c.startswith("motion_")],
    "context": [c for c in FEATURE_NAMES if c.startswith("context_")],
}
DROPOUT_P = 0.15  # per-cue, per-row probability of simulated dropout during training
F02_SAFETY_THRESHOLD = 0.15
RANDOM_SEED = 42


def apply_modality_dropout(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    X = X.copy()
    for cue, cols in CUE_BLOCKS.items():
        drop_mask = rng.random(len(X)) < DROPOUT_P
        if not drop_mask.any():
            continue
        value_cols = [c for c in cols if not c.startswith(f"{cue}_valid_fraction")]
        X.loc[drop_mask, value_cols] = 0.0
        if f"{cue}_valid_fraction" in cols:
            X.loc[drop_mask, f"{cue}_valid_fraction"] = 0.0
        X.loc[drop_mask, f"missing_{cue}"] = 1.0
    return X


def predict_with_safety_override(model, X, f02_idx):
    proba = model.predict_proba(X)
    argmax_idx = proba.argmax(axis=1)
    preds = model.classes_[argmax_idx]
    escalate = proba[:, f02_idx] >= F02_SAFETY_THRESHOLD
    preds = np.where(escalate, "F02", preds)
    return preds, proba


def main():
    df = pd.read_parquet(FEATURES_PATH)
    train_df = df[df["split_scenario"] == "train"].reset_index(drop=True)
    val_df = df[df["split_scenario"] == "val"].reset_index(drop=True)
    test_df = df[df["split_scenario"] == "test"].reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_SEED)
    X_train = apply_modality_dropout(train_df[FEATURE_NAMES], rng)
    y_train = train_df["intent"]

    model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    f02_idx = list(model.classes_).index("F02")

    print(f"[gbt] trained on {len(train_df)} clips, {len(FEATURE_NAMES)} features, "
          f"modality dropout p={DROPOUT_P}, F02 safety threshold={F02_SAFETY_THRESHOLD}")

    fallback = fit_fallback(train_df)
    rule_preds_all = predict_all(df, fallback_intent=fallback)
    df["rule_pred"] = rule_preds_all

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            continue
        X = split_df[FEATURE_NAMES]
        preds, proba = predict_with_safety_override(model, X, f02_idx)
        acc = (preds == split_df["intent"].values).mean()

        rule_sub = df[df["clip_id"].isin(split_df["clip_id"])]
        rule_acc = (rule_sub["rule_pred"] == rule_sub["intent"]).mean()

        print(f"\n[gbt] split_scenario={split_name}: n={len(split_df)}")
        print(f"  GBT accuracy:  {acc:.3f}")
        print(f"  rule accuracy: {rule_acc:.3f}  (same clips, for direct comparison)")

        if split_name == "test":
            print("\n[gbt] per-class recall (test):")
            for cls in sorted(df["intent"].unique()):
                mask = split_df["intent"].values == cls
                n = mask.sum()
                if n == 0:
                    continue
                recall = (preds[mask] == cls).mean()
                print(f"  {cls}: n={n}, recall={recall:.3f}")

            f02_mask = split_df["intent"].values == "F02"
            f02_recall = (preds[f02_mask] == "F02").mean() if f02_mask.sum() else float("nan")
            f02_false_neg = int((f02_mask & (preds != "F02")).sum())
            print(f"\n[gbt] F02 test recall: {f02_recall:.3f} ({f02_false_neg} false negatives / {f02_mask.sum()} true F02 clips)")

    print("\n[gbt] feature importances (top 10, gain):")
    importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print(importances.head(10))


if __name__ == "__main__":
    main()
