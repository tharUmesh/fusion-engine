"""
Phase 3 rule-based baseline -- PROVISIONAL (emotion + gesture cue models are
expected to be replaced, same as motion already was; this baseline and its
accuracy number are a moving target, not a final result).

Encodes scenarios.csv's own authoring logic as explicit priority-ordered
IF-THEN rules, derived by inspecting all 22 base scenarios' (Intended
Emotion, Intended Gesture, Intended Motion, Context) -> Intent mapping
directly (see table below). This is what fusion/gbt.py must beat.

Two IRREDUCIBLE ambiguities were found in the authored table itself while
deriving these rules -- flagged here, not hidden, because they cap this
baseline's (and arguably any 4-cue fusion model's) achievable accuracy:

  1. F02 vs F07: S05 (classroom, Anger, both_hands_up, standing) = F02, but
     S24 (kitchen, Anger, both_hands_up, standing) = F07 -- the IDENTICAL
     emotion/gesture/motion combination maps to two different intents,
     distinguished ONLY by scene. Handled below with a narrow kitchen+Anger
     carve-out checked before the general F02 (emergency) catch-all -- per
     the handover doc's "asymmetric cost: any meaningful evidence of F02
     escalates" instruction, the carve-out is intentionally narrow so most
     both_hands_up detections still escalate to F02.
  2. F04 vs F10: S21 (kitchen, Sad, thumbs_down, sitting) = F04, but S28
     (kitchen, Sad, thumbs_down, sitting) = F10 -- SAME combination, scene
     included, maps to two different intents with NO distinguishing signal
     anywhere in the 4 measured cues. This one is not resolvable by any
     rule over this feature set; the tie-break below (-> F04, the
     majority of the two) is arbitrary and will misclassify every true F10
     case. This is a genuine ceiling on rule-based (and cue-fusion)
     accuracy, not a bug to fix here -- see MODEL_ANALYSIS.md /
     HRI_Fusion_Engine_Handover.md's own note that activity/engagement/
     object-target are permanently unmeasured and might be what actually
     disambiguates this pair.

Source mapping (base scenario -> Intent, from scenarios.csv):
  F01: S02(happy,wave,walk toward,classroom) S12(happy,thumbs_up,sitting,classroom) S18(happy,thumbs_up,standing,kitchen)
  F02: S05(angry,both_hands_up,standing,classroom) S09(surprise,both_hands_up,standing,classroom)
       S19(fear,both_hands_up,stepping_back*,kitchen) S26(surprise,both_hands_up,stepping_back,kitchen)
       (*S19's "Move backward (run)" has no equivalent in the new 4-class motion taxonomy -- see
       pipeline/canonical_map.py's note; not motion-gated here, gesture alone is enough for F02)
  F03: S07(neutral,beckoning,sitting,classroom) S20(neutral,beckoning,walking,kitchen) S29(neutral,point,walking,kitchen)
  F04: S01(neutral,raise_hand,sitting,classroom) S04(sad,thumbs_down,sitting,classroom) S21(sad,thumbs_down,sitting,kitchen)
  F05: S03(neutral,point,sitting,classroom) S11(happy,raise_hand,sitting,classroom) S22(neutral,point,standing,kitchen)
  F06: S08(neutral,[MISSING],walking,classroom) S27(disgust,point,stepping_back,kitchen)
  F07: S24(angry,both_hands_up,standing,kitchen) -- see ambiguity #1 above
  F08: S06(disgust,thumbs_down,stepping_back,classroom) S23(disgust,thumbs_down,stepping_back,kitchen)
  F09: S25(happy,wave,walking,kitchen)
  F10: S28(sad,thumbs_down,sitting,kitchen) -- see ambiguity #2 above

Reads a clip's Phase 2 feature row (pipeline/aggregate.py's FEATURE_NAMES)
and returns a predicted intent code. Priority order below is emergency
(F02) first, then most-to-least specific pattern, ending in a corpus-mode
fallback for anything unmatched.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from aggregate import EMOTION_CLASSES, GESTURE_CLASSES, MOTION_CLASSES, CONTEXT_CLASSES  # noqa: E402

DEFAULT_FALLBACK_INTENT = "F05"  # overridden by fit_fallback() with the training corpus's mode


def _dominant(row, prefix, classes):
    """Reads the one-hot/mean-prob block back out as a single dominant label
    (argmax), or None if the cue is flagged missing for this clip."""
    if row.get(f"missing_{prefix}", 0.0) >= 1.0:
        return None
    vals = [row[f"{prefix}_{c}"] for c in classes]
    if max(vals) <= 0.0:
        return None
    return classes[int(np.argmax(vals))]


def predict_intent(row, fallback_intent=DEFAULT_FALLBACK_INTENT):
    emotion = _dominant(row, "emotion", EMOTION_CLASSES)
    gesture = _dominant(row, "gesture", GESTURE_CLASSES)
    motion = _dominant(row, "motion", MOTION_CLASSES)
    context = _dominant(row, "context", CONTEXT_CLASSES)

    # 1. Emergency escalation -- both_hands_up almost always means F02.
    #    Narrow authored exception: kitchen + Anger + standing -> F07.
    if gesture == "both_hands_up":
        if context == "kitchen" and emotion == "Anger" and motion == "standing":
            return "F07"
        return "F02"

    # 2. F08: disgust + thumbs_down + stepping_back (consistent both scenes).
    if gesture == "thumbs_down" and motion == "stepping_back" and emotion == "Disgust":
        return "F08"

    # 3. F04/F10 collision (see module docstring, ambiguity #2) -- sitting +
    #    (raise_hand or thumbs_down) always predicted as F04; true F10
    #    clips are unresolvable with these cues and will be misclassified.
    if gesture in ("raise_hand", "thumbs_down") and motion == "sitting":
        return "F04"

    # 4. F05: point or raise_hand while NOT walking (sitting/standing).
    if gesture in ("point", "raise_hand") and motion in ("sitting", "standing"):
        return "F05"

    # 5. F03: beckoning (any motion), or point while walking.
    if gesture == "beckoning":
        return "F03"
    if gesture == "point" and motion == "walking":
        return "F03"

    # 6. F01/F09: happy + (wave or thumbs_up). Scene disambiguates, per
    #    ambiguity #1's sibling pattern in the table (kitchen+walking -> F09).
    if emotion == "Happy" and gesture in ("wave", "thumbs_up"):
        if context == "kitchen" and motion == "walking":
            return "F09"
        return "F01"

    # 7. F06: point + stepping_back (S27's pattern). S08's own [MISSING]
    #    intended gesture has no measurable equivalent, so that half of
    #    F06's authored pattern is unreachable from measured cues alone.
    if gesture == "point" and motion == "stepping_back":
        return "F06"

    return fallback_intent


def fit_fallback(train_df):
    """Corpus-mode fallback for clips matching none of the above rules."""
    return train_df["intent"].mode().iloc[0]


def predict_all(df, fallback_intent=DEFAULT_FALLBACK_INTENT):
    return df.apply(lambda row: predict_intent(row, fallback_intent), axis=1)


if __name__ == "__main__":
    FEATURES_PATH = os.path.join(REPO_ROOT, "data", "features", "clip_features.parquet")
    df = pd.read_parquet(FEATURES_PATH)
    train_df = df[df["split_scenario"] == "train"]
    fallback = fit_fallback(train_df)
    print(f"[rule_based] fallback intent (train-set mode): {fallback}")

    preds = predict_all(df, fallback_intent=fallback)
    df["rule_pred"] = preds
    overall_acc = (df["rule_pred"] == df["intent"]).mean()
    print(f"[rule_based] overall accuracy (all {len(df)} clips, includes train -- not a test number): {overall_acc:.3f}")

    for split_name in ["train", "val", "test"]:
        sub = df[df["split_scenario"] == split_name]
        if len(sub) == 0:
            continue
        acc = (sub["rule_pred"] == sub["intent"]).mean()
        print(f"[rule_based] split_scenario={split_name}: n={len(sub)}, accuracy={acc:.3f}")

    print("\n[rule_based] per-class recall (split_scenario=test):")
    test_df = df[df["split_scenario"] == "test"]
    for cls in sorted(df["intent"].unique()):
        sub = test_df[test_df["intent"] == cls]
        if len(sub) == 0:
            continue
        recall = (sub["rule_pred"] == cls).mean()
        print(f"  {cls}: n={len(sub)}, recall={recall:.3f}")
