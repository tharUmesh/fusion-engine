"""
Maps scenarios.csv's free-text "Intended <cue>" columns to each cue model's
own canonical label vocabulary (the same vocabulary the runners emit in
NormalisedFrameCue.label), so the agreement report compares like with like.

This mapping is a data-transcription concern only -- it does not touch any
cue-model decision logic, and it is NOT part of the NormalisedFrameCue
schema. See MODEL_ANALYSIS.md / Integration_API.md for the native label
vocabularies this maps into.

Stdlib-only, importable from any venv.
"""

# scenarios.csv "Intended Emotion" -> Emotion Repo's EMOTION_LABELS
EMOTION_MAP = {
    "angry": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
}

# scenarios.csv "Intended Gesture" -> gesture_runner's canonical vocabulary
# (GESTURE_SCENARIO_TO_CANONICAL's value set in runners/common/constants.py).
# "[MISSING]" is a deliberately-absent cue in the authored scenario (S08) --
# mapped to None, meaning "no intended gesture to compare against", not
# "the model should have detected nothing".
GESTURE_MAP = {
    "point / (writing)": "point",
    "raise hand": "raise_hand",
    "[missing]": None,
    "beckoning": "beckoning",
    "both hands up": "both_hands_up",
    "point": "point",
    "thumbs down": "thumbs_down",
    "thumbs up": "thumbs_up",
    "wave": "wave",
}

# scenarios.csv "Intended Motion" -> Motion Repo's MOTION_LABELS.
# "stepping back" has no dedicated class in the runtime 8-class taxonomy --
# action_recognizer.py's own authors merged "Walk Toward"/"Step Back" into a
# single "Walking" class (direction from a fixed camera was judged
# unreliable, see MODEL_ANALYSIS.md #3.3) -- so "stepping back" maps to
# "Walking" here too, consistently with that merge decision.
MOTION_MAP = {
    "move backward (run)": "Run Backward",
    "sitting": "Sitting Still",
    "stand": "Standing Still",
    "stepping back": "Walking",
    "walk": "Walking",
    "walk (toward)": "Walking",
    "walking": "Walking",
}

# scenarios.csv "Context"/"Intended Context" -> Context Repo's SCENE_LABELS
# (already matching directly, kept as an explicit map for consistency/audit).
CONTEXT_MAP = {
    "classroom": "classroom",
    "kitchen": "kitchen",
}


def map_intended(cue: str, raw_value: str):
    """Returns the canonical label for a cue's intended value, or None if
    the value is missing/unmapped (deliberately-missing cues, blank cells)."""
    if raw_value is None:
        return None
    key = raw_value.strip().lower()
    if key in ("", "[missing]"):
        return None
    table = {
        "emotion": EMOTION_MAP,
        "gesture": GESTURE_MAP,
        "motion": MOTION_MAP,
        "context": CONTEXT_MAP,
    }[cue]
    mapped = table.get(key)
    if mapped is None and key not in ("[missing]",):
        raise KeyError(f"No canonical mapping for {cue}='{raw_value}' -- add it to canonical_map.py")
    return mapped
