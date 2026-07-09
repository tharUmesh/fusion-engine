"""
Confidence floors used by all four runners to compute NormalisedFrameCue.valid.
Values match the handover document's configs/schema.yaml plan (section 3).
Stdlib-only — see schema.py for why.
"""

CONFIDENCE_FLOOR = {
    "emotion": 0.50,
    "gesture": 0.80,
    "motion": 0.50,
    "context": 0.50,
}

# Canonical gesture vocabulary (schema.yaml's gesture_classes), used by
# gesture_runner.py to map the native scenario-resolver strings.
GESTURE_SCENARIO_TO_CANONICAL = {
    "Wave": "wave",
    "Brief wave": "wave",
    "Arms waving": "wave",
    "Pointing": "point",
    "Thumbs up": "thumbs_up",
    "Thumbs down": "thumbs_down",
    "One hand raised": "raise_hand",
    "Arms up": "both_hands_up",
    "Beckoning": "beckoning",
    "None": "Unknown",
}
