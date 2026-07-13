"""
example_webcam.py

Minimal end-to-end integration example:
webcam → MediaPipe Pose → NTU joint conversion → MotionInference.

Prints one prediction line per frame once the 30-frame window is full.
This is deliberately UI-free — copy the loop body into your own
pipeline and replace the webcam with your camera source.

Usage:
    python example_webcam.py
"""

import cv2
import numpy as np
import mediapipe as mp

from inference import MotionInference
from skeleton_utils import mediapipe_to_ntu25

CHECKPOINT = "checkpoints/best_model_finetuned.pt"


def main():
    engine = MotionInference(CHECKPOINT)

    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        static_image_mode=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam — try VideoCapture(1).")
    cap.set(cv2.CAP_PROP_FPS, 30)  # model expects ~30 fps input

    print("Running. Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_world_landmarks is not None:
                joints_25 = mediapipe_to_ntu25(
                    results.pose_world_landmarks.landmark
                )
                result = engine.update(joints_25)
            else:
                # No person: feed zeros to keep the buffer advancing.
                # If the person is gone for long, call engine.reset()
                # instead so stale frames don't pollute the window.
                result = engine.update(np.zeros((25, 3), dtype=np.float32))

            if result.label != "buffering":
                print(f"{result.label:<14} conf={result.confidence:.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pose.close()


if __name__ == "__main__":
    main()
