"""
skeleton_utils.py

Converts MediaPipe Pose *world* landmarks into the (25, 3) NTU-layout
joint array expected by MotionInference.update().

This conversion MUST be applied — it is part of the model's input
contract. It was extracted verbatim from the original project's
webcam_demo.py (the reference integration).

Three things happen here:
  1. MediaPipe landmark indices are remapped to NTU joint indices.
  2. X and Y are negated: MediaPipe world landmarks use X-right/Y-down,
     while the Kinect/NTU convention used during training is
     X-left/Y-up. Negating both is a 180° rotation about Z, so it
     corrects the axes without mirroring the skeleton.
  3. The three spine joints NTU has but MediaPipe lacks are
     approximated from shoulder/hip midpoints.
"""

import numpy as np

# MediaPipe landmark index → NTU joint index
MP_TO_NTU = {
    0:  3,   # nose        → head
    11: 4,   # l_shoulder  → left_shoulder
    12: 8,   # r_shoulder  → right_shoulder
    13: 5,   # l_elbow     → left_elbow
    14: 9,   # r_elbow     → right_elbow
    15: 6,   # l_wrist     → left_wrist
    16: 10,  # r_wrist     → right_wrist
    23: 12,  # l_hip       → left_hip
    24: 16,  # r_hip       → right_hip
    25: 13,  # l_knee      → left_knee
    26: 17,  # r_knee      → right_knee
    27: 14,  # l_ankle     → left_ankle
    28: 18,  # r_ankle     → right_ankle
}


def mediapipe_to_ntu25(world_landmarks) -> np.ndarray:
    """
    world_landmarks: results.pose_world_landmarks.landmark from
                     MediaPipe Pose (33 landmarks, metres).

    Returns a (25, 3) float32 array in NTU joint layout, ready to pass
    to MotionInference.update().
    """
    joints_25 = np.zeros((25, 3), dtype=np.float32)

    for mp_idx, ntu_idx in MP_TO_NTU.items():
        lm = world_landmarks[mp_idx]
        joints_25[ntu_idx] = [-lm.x, -lm.y, lm.z]

    # Approximate spine joints from shoulder/hip midpoints
    joints_25[0] = (joints_25[12] + joints_25[16]) / 2       # base_spine
    joints_25[1] = (joints_25[4]  + joints_25[8])  / 2       # mid_spine
    joints_25[2] = joints_25[1] * 0.5 + joints_25[3] * 0.5   # neck

    return joints_25
