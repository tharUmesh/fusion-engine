"""
=======================================================================
HRI Motion Dataset Generator v2
=======================================================================
Generates synthetic MediaPipe-format skeleton sequences for 9 motion
classes derived from the HRI_Dataset_Table scenarios.

Classes:
  0: Sitting Still        – Seated, minimal movement
  1: Standing Still       – Standing, stationary
  2: Walk Toward          – Normal pace approach (Z increases)
  3: Step/Walk Back       – Stepping or walking backward (Z decreases)
  4: Walk Across          – Lateral walking movement (X changes)
  5: Run Backward         – Fast backward retreat
  6: Run (Fast Movement)  – Fast running in any direction
  7: Leaning Forward      – Upper body lean toward camera
  8: Frozen/Rigid Stand   – Standing completely still (freeze response)

Output:
  extracted_keypoints_v2/<class_id>_<class_name>/synthetic_*.npy
  Each .npy file is shape (30, 33, 3)

Usage:
  python 1_prepare_dataset_v2.py
"""

import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 200
FRAMES_PER_SEQUENCE = 30
NUM_KEYPOINTS = 33

MOTION_LABELS = [
    "Sitting Still",        # 0
    "Standing Still",       # 1
    "Walk Toward",          # 2
    "Step/Walk Back",       # 3
    "Walk Across",          # 4
    "Run Backward",         # 5
    "Run (Fast Movement)",  # 6
    "Leaning Forward",      # 7
    "Frozen/Rigid Stand",   # 8
]

NUM_CLASSES = len(MOTION_LABELS)

KEYPOINTS_DIR = Path("extracted_keypoints_v2")
KEYPOINTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# SKELETON TEMPLATES (MediaPipe 33 keypoints, normalized 0-1)
# ──────────────────────────────────────────────────────────────────────

# Standing person template
STANDING_POSE = np.array([
    [0.50, 0.20, 0.00],   # 0:  Nose
    [0.48, 0.18, 0.00],   # 1:  Left Eye
    [0.52, 0.18, 0.00],   # 2:  Right Eye
    [0.46, 0.17, 0.00],   # 3:  Left Ear
    [0.54, 0.17, 0.00],   # 4:  Right Ear
    [0.00, 0.00, 0.00],   # 5:  (unused)
    [0.00, 0.00, 0.00],   # 6:  (unused)
    [0.00, 0.00, 0.00],   # 7:  (unused)
    [0.00, 0.00, 0.00],   # 8:  (unused)
    [0.48, 0.22, 0.00],   # 9:  Mouth left
    [0.52, 0.22, 0.00],   # 10: Mouth right
    [0.45, 0.35, 0.00],   # 11: Left Shoulder
    [0.55, 0.35, 0.00],   # 12: Right Shoulder
    [0.42, 0.50, 0.00],   # 13: Left Elbow
    [0.58, 0.50, 0.00],   # 14: Right Elbow
    [0.40, 0.65, 0.00],   # 15: Left Wrist
    [0.60, 0.65, 0.00],   # 16: Right Wrist
    [0.00, 0.00, 0.00],   # 17: (unused)
    [0.00, 0.00, 0.00],   # 18: (unused)
    [0.00, 0.00, 0.00],   # 19: (unused)
    [0.00, 0.00, 0.00],   # 20: (unused)
    [0.00, 0.00, 0.00],   # 21: (unused)
    [0.00, 0.00, 0.00],   # 22: (unused)
    [0.46, 0.60, 0.00],   # 23: Left Hip
    [0.54, 0.60, 0.00],   # 24: Right Hip
    [0.46, 0.75, 0.00],   # 25: Left Knee
    [0.54, 0.75, 0.00],   # 26: Right Knee
    [0.46, 0.90, 0.00],   # 27: Left Ankle
    [0.54, 0.90, 0.00],   # 28: Right Ankle
    [0.00, 0.00, 0.00],   # 29: (unused)
    [0.00, 0.00, 0.00],   # 30: (unused)
    [0.00, 0.00, 0.00],   # 31: (unused)
    [0.00, 0.00, 0.00],   # 32: (unused)
])

# Sitting person template (hips lowered, knees bent forward)
SITTING_POSE = np.array([
    [0.50, 0.15, 0.00],   # 0:  Nose (higher relative to body)
    [0.48, 0.13, 0.00],   # 1:  Left Eye
    [0.52, 0.13, 0.00],   # 2:  Right Eye
    [0.46, 0.12, 0.00],   # 3:  Left Ear
    [0.54, 0.12, 0.00],   # 4:  Right Ear
    [0.00, 0.00, 0.00],   # 5:  (unused)
    [0.00, 0.00, 0.00],   # 6:  (unused)
    [0.00, 0.00, 0.00],   # 7:  (unused)
    [0.00, 0.00, 0.00],   # 8:  (unused)
    [0.48, 0.17, 0.00],   # 9:  Mouth left
    [0.52, 0.17, 0.00],   # 10: Mouth right
    [0.45, 0.28, 0.00],   # 11: Left Shoulder
    [0.55, 0.28, 0.00],   # 12: Right Shoulder
    [0.42, 0.42, 0.00],   # 13: Left Elbow
    [0.58, 0.42, 0.00],   # 14: Right Elbow
    [0.42, 0.55, 0.00],   # 15: Left Wrist (resting on lap)
    [0.58, 0.55, 0.00],   # 16: Right Wrist
    [0.00, 0.00, 0.00],   # 17: (unused)
    [0.00, 0.00, 0.00],   # 18: (unused)
    [0.00, 0.00, 0.00],   # 19: (unused)
    [0.00, 0.00, 0.00],   # 20: (unused)
    [0.00, 0.00, 0.00],   # 21: (unused)
    [0.00, 0.00, 0.00],   # 22: (unused)
    [0.46, 0.55, 0.00],   # 23: Left Hip
    [0.54, 0.55, 0.00],   # 24: Right Hip
    [0.44, 0.58, 0.05],   # 25: Left Knee (bent forward)
    [0.56, 0.58, 0.05],   # 26: Right Knee
    [0.44, 0.75, 0.00],   # 27: Left Ankle (under chair)
    [0.56, 0.75, 0.00],   # 28: Right Ankle
    [0.00, 0.00, 0.00],   # 29: (unused)
    [0.00, 0.00, 0.00],   # 30: (unused)
    [0.00, 0.00, 0.00],   # 31: (unused)
    [0.00, 0.00, 0.00],   # 32: (unused)
])

# Leaning forward template (upper body tilted)
LEANING_POSE = np.array([
    [0.50, 0.22, 0.08],   # 0:  Nose (shifted forward in Z)
    [0.48, 0.20, 0.07],   # 1:  Left Eye
    [0.52, 0.20, 0.07],   # 2:  Right Eye
    [0.46, 0.19, 0.06],   # 3:  Left Ear
    [0.54, 0.19, 0.06],   # 4:  Right Ear
    [0.00, 0.00, 0.00],   # 5:  (unused)
    [0.00, 0.00, 0.00],   # 6:  (unused)
    [0.00, 0.00, 0.00],   # 7:  (unused)
    [0.00, 0.00, 0.00],   # 8:  (unused)
    [0.48, 0.24, 0.07],   # 9:  Mouth left
    [0.52, 0.24, 0.07],   # 10: Mouth right
    [0.45, 0.37, 0.04],   # 11: Left Shoulder
    [0.55, 0.37, 0.04],   # 12: Right Shoulder
    [0.42, 0.52, 0.02],   # 13: Left Elbow
    [0.58, 0.52, 0.02],   # 14: Right Elbow
    [0.40, 0.65, 0.00],   # 15: Left Wrist
    [0.60, 0.65, 0.00],   # 16: Right Wrist
    [0.00, 0.00, 0.00],   # 17: (unused)
    [0.00, 0.00, 0.00],   # 18: (unused)
    [0.00, 0.00, 0.00],   # 19: (unused)
    [0.00, 0.00, 0.00],   # 20: (unused)
    [0.00, 0.00, 0.00],   # 21: (unused)
    [0.00, 0.00, 0.00],   # 22: (unused)
    [0.46, 0.60, 0.00],   # 23: Left Hip
    [0.54, 0.60, 0.00],   # 24: Right Hip
    [0.46, 0.75, 0.00],   # 25: Left Knee
    [0.54, 0.75, 0.00],   # 26: Right Knee
    [0.46, 0.90, 0.00],   # 27: Left Ankle
    [0.54, 0.90, 0.00],   # 28: Right Ankle
    [0.00, 0.00, 0.00],   # 29: (unused)
    [0.00, 0.00, 0.00],   # 30: (unused)
    [0.00, 0.00, 0.00],   # 31: (unused)
    [0.00, 0.00, 0.00],   # 32: (unused)
])


# ──────────────────────────────────────────────────────────────────────
# AUGMENTATION HELPERS
# ──────────────────────────────────────────────────────────────────────
def add_noise(keypoints, noise_level=0.002):
    """Add Gaussian jitter to keypoints."""
    noise = np.random.normal(0, noise_level, keypoints.shape)
    return np.clip(keypoints + noise, -0.2, 1.2)

def random_start_offset():
    """Random X/Y offset for skeleton starting position."""
    return np.array([
        np.random.uniform(-0.1, 0.1),   # X offset
        np.random.uniform(-0.05, 0.05), # Y offset
        0.0                             # Z stays
    ])

def random_scale():
    """Random scale factor to simulate distance variation."""
    return np.random.uniform(0.85, 1.15)

def random_speed_factor(base_low=0.7, base_high=1.3):
    """Random speed multiplier within a class."""
    return np.random.uniform(base_low, base_high)

def walking_bobbing(t, frequency, amplitude):
    """Simulate vertical bobbing during walking gait."""
    return np.sin(t * frequency * np.pi) * amplitude

def apply_augmentations(template, offset=True, scale=True):
    """Apply random starting position and scale augmentation."""
    kp = template.copy()
    if scale:
        s = random_scale()
        center = np.array([0.5, 0.5, 0.0])
        kp = center + (kp - center) * s
    if offset:
        off = random_start_offset()
        # Only apply offset to non-zero keypoints
        mask = np.any(kp != 0, axis=1)
        kp[mask] += off
    return kp


# ──────────────────────────────────────────────────────────────────────
# MOTION GENERATORS
# ──────────────────────────────────────────────────────────────────────

def generate_sitting_still(n=SAMPLES_PER_CLASS):
    """0: Sitting Still - Seated person with distinct micro-movements (breathing, fidgeting)."""
    sequences = []
    for _ in range(n):
        base = apply_augmentations(SITTING_POSE)
        # Higher noise than frozen stand to create distinguishable velocity signal
        noise_level = np.random.uniform(0.003, 0.007)
        # Random oscillation frequencies for variation between samples
        breath_freq = np.random.uniform(1.5, 3.0)
        fidget_freq = np.random.uniform(0.8, 2.5)
        head_nod_amp = np.random.uniform(0.004, 0.010)
        wrist_fidget_amp = np.random.uniform(0.010, 0.022)
        breath_amp = np.random.uniform(0.006, 0.014)
        seq = []
        for f in range(FRAMES_PER_SEQUENCE):
            t = f / FRAMES_PER_SEQUENCE
            kp = base.copy()
            # Breathing: shoulders rise and fall
            breath = np.sin(t * breath_freq * 2 * np.pi) * breath_amp
            kp[11, 1] += breath   # Left shoulder
            kp[12, 1] += breath   # Right shoulder
            # Head nodding (reading/looking at desk)
            kp[0, 1] += np.sin(t * fidget_freq * np.pi) * head_nod_amp
            kp[0, 2] += np.cos(t * fidget_freq * np.pi) * head_nod_amp * 0.5
            # Wrist fidgeting (typing, writing on lap)
            kp[15, 0] += np.sin(t * fidget_freq * 2 * np.pi) * wrist_fidget_amp
            kp[15, 1] += np.cos(t * fidget_freq * 2 * np.pi) * wrist_fidget_amp * 0.5
            kp[16, 0] -= np.sin(t * fidget_freq * 2 * np.pi) * wrist_fidget_amp
            kp[16, 1] += np.cos(t * fidget_freq * 2 * np.pi) * wrist_fidget_amp * 0.5
            kp = add_noise(kp, noise_level=noise_level)
            seq.append(kp)
        sequences.append(np.array(seq))
    return sequences


def generate_standing_still(n=SAMPLES_PER_CLASS):
    """1: Standing Still - Standing person with natural micro-movements."""
    sequences = []
    for _ in range(n):
        base = apply_augmentations(STANDING_POSE)
        noise_level = np.random.uniform(0.001, 0.003)
        seq = []
        for f in range(FRAMES_PER_SEQUENCE):
            t = f / FRAMES_PER_SEQUENCE
            kp = base.copy()
            # Weight shift (slight X sway)
            kp[:, 0] += np.sin(t * 2 * np.pi) * np.random.uniform(0.005, 0.015)
            # Breathing
            kp[11, 1] += np.sin(t * 3 * np.pi) * 0.004
            kp[12, 1] += np.sin(t * 3 * np.pi) * 0.004
            kp = add_noise(kp, noise_level=noise_level)
            seq.append(kp)
        sequences.append(np.array(seq))
    return sequences


def generate_walk_toward(n=SAMPLES_PER_CLASS):
    """2: Walk Toward - Normal pace approach (Z increases)."""
    sequences = []
    for _ in range(n):
        base = apply_augmentations(STANDING_POSE)
        speed = random_speed_factor(0.7, 1.3)
        z_range = np.random.uniform(0.15, 0.35) * speed
        bob_amp = np.random.uniform(0.015, 0.03)
        bob_freq = np.random.uniform(3.0, 5.0)
        noise_level = np.random.uniform(0.002, 0.004)
        seq = []
        for f in range(FRAMES_PER_SEQUENCE):
            t = f / FRAMES_PER_SEQUENCE
            kp = base.copy()
            # Move toward camera (Z increases)
            kp[:, 2] += t * z_range
            # Walking bobbing
            kp[:, 1] += walking_bobbing(t, bob_freq, bob_amp)
            # Slight arm swing
            kp[15, 1] += np.sin(t * bob_freq * np.pi) * 0.02  # Left wrist
            kp[16, 1] -= np.sin(t * bob_freq * np.pi) * 0.02  # Right wrist
            kp = add_noise(kp, noise_level=noise_level)
            seq.append(kp)
        sequences.append(np.array(seq))
    return sequences


def generate_step_walk_back(n=SAMPLES_PER_CLASS):
    """3: Step/Walk Back - Stepping or walking backward (Z decreases)."""
    sequences = []
    for _ in range(n):
        base = apply_augmentations(STANDING_POSE)
        speed = random_speed_factor(0.6, 1.2)
        z_range = np.random.uniform(0.12, 0.30) * speed
        bob_amp = np.random.uniform(0.01, 0.025)
        bob_freq = np.random.uniform(2.5, 4.5)
        noise_level = np.random.uniform(0.002, 0.004)
        seq = []
        for f in range(FRAMES_PER_SEQUENCE):
            t = f / FRAMES_PER_SEQUENCE
            kp = base.copy()
            # Move away from camera (Z decreases)
            kp[:, 2] -= t * z_range
            # Walking bobbing
            kp[:, 1] += walking_bobbing(t, bob_freq, bob_amp)
            # Arm swing (backwards gait)
            kp[15, 1] -= np.sin(t * bob_freq * np.pi) * 0.015
            kp[16, 1] += np.sin(t * bob_freq * np.pi) * 0.015
            kp = add_noise(kp, noise_level=noise_level)
            seq.append(kp)
        sequences.append(np.array(seq))
    return sequences


def generate_walk_across(n=SAMPLES_PER_CLASS):
    """4: Walk Across - Lateral walking movement (X changes)."""
    sequences = []
    for _ in range(n):
        base = apply_augmentations(STANDING_POSE)
        speed = random_speed_factor(0.7, 1.3)
        x_range = np.random.uniform(0.2, 0.5) * speed
        # Random direction: left or right
        direction = np.random.choice([-1, 1])
        bob_amp = np.random.uniform(0.015, 0.03)
        bob_freq = np.random.uniform(3.0, 5.0)
        noise_level = np.random.uniform(0.002, 0.004)
        seq = []
        for f in range(FRAMES_PER_SEQUENCE):
            t = f / FRAMES_PER_SEQUENCE
            kp = base.copy()
            # Lateral movement
            kp[:, 0] += t * x_range * direction
            # Walking bobbing
            kp[:, 1] += walking_bobbing(t, bob_freq, bob_amp)
            # Arm swing
            kp[15, 0] += np.sin(t * bob_freq * np.pi) * 0.02 * direction
            kp[16, 0] -= np.sin(t * bob_freq * np.pi) * 0.02 * direction
            kp = add_noise(kp, noise_level=noise_level)
            seq.append(kp)
        sequences.append(np.array(seq))
    return sequences


def generate_run_backward(n=SAMPLES_PER_CLASS):
    """5: Run Backward - Fast backward retreat (Z decreases rapidly)."""
    sequences = []
    for _ in range(n):
        base = apply_augmentations(STANDING_POSE)
        speed = random_speed_factor(0.8, 1.4)
        z_range = np.random.uniform(0.35, 0.60) * speed
        bob_amp = np.random.uniform(0.03, 0.05)
        bob_freq = np.random.uniform(5.0, 7.0)
        noise_level = np.random.uniform(0.003, 0.006)
        seq = []
        for f in range(FRAMES_PER_SEQUENCE):
            t = f / FRAMES_PER_SEQUENCE
            kp = base.copy()
            # Fast Z decrease (running away)
            kp[:, 2] -= t * z_range
            # Strong bobbing (running gait)
            kp[:, 1] += walking_bobbing(t, bob_freq, bob_amp)
            # Large arm swing
            kp[15, 1] -= np.sin(t * bob_freq * np.pi) * 0.04
            kp[16, 1] += np.sin(t * bob_freq * np.pi) * 0.04
            # Slight lateral sway
            kp[:, 0] += np.sin(t * 3 * np.pi) * 0.01
            kp = add_noise(kp, noise_level=noise_level)
            seq.append(kp)
        sequences.append(np.array(seq))
    return sequences


def generate_run_fast(n=SAMPLES_PER_CLASS):
    """6: Run (Fast Movement) - Fast running, mixed directions."""
    sequences = []
    for _ in range(n):
        base = apply_augmentations(STANDING_POSE)
        speed = random_speed_factor(0.8, 1.5)
        # Random primary direction bias
        x_bias = np.random.uniform(-0.4, 0.4)
        z_bias = np.random.uniform(-0.3, 0.3)
        total_move = np.sqrt(x_bias**2 + z_bias**2)
        if total_move < 0.3:
            # Ensure minimum movement speed for "running"
            scale_up = 0.3 / max(total_move, 0.01)
            x_bias *= scale_up
            z_bias *= scale_up

        bob_amp = np.random.uniform(0.03, 0.06)
        bob_freq = np.random.uniform(5.0, 8.0)
        noise_level = np.random.uniform(0.004, 0.007)
        seq = []
        for f in range(FRAMES_PER_SEQUENCE):
            t = f / FRAMES_PER_SEQUENCE
            kp = base.copy()
            # Fast movement in mixed direction
            kp[:, 0] += t * x_bias * speed
            kp[:, 2] += t * z_bias * speed
            # Heavy bobbing
            kp[:, 1] += walking_bobbing(t, bob_freq, bob_amp)
            # Large arm swing
            kp[15, 1] += np.sin(t * bob_freq * np.pi) * 0.05
            kp[16, 1] -= np.sin(t * bob_freq * np.pi) * 0.05
            kp[13, 1] += np.sin(t * bob_freq * np.pi) * 0.03
            kp[14, 1] -= np.sin(t * bob_freq * np.pi) * 0.03
            kp = add_noise(kp, noise_level=noise_level)
            seq.append(kp)
        sequences.append(np.array(seq))
    return sequences


def generate_leaning_forward(n=SAMPLES_PER_CLASS):
    """7: Leaning Forward - Upper body leans toward camera progressively."""
    sequences = []
    for _ in range(n):
        base = apply_augmentations(STANDING_POSE)
        lean_amount = np.random.uniform(0.04, 0.10)
        noise_level = np.random.uniform(0.001, 0.003)
        seq = []
        # Upper body keypoint indices (head, shoulders, elbows, wrists)
        upper_body = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16]
        for f in range(FRAMES_PER_SEQUENCE):
            t = f / FRAMES_PER_SEQUENCE
            kp = base.copy()
            # Progressive lean: upper body Z increases, lower stays
            for idx in upper_body:
                if np.any(kp[idx] != 0):
                    # More lean for higher keypoints (head leans more than shoulders)
                    lean_factor = 1.0
                    if idx in [0, 1, 2, 3, 4, 9, 10]:  # Head/face
                        lean_factor = 1.5
                    elif idx in [11, 12]:  # Shoulders
                        lean_factor = 1.0
                    elif idx in [13, 14]:  # Elbows
                        lean_factor = 0.7
                    elif idx in [15, 16]:  # Wrists
                        lean_factor = 0.5
                    kp[idx, 2] += t * lean_amount * lean_factor
                    # Slight downward Y shift for head (looking down slightly)
                    if idx in [0, 1, 2, 3, 4, 9, 10]:
                        kp[idx, 1] += t * lean_amount * 0.3
            kp = add_noise(kp, noise_level=noise_level)
            seq.append(kp)
        sequences.append(np.array(seq))
    return sequences


def generate_frozen_rigid_stand(n=SAMPLES_PER_CLASS):
    """8: Frozen/Rigid Stand - Standing completely still (freeze response)."""
    sequences = []
    for _ in range(n):
        base = apply_augmentations(STANDING_POSE)
        # Extremely low noise — person is frozen
        noise_level = np.random.uniform(0.0003, 0.0008)
        seq = np.tile(base, (FRAMES_PER_SEQUENCE, 1, 1))
        seq = add_noise(seq, noise_level=noise_level)
        # NO oscillation, NO breathing, NO sway — completely rigid
        sequences.append(seq)
    return sequences


# ──────────────────────────────────────────────────────────────────────
# MAIN GENERATION
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HRI Motion Dataset Generator v2")
    print("=" * 60)
    print(f"\nClasses: {NUM_CLASSES}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Frames per sequence: {FRAMES_PER_SEQUENCE}")
    print(f"Total sequences: {SAMPLES_PER_CLASS * NUM_CLASSES}")
    print()

    generators = {
        0: generate_sitting_still,
        1: generate_standing_still,
        2: generate_walk_toward,
        3: generate_step_walk_back,
        4: generate_walk_across,
        5: generate_run_backward,
        6: generate_run_fast,
        7: generate_leaning_forward,
        8: generate_frozen_rigid_stand,
    }

    total_saved = 0

    for class_id in range(NUM_CLASSES):
        label = MOTION_LABELS[class_id]
        dir_name = f"{class_id}_{label.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"
        class_dir = KEYPOINTS_DIR / dir_name
        class_dir.mkdir(exist_ok=True)

        print(f"\nGenerating class {class_id}: {label} ...")
        sequences = generators[class_id](SAMPLES_PER_CLASS)

        for seq_idx, seq in enumerate(tqdm(sequences, desc=label, leave=False)):
            seq = np.array(seq)
            assert seq.shape == (FRAMES_PER_SEQUENCE, NUM_KEYPOINTS, 3), \
                f"Wrong shape: {seq.shape}, expected ({FRAMES_PER_SEQUENCE}, {NUM_KEYPOINTS}, 3)"
            filename = class_dir / f"synthetic_{class_id}_{seq_idx:04d}.npy"
            np.save(filename, seq)
            total_saved += 1

    # Save dataset info
    dataset_info = {
        "version": "v2",
        "motion_labels": MOTION_LABELS,
        "samples_per_class": SAMPLES_PER_CLASS,
        "frames_per_sequence": FRAMES_PER_SEQUENCE,
        "num_keypoints": NUM_KEYPOINTS,
        "total_sequences": total_saved,
    }
    info_path = Path("dataset_info_v2.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    for i, label in enumerate(MOTION_LABELS):
        dir_name = f"{i}_{label.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"
        class_dir = KEYPOINTS_DIR / dir_name
        npy_count = len(list(class_dir.glob("*.npy")))
        print(f"  {i}: {label:25s} — {npy_count:4d} sequences")
    print(f"\nTotal saved: {total_saved}")
    print(f"Dataset info: {info_path}")
    print(f"Output dir:   {KEYPOINTS_DIR}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
