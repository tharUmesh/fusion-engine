"""
Real-time Human Motion Recognizer using PyTorch LSTM
=====================================================
Uses:
  - MediaPipe Pose  → 33 body keypoints
  - PyTorch LSTM    → Trained on 9 HRI motion classes (100% accuracy)

Usage:
    # Webcam
    python action_recognizer.py --webcam

    # Video file
    python action_recognizer.py --video path/to/video.mp4

    # Video file + save output
    python action_recognizer.py --video path/to/video.mp4 --save output.avi
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# MOTION LABELS
# ─────────────────────────────────────────────────────────────────────────────
# 8 motion classes (Walk Toward + Step/Walk Back merged into Walking —
# direction detection from a fixed/phone camera is unreliable)
MOTION_LABELS = [
    "Sitting Still",       # 0  person seated & still
    "Standing Still",      # 1  person upright & not moving
    "Walking",             # 2  person walking (any direction)
    "Walk Across",         # 3  person walking sideways across frame
    "Run Backward",        # 4  person running away/backward
    "Run (Fast Movement)", # 5  person running fast
    "Leaning Forward",     # 6  person bending forward / crouching toward robot
    "Frozen/Rigid Stand",  # 7  person standing completely motionless
]

# Colour per label (BGR)
LABEL_COLORS = {
    "Sitting Still":       (160, 160, 160),  # Grey
    "Standing Still":      (200, 200, 200),  # Light Grey
    "Walking":             (0,   210, 255),  # Yellow
    "Walk Across":         (0,   255, 100),  # Green
    "Run Backward":        (0,    80, 255),  # Orange-red
    "Run (Fast Movement)": (0,   255, 200),  # Turquoise
    "Leaning Forward":     (220, 255,   0),  # Lime
    "Frozen/Rigid Stand":  (50,   50, 150),  # Dark Red/Blue
}

# ─────────────────────────────────────────────────────────────────────────────
# LSTM MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class MotionLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=3, num_classes=9, dropout=0.4):
        super(MotionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.reshape(batch_size, seq_len, -1)  # (batch, seq_len, 99)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        out = self.fc(last_hidden)
        return out

# ─────────────────────────────────────────────────────────────────────────────
# POSE CLASSIFIER (Rule-based)
# ─────────────────────────────────────────────────────────────────────────────
class PoseClassifier:
    def classify(self, landmarks) -> str:
        if landmarks is None:
            return "Unknown"
        lm = landmarks.landmark

        # Vertical span: nose (0) vs mid-hip (23/24)
        nose_y     = lm[0].y
        hip_y      = (lm[23].y + lm[24].y) / 2
        knee_y     = (lm[25].y + lm[26].y) / 2
        ankle_y    = (lm[27].y + lm[28].y) / 2
        shoulder_y = (lm[11].y + lm[12].y) / 2

        body_height = abs(ankle_y - nose_y) + 1e-6
        torso_ratio = abs(hip_y - shoulder_y) / body_height
        leg_ratio   = abs(ankle_y - knee_y)  / body_height

        if body_height < 0.25:
            return "Lying"
        if torso_ratio > 0.35 and leg_ratio < 0.15:
            return "Sitting"
        if knee_y < hip_y + 0.05:
            return "Crouching"
        return "Standing"

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE DASHBOARD DRAWING
# Landscape video → right sidebar (320px wide)
# Portrait video  → bottom panel (280px tall, 3-column layout)
# ─────────────────────────────────────────────────────────────────────────────
def draw_dashboard(frame, pose_label, motion_label, confidence, probabilities, fps, landmarks, mp_drawing, mp_pose):
    """Adaptive dashboard: right sidebar for landscape, bottom panel for portrait videos."""
    h, w = frame.shape[:2]
    is_portrait = h > w
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw MediaPipe skeleton on frame
    if landmarks:
        mp_drawing.draw_landmarks(
            frame, landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2),
        )

    if is_portrait:
        # ── PORTRAIT LAYOUT: bottom panel ──────────────────────────────────
        panel_h = 280
        canvas = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        cv2.rectangle(canvas, (0, h), (w, h + panel_h), (20, 24, 33), -1)
        cv2.line(canvas, (0, h), (w, h), (43, 52, 69), 2)

        # Row 1: Title + FPS + Pose
        y = h + 30
        cv2.putText(canvas, "HRI MOTION ANALYZER", (15, y), font, 0.65, (230, 235, 245), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS:{fps:.1f}  Pose:{pose_label}", (w - 210, y), font, 0.45, (150, 160, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, (10, y + 8), (w - 10, y + 8), (60, 75, 100), 1)

        # Row 2: Active Motion Badge (full width)
        y += 32
        color = LABEL_COLORS.get(motion_label, (255, 255, 255))
        cv2.rectangle(canvas, (10, y - 20), (w - 10, y + 10), color, -1)
        text_color = (255, 255, 255) if np.mean(color) < 120 else (20, 20, 20)
        cv2.putText(canvas, f"  {motion_label}   {confidence*100:.1f}%", (18, y - 2), font, 0.65, text_color, 2, cv2.LINE_AA)

        # Divider
        y += 22
        cv2.line(canvas, (10, y), (w - 10, y), (60, 75, 100), 1)
        y += 18

        # Rows: 3-column class probability bars (dynamic, works for any label count)
        col_w = w // 3
        bar_max_px = max(col_w - 92, 20)
        n = len(MOTION_LABELS)
        per_col = (n + 2) // 3   # ceil(n/3) — e.g. 8 classes → 3, 3, 2
        for col_idx in range(3):
            bx = col_idx * col_w + 6
            by = y
            start = col_idx * per_col
            end   = min(start + per_col, n)
            for i in range(start, end):
                lbl = MOTION_LABELS[i]
                prob = probabilities[i] if (probabilities is not None and i < len(probabilities)) else 0.0
                bc = LABEL_COLORS.get(lbl, (100, 100, 100))
                cv2.putText(canvas, lbl[:11], (bx, by), font, 0.37, (170, 180, 200), 1, cv2.LINE_AA)
                bx_bar = bx + 76
                cv2.rectangle(canvas, (bx_bar, by - 9), (bx_bar + bar_max_px, by + 2), (40, 48, 64), -1)
                fw_ = int(prob * bar_max_px)
                if fw_ > 0:
                    cv2.rectangle(canvas, (bx_bar, by - 9), (bx_bar + fw_, by + 2), bc, -1)
                cv2.putText(canvas, f"{prob*100:.0f}%", (bx_bar + bar_max_px + 3, by), font, 0.34, (120, 130, 150), 1, cv2.LINE_AA)
                by += 20

    else:
        # ── LANDSCAPE LAYOUT: right sidebar ────────────────────────────────
        sidebar_w = 320
        canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        cv2.rectangle(canvas, (w, 0), (w + sidebar_w, h), (20, 24, 33), -1)
        cv2.line(canvas, (w, 0), (w, h), (43, 52, 69), 2)

        cv2.putText(canvas, "HRI MOTION ANALYZER", (w + 20, 35), font, 0.65, (230, 235, 245), 2, cv2.LINE_AA)
        cv2.line(canvas, (w + 20, 48), (w + sidebar_w - 20, 48), (60, 75, 100), 1)

        cv2.putText(canvas, f"FPS: {fps:.1f}", (w + 20, 75), font, 0.5, (150, 160, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Pose: {pose_label}", (w + 140, 75), font, 0.5, (150, 160, 180), 1, cv2.LINE_AA)

        y_offset = 110
        cv2.putText(canvas, "ACTIVE MOTION:", (w + 20, y_offset), font, 0.55, (100, 120, 150), 1, cv2.LINE_AA)
        y_offset += 25

        color = LABEL_COLORS.get(motion_label, (255, 255, 255))
        cv2.rectangle(canvas, (w + 20, y_offset - 20), (w + sidebar_w - 20, y_offset + 10), color, -1)
        text_color = (255, 255, 255) if np.mean(color) < 120 else (20, 20, 20)
        cv2.putText(canvas, f"{motion_label} ({confidence*100:.1f}%)", (w + 30, y_offset - 1), font, 0.55, text_color, 2, cv2.LINE_AA)

        y_offset += 40
        cv2.line(canvas, (w + 20, y_offset), (w + sidebar_w - 20, y_offset), (60, 75, 100), 1)
        y_offset += 25

        cv2.putText(canvas, "CLASS CONFIDENCE:", (w + 20, y_offset), font, 0.55, (100, 120, 150), 1, cv2.LINE_AA)
        y_offset += 25

        bar_max_w = 150
        for i, label in enumerate(MOTION_LABELS):
            prob = probabilities[i] if probabilities is not None else 0.0
            cv2.putText(canvas, label[:14], (w + 20, y_offset), font, 0.45, (170, 180, 200), 1, cv2.LINE_AA)
            bar_w = int(prob * bar_max_w)
            bar_color = LABEL_COLORS.get(label, (100, 100, 100))
            cv2.rectangle(canvas, (w + 130, y_offset - 10), (w + 130 + bar_max_w, y_offset + 2), (40, 48, 64), -1)
            if bar_w > 0:
                cv2.rectangle(canvas, (w + 130, y_offset - 10), (w + 130 + bar_w, y_offset + 2), bar_color, -1)
            cv2.putText(canvas, f"{prob*100:.0f}%", (w + 135 + bar_max_w, y_offset), font, 0.4, (120, 130, 150), 1, cv2.LINE_AA)
            y_offset += 22

    return canvas

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def run(source, model_path, save_path=None, no_show=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load LSTM Model
    print(f"[INFO] Loading LSTM model from: {model_path}")
    model = MotionLSTM(input_size=99, hidden_size=128, num_layers=3, num_classes=9, dropout=0.4)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        sys.exit(1)
        
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_classifier = PoseClassifier()
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)
        
    # ── Smart Auto-Rotation ──────────────────────────────────────────────────
    # Some decoders (e.g. Windows + .MOV) already apply the EXIF rotation,
    # so we read one test frame first and check whether the actual aspect ratio
    # already matches the expected orientation.  Only rotate if needed.
    orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else cap.get(48)
    rotation_code = None
    if orientation in (90, 270):
        # These flags say the video *should* be portrait (h > w after rotation)
        _ret0, _f0 = cap.read()
        if _ret0:
            _fh, _fw = _f0.shape[:2]
            already_portrait = _fh > _fw
            if not already_portrait:
                # Frame is still landscape — rotation not yet applied → apply it
                rotation_code = cv2.ROTATE_90_CLOCKWISE if orientation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
            # Rewind so the main loop sees all frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif orientation == 180:
        rotation_code = cv2.ROTATE_180

    def resize_with_aspect_ratio(image, max_dim=960):
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    
    print("[INFO] Press 'Q' to quit.")

    # ── Keypoint sliding window (30 frames) ─────────────────────────────────
    keypoints_queue = deque(maxlen=30)

    # ── Physics-based motion classification parameters ───────────────────────
    # Primary motion signal: hips only!
    # Ankles are excluded because occlusion (e.g. skirts, gas cylinders) causes
    # MediaPipe ankles to jitter wildly, leading to false walking triggers.
    HIP_IDX       = [23, 24]          # left + right hip
    BODY_IDX      = HIP_IDX           # hips only for speed measurement

    # Speed thresholds (normalised coords * 100, averaged per frame)
    STATIC_THRESH = 0.22   # below → person is static
    RUN_THRESH    = 1.30   # above → running/fast movement

    # Smoothing (EMA on probability vector)
    SMOOTH_ALPHA  = 0.25
    smooth_probs  = np.ones(8) / 8


    frame_idx   = 0
    t_prev      = time.time()
    fps_display = 0.0
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Rotate if needed
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
                
            # Downscale for performance
            frame = resize_with_aspect_ratio(frame, max_dim=960)
            
            # Lazy initialize VideoWriter after rotation & resizing
            if save_path and writer is None:
                fh, fw = frame.shape[:2]
                is_portrait_frame = fh > fw
                if is_portrait_frame:
                    out_w, out_h = fw, fh + 280   # bottom panel
                else:
                    out_w, out_h = fw + 320, fh   # right sidebar
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(save_path, fourcc, fps_cam, (out_w, out_h))
                print(f"[INFO] Saving output to: {save_path} with canvas size: {out_w}x{out_h}")
                
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            
            landmarks = results.pose_landmarks

            # Rule-based posture label (used both in dashboard and hybrid logic)
            pose_label = pose_classifier.classify(landmarks)

            # ── Keypoint extraction ───────────────────────────────────────────
            if landmarks:
                pts = np.zeros((33, 3), dtype=np.float32)
                for i, lm in enumerate(landmarks.landmark):
                    pts[i] = [lm.x, lm.y, lm.z]
                keypoints_queue.append(pts)
            else:
                keypoints_queue.clear()

            # ── Defaults ─────────────────────────────────────────────────────
            motion_label  = "Standing Still"
            confidence    = 0.90
            probabilities = np.zeros(8)
            probabilities[1] = 0.90

            if len(keypoints_queue) >= 4 and landmarks:
                arr = np.array(keypoints_queue)   # (seq_len, 33, 3)
                vel = np.diff(arr, axis=0)         # (seq_len-1, 33, 3)

                # ── 1. Body speed: hips + ankles only ────────────────────────
                # Using only the large leg/hip joints — completely ignores arms.
                body_vel   = vel[:, BODY_IDX, :2]  # XY only, (seq-1, 4, 2)
                body_speed = float(np.mean(np.abs(body_vel)) * 100.0)

                # ── 2. Trend detection (last 15 frames) ──────────────────────
                # We analyze both hips and shoulders translation. For a true walk,
                # the entire torso (shoulders + hips) must translate in the same direction.
                look_back = min(15, len(arr) - 1)
                
                # Hips width (scale indicator: moving closer → wider hips)
                hip_w = np.linalg.norm(arr[-look_back-1:, 23, :2] - arr[-look_back-1:, 24, :2], axis=1)
                dh = float(hip_w[-1] - hip_w[0])
                path_h = float(np.sum(np.abs(np.diff(hip_w))))
                eff_h = abs(dh) / (path_h + 1e-5)
                
                # Hip positions (average of left/right hip)
                hips_xy = arr[-look_back-1:, HIP_IDX, :2].mean(axis=1) # (seq, 2)
                dx_hip = float(hips_xy[-1, 0] - hips_xy[0, 0])
                dy_hip = float(hips_xy[-1, 1] - hips_xy[0, 1])
                path_x_hip = float(np.sum(np.abs(np.diff(hips_xy[:, 0]))))
                path_y_hip = float(np.sum(np.abs(np.diff(hips_xy[:, 1]))))
                eff_x_hip = abs(dx_hip) / (path_x_hip + 1e-5)
                eff_y_hip = abs(dy_hip) / (path_y_hip + 1e-5)

                # Shoulder positions (average of left/right shoulder, indices 11, 12)
                sh_xy = arr[-look_back-1:, [11, 12], :2].mean(axis=1) # (seq, 2)
                dx_sh = float(sh_xy[-1, 0] - sh_xy[0, 0])
                dy_sh = float(sh_xy[-1, 1] - sh_xy[0, 1])
                path_x_sh = float(np.sum(np.abs(np.diff(sh_xy[:, 0]))))
                path_y_sh = float(np.sum(np.abs(np.diff(sh_xy[:, 1]))))
                eff_x_sh = abs(dx_sh) / (path_x_sh + 1e-5)
                eff_y_sh = abs(dy_sh) / (path_y_sh + 1e-5)

                # ── Hips + Shoulders same-direction translation checks ──────
                is_translating_across = False
                if abs(dx_hip) > 0.024 and eff_x_hip > 0.70:
                    if abs(dx_sh) > 0.020 and eff_x_sh > 0.70:
                        if np.sign(dx_hip) == np.sign(dx_sh):
                            is_translating_across = True

                is_translating_vert = False
                if abs(dy_hip) > 0.020 and eff_y_hip > 0.70:
                    if abs(dy_sh) > 0.016 and eff_y_sh > 0.70:
                        if np.sign(dy_hip) == np.sign(dy_sh):
                            is_translating_vert = True

                # Determine if there is a directed walking movement (translating in space)
                is_directed_walk = False
                walk_type = "Walking"
                
                # Case A: Walk Across (horizontal translation dominates vertical translation)
                if is_translating_across and abs(dx_hip) > abs(dy_hip) * 1.5:
                    is_directed_walk = True
                    walk_type = "Walk Across"
                # Case B: Walking toward/away/general (vertical translation OR scale change)
                elif is_translating_vert or (abs(dh) > 0.012 and eff_h > 0.70):
                    is_directed_walk = True
                    walk_type = "Walking"

                # 8 classes: Sitting Still, Standing Still, Walking,
                # Walk Across, Run Backward, Run (Fast Move), Leaning Fwd, Frozen
                probs = np.zeros(8)

                if body_speed >= RUN_THRESH:
                    # ── RUNNING: Fast movement ──────────────────────────────────
                    if dx_hip < -0.01:
                        motion_label = "Run Backward"
                        probs[4] = 0.85
                    else:
                        motion_label = "Run (Fast Movement)"
                        probs[5] = 0.85

                elif is_directed_walk:
                    # ── WALKING: Only if steady directional movement is confirmed ─
                    if walk_type == "Walk Across":
                        motion_label = "Walk Across"
                        probs[3] = 0.80
                    else:
                        motion_label = "Walking"
                        probs[2] = 0.80

                else:
                    # ── STATIC: Otherwise, they are standing/sitting/crouching ───
                    if pose_label == "Sitting":
                        motion_label = "Sitting Still"
                        probs[0] = 0.95
                    elif pose_label == "Crouching":
                        motion_label = "Leaning Forward"
                        probs[6] = 0.90
                    elif body_speed < 0.08:
                        motion_label = "Frozen/Rigid Stand"
                        probs[7] = 0.90
                    else:
                        motion_label = "Standing Still"
                        probs[1] = 0.90

                # ── EMA smooth ────────────────────────────────────────────────
                smooth_probs = SMOOTH_ALPHA * probs + (1 - SMOOTH_ALPHA) * smooth_probs
                voted_idx    = int(np.argmax(smooth_probs))
                motion_label  = MOTION_LABELS[voted_idx]
                confidence    = float(smooth_probs[voted_idx])
                probabilities = smooth_probs



            elif not landmarks:
                smooth_probs = np.ones(8) / 8
                motion_label  = "Standing Still"
                confidence    = 0.0
                probabilities = np.zeros(8)

                
            # Measure FPS
            now = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            
            # Draw premium dashboard
            canvas = draw_dashboard(
                frame, pose_label, motion_label,
                confidence, probabilities, fps_display,
                landmarks, mp_drawing, mp_pose
            )
            
            if writer:
                writer.write(canvas)

            if not no_show:
                # ── Fit canvas to screen before displaying ─────────────────
                # Portrait canvases (e.g. 540×1240) are often taller than the
                # screen, hiding the bottom dashboard panel.  Scale down to fit.
                MAX_DISP_H = 900   # max display height (fits 1080p with taskbar)
                MAX_DISP_W = 1400  # max display width
                ch, cw = canvas.shape[:2]
                scale = min(MAX_DISP_H / ch, MAX_DISP_W / cw, 1.0)
                if scale < 1.0:
                    disp = cv2.resize(canvas,
                                      (int(cw * scale), int(ch * scale)),
                                      interpolation=cv2.INTER_AREA)
                else:
                    disp = canvas
                cv2.imshow("Human Motion Recognizer LSTM", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Execution finished successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENTS PARSING
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human Motion Recognizer using PyTorch LSTM & MediaPipe Pose"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--webcam", action="store_true", help="Use default webcam")
    src.add_argument("--video", type=str, metavar="PATH", help="Path to video file")
    src.add_argument("--camera", type=int, metavar="INDEX", help="External camera device index")
    
    parser.add_argument("--model-path", type=str, metavar="PATH",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "models", "motion_lstm_v2_best.pth"),
                        help="Path to trained LSTM model weights")
    parser.add_argument("--save", type=str, metavar="OUT", help="Save output video path (.avi)")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display OpenCV window (useful for headless/batch execution)")
    
    args = parser.parse_args()
    
    if args.webcam:
        source = 0
    elif args.camera is not None:
        source = args.camera
    else:
        source = args.video
        
    run(source, args.model_path, save_path=args.save, no_show=args.no_show)
