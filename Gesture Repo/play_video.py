#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🎬 Test Video Selector & Player (TFLite Model & Landmark Smoothed Edition)
==========================================================================
Automatically scans the 'testVideo' folder and displays a clean menu.
Uses the trained 6-class TFLite model and real-time landmark smoothing.
"""

import os
import sys
import csv
import copy
import time
import collections
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
from model import KeyPointClassifier

# ── Finger Joint Constants ──────────────────────────────────────────────────
WRIST = 0
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
RING_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17


# ── Feature Extractors & Preprocessing ───────────────────────────────────────

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, _ in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0:
        return temp_landmark_list
    return [n / max_value for n in temp_landmark_list]


def hand_y_norm(lm):
    """Average y of all landmarks. Lower y = hand is higher in the frame."""
    return np.mean([l.y for l in lm])


# ── Stable Distance-Based Motion Tracker ─────────────────────────────────────

class HandMotionTracker:
    def __init__(self):
        # Stably track up to 2 hands using spatial coordinates
        self.x_hist = {0: collections.deque(maxlen=25), 1: collections.deque(maxlen=25)}
        self.y_hist = {0: collections.deque(maxlen=25), 1: collections.deque(maxlen=25)}
        self.beck_hist = {0: collections.deque(maxlen=15), 1: collections.deque(maxlen=15)}
        self.last_positions = {0: None, 1: None}

    def get_track_id(self, wx, wy):
        """Assign ID (0 or 1) based on proximity to previous frame's hand coordinates."""
        if self.last_positions[0] is None and self.last_positions[1] is None:
            self.last_positions[0] = (wx, wy)
            return 0

        if self.last_positions[0] is not None and self.last_positions[1] is None:
            d0 = (wx - self.last_positions[0][0])**2 + (wy - self.last_positions[0][1])**2
            if d0 < 0.06:
                self.last_positions[0] = (wx, wy)
                return 0
            else:
                self.last_positions[1] = (wx, wy)
                return 1

        if self.last_positions[0] is None and self.last_positions[1] is not None:
            d1 = (wx - self.last_positions[1][0])**2 + (wy - self.last_positions[1][1])**2
            if d1 < 0.06:
                self.last_positions[1] = (wx, wy)
                return 1
            else:
                self.last_positions[0] = (wx, wy)
                return 0

        d0 = (wx - self.last_positions[0][0])**2 + (wy - self.last_positions[0][1])**2
        d1 = (wx - self.last_positions[1][0])**2 + (wy - self.last_positions[1][1])**2
        if d0 < d1:
            self.last_positions[0] = (wx, wy)
            return 0
        else:
            self.last_positions[1] = (wx, wy)
            return 1

    def update(self, hand_id, wx, wy, beck_angle):
        self.x_hist[hand_id].append(wx)
        self.y_hist[hand_id].append(wy)
        self.beck_hist[hand_id].append(beck_angle)

    def is_raised(self, hand_id, current_hy):
        """
        Check if the hand is raised.
        Returns True if:
        1. The hand is currently very high in the frame (static hold: current_hy < 0.35)
        2. OR the hand has recently moved from a lower position to a higher position (dynamic raise).
        """
        if current_hy < 0.35:
            return True
        h = self.y_hist[hand_id]
        if len(h) < 10:
            return current_hy < 0.42
        
        arr = list(h)
        first_half = arr[:len(arr)//2]
        max_past_y = max(first_half) # Lowest position in the past (highest y coordinate)
        current_y = arr[-1]          # Current position
        
        # If it moved up by at least 0.12 and is now high in the frame
        if (max_past_y - current_y) > 0.12 and current_y < 0.45:
            return True
        return current_hy < 0.42

    def is_waving(self, hand_id):
        h = self.x_hist[hand_id]
        if len(h) < 10:
            return False, False

        arr = np.array(h)
        excursion = arr.max() - arr.min()
        diffs = np.diff(arr)
        reversals = int(np.sum(np.diff(np.sign(diffs)) != 0))

        # Horizontal waving rules
        wave = excursion > 0.14 and reversals >= 2
        brief_wave = excursion > 0.08 and reversals >= 1 and len(h) <= 15
        return wave, brief_wave

    def is_beckoning(self, hand_id):
        h = self.beck_hist[hand_id]
        if len(h) < 8:
            return False

        arr = np.array(h)
        excursion = arr.max() - arr.min()
        diffs = np.diff(arr)
        reversals = int(np.sum(np.diff(np.sign(diffs)) != 0))

        return excursion > 0.15 and reversals >= 2

    def reset_inactive(self, active_ids):
        """Clear tracking history for hands that disappeared from the frame."""
        for i in [0, 1]:
            if i not in active_ids:
                self.last_positions[i] = None
                self.x_hist[i].clear()
                self.y_hist[i].clear()
                self.beck_hist[i].clear()


# ── Gesture Engine ──────────────────────────────────────────────────────────

class GestureEngine:
    GESTURE_COLORS = {
        "One Hand Raised": (0, 220, 100),
        "Brief Wave": (255, 200, 0),
        "Pointing": (0, 180, 255),
        "None": (160, 160, 160),
        "Arms Waving": (255, 80, 200),
        "Wave": (255, 140, 0),
        "Beckoning": (100, 255, 220),
        "Arms Up": (0, 100, 255),
        "No hands": (80, 80, 80),
        "Thumbs up": (0, 255, 0),
        "Thumbs down": (0, 0, 255),
    }

    def __init__(self):
        self.tracker = HandMotionTracker()
        self.last_gesture = "None"
        self.last_time = time.time()
        
        # Landmark smoothing state (Exponential Moving Average)
        self.smoothed_landmarks = {0: None, 1: None}
        self.alpha = 0.45  # Smoothing factor (alpha = 0.45)

        # Initialize TFLite model classifier
        self.keypoint_classifier = KeyPointClassifier()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_labels = [row[0] for row in csv.reader(f)]

    def stable_update(self, gesture):
        now = time.time()
        if gesture != self.last_gesture:
            if now - self.last_time > 0.3:  # 300ms smoothing delay
                self.last_gesture = gesture
                self.last_time = now
        return self.last_gesture

    def process(self, hand_results, frame_shape):
        if not hand_results:
            self.tracker.reset_inactive([])
            # Reset smoothing states
            for i in [0, 1]:
                self.smoothed_landmarks[i] = None
            return self.stable_update("No hands"), self.GESTURE_COLORS["No hands"]

        n_hands = len(hand_results)
        per_hand = []
        active_ids = []

        for hl in hand_results:
            lm = hl.landmark
            wx = lm[WRIST].x
            wy = lm[WRIST].y

            # Compute stable spatial ID
            hand_id = self.tracker.get_track_id(wx, wy)
            active_ids.append(hand_id)

            # ── Real-time Landmark Smoothing Filter ──
            raw_landmark_list = calc_landmark_list(np.zeros(frame_shape), hl)
            if self.smoothed_landmarks[hand_id] is None:
                self.smoothed_landmarks[hand_id] = raw_landmark_list
            else:
                self.smoothed_landmarks[hand_id] = [
                    [
                        int(self.alpha * curr[0] + (1.0 - self.alpha) * prev[0]),
                        int(self.alpha * curr[1] + (1.0 - self.alpha) * prev[1])
                    ]
                    for curr, prev in zip(raw_landmark_list, self.smoothed_landmarks[hand_id])
                ]
            smoothed_list = self.smoothed_landmarks[hand_id]

            # Write smoothed coordinates back to MediaPipe's landmarks so drawing utility remains stable
            for idx, pt in enumerate(smoothed_list):
                hl.landmark[idx].x = pt[0] / frame_shape[1]
                hl.landmark[idx].y = pt[1] / frame_shape[0]

            hy = hand_y_norm(hl.landmark)
            
            # Calculate hand scale (distance from Wrist to Middle MCP)
            w_x, w_y = hl.landmark[WRIST].x, hl.landmark[WRIST].y
            m_x, m_y = hl.landmark[9].x, hl.landmark[9].y
            hand_scale = np.sqrt((w_x - m_x)**2 + (w_y - m_y)**2)
            if hand_scale == 0:
                hand_scale = 1.0
            
            beck_a = (hl.landmark[INDEX_TIP].y - hl.landmark[6].y) / hand_scale

            # Predict static pose index using TFLite model on smoothed coordinates
            pre_processed_landmark_list = pre_process_landmark(smoothed_list)
            hand_sign_id, hand_sign_conf = self.keypoint_classifier(pre_processed_landmark_list)

            # Filter low-confidence predictions
            if hand_sign_id in [2, 3, 4, 5] and hand_sign_conf < 0.80:
                hand_sign_id = -1

            # Update tracker history
            track_wx = wx if hand_sign_id in [0, 1, 2, 5, -1] else wx
            track_beck = beck_a if hand_sign_id in [0, 1, 5, -1] else 0
            self.tracker.update(hand_id, track_wx, hy, track_beck)

            per_hand.append({
                'sign': hand_sign_id,
                'sign_conf': hand_sign_conf,
                'hy': hy,
                'wx': wx,
                'id': hand_id,
                'lm': hl.landmark
            })

        # Clear inactive hands' smoothing states
        for i in [0, 1]:
            if i not in active_ids:
                self.smoothed_landmarks[i] = None
        self.tracker.reset_inactive(active_ids)

        # ── Two-Hand Scenarios (Arms Up / Arms Waving) ──────────────────────
        if n_hands == 2:
            h0 = per_hand[0]
            h1 = per_hand[1]

            h0_raised = self.tracker.is_raised(h0['id'], h0['hy'])
            h1_raised = self.tracker.is_raised(h1['id'], h1['hy'])
            both_high = h0_raised and h1_raised
            both_raising_poses = (h0['sign'] in [0, 1, -1] and h1['sign'] in [0, 1, -1])
            if both_raising_poses and both_high:
                return self.stable_update("Arms Up"), self.GESTURE_COLORS["Arms Up"]

            w0, bw0 = self.tracker.is_waving(h0['id'])
            w1, bw1 = self.tracker.is_waving(h1['id'])
            if (w0 or bw0) and (w1 or bw1):
                return self.stable_update("Arms Waving"), self.GESTURE_COLORS["Arms Waving"]

        # ── Single-Hand Scenarios (Using dominant hand - the highest hand) ───
        primary = min(per_hand, key=lambda x: x['hy'])
        sign = primary['sign']
        hy = primary['hy']
        hid = primary['id']

        is_wave, is_brief = self.tracker.is_waving(hid)

        # Beckoning
        if sign == 5 or self.tracker.is_beckoning(hid):
            return self.stable_update("Beckoning"), self.GESTURE_COLORS["Beckoning"]

        # Wave
        if is_wave and sign == 0:
            return self.stable_update("Wave"), self.GESTURE_COLORS["Wave"]

        # Brief Wave
        if is_brief and sign == 0:
            return self.stable_update("Brief Wave"), self.GESTURE_COLORS["Brief Wave"]

        # Pointing
        if sign == 2:
            return self.stable_update("Pointing"), self.GESTURE_COLORS["Pointing"]

        # Thumbs up
        if sign == 3:
            return self.stable_update("Thumbs up"), self.GESTURE_COLORS["Thumbs up"]

        # Thumbs down
        if sign == 4:
            return self.stable_update("Thumbs down"), self.GESTURE_COLORS["Thumbs down"]

        # One Hand Raised: Open hand, fist, or unknown held high, stationary
        if sign in [0, 1, -1] and self.tracker.is_raised(hid, hy):
            return self.stable_update("One Hand Raised"), self.GESTURE_COLORS["One Hand Raised"]

        return self.stable_update("None"), self.GESTURE_COLORS["None"]


# ── Video Selection UI ──────────────────────────────────────────────────────

def get_video_list(folder='testVideo2'):
    if not os.path.exists(folder):
        return []
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
    videos = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(video_extensions):
            full_path = os.path.join(folder, f)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            
            cap = cv.VideoCapture(full_path)
            fps = cap.get(cv.CAP_PROP_FPS)
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            videos.append({
                'name': f,
                'path': full_path,
                'size_mb': size_mb,
                'fps': fps,
                'frames': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            })
    
    return videos


def display_menu(videos):
    print("\n" + "=" * 70)
    print("  🎬  HRI Gesture Detection — Test Video Player (TFLite)")
    print("=" * 70)
    print(f"\n  Found {len(videos)} videos in testVideo/ folder:\n")
    print(f"  {'#':<4} {'Video Name':<20} {'Size':<10} {'Duration':<10} {'Resolution':<12}")
    print(f"  {'─'*4} {'─'*20} {'─'*10} {'─'*10} {'─'*12}")
    
    for i, v in enumerate(videos, 1):
        duration_str = f"{v['duration']:.1f}s"
        size_str = f"{v['size_mb']:.1f} MB"
        res_str = f"{v['width']}x{v['height']}"
        print(f"  {i:<4} {v['name']:<20} {size_str:<10} {duration_str:<10} {res_str:<12}")
    
    print(f"\n  {'─'*56}")
    print(f"  0    ▶ Run ALL videos sequentially (No prompts)")
    print(f"  q    ✖ Quit")
    print(f"  {'─'*56}")


def select_video(videos):
    while True:
        display_menu(videos)
        choice = input("\n  ⏎ Enter your choice: ").strip().lower()
        
        if choice == 'q':
            return None
        
        if choice == '0':
            return videos
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(videos):
                return [videos[idx - 1]]
            else:
                print(f"\n  ⚠️  Invalid choice. Enter 1-{len(videos)}, 0 (all), or q (quit).")
        except ValueError:
            print(f"\n  ⚠️  Invalid input. Enter a number or 'q'.")


def resize_with_aspect_ratio(image, max_dim=960):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)


# ── Process Single Video ────────────────────────────────────────────────────

def process_video(video_path, video_name):
    print(f"\n  🎬 Playing: {video_name}")
    print(f"  📂 Path: {video_path}")
    print(f"  ⏸  Press ESC to stop, SPACE to pause/resume\n")
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Could not open video: {video_path}")
        return False
    orientation = cap.get(cv.CAP_PROP_ORIENTATION_META) if hasattr(cv, 'CAP_PROP_ORIENTATION_META') else cap.get(48)
    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        model_complexity=1,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    engine = GestureEngine()
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    TARGET_FPS = 15
    FRAME_DELAY = 1.0 / TARGET_FPS
    frame_count = 0
    last_frame_time = time.time()
    paused = False
    user_aborted = False

    while True:
        if paused:
            key = cv.waitKey(100)
            if key == 32:
                paused = False
                last_frame_time = time.time()
            elif key == 27:
                user_aborted = True
                break
            continue

        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < FRAME_DELAY:
            sleep_ms = int((FRAME_DELAY - elapsed) * 1000)
            key = cv.waitKey(max(1, sleep_ms))
        else:
            key = cv.waitKey(1)

        last_frame_time = time.time()

        if key == 27:
            user_aborted = True
            break
        elif key == 32:
            paused = True
            continue

        ret, image = cap.read()
        if not ret:
            break

        # Apply rotation based on EXIF metadata
        if orientation == 90:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        elif orientation == 180:
            image = cv.rotate(image, cv.ROTATE_180)
        elif orientation == 270:
            image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

        # Resize for performance and proper display scale
        image = resize_with_aspect_ratio(image, max_dim=960)

        frame_count += 1
        fps = cvFpsCalc.get()
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        res = hands.process(image_rgb)

        if res.multi_hand_landmarks:
            # First process with GestureEngine to smooth coordinates in-place
            gesture, color = engine.process(res.multi_hand_landmarks, image.shape)
            
            # Draw overlay using smoothed landmarks
            for hl in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    debug_image, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 150), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2)
                )
        else:
            gesture, color = engine.process(None, image.shape)

        # Draw video info banner
        cv.rectangle(debug_image, (0, 0), (debug_image.shape[1], 80), (15, 15, 15), -1)
        cv.putText(debug_image, f"Video: {video_name}  |  FPS: {fps}  |  Frame: {frame_count}",
                  (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv.LINE_AA)
        
        cv.putText(debug_image, "Scenario: " + gesture,
                  (10, 62), cv.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv.LINE_AA)

        # Controls hint
        h = debug_image.shape[0]
        cv.rectangle(debug_image, (0, h - 35), (debug_image.shape[1], h), (0,0,0), -1)
        cv.putText(debug_image, "ESC: Return to Menu  |  SPACE: Pause",
                  (10, h - 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv.LINE_AA)

        cv.imshow('HRI Gesture Detection - Video Player', debug_image)

    cap.release()
    hands.close()
    cv.destroyAllWindows()
    return user_aborted


def main():
    while True:
        videos = get_video_list('testVideo2')
        
        if not videos:
            print("\n  ❌ No videos found in testVideo/ folder!")
            sys.exit(1)
        
        selected = select_video(videos)
        
        if selected is None:
            print("\n  👋 Goodbye!")
            break
        
        is_play_all = len(selected) > 1
        
        for i, video in enumerate(selected):
            if is_play_all:
                print(f"\n  📹 Playing video {i+1}/{len(selected)}: {video['name']}")
            
            # process_video returns True if the user pressed ESC to abort
            aborted = process_video(video['path'], video['name'])
            
            if aborted:
                print("\n  ⏸ Playback stopped by user.")
                break
                
            if is_play_all and i < len(selected) - 1:
                print("  ⏭ Moving to next video automatically in 1 second...")
                time.sleep(1.0)
                
        print("\n  ** Finished! Returning to selector menu...")


if __name__ == '__main__':
    main()
