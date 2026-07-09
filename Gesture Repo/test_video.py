#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test video processing for gesture detection.
Usage: python test_video.py --video testVideo/6.mp4
"""

import csv
import copy
import argparse
import itertools
import time
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def resize_with_aspect_ratio(image, max_dim=960):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)


def check_hand_raised(y_history, current_y):
    """
    Checks if a hand is raised based on current height and history.
    y_history: deque of normalized average y coordinates of the hand.
    current_y: current normalized average y coordinate.
    Returns True if raised, False otherwise.
    """
    # 1. Static check: hand is held very high in the frame
    if current_y < 0.35:
        return True
        
    # 2. Dynamic check: hand moved from lower to higher position
    if len(y_history) >= 10:
        y_list = list(y_history)
        first_half = y_list[:len(y_list)//2]
        max_past_y = max(first_half) # Lowest position in the past (highest y value)
        
        # If hand moved upwards by at least 0.12 (normalized y decreases)
        # and is currently at a raised height (current_y < 0.45)
        if (max_past_y - current_y) > 0.12 and current_y < 0.45:
            return True
            
    # 3. Fallback: simple height threshold
    return current_y < 0.42


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="testVideo/6.mp4")
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.45)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.45)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cap = cv.VideoCapture(args.video)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    orientation = cap.get(cv.CAP_PROP_ORIENTATION_META) if hasattr(cv, 'CAP_PROP_ORIENTATION_META') else cap.get(48)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv',
              encoding='utf-8-sig') as f:
        history_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    smoothed_landmarks = {0: None, 1: None}
    alpha = 0.45

    point_history = {0: deque(maxlen=history_length), 1: deque(maxlen=history_length)}
    finger_gesture_history = {0: deque(maxlen=history_length), 1: deque(maxlen=history_length)}
    y_hist = {0: deque(maxlen=25), 1: deque(maxlen=25)}

    for i in range(2):
        for _ in range(history_length):
            point_history[i].append([0, 0])
            finger_gesture_history[i].append(0)

    mode = 0
    last_frame_time = time.time()
    display_scenario_text = "None"

    TARGET_FPS = 15
    FRAME_DELAY = 1.0 / TARGET_FPS
    frame_count = 0

    while True:
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < FRAME_DELAY:
            sleep_ms = int((FRAME_DELAY - elapsed) * 1000)
            if sleep_ms > 0:
                key = cv.waitKey(sleep_ms)
            else:
                key = cv.waitKey(1)
        else:
            key = cv.waitKey(1)

        last_frame_time = time.time()

        fps = cvFpsCalc.get()
        if key == 27: break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret: break

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
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        detected_hand_indices = []
        global_scenario_text = "None"
        hand_states = {}

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                   results.multi_handedness):
                hand_index = handedness.classification[0].index
                detected_hand_indices.append(hand_index)

                # ── Real-time Landmark Smoothing Filter ──
                raw_landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                if smoothed_landmarks[hand_index] is None:
                    smoothed_landmarks[hand_index] = raw_landmark_list
                else:
                    smoothed_landmarks[hand_index] = [
                        [
                            int(alpha * curr[0] + (1.0 - alpha) * prev[0]),
                            int(alpha * curr[1] + (1.0 - alpha) * prev[1])
                        ]
                        for curr, prev in zip(raw_landmark_list, smoothed_landmarks[hand_index])
                    ]
                smoothed_list = smoothed_landmarks[hand_index]

                # Overwrite coordinate values for drawing & calculation stability
                for idx, pt in enumerate(smoothed_list):
                    hand_landmarks.landmark[idx].x = pt[0] / debug_image.shape[1]
                    hand_landmarks.landmark[idx].y = pt[1] / debug_image.shape[0]

                # Compute normalized height and append to history
                hy = np.mean([l.y for l in hand_landmarks.landmark])
                y_hist[hand_index].append(hy)

                brect = calc_bounding_rect(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(smoothed_list)
                pre_processed_history_list = pre_process_point_history(
                    debug_image, point_history[hand_index])

                hand_sign_id, hand_sign_conf = keypoint_classifier(
                    pre_processed_landmark_list)

                if hand_sign_id in [2, 3, 4, 5] and hand_sign_conf < 0.80:
                    hand_sign_id = -1

                if hand_sign_id in [0, 1, 2, 5, -1]:
                    point_history[hand_index].append(smoothed_list[8])
                else:
                    point_history[hand_index].append([0, 0])

                finger_gesture_id = 0
                fg_conf = 0.0
                wave_detected, wave_amplitude = detect_wave(point_history[hand_index], brect[2] - brect[0])

                if len(pre_processed_history_list) == (history_length * 2):
                    if wave_detected:
                        finger_gesture_id = 4
                        fg_conf = 0.95
                    elif detect_come_here(point_history[hand_index], brect[3] - brect[1]):
                        finger_gesture_id = 5
                        fg_conf = 0.92
                    elif hand_sign_id == 2:
                        finger_gesture_id, fg_conf = point_history_classifier(
                            pre_processed_history_list)

                finger_gesture_history[hand_index].append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history[hand_index]).most_common()
                current_fg_id = most_common_fg_id[0][0]

                hand_states[hand_index] = {
                    'sign': hand_sign_id,
                    'action': current_fg_id,
                    'brect': brect,
                    'wave_amp': wave_amplitude,
                    'sign_conf': hand_sign_conf,
                    'action_conf': fg_conf,
                    'hy': hy,
                    'id': hand_index
                }

                debug_image = draw_bounding_rect(True, debug_image, brect)
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                debug_image = draw_point_history(debug_image, point_history[hand_index])

                sign_name = keypoint_labels[hand_sign_id] if hand_sign_id != -1 else "Unknown"
                label = f"{handedness.classification[0].label}: {sign_name} ({int(hand_sign_conf * 100)}%)"
                cv.rectangle(debug_image, (brect[0], brect[1]),
                           (brect[2], brect[1] - 22), (0, 0, 0), -1)
                cv.putText(debug_image, label, (brect[0] + 5, brect[1] - 4),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        for i in range(2):
            if i not in detected_hand_indices:
                point_history[i].append([0, 0])
                smoothed_landmarks[i] = None
                y_hist[i].clear()

        # === Global Scenario Resolution (8 Gestures, No Reaching) ===
        num_hands = len(hand_states)
        global_conf = 0.0

        if num_hands == 2:
            h1 = hand_states[list(hand_states.keys())[0]]
            h2 = hand_states[list(hand_states.keys())[1]]

            h1_raised = check_hand_raised(y_hist[h1['id']], h1['hy'])
            h2_raised = check_hand_raised(y_hist[h2['id']], h2['hy'])

            if h1['action'] == 4 and h2['action'] == 4:
                global_scenario_text = "Arms waving"
                global_conf = min(h1['action_conf'], h2['action_conf'])
            elif h1['sign'] in [0, 1, -1] and h2['sign'] in [0, 1, -1] and h1_raised and h2_raised:
                global_scenario_text = "Arms up"
                c1 = h1['sign_conf'] if h1['sign'] != -1 else 0.85
                c2 = h2['sign_conf'] if h2['sign'] != -1 else 0.85
                global_conf = min(c1, c2)
            elif h1['action'] == 4 or h2['action'] == 4:
                global_scenario_text = "Wave"
                global_conf = h1['action_conf'] if h1['action'] == 4 else h2['action_conf']
            elif h1['action'] == 5 or h2['action'] == 5 or h1['sign'] == 5 or h2['sign'] == 5:
                global_scenario_text = "Beckoning"
                if h1['action'] == 5 or h1['sign'] == 5:
                    global_conf = max(h1['action_conf'], h1['sign_conf'] if h1['sign'] == 5 else 0.0)
                else:
                    global_conf = max(h2['action_conf'], h2['sign_conf'] if h2['sign'] == 5 else 0.0)
            elif h1['sign'] == 2 or h2['sign'] == 2:
                global_scenario_text = "Pointing"
                global_conf = h1['sign_conf'] if h1['sign'] == 2 else h2['sign_conf']
            elif h1['sign'] == 3 or h2['sign'] == 3:
                global_scenario_text = "Thumbs up"
                global_conf = h1['sign_conf'] if h1['sign'] == 3 else h2['sign_conf']
            elif h1['sign'] == 4 or h2['sign'] == 4:
                global_scenario_text = "Thumbs down"
                global_conf = h1['sign_conf'] if h1['sign'] == 4 else h2['sign_conf']
            elif (h1['sign'] in [0, 1, -1] and h1_raised) or (h2['sign'] in [0, 1, -1] and h2_raised):
                target_h = h1 if (h1['sign'] in [0, 1, -1] and h1_raised) else h2
                global_scenario_text = "One hand raised"
                global_conf = target_h['sign_conf'] if target_h['sign'] != -1 else 0.85
            else:
                global_scenario_text = "None"

        elif num_hands == 1:
            h1 = hand_states[list(hand_states.keys())[0]]
            h1_raised = check_hand_raised(y_hist[h1['id']], h1['hy'])

            if h1['action'] == 4:
                if h1['wave_amp'] > 150:
                    global_scenario_text = "Wave"
                else:
                    global_scenario_text = "Brief wave"
                global_conf = h1['action_conf']
            elif h1['action'] == 5 or h1['sign'] == 5:
                global_scenario_text = "Beckoning"
                global_conf = max(h1['action_conf'], h1['sign_conf'] if h1['sign'] == 5 else 0.0)
            elif h1['sign'] == 2:
                global_scenario_text = "Pointing"
                global_conf = h1['sign_conf']
            elif h1['sign'] == 3:
                global_scenario_text = "Thumbs up"
                global_conf = h1['sign_conf']
            elif h1['sign'] == 4:
                global_scenario_text = "Thumbs down"
                global_conf = h1['sign_conf']
            elif h1['sign'] in [0, 1, -1] and h1_raised:
                global_scenario_text = "One hand raised"
                global_conf = h1['sign_conf'] if h1['sign'] != -1 else 0.85
            else:
                global_scenario_text = "None"

        if global_scenario_text != "None" and global_conf > 0:
            global_scenario_text += f" ({int(global_conf * 100)}%)"

        if frame_count % 3 == 0:
            display_scenario_text = global_scenario_text

        cv.putText(debug_image, "Scenario: " + display_scenario_text,
                  (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(debug_image, "Scenario: " + display_scenario_text,
                  (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv.LINE_AA)

        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow('HRI Scenario Detection', debug_image)

    cap.release()
    cv.destroyAllWindows()


# === Helper Functions ===

def detect_wave(point_history, hand_width):
    valid_points = [p for p in point_history if p != [0, 0]]
    if len(valid_points) < 8: return False, 0
    x_coords = [p[0] for p in valid_points]
    y_coords = [p[1] for p in valid_points]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    
    # Scale-invariant threshold: movement must be at least 30% of hand width (min 15 pixels)
    thresh = max(15, int(0.30 * hand_width))
    if x_range < thresh or x_range < y_range: return False, x_range
    
    direction_changes = 0
    for i in range(2, len(x_coords)):
        prev_diff = x_coords[i - 1] - x_coords[i - 2]
        curr_diff = x_coords[i] - x_coords[i - 1]
        if (prev_diff > 1 and curr_diff < -1) or (prev_diff < -1 and curr_diff > 1):
            direction_changes += 1
    return direction_changes >= 2, x_range


def detect_come_here(point_history, hand_height):
    valid_points = [p for p in point_history if p != [0, 0]]
    if len(valid_points) < 8: return False
    y_coords = [p[1] for p in valid_points]
    y_range = max(y_coords) - min(y_coords)
    
    # Scale-invariant threshold: movement must be at least 25% of hand height (min 15 pixels)
    thresh = max(15, int(0.25 * hand_height))
    if y_range < thresh: return False
    
    direction_changes = 0
    for i in range(2, len(y_coords)):
        prev_diff = y_coords[i - 1] - y_coords[i - 2]
        curr_diff = y_coords[i] - y_coords[i - 1]
        if (prev_diff > 1 and curr_diff < -1) or (prev_diff < -1 and curr_diff > 1):
            direction_changes += 1
    return direction_changes >= 2


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57: number = key - 48
    if key == 110: mode = 0
    if key == 107: mode = 1
    if key == 104: mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array,
                                   [np.array((landmark_x, landmark_y))], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


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
    return [n / max_value for n in temp_landmark_list]


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    if not temp_point_history: return []
    base_x, base_y = temp_point_history[0][0], temp_point_history[0][1]
    for index, point in enumerate(temp_point_history):
        temp_point_history[index][0] = (point[0] - base_x) / image_width
        temp_point_history[index][1] = (point[1] - base_y) / image_height
    return list(itertools.chain.from_iterable(temp_point_history))


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (0, 255, 0), 2)
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]),
                     1 + int(index / 2), (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30),
              cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
