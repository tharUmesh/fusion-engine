"""
Standalone Gesture runner. Reimplements test_video.py's per-frame loop
headlessly (that script is GUI-coupled with no return value -- see
Integration_API.md #3.2), reusing the repo's own trained classifiers
(model.KeyPointClassifier, model.PointHistoryClassifier) and helper functions
verbatim, then emits one NormalisedFrameCue per frame from the global
scenario resolver (test_video.py's own 8-scenario logic, copied unmodified
except for the fix noted below).

Correctness fix applied here (see Integration_API.md #2.2 / MODEL_ANALYSIS.md
#2.9): test_video.py's pre_process_landmark lacks the divide-by-zero guard
present in app.py/play_video.py's copy of the same function -- a degenerate
all-landmarks-at-wrist frame would raise ZeroDivisionError and crash batch
processing. Patched here.

Must run with CWD (or with paths resolved) relative to the Gesture Repo,
since KeyPointClassifier/PointHistoryClassifier load their .tflite files via
CWD-relative paths, not paths relative to their own file location -- this
runner chdir()s into the Gesture Repo before importing them.

In batch mode, the two TFLite classifiers (stateless, expensive to reload)
are loaded ONCE and reused across all clips; the MediaPipe Hands tracker and
all per-hand deques/histories (stateful across frames) are recreated fresh
per clip, matching the native script's behaviour of always starting a new
process per video (see MODEL_ANALYSIS.md's per-frame state discussion).

Run inside .venvs/gesture (numpy==1.26.4, opencv-python==4.9.0.80,
mediapipe==0.10.11, tensorflow==2.15.1, protobuf==3.20.3 -- see
Integration_API.md #4).

Usage:
    # single clip
    .venvs/gesture/Scripts/python.exe runners/gesture_runner.py --clip <path> --out <out.jsonl>

    # batch mode: loads the classifiers ONCE, loops every clip in clips.csv
    .venvs/gesture/Scripts/python.exe runners/gesture_runner.py \
        --manifest Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/clips.csv \
        --clips-root Data/Dataset/hri-multimodal-intent-v1.0.0 \
        --out data/measured/gesture_frame_cues.jsonl
"""
import argparse
import copy
import itertools
import os
import sys
import time
from collections import Counter, deque

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS_DIR)
from common.schema import NormalisedFrameCue, write_jsonl, append_batch, read_manifest  # noqa: E402
from common.constants import CONFIDENCE_FLOOR, GESTURE_SCENARIO_TO_CANONICAL  # noqa: E402

CUE = "gesture"
FLOOR = CONFIDENCE_FLOOR[CUE]

GESTURE_REPO = os.path.join(os.path.dirname(RUNNERS_DIR), "Gesture Repo")


# ── Helper functions, copied verbatim from Gesture Repo/test_video.py ──────
# (see MODEL_ANALYSIS.md #2 for provenance; only pre_process_landmark is
# patched, as noted above)

def resize_with_aspect_ratio(image, max_dim, cv):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)


def check_hand_raised(y_history, current_y):
    if current_y < 0.35:
        return True
    if len(y_history) >= 10:
        y_list = list(y_history)
        first_half = y_list[:len(y_list) // 2]
        max_past_y = max(first_half)
        if (max_past_y - current_y) > 0.12 and current_y < 0.45:
            return True
    return current_y < 0.42


def detect_wave(point_history, hand_width):
    valid_points = [p for p in point_history if p != [0, 0]]
    if len(valid_points) < 8:
        return False, 0
    x_coords = [p[0] for p in valid_points]
    y_coords = [p[1] for p in valid_points]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    thresh = max(15, int(0.30 * hand_width))
    if x_range < thresh or x_range < y_range:
        return False, x_range
    direction_changes = 0
    for i in range(2, len(x_coords)):
        prev_diff = x_coords[i - 1] - x_coords[i - 2]
        curr_diff = x_coords[i] - x_coords[i - 1]
        if (prev_diff > 1 and curr_diff < -1) or (prev_diff < -1 and curr_diff > 1):
            direction_changes += 1
    return direction_changes >= 2, x_range


def detect_come_here(point_history, hand_height):
    valid_points = [p for p in point_history if p != [0, 0]]
    if len(valid_points) < 8:
        return False
    y_coords = [p[1] for p in valid_points]
    y_range = max(y_coords) - min(y_coords)
    thresh = max(15, int(0.25 * hand_height))
    if y_range < thresh:
        return False
    direction_changes = 0
    for i in range(2, len(y_coords)):
        prev_diff = y_coords[i - 1] - y_coords[i - 2]
        curr_diff = y_coords[i] - y_coords[i - 1]
        if (prev_diff > 1 and curr_diff < -1) or (prev_diff < -1 and curr_diff > 1):
            direction_changes += 1
    return direction_changes >= 2


def calc_bounding_rect(image, landmarks, cv, np):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [np.array((landmark_x, landmark_y))], axis=0)
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
    """Patched: adds the divide-by-zero guard present in app.py/play_video.py
    but missing from test_video.py's own copy (see MODEL_ANALYSIS.md #2.9)."""
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, _ in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 0
    if max_value == 0:
        return [0.0 for _ in temp_landmark_list]
    return [n / max_value for n in temp_landmark_list]


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    if not temp_point_history:
        return []
    base_x, base_y = temp_point_history[0][0], temp_point_history[0][1]
    for index, point in enumerate(temp_point_history):
        temp_point_history[index][0] = (point[0] - base_x) / image_width
        temp_point_history[index][1] = (point[1] - base_y) / image_height
    return list(itertools.chain.from_iterable(temp_point_history))


def _setup(mp):
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.45, min_tracking_confidence=0.45)


def process_clip(clip_path: str, keypoint_classifier, point_history_classifier, cv, np, mp):
    """Pure per-clip logic. Creates a fresh MediaPipe Hands tracker and all
    per-hand history buffers for this clip (see module docstring)."""
    hands = _setup(mp)

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

    cap = cv.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")
    orientation = cap.get(cv.CAP_PROP_ORIENTATION_META) if hasattr(cv, "CAP_PROP_ORIENTATION_META") else cap.get(48)

    records = []
    frame_idx = -1
    while True:
        ret, image = cap.read()
        if not ret:
            break
        frame_idx += 1

        if orientation == 90:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        elif orientation == 180:
            image = cv.rotate(image, cv.ROTATE_180)
        elif orientation == 270:
            image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

        image = resize_with_aspect_ratio(image, max_dim=960, cv=cv)
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        detected_hand_indices = []
        hand_states = {}

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_index = handedness.classification[0].index
                detected_hand_indices.append(hand_index)

                raw_landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                if smoothed_landmarks[hand_index] is None:
                    smoothed_landmarks[hand_index] = raw_landmark_list
                else:
                    smoothed_landmarks[hand_index] = [
                        [int(alpha * curr[0] + (1.0 - alpha) * prev[0]),
                         int(alpha * curr[1] + (1.0 - alpha) * prev[1])]
                        for curr, prev in zip(raw_landmark_list, smoothed_landmarks[hand_index])
                    ]
                smoothed_list = smoothed_landmarks[hand_index]

                for idx, pt in enumerate(smoothed_list):
                    hand_landmarks.landmark[idx].x = pt[0] / debug_image.shape[1]
                    hand_landmarks.landmark[idx].y = pt[1] / debug_image.shape[0]

                hy = np.mean([l.y for l in hand_landmarks.landmark])
                y_hist[hand_index].append(hy)

                brect = calc_bounding_rect(debug_image, hand_landmarks, cv, np)
                pre_processed_landmark_list = pre_process_landmark(smoothed_list)
                pre_processed_history_list = pre_process_point_history(debug_image, point_history[hand_index])

                hand_sign_id, hand_sign_conf = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id in [2, 3, 4, 5] and hand_sign_conf < 0.80:
                    hand_sign_id = -1

                if hand_sign_id in [0, 1, 2, 5, -1]:
                    point_history[hand_index].append(smoothed_list[8])
                else:
                    point_history[hand_index].append([0, 0])

                finger_gesture_id, fg_conf = 0, 0.0
                wave_detected, wave_amplitude = detect_wave(point_history[hand_index], brect[2] - brect[0])

                if len(pre_processed_history_list) == (history_length * 2):
                    if wave_detected:
                        finger_gesture_id, fg_conf = 4, 0.95
                    elif detect_come_here(point_history[hand_index], brect[3] - brect[1]):
                        finger_gesture_id, fg_conf = 5, 0.92
                    elif hand_sign_id == 2:
                        finger_gesture_id, fg_conf = point_history_classifier(pre_processed_history_list)

                finger_gesture_history[hand_index].append(finger_gesture_id)
                current_fg_id = Counter(finger_gesture_history[hand_index]).most_common()[0][0]

                hand_states[hand_index] = {
                    'sign': hand_sign_id, 'action': current_fg_id, 'brect': brect,
                    'wave_amp': wave_amplitude, 'sign_conf': hand_sign_conf,
                    'action_conf': fg_conf, 'hy': hy, 'id': hand_index,
                }

        for i in range(2):
            if i not in detected_hand_indices:
                point_history[i].append([0, 0])
                smoothed_landmarks[i] = None
                y_hist[i].clear()

        # === Global Scenario Resolution (copied verbatim from test_video.py) ===
        global_scenario_text = "None"
        global_conf = 0.0
        num_hands = len(hand_states)

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
                global_scenario_text = "Wave" if h1['wave_amp'] > 150 else "Brief wave"
                global_conf = h1['action_conf']
            elif h1['action'] == 5 or h1['sign'] == 5:
                global_scenario_text = "Beckoning"
                global_conf = max(h1['action_conf'], h1['sign_conf'] if h1['sign'] == 5 else 0.0)
            elif h1['sign'] == 2:
                global_scenario_text, global_conf = "Pointing", h1['sign_conf']
            elif h1['sign'] == 3:
                global_scenario_text, global_conf = "Thumbs up", h1['sign_conf']
            elif h1['sign'] == 4:
                global_scenario_text, global_conf = "Thumbs down", h1['sign_conf']
            elif h1['sign'] in [0, 1, -1] and h1_raised:
                global_scenario_text = "One hand raised"
                global_conf = h1['sign_conf'] if h1['sign'] != -1 else 0.85
            else:
                global_scenario_text = "None"

        label = GESTURE_SCENARIO_TO_CANONICAL.get(global_scenario_text, "Unknown")
        confidence = float(global_conf) if global_scenario_text != "None" else 0.0

        records.append(NormalisedFrameCue(
            cue=CUE, frame_idx=frame_idx, label=label, confidence=confidence,
            probs={}, valid=(label != "Unknown" and confidence >= FLOOR),
            extra={"point_direction": None, "motion_direction": "none", "point_target": "unknown"}))

    cap.release()
    hands.close()
    return records


def _load_classifiers():
    os.chdir(GESTURE_REPO)  # KeyPointClassifier/PointHistoryClassifier use CWD-relative paths
    sys.path.insert(0, GESTURE_REPO)
    import cv2 as cv
    import numpy as np
    import mediapipe as mp
    from model import KeyPointClassifier, PointHistoryClassifier
    return KeyPointClassifier(), PointHistoryClassifier(), cv, np, mp


def run_single(clip_path: str, out_path: str):
    clip_path = os.path.abspath(clip_path)
    out_path = os.path.abspath(out_path)
    kc, phc, cv, np, mp = _load_classifiers()
    records = process_clip(clip_path, kc, phc, cv, np, mp)
    write_jsonl(records, out_path)
    print(f"[gesture_runner] {len(records)} frames -> {out_path}")


def run_batch(manifest_csv: str, clips_root: str, out_path: str, limit=None, resume=False):
    manifest_csv = os.path.abspath(manifest_csv)
    clips_root = os.path.abspath(clips_root)
    out_path = os.path.abspath(out_path)

    rows = read_manifest(manifest_csv)
    if limit:
        rows = rows[:limit]

    done_ids = set()
    mode = "a"
    if resume and os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(line.split('"clip_id": "', 1)[1].split('"', 1)[0])
                except IndexError:
                    pass
        print(f"[gesture_runner] resuming: {len(done_ids)} clips already done")
    else:
        mode = "w"

    kc, phc, cv, np, mp = _load_classifiers()

    t0 = time.time()
    n_done = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i, row in enumerate(rows):
            clip_id = row["clip_id"]
            if clip_id in done_ids:
                continue
            clip_path = os.path.join(clips_root, row["filepath"])
            try:
                records = process_clip(clip_path, kc, phc, cv, np, mp)
            except Exception as e:
                print(f"[gesture_runner] ERROR on {clip_id} ({clip_path}): {e}")
                continue
            append_batch(f, clip_id, records)
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(rows) - len(done_ids) - n_done) / rate if rate > 0 else float("inf")
                print(f"[gesture_runner] {i+1}/{len(rows)} clips ({n_done} this run, "
                      f"{rate:.2f} clips/s, ~{remaining/60:.1f} min remaining)")

    print(f"[gesture_runner] batch done: {n_done} clips processed -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="single-clip mode: path to one clip")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--manifest", help="batch mode: path to clips.csv")
    ap.add_argument("--clips-root", help="batch mode: dataset root (filepath column is relative to this)")
    ap.add_argument("--limit", type=int, default=None, help="batch mode: only process first N rows (testing)")
    ap.add_argument("--resume", action="store_true", help="batch mode: skip clip_ids already present in --out")
    args = ap.parse_args()

    if args.manifest:
        if not args.clips_root:
            raise SystemExit("--clips-root is required with --manifest")
        run_batch(args.manifest, args.clips_root, args.out, limit=args.limit, resume=args.resume)
    else:
        if not args.clip:
            raise SystemExit("either --clip or --manifest is required")
        run_single(args.clip, args.out)
