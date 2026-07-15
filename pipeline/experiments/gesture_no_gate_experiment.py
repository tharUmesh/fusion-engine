"""
Investigation (report-only, per explicit instruction -- does NOT change
runners/gesture_runner.py or the 0.80 sensitive-gesture gate in production):

Quantifies how much of the gesture cue's ~36% insufficient-valid-frames rate
(457/1270 clips below the 0.40 valid_fraction threshold, see
reports/phase0_agreement.md) is caused specifically by the 0.80
sensitive-gesture confidence gate (test_video.py / gesture_runner.py:
`if hand_sign_id in [2, 3, 4, 5] and hand_sign_conf < 0.80: hand_sign_id = -1`).

Method: re-run the exact same per-frame pipeline gesture_runner.py uses
(imported from it unmodified -- only process_clip is duplicated here, with
the gate line removed), over the full 1270-clip dataset, in the same
.venvs/gesture environment. Compare per-clip valid_fraction / dominant_label
with the gate ON (already in pipeline/measured/gesture_frame_cues.jsonl) vs
OFF (this script's output), using the identical 0.40 threshold
aggregate_clip_cues.py already uses.

Output written to pipeline/experiments/ -- NEVER pipeline/measured/ -- so
this cannot be mistaken for, or accidentally overwrite, the real Phase 0
gesture output.
"""
import argparse
import copy
import os
import sys
import time
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNERS_DIR = os.path.join(REPO_ROOT, "runners")
sys.path.insert(0, RUNNERS_DIR)

from common.schema import NormalisedFrameCue, append_batch, read_manifest  # noqa: E402
from common.constants import CONFIDENCE_FLOOR, GESTURE_SCENARIO_TO_CANONICAL  # noqa: E402
import gesture_runner as gr  # noqa: E402 -- reused unmodified: helper fns, classifiers, GESTURE_REPO

CUE = "gesture"
FLOOR = CONFIDENCE_FLOOR[CUE]


def process_clip_no_gate(clip_path: str, keypoint_classifier, point_history_classifier, cv, np, mp):
    """Byte-for-byte copy of gesture_runner.process_clip, EXCEPT the 0.80
    sensitive-gesture gate (`if hand_sign_id in [2,3,4,5] and hand_sign_conf
    < 0.80: hand_sign_id = -1`) is never applied -- hand_sign_id is left as
    the raw keypoint-classifier output. Everything else (smoothing, point
    history, wave/come-here heuristics, global scenario resolution) is
    unchanged, so this isolates the gate's effect specifically."""
    hands = gr._setup(mp)

    history_length = 16
    smoothed_landmarks = {0: None, 1: None}
    alpha = 0.45
    point_history = {0: __import__("collections").deque(maxlen=history_length),
                      1: __import__("collections").deque(maxlen=history_length)}
    finger_gesture_history = {0: __import__("collections").deque(maxlen=history_length),
                                1: __import__("collections").deque(maxlen=history_length)}
    y_hist = {0: __import__("collections").deque(maxlen=25), 1: __import__("collections").deque(maxlen=25)}
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

        image = gr.resize_with_aspect_ratio(image, max_dim=960, cv=cv)
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

                raw_landmark_list = gr.calc_landmark_list(debug_image, hand_landmarks)
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

                brect = gr.calc_bounding_rect(debug_image, hand_landmarks, cv, np)
                pre_processed_landmark_list = gr.pre_process_landmark(smoothed_list)
                pre_processed_history_list = gr.pre_process_point_history(debug_image, point_history[hand_index])

                hand_sign_id, hand_sign_conf = keypoint_classifier(pre_processed_landmark_list)
                # <-- THE ONLY DELIBERATE DIFFERENCE FROM gesture_runner.process_clip:
                # the 0.80 sensitive-gesture gate is NOT applied here. hand_sign_id
                # stays as the raw classifier output regardless of hand_sign_conf.

                if hand_sign_id in [0, 1, 2, 5, -1]:
                    point_history[hand_index].append(smoothed_list[8])
                else:
                    point_history[hand_index].append([0, 0])

                finger_gesture_id, fg_conf = 0, 0.0
                wave_detected, wave_amplitude = gr.detect_wave(point_history[hand_index], brect[2] - brect[0])

                if len(pre_processed_history_list) == (history_length * 2):
                    if wave_detected:
                        finger_gesture_id, fg_conf = 4, 0.95
                    elif gr.detect_come_here(point_history[hand_index], brect[3] - brect[1]):
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

        global_scenario_text = "None"
        global_conf = 0.0
        num_hands = len(hand_states)

        if num_hands == 2:
            h1 = hand_states[list(hand_states.keys())[0]]
            h2 = hand_states[list(hand_states.keys())[1]]
            h1_raised = gr.check_hand_raised(y_hist[h1['id']], h1['hy'])
            h2_raised = gr.check_hand_raised(y_hist[h2['id']], h2['hy'])

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
            h1_raised = gr.check_hand_raised(y_hist[h1['id']], h1['hy'])

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


def run_batch(manifest_csv, clips_root, out_path, limit=None, resume=False):
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
        print(f"[gesture_no_gate] resuming: {len(done_ids)} clips already done")
    else:
        mode = "w"

    kc, phc, cv, np, mp = gr._load_classifiers()

    t0 = time.time()
    n_done = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i, row in enumerate(rows):
            clip_id = row["clip_id"]
            if clip_id in done_ids:
                continue
            clip_path = os.path.join(clips_root, row["filepath"])
            try:
                records = process_clip_no_gate(clip_path, kc, phc, cv, np, mp)
            except Exception as e:
                print(f"[gesture_no_gate] ERROR on {clip_id} ({clip_path}): {e}")
                continue
            append_batch(f, clip_id, records)
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(rows) - len(done_ids) - n_done) / rate if rate > 0 else float("inf")
                print(f"[gesture_no_gate] {i+1}/{len(rows)} clips ({n_done} this run, "
                      f"{rate:.2f} clips/s, ~{remaining/60:.1f} min remaining)")

    print(f"[gesture_no_gate] batch done: {n_done} clips processed -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clips-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_batch(args.manifest, args.clips_root, args.out, limit=args.limit, resume=args.resume)
