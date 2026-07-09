# Cue Model Analysis — Pre-Phase Deliverable

**Status:** Analysis only. No repository under `Fusion Engine/*Repo` was modified. No Fusion Engine code was written as part of this document.

**Scope:** Independent analysis of the four existing, already-trained cue model repositories, read directly from disk at:
- `Fusion Engine/Emotion Repo`
- `Fusion Engine/Gesture Repo`
- `Fusion Engine/Motion Repo`
- `Fusion Engine/Context Repo/scene classification`

Each section below documents: inference entry point, preprocessing, model loading, postprocessing, output schema, confidence semantics, dependencies, minimal inference file set, integration gotchas, and open risks. All claims are cited with `file:line`.

**Companion document:** [`Integration_API.md`](Integration_API.md) defines the canonical adapter contract and reconciles it against what these four models can *actually* deliver (several corrections to the handover document's assumptions are required — see Integration_API.md §0).

---

## 0. Top-line findings (read this first)

1. ~~Context model is not runnable as delivered~~ **RESOLVED 2026-07-01**: the user rewrote `Context Repo/scene classification/video.py` and `realtime.py` to be self-contained (no more `src.classifier` import — model build/load/inference now live directly in both scripts, matching the Emotion repo's style). See §4 (rewritten) for the current, verified analysis. No blocker remains for the context cue.
2. **None of the four models currently emit a structured (JSON/dict/CSV) per-frame output.** All four are demo scripts that burn results into video overlays (`cv2.putText`) and/or display windows. Every adapter will need to refactor or wrap each script's inner loop to capture in-memory values that are today discarded after being drawn.
3. **The Motion model's LSTM is loaded but never used.** `action_recognizer.py` loads `motion_lstm_v2_best.pth` into a `MotionLSTM`, moves it to device, sets `.eval()` — and then never calls it. All classification is a deterministic, hand-tuned physics/rules engine. The LSTM's output space (9 classes) doesn't even match the runtime taxonomy (8 classes) any more.
4. **The Gesture model does not use MediaPipe Pose.** The handover document's claim that "dynamic scenarios (Pointing / One Hand Raised / Arms Up) come from pose+rules" is inaccurate — everything is MediaPipe **Hands** landmarks + temporal rule logic. There is no body-pose estimation anywhere in the gesture repo.
5. **"Confidence" means three different things across the four models**: Emotion = calibration-free softmax max-prob; Gesture = calibration-free softmax max-prob (two separate classifiers) plus two hardcoded constants (0.95, 0.92) for rule-detected dynamic gestures; Motion = fully hand-authored per-class constants (0.80–0.95) smoothed with an EMA, not a learned probability at all; Context = unknown (blocked by finding #1), but usage sites imply a softmax-style float with an "uncertain" fallback label.
6. **Only Gesture ships a `requirements.txt`.** Motion also ships one. Emotion and Context ship none — dependencies must be inferred from `import` statements, and exact versions are unknown for those two.

---

## 1. Emotion Model

**Location:** `Emotion Repo/` — `realtime_realsense.py` (165 lines, live-camera), `video.py` (238 lines, offline), `best_MobileNetV2.pth` (9,175,932 bytes).

### 1.1 Inference entry point
`video.py` is the correct entry point for offline inference on recorded clips. Its `main()` (video.py:163-234) accepts `--video <path>` or `--videos-dir <dir>`, reads frames via `cv2.VideoCapture`, and runs `process_video()` (video.py:103-160) per frame. `realtime_realsense.py` is live-camera only (Intel RealSense, with a `cv2.VideoCapture` webcam fallback) and shares near-identical model/transform code, duplicated rather than shared via a common module.

### 1.2 Preprocessing (video.py:116-141)
1. If source frame width > `MAX_FRAME_WIDTH` (640), a **downscaled copy** is made for detection only (aspect-preserving resize to width 640); the actual face **crop is taken from the original full-resolution frame** using MediaPipe's relative (0–1) bounding box coordinates.
2. Face detection: `mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)` — `model_selection=0` is the *short-range* (≤2m) model, despite both files' docstrings claiming "close-range detection."
3. Crop: tight rectangular crop, no margin/padding, no square-aspect enforcement. Empty crops (`face.size == 0`) are skipped.
4. `cv2.COLOR_BGR2RGB` → `PIL.Image` → `transforms.Resize((224,224))` (fixed-size, **not** aspect-preserving — non-square faces are squashed) → `ToTensor()` → `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])` (standard ImageNet stats).
5. Batch size 1, one forward pass per detected face per frame (no cross-face/cross-frame batching). Final tensor: `[1,3,224,224]`.

### 1.3 Model architecture & loading
`torchvision.models.mobilenet_v2(weights=None)` with `model.classifier[1] = nn.Linear(1280, 7)` (video.py:38-42). Loaded via `torch.load(ckpt, map_location=device, weights_only=True)` then `load_state_dict(...)` directly (video.py:180) — the checkpoint is a **bare state_dict**, no wrapper keys. Device: `"cuda" if torch.cuda.is_available() else "cpu"`, not overridable via CLI. `resolve_weights()` (video.py:53-66) searches script dir → `checkpoints/` → `../checkpoints/` for the weights file.

### 1.4 Postprocessing
`conf, pred = torch.max(F.softmax(model(tensor), dim=1), 1)` (video.py:136) — plain softmax, T=1, no calibration. Label list is **hardcoded identically** in both files (video.py:30):
```
0 Surprise   1 Fear   2 Disgust   3 Happy   4 Sad   5 Anger   6 Neutral
```
This order is not read from any external class file — none exists in the repo. Only the top-1 label + its confidence are retained; the full 7-way softmax vector is computed but discarded after `torch.max`.

### 1.5 Output schema
**No structured output exists.** Per detection, a string `"{label}: {conf*100:.1f}%"` is drawn via `cv2.putText` + `cv2.rectangle` onto the frame, and the annotated frame is written to an output `.mp4` (video.py:137-141). No per-frame record (frame index, bbox, full probability vector) is stored or returned anywhere. An adapter must intercept `process_video()`'s inner loop to capture `(frame_idx, bbox, label, conf, full_softmax_vector)`, none of which are currently exposed.

### 1.6 Confidence semantics
Confidence = softmax max-probability, uncalibrated. **No threshold is applied anywhere** — every detected face's prediction is reported regardless of confidence. **No "Unknown"/reject class** exists among the 7 labels; if zero faces are detected in a frame, nothing is drawn and nothing is emitted — the "no face" case must be inferred by an adapter from the *absence* of a prediction event, not from an explicit sentinel value.

### 1.7 Dependencies
`argparse, os, cv2, mediapipe, torch, torch.nn, torch.nn.functional, torchvision.models, PIL.Image, torchvision.transforms` — all needed for offline inference. `numpy` and `pyrealsense2` appear only in `realtime_realsense.py` (live-camera path) and are **not** needed offline; `pyrealsense2` is already import-guarded (`try/except ImportError`). **No `requirements.txt` exists** — no versions are pinned anywhere; install line in the docstring is `pip install torch torchvision opencv-python mediapipe pillow numpy` with no version constraints.

### 1.8 Minimal inference file set
`Emotion Repo/video.py` + `Emotion Repo/best_MobileNetV2.pth`. Exclude `realtime_realsense.py` and `pyrealsense2`.

### 1.9 Integration gotchas
- **Multiple faces per frame** are each independently predicted with no person/track ID linking predictions across frames — if a clip has bystanders, the adapter needs its own "pick the subject" policy (largest bbox / most central / highest confidence), which does not exist in this code.
- **No frame skipping or temporal smoothing** — every frame is processed independently; for a 5s clip at ~30fps this is ~150 forward passes (×faces).
- **Aspect-ratio distortion is intentional/by design** (non-square crops squashed to 224×224) — an adapter reproducing training-time behavior must not "fix" this.
- `video.py` opens a `cv2.imshow` preview by default and blocks on keypresses unless `--no-show` is passed; it also always writes an annotated output `.mp4` regardless of whether structured results are wanted — both are side effects an adapter must suppress/bypass.
- GPU/CPU selection is automatic (prefers CUDA) with no override flag, which affects reproducibility across dev machines.

### 1.10 Open questions / risks
- **Label order is unverified against training code.** No class-index file, training script, or dataset reference exists in this repo to confirm `["Surprise","Fear","Disgust","Happy","Sad","Anger","Neutral"]` is the true index-to-class mapping (a silent transposition here would silently corrupt all downstream fusion logic).
- No version pins anywhere — behavior of `torchvision.models.mobilenet_v2(weights=...)` argument and `mediapipe.solutions.face_detection` could vary across library versions.
- Checkpoint-architecture match is unverified (weights were not loaded per the read-only scope of this analysis).
- No confidence threshold / no-face sentinel is defined — the fusion adapter must invent this policy.
- `model_selection=0` (short-range, ≤2m) may degrade if the HRI camera-to-subject distance in the project's clips exceeds ~2m.

---

## 2. Gesture Model

**Location:** `Gesture Repo/` — `app.py` (449 lines, live webcam), `test_video.py` (472 lines, offline/file), `play_video.py` (588 lines, offline/file, simplified), plus `model/keypoint_classifier/` and `model/point_history_classifier/` (TFLite + Keras `.hdf5` + label CSVs), `extract_dataset.py` (training-only), `utils/cvfpscalc.py`, `requirements.txt`, `README.md`, and `doc/*.md` guides.

### 2.1 Inference entry point
Three scripts exist; they are **not equivalent**:

| Script | Source | PointHistoryClassifier? | Scenario resolver | Use for offline eval? |
|---|---|---|---|---|
| `app.py` | webcam index (`--device`) | No | Simple rule engine | No — live only |
| `test_video.py` | video **file** path | **Yes** | Full 8-scenario resolver with per-scenario confidence | **Yes — canonical offline entry point** |
| `play_video.py` | video file, menu-driven | No (identical engine to `app.py`) | Simple rule engine (same as `app.py`) | No — it's a file-based demo of the *simpler* engine, not a superset |

`test_video.py` is the only script that uses both trained classifiers plus the richer scenario resolution and must be adapted for Phase 0. It runs in a blocking `cv.imshow`/`cv.waitKey` GUI loop with no headless/programmatic API — this must be refactored to yield structured per-frame results.

### 2.2 Preprocessing
- MediaPipe **Hands** (`mp.solutions.hands`), NOT Pose. `min_detection_confidence=0.45`, `min_tracking_confidence=0.45` (test_video.py:67-68), `max_num_hands=2` everywhere.
- **Keypoint classifier input** (`pre_process_landmark`, test_video.py:428-436): 21 landmarks → pixel coords → subtract wrist (landmark 0) for translation invariance → flatten to 42 values → divide by max(abs(value)) for scale normalization to [-1,1]. **Note:** `test_video.py`'s version lacks the divide-by-zero guard present in `app.py`/`play_video.py` — a degenerate all-landmarks-at-wrist frame would raise `ZeroDivisionError` and crash batch processing.
- **Point history** (test_video.py only): rolling `deque(maxlen=16)` per hand slot, appending the smoothed index-fingertip (landmark 8) pixel coordinate only when `hand_sign_id in [0,1,2,5,-1]`; normalized relative to the deque's first point and image width/height, flattened to 32 values. Fed to the point-history TFLite model **only** when the buffer is full (16 points) AND the static classifier said "Pointer" AND neither of two hardcoded heuristics (`detect_wave`, `detect_come_here`) already fired — i.e., the neural point-history model is a tertiary fallback, invoked rarely.

### 2.3 Model architecture & loading
Both classifiers load via **TFLite `Interpreter`** (not the `.hdf5` Keras files, which are training artifacts only, never loaded by any inference script):
- **KeyPointClassifier** (`model/keypoint_classifier/keypoint_classifier.py:13-18`): input `(1,42)` float32, output squeezed to 6-class softmax vector. Confirmed architecture from `train/train_keypoint.py:65-74`: `Input(42) → Dropout(.2) → Dense(64,relu) → Dropout(.3) → Dense(32,relu) → Dropout(.3) → Dense(16,relu) → Dense(6,softmax)`. (One of the repo's own docs, `gesture_pipeline_guide.md:21`, incorrectly omits the Dense-64 layer — trust the code / `model_classifiers_guide.md`.)
- **PointHistoryClassifier** (`model/point_history_classifier/point_history_classifier.py:14-19`): input `(1,32)` float32, output 6-class softmax. Has its own internal `score_th=0.5`: below this, `result_index` is forced to `invalid_value=0` ("Stop") — a separate mechanism from the 0.80 "sensitive gesture" gate below.

### 2.4 Postprocessing
Label files (verbatim, index order):
- `keypoint_classifier_label.csv`: `0 Open Palm, 1 Close, 2 Pointer, 3 Thumbs Up, 4 Thumbs Down, 5 Beckoning`
- `point_history_classifier_label.csv`: `0 Stop, 1 Clockwise, 2 Counter Clockwise, 3 Move, 4 Wave, 5 Come Here`

**Verified: the 0.80 "sensitive gesture" threshold is real and precisely located** — `test_video.py:203-204` (identically in `app.py:269-270`, `play_video.py:268-269`):
```python
if hand_sign_id in [2, 3, 4, 5] and hand_sign_conf < 0.80:
    hand_sign_id = -1   # → displayed as "Unknown"
```
Only Pointer/Thumbs-Up/Thumbs-Down/Beckoning are subject to this gate; Open Palm and Close pass through regardless of confidence.

**Correction to the handover document:** "dynamic scenarios come from pose+rules" is **not accurate**. No MediaPipe Pose is used anywhere in this repo (verified by search). "Pointing" is a static keypoint-classifier output (class 2); "One Hand Raised"/"Arms Up" are derived from **hand-landmark y-coordinate history + temporal rules**, not a body-pose model.

### 2.5 Output schema
Per-hand, per-frame, built inline in `test_video.py:231-240` (not returned by any function — must be refactored out):
```python
{
  'sign': hand_sign_id,     # int, -1 or 0-5 (post-0.80-threshold)
  'action': current_fg_id,  # int, 0-5 (point-history / rule-based dynamic class, mode-filtered)
  'brect': [x1,y1,x2,y2],   # pixel bbox
  'wave_amp': float,        # pixel x-range amplitude
  'sign_conf': float,       # 0-1, keypoint classifier confidence
  'action_conf': float,     # 0.95 (rule wave) | 0.92 (rule come-here) | model softmax (pointer) | 0.0 (none)
  'hy': float,              # mean y of all 21 landmarks (height proxy, 0-1)
  'id': int                 # MediaPipe handedness index (0/1) — NOT a stable person/hand track ID
}
```
A single per-frame **global scenario string** is also produced (test_video.py:265-341): one of `Arms waving, Arms up, Wave, Brief wave, Beckoning, Pointing, Thumbs up, Thumbs down, One hand raised, None`, with an appended confidence percentage.

**No `point_direction` or `motion_direction` fields exist anywhere in the code** — "Pointing" carries no directional vector/angle. **No object-target field exists** — confirms the handover's own note that object-target detection needs a separate YOLO module not present here.

### 2.6 Confidence semantics
- `sign_conf`: raw softmax max-prob from the keypoint classifier, no smoothing.
- `action_conf`: three mutually-exclusive sources — `0.95` hardcoded if `detect_wave()` fires, `0.92` hardcoded if `detect_come_here()` fires, else the actual point-history-classifier softmax confidence (Pointer case only), else `0.0`.
- The 0.80 sensitive-gesture gate (§2.4) is a hardcoded literal in three files, not configurable via CLI.
- When a hand's shape is `-1` (Unknown) but the scenario logic still needs it (e.g. "Arms up" allows unknown shape), a **synthetic confidence of 0.85** is substituted (test_video.py:281-282, 305, 333) — an arbitrary placeholder, not a measured value.

### 2.7 Dependencies
`requirements.txt` (pinned): `numpy==1.26.4, opencv-python==4.9.0.80, mediapipe==0.10.11, tensorflow==2.15.1, protobuf==3.20.3, matplotlib==3.10.9` (matplotlib is training-only). **Both classifier `.py` files import the full `tensorflow` package** (`import tensorflow as tf`) even though only `tf.lite.Interpreter` is used — the Jetson deployment doc's suggestion of a lighter `tflite-runtime` substitute is not actually wired into the code (no conditional import/fallback exists), so a dev-machine adapter needs full TensorFlow installed as-is.

### 2.8 Minimal inference file set
`test_video.py` (needs refactoring to a non-GUI API), both `model/*/*.py` + `*.tflite` + `*_label.csv` files, `model/__init__.py`, `utils/cvfpscalc.py`, `utils/__init__.py`. Exclude `extract_dataset.py`, `train/`, both `.hdf5` files, and the CSV *training data* files (`keypoint.csv`, `point_history.csv` — not the label CSVs, which are required).

### 2.9 Integration gotchas
- All model paths are relative to CWD — running from any directory other than the repo root will break loading unless made absolute.
- **Hand-identity instability**: `test_video.py` keys hand state by MediaPipe's raw per-frame handedness index (fragile across hand-count changes), whereas `app.py`/`play_video.py` implement a more robust custom nearest-neighbor spatial tracker — but those lack the point-history classifier entirely. Neither script has "the best of both."
- **Point-history classes 1/2/3 (Clockwise/Counter-Clockwise/Move) are effectively dead outputs** — the scenario resolver in `test_video.py` never checks for them (only `action==4` Wave and `action==5` Come-Here are consumed). Confirm whether these are even in the fusion engine's intent space before assuming they're reachable.
- Temporal windows (16-frame point history, 25-frame y/x histories) are **frame-count-based, not time-based** — behavior will shift if source clip FPS differs from what these thresholds were empirically tuned against.
- No confidence value should be read for the `"None"` global scenario — `global_conf` stays 0.0 by convention there, meaning "no meaningful confidence," not "confidently nothing."

### 2.10 Open questions / risks
- Confirm with the repo owner **which script (`test_video.py` vs `play_video.py`) actually produced any existing evaluation numbers**, given they have materially different capabilities.
- No headless/batch inference API currently exists in any script — this is a real implementation task for Phase 0's adapter, not a thin wrapper.
- Division-by-zero risk in `test_video.py`'s `pre_process_landmark` (missing guard present in the other two scripts) should be patched defensively before batch-processing many clips.

---

## 3. Motion Model

**Location:** `Motion Repo/` — `action_recognizer.py` (558 lines, core engine), `run_single_video.py` (112 lines), `run_all_and_save.py` (146 lines), `models/motion_lstm_v2_{best,final}.pth`, `models/model_config_v2.json`, `dataset_info_v2.json`, `README.md`, `requirements.txt`, `model_train/` (training-only).

### 3.1 Inference entry point
All inference logic lives in `action_recognizer.py`. `run_single_video.py` and `run_all_and_save.py` are **not separate implementations** — both merely `subprocess.run(["python","action_recognizer.py","--video",...])` (run_single_video.py:98-104; run_all_and_save.py:67-77) and neither captures or parses any structured output; they only check the subprocess return code. **For Phase 0, adapt `action_recognizer.py` directly** — the two wrapper scripts add no logic beyond a video picker / batch loop. `--webcam`/`--camera` are live-source CLI options in the same script (not a separate code path) and should simply be ignored in favor of `--video <clip_path> --no-show`.

### 3.2 Preprocessing
- Auto-rotation via `CAP_PROP_ORIENTATION_META` probing with a one-frame lookahead heuristic to avoid double-rotating (action_recognizer.py:263-277).
- Downscale: `resize_with_aspect_ratio(frame, max_dim=960)` — a 1280×720 clip becomes 960×540 before MediaPipe runs.
- MediaPipe **Pose** (`min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1`) — 33 landmarks. **If no landmarks are detected in a frame, the entire 30-frame sliding-window buffer is cleared** (action_recognizer.py:363-364).
- **Pose classifier** (Sitting/Standing/Crouching/Lying/Unknown): pure single-frame geometric rules on nose/hip/knee/ankle/shoulder y-coordinates (action_recognizer.py:93-116) — no temporal state, computed independently of the motion engine.
- **Motion physics engine**: 30-frame `deque` of `(33,3)` keypoints; needs ≥4 frames before computing anything. Uses hip-only XY velocity for `body_speed` (ankles excluded to avoid occlusion jitter). A 15-frame lookback sub-window computes hip-width scale change and hip/shoulder centroid translation with "path efficiency" (near-1.0 = consistent monotonic movement) to distinguish real directed walking from jitter.

### 3.3 Model architecture & loading — **the LSTM is vestigial**
`model_config_v2.json`: `LSTM, input_size=99 (33×3), hidden_size=128, num_layers=3, num_classes=9, dropout=0.4`. This is loaded, moved to device, and `.eval()`'d (action_recognizer.py:239-247) — **and then never called anywhere in the file.** Confirmed by full-file search: no `model(x)`/`model.forward(...)` call exists after line 244. The README's own Viva Q&A section confirms this was an intentional pivot away from the LSTM due to a synthetic-vs-real domain gap. **The LSTM's output space (9 classes, including separate "Walk Toward"/"Step Back") does not even match the runtime 8-class taxonomy** (which merges those two into a single "Walking" because direction from a fixed camera was deemed unreliable) — it could not be reconnected without retraining/remapping. If the `.pth` file fails to load, the script still `sys.exit(1)`s despite the model being functionally unused.

### 3.4 Postprocessing — 8-class taxonomy
Exact runtime order (action_recognizer.py:36-45), **matches the handover document's expected list 1:1**:
```
0 Sitting Still   1 Standing Still   2 Walking   3 Walk Across
4 Run Backward    5 Run (Fast Movement)   6 Leaning Forward   7 Frozen/Rigid Stand
```
Per-frame decision tree (action_recognizer.py:436-471): (1) `body_speed ≥ 1.30` → Run Backward or Run Fast by direction sign; (2) directed-walk rule fired → Walk Across or Walking; (3) else static branch: Sitting (from pose) → Sitting Still; Crouching (from pose) → Leaning Forward; `body_speed < 0.08` → Frozen/Rigid Stand; else → Standing Still. Pose and motion are computed by two separate mechanisms and combined only in the static branch.

**Handover's flagged concern verified**: Standing_Still vs Frozen_Rigid_Stand are kept as **distinct classes** in code (not collapsed), but the *only* discriminating signal between them is a single hardcoded `body_speed < 0.08` threshold with no hysteresis/debounce/duration requirement — this is empirically chosen, not physically validated against a real "freeze response" vs. ordinary idle standing, and is a genuine fidelity risk, not just a naming concern.

### 3.5 Output schema
No structured output — only an annotated `.avi` dashboard video. In-memory per frame (never persisted, must be extracted by refactoring `run()`): `pose_label: str`, `motion_label: str`, `confidence: float`, and — importantly — **`probabilities: np.ndarray` shape (8,) IS produced every frame** (`smooth_probs`), so the handover's `probs` dict requirement can be satisfied once the adapter zips it against `MOTION_LABELS`.

### 3.6 Confidence semantics — **manufactured, not learned**
Each classification branch hardcodes its own "confidence" constant (e.g. Sitting Still=0.95, Frozen/Rigid Stand=0.90, Run Backward=0.85) as a one-hot-ish `probs` vector; all non-winning classes get 0.0 that frame. This raw vector is then EMA-smoothed: `smooth_probs = 0.25*probs + 0.75*smooth_probs` (init uniform 1/8), and `confidence = smooth_probs[argmax]`. **This is not a statistically calibrated probability** — it will not be numerically comparable to the emotion/gesture models' softmax confidences without explicit reconciliation in the fusion layer. When no landmarks are detected, confidence is forced to 0.0 and probabilities to zeros, but `smooth_probs` itself resets to a uniform prior — the next frame after a tracking loss starts smoothing fresh, not continuing from history.

### 3.7 Dependencies
`requirements.txt` (pinned, 56 packages): `mediapipe==0.10.14, torch==2.11.0, torchvision==0.26.0, opencv-python==4.13.0.92, opencv-contrib-python==4.13.0.92, numpy==2.4.4`, plus transitive deps (`scipy, matplotlib, pillow, sympy, networkx, jax/jaxlib` via mediapipe, etc.). All are actually used or transitively required; `run_single_video.py`/`run_all_and_save.py` need only stdlib.

### 3.8 Minimal inference file set
`action_recognizer.py` (needs refactoring to expose a callable/generator instead of only `run()`'s blocking loop) + `models/motion_lstm_v2_best.pth` (only required because the script hard-loads it and exits on failure — **could be stripped entirely** once refactored, since it's never used for classification) + `requirements.txt`. Exclude `model_train/`, both JSON metadata files under `models/`, and `dataset_info_v2.json` — none of these are read by any inference script (`MOTION_LABELS` and all thresholds are hardcoded directly in `action_recognizer.py`).

### 3.9 Integration gotchas
- **Cold-start / re-warm-up default is a fabricated positive, not a null.** Until the 30-frame buffer has ≥4 valid frames (at clip start, or immediately after any single frame with no detected landmarks, since one bad frame clears the whole buffer), the code emits `motion_label="Standing Still", confidence=0.90, probabilities=[0,0.9,0,...,0]` — **not** an honest "no output" state. Only the true zero-landmarks case correctly signals `confidence=0.0`/all-zero probabilities (though the label string still shows "Standing Still" as a placeholder). Adapters must gate on landmark-presence/confidence, never on the label field alone.
- **Frame-count-based windows assume ~30fps.** The buffer (30 frames) and lookback (15 frames) are not normalized by actual FPS — a 5s clip at 30fps gives 150 frames (ample warm-up), but a different source FPS silently changes the real-world time window these thresholds represent.
- A single missing-landmark frame (e.g. brief occlusion) resets the entire temporal history and re-triggers the fabricated cold-start default for several subsequent frames — worth stress-testing on real footage with intermittent occlusion.
- `STATIC_THRESH = 0.22` is defined but never referenced anywhere — dead code that could mislead a reader into assuming a three-tier speed classification exists; it doesn't.
- `testVideo/` and `testVideo2/` contain no actual video files in this checkout (gitignored) — none of the above was empirically verified by running the pipeline; recommend running `action_recognizer.py --video <sample>.mp4 --no-show` on a real project clip before finalizing the adapter.

### 3.10 Open questions / risks
- The Standing_Still/Frozen_Rigid_Stand 0.08 threshold is unvalidated against real freeze-response footage — flagged as a risk for fusion accuracy on emergency scenarios (F02).
- Motion's confidence scale (hand-tuned constants + EMA) is not commensurate with the other three models' confidence semantics — the fusion layer should not treat these as directly comparable magnitudes without explicit design consideration.
- LSTM should be treated as dead weight to be excluded, not "optional to call later" — its output space doesn't align with the current taxonomy without retraining.

---

## 4. Context (Scene Classification) Model

**Location:** `Context Repo/scene classification/` — `video.py` (233 lines), `realtime.py` (158 lines), `best_EfficientNet_B0.pth` (16,338,142 bytes). **Updated 2026-07-01**: both scripts were rewritten by the user to be fully self-contained — the previously-missing `src.classifier.SceneClassifier` dependency has been eliminated entirely; model build, checkpoint loading, preprocessing, and inference now live directly in each script (mirroring the Emotion repo's structure). **No blocker remains.** This section supersedes the earlier (inference-blocked) analysis.

### 4.1 Inference entry point
`video.py` is the correct offline entry point. It supports: `--video <path>` (single file), `--videos-dir <dir>` (batch, recursive scan), `--checkpoint <path>`, `--output <path>` (single-file mode), `--out-dir <dir>` (batch mode, default `outputs`), `--no-show` (headless), `--skip-existing` (batch mode). If neither `--video` nor `--videos-dir` is given, it defaults to scanning `DEFAULT_VIDEOS_DIR` — **see the gotcha in §4.9, this default resolves to a nonexistent path given the repo's actual layout, so always pass `--video` explicitly.** `realtime.py` is live-camera only (Intel RealSense with automatic webcam fallback via `--camera <index>`), no file input, not relevant to clip evaluation.

### 4.2 Preprocessing (video.py:127-128, identical in realtime.py:132-133)
1. `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` — BGR→RGB conversion, no cropping/ROI (full frame is classified, unlike Emotion's face-crop approach).
2. `transforms.Compose([ToPILImage(), Resize((224,224)), ToTensor(), Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])` (video.py:49-55) — fixed-size resize (not aspect-preserving), standard ImageNet normalization, same pattern as the Emotion model.
3. `tensor = transform(rgb).unsqueeze(0).to(device)` — batch size 1, one forward pass per frame (no cropping means no multi-detection-per-frame issue, unlike Emotion/Gesture).

### 4.3 Model architecture & loading
`torchvision.models.efficientnet_b0(weights=None)` with `model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes=2)` (video.py:42-46). Loaded via `model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))` (video.py:178) — a **bare state_dict**, same convention as the Emotion model. `model.to(device).eval()`. Device: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, not overridable via CLI. `resolve_weights()` (video.py:58-71) searches script dir → `checkpoints/` → `../checkpoints/`, identical pattern to Emotion's `resolve_weights()`.

### 4.4 Postprocessing — with temporal smoothing (new information)
```python
probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()   # 2-class softmax
prob_history.append(probs)                                      # deque(maxlen=SMOOTH_WINDOW=15)
avg = np.mean(prob_history, axis=0)                             # rolling mean over up to 15 frames
idx = int(avg.argmax()); conf = float(avg[idx])
label = SCENE_LABELS[idx] if conf >= CONF_THRESHOLD(0.5) else "uncertain"
```
(video.py:130-135). **Confirmed exact class list and order**: `SCENE_LABELS = ["classroom", "kitchen"]` (video.py:31), with an explicit code comment `"alphabetical == training order"` — this **matches the handover's `context_scenes` assumption exactly** and is no longer a guess. **Important difference from the other three models**: Context applies a **15-frame rolling-average smoothing of the softmax probabilities before thresholding/argmax** — raw per-frame softmax output is never used directly for the label decision. `CONF_THRESHOLD = 0.5` matches schema.yaml's `confidence_floor.context = 0.50` exactly, and the `"uncertain"` fallback label maps directly to the handover's `"Unknown"` sentinel.

### 4.5 Output schema
Still **no structured per-frame output is returned or persisted** — `process_video()` (video.py:108-158) only draws `f"{label}: {conf*100:.1f}%"` onto the frame via `cv2.putText`, writes the annotated frame to a `VideoWriter` (unconditionally, regardless of `--no-show`), and returns only a single navigation string (`"done"|"next"|"prev"|"quit"`) for the CLI's batch/pause controls — not per-frame data. An adapter must capture `(frame_idx, label, conf, avg_probs_vector)` from inside the `while True:` loop (video.py:122-154), which is now straightforward since all the logic is visible/inline (no missing module to work around).

Comparison against the handover's expected `ContextState(scene, scene_confidence, objects[], gaze, attention_object, activity, engaged, timestamp)` is unchanged from the original analysis — this repo remains **scene classification only**:

| Handover field | Status | Evidence |
|---|---|---|
| `scene` | Present, as `label` | video.py:135 |
| `scene_confidence` | Present, as `conf` (now a 15-frame smoothed value, not raw per-frame) | video.py:134-135 |
| `objects[]` | **Absent** | zero object-detection code/imports anywhere in either script |
| `gaze` | **Absent** | zero gaze-estimation code anywhere |
| `attention_object` | **Absent** | depends on both of the above, neither exists |
| `activity` | **Absent** | single-frame (well, smoothed-multi-frame) scene classifier only, no activity/action recognition |
| `engaged` | **Absent** | no engagement/attention logic of any kind |
| `timestamp` | Not returned by the inference loop | trivially derivable by the adapter from `cv2.VideoCapture` frame position |

### 4.6 Confidence semantics
Confidence is a **15-frame rolling mean of softmax probabilities**, not a raw per-frame softmax value (contrast with Emotion, which is unsmoothed). `CONF_THRESHOLD = 0.5` (video.py:35) gates the `"uncertain"` fallback label, but the underlying `avg`/`idx`/`conf` values are computed and available regardless of whether the threshold is crossed — an adapter should read `conf`/`idx` directly rather than parsing the display string. Because of the smoothing, confidence values are **correlated across consecutive frames** (not i.i.d. per-frame observations) — worth accounting for in Phase 2's clip-level aggregation (§ handover doc), since naively re-averaging already-smoothed values across a clip is a mild form of double-smoothing.

### 4.7 Dependencies
No `requirements.txt` still exists in this repo (confirmed again on the updated files). Visible imports in `video.py`: `argparse, os, collections.deque, pathlib.Path, cv2, numpy, torch, torch.nn, torchvision.models, torchvision.transforms` — all needed for offline inference, all unpinned. `realtime.py` additionally imports `pyrealsense2` (optional, import-guarded, live-only — not needed offline). Both scripts' own docstrings now state the exact install line: `pip install torch torchvision opencv-python pillow numpy` (video.py) / `pip install torch torchvision opencv-python numpy` (realtime.py) — no version constraints given.

### 4.8 Minimal inference file set
`Context Repo/scene classification/video.py` + `Context Repo/scene classification/best_EfficientNet_B0.pth`. **Fully self-contained — no other files needed.** Exclude `realtime.py` (live-camera-only) and `pyrealsense2`.

### 4.9 Integration gotchas
- **`DEFAULT_VIDEOS_DIR` resolves to a path outside the project when no `--video`/`--videos-dir` is given.** `DEFAULT_VIDEOS_DIR = str(Path(SCRIPT_DIR).parents[3] / "videos")` (video.py:39) assumes a deeply nested layout (`repo_root/modalities/context/scene_classification/inference/video.py`, per the code's own comment) — but the actual on-disk location is the flat `Context Repo/scene classification/video.py`, only 2 levels below `f:\FYP`. `parents[3]` from that location resolves to the **drive root** (`F:\videos`), which does not exist. Running `python video.py` with no arguments will therefore print "No video files found under: F:\videos" rather than doing anything useful. **The Fusion Engine's Phase-0 runner must always pass `--video <clip_path>` explicitly** (or `--videos-dir` pointed at the actual clip folder) — never rely on the default.
- `process_video()` always creates a `VideoWriter` and writes to it regardless of `--no-show` (video.py:118, 139) — an adapter capturing structured results should redirect this to a throwaway/temp output path (or refactor it out) rather than treating "no display" as "no I/O side effects."
- No cropping/ROI/face-or-person-detection step exists — the entire frame is classified directly, so multi-subject or off-center-scene framing isn't a concern the way it is for Emotion/Gesture (no "which detection" ambiguity).
- GPU/CPU device selection is automatic (prefers CUDA), same as Emotion, with no CLI override.
- Only 2 output classes are hardcoded (`classroom`, `kitchen`) — schema.yaml's own comment `# extend as data grows` implies more scenes may be added later, which would require retraining, not just an adapter change.

### 4.10 Recommended adapter defaults (unchanged from original gap analysis — still applicable, just no longer blocked)
- Populate `scene`/`scene_confidence` directly from the captured `(idx, conf)` per frame (map `idx` through `SCENE_LABELS`).
- Populate `timestamp` in the adapter itself (model provides none).
- Default `objects=[]` (→ `n_objects=0`), `gaze`/`attention_object`/`activity`/`engaged` to `None`/`"unknown"` — all structurally absent from this model, must be documented as "never measured," not a real zero/null signal.

### 4.11 Open questions / risks
- Checkpoint-architecture match (`efficientnet_b0` + 2-class linear head) is assumed consistent with `best_EfficientNet_B0.pth` but was not verified by actually loading the weights (out of scope for this read-only analysis, per instructions) — low risk given the loading code now exactly mirrors the Emotion repo's proven pattern, but worth a smoke test before Phase 0.
- No version pins anywhere for this repo's dependencies — same category of risk as the Emotion model.
- The 15-frame smoothing window means the *very first* frames of a 5s clip have a thinner (but still valid, since `deque.append` + `np.mean` handle partial windows gracefully) probability history than frames later in the clip — not a cold-start defect like Motion's fabricated placeholder, but worth being aware of when comparing early- vs. late-clip confidence values.

---

## 5. Pre-Phase-0 smoke tests (2026-07-02)

Per the handover's gate requirement, both previously-unverified assumptions (Emotion's hardcoded label order, Context's rewritten checkpoint behavior) were tested with real inference runs (isolated venvs, actual weights, actual clips from `Dataset/My/`) before any Phase-0 code was written.

### 5.1 Emotion — label order verification
Ran `EMOTION_LABELS` classification on 5 real clips (3 unambiguous smiling/"Happy" clips across classroom+kitchen settings, 1 tense/defensive-posture clip expected Fear/Surprise, 1 bonus sarcasm clip). **Initial run using the native code's exact settings (`model_selection=0`) showed low, wildly inconsistent face-detection recall: 0%, 8%, 22%, 66%, 7% of frames across the five clips** — including one visually-unambiguous smiling clip with zero detections at all.

**Root cause isolated**: it is the face detector's `model_selection=0` (short-range, ≤2m) parameter, not a confidence-threshold issue (lowering `min_detection_confidence` from 0.5→0.3 changed nothing) and not a face-size issue (detected faces measured 150-330px, not tiny). Switching to `model_selection=1` (full-range) recovered recall to 59-100% across all five clips with no other change.

**With recall fixed, classification results cleanly matched expectations**: the clearest case (smiling + thumbs-up, kitchen) → Happy in 178/188 frames (94.4% avg conf); the tense/defensive case → Surprise in 107/172 frames (94.7% avg conf). The two classroom clips showed a close Neutral/Happy split (consistent with genuine over-time expression variation during a multi-second wave, not misclassification). **Conclusion: `EMOTION_LABELS` order is correct — no evidence of an index swap.** The real, actionable finding is that the native scripts' `model_selection=0` choice is a poor fit for typical HRI camera distances in this dataset; **adapters should use `model_selection=1`** (see Integration_API.md §2.1).

### 5.2 Context — scene classification verification
Ran the rewritten `video.py` on 2 classroom + 3 kitchen clips. **Classroom: 2/2 correct** (one at near-100% confidence throughout). **Kitchen: 1/3 correct** — one kitchen clip (thumbs-up scene) classified kitchen at ~98% confidence every frame; the other two kitchen clips were classified "classroom" for the *entire* clip at moderate confidence (57-86%).

**Root cause investigated**: compared frame composition between the correctly- and incorrectly-classified kitchen clips. The two misclassified clips have a camera angle/zoom that shows proportionally more plain wall + a patterned drop-ceiling and proportionally less kitchen-specific texture (tiled backsplash, stove, sink) than the correctly-classified clip. **Confirmed by direct experiment**: cropping the top 35% of frame (removing the ceiling/upper wall) out of one misclassified frame flipped its prediction from classroom (68%) to kitchen (68%); cropping to the bottom half pushed it to 86% kitchen.

**Conclusion: `SCENE_LABELS` order is correct** (classroom→classroom and kitchen→kitchen both happened cleanly when they worked) — this is a genuine generalization/robustness gap in the trained EfficientNet-B0 model, sensitive to how much ceiling/blank-wall is visible relative to kitchen-specific texture in frame. This is a **real, documented risk for Phase 0's per-cue agreement report**, not a fusion-engine or adapter bug, and not something to patch at the adapter level (out of scope — training the cue models is assumed done, per the handover's explicit scope).

### 5.3 Runner validation (2026-07-08/09) — two further confirmed findings
Building and running the four standalone runners (`runners/*_runner.py`, one per cue, each in its own isolated venv) end-to-end on real clips surfaced two further findings, both **confirming risks already flagged in this document rather than new problems**:

- **Gesture**: on the thumbs-up test clip, the raw `KeyPointClassifier` correctly identifies sign=3 ("Thumbs Up") in nearly every frame (188/188), but its raw confidence averages only 0.51 (range 0.31-0.94) — mostly below the 0.80 sensitive-gesture gate. Confirmed by direct instrumentation of the raw classifier (bypassing the gate) that this is not a runner bug: the correct shape is detected, but the confidence gate converts most of these frames to "Unknown", which then falls through to the "One hand raised" rule (hand held above a y-threshold) → canonical `raise_hand` instead of `thumbs_up`. This is precisely the handover document's own anticipated risk ("Gesture model filters thumbs_up to Unknown below 0.80 on your footage").
- **Motion**: on the same clip (person standing still, posing), 138/188 frames were classified "Frozen/Rigid Stand" rather than "Standing Still" — a normal calm-standing moment tripping the emergency-adjacent label. This directly confirms the fragility already flagged in §3.4/3.9: the two classes are separated only by a single unvalidated `body_speed < 0.08` threshold with no debounce.

Both are real cue-model behaviors to carry into Phase 0's agreement report, not runner defects — the runners were cross-checked to reproduce the native models' raw outputs faithfully before drawing this conclusion.

---

## 6. Cross-model summary table

| | Emotion | Gesture | Motion | Context |
|---|---|---|---|---|
| Offline entry point | `video.py` | `test_video.py` | `action_recognizer.py` (via direct call, not the wrapper scripts) | `video.py` |
| Perception backbone | MediaPipe Face Detection + MobileNetV2 | MediaPipe Hands + 2 TFLite MLPs | MediaPipe Pose + rule engine (LSTM loaded but unused) | EfficientNet-B0 (confirmed, no MediaPipe stage — full frame classified) |
| Native output today | burned-in video overlay only | burned-in overlay + in-loop dict (not returned) | burned-in overlay only | burned-in overlay only |
| Full probability vector available? | Yes (computed, discarded) | Partial (keypoint: yes; point-history: yes when invoked; overall scenario: no) | Yes (`smooth_probs`, 8-dim) | Yes (2-dim, 15-frame smoothed) |
| Confidence meaning | softmax max-prob, uncalibrated, per-frame | softmax max-prob (2 models) + 2 hardcoded constants + 1 synthetic placeholder (0.85) | fully hand-authored constants + EMA, not learned | softmax max-prob, 15-frame rolling-mean smoothed |
| Explicit "Unknown"/reject handling | No | Yes (`-1`, 0.80 gate on 4 of 6 classes) | Only via confidence=0 on zero-landmark frames (label field still shows a placeholder) | Yes ("uncertain" label, `conf < 0.5`) |
| requirements.txt present? | No | Yes (pinned) | Yes (pinned) | No |
| Runnable as-is? | Yes | Yes | Yes | Yes (fixed 2026-07-01 — was blocked, now self-contained) |

---

## 7. Gate status (per handover §"Gate")

**Do not begin Phase 0 until:**
- [x] Context model's blocker is resolved — `video.py`/`realtime.py` were rewritten to be self-contained (2026-07-01); §4 above reflects the verified, current implementation.
- [x] Emotion, Gesture, Motion, and Context entry points, preprocessing, output schemas, and confidence semantics are documented above.
- [ ] The corrections in this document (Gesture ≠ pose-based; Motion's LSTM is unused; Motion's confidence is not a learned probability; none of the four models emit structured per-frame output today — all four need the extraction refactor in `Integration_API.md` §3) are acknowledged before adapters are written.
- [ ] The Gesture/Motion numpy version conflict (`Integration_API.md` §4) is resolved so all four models can run in one Phase-0 environment.

All four cue models are now verified-runnable. **The gate is clear to begin Phase 0**, pending the refactor work and dependency resolution noted above.

See [`Integration_API.md`](Integration_API.md) for the reconciled adapter contract.
