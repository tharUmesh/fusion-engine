# Integration API — Cue Model → Fusion Engine Adapter Contract

**Status:** Specification/reconciliation document only. No adapter code has been written. This defines what Phase 2's `adapters/*.py` (per the handover document) will need to do, based on the verified findings in [`MODEL_ANALYSIS.md`](MODEL_ANALYSIS.md).

**Purpose:** The handover document (`HRI_Fusion_Engine_Handover.md`, §4) already specifies a target common schema (`NormalisedFrameCue`) and per-adapter mapping notes (§4.1–4.4). This document reconciles that target against what the four repositories can *actually* deliver today, and specifies the concrete refactor each repo needs before an adapter can call into it programmatically (all four are currently CLI/GUI demo scripts, not importable inference APIs).

---

## 0. Corrections to the handover document's assumptions

**Pre-Phase-0 gate (2026-07-02): both smoke tests passed.** Emotion's and Context's hardcoded label orders were verified against real inference runs on known-content clips — no evidence of an index swap in either model. Two real, non-blocking findings surfaced and are already folded into corrections 6-7 below and the per-model sections: Emotion's native face detector (`model_selection=0`) has poor recall at real HRI camera distances (fix: use `model_selection=1`), and Context's EfficientNet-B0 has a genuine accuracy gap tied to camera framing (ceiling/wall proportion vs. kitchen-specific texture) that is not fixable at the adapter level. See MODEL_ANALYSIS.md §5 for the full experiments.


Before writing adapters, the following corrections from `MODEL_ANALYSIS.md` should be folded into the handover's §3/§4:

1. **§4.2 (Gesture adapter) — "dynamic scenarios come from pose+rules" is wrong.** There is no MediaPipe Pose usage in the Gesture repo. Replace with: "dynamic scenarios come from MediaPipe **Hands** landmark history + temporal rule logic (wave/come-here heuristics), with a point-history TFLite classifier invoked only as a tertiary fallback for the Pointer case."
2. **§4.3 (Motion adapter) — the LSTM should not be described as part of the inference path.** `action_recognizer.py` loads `motion_lstm_v2_best.pth` but never calls it; classification is 100% a deterministic physics/rule engine. The adapter should not attempt to load or depend on the `.pth` file at all once `action_recognizer.py` is refactored (see §3 below).
3. **Confidence is not commensurate across models.** The handover implicitly treats `confidence` as a uniform concept for `confidence_floor` thresholding (schema.yaml). In reality: Emotion/Gesture confidences are calibration-free softmax max-probs; Motion's confidence is a hand-authored constant per class, EMA-smoothed (not a probability derived from data); Context's is unknown pending the missing classifier module. **Recommendation:** keep the per-cue `confidence_floor` values in schema.yaml (already cue-specific, which is good), but do not assume cross-cue confidence comparability anywhere else in the fusion logic (e.g., avoid a single global confidence-weighted vote across cues without normalization).
4. **§4.4 (Context adapter) — this repo cannot deliver `objects`, `gaze`, `attention_object`, `activity`, or `engaged` at all**, not just "gaze behind a flag." Only `scene` and `scene_confidence` are available (**confirmed** — see below, no longer pending). The `extra` dict the adapter builds will have `activity`/`engaged` permanently `None` and `n_objects` permanently `0` for this cue until a different/extended model is substituted — this must be a hardcoded, documented adapter behavior, not a silently-degraded runtime value. **Update 2026-07-01**: the Context repo's `video.py`/`realtime.py` were rewritten to be self-contained (no more missing-module blocker) — `scene`/`scene_confidence` are now fully verified: `SCENE_LABELS = ["classroom","kitchen"]`, matching schema.yaml exactly, and confidence is a 15-frame rolling-mean-smoothed softmax (not raw per-frame, unlike the other three cues — see §2.4 below).
5. **None of the four models expose a callable, structured, per-frame inference API today.** All four are monolithic scripts combining capture, inference, and rendering in one blocking loop (`cv2.imshow`/`cv2.waitKey`). Phase 0's `pipeline/run_cue_models.py` cannot simply `import` and call these repos as-is — each requires a minimal, scoped refactor (detailed per-model in §3) to extract a callable/generator that yields per-frame results without GUI side effects. This refactor touches only the inference repos' entry-point files (or, preferably, is done via a thin non-invasive wrapper module inside the Fusion Engine repo that duplicates just the per-frame loop body, so the original cue-model repos remain untouched — see §3.5 for the recommended approach).

---

## 1. Canonical per-frame schema (target, unchanged from handover §4)

```python
@dataclass
class NormalisedFrameCue:
    cue: str                      # "emotion" | "gesture" | "motion" | "context"
    frame_idx: int
    label: str                    # canonical label from schema.yaml (or "Unknown")
    confidence: float              # 0..1
    probs: dict[str, float]        # full distribution over that cue's classes ({} if N/A)
    valid: bool                    # confidence >= floor AND label != "Unknown"
    extra: dict                    # cue-specific extras
```

Each section below gives the exact native → normalised mapping, keyed to verified code locations.

---

## 2. Per-model adapter mapping

### 2.1 Emotion → `NormalisedFrameCue`

| Normalised field | Source in native code | Notes |
|---|---|---|
| `label` | `EMOTION_LABELS[pred.item()]` (video.py:30, 136) | 7 classes: `Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral` — 1:1 with schema.yaml's `emotion_classes`, order **unverified against training code** (no class-index file exists) |
| `confidence` | `conf.item()` — softmax max-prob (video.py:136) | uncalibrated |
| `probs` | full `F.softmax(model(tensor), dim=1)` vector, zipped against `EMOTION_LABELS` | currently computed then discarded by the script — adapter must capture it before it's dropped |
| `valid` | `confidence >= confidence_floor.emotion` (0.50 per schema.yaml) AND a face was detected that frame | model has no built-in threshold or Unknown class — the adapter must implement both the floor check and the "no face detected" case itself |
| `extra` | none defined by handover | recommend adding `bbox` (from `x,y,bw,bh`, video.py:128-130) so the fusion engine can later disambiguate multiple faces per frame |

**Verified fix (smoke-tested 2026-07-02): use `model_selection=1`, not the native code's `model_selection=0`.** The native short-range face detector (`video.py:189`) misses 34-100% of frames on real HRI-distance footage (measured 0%, 8%, 22%, 66%, 7% detection rates across 5 test clips); switching only this one parameter recovered 59-100% recall with no other change, and with recall fixed, classification results cleanly matched ground truth on the clearest test cases. This is a one-line change in the adapter's face-detector construction, not a change to the classifier itself — see MODEL_ANALYSIS.md §5.1 for the full experiment.

**No-face-detected frames**: the native script emits nothing. The adapter must emit a `NormalisedFrameCue` with `label="Unknown"`, `confidence=0.0`, `probs={}`, `valid=False` for such frames, since downstream aggregation (Phase 2) expects one record per frame, not sparse frames.

**Multi-face policy (undefined in native code)**: the adapter must pick one face per frame if HRI clips can contain bystanders. Recommend: largest bounding-box area, tie-broken by highest confidence. This policy must be documented in the adapter's code as a fusion-engine decision, not inherited from the model.

### 2.2 Gesture → `NormalisedFrameCue`

| Normalised field | Source in native code | Notes |
|---|---|---|
| `label` | `keypoint_classifier_label.csv` mapped via `hand_sign_id` (test_video.py:203-204, 252), OR the global scenario string (test_video.py:265-341) | **Decide at adapter-design time**: normalise per-hand static shape, or the fused global scenario string. The handover's `gesture_classes` list (`wave, point, thumbs_up, thumbs_down, raise_hand, both_hands_up, beckoning, Unknown`) matches the **global scenario** vocabulary more closely than the raw 6-class keypoint labels — recommend the adapter consume `global_scenario_text` (mapped: Pointing→point, Thumbs up→thumbs_up, Thumbs down→thumbs_down, Wave/Brief wave/Arms waving→wave, One hand raised→raise_hand, Arms up→both_hands_up, Beckoning→beckoning, None→Unknown) |
| `confidence` | `global_conf` (test_video.py:265-341) for the scenario-level mapping above | already a 0-1 float; note it is `0.0` and not appended for the `"None"` case (§2.4/2.6 of MODEL_ANALYSIS.md) — must not be read as "confidently no gesture" |
| `probs` | `{}` recommended | the global scenario resolver does not produce a full distribution over the 8 gesture_classes — only a single winning label + confidence. Per-handover §4.2, `probs={}` is explicitly acceptable ("handle in aggregation") |
| `valid` | `(label != "Unknown") and (confidence >= confidence_floor.gesture=0.80)` | matches the model's own internal 0.80 sensitive-gesture gate (test_video.py:203-204) for the 4 gated classes, but note Open Palm/Close pass the model's internal gate regardless of confidence — the adapter's `valid` check must still apply the 0.80 floor uniformly per schema.yaml, independent of the model's internal (narrower) gating |
| `extra` | `point_direction`, `motion_direction` per handover §4.2 | **Neither is computed anywhere in the gesture repo.** Set `extra["point_direction"] = None` and `extra["motion_direction"] = "none"` unconditionally — this is not a partial gap, it's a total absence, and should be documented as such rather than left silently null |
| `extra["point_target"]` | N/A | hardcode `"unknown"` per handover — confirmed no YOLO/object-detection code exists anywhere in this repo to source it from |

**Which script to adapt**: `test_video.py`, not `app.py` or `play_video.py` (see MODEL_ANALYSIS.md §2.1). This script must be refactored to remove the `cv.imshow`/`cv.waitKey` loop and instead yield `(frame_idx, hand_states, global_scenario_text, global_conf)` per frame.

**Dead-code caution**: point-history classes `Clockwise`/`Counter Clockwise`/`Move` are never surfaced by the current scenario resolver — confirm with the fusion engine's intent taxonomy whether these are expected before wiring them in; if not expected, no adapter work is needed for them.

### 2.3 Motion → `NormalisedFrameCue`

| Normalised field | Source in native code | Notes |
|---|---|---|
| `label` | `motion_label` (one of `MOTION_LABELS`, action_recognizer.py:36-45) | matches handover's `motion_classes` list 1:1 in order and meaning; adapter must map code's display strings (e.g. `"Run (Fast Movement)"`) to schema.yaml's snake_case (`Run_Fast`) |
| `confidence` | `confidence = smooth_probs[argmax(smooth_probs)]` | **not a learned probability** — a hand-authored constant per class, EMA-smoothed (§3.6 of MODEL_ANALYSIS.md). Still usable numerically for the `confidence_floor.motion=0.50` threshold, but should not be assumed calibrated relative to other cues |
| `probs` | `smooth_probs` (8-dim np.ndarray) zipped against `MOTION_LABELS` | **this cue does produce a real full distribution every frame** — unlike Gesture, no `{}` fallback needed here |
| `valid` | `confidence >= confidence_floor.motion (0.50)` | **must additionally gate on landmark presence, not just the confidence field** — the cold-start/re-warm-up case (< 4 buffered frames) fabricates `label="Standing Still", confidence=0.90` (a value that passes the floor!) even though it is not a real detection. The adapter must special-case this: treat frames where `len(keypoints_queue) < 4` (equivalently: fewer than 4 frames since the last landmark-loss reset) as `valid=False` regardless of the fabricated confidence value. This is the single most important correction needed relative to a naive floor-based `valid` check |
| `extra` | `{"pose": pose_label}` per handover §4.3 | `pose_label` (Sitting/Standing/Crouching/Lying/Unknown) comes from the separate single-frame geometric classifier (action_recognizer.py:93-116) — already available every frame, independent of the motion buffer's warm-up state |

**Which script to adapt**: `action_recognizer.py` directly (not `run_single_video.py`/`run_all_and_save.py`, which are subprocess wrappers with no structured output — see MODEL_ANALYSIS.md §3.1). The refactor should also **remove the LSTM load entirely** (dead weight, `sys.exit(1)`-on-missing-file risk with zero functional benefit) once confirmed safe to drop.

**Standing_Still vs Frozen_Rigid_Stand**: kept distinct in code as required by the handover checklist, but the discriminating signal (single `body_speed < 0.08` threshold, no debounce) is a real fidelity risk for the emergency-adjacent scenario (#38 in the dataset table) — flag this in Phase 0's agreement report, not something an adapter can fix by itself.

### 2.4 Context → `NormalisedFrameCue` — **unblocked, verified 2026-07-01**

| Normalised field | Source in native code | Notes |
|---|---|---|
| `label` | `SCENE_LABELS[idx]` where `idx = avg.argmax()` (video.py:133-135) | `SCENE_LABELS = ["classroom", "kitchen"]` (video.py:31) — confirmed exact match to schema.yaml's `context_scenes` |
| `confidence` | `conf = float(avg[idx])`, where `avg` is a **15-frame rolling mean** of per-frame softmax outputs (`prob_history = deque(maxlen=15)`, video.py:120, 130-134) | **Not a raw per-frame value** — unlike Emotion/Gesture/Motion's per-frame confidence, this is smoothed over the trailing 15 frames. Adapter should read `conf` directly (not parse the `"uncertain"` display string) |
| `probs` | `dict(zip(SCENE_LABELS, avg))` | a real, full 2-class distribution is available every frame (`avg`, the smoothed softmax vector) — populate `probs`, don't default to `{}` |
| `valid` | `confidence >= confidence_floor.context (0.50)` — this already matches the native `CONF_THRESHOLD = 0.5` (video.py:35) exactly, so the model's own `"uncertain"` fallback and the schema.yaml floor agree by construction | the native `"uncertain"` label (video.py:135) should map to the handover's `"Unknown"` sentinel |
| `extra` | `{"activity": None, "engaged": None, "n_objects": 0}` | **all three fields remain structurally absent from this model** — no object detection, activity recognition, or engagement logic exists anywhere in either script, even after the rewrite. These must be hardcoded placeholders, clearly commented as "not measured by this model," so downstream fusion code never mistakes `n_objects=0` for a real "zero objects observed" signal |
| gaze / `attention_object` | N/A | per `use_gaze_features: false`, ignore entirely — consistent with the handover's own design; this remains a permanent structural absence, not a temporarily-disabled feature |

**Which script to adapt**: `video.py` (self-contained now, no missing dependency). Refactor the `while True:` loop in `process_video()` (video.py:122-154) to yield `(frame_idx, idx, conf, avg_probs)` instead of only drawing/writing — same pattern as the other three adapters. **Do not rely on `DEFAULT_VIDEOS_DIR`** (video.py:39) — it resolves to a nonexistent path (`F:\videos`) given this repo's actual flat layout; always invoke with an explicit `--video <clip_path>`.

**Known accuracy risk (smoke-tested 2026-07-02, not fixable at the adapter level): the model is sensitive to how much ceiling/blank-wall vs. kitchen-specific texture (tiled backsplash, stove, sink) is visible in frame.** 2 of 3 real kitchen test clips were misclassified as "classroom" at moderate confidence (57-86%) when the camera angle showed more ceiling/plain wall; a direct crop experiment confirmed removing the ceiling from the same frame flips the prediction back to "kitchen." Label order itself is correct (classroom→classroom and kitchen→kitchen both happen cleanly when they work) — this is a genuine model generalization gap, out of scope to fix here (training the cue models is assumed done). **Carry this into Phase 0's per-cue agreement report as an expected source of context-cue disagreement**, not a fusion-engine bug.

---

## 3. Required refactor per repo (Phase-0 prerequisite work)

None of the four repos expose a function/generator that returns structured per-frame results without also doing I/O (display, video writing) as a side effect. The handover's `pipeline/run_cue_models.py` needs each of the following minimal extraction points. **Recommendation: do this as new thin wrapper modules inside the Fusion Engine's `adapters/` package that import and call into a refactored copy of each script's per-frame logic — not by modifying the original cue-model repos in place**, so the cue-model repos remain untouched per the task's constraint.

1. **Emotion** (`video.py`): extract the body of the `for det in results.detections:` loop (video.py:126-141) into a function `infer_frame(frame, model, device) -> list[FaceResult]` returning `(bbox, label, confidence, probs_vector)` per detection, with no `cv2.putText`/`cv2.imshow`/`VideoWriter` calls.
2. **Gesture** (`test_video.py`): extract the per-frame body (roughly test_video.py:143-341, the section between frame read and the drawing calls) into a function/generator that yields `(frame_idx, hand_states: dict, global_scenario_text: str, global_conf: float)`, with `cv.imshow`/`cv.waitKey`/drawing removed. Also patch the missing divide-by-zero guard in `pre_process_landmark` (test_video.py:428-436) to match `app.py`'s guarded version.
3. **Motion** (`action_recognizer.py`): extract the per-frame body of `run()` (lines ~322-521) into a generator yielding `(frame_idx, pose_label, motion_label, confidence, probabilities: np.ndarray)`, with the LSTM load (lines 239-247) and all `cv2.imshow`/`VideoWriter`/dashboard-drawing calls removed.
4. **Context** (`video.py`): extract the per-frame body of `process_video()`'s `while True:` loop (video.py:122-154) into a generator yielding `(frame_idx, idx, conf, avg_probs: np.ndarray)`, with `cv2.imshow`/`cv2.waitKey`/`VideoWriter` calls removed. Also bypass `DEFAULT_VIDEOS_DIR` entirely — always call with an explicit clip path.
5. **General pattern**: every extraction above should preserve the exact preprocessing math (resize dimensions, normalization constants, confidence thresholds, landmark indices) verified in `MODEL_ANALYSIS.md` — the goal is capturing values the scripts already compute internally, not re-deriving or "improving" them at this stage.

---

## 4. Dependency install summary (offline evaluation, dev machine)

| Model | Required packages | Version source |
|---|---|---|
| Emotion | `torch, torchvision, opencv-python, mediapipe, pillow` | unpinned (no requirements.txt) |
| Gesture | `numpy==1.26.4, opencv-python==4.9.0.80, mediapipe==0.10.11, tensorflow==2.15.1, protobuf==3.20.3` | `Gesture Repo/requirements.txt` (matplotlib excluded — training-only) |
| Motion | `mediapipe==0.10.14, torch==2.11.0, torchvision==0.26.0, opencv-python==4.13.0.92, opencv-contrib-python==4.13.0.92, numpy==2.4.4` (+ transitive deps) | `Motion Repo/requirements.txt` |
| Context | `torch, torchvision, opencv-python, pillow, numpy` | unpinned (no requirements.txt), but now confirmed by direct inspection (self-contained script) |

**Version conflict risk**: Gesture pins `numpy==1.26.4` while Motion pins `numpy==2.4.4` — these are **incompatible in the same environment** (NumPy 2.x is not backward compatible with 1.x for compiled extensions). The Fusion Engine's Phase-0 runner will need either separate virtual environments per cue model, or a resolution to a single mutually-compatible numpy version (test both models against numpy 2.x first, since Gesture's pin is likely just "last known-good at authoring time" rather than a hard requirement — TensorFlow 2.15.1 and mediapipe 0.10.11 compatibility with numpy 2.x should be verified empirically before assuming the older pin is load-bearing).

---

## 5. Summary of blockers before adapters can be implemented

1. ~~Context model's `src/classifier.py` is missing~~ **RESOLVED 2026-07-01** — `video.py`/`realtime.py` are now self-contained; `SCENE_LABELS`, preprocessing, and confidence semantics are fully verified (§2.4). No remaining blocker for this cue.
2. **All four repos need the per-frame extraction refactors in §3** before `pipeline/run_cue_models.py` can call them in a loop — this is real implementation work, not configuration.
3. **Gesture/Motion numpy version conflict (§4)** needs resolving before both can run in one Phase-0 environment.
4. **Emotion's label order is still unverified against ground truth** (no class-index file or training script exists in that repo) — should be spot-checked against a few known clips as part of Phase 0's agreement report. Context's label order (`classroom, kitchen`) is now confirmed directly from code, no longer a risk.
5. **All four adapters need the per-frame extraction refactor before any code can be written against them** — no repo currently exposes a callable/generator API; this is the main implementation task remaining before Phase 0 can start in earnest.
