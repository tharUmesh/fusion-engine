# HRI Intent Fusion Engine — Engineering Handover

**Audience:** an engineer or an AI coding agent (Claude Code, etc.) building the system.
**Scope of this document:** the **fusion layer only** — everything downstream of the four cue models, up to a predicted intent, evaluated **offline** on recorded clips.
**Out of scope:** training the cue models (assumed done), the live camera path, robot action execution, object-target detection (YOLO).

---

## 0. Read this first — the mental model

```
recorded clip (5s, 720p)
        │
        ▼
┌───────────────────────────────────────────────┐
│  4 cue models (ALREADY TRAINED — do not touch) │
│  emotion / gesture / motion / context          │
│  each emits a result PER FRAME                  │
└───────────────────────────────────────────────┘
        │  per-frame outputs
        ▼
┌───────────────────────────────────────────────┐
│  ADAPTER per model                              │
│  normalises native output → common schema       │
└───────────────────────────────────────────────┘
        │  per-frame normalised cues
        ▼
┌───────────────────────────────────────────────┐
│  CLIP AGGREGATOR (Option A)                     │
│  collapse ~150 frames → ONE feature vector/clip │
└───────────────────────────────────────────────┘
        │  one feature vector per clip
        ▼
┌───────────────────────────────────────────────┐
│  FUSION ENGINE                                  │
│   • Rule-based baseline   (the bar to beat)      │
│   • GBT fusion            (THE DELIVERABLE)      │
│   • Transformer           (the experiment)       │
└───────────────────────────────────────────────┘
        │
        ▼
  predicted intent (F-code) + probability dist + per-cue attribution + safety flag
```

**Three rules that govern the whole build:**

1. **The unit of learning is the *scenario* (cue-combination), not the *clip*.** You have ~26 scenarios × ~50 variations. The ~50 variations of one scenario are the *same lesson seen under different conditions* — they teach robustness, not new patterns. Every split, every metric must respect this (see §6).
2. **Train on what the models *actually emit*, never on the authored table.** The dataset table is the *intent ground truth* and the *scene reference*. The training *features* come from running the real models over the real clips (Phase 0). Skipping this creates a train/runtime mismatch that silently breaks deployment.
3. **GBT first and locked, then the transformer as a controlled experiment.** Never let the transformer block the working system.

---

## 0b. Known cue-model findings (from repo analysis) — fold these in before building

The four cue models were built by other team members and analysed (see `MODEL_ANALYSIS.md` / `Integration_API.md`). The following concrete findings shape the runners built for Phase 0a/0. Ordered by how badly each would bite if ignored. **Status as of 2026-07-09: gates 1, 2, 3, 5 below are now VERIFIED (not just flagged) — see MODEL_ANALYSIS.md §5 for the actual smoke-test evidence.**

1. ~~Emotion label order is unverified~~ **VERIFIED 2026-07-02** (MODEL_ANALYSIS.md §5.1): ran real inference on 5 known-expression clips. No evidence of an index swap — a genuinely happy+thumbs-up clip produced `Happy` at 94.4% avg confidence across 178/188 frames. **Separately, a real bug was found and fixed**: the native face detector's `model_selection=0` (short-range) misses 34-100% of frames on this project's camera distances; `emotion_runner.py` uses `model_selection=1` instead — this is now baked into the runner, not an open risk.
2. **Motion cold-start fabricates a confident label (silent trap) — FIXED in `motion_runner.py`.** With `<4` buffered frames, the native code emits `label="Standing Still", confidence=0.90` — which passes the 0.50 floor. The runner sets `valid=False` whenever `<4` frames are buffered, **including mid-clip after an occlusion resets the buffer, not just at clip start** (verified: `is_cold_start` is re-derived every frame from the live buffer length, not a one-shot flag — see `runners/motion_runner.py` module docstring and the `mid_clip_occlusion_resets` diagnostic column in its `--stats-out` CSV).
3. **"Confidence" is not commensurate across models — and Motion's is fake.** Motion confidence is a hand-authored constant per class (0.80–0.95), EMA-smoothed, *not* a learned probability. Emotion/Gesture are uncalibrated softmax. Context is 15-frame-smoothed softmax. Consequences: (a) never build a global confidence-weighted vote across cues; (b) calibration (Phase 4) applies to emotion/context only — you cannot calibrate a constant; (c) do **not** feed motion's confidence as a real-valued probability feature — use its class-probability *shape* and/or its validity bit instead.
4. **Context cannot deliver `activity`, `engaged`, or `objects` — permanently.** No such code exists (not just gaze). These `extra` fields are hardcoded "not measured" placeholders forever (see `context_runner.py`'s `NOT_MEASURED_EXTRA`). **Scope impact:** scenarios that lean on focused-activity / engagement must be resolved from scene + motion + gesture alone, or they degrade. Acknowledge this as a v1 scope cut, not a bug.
5. ~~Context repo checkpoint match is unverified~~ **VERIFIED 2026-07-02** (MODEL_ANALYSIS.md §5.2): loaded `best_EfficientNet_B0.pth` into the rewritten `video.py` and ran it on real classroom/kitchen clips. 2/2 classroom correct; 1/3 kitchen correct — the 2 misses were root-caused to camera framing (ceiling/wall proportion vs. kitchen-specific texture), confirmed by a direct crop experiment, **not** an architecture/checkpoint mismatch. This is a real, permanent accuracy limitation of the trained model to carry into the agreement report — not a blocker.
6. **Context confidence is already 15-frame-smoothed** → clip-level aggregation averages already-smoothed values (mild double-smoothing). Low severity; don't over-interpret context confidence stability.
7. **Gesture's 0.80 sensitive-gesture gate suppresses correct detections on real footage (confirmed, not a bug).** On a real thumbs-up test clip, the raw keypoint classifier correctly identifies "Thumbs Up" in 188/188 frames, but its raw confidence averages only 0.51 — mostly below the 0.80 gate — so most frames get downgraded to `Unknown` and then reclassified as `raise_hand` via the "hand held above y-threshold" fallback rule. **This is the single most impactful systematic disagreement pattern in the Phase 0 agreement report** (see `reports/phase0_agreement.md`, `KNOWN_SYSTEMATIC_PATTERNS` in `pipeline/agreement_report.py`).
8. **Motion's `Standing Still` vs `Frozen/Rigid Stand` threshold is fragile (confirmed, not a bug).** On the same clip (person calmly posing, not moving much), 138/188 frames were classified `Frozen/Rigid Stand` — the single `body_speed < 0.08` threshold with no debounce over-triggers on ordinary stillness. Also surfaced as a known systematic pattern in the agreement report.

Cosmetic corrections (no plan impact): gesture is MediaPipe **Hands**-based, not Pose-based; motion's LSTM is loaded but never used and is **not loaded at all** by `motion_runner.py` (dropped from the inference path entirely, not just ignored after loading).

**Division of labour for verification — retired.** The Claude Code / human split originally planned here is no longer needed: findings 1, 2, 3, 5 were verified mechanically (real inference on known-content clips, cross-checked by direct instrumentation of raw model outputs before/after each gate — see MODEL_ANALYSIS.md §5 for the exact experiments). No model-owner interviews were required.

---

## 1. Repository layout

**Superseded 2026-07-09 — this is the actual layout in use, not the originally-planned one.** The dataset moved to a versioned, pre-packaged location with its own manifests (no hand-authoring needed), and Phase 0a's "adapters" became "runners" (below) with an added batch mode. Kept close to the original plan otherwise.

```
Fusion Engine/                                    # repo root (this repo)
├── Data/Dataset/hri-multimodal-intent-v1.0.0/     # READ-ONLY input dataset (not "data/", see below)
│   ├── raw/clips/<classroom|kitchen>/<scenario_id>/<clip_id>.mp4
│   └── annotations/
│       ├── scenarios.csv       # intent ground truth, one row per BASE scenario (S01..S29) — pre-existing, not hand-authored
│       ├── clips.csv           # one row per clip: clip_id, scenario_id (e.g. S01_F04), subject_id (P01..P09), filepath, fps, frame_count, sha256...
│       └── splits.csv          # generated by pipeline/build_splits.py — see §2a
├── Emotion Repo/ Gesture Repo/ Motion Repo/ Context Repo/   # the 4 cue-model repos (UNMODIFIED, per Pre-Phase rules)
├── runners/                        # Phase 0a deliverable: 4 standalone runners, replaces adapters/
│   ├── common/
│   │   ├── schema.py            # NormalisedFrameCue dataclass (the hard schema contract) + JSONL I/O helpers
│   │   └── constants.py         # confidence floors + gesture label-canonicalization map
│   ├── emotion_runner.py
│   ├── gesture_runner.py
│   ├── motion_runner.py
│   └── context_runner.py        # each: --clip/--out (single) or --manifest/--clips-root/--out (batch, loads model ONCE)
├── .venvs/                         # one isolated venv per runner (emotion/gesture/motion/context) — see §Phase 0a
├── pipeline/
│   ├── canonical_map.py         # scenarios.csv free-text intended-cue values -> each model's canonical label vocab
│   ├── build_splits.py          # Phase 0: writes Data/.../annotations/splits.csv (scenario-grouped + subject-grouped)
│   ├── aggregate_clip_cues.py   # Phase 0 LIGHTWEIGHT aggregation: majority-vote label per (clip, cue) over valid frames only
│   │                             #   NOT the Phase 2 feature-vector builder below — do not conflate the two
│   ├── agreement_report.py      # Phase 0 deliverable: intended vs measured per scenario per cue, cue_corrupted flags
│   ├── measured/                # Phase 0 output: <cue>_frame_cues.jsonl (one per cue, all clips, clip_id-tagged)
│   │                             #   + clip_cues.csv (aggregate_clip_cues.py output) + motion_stats.csv (occlusion diagnostic)
│   ├── aggregate.py             # Phase 2 (NOT YET BUILT): frame cues → full clip feature vector (~30-40 dims)
│   └── build_features.py        # Phase 2 (NOT YET BUILT): orchestration → clip_features.parquet
├── fusion/                         # Phase 3+ (NOT YET BUILT)
│   ├── rule_based.py
│   ├── gbt.py
│   └── transformer.py
├── eval/                           # Phase 5 (NOT YET BUILT) — leave-one-scenario-out CV harness, see §6
├── reports/
│   └── phase0_agreement.md      # THE Phase 0 deliverable — see Phase 0 section below
├── MODEL_ANALYSIS.md               # Pre-Phase cue-model analysis + all verification-gate evidence
└── Integration_API.md              # Pre-Phase adapter-contract reconciliation
```

**Why JSONL instead of parquet for `pipeline/measured/`:** the four runners each live in an isolated venv with deliberately minimal, pinned dependencies (see Phase 0a) — adding `pyarrow`/`pandas` to all four just for parquet I/O would be scope creep. Everything downstream (`aggregate_clip_cues.py`, `agreement_report.py`) is stdlib-only `csv`/`json`, by design (see the hard-contract rule in Phase 0 below). Revisit parquet only if/when file size becomes a real problem.

**`data/labels/`, `data/measured/`, `data/features/` (originally planned) do not exist as such** — Windows' case-insensitive filesystem collides `data/` with the existing `Data/` (dataset) folder, so pipeline output instead lives under `pipeline/measured/` and the ground-truth files are read directly from `Data/Dataset/.../annotations/` rather than copied.

---

## 2. Ground-truth label files — `Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/`

**Superseded 2026-07-09.** No hand-authoring needed — the dataset ships its own manifests. Three CSVs, not one:

**`scenarios.csv`** — one row per **base scenario** (23 rows: S01-S09, S11, S12, S18-S29):

| column | meaning | example |
|---|---|---|
| `Scenario ID` | base scenario id | `S18` |
| `Context` | scene reference | `kitchen` |
| `Intent` | target label (F-code) | `F01` |
| `Intended Emotion` | authored cue (free text) | `happy` |
| `Intended Gesture` | authored cue (free text, sometimes `[MISSING]`) | `thumbs up` |
| `Intended Motion` | authored cue (free text) | `stand` |
| `Intended Context` | scene reference (redundant with `Context` in this version) | `kitchen` |

Free-text intended values are **not** already in each model's canonical label vocabulary (e.g. `"both hands up"` vs. gesture's `both_hands_up`) — `pipeline/canonical_map.py` maps each cue's raw values to its runner's canonical labels (documented per-value, including the `stepping back` → `Walking` merge decision inherited from the motion model's own 8-class taxonomy). `[MISSING]` (scenario S08's gesture) maps to `None` — a deliberately-absent intended cue, excluded from agreement comparison, not "the model should detect nothing."

**`clips.csv`** — one row per **clip** (1269 rows after removing 1 row referencing a missing file, see Phase 0 notes): `clip_id, scenario_id (composite, e.g. S01_F04 = base scenario + intent code), subject_id (P01-P09), filepath, orig_filename, duration_s, fps, resolution, frame_count, sha256, recorded_at`. Join to `scenarios.csv` via `scenario_id.split("_")[0]`.

**`splits.csv`** — generated by `pipeline/build_splits.py`, see §2a below. Not hand-authored; regenerate by rerunning the script if `clips.csv` changes.

> The `Intended *` columns are **reference only**. They are NOT model inputs. They exist so Phase 0 can check whether the models actually reproduce them.

### 2a. `splits.csv` — grouped train/val/test, two independent groupings

Per clip, three columns:
- **`split_scenario`** (primary fusion evaluation split) — all clips of one `scenario_id` (composite, e.g. `S01_F04`) assigned to exactly one of train/val/test. Tests generalization to **unseen cue combinations**.
- **`split_subject`** — all clips of one `subject_id` assigned to exactly one of train/val/test. Tests generalization to **unseen people**.
- **`split_random_leaky_DO_NOT_USE_FOR_EVAL`** — ordinary clip-level random split, no grouping. Included only as a documented optimistic-upper-bound contrast (near-duplicate clips can straddle train/test) — **never use as the headline number.**

Both grouped splits use a greedy largest-group-first allocation targeting 70/15/15 by clip count (group sizes vary 11-73 clips per scenario, 10-377 per subject). `build_splits.py` **asserts** no `scenario_id`/`subject_id` spans multiple splits before writing anything.

**Important, corrected finding:** `clips.csv`'s `subject_id` column initially contained a data error making it look like subject and scenario were in a 1:1 relationship (23 "subjects" = 23 scenarios) — which would have made `split_scenario` and `split_subject` identical partitions, defeating the point of having two splits. This was a data bug, now fixed (9 real subjects, P01-P09, each performing 2-22 scenarios) — `split_scenario` and `split_subject` are now genuinely different partitions (verified: `split_scenario == split_subject` is `False` for most clips).

**Distinct from Phase 5's leave-one-scenario-out CV (§ below):** `splits.csv` is a single static train/val/test assignment for Phase 2+ model development. Phase 5's rotating leave-one-scenario-out cross-validation (§6) is a separate, later evaluation protocol applied *to* a trained fusion model — the two are not interchangeable and this document does not conflate them.

---

## 3. Canonical schema (`configs/schema.yaml`) — single source of truth

Everything reads from this. Define it once.

```yaml
emotion_classes: [Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral]
gesture_classes: [wave, point, thumbs_up, thumbs_down, raise_hand, both_hands_up, beckoning, Unknown]
pose_classes:    [Sitting, Standing, Crouching, Lying, Unknown]
motion_classes:  [Sitting_Still, Standing_Still, Walking, Walk_Across, Run_Backward, Run_Fast, Leaning_Forward, Frozen_Rigid_Stand]
context_scenes:  [classroom, kitchen]          # extend as data grows
context_activities: [studying, cooking, idle, unknown]

intent_classes:  [F01, F02, F03, F04, F05, F06, F07, F08, F09, F10]

# per-cue confidence floor below which a frame's cue is treated as MISSING
confidence_floor:
  emotion: 0.50
  gesture: 0.80      # matches the gesture model's own sensitive-gesture filter
  motion:  0.50
  context: 0.50

# fraction of valid frames below which the WHOLE CLIP's cue is flagged missing
clip_missing_threshold: 0.40

use_gaze_features: false   # context gaze is unfinished — keep OFF until ready
```

---

## 4. Common cue schema + adapter contract (`adapters/base.py`)

Every adapter converts one model's **per-frame native output** into this normalised per-frame record:

```python
@dataclass
class NormalisedFrameCue:
    cue: str                      # "emotion" | "gesture" | "motion" | "context"
    frame_idx: int
    label: str                    # canonical label from schema.yaml (or "Unknown")
    confidence: float             # 0..1
    probs: dict[str, float]       # full distribution over that cue's classes ({} if N/A)
    valid: bool                   # confidence >= floor AND label != "Unknown"
    extra: dict                   # cue-specific extras (direction, activity, engaged...)
```

```python
class CueAdapter(ABC):
    @abstractmethod
    def normalise(self, raw_model_output, frame_idx: int) -> NormalisedFrameCue: ...
```

### 4.1 Emotion adapter
Native: `label`, `confidence`, 7-class `probs` dict. Nearly 1:1.
- Map class names to canonical (`Anger` stays `Anger`, etc.).
- `valid = confidence >= confidence_floor.emotion`.
- `probs` = the full 7-way dict.

### 4.2 Gesture adapter
Native: `class_id` + `confidence`, mapped to label; model already returns `Unknown` for sensitive gestures below 0.80; dynamic scenarios (Pointing / One Hand Raised / Arms Up) come from pose+rules.
- Map to canonical `gesture_classes`.
- `valid = (label != "Unknown") and (confidence >= confidence_floor.gesture)`.
- `probs` = `{}` if the model only gives top-1 conf (acceptable — handle in aggregation).
- `extra` = `{"point_direction": <vector or null>, "motion_direction": <toward/away/none>}`
  - **Object-target is NOT available in v1** (needs YOLO). Set `extra["point_target"] = "unknown"`. Rows whose disambiguation needs point-target (e.g. classroom row 16, kitchen row 36) are **expected partial failures in v1** — document, don't fix now.

### 4.3 Motion adapter
Native: `pose` label, `active_motion` label, `confidence`, 8-class `probabilities` dict. (The repo's LSTM is loaded but never used — ignore it; classification is a deterministic rule engine.)
- Two outputs → emit `cue="motion"` with `label=active_motion`, `probs=probabilities`, and `extra={"pose": pose}`. Pose becomes a separate feature column in aggregation.
- **`confidence` is a hand-authored constant per class (EMA-smoothed), NOT a learned probability.** Use it only as a coarse validity gate; do not feed it as a real-valued probability feature and do not calibrate it.
- **Cold-start guard (required):** when `<4` frames are buffered, the model fabricates `label="Standing Still", confidence=0.90`, which passes the floor. Set `valid=False` whenever `<4` frames are buffered, regardless of the confidence value.
- Otherwise `valid = confidence >= confidence_floor.motion` AND landmarks present.
- **Critical mapping check:** the table's `stand` is ambiguous — `Standing_Still` (idle) vs `Frozen_Rigid_Stand` (row 38 = emergency). Do NOT collapse these. Keep both classes distinct; they carry different intents.

### 4.4 Context adapter
Native: `ContextState(scene, scene_confidence, ...)`. **In practice this model delivers only `scene` and `scene_confidence`** — `objects`, `gaze`, `attention_object`, `activity`, `engaged` do not exist in the code and never will for this model.
- `label = scene`, `confidence = scene_confidence`, `probs = {classroom: p, kitchen: p}` (real 2-class distribution available).
- `confidence` is a **15-frame rolling-mean** of softmax — already smoothed. Read it directly; don't re-derive from the display string; don't over-average in aggregation (mild double-smoothing).
- Map the native `"uncertain"` label to `"Unknown"`.
- `extra = {"activity": None, "engaged": None, "n_objects": 0}` — **hardcoded "not measured" placeholders, clearly commented**, so fusion never reads `n_objects=0` as a real observation. Scenarios needing activity/engagement (rows 3, 22, 37) must resolve from scene+motion+gesture — expected partial degradation in v1.
- **Gaze is behind a flag and structurally absent.** Keep `use_gaze_features: false`. When/if a future model adds gaze, add `extra["looking_at_robot"]`, `extra["engagement"]` — no restructuring needed.

---

## 5. Phases

> For each phase: **Objective · Deliverable · Success criteria · Failure points · Experiments.**

### Phase 0a — Cue-model verification & uniform inference interface *(hard gate before Phase 0)* — **COMPLETE 2026-07-09**
**Objective:** Confirm the four models are *correct on this dataset* (not just "confident"), and wrap each behind one uniform interface so the Phase-0 orchestrator never knows a model's origin — while sidestepping the dependency conflict.

**Deliverables — all built:**
- **Verification evidence**: `MODEL_ANALYSIS.md` §5 (smoke tests + root-cause experiments for every gate below) instead of a separate `reports/phase0a_verification.md` — the analysis doc already carried this content, no need for a second file.
- **Four standalone runners** (`runners/{emotion,gesture,motion,context}_runner.py`), each importing refactored per-frame logic from the original (unmodified) cue-model repos. Each supports two modes:
  - single-clip: `--clip <path> --out <out.jsonl>`
  - **batch** (added once full-dataset scale — 1269 clips — made subprocess-per-clip impractical): `--manifest clips.csv --clips-root <dataset root> --out <combined.jsonl> [--resume]` — loads the model **once**, loops every clip in one process. Per-clip stateful trackers (MediaPipe Hands/Pose, gesture history deques) are recreated fresh per clip to match the native scripts' one-process-per-video behaviour; stateless classifiers (TFLite keypoint/point-history classifiers, PoseClassifier) are loaded once and reused.

**The uniform interface is at the *process* boundary, not the function boundary — confirmed necessary in practice.** Four isolated venvs under `.venvs/`: `emotion`, `gesture`, `motion`, `context`. **Documented deviation from pinned requirements.txt:** `motion` uses `mediapipe==0.10.11`, not the repo's pinned `0.10.14` — `0.10.14` had a genuine pip dependency conflict with the protobuf version mediapipe itself requires in this environment; `0.10.11` is the same version already proven working for `emotion`/`gesture` and exposes an identical `mp.solutions.pose` API. No behavioural difference expected or observed.

**Verification gates — all passed:**
- ✅ **Emotion label order** — verified by real inference (MODEL_ANALYSIS.md §5.1), not by obtaining training code. No swap found; a real face-detector parameter bug (`model_selection=0`→`1`) was found and fixed instead.
- ✅ **Context checkpoint match** — verified by real inference on classroom/kitchen clips (MODEL_ANALYSIS.md §5.2). Loads and runs correctly; 2 kitchen misclassifications root-caused to camera framing, not a checkpoint mismatch.
- ✅ **Motion confidence semantics + cold-start** — confirmed constant-based/EMA (code-read) and the cold-start guard is implemented and empirically verified to also catch mid-clip re-warm-up after occlusion (MODEL_ANALYSIS.md §5.3, `runners/motion_runner.py`'s `--stats-out` diagnostic).
- ✅ **Per-cue smoke run** — all four runners executed end-to-end on real clips, batch mode additionally smoke-tested with `--limit 3` before the full run.

**A runner bug was found and fixed during the full-dataset run, not before:** `process_clip()` originally raised `SystemExit` on an unreadable clip file; `SystemExit` inherits from `BaseException`, not `Exception`, so `run_batch()`'s per-clip `except Exception` did not catch it and one bad file crashed the entire multi-hour batch job. Fixed by raising `RuntimeError` instead (all 4 runners) — a lesson for any future runner: never let a per-clip failure be fatal to the whole batch, and test that exception-handling shape explicitly, not just the happy path.

**Data-integrity finding:** `clips.csv` referenced one clip file (`S07_F03_c060.mp4`) that did not exist on disk — this crashed both `gesture_runner.py` and `motion_runner.py` batch runs mid-flight (at the exact same manifest row, confirming it wasn't a runner-specific issue). Resolved by removing the row from `clips.csv` (now 1269 clips, was 1270).

**Known performance characteristics (informs hardware decisions):** all four models run CPU-only in this environment. Steady-state measured throughput per model: **~0.08-0.17 clips/second** (≈290-610 clips/hour), with a roughly 2x slowdown observed after ~250-300 clips into a run (likely CPU contention from the four parallel processes competing for the same cores once all are fully warmed up — not yet root-caused further). At this rate, a full 1269-clip run takes **roughly 2-5 hours per model**, all four running in parallel (bounded by the slowest). Model loading itself (mediapipe's import chain pulls in TensorFlow) costs a fixed ~70s per process, negligible against total runtime. If moving to different hardware, benchmark with `--limit 25` first to get a fresh rate estimate before committing to a full run.

**Forbidden during 0a — honored throughout:** no preprocessing/threshold constants were changed from the verified originals; the four cue-model repos were not modified; motion confidence was not treated as a calibrated probability anywhere.

---

### Phase 0 — Generate the measured-cue dataset *(the gate)* — **IN PROGRESS 2026-07-09**
**Objective:** Run all four models over every clip and record their *real* per-frame outputs. **Then verify the models actually reproduce the table's intended cues.** This is the highest-risk unverified assumption in the project.

**Hard contract (per explicit instruction — enforced in code, not just convention):** `aggregate_clip_cues.py` and `agreement_report.py` read **only** `NormalisedFrameCue` schema fields (`cue, frame_idx, label, confidence, probs, valid, extra`) plus the `clip_id` envelope the batch runners add. No cue-model-specific assumption (confidence distributions, specific misclassification patterns) is embedded in aggregation/report code — when a cue model is replaced later, only its runner should need to change.

**Deliverable (actual paths, superseding the originally-planned `data/measured/frame_cues.parquet`):**
- `pipeline/measured/{emotion,gesture,motion,context}_frame_cues.jsonl` — one combined JSONL per cue, all clips, `clip_id`-tagged (see §1 for why JSONL not parquet).
- `pipeline/measured/motion_stats.csv` — per-clip valid/invalid frame counts + cold-start/occlusion diagnostic (`total_frames, valid_frames, invalid_frames, cold_start_frames, no_landmark_frames, mid_clip_occlusion_resets`).
- `pipeline/measured/clip_cues.csv` (`aggregate_clip_cues.py`) — lightweight per-(clip, cue) majority-vote label over valid frames only, **not** the Phase 2 feature vector.
- `reports/phase0_agreement.md` + `.csv` (`agreement_report.py`): for every (scenario, cue) pair, intended (from `scenarios.csv`, canonicalized) vs. measured clip-level dominant cue, with:
  - `cue_corrupted` flag when the scenario-level measured dominant disagrees with intended
  - explicit surfacing of the two known systematic patterns: `thumbs_up→raise_hand` (gesture's 0.80 gate) and `Standing Still→Frozen/Rigid Stand` (motion's threshold) — see §0b findings 7/8
  - per-scenario clip counts, with singleton scenarios (`n_clips_total==1`) flagged separately as low-confidence signal
  - a clear **STOP** banner printed when any `cue_corrupted` scenario is found — **per explicit instruction, this pipeline halts for human review after this report; it does not auto-proceed to Phase 2 feature-building.**

**No cue-model gate/threshold/decision logic was modified to produce this data.** Corrupted outputs (disagreements) are recorded as findings in the report, not patched in the runners.

**Success criteria:** all four runners complete over the full 1269-clip dataset without crashing (see Phase 0a's runner-bug fix); `phase0_agreement.md` produced; every `cue_corrupted` scenario reviewed before Phase 2 begins.

**Failure points:**
- Emotion model misreads an emotion at distance → sarcasm/conflict cases unrecoverable *before fusion runs*. If this happens, it's a **finding**, not a bug to patch in fusion.
- Gesture model filters `thumbs_up` to `Unknown` below 0.80 on this footage → **confirmed happening**, see §0b finding 7.
- If agreement is low, **stop and report** — no fusion model can recover from wrong inputs. This may redirect effort to cue-model thresholds or re-recording, which is exactly what you want to learn now.

**Experiments:** confusion matrix per cue (model output vs intended) — see `phase0_agreement.md`'s full per-scenario-per-cue table; list of worst-agreement scenarios (the `cue_corrupted` rows).

---

### Phase 1 — *(folded into Phase 0)*
Vocabulary reconciliation is no longer a separate step (models assumed to emit the table's cue vocabulary). The adapters in §4 do the light normalisation. Keep the §3 schema as the contract.

---

### Phase 2 — Feature schema + adapters + aggregation
**Objective:** Turn per-frame measured cues into **one feature vector per clip** (Option A).

**Deliverable:** `pipeline/aggregate.py`, `pipeline/build_features.py` → `data/features/clip_features.parquet`.

**Aggregation rules (Option A):**
- **Probability cues (emotion, motion):** `mean` of the per-frame probability vectors over *valid* frames → N features each. Add `max_confidence` (max over frames) and `valid_fraction`.
- **Label-only cue (gesture):** majority-vote label over valid frames (one-hot), `mean_confidence` of winning label, `valid_fraction`.
- **Context:** one-hot `scene`, one-hot `activity`, `engaged` (bool→0/1), `scene_confidence`.
- **Missingness mask (one bit per cue):** `missing_<cue> = (valid_fraction < clip_missing_threshold)`. This is the explicit signal the fusion engine uses for graceful degradation — and gesture's own `Unknown` feeds it directly.
- **Spatial extras:** `motion_direction` (toward/away/none → one-hot), `point_target = unknown` (placeholder).

**Final feature vector (~30–40 dims):** 7 emotion-mean + emotion conf + 8 motion-mean + motion conf + pose one-hot(5) + gesture one-hot(8) + gesture conf + context scene one-hot + activity one-hot + engaged + 4 missing-bits + direction one-hot. Freeze this layout; both GBT and transformer consume it identically.

**Success criteria:** every clip → exactly one fixed-length vector; no NaNs; missing-cue clips correctly flagged.

**Failure points:** silently dropping all frames of a cue (→ all-zero vector indistinguishable from a real zero). Always pair zeros with the explicit missing-bit so the model can tell "absent" from "confidently zero".

---

### Phase 3 — Rule-based baseline *(the bar to beat)*
**Objective:** Encode the table's authoring logic as explicit IF-THEN rules. This is what GBT must beat — and the diagnostic for whether you even have a fusion problem.

**Deliverable:** `fusion/rule_based.py` — reads a clip feature vector, applies hand-written rules (priority-ordered, emergency first), returns intent.

**Success criteria:** runs on all clips; produces a baseline accuracy number under the §6 protocol.

**Failure points / the key diagnostic:**
- If the rule baseline already scores ~100% on the test set → **you do not yet have a fusion problem you can prove**; your data is rule-separable. Report this honestly; it shapes the paper (see earlier discussion) and tells you where to collect harder data.
- Build emergency rules with **asymmetric cost**: any meaningful evidence of F02 escalates, even at low confidence.

**Experiments:** ablate one cue at a time from the rule system → which scenarios break? Those are the cases that genuinely *need* fusion.

---

### Phase 4 — GBT fusion *(THE DELIVERABLE)*
**Objective:** A learned, interaction-aware fusion model that beats the rule baseline on conflict + missing-cue cases, runs trivially on Jetson, and explains its decisions.

**Deliverable:** `fusion/gbt.py` — LightGBM or XGBoost multiclass classifier over the Phase-2 features, with:
- **Calibration** (isotonic/Platt on the CV folds) so output probabilities are trustworthy.
- **Modality-dropout augmentation during training:** randomly set a cue's features to zero + missing-bit=1, so the model learns to redistribute weight (your training data has almost no missingness but your test set is full of it — without this, T03/T05 fail).
- **Class weighting** for imbalance; **F02 never down-weighted**.
- **SHAP per-prediction attribution** → the "which cue drove this decision" output (this is your attention-equivalent explainability).
- **Safety override:** non-trivial F02 mass → escalate regardless of argmax.

**Outputs per clip:** intent class, full probability distribution, confidence scalar (max-prob/entropy), per-cue SHAP attribution, safety flag.

**Success criteria:**
- Beats **all** baselines (majority / single-best-cue / rule-based) under leave-one-scenario-out CV.
- Specifically beats rules on **conflict cases** and **missing-cue cases**.
- **Zero F02 false-negatives** on the test set.
- Runs < a few ms per clip on Jetson (it will — GBT is microseconds; profile the *whole* pipeline, perception dominates).

**Failure points:**
- **Split leakage** — near-duplicate variations straddling train/test inflate accuracy. The §6 grouped split prevents this; verify it.
- Gains only on *seen* combinations (memorization) — check held-out scenario performance, not training fit.

**Experiments:** per-class confusion matrix; missing-cue robustness vs rule baseline; SHAP summary showing emotion overriding gesture on sarcasm cases.

---

### Phase 5 — Evaluation harness *(shared — scores ANY fusion model)*
**Objective:** One harness both GBT and the transformer are scored by, so the comparison is fair.

**Deliverable:** `eval/run_eval.py`, `eval/splits.py`, `eval/metrics.py`.

**Protocol (non-negotiable):**
- **Leave-one-scenario-out cross-validation**, grouping *all variations of a scenario together* so copies never straddle the split.
- Report: overall accuracy, **per-class** recall (F02 highlighted), conflict-case accuracy, missing-cue-case accuracy, and the **train/test accuracy gap** (this is the memorization detector used in Phase 6).
- Bootstrap confidence intervals — at this sample size, a "win" must survive variance to count.

**Success criteria:** identical harness produces comparable numbers for every model; results reproducible from a fixed seed.

---

### Phase 6 — Transformer experiment *(the controlled comparison)*
**Objective:** Produce *evidence* that a from-scratch transformer is data-starved at this scale — not an assertion. The result must be defensible: it has to fail *despite being built correctly*.

**Deliverable:** `fusion/transformer.py` — a small, heavily-regularised encoder over the **same** Phase-2 feature vectors (treat the cue groups as tokens), same splits, same harness as GBT.

**The persuasive outputs (this is what goes in the report):**
1. **Train/test gap:** transformer ≈ near-100% train, collapses on held-out scenarios; GBT's gap is smaller. The gap *is* the memorization evidence.
2. **Data-scaling curve:** train both on 25/50/100% of *scenarios*; plot test accuracy vs #scenarios. Transformer still climbing steeply at 100% while GBT plateaus = "transformer is data-starved, GBT is data-matched" in one chart.
3. **Honest tuning:** try 2–3 regularisation strengths / sizes so no one can say you didn't try.

**Success criteria:** a clean, fair comparison with the scaling curve. (A transformer that *loses fairly* is the successful outcome of this experiment.)

**Failure points:** sloppy build → "it failed because you built it badly." Reuse the GBT pipeline exactly; only the model class changes.

**Cheaper substitute if time-tight:** a small MLP shows nearly the same memorization story for far less effort. Include it as a fallback if the transformer slips.

---

### Phase 7 — Write-up
Summarise: rule baseline vs GBT vs transformer under the shared protocol; the scaling curve; the SHAP attribution examples; the Phase-0 cue-agreement findings; and an honest scoping of claims (**graceful degradation** + **learned cue interactions**, *not* "understands novel intent").

---

## 6. Critical correctness checklist (paste into the agent's instructions)

- [x] **Emotion label order verified** — by real inference on known clips (MODEL_ANALYSIS.md §5.1), not training-code comparison.
- [x] **Context checkpoint smoke-tested** on known clips (MODEL_ANALYSIS.md §5.2).
- [x] **Motion cold-start guard**: `<4` buffered frames → `valid=False` regardless of the fabricated 0.90 confidence — **and verified to also re-trigger mid-clip after an occlusion resets the buffer**, not just at clip start (`runners/motion_runner.py`).
- [ ] **Motion confidence not used as a real-valued probability feature** (it's a hand-authored constant). *(Applies from Phase 2 onward — not yet built.)*
- [ ] **No global confidence-weighted vote across cues** (confidences aren't commensurate). *(Applies from Phase 3+ — not yet built.)*
- [x] Context `activity`/`engaged`/`n_objects` are hardcoded "not measured" placeholders, not real zeros (`context_runner.py`).
- [x] **Uniform runners are venv-isolated** (`.venvs/{emotion,gesture,motion,context}`); no attempt to unify environments.
- [x] Cue-model repos **not modified**; runners are new wrapper modules (`runners/*.py`).
- [x] **Aggregation/report code depends only on the `NormalisedFrameCue` schema** — no cue-model-specific assumptions embedded (`pipeline/aggregate_clip_cues.py`, `pipeline/agreement_report.py`).
- [ ] Features come from **measured model outputs** (Phase 0), never the authored table. *(Phase 0 in progress; Phase 2 not yet built — this rule will apply then.)*
- [x] `splits.csv` is **scenario-grouped** (`split_scenario`, primary) with a **separate subject-grouped** column (`split_subject`) — no scenario_id/subject_id spans multiple splits (asserted in `build_splits.py`). A random clip-level split is included only as a clearly-labelled, documented non-primary contrast (`split_random_leaky_DO_NOT_USE_FOR_EVAL`).
- [ ] **Missing-bit per cue** present and paired with zeroed features. *(Phase 2, not yet built.)*
- [ ] **Modality-dropout** applied in GBT training. *(Phase 4, not yet built.)*
- [ ] Probabilities **calibrated** before confidence-based logic — **emotion/context only**. *(Phase 4, not yet built.)*
- [ ] **F02 never down-weighted**; safety override implemented. *(Phase 4, not yet built.)*
- [x] `Standing_Still` vs `Frozen_Rigid_Stand` kept distinct in the runner output — **and their frequent confusion is now a tracked, named finding** (§0b #8), not silently absorbed.
- [x] Gaze features OFF — structurally absent from the context model, not just flagged off.
- [x] `point_target = unknown` in v1 (`gesture_runner.py`'s `extra`); point-target-dependent scenarios not yet separately enumerated against this dataset's scenario table.
- [ ] Transformer uses the **same features + same splits + same harness** as GBT. *(Phase 6, not yet built.)*
- [ ] Every reported "win" has a bootstrap confidence interval. *(Phase 5, not yet built.)*

---

## 7. Suggested build order for the AI coding tool

1. ~~`configs/schema.yaml` + `data/labels/scenarios.csv` (hand-authored)~~ — **superseded**: dataset ships its own `scenarios.csv`/`clips.csv`; canonical vocab lives in `runners/common/constants.py` (confidence floors) and each runner's own native label lists — no separate `schema.yaml` was needed in practice.
2. ✅ **Phase 0a — verification + uniform runners.** Done via real inference on known clips rather than model-owner interviews (§0b). Four venv-isolated batch-capable runners built and smoke-tested.
3. ✅ Runner-level correctness fixes baked in (motion cold-start guard incl. mid-clip occlusion, motion LSTM excluded, gesture div-by-zero patch, context Unknown mapping + placeholders, emotion `model_selection=1`).
4. 🔶 **`pipeline/build_splits.py` → `splits.csv`; `pipeline/canonical_map.py`; batch-run the four runners over all 1269 clips → `pipeline/measured/*.jsonl`; `aggregate_clip_cues.py` → `clip_cues.csv`; `agreement_report.py` → `phase0_agreement.md`. IN PROGRESS as of 2026-07-09 — full-dataset batch run is CPU-bound and slow (§Phase 0a performance note); consider a faster machine if this needs to repeat often.**
5. **STOP HERE per explicit instruction — do not proceed to step 6 until `phase0_agreement.md` has been reviewed.**
6. `pipeline/aggregate.py` + `build_features.py` → `clip_features.parquet`. *(Not started.)*
7. `eval/` harness. *(Not started.)*
8. `fusion/rule_based.py` → get the baseline number. *(Not started.)*
9. `fusion/gbt.py` → beat it → **LOCK THIS. This is the working deliverable.** *(Not started.)*
10. `fusion/transformer.py` → run the comparison experiment. *(Not started.)*
11. `reports/` → plots + write-up. *(Not started.)*

**Stop-and-report gate 1:** ✅ cleared — Phase 0a verification passed (§0b).
**Stop-and-report gate 2 (current gate):** after Phase 0's agreement report, if `cue_corrupted` scenarios are found (expected: at least the two known systematic patterns, §0b #7/#8), halt and surface it before building any feature/fusion code. That finding is more valuable than a fusion model trained on bad inputs.
