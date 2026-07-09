# Migrating Phase 0 to another machine

Steps to pick up the full-dataset batch run on a different (ideally GPU) machine.

## 1. Clone

```
git clone https://github.com/tharUmesh/fusion-engine.git
cd "Fusion Engine"
```

The repo already includes the small stuff: cue-model code + weights (`Emotion/Gesture/Motion/Context Repo/`),
the runners (`runners/`), the pipeline scripts (`pipeline/`), the docs, and the dataset's
**annotations only** (`Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/{clips.csv,scenarios.csv,splits.csv}`).

**Not included** (see `.gitignore`): `.venvs/` (7.5GB, platform-specific) and
`Data/Dataset/hri-multimodal-intent-v1.0.0/raw/` (2.5GB of actual video clips) — bring the `raw/`
folder over separately and place it at exactly:

```
Fusion Engine/Data/Dataset/hri-multimodal-intent-v1.0.0/raw/clips/classroom/...
Fusion Engine/Data/Dataset/hri-multimodal-intent-v1.0.0/raw/clips/kitchen/...
```

i.e. as a sibling of the already-present `annotations/` folder. `clips.csv`'s `filepath` column is relative
to `Data/Dataset/hri-multimodal-intent-v1.0.0/`, so the structure must match exactly or the runners will report
"Cannot open clip" for everything.

**Verify before running anything:**
```
python -c "
import csv, os
root = 'Data/Dataset/hri-multimodal-intent-v1.0.0'
missing = [r['clip_id'] for r in csv.DictReader(open(root+'/annotations/clips.csv', encoding='utf-8'))
           if not os.path.isfile(os.path.join(root, r['filepath']))]
print(f'{len(missing)} missing clip files')
"
```
Should print `0 missing clip files`.

## 2. Recreate the four venvs

Each cue model needs its own isolated venv (see `Integration_API.md` §4 for why — Gesture needs
`numpy==1.26`, Motion's mediapipe combo needs a different protobuf/tensorflow resolution than a naive
install gives you). Python **3.10** was used here (some pinned wheels, e.g. `mediapipe==0.10.11`, are
`cp310`-specific on Windows — check wheel availability if using a different Python minor version).

**If the new machine has a CUDA GPU**, replace the CPU-only torch install line below with the CUDA build
matching the box's CUDA version (check with `nvidia-smi`), e.g.:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(swap `cu121` for whatever matches). This will make Emotion/Context/Motion's torch inference use the GPU
automatically — the runners already do `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, no
code changes needed. **Gesture and Motion's MediaPipe hand/pose detection will likely stay CPU-bound
regardless** — MediaPipe's legacy Python "solutions" API has limited GPU delegate support on Windows, so
don't expect a GPU to speed those two up as much as Emotion/Context.

```bash
# Emotion
python -m venv .venvs/emotion
.venvs/emotion/Scripts/python.exe -m pip install --upgrade pip
.venvs/emotion/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or cuXXX, see above
.venvs/emotion/Scripts/python.exe -m pip install "mediapipe==0.10.11" "tensorflow==2.15.1" "protobuf==3.20.3" opencv-python pillow "numpy==1.26.4"

# Context
python -m venv .venvs/context
.venvs/context/Scripts/python.exe -m pip install --upgrade pip
.venvs/context/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or cuXXX
.venvs/context/Scripts/python.exe -m pip install opencv-python pillow numpy

# Gesture
python -m venv .venvs/gesture
.venvs/gesture/Scripts/python.exe -m pip install --upgrade pip
.venvs/gesture/Scripts/python.exe -m pip install "numpy==1.26.4" "opencv-python==4.9.0.80" "mediapipe==0.10.11" "tensorflow==2.15.1" "protobuf==3.20.3"

# Motion
python -m venv .venvs/motion
.venvs/motion/Scripts/python.exe -m pip install --upgrade pip
.venvs/motion/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or cuXXX
.venvs/motion/Scripts/python.exe -m pip install "mediapipe==0.10.11" "tensorflow==2.15.1" "protobuf==3.20.3" opencv-python opencv-contrib-python "numpy==1.26.4"
```

**Note the `mediapipe==0.10.11` deviation** from Motion's own `requirements.txt` pin (`0.10.14`) — that
exact version has a real pip dependency conflict with protobuf in this setup; `0.10.11` is confirmed
equivalent for our purposes (same `mp.solutions.pose`/`mp.solutions.hands` API). See
`HRI_Fusion_Engine_Handover.md` §Phase 0a for the full explanation.

**Verify each venv before the full run** (should print version info with no errors):
```
.venvs/emotion/Scripts/python.exe -c "import torch, cv2, mediapipe as mp; print(mp.solutions.face_detection)"
.venvs/context/Scripts/python.exe -c "import torch, cv2; print('ok')"
.venvs/gesture/Scripts/python.exe -c "import cv2, mediapipe as mp; print(mp.solutions.hands)"
.venvs/motion/Scripts/python.exe -c "import torch, cv2, mediapipe as mp; print(mp.solutions.pose)"
```

## 3. Benchmark before committing to the full run

```bash
DATASET="Data/Dataset/hri-multimodal-intent-v1.0.0"
mkdir -p pipeline/measured
time .venvs/emotion/Scripts/python.exe runners/emotion_runner.py \
  --manifest "$DATASET/annotations/clips.csv" --clips-root "$DATASET" \
  --out /tmp/bench_emotion.jsonl --limit 25
```
On the original CPU-only machine this measured ~0.08-0.17 clips/s per model (a full 1269-clip run took an
estimated 2-5 hours per model, four running in parallel). Compare your new machine's rate before assuming
it'll be faster — Gesture/Motion in particular may not benefit much from a GPU (see note above).

## 4. Full run

```bash
DATASET="Data/Dataset/hri-multimodal-intent-v1.0.0"

.venvs/emotion/Scripts/python.exe runners/emotion_runner.py --manifest "$DATASET/annotations/clips.csv" --clips-root "$DATASET" --out "pipeline/measured/emotion_frame_cues.jsonl" --resume > pipeline/measured/emotion_run.log 2>&1 &
.venvs/context/Scripts/python.exe runners/context_runner.py --manifest "$DATASET/annotations/clips.csv" --clips-root "$DATASET" --out "pipeline/measured/context_frame_cues.jsonl" --resume > pipeline/measured/context_run.log 2>&1 &
.venvs/gesture/Scripts/python.exe runners/gesture_runner.py --manifest "$DATASET/annotations/clips.csv" --clips-root "$DATASET" --out "pipeline/measured/gesture_frame_cues.jsonl" --resume > pipeline/measured/gesture_run.log 2>&1 &
.venvs/motion/Scripts/python.exe runners/motion_runner.py --manifest "$DATASET/annotations/clips.csv" --clips-root "$DATASET" --out "pipeline/measured/motion_frame_cues.jsonl" --stats-out "pipeline/measured/motion_stats.csv" --resume > pipeline/measured/motion_run.log 2>&1 &

wait
```

`--resume` is safe to always include — on a fresh machine with no prior `pipeline/measured/*.jsonl` it's a
no-op; if a run gets interrupted, rerunning the same command skips clips already recorded (matched by
`clip_id`).

Progress: the runners' own `print()` progress lines (every 25 clips) go to the `.log` files but may be
buffered when redirected — check actual progress via line count instead:
```bash
wc -l pipeline/measured/*_frame_cues.jsonl
```
Expect roughly `1269 x avg_frames_per_clip` (~112) ≈ 142,000 lines per file when a model finishes.

## 5. After all four finish

```bash
# any plain python works for these two -- stdlib only, no venv needed
python pipeline/aggregate_clip_cues.py
python pipeline/agreement_report.py
```

Read `reports/phase0_agreement.md`. **Per the project's own rule: do not proceed to Phase 2
feature-building until this report has been reviewed** — it will flag `cue_corrupted` scenarios,
specifically including the two already-known systematic patterns (`thumbs_up→raise_hand`,
`Standing Still→Frozen/Rigid Stand`).

## 6. Bringing results back (if you want to switch back to the original machine)

Copy `pipeline/measured/` (the `.jsonl`/`.csv` outputs) and `reports/` back — those are the only new
artifacts from this run that aren't already in git (they're gitignored to keep the repo small; commit them
manually with `git add -f pipeline/measured reports` if you want them version-controlled after they're
complete, or just copy the folder).
