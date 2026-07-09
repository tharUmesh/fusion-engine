"""
Phase 0 agreement report: per scenario, per cue, compares the INTENDED cue
(from scenarios.csv, transcribed by whoever authored the dataset) against the
MEASURED clip-level dominant cue (majority vote over valid frames, from
aggregate_clip_cues.py's output) -- and flags scenarios where they disagree
as cue_corrupted.

This script does NOT modify any cue-model gate/threshold/decision logic. It
only compares what the models already, faithfully emitted (via
aggregate_clip_cues.py, which itself only reads the NormalisedFrameCue
schema) against the authored ground truth. Disagreements are recorded as
findings, not "fixed".

Two systematic disagreement patterns are explicitly surfaced everywhere they
occur, because they were already identified as root-caused, not-a-bug model
behaviours during runner validation (see MODEL_ANALYSIS.md #5.3):
  - thumbs_up -> raise_hand   (gesture's 0.80 sensitive-gesture confidence gate)
  - Standing Still -> Frozen/Rigid Stand   (motion's body_speed<0.08 threshold)

Singleton scenarios (exactly 1 clip) are flagged separately -- a "majority"
of one clip is not a statistically meaningful agreement/disagreement signal.

Stdlib-only. Reads:
  - Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/clips.csv
  - Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/scenarios.csv
  - pipeline/measured/clip_cues.csv (from aggregate_clip_cues.py)
Writes:
  - reports/phase0_agreement.csv
  - reports/phase0_agreement.md
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from canonical_map import map_intended  # noqa: E402

DATASET_ROOT = os.path.join(REPO_ROOT, "Data", "Dataset", "hri-multimodal-intent-v1.0.0")
CLIPS_CSV = os.path.join(DATASET_ROOT, "annotations", "clips.csv")
SCENARIOS_CSV = os.path.join(DATASET_ROOT, "annotations", "scenarios.csv")
CLIP_CUES_CSV = os.path.join(REPO_ROOT, "pipeline", "measured", "clip_cues.csv")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

CUE_INTENDED_COL = {
    "emotion": "Intended Emotion",
    "gesture": "Intended Gesture",
    "motion": "Intended Motion",
    "context": "Context",  # scenarios.csv has both "Context" and "Intended Context"; identical in this version
}

# Known, already-root-caused systematic disagreement patterns (see module
# docstring) -- surfaced with a dedicated flag wherever they occur, on top of
# the general cue_corrupted flag.
KNOWN_SYSTEMATIC_PATTERNS = {
    ("gesture", "thumbs_up", "raise_hand"): "gesture 0.80 confidence gate (MODEL_ANALYSIS.md #5.3)",
    ("motion", "Standing Still", "Frozen/Rigid Stand"): "motion body_speed<0.08 threshold (MODEL_ANALYSIS.md #5.3/#3.9)",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    if not os.path.isfile(CLIP_CUES_CSV):
        raise SystemExit(f"{CLIP_CUES_CSV} not found -- run aggregate_clip_cues.py first")

    clips = read_csv(CLIPS_CSV)
    scenarios = read_csv(SCENARIOS_CSV)
    clip_cues = read_csv(CLIP_CUES_CSV)

    clip_to_scenario = {r["clip_id"]: r["scenario_id"] for r in clips}
    scenario_base = {r["scenario_id"]: r for r in scenarios}  # keyed by "S01" etc.

    # clip_cues.csv rows: (clip_id, cue) -> dict
    cc_index = defaultdict(dict)  # scenario_id -> cue -> list of clip-level rows
    for r in clip_cues:
        clip_id = r["clip_id"]
        scenario_id = clip_to_scenario.get(clip_id)
        if scenario_id is None:
            continue
        cue = r["cue"]
        r["insufficient_valid_frames"] = (r["insufficient_valid_frames"] == "True")
        cc_index[scenario_id].setdefault(cue, []).append(r)

    all_scenario_ids = sorted(clip_to_scenario_scenarios(clip_to_scenario))

    report_rows = []
    for scenario_id in all_scenario_ids:
        base = scenario_id.split("_")[0]
        scen_row = scenario_base.get(base)
        if scen_row is None:
            print(f"WARNING: no scenarios.csv entry for base scenario '{base}' (from {scenario_id}) -- skipping")
            continue

        for cue, intended_col in CUE_INTENDED_COL.items():
            intended_raw = scen_row.get(intended_col, "")
            try:
                intended_canonical = map_intended(cue, intended_raw)
            except KeyError as e:
                print(f"WARNING: {e}")
                intended_canonical = None

            clip_rows = cc_index.get(scenario_id, {}).get(cue, [])
            n_clips_total = len(clip_rows)
            usable = [r for r in clip_rows if not r["insufficient_valid_frames"] and r["dominant_label"]]
            n_insufficient = n_clips_total - len(usable)

            measured_votes = Counter(r["dominant_label"] for r in usable)
            scenario_measured_dominant = measured_votes.most_common(1)[0][0] if measured_votes else None
            n_dominant_votes = measured_votes.most_common(1)[0][1] if measured_votes else 0

            n_agreeing = sum(1 for r in usable if r["dominant_label"] == intended_canonical) if intended_canonical else None
            n_disagreeing = (len(usable) - n_agreeing) if n_agreeing is not None else None

            has_intended = intended_canonical is not None
            has_measured = scenario_measured_dominant is not None
            cue_corrupted = bool(has_intended and has_measured and scenario_measured_dominant != intended_canonical)

            pattern_key = (cue, intended_canonical, scenario_measured_dominant)
            known_pattern = KNOWN_SYSTEMATIC_PATTERNS.get(pattern_key)

            report_rows.append({
                "scenario_id": scenario_id,
                "cue": cue,
                "intended_raw": intended_raw,
                "intended_canonical": intended_canonical if has_intended else "(no intended value)",
                "measured_dominant": scenario_measured_dominant if has_measured else "(no valid measurement)",
                "n_clips_total": n_clips_total,
                "n_clips_usable": len(usable),
                "n_clips_insufficient_valid_frames": n_insufficient,
                "n_dominant_votes": n_dominant_votes,
                "n_clips_agreeing": n_agreeing if n_agreeing is not None else "",
                "n_clips_disagreeing": n_disagreeing if n_disagreeing is not None else "",
                "cue_corrupted": cue_corrupted,
                "is_singleton_scenario": n_clips_total == 1,
                "known_systematic_pattern": known_pattern or "",
            })

    REPORTS_DIR_ensure()
    csv_path = os.path.join(REPORTS_DIR, "phase0_agreement.csv")
    fieldnames = list(report_rows[0].keys()) if report_rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Wrote {csv_path} ({len(report_rows)} scenario x cue rows)")

    write_markdown_report(report_rows)

    corrupted = [r for r in report_rows if r["cue_corrupted"]]
    singleton_corrupted = [r for r in corrupted if r["is_singleton_scenario"]]
    print(f"\n{'='*70}")
    print(f"cue_corrupted scenarios: {len(corrupted)} / {len(report_rows)} (scenario, cue) pairs")
    print(f"  of which singleton-scenario (n_clips_total==1, low confidence): {len(singleton_corrupted)}")
    if corrupted:
        print("STOP: cue_corrupted scenarios found -- review reports/phase0_agreement.md "
              "before proceeding to feature-building.")
    print(f"{'='*70}")


def clip_to_scenario_scenarios(clip_to_scenario):
    return set(clip_to_scenario.values())


def REPORTS_DIR_ensure():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def write_markdown_report(rows):
    path = os.path.join(REPORTS_DIR, "phase0_agreement.md")
    corrupted = [r for r in rows if r["cue_corrupted"]]
    singletons_corrupted = [r for r in corrupted if r["is_singleton_scenario"]]
    known_pattern_rows = [r for r in rows if r["known_systematic_pattern"]]

    lines = []
    lines.append("# Phase 0 — Cue Agreement Report")
    lines.append("")
    lines.append("Intended cue (scenarios.csv, authored) vs. measured clip-level dominant cue")
    lines.append("(majority vote over VALID frames only, per `aggregate_clip_cues.py`). No cue-model")
    lines.append("gate/threshold/decision logic was modified to produce this data — disagreements are")
    lines.append("recorded as findings, not patched. **Do not proceed to feature-building until this")
    lines.append("report has been reviewed.**")
    lines.append("")
    lines.append(f"**Total (scenario, cue) pairs evaluated:** {len(rows)}")
    lines.append(f"**cue_corrupted (measured dominant disagrees with intended):** {len(corrupted)}")
    lines.append(f"**...of which singleton scenarios (n_clips==1, low-confidence signal):** {len(singletons_corrupted)}")
    lines.append("")

    lines.append("## Known systematic disagreement patterns")
    lines.append("")
    lines.append("These two patterns were already root-caused during runner validation (not runner bugs —")
    lines.append("see `MODEL_ANALYSIS.md` §5.3) and are explicitly surfaced here wherever they recur:")
    lines.append("")
    lines.append("| scenario_id | cue | intended | measured | pattern |")
    lines.append("|---|---|---|---|---|")
    if known_pattern_rows:
        for r in known_pattern_rows:
            lines.append(f"| {r['scenario_id']} | {r['cue']} | {r['intended_canonical']} | "
                          f"{r['measured_dominant']} | {r['known_systematic_pattern']} |")
    else:
        lines.append("| (none observed in this run) | | | | |")
    lines.append("")

    lines.append("## All cue_corrupted scenarios")
    lines.append("")
    lines.append("| scenario_id | cue | intended | measured | clips (usable/total) | agree/disagree | singleton? | known pattern |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(corrupted, key=lambda r: (r["cue"], r["scenario_id"])):
        lines.append(
            f"| {r['scenario_id']} | {r['cue']} | {r['intended_canonical']} | {r['measured_dominant']} | "
            f"{r['n_clips_usable']}/{r['n_clips_total']} | {r['n_clips_agreeing']}/{r['n_clips_disagreeing']} | "
            f"{'**YES**' if r['is_singleton_scenario'] else 'no'} | {r['known_systematic_pattern'] or '—'} |")
    if not corrupted:
        lines.append("| (none) | | | | | | | |")
    lines.append("")

    lines.append("## Full per-scenario-per-cue table")
    lines.append("")
    lines.append("| scenario_id | cue | intended | measured | clips (usable/total) | insufficient-valid | corrupted? | singleton? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["scenario_id"], r["cue"])):
        lines.append(
            f"| {r['scenario_id']} | {r['cue']} | {r['intended_canonical']} | {r['measured_dominant']} | "
            f"{r['n_clips_usable']}/{r['n_clips_total']} | {r['n_clips_insufficient_valid_frames']} | "
            f"{'YES' if r['cue_corrupted'] else 'no'} | {'YES' if r['is_singleton_scenario'] else 'no'} |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
