"""
Builds Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/splits.csv.

Produces TWO independent grouped train/val/test assignments, as separate
columns on the same clip-level file:

  split_scenario -- all clips of one scenario_id assigned to exactly one of
                    train/val/test. Tests generalization to unseen cue
                    combinations. This is the PRIMARY fusion evaluation split.
  split_subject  -- all clips of one subject_id assigned to exactly one of
                    train/val/test. Tests generalization to unseen people.

IMPORTANT DATASET-SPECIFIC FINDING: in hri-multimodal-intent-v1.0.0, subject_id
and scenario_id are in a perfect 1:1 relationship (23 subjects, 23 scenarios,
each subject performed exactly one scenario). This means split_scenario and
split_subject are, in this dataset version, THE SAME PARTITION of clips --
"unseen scenario" and "unseen subject" cannot be tested independently here.
This script verifies and reports that confound explicitly rather than hiding
it; it does not silently produce two splits that look independent but aren't.

A third, clearly-labelled column is also included: split_random_leaky_DO_NOT_USE_FOR_EVAL,
an ordinary clip-level random split with NO grouping. It is an optimistic
upper bound only (near-duplicate clips from the same scenario/subject can
straddle train/test), included solely as a documented point of methodological
contrast -- never use it as a headline number.

Both grouped splits use a greedy largest-group-first bin-packing allocation
targeting 70/15/15 by CLIP COUNT (not group count, since group sizes vary
11-73 clips) while keeping every group intact in one split.

Before writing, this script verifies (asserts) that no scenario_id or
subject_id spans multiple splits. Stdlib-only.
"""
import csv
import os
import random
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(REPO_ROOT, "Data", "Dataset", "hri-multimodal-intent-v1.0.0")
CLIPS_CSV = os.path.join(DATASET_ROOT, "annotations", "clips.csv")
OUT_CSV = os.path.join(DATASET_ROOT, "annotations", "splits.csv")

TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def read_clips(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def greedy_grouped_split(rows, group_key):
    """Assigns each distinct value of group_key entirely to one split,
    greedily filling the split currently furthest below its target share of
    total clip count. Returns {group_value: split_name}."""
    group_counts = defaultdict(int)
    for r in rows:
        group_counts[r[group_key]] += 1

    total = sum(group_counts.values())
    targets = {s: TARGET_RATIOS[s] * total for s in TARGET_RATIOS}
    current = {s: 0 for s in TARGET_RATIOS}

    # Largest group first -- standard greedy bin-packing heuristic, keeps the
    # final proportions closer to target than smallest-first would.
    ordered_groups = sorted(group_counts.items(), key=lambda kv: -kv[1])

    assignment = {}
    for group_value, count in ordered_groups:
        # pick the split with the largest remaining deficit (target - current)
        best_split = max(TARGET_RATIOS, key=lambda s: targets[s] - current[s])
        assignment[group_value] = best_split
        current[best_split] += count

    return assignment, current, targets


def verify_no_leakage(rows, group_key, assignment_col):
    """Raises if any group_key value spans more than one split value."""
    seen = {}
    for r in rows:
        g = r[group_key]
        s = r[assignment_col]
        if g in seen and seen[g] != s:
            raise AssertionError(
                f"LEAKAGE: {group_key}={g} spans splits {seen[g]!r} and {s!r} in column {assignment_col}")
        seen[g] = s
    return True


def main():
    rows = read_clips(CLIPS_CSV)
    print(f"Loaded {len(rows)} clips from {CLIPS_CSV}")

    scenario_assignment, scen_counts, scen_targets = greedy_grouped_split(rows, "scenario_id")
    subject_assignment, subj_counts, subj_targets = greedy_grouped_split(rows, "subject_id")

    for r in rows:
        r["split_scenario"] = scenario_assignment[r["scenario_id"]]
        r["split_subject"] = subject_assignment[r["subject_id"]]

    # Random clip-level split -- included ONLY as a labelled, documented
    # optimistic-upper-bound contrast. Never the primary split.
    rng = random.Random(RANDOM_SEED)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TARGET_RATIOS["train"])
    n_val = int(n * TARGET_RATIOS["val"])
    for i, r in enumerate(shuffled):
        if i < n_train:
            r["split_random_leaky_DO_NOT_USE_FOR_EVAL"] = "train"
        elif i < n_train + n_val:
            r["split_random_leaky_DO_NOT_USE_FOR_EVAL"] = "val"
        else:
            r["split_random_leaky_DO_NOT_USE_FOR_EVAL"] = "test"

    # ── Verify no leakage before writing anything ──────────────────────────
    verify_no_leakage(rows, "scenario_id", "split_scenario")
    verify_no_leakage(rows, "subject_id", "split_subject")
    print("Leakage check passed: no scenario_id or subject_id spans multiple splits.")

    # ── Report the subject/scenario confound explicitly ────────────────────
    scenario_to_subjects = defaultdict(set)
    subject_to_scenarios = defaultdict(set)
    for r in rows:
        scenario_to_subjects[r["scenario_id"]].add(r["subject_id"])
        subject_to_scenarios[r["subject_id"]].add(r["scenario_id"])
    max_subjects_per_scenario = max(len(v) for v in scenario_to_subjects.values())
    max_scenarios_per_subject = max(len(v) for v in subject_to_scenarios.values())
    is_confounded = (max_subjects_per_scenario == 1 and max_scenarios_per_subject == 1)

    identical_partition = all(
        scenario_assignment[r["scenario_id"]] == subject_assignment[r["subject_id"]] for r in rows)

    print(f"\nSubject/scenario relationship: max {max_subjects_per_scenario} subject(s)/scenario, "
          f"max {max_scenarios_per_subject} scenario(s)/subject -> "
          f"{'CONFOUNDED (1:1)' if is_confounded else 'independent'}")
    print(f"split_scenario == split_subject for every clip: {identical_partition}")

    print("\nsplit_scenario clip counts:", dict(scen_counts),
          "targets:", {k: round(v) for k, v in scen_targets.items()})
    print("split_subject  clip counts:", dict(subj_counts),
          "targets:", {k: round(v) for k, v in subj_targets.items()})

    print("\nscenario -> split_scenario:")
    for scen, split in sorted(scenario_assignment.items()):
        print(f"  {scen}: {split}")

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUT_CSV}")

    return {
        "is_confounded": is_confounded,
        "identical_partition": identical_partition,
        "scenario_assignment": scenario_assignment,
        "subject_assignment": subject_assignment,
        "scen_counts": dict(scen_counts),
        "subj_counts": dict(subj_counts),
    }


if __name__ == "__main__":
    main()
