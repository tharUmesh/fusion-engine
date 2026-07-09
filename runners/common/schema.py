"""
Canonical per-frame cue record, shared by all four runners.

Stdlib-only by design: each runner executes in its own isolated venv with its
own pinned third-party dependencies (see Integration_API.md #4), so this
module must import cleanly everywhere without adding a dependency of its own.
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass
class NormalisedFrameCue:
    cue: str                      # "emotion" | "gesture" | "motion" | "context"
    frame_idx: int
    label: str                    # canonical label (or "Unknown")
    confidence: float             # 0..1
    probs: Dict[str, float] = field(default_factory=dict)
    valid: bool = False
    extra: dict = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json_line() + "\n")


def append_batch(f, clip_id: str, records):
    """Batch mode: one combined JSONL per model, each line = NormalisedFrameCue
    fields plus a clip_id envelope (frame_idx alone isn't unique across clips)."""
    for r in records:
        d = asdict(r)
        d["clip_id"] = clip_id
        f.write(json.dumps(d) + "\n")


def read_manifest(manifest_csv):
    """Minimal stdlib CSV reader for Data/Dataset/.../annotations/clips.csv."""
    import csv
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
