#!/usr/bin/env python3
"""Run the trench alignment preflight on one level's staged candidates.

The audit's ``unrestricted`` fresh-cover verdict is a function of the dig
raster, the occupancy (padding) and the finite-section metadata only
(``analyze_map`` -> ``action_table`` / ``monotone_closure(require_dump=False)``;
dumpability and accepted-dump cells enter only the ``same_base_accepted_dump``
branch). Sibling conditions of a pair slot share all three by construction, so
this wrapper asserts that equality slot by slot and audits one representative
condition per level; the verdict is then recorded for every sibling. Any slot
whose siblings differ is audited per sibling instead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

TOOL = c.pinned.TERRA_ROOT / "tools" / "audit_trench_alignment_feasibility.py"
SECTION_KEYS = ("axes_ABC", "trench_segments_yx", "trench_half_width_tiles")


def key_of(dataset: Path, slot: int):
    target = np.load(dataset / "images" / f"img_{slot}.npy")
    occupancy = np.load(dataset / "occupancy" / f"img_{slot}.npy")
    metadata = json.loads((dataset / "metadata" / f"trench_{slot}.json").read_text())
    return ((target < 0).tobytes(), np.asarray(occupancy, dtype=bool).tobytes(),
            json.dumps({k: metadata.get(k) for k in SECTION_KEYS}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", choices=("default", "yawonly"), default="default",
                        help="default: the tool's 2122b2df defaults (v2, on-the-line bound "
                        "EnvConfig.trench_dig_max_offset_m); yawonly: --max-offset-m 0, the "
                        "v2 yaw-parallel contract the current bank was re-admitted under "
                        "(tools/trench_align_v2_revalidation_20260901/preflight_full_v2.json)")
    args = parser.parse_args()
    level_dir = args.level_dir.resolve()
    datasets = sorted(p.name for p in level_dir.iterdir()
                      if (p / "manifest.jsonl").is_file())
    rows = {name: c.read_jsonl(level_dir / name / "manifest.jsonl") for name in datasets}
    trench = [n for n in datasets if rows[n] and rows[n][0]["family"] == "trench"]
    counts = {len(rows[n]) for n in trench}
    if len(counts) != 1:
        raise SystemExit(f"sibling datasets differ in length: { {n: len(rows[n]) for n in trench} }")
    slots = counts.pop()
    representative = trench[0]
    unequal = []
    for slot in range(1, slots + 1):
        pairs = {rows[n][slot - 1]["pair_slot_id"] for n in trench}
        if len(pairs) != 1:
            raise SystemExit(f"slot {slot}: siblings are not one pair slot: {pairs}")
        keys = {key_of(level_dir / n, slot) for n in trench}
        if len(keys) != 1:
            unequal.append(slot)
    audited = [representative] if not unequal else trench
    output = level_dir / f"preflight_{args.tag}.json"
    command = [sys.executable, str(TOOL), "--layout", "exact", "--bank", str(level_dir),
               "--workers", str(args.workers), "--witness-actions", "counts",
               "--output", str(output)]
    for name in audited:
        command += ["--dataset", name]
    if args.limit:
        command += ["--limit", str(args.limit)]
    if args.tag == "yawonly":
        command += ["--max-offset-m", "0"]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    report = json.loads(output.read_text())
    verdict = {}
    for result in report["results"]:
        slot = int(result["label"].rsplit(":", 1)[1])
        dataset = result["dataset"]
        complete = bool(result["unrestricted"]["complete"])
        names = trench if dataset == representative and not unequal else [dataset]
        for name in names:
            verdict[f"{name}:{slot}"] = {
                "complete": complete,
                "remaining_cells": result["unrestricted"]["remaining"],
                "target_cells": result["target_cells"],
                "audited_as": f"{dataset}:{slot}",
            }
    summary = {
        "level_dir": str(level_dir),
        "tag": args.tag,
        "trench_datasets": trench,
        "audited_datasets": audited,
        "slots": slots,
        "sibling_inputs_identical": not unequal,
        "unequal_slots": unequal,
        "contract": {k: report["contract"][k] for k in (
            "maps", "workers", "wall_seconds", "incomplete_fresh_cover_count",
            "missing_finite_metadata_count", "gate_semantics", "max_offset_m",
            "standoff_band_enforced", "on_line_clause")},
        "incomplete": sorted(k for k, v in verdict.items() if not v["complete"]),
        "verdict": verdict,
    }
    (level_dir / f"preflight_verdict_{args.tag}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "verdict"})[:2000])


if __name__ == "__main__":
    main()
