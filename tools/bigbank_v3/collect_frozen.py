#!/usr/bin/env python3
"""Collect every identity the new maps must avoid.

* ``release``: the identities of the current release (all splits of the V8/R2
  registry, the V6 Train-96 registries, the P5 accepted-bank registries). New
  pair slots must not reuse any of these sources (the Train-96 "frozen source"
  rule), whatever the split.
* ``evaluation``: every non-train row of every distinct ``manifest.jsonl`` under
  ``.artifacts`` (158 distinct evaluation manifests: V8 main / gate_main /
  capability panels, TTC known-geometry panels, foundation-study panels, ...),
  plus identities recomputed from their arrays (dig raster sha256 and the
  five-array reset identity), and every non-train registry row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

REGISTRIES = [
    c.OLD_ROOT / "source_registry.jsonl",
    c.ARTIFACTS / "terra_v6main_capfloor34_train96_v1_20260803_a14d8302" / "source_registry.jsonl",
    c.ARTIFACTS / "terra_v6main_capfloor34_train96_v1_20260803_a14d8302" / "evaluation" / "constrained" / "source_registry.jsonl",
    c.ARTIFACTS / "terra_v6main_capfloor34_train96_v1_20260803_a14d8302" / "evaluation" / "capability_floor" / "source_registry.jsonl",
    c.ARTIFACTS / "terra_p5_accepted_bank_20260801_a6e6e5bc_prng_v102" / "source_registry.jsonl",
    c.ARTIFACTS / "terra_p5_accepted_bank_20260801_785c2b6d" / "source_registry.jsonl",
    c.ARTIFACTS / "terra_unconstrained_controls_20260802_0306c3cd" / "source_registry.jsonl",
    c.ARTIFACTS / "terra_test_time_compute_20260921" / "adaptation" / "maps" / "source_registry.jsonl",
]
ID_FIELDS = ("source_id", "parent_source_id", "scenario_id", "parent_scenario_id",
             "dig_sha256", "map_id", "source_scenario_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    release = {field: set() for field in ("source_id", "scenario_id", "map_id")}
    evaluation = {field: set() for field in ID_FIELDS}
    evaluation["array_dig_sha256"] = set()
    evaluation["array_scenario_sha256"] = set()
    registry_counts = {}
    for path in REGISTRIES:
        rows = c.read_jsonl(path)
        registry_counts[str(path)] = dict(Counter(r["split"] for r in rows))
        for row in rows:
            for field in release:
                if row.get(field):
                    release[field].add(row[field])
            if row["split"] != "train":
                for field in ("source_id", "scenario_id", "map_id"):
                    if row.get(field):
                        evaluation[field].add(row[field])

    seen = set()
    manifests = []
    for line in args.manifest_list.read_text().splitlines():
        path = (c.ARTIFACTS / line.strip()).resolve() if not line.startswith("/") else Path(line)
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        manifests.append(path)
    eval_manifests = []
    rows_total = 0
    arrays_total = 0
    for path in manifests:
        try:
            rows = c.read_jsonl(path)
        except json.JSONDecodeError:
            continue
        eval_rows = [r for r in rows if isinstance(r, dict) and r.get("split", "train") != "train"]
        if not eval_rows:
            continue
        eval_manifests.append({"path": str(path), "rows": len(eval_rows),
                               "splits": dict(Counter(r["split"] for r in eval_rows))})
        directory = path.parent
        for row in eval_rows:
            rows_total += 1
            for field in ID_FIELDS:
                value = row.get(field)
                if isinstance(value, str) and value:
                    evaluation[field].add(value)
            slot = row.get("slot_index")
            if slot is None or not (directory / "images" / f"img_{slot}.npy").exists():
                continue
            try:
                arrays = c.load_arrays(directory, int(slot))
            except (FileNotFoundError, ValueError):
                arrays = None
            if arrays is None:
                continue
            arrays_total += 1
            evaluation["array_dig_sha256"].add(c.dig_sha(arrays["images"]))
            evaluation["array_scenario_sha256"].add(c.reset_array_scenario_sha256(arrays))
    payload = {
        "registries": registry_counts,
        "evaluation_manifests": eval_manifests,
        "evaluation_rows": rows_total,
        "evaluation_rows_with_arrays": arrays_total,
        "release": {k: sorted(v) for k, v in release.items()},
        "evaluation": {k: sorted(v) for k, v in evaluation.items()},
    }
    args.output.write_text(json.dumps(payload) + "\n")
    print(json.dumps({"eval_manifests": len(eval_manifests), "eval_rows": rows_total,
                      "eval_rows_with_arrays": arrays_total,
                      "release": {k: len(v) for k, v in release.items()},
                      "evaluation": {k: len(v) for k, v in evaluation.items()}}))


if __name__ == "__main__":
    main()
