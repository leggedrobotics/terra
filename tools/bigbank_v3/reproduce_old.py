#!/usr/bin/env python3
"""Re-derive existing generalist-bank maps from generator outputs and byte-compare.

V6 constrained / control maps start from the P5 candidate pool (the generator
output that ``v6_extension.py`` reproduces byte-for-byte); V7 maps are
regenerated from ``generate_scenarios(96, 2026080801)``. Every compared map must
match the current pooled bank in all five reset arrays, the metadata sidecar and
the manifest row (pooling fields excepted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

POOL_FIELDS = ("slot_index", "pooled_from_dataset", "pooled_from_slot_index")


def p5_map(map_id: str):
    index = int(map_id.rsplit("-", 1)[1])
    arrays = c.load_arrays(c.P5_POOL / "dataset", index)
    metadata = json.loads((c.P5_POOL / "dataset" / "metadata" / f"trench_{index}.json").read_text())
    record_path = c.P5_POOL / "review_metadata" / f"img_{index}.json"
    return index, arrays, metadata, json.loads(record_path.read_text()), record_path


def compare(label, arrays, metadata, row, old_row, report):
    old_arrays = c.load_arrays(c.OLD_POOL, old_row["slot_index"])
    old_metadata = json.loads((c.OLD_POOL / "metadata" / f"trench_{old_row['slot_index']}.json").read_text())
    expected = {k: v for k, v in old_row.items() if k not in POOL_FIELDS}
    produced = {k: v for k, v in row.items() if k not in POOL_FIELDS}
    ok_arrays = c.arrays_equal(arrays, old_arrays)
    ok_metadata = metadata == old_metadata
    ok_row = produced == expected
    report["maps"] += 1
    report["arrays_identical"] += int(ok_arrays)
    report["metadata_identical"] += int(ok_metadata)
    report["rows_identical"] += int(ok_row)
    if not (ok_arrays and ok_metadata and ok_row):
        diff = sorted(k for k in set(produced) | set(expected) if produced.get(k) != expected.get(k))
        mdiff = sorted(k for k in set(metadata) | set(old_metadata)
                       if metadata.get(k) != old_metadata.get(k))
        report["failures"].append({"map": label, "arrays": ok_arrays, "row_keys": diff,
                                   "metadata_keys": mdiff})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-condition", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = c.old_pool_index()
    report = {"maps": 0, "arrays_identical": 0, "metadata_identical": 0, "rows_identical": 0,
              "failures": [], "by_family": {}}
    for directory in index["order"]:
        rows = index["by_dir"][directory]
        condition = rows[0]["primary_cell"]
        if condition.startswith("v7-"):
            continue
        n = len(rows)
        picks = sorted({0, 63, 64, n - 1, *range(0, n, max(1, n // args.per_condition))})
        for position in picks:
            old = rows[position]
            if condition in c.CONTROL_PARENTS:
                _, arrays, metadata, record, record_path = p5_map(old["parent_map_id"])
                out = c.control_map(control_id=condition, parent_arrays=arrays,
                                    parent_metadata=metadata, parent_record=record,
                                    parent_record_path=record_path)
            else:
                sample, arrays, metadata, record, record_path = p5_map(old["map_id"])
                out = c.v6_constrained_map(condition_id=condition, family=old["family"],
                                           arrays=arrays, metadata=metadata, record=record,
                                           record_path=record_path, sample_index=sample)
            compare(f"{directory}:{old['pooled_from_slot_index']}", *out, old, report)
    v7_rows = {}
    for directory in index["order"]:
        rows = index["by_dir"][directory]
        if rows[0]["primary_cell"].startswith("v7-"):
            v7_rows[rows[0]["primary_cell"]] = rows
    for condition, k, arrays, metadata, row in c.v7_maps(96, c.V7_TRAIN_SEED):
        compare(f"{condition.condition_id}:{k}", arrays, metadata, row,
                v7_rows[condition.condition_id][k], report)
    report["passed"] = not report["failures"]
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}),
          json.dumps(report["failures"][:5]))


if __name__ == "__main__":
    main()
