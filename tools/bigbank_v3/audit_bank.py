#!/usr/bin/env python3
"""Disjointness, duplicate and distribution-parity audit of train_v3_generalist_512."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

BANK = c.ROOT / "bank"
POOLED = BANK / "train_v3_generalist_512"
TREATMENT = c.ARTIFACTS / "terra_v8_r2_training_inputs_20260810" / "treatment_bank"
TTC = c.ARTIFACTS / "terra_test_time_compute_20260921" / "adaptation" / "maps"
NAMED_EVAL = {
    **{f"v8_main/{s}": TREATMENT / "evaluation" / "main" / s
       for s in ("promotion", "development", "sealed")},
    **{f"gate_main/{s}": c.OLD_ROOT / "evaluation" / "gate_main" / s
       for s in ("promotion", "development", "sealed")},
    **{f"capability_floor/{s}": c.OLD_ROOT / "evaluation" / "capability_floor" / s
       for s in ("promotion", "development", "sealed")},
    **{f"ttc_known_geometry/{case}": TTC / case / "eval"
       for case in ("trn-straight-side1", "trn-tee-side2", "trn-net4-side1-road")},
}
FOUR = ndi.generate_binary_structure(2, 1)
EIGHT = ndi.generate_binary_structure(2, 2)


def slot_identity(dataset: Path, slot: int):
    arrays = c.load_arrays(dataset, slot)
    return arrays, c.dig_sha(arrays["images"]), c.reset_array_scenario_sha256(arrays)


def stats(arrays, metadata) -> dict[str, float]:
    target = np.asarray(arrays["images"])
    occupancy = np.asarray(arrays["occupancy"], dtype=bool)
    dump = np.asarray(arrays["dumpability"], dtype=bool)
    distance = np.asarray(arrays["distance"], dtype=np.float64)
    dig = target < 0
    accepted = target > 0
    _, dig_components = ndi.label(dig, structure=FOUR)
    _, obstacles = ndi.label(occupancy, structure=EIGHT)
    return {
        "dig_cells": float(dig.sum()),
        "dig_components": float(dig_components),
        "dump_to_dig_ratio": float(accepted.sum() / max(1, dig.sum())),
        "distance_mean_dig": float(distance[dig].mean()) if dig.any() else 0.0,
        "distance_max_traversable": float(distance[~occupancy].max()),
        "trench_sections": float(len(metadata.get("trench_segments_yx") or [])),
        "obstacle_count": float(obstacles),
        "protected_cells": float((~dump & ~occupancy & (target >= 0)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, default=c.ROOT / "receipts" / "frozen_identities.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text())["evaluation"]
    eval_sets = {
        "dig": set(frozen["array_dig_sha256"]) | set(frozen["dig_sha256"]),
        "source": set(frozen["source_id"]) | set(frozen["parent_source_id"]),
        "scenario": set(frozen["scenario_id"]) | set(frozen["array_scenario_sha256"])
        | set(frozen["parent_scenario_id"]),
        "reset_arrays": set(frozen["array_scenario_sha256"]),
        "map_id": set(frozen["map_id"]),
    }
    rows = c.read_jsonl(POOLED / "manifest.jsonl")
    identities = []
    per_condition = defaultdict(lambda: {"old": defaultdict(list), "new": defaultdict(list)})
    mismatched_scenario = 0
    for row in rows:
        arrays, dig, reset = slot_identity(POOLED, row["slot_index"])
        metadata = json.loads((POOLED / "metadata" / f"trench_{row['slot_index']}.json").read_text())
        if reset != row["scenario_id"]:
            mismatched_scenario += 1
        identities.append((row, dig, reset))
        for key, value in stats(arrays, metadata).items():
            per_condition[row["primary_cell"]][row["train_v3_origin"]][key].append(value)

    overlap = {"old": Counter(), "new": Counter()}
    for row, dig, reset in identities:
        part = overlap[row["train_v3_origin"]]
        part["dig_sha256"] += dig in eval_sets["dig"]
        part["source_id"] += row["source_id"] in eval_sets["source"]
        part["parent_source_id"] += bool(row.get("parent_source_id")) and row["parent_source_id"] in eval_sets["source"]
        part["scenario_id"] += row["scenario_id"] in eval_sets["scenario"]
        part["reset_array_identity"] += reset in eval_sets["reset_arrays"]
        part["map_id"] += row["map_id"] in eval_sets["map_id"]

    named = {}
    new_rows = [(r, d, s) for r, d, s in identities if r["train_v3_origin"] == "new"]
    old_rows = [(r, d, s) for r, d, s in identities if r["train_v3_origin"] == "old"]
    for name, directory in NAMED_EVAL.items():
        eval_rows = c.read_jsonl(directory / "manifest.jsonl")
        digs, resets, sources, scenarios = set(), set(), set(), set()
        for row in eval_rows:
            _, dig, reset = slot_identity(directory, row["slot_index"])
            digs.add(dig)
            resets.add(reset)
            scenarios.add(row["scenario_id"])
            sources.add(row["source_id"])
            if row.get("parent_source_id"):
                sources.add(row["parent_source_id"])
        entry = {"slots": len(eval_rows)}
        for label, part in (("new", new_rows), ("old", old_rows)):
            entry[label] = {
                "dig_sha256": sum(d in digs for _, d, _ in part),
                "source_or_parent_source": sum(
                    r["source_id"] in sources or r.get("parent_source_id") in sources
                    for r, _, _ in part),
                "scenario_id": sum(r["scenario_id"] in scenarios for r, _, _ in part),
                "reset_array_identity": sum(s in resets for _, _, s in part),
            }
        named[name] = entry

    scenario_slots = defaultdict(list)
    dig_pairs = defaultdict(set)
    source_pairs = defaultdict(set)
    for row, dig, reset in identities:
        scenario_slots[reset].append(row["slot_index"])
        dig_pairs[dig].add((row["pair_slot_id"], row["train_v3_origin"]))
        source_pairs[row["source_id"]].add((row["pair_slot_id"], row["train_v3_origin"]))
    duplicate_scenarios = []
    for reset, slots in scenario_slots.items():
        if len(slots) > 1:
            origins = sorted({rows[s - 1]["train_v3_origin"] for s in slots})
            duplicate_scenarios.append({"slots": slots, "origins": origins,
                                        "conditions": sorted({rows[s - 1]["primary_cell"] for s in slots})})
    def shared(groups):
        result = {"old_only": 0, "involving_new": 0}
        for members in groups.values():
            pairs = {p for p, _ in members}
            if len(pairs) > 1:
                key = "involving_new" if any(o == "new" for _, o in members) else "old_only"
                result[key] += 1
        return result

    parity = {}
    flags = []
    for condition, parts in sorted(per_condition.items()):
        entry = {}
        for key in parts["old"]:
            old = np.asarray(parts["old"][key])
            new = np.asarray(parts["new"][key]) if parts["new"][key] else None
            item = {"old_median": float(np.median(old)), "old_p10": float(np.percentile(old, 10)),
                    "old_p90": float(np.percentile(old, 90))}
            if new is not None:
                item.update({"new_median": float(np.median(new)),
                             "new_p10": float(np.percentile(new, 10)),
                             "new_p90": float(np.percentile(new, 90))})
                base = abs(item["old_median"])
                delta = abs(item["new_median"] - item["old_median"])
                rel = delta / base if base > 0 else (0.0 if delta == 0 else float("inf"))
                item["median_rel_diff"] = rel
                if rel > 0.15:
                    flags.append({"condition": condition, "stat": key, **item})
            entry[key] = item
        entry["counts"] = {"old": len(parts["old"]["dig_cells"]), "new": len(parts["new"]["dig_cells"])}
        parity[condition] = entry

    report = {
        "slots": len(rows),
        "new_slots": len(new_rows),
        "old_slots": len(old_rows),
        "manifest_scenario_vs_arrays_mismatch": mismatched_scenario,
        "eval_universe_sizes": {k: len(v) for k, v in eval_sets.items()},
        "overlap_with_all_evaluation_identities": {k: dict(v) for k, v in overlap.items()},
        "named_evaluation_sets": named,
        "within_bank": {
            "unique_map_ids": len({r["map_id"] for r in rows}),
            "unique_scenario_ids": len({r["scenario_id"] for r in rows}),
            "duplicate_reset_array_groups": duplicate_scenarios,
            "dig_shared_by_distinct_pair_slots": shared(dig_pairs),
            "source_shared_by_distinct_pair_slots": shared(source_pairs),
        },
        "parity_flags_median_gt_15pct": flags,
        "parity": parity,
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("slots", "new_slots", "old_slots",
                                             "manifest_scenario_vs_arrays_mismatch",
                                             "overlap_with_all_evaluation_identities",
                                             "within_bank")}, indent=1)[:6000])
    print("parity flags:", json.dumps(flags, indent=0)[:4000])


if __name__ == "__main__":
    main()
