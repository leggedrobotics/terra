#!/usr/bin/env python3
"""Rank, transform and identity-filter new candidate maps per geometry level.

Admission follows the Train-96 builder that appended maps to the current bank
(``_candidate_slots`` / ``_select_additions``): whole complete pair slots
(every sibling condition on one shared dig), source-disjoint from every source
of the frozen release (any split) and across levels, levels served
least-supported first, and within a level the stable-hash order of the
pair-slot id with new dihedral shapes preferred. Every candidate then goes
through the exact transform chain (``common``) and is dropped when any final
identity (dig raster, source / parent source, reset-array scenario) collides
with an evaluation identity, a frozen-release identity or an already accepted
new slot. V7 candidates follow the V8 builder: generator order of one new-seed
batch, identity-filtered the same way.

Each accepted candidate is written, per condition, as an exact-layout dataset
under ``candidates/<level>/<condition>`` (slot = rank within the level), ready
for the trench preflight and the final assembly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

G = c.generator
CANDIDATES = c.ROOT / "candidates"


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def conditions_by_level() -> dict[str, list[tuple[int, Any]]]:
    result: dict[str, list[tuple[int, Any]]] = {}
    for index, condition in enumerate(G.DATASETS["main"].conditions):
        result.setdefault(condition.dig_bank_level, []).append((index, condition))
    return result


def load_frozen(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text())
    frozen = {f"release_{k}": set(v) for k, v in payload["release"].items()}
    frozen.update({f"eval_{k}": set(v) for k, v in payload["evaluation"].items()})
    return frozen


def generator_slots(level: str, siblings) -> list[dict[str, Any]]:
    """Complete pair slots with map_index >= 320 from the level's generator runs."""
    slots = []
    ids = {condition.id for _, condition in siblings}
    for shard in sorted((c.GENERATION / "v6").glob(f"{level.replace('@', '_at_')}_*")):
        summary = json.loads((shard / "shard_summary.json").read_text())
        if summary["level"] != level:
            continue
        for slot in summary["slots"]:
            if slot["status"] not in ("complete", "complete_reroll"):
                continue
            members = {}
            for condition_index, condition in siblings:
                sample = 10000 * condition_index + slot["map_index"]
                members[condition.id] = (shard, sample)
            slots.append({"map_index": slot["map_index"], "members": members,
                          "status": slot["status"]})
    if len(siblings) == 1:
        condition = siblings[0][1]
        root = c.GENERATION / "stock999" / condition.id
        if (root / "manifest.csv").is_file():
            with (root / "manifest.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["condition_id"] != condition.id or int(row["map_index"]) < 320:
                        continue
                    slots.append({"map_index": int(row["map_index"]),
                                  "members": {condition.id: (root, int(row["sample_index"]))},
                                  "status": "complete" if row["shared_dig"] == "1"
                                  else "complete_reroll"})
    by_index = {}
    for slot in slots:
        if slot["map_index"] in by_index:
            raise RuntimeError(f"{level}:{slot['map_index']} generated twice")
        by_index[slot["map_index"]] = slot
    for slot in by_index.values():
        records = {}
        for condition_id, (root, sample) in slot["members"].items():
            path = root / "review_metadata" / f"img_{sample}.json"
            record = json.loads(path.read_text())
            if record["condition_id"] != condition_id:
                raise RuntimeError(f"{path}: condition mismatch")
            records[condition_id] = (record, path)
        sources = {r["source_group_id"] for r, _ in records.values()}
        digs = {r["dig_sha256"] for r, _ in records.values()}
        if set(records) != ids or len(sources) != 1 or len(digs) != 1:
            raise RuntimeError(f"{level}:{slot['map_index']} is not a complete pair slot")
        slot["records"] = records
        slot["pair_slot_id"] = f"{level}:{slot['map_index']}"
        slot["source_id"] = sources.pop()
        slot["wide_dig_sha256"] = digs.pop()
        root, sample = next(iter(slot["members"].values()))
        dig = np.load(root / "dataset" / "images" / f"img_{sample}.npy") < 0
        slot["dihedral_id"] = G._dihedral_normalized_identity(dig)
    return [by_index[k] for k in sorted(by_index)]


def existing_dihedral(level: str, siblings, old) -> set[str]:
    """Dihedral ids of the level's representative condition in the current bank."""
    representative = sorted(condition.id for _, condition in siblings)[0]
    rows = [r for r in old["rows"] if r["primary_cell"] == representative]
    result = set()
    for row in rows:
        index = int(row["map_id"].rsplit("-", 1)[1])
        dig = np.load(c.P5_POOL / "dataset" / "images" / f"img_{index}.npy") < 0
        result.add(G._dihedral_normalized_identity(dig))
    return result


def rank_slots(slots, frozen_sources, used_sources, dihedral) -> list[dict[str, Any]]:
    """_select_additions without a count: the full admission order."""
    available = [s for s in slots if s["source_id"] not in frozen_sources
                 and s["source_id"] not in used_sources]
    ranked = []
    taken_sources = set(used_sources)
    seen = set(dihedral)
    remaining = list(available)
    while True:
        remaining = [s for s in remaining if s["source_id"] not in taken_sources]
        if not remaining:
            break
        unseen = [s for s in remaining if s["dihedral_id"] not in seen]
        choice = min(unseen or remaining,
                     key=lambda s: (stable_hash(s["pair_slot_id"]), s["pair_slot_id"]))
        remaining.remove(choice)
        ranked.append(choice)
        taken_sources.add(choice["source_id"])
        seen.add(choice["dihedral_id"])
    return ranked


def prepare_dataset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    for folder in (*c.RESET_ARRAY_FOLDERS, "metadata"):
        (path / folder).mkdir(parents=True)


def identity_conflict(row, arrays, frozen, accepted) -> str:
    digest = c.dig_sha(arrays["images"])
    scenario = row["scenario_id"]
    for value, fields, reason in (
        (row["source_id"], ("eval_source_id", "eval_parent_source_id"), "eval_source"),
        (row.get("parent_source_id"), ("eval_source_id", "eval_parent_source_id"), "eval_parent_source"),
        (digest, ("eval_array_dig_sha256", "eval_dig_sha256"), "eval_dig"),
        (scenario, ("eval_array_scenario_sha256", "eval_scenario_id", "eval_parent_scenario_id"), "eval_scenario"),
        (row["map_id"], ("eval_map_id", "release_map_id"), "map_id"),
        (row["source_id"], ("release_source_id",), "release_source"),
        (row.get("parent_source_id"), ("release_source_id",), "release_parent_source"),
        (scenario, ("release_scenario_id",), "release_scenario"),
    ):
        if value and any(value in frozen[field] for field in fields):
            return reason
    if digest in accepted["dig"]:
        return "new_duplicate_dig"
    if scenario in accepted["scenario"]:
        return "new_duplicate_scenario"
    if row["source_id"] in accepted["source"]:
        return "new_duplicate_source"
    return ""


def stage_v6_level(level, siblings, ranked, frozen, accepted, old_digs, limit, report):
    level_dir = CANDIDATES / level.replace("@", "_at_")
    condition_ids = [condition.id for _, condition in siblings]
    controls = [cid for cid, parent in c.CONTROL_PARENTS.items() if parent in condition_ids]
    outputs = condition_ids + controls
    for condition_id in outputs:
        prepare_dataset(level_dir / condition_id)
    rows = {cid: [] for cid in outputs}
    rejected = Counter()
    rank = 0
    for slot in ranked:
        if rank >= limit:
            break
        produced = {}
        for condition_index, condition in siblings:
            record, record_path = slot["records"][condition.id]
            root, sample = slot["members"][condition.id]
            arrays = c.load_arrays(root / "dataset", sample)
            metadata = json.loads((root / "dataset" / "metadata" / f"trench_{sample}.json").read_text())
            produced[condition.id] = c.v6_constrained_map(
                condition_id=condition.id, family=condition.family, arrays=arrays,
                metadata=metadata, record=record, record_path=record_path, sample_index=sample)
            for control_id in controls:
                if c.CONTROL_PARENTS[control_id] == condition.id:
                    produced[control_id] = c.control_map(
                        control_id=control_id, parent_arrays=arrays,
                        parent_metadata=metadata, parent_record=record,
                        parent_record_path=record_path)
        reason = ""
        slot_digs, slot_scenarios, slot_sources = set(), set(), set()
        for condition_id, (arrays, metadata, row) in produced.items():
            reason = identity_conflict(row, arrays, frozen, accepted)
            if not reason and c.dig_sha(arrays["images"]) in old_digs:
                reason = "old_train_dig"
            if reason:
                reason = f"{reason}:{condition_id}"
                break
            slot_digs.add(c.dig_sha(arrays["images"]))
            slot_scenarios.add(row["scenario_id"])
            slot_sources.add(row["source_id"])
        if not reason and len(slot_scenarios) != len(produced):
            reason = "sibling_scenario_collision"
        if reason:
            rejected[reason.split(":")[0]] += 1
            report["rejections"].append({"pair_slot_id": slot["pair_slot_id"], "reason": reason})
            continue
        rank += 1
        accepted["dig"].update(slot_digs)
        accepted["scenario"].update(slot_scenarios)
        accepted["source"].update(slot_sources)
        for condition_id, (arrays, metadata, row) in produced.items():
            row = {**row, "slot_index": rank,
                   "train_v3_origin": "new",
                   "generator_level": level,
                   "generator_map_index": slot["map_index"],
                   "generator_status": slot["status"],
                   "generator_run": str(slot["members"][
                       condition_id if condition_id in slot["members"]
                       else c.CONTROL_PARENTS[condition_id]][0].relative_to(c.GENERATION))}
            c.save_map(level_dir / condition_id, rank, arrays, metadata)
            rows[condition_id].append(row)
    for condition_id, condition_rows in rows.items():
        dataset = level_dir / condition_id
        (dataset / "manifest.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in condition_rows))
        (dataset / "dataset.json").write_text(json.dumps(
            {"schema": "terra_bigbank_v3_candidates_v1", "level": level,
             "condition_id": condition_id, "slot_count": len(condition_rows)}, indent=2) + "\n")
    report["levels"][level] = {
        "generated_complete_slots": report["levels"].get(level, {}).get("generated_complete_slots"),
        "ranked_slots": len(ranked),
        "staged_slots": rank,
        "rejected": dict(rejected),
        "conditions": outputs,
    }
    return rank


def stage_v7(frozen, accepted, count, seed, limit, report):
    level_dir = CANDIDATES / "v7"
    rows: dict[str, list] = {}
    rejected = Counter()
    ranks: Counter = Counter()
    for condition, k, arrays, metadata, row in c.v7_maps(count, seed):
        cid = condition.condition_id
        if cid not in rows:
            rows[cid] = []
            prepare_dataset(level_dir / cid)
        if ranks[cid] >= limit:
            continue
        row = {**row, "map_id": f"v8x:train:{cid}:s{seed}:{k:04d}"}
        reason = identity_conflict(row, arrays, frozen, accepted)
        if reason:
            rejected[reason] += 1
            report["rejections"].append({"map_id": row["map_id"], "reason": reason})
            continue
        ranks[cid] += 1
        slot = ranks[cid]
        accepted["dig"].add(c.dig_sha(arrays["images"]))
        accepted["scenario"].add(row["scenario_id"])
        accepted["source"].add(row["source_id"])
        c.save_map(level_dir / cid, slot, arrays, metadata)
        rows[cid].append({**row, "slot_index": slot, "train_v3_origin": "new",
                          "generator_seed": seed, "generator_scenario_index": k,
                          "generator_batch_size": count})
    for cid, condition_rows in rows.items():
        (level_dir / cid / "manifest.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in condition_rows))
        (level_dir / cid / "dataset.json").write_text(json.dumps(
            {"schema": "terra_bigbank_v3_candidates_v1", "level": "v7",
             "condition_id": cid, "slot_count": len(condition_rows)}, indent=2) + "\n")
    report["levels"]["v7"] = {"seed": seed, "batch_per_geometry": count,
                              "staged": dict(ranks), "rejected": dict(rejected)}


def old_train_digs(old) -> set[str]:
    return {c.dig_sha(np.load(c.OLD_POOL / "images" / f"img_{r['slot_index']}.npy"))
            for r in old["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="")
    parser.add_argument("--v7", action="store_true")
    parser.add_argument("--v7-seed", type=int, default=2026092302)
    parser.add_argument("--v7-count", type=int, default=424)
    parser.add_argument("--foundation-limit", type=int, default=416)
    parser.add_argument("--trench-limit", type=int, default=468)
    parser.add_argument("--frozen", type=Path, default=c.ROOT / "receipts" / "frozen_identities.json")
    parser.add_argument("--accepted-state", type=Path, default=c.ROOT / "receipts" / "accepted_state.json")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    frozen = load_frozen(args.frozen)
    old = c.old_pool_index()
    old_digs = old_train_digs(old)
    # Identities accepted by other levels (per-level state keeps re-staging idempotent).
    state = json.loads(args.accepted_state.read_text()) if args.accepted_state.is_file() else {}
    levels = [level for level in args.levels.split(",") if level]
    staged_now = set(levels) | ({"v7"} if args.v7 else set())
    accepted = {k: set() for k in ("dig", "scenario", "source")}
    used_sources = set()
    for level, entry in state.items():
        if set(level.split(",")) & staged_now:
            continue
        for key in accepted:
            accepted[key].update(entry[key])
        used_sources.update(entry["used_sources"])
    before = {k: set(v) for k, v in accepted.items()}
    before_used = set(used_sources)
    report = {"levels": {}, "rejections": []}
    by_level = conditions_by_level()
    prepared = []
    for level in levels:
        siblings = by_level[level]
        slots = generator_slots(level, siblings)
        prepared.append((level, siblings, slots))
    frozen_sources = frozen["release_source_id"] | frozen["eval_source_id"] | frozen["eval_parent_source_id"]
    prepared.sort(key=lambda item: (
        len([s for s in item[2] if s["source_id"] not in frozen_sources]),
        stable_hash(item[0]), item[0]))
    for level, siblings, slots in prepared:
        ranked = rank_slots(slots, frozen_sources, used_sources,
                            existing_dihedral(level, siblings, old))
        report["levels"][level] = {"generated_complete_slots": len(slots)}
        family = siblings[0][1].family
        limit = args.trench_limit if family == "trench" else args.foundation_limit
        staged = stage_v6_level(level, siblings, ranked, frozen, accepted, old_digs, limit, report)
        # generator source ids of staged slots block every later level (cross-level rule)
        rows = c.read_jsonl(CANDIDATES / level.replace("@", "_at_") / siblings[0][1].id / "manifest.jsonl")
        staged_pairs = {r["pair_slot_id"] for r in rows}
        used_sources.update(s["source_id"] for s in ranked if s["pair_slot_id"] in staged_pairs)
        print(f"{level}: complete={len(slots)} ranked={len(ranked)} staged={staged}", flush=True)
    if args.v7:
        stage_v7(frozen, accepted, args.v7_count, args.v7_seed, args.foundation_limit, report)
        print("v7:", report["levels"]["v7"], flush=True)
    # attribute the identities added in this run to the staged levels
    for level in [key for key in state if set(key.split(",")) & staged_now]:
        state.pop(level)
    added = {k: sorted(accepted[k] - before[k]) for k in accepted}
    state[",".join(sorted(staged_now))] = added | {"used_sources": sorted(used_sources - before_used)}
    args.accepted_state.write_text(json.dumps(state) + "\n")
    report["wall_seconds"] = round(time.time() - started, 1)
    args.report.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
