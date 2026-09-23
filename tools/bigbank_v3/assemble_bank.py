#!/usr/bin/env python3
"""Assemble train_v3_generalist_512 from the old pool and the staged candidates.

Per condition: the current bank's 96 maps verbatim (original order), then the
first admissible new slots of its level in admission order (trench levels:
preflight-complete slots only), up to 416. The result is written as an
enriched-bank-shaped staging tree (``train/NNN__condition`` with the old
directory names, a source registry = the frozen V8 registry + the new train
rows), pooled with the unmodified ``tools/build_trench_pilot_pooled_train.py``
(``--conditions generalist``), and published under ``bank/`` with real files:
the pooled array symlinks are replaced by hard links to the staging copies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

CANDIDATES = c.ROOT / "candidates"
STAGING = c.ROOT / "staging"
BANK = c.ROOT / "bank"
POOLED_NAME = "train_v3_generalist_512"
POOL_TOOL = c.pinned.TERRA_ROOT / "tools" / "build_trench_pilot_pooled_train.py"
NEW_PER_CONDITION = 416
POOL_FIELDS = ("pooled_from_dataset", "pooled_from_slot_index")


def candidate_dirs() -> dict[str, Path]:
    result = {}
    for level_dir in sorted(p for p in CANDIDATES.iterdir() if p.is_dir()):
        for dataset in sorted(p for p in level_dir.iterdir() if (p / "manifest.jsonl").is_file()):
            if dataset.name in result:
                raise RuntimeError(f"{dataset.name} staged twice")
            result[dataset.name] = dataset
    return result


def excluded_pairs(level_dir: Path) -> tuple[set[str], dict]:
    verdict_path = level_dir / "preflight_verdict_yawonly.json"
    default_path = level_dir / "preflight_verdict_default.json"
    rows = None
    for dataset in level_dir.iterdir():
        if (dataset / "manifest.jsonl").is_file():
            rows = c.read_jsonl(dataset / "manifest.jsonl")
            break
    if not rows or rows[0]["family"] != "trench":
        return set(), {"preflight": "not_applicable"}
    if not verdict_path.is_file():
        raise RuntimeError(f"{level_dir}: trench level without preflight verdict")
    verdict = json.loads(verdict_path.read_text())
    audited = verdict["verdict"]
    excluded = set()
    by_dataset = {}
    for key, value in audited.items():
        name, slot = key.rsplit(":", 1)
        by_dataset.setdefault(name, {})[int(slot)] = value
    for name, slots in by_dataset.items():
        dataset_rows = c.read_jsonl(level_dir / name / "manifest.jsonl")
        for row in dataset_rows:
            value = slots.get(row["slot_index"])
            if value is not None and not value["complete"]:
                excluded.add(row["pair_slot_id"])
    default = json.loads(default_path.read_text()) if default_path.is_file() else None
    return excluded, {"preflight_slots": verdict["slots"],
                      "exclusion_contract": "v2 yaw-parallel (--max-offset-m 0)",
                      "incomplete_pair_slots": sorted(excluded),
                      "audited": verdict["audited_datasets"],
                      "sibling_inputs_identical": verdict["sibling_inputs_identical"],
                      "default_2122b2df_incomplete_maps": (
                          default["contract"]["incomplete_fresh_cover_count"] if default else None),
                      "default_2122b2df_audited_maps": (
                          default["contract"]["maps"] if default else None)}


def link_or_copy(source: Path, destination: Path, hardlink: bool) -> None:
    if hardlink:
        os.link(source, destination)
    else:
        shutil.copyfile(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    if STAGING.exists() or (BANK / POOLED_NAME).exists():
        raise SystemExit("staging or bank output already exists")
    old = c.old_pool_index()
    candidates = candidate_dirs()
    registry_lines = (c.OLD_ROOT / "source_registry.jsonl").read_text().splitlines(keepends=True)
    new_registry_rows = []
    counts = {}
    level_info = {}
    level_selected: dict[Path, list[str]] = {}

    (STAGING / "train").mkdir(parents=True)
    for directory in old["order"]:
        name = directory.split("/", 1)[1]
        condition = name.split("__", 1)[1]
        old_rows = old["by_dir"][directory]
        dataset = STAGING / "train" / name
        for folder in (*c.RESET_ARRAY_FOLDERS, "metadata"):
            (dataset / folder).mkdir(parents=True)
        rows = []
        for position, row in enumerate(old_rows, start=1):
            if row["pooled_from_slot_index"] != position:
                raise RuntimeError(f"{directory}: old slots out of order")
            for folder in c.RESET_ARRAY_FOLDERS:
                shutil.copyfile(c.OLD_POOL / folder / f"img_{row['slot_index']}.npy",
                                dataset / folder / f"img_{position}.npy")
            shutil.copyfile(c.OLD_POOL / "metadata" / f"trench_{row['slot_index']}.json",
                            dataset / "metadata" / f"trench_{position}.json")
            kept = {k: v for k, v in row.items() if k not in POOL_FIELDS}
            rows.append({**kept, "slot_index": position, "train_v3_origin": "old"})

        source = candidates.get(condition)
        new_rows = []
        if source is not None:
            level_dir = source.parent
            if level_dir not in level_selected:
                excluded, info = excluded_pairs(level_dir)
                level_selected[level_dir] = excluded
                level_info[level_dir.name] = info
            excluded = level_selected[level_dir]
            staged_rows = c.read_jsonl(source / "manifest.jsonl")
            chosen_rows = [r for r in staged_rows
                           if r["pair_slot_id"] not in excluded][:NEW_PER_CONDITION]
            info = level_info[level_dir.name]
            info.setdefault("selected", {})[condition] = {
                "staged": len(staged_rows), "selected": len(chosen_rows)}
            if level_dir.name != "v7":
                pairs = [r["pair_slot_id"] for r in chosen_rows]
                if info.setdefault("chosen_pairs", pairs) != pairs:
                    raise RuntimeError(f"{condition}: siblings disagree on the chosen pair slots")
            for offset, row in enumerate(chosen_rows, start=len(rows) + 1):
                for folder in c.RESET_ARRAY_FOLDERS:
                    link_or_copy(source / folder / f"img_{row['slot_index']}.npy",
                                 dataset / folder / f"img_{offset}.npy", hardlink=True)
                shutil.copyfile(source / "metadata" / f"trench_{row['slot_index']}.json",
                                dataset / "metadata" / f"trench_{offset}.json")
                new_rows.append({**row, "slot_index": offset, "candidate_rank": row["slot_index"]})
        rows.extend(new_rows)
        for row in new_rows:
            new_registry_rows.append({k: row[k] for k in (
                "family", "map_id", "primary_cell", "scenario_id", "source_id", "split")})
        (dataset / "manifest.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        counts[condition] = {"old": len(old_rows), "new": len(new_rows), "total": len(rows)}

    registry = STAGING / "source_registry.jsonl"
    new_registry_rows.sort(key=lambda r: (r["primary_cell"], r["map_id"]))
    with registry.open("w") as handle:
        handle.writelines(registry_lines)
        handle.writelines(json.dumps(r, sort_keys=True) + "\n" for r in new_registry_rows)
    registry_sha = c.sha256_file(registry)

    for directory in old["order"]:
        name = directory.split("/", 1)[1]
        dataset = STAGING / "train" / name
        rows = c.read_jsonl(dataset / "manifest.jsonl")
        descriptor = json.loads((c.OLD_ROOT / "train" / name / "dataset.json").read_text())
        enrichment = dict(descriptor["trench_finite_enrichment"])
        enriched = sum(1 for r in rows if r["family"] == "trench")
        enrichment.update({"enriched_slots": enriched,
                           "source_bank": str(STAGING),
                           "source_dataset": f"train/{name}"})
        descriptor.update({"slot_count": len(rows), "unique_identity_count": len(
            {r["map_id"] for r in rows}), "source_registry": "../../source_registry.jsonl",
            "source_registry_sha256": registry_sha, "trench_finite_enrichment": enrichment})
        (dataset / "dataset.json").write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")

    subprocess.run([sys.executable, str(POOL_TOOL), "--bank-root", str(STAGING),
                    "--pooled-name", POOLED_NAME, "--conditions", "generalist",
                    "--manifest-out", str(STAGING / f"{POOLED_NAME}_manifest.json")],
                   check=True)

    pooled = STAGING / POOLED_NAME
    replaced = 0
    for folder in c.RESET_ARRAY_FOLDERS:
        for path in (pooled / folder).iterdir():
            if path.is_symlink():
                target = path.resolve(strict=True)
                path.unlink()
                os.link(target, path)
                replaced += 1
    BANK.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(registry, BANK / "source_registry.jsonl")
    shutil.move(str(pooled), str(BANK / POOLED_NAME))
    shutil.copyfile(STAGING / f"{POOLED_NAME}_manifest.json", BANK / f"{POOLED_NAME}_pooling_manifest.json")
    for info in level_info.values():
        info.pop("chosen_pairs", None)
    report = {
        "counts": counts,
        "levels": level_info,
        "registry_sha256": registry_sha,
        "registry_rows_old": len(registry_lines),
        "registry_rows_new": len(new_registry_rows),
        "symlinks_replaced_by_hardlinks": replaced,
        "wall_seconds": round(time.time() - started, 1),
    }
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "counts"})[:3000])
    print(json.dumps(Counter(v["total"] for v in counts.values())))


if __name__ == "__main__":
    main()
