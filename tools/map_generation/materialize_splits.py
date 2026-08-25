#!/usr/bin/env python3
"""Materialize one generated curriculum bank into four source-disjoint splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "promotion", "development", "sealed")
ARRAY_FOLDERS = (
    "actions",
    "distance",
    "dumpability",
    "images",
    "occupancy",
    "trench_axis_owners",
)
METADATA_FOLDER = "metadata"
SCHEMA = "terra_curriculum_split_bank_v1"


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_group_id(row: dict[str, str]) -> str:
    """Keep every declared counterfactual slot in one split."""
    return row["pair_slot_id"]


def _read_rows(manifest_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {manifest_path}")
        fieldnames = list(reader.fieldnames)
        required = {
            "condition_id",
            "dig_sha256",
            "map_id",
            "pair_slot_id",
            "sample_index",
            "scenario_sha256",
            "source_group_id",
        }
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(missing)}")
        if "split" in fieldnames or "split_group_id" in fieldnames:
            raise ValueError("input manifest is already split-materialized")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest contains no scenarios")

    seen_sample_indices: dict[int, str] = {}
    for row in rows:
        for field in required:
            if not row[field].strip():
                raise ValueError(f"{row.get('map_id', '<unknown>')}: empty {field}")
        try:
            sample_index = int(row["sample_index"])
        except ValueError as exc:
            raise ValueError(
                f"{row['map_id']}: sample_index is not an integer"
            ) from exc
        if sample_index < 0:
            raise ValueError(f"{row['map_id']}: sample_index must be non-negative")
        previous = seen_sample_indices.get(sample_index)
        if previous is not None:
            raise ValueError(
                f"sample_index {sample_index} is shared by {previous} and "
                f"{row['map_id']}"
            )
        seen_sample_indices[sample_index] = row["map_id"]

        row["split_group_id"] = _atomic_group_id(row)

    return rows, fieldnames


def select_complete_pair_slots(
    rows: list[dict[str, str]], required_per_condition: int
) -> tuple[list[dict[str, str]], dict]:
    """Select exact pair slots without reusing a source across levels."""
    by_level: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        try:
            level, _ = row["pair_slot_id"].rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"{row['map_id']}: invalid pair_slot_id {row['pair_slot_id']!r}"
            ) from exc
        by_level[level][row["pair_slot_id"]].append(row)

    complete_by_level: dict[str, list[str]] = {}
    sources_by_slot: dict[str, set[str]] = {}
    incomplete_slots: list[str] = []
    for level, slots in sorted(by_level.items()):
        expected_conditions = {
            row["condition_id"] for slot_rows in slots.values() for row in slot_rows
        }
        complete = []
        for pair_slot, slot_rows in slots.items():
            conditions = {row["condition_id"] for row in slot_rows}
            dig_identities = {row["dig_sha256"] for row in slot_rows}
            if conditions != expected_conditions or len(dig_identities) != 1:
                incomplete_slots.append(pair_slot)
                continue
            complete.append(pair_slot)
            sources_by_slot[pair_slot] = {
                row["source_group_id"] for row in slot_rows
            }
        complete.sort(key=lambda value: (_stable_hash(value), value))
        if len(complete) < required_per_condition:
            raise RuntimeError(
                f"{level}: only {len(complete)} exact pair slots remain after "
                f"dropping rerolls; {required_per_condition} required. "
                "Generate a larger candidate bank."
            )
        complete_by_level[level] = complete

    # Give the least-supported level first choice. Once a source is selected,
    # no slot from another level may reuse it: otherwise post-hoc assignment
    # could leak that source across train/evaluation splits.
    selected_slots: set[str] = set()
    selected_sources: set[str] = set()
    source_conflict_slots: list[str] = []
    levels = sorted(
        complete_by_level,
        key=lambda level: (
            len(complete_by_level[level]),
            _stable_hash(level),
            level,
        ),
    )
    for level in levels:
        selected_for_level: list[str] = []
        conflicts_for_level: list[str] = []
        for pair_slot in complete_by_level[level]:
            if sources_by_slot[pair_slot] & selected_sources:
                conflicts_for_level.append(pair_slot)
                continue
            selected_for_level.append(pair_slot)
            selected_sources.update(sources_by_slot[pair_slot])
            if len(selected_for_level) == required_per_condition:
                break
        if len(selected_for_level) < required_per_condition:
            raise RuntimeError(
                f"{level}: only {len(selected_for_level)} source-disjoint exact "
                "pair slots remain after dropping rerolls and cross-level "
                f"source conflicts; {required_per_condition} required. "
                "Generate a larger candidate bank."
            )
        selected_slots.update(selected_for_level)
        source_conflict_slots.extend(conflicts_for_level)

    selected = [row for row in rows if row["pair_slot_id"] in selected_slots]
    return selected, {
        "input_pair_slots": sum(len(slots) for slots in by_level.values()),
        "selected_pair_slots": len(selected_slots),
        "incomplete_pair_slots": sorted(incomplete_slots),
        "source_conflict_pair_slots": sorted(source_conflict_slots),
    }


def _validate_arrays(rows: list[dict[str, str]], dataset_path: Path) -> None:
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"missing dataset directory: {dataset_path}")
    for folder in ARRAY_FOLDERS:
        array_folder = dataset_path / folder
        if not array_folder.is_dir():
            raise FileNotFoundError(f"missing array folder: {array_folder}")
    metadata_folder = dataset_path / METADATA_FOLDER
    if not metadata_folder.is_dir():
        raise FileNotFoundError(f"missing metadata folder: {metadata_folder}")
    for row in rows:
        sample_index = int(row["sample_index"])
        filename = f"img_{sample_index}.npy"
        for folder in ARRAY_FOLDERS:
            path = dataset_path / folder / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"{row['map_id']}: missing {folder} array: {path}"
                )
        metadata_path = metadata_folder / f"trench_{sample_index}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"{row['map_id']}: missing trench metadata: {metadata_path}"
            )


def _choose_shared_split(
    group_id: str,
    conditions: tuple[str, ...],
    remaining: dict[str, dict[str, int]],
    requested: dict[str, int],
) -> str:
    candidates = [
        split
        for split in SPLITS
        if all(remaining[condition][split] > 0 for condition in conditions)
    ]
    if not candidates:
        joined = ", ".join(conditions)
        raise RuntimeError(
            "cannot meet exact split counts without leaking a shared source "
            f"group: {group_id} spans [{joined}]. Regenerate a split-aware bank."
        )

    def score(split: str) -> tuple[float, float, str]:
        fractions = [
            remaining[condition][split] / requested[split]
            for condition in conditions
        ]
        return (
            -min(fractions),
            -sum(fractions),
            _stable_hash(f"{group_id}\0{split}"),
        )

    return min(candidates, key=score)


def assign_splits(
    rows: list[dict[str, str]], requested: dict[str, int]
) -> dict[str, str]:
    """Assign each declared pair slot to one split or fail without a solver."""
    if set(requested) != set(SPLITS):
        raise ValueError(f"split counts must name exactly: {', '.join(SPLITS)}")
    if any(not isinstance(count, int) or count < 0 for count in requested.values()):
        raise ValueError("split counts must be non-negative integers")
    requested_total = sum(requested.values())
    if requested_total == 0:
        raise ValueError("at least one split count must be positive")

    group_conditions: dict[str, set[str]] = defaultdict(set)
    condition_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_id = row["split_group_id"]
        condition = row["condition_id"]
        group_conditions[group_id].add(condition)
        condition_groups[condition].add(group_id)

    for condition, groups in sorted(condition_groups.items()):
        if len(groups) != requested_total:
            raise ValueError(
                f"{condition}: requested {requested_total} pair slots but "
                f"manifest contains {len(groups)}"
            )

    remaining = {
        condition: dict(requested) for condition in sorted(condition_groups)
    }
    assignments: dict[str, str] = {}
    shared_groups = [
        group_id
        for group_id, conditions in group_conditions.items()
        if len(conditions) > 1
    ]
    shared_groups.sort(
        key=lambda group_id: (
            -len(group_conditions[group_id]),
            _stable_hash(group_id),
            group_id,
        )
    )

    for group_id in shared_groups:
        conditions = tuple(sorted(group_conditions[group_id]))
        split = _choose_shared_split(
            group_id, conditions, remaining, requested
        )
        assignments[group_id] = split
        for condition in conditions:
            remaining[condition][split] -= 1

    for condition in sorted(condition_groups):
        private_groups = [
            group_id
            for group_id in condition_groups[condition]
            if len(group_conditions[group_id]) == 1
        ]
        private_groups.sort(key=lambda value: (_stable_hash(value), value))
        needed = sum(remaining[condition].values())
        if len(private_groups) != needed:
            raise RuntimeError(
                f"{condition}: exact post-hoc split needs {needed} private "
                f"groups but has {len(private_groups)}. Regenerate a split-aware bank."
            )
        offset = 0
        for split in SPLITS:
            count = remaining[condition][split]
            for group_id in private_groups[offset : offset + count]:
                assignments[group_id] = split
            offset += count
            remaining[condition][split] = 0

    if len(assignments) != len(group_conditions):
        raise RuntimeError("internal error: not every pair slot was assigned")
    if any(
        count
        for condition_counts in remaining.values()
        for count in condition_counts.values()
    ):
        raise RuntimeError("internal error: split counts were not fully assigned")
    return assignments


def _write_csv(
    path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_splits(
    manifest_path: Path,
    dataset_path: Path,
    output_path: Path,
    requested: dict[str, int],
) -> dict:
    rows, input_fieldnames = _read_rows(manifest_path)
    _validate_arrays(rows, dataset_path)
    input_scenarios = len(rows)
    rows, pair_audit = select_complete_pair_slots(rows, sum(requested.values()))
    assignments = assign_splits(rows, requested)

    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.mkdir(parents=True)
    output_fieldnames = [*input_fieldnames, "split_group_id", "split"]

    for row in rows:
        row["split"] = assignments[row["split_group_id"]]
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_splits[row["source_group_id"]].add(row["split"])
    leaked_sources = {
        source: sorted(splits)
        for source, splits in source_splits.items()
        if len(splits) > 1
    }
    if leaked_sources:
        first_source = sorted(leaked_sources)[0]
        raise RuntimeError(
            "pair-slot assignment would leak a realized source group across "
            f"splits: {first_source} -> {leaked_sources[first_source]}. "
            "Regenerate a split-aware bank."
        )
    rows.sort(
        key=lambda row: (
            row["condition_id"],
            int(row["sample_index"]),
            row["map_id"],
        )
    )
    _write_csv(output_path / "manifest.csv", rows, output_fieldnames)

    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        split_root = output_path / split
        split_root.mkdir()
        _write_csv(split_root / "manifest.csv", split_rows, output_fieldnames)
        for folder in ARRAY_FOLDERS:
            (split_root / "dataset" / folder).mkdir(parents=True)
        (split_root / "dataset" / METADATA_FOLDER).mkdir(parents=True)
        for row in split_rows:
            sample_index = int(row["sample_index"])
            filename = f"img_{sample_index}.npy"
            for folder in ARRAY_FOLDERS:
                shutil.copy2(
                    dataset_path / folder / filename,
                    split_root / "dataset" / folder / filename,
                )
            shutil.copy2(
                dataset_path
                / METADATA_FOLDER
                / f"trench_{sample_index}.json",
                split_root
                / "dataset"
                / METADATA_FOLDER
                / f"trench_{sample_index}.json",
            )

    conditions: dict[str, dict[str, dict[str, int]]] = {}
    for condition in sorted({row["condition_id"] for row in rows}):
        condition_rows = [row for row in rows if row["condition_id"] == condition]
        conditions[condition] = {}
        for split in SPLITS:
            split_rows = [row for row in condition_rows if row["split"] == split]
            conditions[condition][split] = {
                "pair_slots": len({row["split_group_id"] for row in split_rows}),
                "source_groups": len(
                    {row["source_group_id"] for row in split_rows}
                ),
                "scenarios": len(split_rows),
            }
            if conditions[condition][split]["pair_slots"] != requested[split]:
                raise RuntimeError(
                    f"{condition}/{split}: materialized pair-slot count "
                    "does not match the request"
                )

    assignment_lines = [
        f"{group_id},{assignments[group_id]}\n"
        for group_id in sorted(assignments)
    ]
    summary = {
        "schema": SCHEMA,
        "requested_pair_slots_per_condition": requested,
        "conditions": conditions,
        "pair_slots": len(assignments),
        "realized_source_groups": len(source_splits),
        "input_scenarios": input_scenarios,
        "scenarios": len(rows),
        "pair_integrity": pair_audit,
        "assignment_sha256": hashlib.sha256(
            "".join(assignment_lines).encode("utf-8")
        ).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a generated Terra curriculum bank into exact, "
            "source-disjoint splits."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for split in SPLITS:
        parser.add_argument(f"--{split}", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = {split: getattr(args, split) for split in SPLITS}
    summary = materialize_splits(
        args.manifest, args.dataset, args.output, requested
    )
    print(
        f"materialized {summary['scenarios']} scenarios across "
        f"{len(summary['conditions'])} conditions"
    )


if __name__ == "__main__":
    main()
