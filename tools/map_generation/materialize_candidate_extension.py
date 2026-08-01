#!/usr/bin/env python3
"""Extend a complete curriculum candidate bank with whole-level shards.

The generator's ``--only`` mode always emits a deterministic prefix from map
index zero.  This materializer verifies that prefix against a complete base
bank, then carries only the higher map indices into a new candidate bank.  It
does not resume generation and never mutates an input bank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.map_generation import generate_curriculum_bank as generator

SCHEMA = "terra_curriculum_candidate_extension_v1"
ARRAY_FOLDERS = tuple(generator.ARRAY_FOLDERS)
METADATA_FOLDER = "metadata"
REQUIRED_ROW_FIELDS = {
    "condition_id",
    "condition_index",
    "dig_sha256",
    "map_id",
    "map_index",
    "pair_slot_id",
    "sample_index",
    "scenario_sha256",
    "schema",
    "seed_base",
    "source_group_id",
}
COMPATIBILITY_FIELDS = (
    "schema",
    "dataset",
    "seed_base",
    "taxonomy_version",
    "taxonomy_release",
    "generator",
    "source_foundations_sha256",
    "source_foundations_images",
    "tile_size_m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_equal(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(1 << 20)
            right_chunk = right.read(1 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        fieldnames = list(reader.fieldnames)
        missing = sorted(REQUIRED_ROW_FIELDS - set(fieldnames))
        if missing:
            raise ValueError(f"{path}: missing manifest fields {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return rows, fieldnames


def _map_id_prefix(row: dict[str, str], bank: Path) -> str:
    sample_index = int(row["sample_index"])
    suffix = f"-{sample_index:04d}"
    map_id = row["map_id"]
    if not map_id.endswith(suffix):
        raise ValueError(
            f"{bank}: map_id {map_id!r} does not end in sample index {suffix!r}"
        )
    return map_id[: -len(suffix)]


def _row_file(bank: Path, folder: str, sample_index: int) -> Path:
    return bank / "dataset" / folder / f"img_{sample_index}.npy"


def _metadata_file(bank: Path, sample_index: int) -> Path:
    return bank / "dataset" / METADATA_FOLDER / f"trench_{sample_index}.json"


def _read_row_metadata(bank: Path, row: dict[str, str]) -> dict[str, Any]:
    sample_index = int(row["sample_index"])
    payload = _read_json(_metadata_file(bank, sample_index))
    if "map_id" in payload and payload["map_id"] != row["map_id"]:
        raise ValueError(
            f"{bank}: metadata map_id {payload['map_id']!r} does not match "
            f"manifest map_id {row['map_id']!r}"
        )
    return payload


def _validate_summary(
    bank: Path,
    summary: dict[str, Any],
    observed_counts: dict[str, int],
) -> None:
    built = summary.get("conditions_built_this_run")
    if not isinstance(built, list) or set(built) != set(observed_counts):
        raise ValueError(
            f"{bank}: generation summary condition set does not match manifest"
        )
    declared_counts = summary.get("maps_per_condition")
    if declared_counts != observed_counts:
        raise ValueError(f"{bank}: generation summary map counts do not match manifest")
    if summary.get("accepted_maps") != sum(observed_counts.values()):
        raise ValueError(f"{bank}: generation summary accepted_maps is stale")
    if summary.get("condition_count") != len(generator.MAIN_CONDITIONS):
        raise ValueError(f"{bank}: generation summary has the wrong registry size")
    for field in COMPATIBILITY_FIELDS:
        if field not in summary:
            raise ValueError(f"{bank}: generation summary is missing {field!r}")


def _validate_bank(bank: Path) -> dict[str, Any]:
    bank = bank.resolve()
    manifest_path = bank / "manifest.csv"
    summary_path = bank / "generation_summary.json"
    rows, fieldnames = _read_manifest(manifest_path)
    summary = _read_json(summary_path)

    conditions = tuple(generator.MAIN_CONDITIONS)
    condition_by_id = {condition.id: condition for condition in conditions}
    condition_index = {
        condition.id: index for index, condition in enumerate(conditions)
    }
    rows_by_condition: dict[str, dict[int, dict[str, str]]] = {}
    sample_indices: set[int] = set()
    map_ids: set[str] = set()
    scenario_ids: set[str] = set()
    prefixes: set[str] = set()

    for row in rows:
        condition_id = row["condition_id"]
        condition = condition_by_id.get(condition_id)
        if condition is None:
            raise ValueError(f"{bank}: unknown condition {condition_id!r}")
        try:
            map_index = int(row["map_index"])
            sample_index = int(row["sample_index"])
            row_condition_index = int(row["condition_index"])
            row_seed_base = int(row["seed_base"])
        except ValueError as error:
            raise ValueError(
                f"{bank}: non-integer row identity in {row['map_id']!r}"
            ) from error
        if map_index < 0:
            raise ValueError(f"{bank}: negative map_index in {row['map_id']!r}")
        expected_condition_index = condition_index[condition_id]
        expected_sample_index = generator.sample_index_of(
            expected_condition_index, map_index
        )
        if row_condition_index != expected_condition_index:
            raise ValueError(
                f"{bank}: {row['map_id']} has condition_index "
                f"{row_condition_index}, expected {expected_condition_index}"
            )
        if sample_index != expected_sample_index:
            raise ValueError(
                f"{bank}: {row['map_id']} has sample_index {sample_index}, "
                f"expected {expected_sample_index}"
            )
        expected_pair_slot = f"{condition.dig_bank_level}:{map_index}"
        if row["pair_slot_id"] != expected_pair_slot:
            raise ValueError(
                f"{bank}: {row['map_id']} has pair_slot_id "
                f"{row['pair_slot_id']!r}, expected {expected_pair_slot!r}"
            )
        if row_seed_base != generator.SEED_BASE:
            raise ValueError(f"{bank}: {row['map_id']} has the wrong seed base")
        if row["schema"] != generator.SCHEMA:
            raise ValueError(f"{bank}: {row['map_id']} has the wrong row schema")
        if not row["source_group_id"]:
            raise ValueError(f"{bank}: {row['map_id']} has an empty source group")

        condition_rows = rows_by_condition.setdefault(condition_id, {})
        if map_index in condition_rows:
            raise ValueError(
                f"{bank}: duplicate condition/map index {condition_id}/{map_index}"
            )
        condition_rows[map_index] = row
        if sample_index in sample_indices:
            raise ValueError(f"{bank}: sample_index collision {sample_index}")
        if row["map_id"] in map_ids:
            raise ValueError(f"{bank}: map_id collision {row['map_id']!r}")
        if row["scenario_sha256"] in scenario_ids:
            raise ValueError(f"{bank}: scenario collision {row['scenario_sha256']!r}")
        sample_indices.add(sample_index)
        map_ids.add(row["map_id"])
        scenario_ids.add(row["scenario_sha256"])
        prefixes.add(_map_id_prefix(row, bank))

        for folder in ARRAY_FOLDERS:
            array = _row_file(bank, folder, sample_index)
            if not array.is_file():
                raise FileNotFoundError(f"{bank}: missing array {array}")
        metadata = _metadata_file(bank, sample_index)
        if not metadata.is_file():
            raise FileNotFoundError(f"{bank}: missing metadata {metadata}")

    if len(prefixes) != 1:
        raise ValueError(f"{bank}: manifest uses multiple map_id prefixes")

    counts: dict[str, int] = {}
    for condition_id, condition_rows in rows_by_condition.items():
        indices = sorted(condition_rows)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"{bank}: {condition_id} map indices are not contiguous from zero"
            )
        counts[condition_id] = len(indices)
    if len(set(counts.values())) != 1:
        raise ValueError(f"{bank}: selected conditions have unequal map counts")

    _validate_summary(bank, summary, counts)
    return {
        "path": bank,
        "rows": rows,
        "fieldnames": fieldnames,
        "summary": summary,
        "counts": counts,
        "rows_by_condition": rows_by_condition,
        "map_id_prefix": prefixes.pop(),
        "manifest_sha256": _sha256(manifest_path),
        "summary_sha256": _sha256(summary_path),
    }


def _canonical_levels() -> dict[str, set[str]]:
    levels: dict[str, set[str]] = {}
    for condition in generator.MAIN_CONDITIONS:
        levels.setdefault(condition.dig_bank_level, set()).add(condition.id)
    return levels


def _validate_coverage(base: dict[str, Any], extensions: list[dict[str, Any]]) -> None:
    canonical_conditions = {condition.id for condition in generator.MAIN_CONDITIONS}
    if set(base["counts"]) != canonical_conditions:
        missing = sorted(canonical_conditions - set(base["counts"]))
        extra = sorted(set(base["counts"]) - canonical_conditions)
        raise ValueError(
            f"{base['path']}: base bank is not complete; missing={missing}, extra={extra}"
        )
    if len(set(base["counts"].values())) != 1:
        raise ValueError(f"{base['path']}: base condition counts are unequal")

    canonical_levels = _canonical_levels()
    claimed_levels: set[str] = set()
    base_count = next(iter(base["counts"].values()))
    for extension in extensions:
        observed = set(extension["counts"])
        levels = {
            condition.dig_bank_level
            for condition in generator.MAIN_CONDITIONS
            if condition.id in observed
        }
        expected = set().union(*(canonical_levels[level] for level in levels))
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"{extension['path']}: extension must contain whole dig-bank "
                f"levels; missing={missing}, extra={extra}"
            )
        repeated = claimed_levels & levels
        if repeated:
            raise ValueError(
                f"dig-bank levels occur in multiple extensions: {sorted(repeated)}"
            )
        extension_count = next(iter(extension["counts"].values()))
        if extension_count <= base_count:
            raise ValueError(
                f"{extension['path']}: extension has {extension_count} maps per "
                f"condition, base already has {base_count}"
            )
        extension["levels"] = sorted(levels)
        extension["map_count"] = extension_count
        claimed_levels.update(levels)


def _validate_compatibility(
    base: dict[str, Any], extensions: list[dict[str, Any]]
) -> None:
    for extension in extensions:
        extra_fields = set(extension["fieldnames"]) - set(base["fieldnames"])
        if extra_fields:
            raise ValueError(
                f"{extension['path']}: extension manifest has fields absent "
                f"from the base manifest: {sorted(extra_fields)}"
            )
        for field in COMPATIBILITY_FIELDS:
            if extension["summary"][field] != base["summary"][field]:
                raise ValueError(
                    f"{extension['path']}: incompatible generation field {field!r}"
                )


def _compare_overlap(
    base: dict[str, Any], extension: dict[str, Any]
) -> tuple[int, int]:
    base_count = next(iter(base["counts"].values()))
    overlap_rows = 0
    overlap_arrays = 0
    for condition_id in sorted(extension["counts"]):
        for map_index in range(base_count):
            base_row = base["rows_by_condition"][condition_id][map_index]
            extension_row = extension["rows_by_condition"][condition_id][map_index]
            for field in base["fieldnames"]:
                if field == "map_id":
                    continue
                if base_row.get(field, "") != extension_row.get(field, ""):
                    raise ValueError(
                        f"{extension['path']}: overlap identity differs at "
                        f"{condition_id}/{map_index} field {field!r}"
                    )
            sample_index = int(base_row["sample_index"])
            for folder in ARRAY_FOLDERS:
                base_array = _row_file(base["path"], folder, sample_index)
                extension_array = _row_file(extension["path"], folder, sample_index)
                if not _files_equal(base_array, extension_array):
                    raise ValueError(
                        f"{extension['path']}: overlap array bytes differ at "
                        f"{condition_id}/{map_index}/{folder}"
                    )
                overlap_arrays += 1
            base_metadata = _read_row_metadata(base["path"], base_row)
            extension_metadata = _read_row_metadata(extension["path"], extension_row)
            base_metadata_map_id = base_metadata.pop("map_id", None)
            extension_metadata_map_id = extension_metadata.pop("map_id", None)
            if (base_metadata_map_id is None) != (
                extension_metadata_map_id is None
            ) or base_metadata != extension_metadata:
                raise ValueError(
                    f"{extension['path']}: overlap metadata differs at "
                    f"{condition_id}/{map_index}"
                )
            overlap_rows += 1
    return overlap_rows, overlap_arrays


def _write_manifest(
    path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_row_dataset(
    source: Path,
    output: Path,
    source_row: dict[str, str],
    output_row: dict[str, str],
) -> None:
    sample_index = int(source_row["sample_index"])
    if sample_index != int(output_row["sample_index"]):
        raise RuntimeError("source and output rows have different sample indices")
    for folder in ARRAY_FOLDERS:
        destination = _row_file(output, folder, sample_index)
        if destination.exists():
            raise RuntimeError(f"output array collision: {destination}")
        shutil.copy2(_row_file(source, folder, sample_index), destination)
    metadata = _metadata_file(output, sample_index)
    if metadata.exists():
        raise RuntimeError(f"output metadata collision: {metadata}")
    payload = _read_row_metadata(source, source_row)
    if "map_id" not in payload or source_row["map_id"] == output_row["map_id"]:
        shutil.copy2(_metadata_file(source, sample_index), metadata)
        return
    payload["map_id"] = output_row["map_id"]
    metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def materialize_candidate_extension(
    base_path: Path,
    extension_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    if not extension_paths:
        raise ValueError("at least one --extension bank is required")
    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    base = _validate_bank(base_path)
    extensions = [_validate_bank(path) for path in extension_paths]
    _validate_coverage(base, extensions)
    _validate_compatibility(base, extensions)

    overlap_receipts: dict[Path, tuple[int, int]] = {}
    for extension in extensions:
        overlap_receipts[extension["path"]] = _compare_overlap(base, extension)

    rows_with_sources: list[tuple[dict[str, str], Path, dict[str, str]]] = [
        (dict(row), base["path"], row) for row in base["rows"]
    ]
    base_count = next(iter(base["counts"].values()))
    for extension in extensions:
        for condition_id in sorted(extension["counts"]):
            for map_index in range(base_count, extension["map_count"]):
                source_row = extension["rows_by_condition"][condition_id][map_index]
                output_row = {
                    field: source_row.get(field, "") for field in base["fieldnames"]
                }
                sample_index = int(output_row["sample_index"])
                output_row["map_id"] = f"{base['map_id_prefix']}-{sample_index:04d}"
                rows_with_sources.append((output_row, extension["path"], source_row))

    sample_indices: set[int] = set()
    map_ids: set[str] = set()
    scenario_ids: set[str] = set()
    for row, _, _ in rows_with_sources:
        sample_index = int(row["sample_index"])
        if sample_index in sample_indices:
            raise ValueError(f"output sample_index collision: {sample_index}")
        if row["map_id"] in map_ids:
            raise ValueError(f"output map_id collision: {row['map_id']!r}")
        if row["scenario_sha256"] in scenario_ids:
            raise ValueError(f"output scenario collision: {row['scenario_sha256']!r}")
        sample_indices.add(sample_index)
        map_ids.add(row["map_id"])
        scenario_ids.add(row["scenario_sha256"])

    rows_with_sources.sort(key=lambda item: int(item[0]["sample_index"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for folder in ARRAY_FOLDERS:
            (staging / "dataset" / folder).mkdir(parents=True)
        (staging / "dataset" / METADATA_FOLDER).mkdir(parents=True)
        for output_row, source, source_row in rows_with_sources:
            _copy_row_dataset(source, staging, source_row, output_row)

        output_rows = [row for row, _, _ in rows_with_sources]
        _write_manifest(staging / "manifest.csv", output_rows, base["fieldnames"])
        final_counts: dict[str, int] = {}
        for row in output_rows:
            condition_id = row["condition_id"]
            final_counts[condition_id] = final_counts.get(condition_id, 0) + 1

        extension_receipts = []
        for extension in extensions:
            overlap_rows, overlap_arrays = overlap_receipts[extension["path"]]
            appended = sum(
                extension["counts"][condition] - base_count
                for condition in extension["counts"]
            )
            extension_receipts.append(
                {
                    "path": str(extension["path"]),
                    "manifest_sha256": extension["manifest_sha256"],
                    "generation_summary_sha256": extension["summary_sha256"],
                    "conditions": sorted(extension["counts"]),
                    "dig_bank_levels": extension["levels"],
                    "maps_per_condition": extension["map_count"],
                    "scenarios": len(extension["rows"]),
                    "overlap_rows_verified": overlap_rows,
                    "overlap_array_files_verified": overlap_arrays,
                    "appended_scenarios": appended,
                }
            )

        receipt = {
            "schema": SCHEMA,
            "base": {
                "path": str(base["path"]),
                "manifest_sha256": base["manifest_sha256"],
                "generation_summary_sha256": base["summary_sha256"],
                "maps_per_condition": base_count,
                "scenarios": len(base["rows"]),
            },
            "extensions": extension_receipts,
            "compatibility": {
                field: base["summary"][field] for field in COMPATIBILITY_FIELDS
            },
            "output": {
                "map_id_prefix": base["map_id_prefix"],
                "maps_per_condition": final_counts,
                "scenarios": len(output_rows),
                "array_files": len(output_rows) * len(ARRAY_FOLDERS),
                "metadata_files": len(output_rows),
                "manifest_sha256": _sha256(staging / "manifest.csv"),
            },
        }
        (staging / "extension_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise FileExistsError(f"output appeared during materialization: {output}")
        os.rename(staging, output)
        return receipt
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge deterministic whole-level --only candidate shards into a "
            "complete base bank."
        )
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = materialize_candidate_extension(args.base, args.extension, args.output)
    print(
        f"materialized {receipt['output']['scenarios']} scenarios from "
        f"{len(receipt['extensions'])} extension bank(s)"
    )


if __name__ == "__main__":
    main()
