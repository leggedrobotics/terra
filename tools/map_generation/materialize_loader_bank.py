#!/usr/bin/env python3
"""Convert one reviewed split bank into Terra's exact loader layout.

This is deliberately a narrow bridge. It accepts only the output of
``materialize_splits.py`` and writes:

* one exact, equal-size loader level per training condition;
* one contiguous loader panel for each evaluation split; and
* one source registry and one frozen environment-protocol receipt.

Review galleries and unsplit generator outputs are rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from terra.benchmark_protocol import (  # noqa: E402
    BENCHMARK_JAX_DEFAULT_PRNG_IMPL,
    BENCHMARK_JAX_THREEFRY_PARTITIONABLE,
)
from terra.benchmark_protocol import canonical_json_sha256  # noqa: E402
from terra.benchmark_protocol import frozen_environment_protocol  # noqa: E402
from terra.config import EnvConfig  # noqa: E402
from terra.maps_buffer import EXACT_DATASET_SCHEMA  # noqa: E402
from terra.maps_buffer import MapsBuffer  # noqa: E402
from terra.maps_buffer import RESET_ARRAY_FOLDERS  # noqa: E402
from terra.maps_buffer import (  # noqa: E402
    RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
)
from terra.maps_buffer import reset_array_scenario_sha256  # noqa: E402
from terra.maps_buffer import validate_exact_dataset_contract  # noqa: E402
from tools.map_generation.compile_condition_review import (  # noqa: E402
    MANIFEST_SHA256 as REVIEW_MANIFEST_SHA256,
    OUTPUT_SCHEMA as REVIEW_ADMISSION_SCHEMA,
    RELEASE_ID as REVIEW_RELEASE_ID,
    REVIEW_DATA_SHA256,
)

SPLIT_BANK_SCHEMA = "terra_curriculum_split_bank_v1"
LOADER_BANK_SCHEMA = "terra_curriculum_loader_bank_v1"
EPISODE_ID_SCHEMA = "terra_episode_id_v1"
SPLITS = ("train", "promotion", "development", "sealed")
EVALUATION_SPLITS = SPLITS[1:]
ARRAY_FOLDERS = RESET_ARRAY_FOLDERS
REQUIRED_COLUMNS = {
    "condition_id",
    "family",
    "map_id",
    "pair_slot_id",
    "sample_index",
    "scenario_sha256",
    "source_group_id",
    "split",
    "tier",
}
DISTANCE_METRIC = "8_connected_cardinal_1_diagonal_sqrt2"
DISTANCE_NORMALIZATION = "per_map_max_to_1"
ACCEPTED_DUMP_CONTRACT = "exact_visible_dump_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _configure_benchmark_prng() -> None:
    """Use the PRNG mode frozen into evaluation episode identities."""
    jax.config.update(
        "jax_default_prng_impl",
        BENCHMARK_JAX_DEFAULT_PRNG_IMPL,
    )
    jax.config.update(
        "jax_threefry_partitionable",
        BENCHMARK_JAX_THREEFRY_PARTITIONABLE,
    )
    actual = (
        jax.config.jax_default_prng_impl,
        bool(jax.config.jax_threefry_partitionable),
    )
    expected = (
        BENCHMARK_JAX_DEFAULT_PRNG_IMPL,
        BENCHMARK_JAX_THREEFRY_PARTITIONABLE,
    )
    if actual != expected:
        raise RuntimeError(
            f"JAX PRNG contract mismatch: runtime={actual}, expected={expected}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"{path} is missing. Review-only or unsplit banks are not accepted; "
            "run materialize_splits.py first."
        )
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing split manifest: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no scenarios")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_review_admission(path: Path, conditions: list[str]) -> dict[str, Any]:
    receipt = _read_json(path)
    if receipt.get("schema") != REVIEW_ADMISSION_SCHEMA:
        raise ValueError(f"{path}: unsupported review admission schema")
    expected_identity = {
        "release": REVIEW_RELEASE_ID,
        "manifest_sha256": REVIEW_MANIFEST_SHA256,
        "review_data_sha256": REVIEW_DATA_SHA256,
    }
    for field, expected in expected_identity.items():
        if receipt.get(field) != expected:
            raise ValueError(f"{path}: {field} does not match the reviewed release")
    accepted = receipt.get("accepted_conditions")
    if (
        not isinstance(accepted, list)
        or not all(isinstance(value, str) and value for value in accepted)
        or accepted != sorted(set(accepted))
    ):
        raise ValueError(f"{path}: accepted_conditions must be unique and sorted")
    if accepted != conditions:
        raise ValueError(
            "review admission conditions do not match the split bank: "
            f"accepted={accepted}, split={conditions}"
        )
    if not SHA256_PATTERN.fullmatch(str(receipt.get("review_bundle_sha256", ""))):
        raise ValueError(f"{path}: review_bundle_sha256 must be a SHA-256 digest")
    return receipt


def _scenario_sha256(dataset: Path, sample_index: int) -> tuple[str, tuple[int, int]]:
    """Recompute the generator's identity from reset-consumed arrays."""
    shape: tuple[int, int] | None = None
    arrays = {}
    for folder in ARRAY_FOLDERS:
        path = dataset / folder / f"img_{sample_index}.npy"
        if not path.is_file():
            raise ValueError(f"missing scenario array: {path}")
        array = np.ascontiguousarray(np.load(path, allow_pickle=False))
        if array.ndim != 2:
            raise ValueError(f"{path} must be a 2-D array, got {array.shape}")
        current_shape = (int(array.shape[0]), int(array.shape[1]))
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError(
                f"{path} has shape {current_shape}; expected {shape} for this scenario"
            )
        arrays[folder] = array
    assert shape is not None
    return reset_array_scenario_sha256(arrays), shape


def _sample_index(row: dict[str, str]) -> int:
    try:
        value = int(row["sample_index"])
    except ValueError as error:
        raise ValueError(f"{row['map_id']}: sample_index is not an integer") from error
    if value < 0:
        raise ValueError(f"{row['map_id']}: sample_index must be non-negative")
    return value


def _family(row: dict[str, str]) -> str:
    family = row["family"].strip().lower()
    if family not in {"foundation", "trench"}:
        raise ValueError(f"{row['map_id']}: unsupported family {row['family']!r}")
    return family


def _branch_depth(row: dict[str, str]) -> str:
    try:
        tier = int(row["tier"])
    except ValueError as error:
        raise ValueError(f"{row['map_id']}: tier is not an integer") from error
    if tier < 0:
        raise ValueError(f"{row['map_id']}: tier must be non-negative")
    return ("Anchor", "One-axis", "Composed")[min(tier, 2)]


def _selected_map_indices(seeds: list[int], count: int) -> np.ndarray:
    _configure_benchmark_prng()
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, dtype=jnp.uint32))
    index_only_buffer = SimpleNamespace(n_maps=count)
    env_cfg = EnvConfig()

    def selected(key):
        _, index, _ = MapsBuffer._select_index(
            index_only_buffer,
            key,
            env_cfg,
        )
        return index

    return np.asarray(jax.vmap(selected)(keys))


def _exact_reset_seeds(count: int) -> list[int]:
    """Find one scalar seed whose reset path selects each exact map slot."""
    if count <= 0:
        raise ValueError("exact reset seeds require a positive map count")
    found: list[int | None] = [None] * count
    start = 0
    while any(seed is None for seed in found):
        candidates = list(range(start, start + 4096))
        indices = _selected_map_indices(candidates, count)
        for seed, index in zip(candidates, indices):
            if found[int(index)] is None:
                found[int(index)] = seed
        start += len(candidates)
        if start > 1_000_000:
            raise RuntimeError(
                f"could not construct exact reset seeds for {count} maps"
            )
    result = [int(seed) for seed in found if seed is not None]
    np.testing.assert_array_equal(
        _selected_map_indices(result, count),
        np.arange(count),
    )
    return result


def episode_id(
    scenario_id: str,
    reset_seed: int,
    environment_protocol_sha256: str,
) -> str:
    """Bind map arrays, reset randomness, and executable protocol."""
    return canonical_json_sha256(
        {
            "schema": EPISODE_ID_SCHEMA,
            "scenario_id": scenario_id,
            "reset_seed": reset_seed,
            "environment_protocol_sha256": environment_protocol_sha256,
        }
    )


def _validate_split_bank(
    split_bank: Path,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, str]]],
    tuple[int, int],
]:
    summary = _read_json(split_bank / "summary.json")
    if summary.get("schema") != SPLIT_BANK_SCHEMA:
        raise ValueError(
            f"{split_bank} is not a final split bank "
            f"({SPLIT_BANK_SCHEMA}); review-only artifacts are rejected"
        )
    requested = summary.get("requested_pair_slots_per_condition")
    if (
        not isinstance(requested, dict)
        or set(requested) != set(SPLITS)
        or any(
            not isinstance(requested[split], int) or requested[split] <= 0
            for split in SPLITS
        )
    ):
        raise ValueError(
            "split summary must request a positive per-condition count for "
            f"exactly {', '.join(SPLITS)}"
        )

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    all_map_ids: set[str] = set()
    all_scenario_ids: set[str] = set()
    pair_slot_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    observed: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "scenarios": set(),
                "pair_slots": set(),
                "source_groups": set(),
            }
        )
    )
    condition_contracts: dict[str, tuple[str, str]] = {}
    conditions: set[str] | None = None
    common_shape: tuple[int, int] | None = None

    for split in SPLITS:
        rows = _read_csv(split_bank / split / "manifest.csv")
        dataset = split_bank / split / "dataset"
        rows_by_split[split] = rows
        split_conditions = {row["condition_id"] for row in rows}
        if conditions is None:
            conditions = split_conditions
        elif split_conditions != conditions:
            raise ValueError(
                f"{split}: condition support differs from train; "
                f"missing={sorted(conditions - split_conditions)}, "
                f"extra={sorted(split_conditions - conditions)}"
            )

        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            empty = sorted(
                field
                for field in REQUIRED_COLUMNS
                if not isinstance(row[field], str) or not row[field].strip()
            )
            if empty:
                raise ValueError(f"{split}: manifest row has empty fields {empty}")
            map_id = row["map_id"].strip()
            source_id = row["source_group_id"].strip()
            scenario_id = row["scenario_sha256"].strip()
            if not SHA256_PATTERN.fullmatch(scenario_id):
                raise ValueError(f"{map_id}: invalid scenario_sha256 {scenario_id!r}")
            if row["split"] != split:
                raise ValueError(
                    f"{map_id}: manifest split is {row['split']!r}, expected {split!r}"
                )
            condition_id = row["condition_id"]
            contract = (_family(row), _branch_depth(row))
            previous_contract = condition_contracts.setdefault(
                condition_id,
                contract,
            )
            if previous_contract != contract:
                raise ValueError(
                    f"{condition_id}: inconsistent family/depth contract; "
                    f"{previous_contract} != {contract}"
                )
            if map_id in all_map_ids:
                raise ValueError(f"map_id collision: {map_id}")
            if scenario_id in all_scenario_ids:
                raise ValueError(f"scenario hash collision: {scenario_id}")
            all_map_ids.add(map_id)
            all_scenario_ids.add(scenario_id)
            pair_slot_id = row["pair_slot_id"]
            pair_slot_splits[pair_slot_id].add(split)
            source_splits[source_id].add(split)
            observed_ids = observed[condition_id][split]
            observed_ids["scenarios"].add(scenario_id)
            observed_ids["pair_slots"].add(pair_slot_id)
            observed_ids["source_groups"].add(source_id)
            counts[condition_id] += 1

            sample_index = _sample_index(row)
            actual_scenario_id, shape = _scenario_sha256(dataset, sample_index)
            if actual_scenario_id != scenario_id:
                raise ValueError(
                    f"{map_id}: scenario hash mismatch; manifest={scenario_id}, "
                    f"arrays={actual_scenario_id}"
                )
            if common_shape is None:
                common_shape = shape
            elif shape != common_shape:
                raise ValueError(
                    f"{map_id}: map shape {shape} differs from {common_shape}"
                )
            metadata = dataset / "metadata" / f"trench_{sample_index}.json"
            if not metadata.is_file():
                raise ValueError(f"{map_id}: missing metadata sidecar {metadata}")

        mismatches = {
            condition: count
            for condition, count in sorted(counts.items())
            if count != requested[split]
        }
        if mismatches:
            raise ValueError(
                f"{split}: condition counts do not match requested "
                f"{requested[split]}: {mismatches}"
            )

    assert conditions is not None
    assert common_shape is not None
    declared_conditions = summary.get("conditions")
    if (
        not isinstance(declared_conditions, dict)
        or set(declared_conditions) != conditions
    ):
        raise ValueError("split summary condition support does not match manifests")
    assignment_sha256 = summary.get("assignment_sha256")
    if not isinstance(assignment_sha256, str) or not SHA256_PATTERN.fullmatch(
        assignment_sha256
    ):
        raise ValueError("split summary has an invalid assignment_sha256")
    for condition in sorted(conditions):
        for split in SPLITS:
            declared = declared_conditions[condition].get(split, {})
            for metric, values in observed[condition][split].items():
                actual = len(values)
                if declared.get(metric) != actual:
                    raise ValueError(
                        f"{condition}/{split}: summary {metric}="
                        f"{declared.get(metric)!r}, observed distinct {metric}={actual}"
                    )
                if actual != requested[split]:
                    raise ValueError(
                        f"{condition}/{split}: distinct {metric}={actual}, "
                        f"requested {requested[split]}"
                    )

    pair_overlaps = {
        pair_slot_id: sorted(split_set)
        for pair_slot_id, split_set in pair_slot_splits.items()
        if len(split_set) > 1
    }
    if pair_overlaps:
        pair_slot_id = sorted(pair_overlaps)[0]
        raise ValueError(
            f"pair-slot leakage: {pair_slot_id} appears in "
            f"{pair_overlaps[pair_slot_id]}"
        )
    overlaps = {
        source_id: sorted(split_set)
        for source_id, split_set in source_splits.items()
        if len(split_set) > 1
    }
    if overlaps:
        source_id = sorted(overlaps)[0]
        raise ValueError(
            f"source leakage: {source_id} appears in {overlaps[source_id]}"
        )
    return summary, rows_by_split, common_shape


def _condition_directory(index: int, condition: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", condition).strip("-")
    if not safe:
        raise ValueError(
            f"condition_id has no filesystem-safe characters: {condition!r}"
        )
    return f"{index:03d}__{safe}"


def _copy_sidecars(
    input_dataset: Path,
    output_dataset: Path,
    row: dict[str, str],
    slot: int,
) -> None:
    sample_index = _sample_index(row)
    for folder in ARRAY_FOLDERS:
        source = input_dataset / folder / f"img_{sample_index}.npy"
        destination = output_dataset / folder / f"img_{slot}.npy"
        shutil.copy2(source, destination)
    shutil.copy2(
        input_dataset / "metadata" / f"trench_{sample_index}.json",
        output_dataset / "metadata" / f"trench_{slot}.json",
    )


def _manifest_row(
    row: dict[str, str],
    slot: int,
    *,
    evaluation: bool,
    environment_protocol_sha256: str,
    reset_seed: int | None = None,
) -> dict[str, Any]:
    scenario_id = row["scenario_sha256"]
    result: dict[str, Any] = {
        "slot_index": slot,
        "map_id": row["map_id"],
        "scenario_id": scenario_id,
        "source_id": row["source_group_id"],
        "split": row["split"],
        "family": _family(row),
        "stratum": "curriculum_v6_main",
        "primary_cell": row["condition_id"],
        "slot_weight": 1.0,
        "identity_slot_multiplicity": 1,
        "pair_slot_id": row["pair_slot_id"],
        "candidate_sample_index": _sample_index(row),
    }
    if evaluation:
        if reset_seed is None:
            raise ValueError("evaluation manifest rows require a reset seed")
        result.update(
            {
                "reset_seed": reset_seed,
                "episode_id": episode_id(
                    scenario_id,
                    reset_seed,
                    environment_protocol_sha256,
                ),
                "environment_protocol_sha256": environment_protocol_sha256,
            }
        )
    return result


def _materialize_dataset(
    *,
    split_bank: Path,
    output: Path,
    rows: list[dict[str, str]],
    shape: tuple[int, int],
    source_registry: Path,
    environment_protocol_sha256: str,
    evaluation: bool,
) -> None:
    if not rows:
        raise ValueError(f"cannot materialize an empty loader dataset at {output}")
    for folder in (*ARRAY_FOLDERS, "metadata"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    input_dataset = split_bank / rows[0]["split"] / "dataset"
    manifest_rows = []
    seen_seeds: dict[int, str] = {}
    seen_episode_ids: set[str] = set()
    reset_seeds = _exact_reset_seeds(len(rows)) if evaluation else [None] * len(rows)
    for slot, row in enumerate(rows, start=1):
        _copy_sidecars(input_dataset, output, row, slot)
        manifest_row = _manifest_row(
            row,
            slot,
            evaluation=evaluation,
            environment_protocol_sha256=environment_protocol_sha256,
            reset_seed=reset_seeds[slot - 1],
        )
        if evaluation:
            seed = manifest_row["reset_seed"]
            previous = seen_seeds.get(seed)
            if previous is not None:
                raise ValueError(
                    "reset-seed hash collision: "
                    f"{previous} and {manifest_row['scenario_id']} -> {seed}"
                )
            seen_seeds[seed] = manifest_row["scenario_id"]
            episode = manifest_row["episode_id"]
            if episode in seen_episode_ids:
                raise ValueError(f"episode hash collision: {episode}")
            seen_episode_ids.add(episode)
        manifest_rows.append(manifest_row)

    _write_jsonl(output / "manifest.jsonl", manifest_rows)
    _write_json(
        output / "dataset.json",
        {
            "schema": EXACT_DATASET_SCHEMA,
            "slot_count": len(rows),
            "unique_identity_count": len(rows),
            "shape": list(shape),
            "distance_metric": DISTANCE_METRIC,
            "distance_normalization": DISTANCE_NORMALIZATION,
            "accepted_dump_contract": ACCEPTED_DUMP_CONTRACT,
            "scenario_identity_contract": (RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT),
            "source_registry": os.path.relpath(source_registry, output),
            "source_registry_sha256": _sha256_file(source_registry),
        },
    )
    validate_exact_dataset_contract(output, len(rows))


def materialize_loader_bank(
    split_bank: Path,
    output: Path,
    terra_revision: str,
    review_admission: Path,
) -> dict[str, Any]:
    """Materialize the only loader-ready view of one final split bank."""
    split_bank = split_bank.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    summary, rows_by_split, shape = _validate_split_bank(split_bank)
    conditions = sorted({row["condition_id"] for row in rows_by_split["train"]})
    review_admission = review_admission.resolve()
    review_receipt = _validate_review_admission(review_admission, conditions)
    environment_protocol = frozen_environment_protocol(terra_revision)
    protocol_hash = environment_protocol["environment_protocol_sha256"]
    if not SHA256_PATTERN.fullmatch(protocol_hash):
        raise RuntimeError("frozen environment protocol returned an invalid hash")
    if environment_protocol["accepted_dump_contract"] != ACCEPTED_DUMP_CONTRACT:
        raise RuntimeError(
            "frozen environment protocol does not use exact_visible_dump_v1"
        )

    all_rows = [row for split in SPLITS for row in rows_by_split[split]]
    registry_rows = [
        {
            "map_id": row["map_id"],
            "scenario_id": row["scenario_sha256"],
            "source_id": row["source_group_id"],
            "split": row["split"],
            "family": _family(row),
            "primary_cell": row["condition_id"],
        }
        for row in sorted(
            all_rows,
            key=lambda row: (
                SPLITS.index(row["split"]),
                row["condition_id"],
                row["map_id"],
            ),
        )
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.tmp-"
    ) as temporary:
        root = Path(temporary)
        source_registry = root / "source_registry.jsonl"
        _write_jsonl(source_registry, registry_rows)
        _write_json(root / "environment_protocol.json", environment_protocol)
        shutil.copyfile(review_admission, root / "review_admission.json")

        training_levels = []
        training_rows = rows_by_split["train"]
        for level_index, condition in enumerate(conditions):
            rows = sorted(
                (row for row in training_rows if row["condition_id"] == condition),
                key=lambda row: (row["map_id"], _sample_index(row)),
            )
            relative = Path("train") / _condition_directory(level_index, condition)
            _materialize_dataset(
                split_bank=split_bank,
                output=root / relative,
                rows=rows,
                shape=shape,
                source_registry=source_registry,
                environment_protocol_sha256=protocol_hash,
                evaluation=False,
            )
            training_levels.append(
                {
                    "level_index": level_index,
                    "condition_id": condition,
                    "family": _family(rows[0]),
                    "branch_depth": _branch_depth(rows[0]),
                    "maps_path": relative.as_posix(),
                    "map_count": len(rows),
                }
            )

        evaluation_panels = {}
        for split in EVALUATION_SPLITS:
            rows = sorted(
                rows_by_split[split],
                key=lambda row: (row["condition_id"], row["map_id"]),
            )
            _materialize_dataset(
                split_bank=split_bank,
                output=root / split,
                rows=rows,
                shape=shape,
                source_registry=source_registry,
                environment_protocol_sha256=protocol_hash,
                evaluation=True,
            )
            evaluation_panels[split] = {
                "maps_path": split,
                "slot_count": len(rows),
                "conditions": len({row["condition_id"] for row in rows}),
            }

        loader_summary = {
            "schema": LOADER_BANK_SCHEMA,
            "source_split_summary_sha256": _sha256_file(split_bank / "summary.json"),
            "source_assignment_sha256": summary["assignment_sha256"],
            "source_registry": "source_registry.jsonl",
            "source_registry_sha256": _sha256_file(source_registry),
            "environment_protocol": "environment_protocol.json",
            "environment_protocol_sha256": protocol_hash,
            "scenario_identity_contract": (RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT),
            "shape": list(shape),
            "train": training_levels,
            "evaluation_panels": evaluation_panels,
        }
        loader_summary.update(
            {
                "review_admission": "review_admission.json",
                "review_admission_sha256": _sha256_file(root / "review_admission.json"),
                "review_release": review_receipt["release"],
                "review_manifest_sha256": review_receipt["manifest_sha256"],
            }
        )
        _write_json(root / "dataset.json", loader_summary)
        root.rename(output)
    return loader_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a final source-disjoint Terra split bank into exact "
            "per-condition training levels and contiguous evaluation panels."
        )
    )
    parser.add_argument("--split-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--terra-revision",
        required=True,
        help="immutable Terra commit or source revision bound into episode_id",
    )
    parser.add_argument(
        "--review-admission",
        required=True,
        type=Path,
        help="validated review_admission.json from compile_condition_review.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize_loader_bank(
        args.split_bank,
        args.output,
        args.terra_revision,
        args.review_admission,
    )
    print(
        f"materialized {len(summary['train'])} training levels and "
        f"{len(summary['evaluation_panels'])} evaluation panels"
    )


if __name__ == "__main__":
    main()
