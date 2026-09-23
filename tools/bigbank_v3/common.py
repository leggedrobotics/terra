"""Shared paths, identity helpers and the per-map transform chain for train_v3.

The chain for every new map is the one the current generalist bank's maps went
through, called through the same pinned functions:

* V6 constrained maps: generator arrays -> (trench) ``derive_v6_trench_width``
  (V8, 1.3 m) -> R2 ``compute_reward_v2_distance_map`` -> (trench)
  ``enrich_sidecar`` (finite sections from the generator ``trench_arms``).
* capability controls: ``make_allfree_arrays`` of the parent generator map ->
  (trench) V8 width with ``allfree=True`` -> R2 -> (trench) enrichment against
  the parent's generator record.
* V7 adjacent foundations: ``generate_scenarios`` -> ``arrays_for_scenario`` ->
  R2.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pinned  # noqa: E402  (binds tools.map_generation to the 60d01307 snapshot)

from pinned import controls, generator, v7review, v8  # noqa: E402

sys.path.insert(0, str(pinned.TERRA_ROOT / "tools"))
import enrich_trench_finite_metadata as enrich  # noqa: E402

from terra.env_generation.distance import compute_reward_v2_distance_map  # noqa: E402
from terra.maps_buffer import RESET_ARRAY_FOLDERS, reset_array_scenario_sha256  # noqa: E402

ARTIFACTS = Path("/home/lorenzo/moleworks/.artifacts")
ROOT = ARTIFACTS / "terra_gru_bigbank_20260923"
GENERATION = ROOT / "generation"
OLD_ROOT = ARTIFACTS / "terra_v8_trench_finite_enriched_20260819"
OLD_POOL = OLD_ROOT / "train_v2_pooled_generalist"
P5_POOL = ARTIFACTS / "terra_p5_candidates320_full_20260801_642756cc"
TILE_SIZE_M = 0.571428571428125
DISTANCE_REF_M = 16.0
DISTANCE_BOUND = 2.5
ENRICH_TOOL = pinned.TERRA_ROOT / "tools" / "enrich_trench_finite_metadata.py"
ENRICH_TOOL_SHA256 = "58506496e50ca74e4812b68d7f960b70edb9418e7d1c0da620184e11e7f75914"
V7_TRAIN_SEED = v8.SPLIT_SEEDS["train"]
V7_NEW_SEED = 2026092301
CONTROL_PARENTS = {
    control_id: definition["parent_condition"]
    for control_id, definition in controls.CONTROL_DEFINITIONS.items()
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dig_sha(target: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(target) < 0).tobytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_arrays(dataset: Path, index: int, folder_prefix: str = "") -> dict[str, np.ndarray]:
    return {
        name: np.load(dataset / folder_prefix / name / f"img_{index}.npy", allow_pickle=False)
        for name in RESET_ARRAY_FOLDERS
    }


def with_r2_distance(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    distance = compute_reward_v2_distance_map(
        np.squeeze(arrays["images"]),
        np.squeeze(arrays["occupancy"]),
        tile_size_m=TILE_SIZE_M,
        distance_ref_m=DISTANCE_REF_M,
        distance_bound=DISTANCE_BOUND,
    )
    return {**arrays, "distance": distance}


def enrich_trench(map_id, metadata, record_path: Path, record, target):
    enriched, _stats = enrich.enrich_sidecar(
        map_id=map_id,
        sidecar=metadata,
        record_path=record_path,
        record=record,
        record_sha256=sha256_file(record_path),
        target_map=target,
        tool_sha256=ENRICH_TOOL_SHA256,
    )
    return enriched


def v8_trench_row(row: dict[str, Any], arrays, metadata, *, allfree: bool):
    """``_derive_v6_trench_row`` on in-memory arrays (same function calls)."""
    parent_scenario_id = row["scenario_id"]
    parent_source_id = row["source_id"]
    derived, metrics = v8.derive_v6_trench_width(arrays, metadata, allfree=allfree)
    derived_metadata = {
        **metadata,
        "schema": "terra_v8_v6_axis_metadata_v4",
        "parent_metadata_schema": metadata.get("schema"),
        "lineage": v8.V6_TRENCH_WIDTH_LINEAGE,
        **metrics,
    }
    digest = v8._dig_sha256(derived["images"] < 0)
    identity = v8._replace_scenario_identity(row, reset_array_scenario_sha256(derived))
    derived_row = {
        **identity,
        "source_id": f"dig:{digest}",
        "dig_sha256": digest,
        "required_dig_volume": metrics["required_dig_volume"],
        "parent_scenario_id": parent_scenario_id,
        "parent_source_id": parent_source_id,
        "lineage": v8.V6_TRENCH_WIDTH_LINEAGE,
        "target_trench_width_m": v8.TARGET_TRENCH_WIDTH_M,
        "target_trench_width_tiles": v8.TARGET_TRENCH_WIDTH_TILES,
    }
    return derived, derived_metadata, derived_row


def v6_constrained_map(*, condition_id, family, arrays, metadata, record, record_path,
                       sample_index):
    """One V6 generator map -> final (arrays, metadata, row) of the current bank."""
    row = {
        "slot_index": None,
        "map_id": record["map_id"],
        "scenario_id": reset_array_scenario_sha256(arrays),
        "source_id": record["source_group_id"],
        "split": "train",
        "family": family,
        "stratum": "curriculum_v6_main",
        "primary_cell": condition_id,
        "slot_weight": 1.0,
        "identity_slot_multiplicity": 1,
        "pair_slot_id": record["pair_slot_id"],
        "candidate_sample_index": int(sample_index),
    }
    if row["scenario_id"] != record["scenario_sha256"]:
        raise RuntimeError(f"{record['map_id']}: generator scenario hash mismatch")
    if family == "trench":
        arrays, metadata, row = v8_trench_row(row, arrays, metadata, allfree=False)
    arrays = with_r2_distance(arrays)
    row["scenario_id"] = reset_array_scenario_sha256(arrays)
    if family == "trench":
        metadata = enrich_trench(row["map_id"], metadata, record_path, record, arrays["images"])
    return arrays, metadata, row


def control_map(*, control_id, parent_arrays, parent_metadata, parent_record,
                parent_record_path):
    """Capability-floor control from its parent's generator map (train96 path)."""
    parent_condition = CONTROL_PARENTS[control_id]
    family = controls.CONTROL_DEFINITIONS[control_id]["family"]
    transformed = controls.make_allfree_arrays(
        parent_arrays["images"], parent_arrays["occupancy"],
        parent_arrays["dumpability"], parent_arrays["actions"],
    )
    metadata = dict(parent_metadata)
    metadata.update(
        {
            "control_condition_id": control_id,
            "control_schema": controls.CONTROL_SCHEMA,
            "dump_layout": "allfree",
            "parent_condition_id": parent_condition,
            "parent_map_id": parent_record["map_id"],
        }
    )
    scenario_id = reset_array_scenario_sha256(transformed)
    dig_volume = int((transformed["images"] < 0).sum())
    dump_cells = int((transformed["images"] > 0).sum())
    row = {
        "slot_index": None,
        "map_id": f"{control_id}:train:{scenario_id[:16]}",
        "scenario_id": scenario_id,
        "source_id": parent_record["source_group_id"],
        "split": "train",
        "family": family,
        "stratum": controls.CONTROL_STRATUM,
        "primary_cell": control_id,
        "slot_weight": 1.0,
        "identity_slot_multiplicity": 1,
        "pair_slot_id": parent_record["pair_slot_id"],
        "parent_condition_id": parent_condition,
        "parent_map_id": parent_record["map_id"],
        "parent_scenario_id": parent_record["scenario_sha256"],
        "accepted_dump_cells": dump_cells,
        "required_dig_volume": dig_volume,
        "single_layer_capacity_ratio": dump_cells / dig_volume,
        "control_definition": "all_legal_non_dig_cells",
    }
    arrays = transformed
    if family == "trench":
        arrays, metadata, row = v8_trench_row(row, arrays, metadata, allfree=True)
    arrays = with_r2_distance(arrays)
    row["scenario_id"] = reset_array_scenario_sha256(arrays)
    if family == "trench":
        metadata = enrich_trench(row["map_id"], metadata, parent_record_path, parent_record,
                                 arrays["images"])
    return arrays, metadata, row


def v7_maps(count: int, seed: int):
    """(condition, k, arrays, metadata, row) for the six V7 foundation conditions."""
    scenarios = v7review.generate_scenarios(count, seed)
    grouped = v8._group_scenarios(scenarios, count)
    for condition in v8.V7_CONDITIONS:
        if condition.family != "foundation":
            continue
        for k, scenario in enumerate(grouped[(condition.family, condition.geometry)]):
            arrays, metrics = v8.arrays_for_scenario(scenario, condition.dump_layout)
            metadata = v8._metadata_for(scenario, condition, metrics)
            row = v8._record_for(scenario=scenario, condition=condition, split="train",
                                 map_index=k, slot=k + 1, arrays=arrays, metrics=metrics)
            arrays = with_r2_distance(arrays)
            row["scenario_id"] = reset_array_scenario_sha256(arrays)
            yield condition, k, arrays, metadata, row


def save_map(dataset: Path, slot: int, arrays, metadata) -> None:
    for name in RESET_ARRAY_FOLDERS:
        np.save(dataset / name / f"img_{slot}.npy", arrays[name])
    (dataset / "metadata" / f"trench_{slot}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def arrays_equal(a, b) -> bool:
    return all(
        a[n].dtype == b[n].dtype and a[n].shape == b[n].shape and a[n].tobytes() == b[n].tobytes()
        for n in RESET_ARRAY_FOLDERS
    )


def old_pool_index() -> dict[str, Any]:
    """Old pooled bank rows grouped by condition directory, in slot order."""
    dataset = json.loads((OLD_POOL / "dataset.json").read_text())
    rows = read_jsonl(OLD_POOL / "manifest.jsonl")
    by_dir: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dir.setdefault(row["pooled_from_dataset"], []).append(row)
    order = dataset["trench_pilot_pooling"]["conditions"]
    if list(by_dir) != order:
        raise RuntimeError("old pool order differs from its receipt")
    return {"dataset": dataset, "rows": rows, "by_dir": by_dir, "order": order}
