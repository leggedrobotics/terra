#!/usr/bin/env python3
"""Build the V8 bank: frozen V6 constraints plus V7 adjacent geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from terra.maps_buffer import (  # noqa: E402
    EXACT_DATASET_SCHEMA,
    RESET_ARRAY_FOLDERS,
    RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
    reset_array_scenario_sha256,
    validate_exact_dataset_contract,
)
from tools.map_generation.generate_prototypes import (  # noqa: E402
    compute_geodesic_distance,
)
from tools.map_generation.generate_v7_geometry_review import (  # noqa: E402
    FOUNDATION_GEOMETRIES,
    MAP_SIZE,
    TILE_SIZE_M,
    TRENCH_GEOMETRIES,
    ReviewScenario,
    generate_scenarios,
)
from tools.map_generation.materialize_loader_bank import (  # noqa: E402
    ACCEPTED_DUMP_CONTRACT,
    DISTANCE_METRIC,
    DISTANCE_NORMALIZATION,
    LOADER_BANK_SCHEMA,
)

RELEASE_ID = "terra_v8_v6_constraints_v7_adjacent_train96_v5"
RELEASE_NAME = "Terra V8 · V6 constraints + V7 adjacent geometry, Train-96 v5"
BUILD_SCHEMA = "terra_v8_combined_bank_build_v5"
MIXTURE_SCHEMA = "terra_v8_training_mixture_v4"
V8_REVIEW_SCHEMA = "terra_v8_review_candidate_v5"

V6_RELEASE_ID = "terra_v6main_capfloor34_train96_v1"
V6_DATASET_SHA256 = "2a1d74eec0ff8115b0922c9f82f14ddb1589aecec2d63f26d8461339b2f66f45"
V6_CONSTRAINED_COUNT = 32
V6_CAPABILITY_FLOOR_IDS = ("fnd-slab-allfree", "trn-straight-allfree")
TARGET_TRENCH_WIDTH_M = 1.3
TARGET_TRENCH_WIDTH_TILES = TARGET_TRENCH_WIDTH_M / TILE_SIZE_M
TARGET_TRENCH_RADIUS_TILES = TARGET_TRENCH_WIDTH_TILES / 2.0
TARGET_TRENCH_END_PADDING_TILES = 0.25
V6_TRENCH_WIDTH_LINEAGE = "v6_constraint_layout_v8_width_1p3m"

SPLIT_COUNTS = {
    "train": 96,
    "promotion": 16,
    "development": 16,
    "sealed": 32,
}
SPLIT_SEEDS = {
    "train": 2026080801,
    "promotion": 2026080802,
    "development": 2026080803,
    "sealed": 2026080804,
}
EVALUATION_SPLITS = ("promotion", "development", "sealed")
V7_LAYOUTS = ("adjacent_generous",)
ENCLOSED_GEOMETRIES = {
    "foundation": frozenset({"courtyard", "bearing_walls", "courtyard_pads"}),
    "trench": frozenset(),
}

V7_GEOMETRY_MASS = {
    "foundation": {
        "slab": 0.25,
        "irregular": 0.15,
        "courtyard": 0.15,
        "bearing_walls": 0.20,
        "pads": 0.15,
        "courtyard_pads": 0.10,
    },
    "trench": {
        "straight": 0.15,
        "dogleg": 0.15,
        "tee": 0.20,
        "cross": 0.10,
        "double_t": 0.20,
        "network3": 0.15,
        "disconnected_pair": 0.05,
    },
}

COLORS = np.asarray(
    [
        (240, 227, 194),
        (230, 138, 46),
        (81, 168, 104),
        (158, 163, 168),
        (32, 33, 36),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class V7Condition:
    condition_id: str
    family: str
    geometry: str
    dump_layout: str
    enclosure_class: str
    branch_depth: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _environment_protocol_sha256(path: Path) -> str:
    protocol = _read_json(path)
    embedded = protocol.get("environment_protocol_sha256")
    payload = {
        key: value
        for key, value in protocol.items()
        if key != "environment_protocol_sha256"
    }
    computed = _canonical_json_sha256(payload)
    if embedded != computed:
        raise ValueError(
            "environment protocol canonical hash mismatch: "
            f"embedded={embedded!r}, computed={computed!r}"
        )
    return computed


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        path.chmod(path.stat().st_mode | 0o200)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        path.chmod(path.stat().st_mode | 0o200)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _geometry_rows() -> tuple[tuple[str, str], ...]:
    return tuple(
        [("foundation", geometry) for geometry in FOUNDATION_GEOMETRIES]
        + [("trench", geometry) for geometry in TRENCH_GEOMETRIES]
    )


def v7_conditions() -> tuple[V7Condition, ...]:
    conditions = []
    for dump_layout in V7_LAYOUTS:
        suffix = "adjacent"
        for family, geometry in _geometry_rows():
            enclosure_class = (
                "enclosed" if geometry in ENCLOSED_GEOMETRIES[family] else "open"
            )
            branch_depth = "Nearby core"
            family_token = "fnd" if family == "foundation" else "trn"
            geometry_token = geometry.replace("_", "-")
            conditions.append(
                V7Condition(
                    condition_id=f"v7-{family_token}-{geometry_token}-{suffix}",
                    family=family,
                    geometry=geometry,
                    dump_layout=dump_layout,
                    enclosure_class=enclosure_class,
                    branch_depth=branch_depth,
                )
            )
    return tuple(conditions)


V7_CONDITIONS = v7_conditions()


def exterior_accepted_mask(dig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exterior accepted ground and enclosed neutral staging ground."""
    dig = np.asarray(dig, dtype=np.bool_)
    if dig.shape != (MAP_SIZE, MAP_SIZE) or not dig.any():
        raise ValueError("V7 dig mask must be a nonempty 64x64 mask")

    non_dig = ~dig
    labels, _ = ndi.label(
        non_dig,
        structure=ndi.generate_binary_structure(rank=2, connectivity=1),
    )
    boundary_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    boundary_labels = boundary_labels[boundary_labels != 0]
    exterior = non_dig & np.isin(labels, boundary_labels)
    enclosed = non_dig & ~exterior
    return exterior, enclosed


def adjacent_generous_mask(dig: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Select complete exterior distance rings until capacity is reached."""
    dig = np.asarray(dig, dtype=np.bool_)
    if dig.shape != (MAP_SIZE, MAP_SIZE) or not dig.any():
        raise ValueError("V7 dig mask must be a nonempty 64x64 mask")
    legal, enclosed = exterior_accepted_mask(dig)
    dig_cells = int(dig.sum())
    allfree_cells = int(legal.sum())
    target_cells = min(
        int(math.ceil(8.0 * dig_cells)),
        int(math.floor(0.80 * allfree_cells)),
    )
    if target_cells < dig_cells:
        raise ValueError("V7 geometry leaves less than one-layer adjacent capacity")

    distance = ndi.distance_transform_edt(~dig)
    candidates = np.unique(distance[legal])
    threshold = None
    accepted = None
    for candidate in candidates:
        mask = legal & (distance <= candidate + 1e-12)
        if int(mask.sum()) >= target_cells:
            threshold = float(candidate)
            accepted = mask
            break
    if accepted is None or threshold is None:
        raise RuntimeError("could not construct adjacent generous support")
    accepted_cells = int(accepted.sum())
    return accepted, {
        "capacity_target_cells": target_cells,
        "accepted_dump_cells": accepted_cells,
        "single_layer_capacity_ratio": accepted_cells / dig_cells,
        "allfree_capacity_ratio": allfree_cells / dig_cells,
        "enclosed_staging_cells": int(enclosed.sum()),
        "apron_outer_distance_tiles": threshold,
        "apron_target_rule": (
            "min(8x_dig,80pct_exterior_allfree),complete_distance_ring"
        ),
    }


def arrays_for_scenario(
    scenario: ReviewScenario, dump_layout: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dig = np.asarray(scenario.dig, dtype=np.bool_)
    target = np.zeros(dig.shape, dtype=np.int8)
    target[dig] = -1
    occupancy = np.zeros(dig.shape, dtype=np.bool_)
    _, enclosed = exterior_accepted_mask(dig)
    dumpability = np.ones(dig.shape, dtype=np.bool_)
    action = np.zeros(dig.shape, dtype=np.int8)

    if dump_layout == "adjacent_generous":
        accepted, metrics = adjacent_generous_mask(dig)
        target[accepted] = 1
        arrays = {
            "images": target,
            "occupancy": occupancy,
            "dumpability": dumpability,
            "actions": action,
            "distance": compute_geodesic_distance(target, occupancy).astype(np.float32),
        }
    else:
        raise ValueError(f"V8 only supports adjacent-generous V7 maps: {dump_layout}")

    accepted = arrays["images"] > 0
    if np.any(accepted & dig) or np.any(accepted & arrays["occupancy"]):
        raise RuntimeError("accepted V7 dump mask overlaps dig or occupancy")
    if np.any(accepted & ~arrays["dumpability"]):
        raise RuntimeError("accepted V7 dump mask contains non-dumpable cells")
    if np.any(accepted & enclosed) or not np.all(arrays["dumpability"][enclosed]):
        raise RuntimeError(
            "enclosed V7 interior must be neutral but physically dumpable"
        )
    return arrays, metrics


def _metadata_for(
    scenario: ReviewScenario, condition: V7Condition, metrics: dict[str, Any]
) -> dict[str, Any]:
    axes = scenario.metadata.get("axes_ABC", [])
    if condition.family == "trench" and not 1 <= len(axes) <= 4:
        raise ValueError(
            f"{scenario.scenario_id}: expected one to four trench axes, got {len(axes)}"
        )
    return {
        "schema": "terra_v8_v7_axis_metadata_v4",
        "geometry": condition.geometry,
        "dump_layout": condition.dump_layout,
        "enclosure_class": condition.enclosure_class,
        "trench_axes_count": len(axes) if condition.family == "trench" else -1,
        "trench_topology": (condition.geometry if condition.family == "trench" else ""),
        "axes_ABC": axes,
        "foundation_border_axes_ABC": [],
        "source_scenario_id": scenario.scenario_id,
        **metrics,
    }


def _prepare_exact_dataset(path: Path) -> None:
    for folder in (*RESET_ARRAY_FOLDERS, "metadata"):
        (path / folder).mkdir(parents=True, exist_ok=True)


def _copy_v6_train_tree(
    v6_bank: Path, output: Path, train_entries: list[dict[str, Any]]
) -> None:
    """Copy the frozen V6 input before deriving V8-owned trench rasters."""
    destination = output / "train"
    shutil.copytree(v6_bank / "train", destination)
    destination.chmod(destination.stat().st_mode | 0o200)
    for entry in train_entries:
        level = output / entry["maps_path"]
        level.chmod(level.stat().st_mode | 0o200)
        metadata = level / "dataset.json"
        metadata.chmod(metadata.stat().st_mode | 0o200)


def _save_slot(
    dataset: Path,
    slot: int,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    for folder in RESET_ARRAY_FOLDERS:
        path = dataset / folder / f"img_{slot}.npy"
        if path.exists():
            path.chmod(path.stat().st_mode | 0o200)
        np.save(path, arrays[folder])
    _write_json(dataset / "metadata" / f"trench_{slot}.json", metadata)


def _load_slot(dataset: Path, slot: int) -> dict[str, np.ndarray]:
    return {
        folder: np.load(dataset / folder / f"img_{slot}.npy")
        for folder in RESET_ARRAY_FOLDERS
    }


def _dig_sha256(dig: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(dig).tobytes()).hexdigest()


def _v6_width_mask(
    original_dig: np.ndarray, axes: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    if not axes:
        raise ValueError("a V6 trench must provide at least one axis")
    rows, columns = np.indices(original_dig.shape, dtype=float)
    distances = []
    projections = []
    lengths = []
    for axis in axes:
        a = float(axis["A"])
        b = float(axis["B"])
        c = float(axis["C"])
        denominator = math.hypot(a, b)
        if denominator <= 0.0:
            raise ValueError(f"degenerate trench axis: {axis}")
        distances.append(np.abs(a * columns + b * rows + c) / denominator)
        projections.append((a * rows - b * columns) / denominator)
        lengths.append(denominator)

    distance_stack = np.stack(distances)
    nearest_axis = np.argmin(distance_stack, axis=0)
    dig = np.zeros_like(original_dig)
    for axis_index, (distance, projection, length) in enumerate(
        zip(distances, projections, lengths, strict=True)
    ):
        support = original_dig & (nearest_axis == axis_index)
        if not support.any():
            raise RuntimeError(f"V6 trench axis {axis_index} has no raster support")
        values = projection[support]
        centre = float(values.min() + values.max()) / 2.0
        dig |= (
            original_dig
            & (distance <= TARGET_TRENCH_RADIUS_TILES + 1e-12)
            & (
                np.abs(projection - centre)
                <= length / 2.0 + TARGET_TRENCH_END_PADDING_TILES
            )
        )
    return dig, np.min(distance_stack, axis=0)


def derive_v6_trench_width(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    allfree: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Keep a V6 constraint layout but enforce V8's 1.3 m trench contract.

    V6 sampled nominal 3- or 5-cell corridors. Its axis metadata is exact, so
    intersecting the old finite trench with a 1.3 m band preserves its segment
    endpoints and topology while removing only the excess lateral width.
    """
    original_target = np.asarray(arrays["images"])
    original_dig = original_target < 0
    if original_target.shape != (MAP_SIZE, MAP_SIZE) or not original_dig.any():
        raise ValueError("V6 trench target must be a nonempty 64x64 mask")

    dig, axis_distance = _v6_width_mask(original_dig, metadata.get("axes_ABC", []))
    if not dig.any():
        raise RuntimeError("the V8 trench-width derivation removed the whole trench")
    _, original_components = ndi.label(original_dig)
    _, derived_components = ndi.label(dig)
    if derived_components != original_components:
        raise RuntimeError(
            "the V8 trench-width derivation changed connected-component count: "
            f"{original_components} -> {derived_components}"
        )

    target = original_target.copy()
    target[original_dig & ~dig] = 0
    target[dig] = -1
    occupancy = np.asarray(arrays["occupancy"], dtype=np.bool_)
    dumpability = np.asarray(arrays["dumpability"], dtype=np.bool_)
    if allfree:
        target[~dig & ~occupancy & dumpability] = 1
    accepted = target > 0
    if np.any(accepted & dig) or np.any(accepted & occupancy):
        raise RuntimeError("derived V6 dump mask overlaps dig or occupancy")
    if np.any(accepted & ~dumpability):
        raise RuntimeError("derived V6 dump mask contains non-dumpable cells")

    derived = {
        **arrays,
        "images": target,
        "distance": compute_geodesic_distance(target, occupancy).astype(np.float32),
    }
    metrics = {
        "parent_dig_cells": int(original_dig.sum()),
        "required_dig_volume": int(dig.sum()),
        "target_trench_width_m": TARGET_TRENCH_WIDTH_M,
        "target_trench_width_tiles": TARGET_TRENCH_WIDTH_TILES,
        "maximum_axis_distance_tiles": float(axis_distance[dig].max()),
        "component_count": int(derived_components),
    }
    return derived, metrics


def _replace_scenario_identity(row: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    derived = {**row, "scenario_id": scenario_id}
    if "episode_id" not in row:
        return derived
    reset_seed = row.get("reset_seed")
    protocol_sha256 = row.get("environment_protocol_sha256")
    if not isinstance(reset_seed, int) or isinstance(reset_seed, bool):
        raise ValueError("evaluation row with episode_id must contain reset_seed")
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
        raise ValueError(
            "evaluation row with episode_id must contain a protocol SHA-256"
        )
    derived["episode_id"] = _canonical_json_sha256(
        {
            "schema": "terra_episode_id_v1",
            "scenario_id": scenario_id,
            "reset_seed": reset_seed,
            "environment_protocol_sha256": protocol_sha256,
        }
    )
    return derived


def _v7_reset_seed(split: str, condition_id: str, map_index: int) -> int:
    digest = _canonical_json_sha256(
        {
            "schema": "terra_v8_reset_seed_v1",
            "split": split,
            "condition_id": condition_id,
            "map_index": map_index,
        }
    )
    return int(digest[:8], 16)


def _attach_episode_identity(
    row: dict[str, Any],
    *,
    reset_seed: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    episode_id = _canonical_json_sha256(
        {
            "schema": "terra_episode_id_v1",
            "scenario_id": row["scenario_id"],
            "reset_seed": reset_seed,
            "environment_protocol_sha256": protocol_sha256,
        }
    )
    return {
        **row,
        "reset_seed": reset_seed,
        "environment_protocol_sha256": protocol_sha256,
        "episode_id": episode_id,
    }


def _derive_v6_trench_row(dataset: Path, row: dict[str, Any]) -> dict[str, Any]:
    slot = int(row["slot_index"])
    arrays = _load_slot(dataset, slot)
    metadata_path = dataset / "metadata" / f"trench_{slot}.json"
    metadata = _read_json(metadata_path)
    parent_scenario_id = row["scenario_id"]
    parent_source_id = row["source_id"]
    derived, metrics = derive_v6_trench_width(
        arrays,
        metadata,
        allfree=row["primary_cell"] == "trn-straight-allfree",
    )
    _save_slot(
        dataset,
        slot,
        derived,
        {
            **metadata,
            "schema": "terra_v8_v6_axis_metadata_v4",
            "parent_metadata_schema": metadata.get("schema"),
            "lineage": V6_TRENCH_WIDTH_LINEAGE,
            **metrics,
        },
    )
    dig_sha = _dig_sha256(derived["images"] < 0)
    identity = _replace_scenario_identity(
        row,
        reset_array_scenario_sha256(derived),
    )
    return {
        **identity,
        "source_id": f"dig:{dig_sha}",
        "dig_sha256": dig_sha,
        "required_dig_volume": metrics["required_dig_volume"],
        "parent_scenario_id": parent_scenario_id,
        "parent_source_id": parent_source_id,
        "lineage": V6_TRENCH_WIDTH_LINEAGE,
        "target_trench_width_m": TARGET_TRENCH_WIDTH_M,
        "target_trench_width_tiles": TARGET_TRENCH_WIDTH_TILES,
    }


def _derive_v6_dataset(dataset: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(dataset / "manifest.jsonl")
    derived_rows = [
        _derive_v6_trench_row(dataset, row) if row["family"] == "trench" else row
        for row in rows
    ]
    _write_jsonl(dataset / "manifest.jsonl", derived_rows)
    return derived_rows


def _source_id(scenario: ReviewScenario) -> str:
    return f"v7-dig:{_dig_sha256(scenario.dig)}"


def _record_for(
    *,
    scenario: ReviewScenario,
    condition: V7Condition,
    split: str,
    map_index: int,
    slot: int,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    scenario_id = reset_array_scenario_sha256(arrays)
    source_id = _source_id(scenario)
    return {
        "slot_index": slot,
        "map_id": (f"v8:{split}:{condition.condition_id}:{map_index:04d}"),
        "scenario_id": scenario_id,
        "source_id": source_id,
        "split": split,
        "family": condition.family,
        "stratum": "v8_v7_geometry_core",
        "primary_cell": condition.condition_id,
        "slot_weight": 1.0,
        "identity_slot_multiplicity": 1,
        "pair_slot_id": source_id,
        "source_scenario_id": scenario.scenario_id,
        "geometry": condition.geometry,
        "dump_layout": condition.dump_layout,
        "enclosure_class": condition.enclosure_class,
        "dig_sha256": source_id.removeprefix("v7-dig:"),
        "required_dig_volume": int(scenario.dig.sum()),
        **(
            {
                "target_trench_width_m": TARGET_TRENCH_WIDTH_M,
                "target_trench_width_tiles": TARGET_TRENCH_WIDTH_TILES,
            }
            if condition.family == "trench"
            else {}
        ),
        **metrics,
    }


def _registry_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "map_id": row["map_id"],
        "scenario_id": row["scenario_id"],
        "source_id": row["source_id"],
        "split": row["split"],
        "family": row["family"],
        "primary_cell": row["primary_cell"],
    }


def _exact_metadata(
    *,
    count: int,
    registry_sha256: str,
    registry_relative_path: str,
    lineage: str,
    included_in_main_macro: bool,
) -> dict[str, Any]:
    return {
        "schema": EXACT_DATASET_SCHEMA,
        "slot_count": count,
        "unique_identity_count": count,
        "shape": [MAP_SIZE, MAP_SIZE],
        "distance_metric": DISTANCE_METRIC,
        "distance_normalization": DISTANCE_NORMALIZATION,
        "accepted_dump_contract": ACCEPTED_DUMP_CONTRACT,
        "scenario_identity_contract": RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
        "source_registry": registry_relative_path,
        "source_registry_sha256": registry_sha256,
        "lineage": lineage,
        "included_in_main_macro": included_in_main_macro,
    }


def _group_scenarios(
    scenarios: list[ReviewScenario], count: int
) -> dict[tuple[str, str], list[ReviewScenario]]:
    grouped: dict[tuple[str, str], list[ReviewScenario]] = {}
    for family, geometry in _geometry_rows():
        selected = [
            scenario
            for scenario in scenarios
            if scenario.family == family and scenario.geometry == geometry
        ]
        if len(selected) != count:
            raise RuntimeError(
                f"{family}/{geometry}: generated {len(selected)}, expected {count}"
            )
        grouped[(family, geometry)] = selected
    return grouped


def _register_v7_split_sources(
    split: str,
    scenarios: list[ReviewScenario],
    source_split: dict[str, str],
) -> None:
    for scenario in scenarios:
        source_id = _source_id(scenario)
        previous = source_split.get(source_id)
        if previous is not None and previous != split:
            raise RuntimeError(
                f"V7 source crosses splits: {source_id} in {previous} and {split}"
            )
        source_split[source_id] = split


def _write_v7_train_levels(
    output: Path,
    grouped: dict[tuple[str, str], list[ReviewScenario]],
    start_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    train_entries = []
    registry_rows = []
    datasets = []
    for offset, condition in enumerate(V7_CONDITIONS):
        level_index = start_index + offset
        directory = output / "train" / f"{level_index:03d}__{condition.condition_id}"
        _prepare_exact_dataset(directory)
        rows = []
        for map_index, scenario in enumerate(
            grouped[(condition.family, condition.geometry)]
        ):
            arrays, metrics = arrays_for_scenario(scenario, condition.dump_layout)
            slot = map_index + 1
            _save_slot(
                directory,
                slot,
                arrays,
                _metadata_for(scenario, condition, metrics),
            )
            row = _record_for(
                scenario=scenario,
                condition=condition,
                split="train",
                map_index=map_index,
                slot=slot,
                arrays=arrays,
                metrics=metrics,
            )
            rows.append(row)
            registry_rows.append(_registry_row(row))
        _write_jsonl(directory / "manifest.jsonl", rows)
        datasets.append(directory)
        train_entries.append(
            {
                "condition_id": condition.condition_id,
                "family": condition.family,
                "branch_depth": condition.branch_depth,
                "level_index": level_index,
                "map_count": SPLIT_COUNTS["train"],
                "maps_path": directory.relative_to(output).as_posix(),
                "lineage": "v7_geometry_adjacent_generous",
                "geometry": condition.geometry,
                "dump_layout": condition.dump_layout,
                "enclosure_class": condition.enclosure_class,
            }
        )
    return train_entries, registry_rows, datasets


def _copy_slot(source: Path, source_slot: int, destination: Path, slot: int) -> None:
    for folder in RESET_ARRAY_FOLDERS:
        shutil.copy2(
            source / folder / f"img_{source_slot}.npy",
            destination / folder / f"img_{slot}.npy",
        )
    shutil.copy2(
        source / "metadata" / f"trench_{source_slot}.json",
        destination / "metadata" / f"trench_{slot}.json",
    )


def _write_main_evaluation_panel(
    *,
    v6_bank: Path,
    v6_index: dict[str, Any],
    output: Path,
    split: str,
    grouped: dict[tuple[str, str], list[ReviewScenario]],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    panel = v6_index["evaluation_panels"][split]
    source = v6_bank / panel["maps_path"]
    source_rows = _read_jsonl(source / "manifest.jsonl")
    destination = output / "evaluation" / "main" / split
    _prepare_exact_dataset(destination)
    protocol_sha256 = _environment_protocol_sha256(output / "environment_protocol.json")

    rows = []
    registry_rows = []
    for source_row in source_rows:
        slot = len(rows) + 1
        _copy_slot(source, int(source_row["slot_index"]), destination, slot)
        row = {**source_row, "slot_index": slot}
        if row["family"] == "trench":
            row = _derive_v6_trench_row(destination, row)
        rows.append(row)
        registry_rows.append(_registry_row(row))

    for condition in V7_CONDITIONS:
        for map_index, scenario in enumerate(
            grouped[(condition.family, condition.geometry)]
        ):
            arrays, metrics = arrays_for_scenario(scenario, condition.dump_layout)
            slot = len(rows) + 1
            _save_slot(
                destination,
                slot,
                arrays,
                _metadata_for(scenario, condition, metrics),
            )
            row = _record_for(
                scenario=scenario,
                condition=condition,
                split=split,
                map_index=map_index,
                slot=slot,
                arrays=arrays,
                metrics=metrics,
            )
            row = _attach_episode_identity(
                row,
                reset_seed=_v7_reset_seed(
                    split,
                    condition.condition_id,
                    map_index,
                ),
                protocol_sha256=protocol_sha256,
            )
            rows.append(row)
            registry_rows.append(_registry_row(row))

    _write_jsonl(destination / "manifest.jsonl", rows)
    condition_count = V6_CONSTRAINED_COUNT + len(V7_CONDITIONS)
    expected = condition_count * SPLIT_COUNTS[split]
    if len(rows) != expected:
        raise RuntimeError(
            f"{split}: wrote {len(rows)} evaluation maps, expected {expected}"
        )
    return (
        {
            "conditions": condition_count,
            "maps_path": destination.relative_to(output).as_posix(),
            "slot_count": len(rows),
        },
        registry_rows,
        destination,
    )


def _validate_registry(rows: list[dict[str, Any]]) -> None:
    by_map: dict[str, tuple[str, str, str]] = {}
    source_splits: dict[str, set[str]] = {}
    for row in rows:
        identity = (row["source_id"], row["split"], row["scenario_id"])
        previous = by_map.get(row["map_id"])
        if previous is not None and previous != identity:
            raise RuntimeError(f"conflicting registry identity for {row['map_id']}")
        by_map[row["map_id"]] = identity
        source_splits.setdefault(row["source_id"], set()).add(row["split"])
    crossing = {
        source_id: sorted(splits)
        for source_id, splits in source_splits.items()
        if len(splits) > 1
    }
    if crossing:
        source_id, splits = next(iter(crossing.items()))
        raise RuntimeError(f"source {source_id} crosses splits {splits}")


def _patch_v6_training_metadata(
    output: Path, train_entries: list[dict[str, Any]], registry_sha256: str
) -> list[Path]:
    datasets = []
    for entry in train_entries:
        dataset = output / entry["maps_path"]
        metadata = _read_json(dataset / "dataset.json")
        metadata["source_registry"] = "../../source_registry.jsonl"
        metadata["source_registry_sha256"] = registry_sha256
        metadata["lineage"] = entry.get("lineage", "v6_train96")
        _write_json(dataset / "dataset.json", metadata)
        datasets.append(dataset)
    return datasets


def _write_v7_dataset_metadata(datasets: list[Path], registry_sha256: str) -> None:
    for dataset in datasets:
        _write_json(
            dataset / "dataset.json",
            _exact_metadata(
                count=SPLIT_COUNTS["train"],
                registry_sha256=registry_sha256,
                registry_relative_path="../../source_registry.jsonl",
                lineage="v7_geometry_adjacent_generous",
                included_in_main_macro=True,
            ),
        )


def _write_panel_metadata(panels: list[Path], split: str, registry_sha256: str) -> None:
    count = (V6_CONSTRAINED_COUNT + len(V7_CONDITIONS)) * SPLIT_COUNTS[split]
    for panel in panels:
        _write_json(
            panel / "dataset.json",
            _exact_metadata(
                count=count,
                registry_sha256=registry_sha256,
                registry_relative_path="../../../source_registry.jsonl",
                lineage="v6_constraints_width_1p3m_plus_v7_adjacent_geometry",
                included_in_main_macro=True,
            ),
        )


def _patch_capability_panel_metadata(
    panel_paths: list[Path], registry_sha256: str
) -> None:
    for panel in panel_paths:
        metadata = _read_json(panel / "dataset.json")
        metadata["source_registry"] = "../../../source_registry.jsonl"
        metadata["source_registry_sha256"] = registry_sha256
        metadata["lineage"] = "v6_capability_floor_v8_width_1p3m"
        _write_json(panel / "dataset.json", metadata)


def _audit_built_datasets(
    datasets: list[Path],
    protocol_sha256: str,
) -> dict[str, Any]:
    source_splits: dict[str, set[str]] = {}
    scenario_count = 0
    derived_v6_trenches = 0
    v7_scenarios = 0
    enclosed_staging_cells = 0
    maximum_axis_distance = 0.0
    for dataset in datasets:
        for row in _read_jsonl(dataset / "manifest.jsonl"):
            slot = int(row["slot_index"])
            arrays = _load_slot(dataset, slot)
            target = arrays["images"]
            dig = target < 0
            accepted = target > 0
            occupancy = arrays["occupancy"].astype(np.bool_)
            dumpability = arrays["dumpability"].astype(np.bool_)
            scenario_count += 1
            if row.get("split") != "train":
                reset_seed = row.get("reset_seed")
                if not isinstance(reset_seed, int) or isinstance(reset_seed, bool):
                    raise RuntimeError(f"{row['map_id']}: invalid reset seed")
                if row.get("environment_protocol_sha256") != protocol_sha256:
                    raise RuntimeError(f"{row['map_id']}: protocol identity changed")
                expected_episode_id = _canonical_json_sha256(
                    {
                        "schema": "terra_episode_id_v1",
                        "scenario_id": row["scenario_id"],
                        "reset_seed": reset_seed,
                        "environment_protocol_sha256": protocol_sha256,
                    }
                )
                if row.get("episode_id") != expected_episode_id:
                    raise RuntimeError(f"{row['map_id']}: invalid episode identity")
            dig_sha = _dig_sha256(dig)
            source_splits.setdefault(dig_sha, set()).add(str(row["split"]))
            if (
                np.any(accepted & dig)
                or np.any(accepted & occupancy)
                or np.any(accepted & ~dumpability)
            ):
                raise RuntimeError(f"{row['map_id']}: illegal accepted dump cell")

            if row.get("lineage") == V6_TRENCH_WIDTH_LINEAGE:
                metadata = _read_json(dataset / "metadata" / f"trench_{slot}.json")
                maximum_axis_distance = max(
                    maximum_axis_distance,
                    float(metadata["maximum_axis_distance_tiles"]),
                )
                _, components = ndi.label(dig)
                if int(metadata["component_count"]) != int(components):
                    raise RuntimeError(
                        f"{row['map_id']}: derived trench topology changed"
                    )
                derived_v6_trenches += 1

            if str(row.get("stratum", "")).startswith("v8_v7"):
                if int(accepted.sum()) < int(row["capacity_target_cells"]):
                    raise RuntimeError(f"{row['map_id']}: adjacent capacity failed")
                _, enclosed = exterior_accepted_mask(dig)
                if np.any(accepted & enclosed) or not np.all(dumpability[enclosed]):
                    raise RuntimeError(
                        f"{row['map_id']}: invalid enclosed staging semantics"
                    )
                enclosed_staging_cells += int(enclosed.sum())
                v7_scenarios += 1

    crossing = {
        source_id: sorted(splits)
        for source_id, splits in source_splits.items()
        if len(splits) > 1
    }
    if crossing:
        source_id, splits = next(iter(crossing.items()))
        raise RuntimeError(f"raw dig hash {source_id} crosses splits {splits}")
    if maximum_axis_distance > TARGET_TRENCH_RADIUS_TILES + 1e-12:
        raise RuntimeError(
            "derived V6 trench exceeds the 1.3 m width radius: "
            f"{maximum_axis_distance} > {TARGET_TRENCH_RADIUS_TILES}"
        )
    return {
        "schema": "terra_v8_combined_bank_audit_v4",
        "scenario_rows_audited": scenario_count,
        "v6_derived_trench_scenarios": derived_v6_trenches,
        "v7_scenarios": v7_scenarios,
        "v7_enclosed_staging_cells": enclosed_staging_cells,
        "maximum_derived_axis_distance_tiles": maximum_axis_distance,
        "maximum_allowed_axis_distance_tiles": TARGET_TRENCH_RADIUS_TILES,
        "component_mismatches": 0,
        "illegal_accepted_dump_scenarios": 0,
        "raw_cross_split_dig_hashes": 0,
    }


def _render_code(arrays: dict[str, np.ndarray]) -> Image.Image:
    target = arrays["images"]
    occupied = arrays["occupancy"].astype(np.bool_)
    dumpable = arrays["dumpability"].astype(np.bool_)
    code = np.zeros(target.shape, dtype=np.uint8)
    code[target < 0] = 1
    code[target > 0] = 2
    code[~dumpable & ~occupied] = 3
    code[occupied] = 4
    return Image.fromarray(COLORS[code], mode="RGB").resize(
        (256, 256), Image.Resampling.NEAREST
    )


def _render_review(
    output: Path, grouped: dict[tuple[str, str], list[ReviewScenario]]
) -> None:
    review = output / "review"
    review.mkdir()
    index_rows = []
    links = []
    for family, geometry in _geometry_rows():
        scenarios = grouped[(family, geometry)]
        positions = [index * (len(scenarios) - 1) // 5 for index in range(6)]
        canvas = Image.new("RGB", (256, 38 + 6 * 286), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 8),
            f"{family} / {geometry}: adjacent generous",
            fill="black",
        )
        for row_index, position in enumerate(positions):
            scenario = scenarios[position]
            adjacent, adjacent_metrics = arrays_for_scenario(
                scenario, "adjacent_generous"
            )
            y = 38 + row_index * 286
            canvas.paste(_render_code(adjacent), (0, y))
            draw.text(
                (8, y + 260),
                (
                    f"{scenario.scenario_id}  dig={int(scenario.dig.sum())}  "
                    f"cap={adjacent_metrics['single_layer_capacity_ratio']:.1f}x"
                ),
                fill="black",
            )
            index_rows.append(
                {
                    "family": family,
                    "geometry": geometry,
                    "source_scenario_id": scenario.scenario_id,
                    "source_id": _source_id(scenario),
                    "dig_cells": int(scenario.dig.sum()),
                    "adjacent_capacity_ratio": adjacent_metrics[
                        "single_layer_capacity_ratio"
                    ],
                    "adjacent_outer_distance_tiles": adjacent_metrics[
                        "apron_outer_distance_tiles"
                    ],
                    "enclosed_staging_cells": adjacent_metrics[
                        "enclosed_staging_cells"
                    ],
                }
            )
        filename = f"{family}__{geometry}.png"
        canvas.save(review / filename)
        links.append(f"- [{family} / {geometry}]({filename})")

    _write_jsonl(review / "index.jsonl", index_rows)
    (review / "README.md").write_text(
        "# Terra V8 adjacent core review\n\n"
        "Orange is excavation and green is accepted dumping. V7 all-free "
        "duplicates were removed because adjacent-generous already provides "
        "the intended easy local support. Closed foundation interiors remain "
        "neutral beige staging ground: temporary spilling is legal there, but "
        "soil left inside prevents exact completion. The V6 constraint layouts "
        "are included in the local site, with every V6 trench re-rasterized at "
        "the shared 1.3 m (2.275 tile) V8 width contract. The original V6 "
        "release remains unchanged.\n\n"
        "The two V6 all-free controls remain separate capability diagnostics, "
        "not duplicate V7 training cells.\n\n" + "\n".join(links) + "\n"
    )


def _training_mixture(v6_index: dict[str, Any]) -> dict[str, Any]:
    v7_ids = [condition.condition_id for condition in V7_CONDITIONS]
    anchors = {
        condition.condition_id: (
            "fnd-slab-allfree"
            if condition.family == "foundation"
            else "trn-straight-allfree"
        )
        for condition in V7_CONDITIONS
    }
    return {
        "schema": MIXTURE_SCHEMA,
        "family_balance": {"foundation": 0.5, "trench": 0.5},
        "v7_geometry_mass_within_family": V7_GEOMETRY_MASS,
        "slot_weights_are_sampler_weights": False,
        "stages": [
            {
                "name": "capability_anchors",
                "new_conditions": list(V6_CAPABILITY_FLOOR_IDS),
                "previous_stage_replay_fraction": 0.0,
            },
            {
                "name": "nearby_geometry_core",
                "new_conditions": v7_ids,
                "requires": anchors,
                "previous_stage_replay_fraction": 0.5,
                "unlock_rule": (
                    "per condition after its family capability anchor; geometry "
                    "siblings are not a total order"
                ),
            },
            {
                "name": "constraint_branches",
                "new_conditions": v6_index["constrained_condition_ids"],
                "previous_stage_replay_fraction": 0.5,
                "unlock_rule": "per V6 branch; no global total ordering",
            },
        ],
        "fixed_protocol": {
            "max_steps_in_episode": 450,
            "rewards_type": "DENSE",
            "apply_trench_rewards": False,
            "full_resets": True,
            "accepted_dump_contract": ACCEPTED_DUMP_CONTRACT,
        },
        "v7_dump_support": (
            "adjacent_generous_only; V7 allfree removed as redundant; V6 allfree "
            "controls retained as separate capability diagnostics"
        ),
    }


def build(v6_bank: Path, output: Path) -> dict[str, Any]:
    v6_bank = v6_bank.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if _sha256_file(v6_bank / "dataset.json") != V6_DATASET_SHA256:
        raise ValueError("V6 input dataset.json hash mismatch")
    v6_index = _read_json(v6_bank / "dataset.json")
    if v6_index.get("release_id") != V6_RELEASE_ID:
        raise ValueError("input is not the frozen V6 Train-96 release")
    if len(v6_index.get("train", [])) != 34:
        raise ValueError("V6 input must contain 34 training conditions")

    output.mkdir(parents=True)
    v6_train_entries = [dict(entry) for entry in v6_index["train"]]
    _copy_v6_train_tree(v6_bank, output, v6_train_entries)
    shutil.copytree(
        v6_bank / "evaluation" / "capability_floor",
        output / "evaluation" / "capability_floor",
    )
    shutil.copy2(v6_bank / "environment_protocol.json", output)
    shutil.copy2(v6_bank / "review_admission.json", output / "v6_review_admission.json")

    registry_rows: list[dict[str, Any]] = []
    for entry in v6_train_entries:
        dataset = output / entry["maps_path"]
        rows = _derive_v6_dataset(dataset)
        registry_rows.extend(_registry_row(row) for row in rows)
        if entry["family"] == "trench":
            entry["lineage"] = V6_TRENCH_WIDTH_LINEAGE
            entry["target_trench_width_m"] = TARGET_TRENCH_WIDTH_M
            entry["target_trench_width_tiles"] = TARGET_TRENCH_WIDTH_TILES

    capability_panel_paths = []
    for split in EVALUATION_SPLITS:
        panel = v6_index["capability_floor_evaluation_panels"][split]
        path = output / panel["maps_path"]
        rows = _derive_v6_dataset(path)
        registry_rows.extend(_registry_row(row) for row in rows)
        capability_panel_paths.append(path)

    source_split: dict[str, str] = {}

    train_scenarios = generate_scenarios(SPLIT_COUNTS["train"], SPLIT_SEEDS["train"])
    _register_v7_split_sources("train", train_scenarios, source_split)
    train_grouped = _group_scenarios(train_scenarios, SPLIT_COUNTS["train"])
    v7_train_entries, new_registry, v7_train_datasets = _write_v7_train_levels(
        output, train_grouped, len(v6_train_entries)
    )
    registry_rows.extend(new_registry)

    evaluation_panels = {}
    panel_paths = []
    development_grouped = None
    for split in EVALUATION_SPLITS:
        scenarios = generate_scenarios(SPLIT_COUNTS[split], SPLIT_SEEDS[split])
        _register_v7_split_sources(split, scenarios, source_split)
        grouped = _group_scenarios(scenarios, SPLIT_COUNTS[split])
        if split == "development":
            development_grouped = grouped
        panel, new_registry, panel_path = _write_main_evaluation_panel(
            v6_bank=v6_bank,
            v6_index=v6_index,
            output=output,
            split=split,
            grouped=grouped,
        )
        evaluation_panels[split] = panel
        registry_rows.extend(new_registry)
        panel_paths.append(panel_path)

    registry_rows.sort(
        key=lambda row: (
            row["split"],
            row["family"],
            row["primary_cell"],
            row["map_id"],
        )
    )
    _validate_registry(registry_rows)
    _write_jsonl(output / "source_registry.jsonl", registry_rows)
    registry_sha256 = _sha256_file(output / "source_registry.jsonl")

    v6_train_datasets = _patch_v6_training_metadata(
        output, v6_train_entries, registry_sha256
    )
    _write_v7_dataset_metadata(v7_train_datasets, registry_sha256)
    for split, panel_path in zip(EVALUATION_SPLITS, panel_paths, strict=True):
        _write_panel_metadata([panel_path], split, registry_sha256)
    _patch_capability_panel_metadata(capability_panel_paths, registry_sha256)

    if development_grouped is None:
        raise RuntimeError("development split was not generated")
    _render_review(output, development_grouped)

    mixture = _training_mixture(v6_index)
    _write_json(output / "training_mixture.json", mixture)
    mixture_sha256 = _sha256_file(output / "training_mixture.json")
    _write_json(
        output / "review_status.json",
        {
            "schema": V8_REVIEW_SCHEMA,
            "status": "pending_v8_visual_review",
            "v6_review": (
                "constraint layouts preserved; trench targets derived at the V8 "
                "1.3 m width contract"
            ),
            "v7_geometry_review": "accepted design input",
            "pending": "V7 adjacent-generous and corrected-width V6 scenarios",
        },
    )

    all_train_entries = v6_train_entries + v7_train_entries
    v7_ids = [condition.condition_id for condition in V7_CONDITIONS]
    main_ids = sorted(v6_index["constrained_condition_ids"] + v7_ids)
    protocol_sha256 = _environment_protocol_sha256(output / "environment_protocol.json")
    audit = _audit_built_datasets(
        v6_train_datasets + v7_train_datasets + panel_paths + capability_panel_paths,
        protocol_sha256,
    )
    _write_json(output / "audit_receipt.json", audit)
    root_index = {
        "schema": LOADER_BANK_SCHEMA,
        "release_id": RELEASE_ID,
        "release_name": RELEASE_NAME,
        "status": "candidate_pending_review",
        "shape": [MAP_SIZE, MAP_SIZE],
        "scenario_identity_contract": RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
        "train_maps_per_condition": SPLIT_COUNTS["train"],
        "train": all_train_entries,
        "source_registry": "source_registry.jsonl",
        "source_registry_sha256": registry_sha256,
        "environment_protocol": "environment_protocol.json",
        "environment_protocol_sha256": protocol_sha256,
        "training_mixture": "training_mixture.json",
        "training_mixture_sha256": mixture_sha256,
        "review_status": "review_status.json",
        "audit_receipt": "audit_receipt.json",
        "audit_receipt_sha256": _sha256_file(output / "audit_receipt.json"),
        "v6_constraint_condition_ids": v6_index["constrained_condition_ids"],
        "v6_capability_floor_condition_ids": list(V6_CAPABILITY_FLOOR_IDS),
        "v7_core_condition_ids": v7_ids,
        "included_in_main_macro": main_ids,
        "evaluation_panels": evaluation_panels,
        "capability_floor_evaluation_panels": v6_index[
            "capability_floor_evaluation_panels"
        ],
        "v6_input": {
            "release_id": V6_RELEASE_ID,
            "dataset_sha256": V6_DATASET_SHA256,
            "foundation_arrays": "copied",
            "trench_constraint_layouts": "preserved",
            "trench_dig_derivation": V6_TRENCH_WIDTH_LINEAGE,
            "target_trench_width_m": TARGET_TRENCH_WIDTH_M,
            "target_trench_width_tiles": TARGET_TRENCH_WIDTH_TILES,
        },
        "v7_input": {
            "generator": "tools/map_generation/generate_v7_geometry_review.py",
            "geometry_count": len(_geometry_rows()),
            "dump_layouts": list(V7_LAYOUTS),
            "split_seeds": SPLIT_SEEDS,
        },
    }
    _write_json(output / "dataset.json", root_index)

    for dataset in v6_train_datasets + v7_train_datasets:
        validate_exact_dataset_contract(dataset, SPLIT_COUNTS["train"])
    for split, panel_path in zip(EVALUATION_SPLITS, panel_paths, strict=True):
        validate_exact_dataset_contract(
            panel_path,
            (V6_CONSTRAINED_COUNT + len(V7_CONDITIONS)) * SPLIT_COUNTS[split],
        )
    for split, panel_path in zip(
        EVALUATION_SPLITS, capability_panel_paths, strict=True
    ):
        validate_exact_dataset_contract(
            panel_path,
            len(V6_CAPABILITY_FLOOR_IDS) * SPLIT_COUNTS[split],
        )

    receipt = {
        "schema": BUILD_SCHEMA,
        "release_id": RELEASE_ID,
        "v6_dataset_sha256": V6_DATASET_SHA256,
        "output_dataset_sha256": _sha256_file(output / "dataset.json"),
        "source_registry_sha256": registry_sha256,
        "train_conditions": len(all_train_entries),
        "train_maps_per_condition": SPLIT_COUNTS["train"],
        "v6_conditions": len(v6_train_entries),
        "v7_conditions": len(v7_train_entries),
        "main_macro_conditions": len(main_ids),
        "evaluation_panels": evaluation_panels,
        "tile_size_m": TILE_SIZE_M,
        "target_trench_width_m": TARGET_TRENCH_WIDTH_M,
        "target_trench_width_tiles": TARGET_TRENCH_WIDTH_TILES,
        "v6_trench_derivation": V6_TRENCH_WIDTH_LINEAGE,
        "audit": audit,
    }
    _write_json(output / "build_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.v6_bank, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
