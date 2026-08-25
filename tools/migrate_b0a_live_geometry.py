#!/usr/bin/env python3
"""Normalize the frozen B0a design bank against live Terra geometry.

This is deliberately a migration receipt, not a benchmark-bank builder.  It
keeps the legacy rasters unchanged, recomputes the inexpensive validator-owned
facts, and materializes one portable initial excavator state per source group.
Exact action-reachable workspace and direct-service validation remain explicit
admission blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import jax
import numpy as np
from scipy import ndimage as ndi

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from terra.benchmark_protocol import BENCHMARK_MAP_SIZE
from terra.benchmark_protocol import BENCHMARK_RELEASE_ID
from terra.benchmark_protocol import frozen_benchmark_protocol
from terra.benchmark_protocol import frozen_environment_protocol
from terra.benchmark_state import agent_to_record
from terra.benchmark_state import sample_benchmark_initial_agent
from terra.benchmark_state import validate_benchmark_initial_agent
from terra.maps_buffer import contained_dump_capacity_sanity_check
from terra.maps_buffer import load_maps_from_disk
from terra.maps_buffer import validate_exact_dataset_contract

MIGRATION_SCHEMA = "terra_b0a_live_geometry_migration_v1"
SUMMARY_SCHEMA = "terra_b0a_live_geometry_migration_summary_v1"
CONDITION_STATUS = "legacy_design_input_not_in_frozen_s2_registry"
MIGRATION_STATUS = "pending_exact_static"
DIRECT_SERVICE_STATUS = "direct_service_blocked_by_cost_gate"

EXPECTED_IDENTITY_COUNT = 256
EXPECTED_SOURCE_GROUP_COUNT = 144
EXPECTED_SOURCE_GROUP_SIZE_COUNTS = {1: 112, 4: 16, 5: 16}
EXPECTED_B0A_FILES_SHA256 = (
    "89a5b5325e4e6872f7899b087ac5d0a8cd444dac30315feee4f342f8e532a347"
)

SPLIT_MAP = {
    "train": "public_train",
    "development": "public_dev",
}
GEOMETRY_MAP = {
    "foundation_osm": ("osm", "connected"),
    "foundation_procedural": ("procedural", "connected"),
    "trench_straight": ("procedural", "trench"),
    "trench_segmented2": ("procedural", "trench"),
    "trench_segmented3": ("procedural", "trench"),
    "trench_T": ("procedural", "trench"),
    "trench_X": ("procedural", "trench"),
    "trench_disconnected": ("procedural", "trench"),
}
TOPOLOGY_MAP = {
    None: None,
    "straight": "straight",
    "segmented_end_to_end_2": "segmented_2",
    "segmented_end_to_end_3": "segmented_3",
    "T": "T",
    "X": "X",
    "disconnected_2": "disconnected",
}
DUMP_LAYOUT_MAP = {
    "all_around": "all_around",
    "broad_apron": "apron",
    "broad_side_cast": "side_cast",
}
SIDE_ACCESS = {"all", "both", "one", "per_segment"}

EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)
FOUR_CONNECTED = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=np.uint8,
)
SHORTEST_PATH_MOVES = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class VerifiedInputIntegrity:
    receipt: dict[str, Any]
    verified_relative_paths: frozenset[str]


@dataclass(frozen=True)
class LoadedLegacyScenario:
    """One legacy identity joined to the arrays consumed by the exact loader."""

    identity: dict[str, Any]
    metadata: dict[str, Any]
    target: np.ndarray
    occupancy: np.ndarray
    dumpability: np.ndarray
    initial_soil: np.ndarray
    reward_distance: np.ndarray
    dataset_directory: Path | None = None
    slot_index: int | None = None
    source_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class MaterializedState:
    state_record: dict[str, Any]
    state_sha256: str
    seed_receipt: dict[str, Any]


StateMaterializer = Callable[
    [str, str, list[LoadedLegacyScenario], Any],
    MaterializedState,
]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_integrity(
    input_root: Path,
    *,
    expected_files_sha256: str = EXPECTED_B0A_FILES_SHA256,
) -> VerifiedInputIntegrity:
    """Verify the frozen B0a checksum tree before any dataset is loaded."""

    input_root = input_root.resolve()
    manifest_path = input_root / "files.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_files_sha256:
        raise ValueError(
            "Frozen B0a files.sha256 changed: "
            f"{manifest_sha256} != {expected_files_sha256}."
        )

    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text().splitlines(),
        start=1,
    ):
        parts = line.split("  ", maxsplit=1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
        ):
            raise ValueError(
                f"Malformed checksum entry at {manifest_path}:{line_number}."
            )
        expected_sha256, relative_text = parts
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_text
        ):
            raise ValueError(
                f"Unsafe checksum path at {manifest_path}:{line_number}: "
                f"{relative_text!r}."
            )
        if relative_text in entries:
            raise ValueError(
                f"Duplicate checksum path in {manifest_path}: {relative_text}."
            )
        entries[relative_text] = expected_sha256

    required_paths = {
        "identities.jsonl",
        "provenance.json",
        "source_registry.jsonl",
        "validation.json",
    }
    missing_required = sorted(required_paths - set(entries))
    if missing_required:
        raise ValueError(
            f"Frozen checksum manifest omits required files: {missing_required}."
        )

    for relative_text, expected_sha256 in sorted(entries.items()):
        path = input_root / relative_text
        resolved = path.resolve()
        if not resolved.is_relative_to(input_root) or not path.is_file():
            raise ValueError(
                f"Checksum target is missing or leaves input root: {relative_text}."
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Frozen B0a checksum mismatch for {relative_text}: "
                f"{actual_sha256} != {expected_sha256}."
            )

    provenance = _read_json(input_root / "provenance.json")
    identity_sha256 = sha256_file(input_root / "identities.jsonl")
    source_registry_sha256 = sha256_file(input_root / "source_registry.jsonl")
    if provenance.get("identity_manifest_sha256") != identity_sha256:
        raise ValueError(
            "provenance.json identity_manifest_sha256 does not match "
            "identities.jsonl."
        )
    if provenance.get("source_registry_sha256") != source_registry_sha256:
        raise ValueError(
            "provenance.json source_registry_sha256 does not match "
            "source_registry.jsonl."
        )
    if entries["identities.jsonl"] != identity_sha256:
        raise ValueError("files.sha256 and provenance disagree on identities.jsonl.")
    if entries["source_registry.jsonl"] != source_registry_sha256:
        raise ValueError(
            "files.sha256 and provenance disagree on source_registry.jsonl."
        )
    return VerifiedInputIntegrity(
        receipt={
            "files_sha256_sha256": manifest_sha256,
            "verified_file_count": len(entries),
            "identity_manifest_sha256": identity_sha256,
            "source_registry_sha256": source_registry_sha256,
            "provenance_sha256": entries["provenance.json"],
            "validation_sha256": entries["validation.json"],
        },
        verified_relative_paths=frozenset(entries),
    )


def sha256_array(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _content_id(kind: str, payload: Mapping[str, Any]) -> str:
    return f"{kind}:sha256:{canonical_json_sha256(payload)}"


def _require_token(mapping: Mapping[Any, Any], value: Any, field: str) -> Any:
    if value not in mapping:
        raise ValueError(f"Unsupported legacy {field}: {value!r}.")
    return mapping[value]


def normalize_factor_vector(
    identity: Mapping[str, Any],
    *,
    separation_p50_tiles: float,
    single_layer_area_ratio: float,
    required_volume: int,
) -> dict[str, Any]:
    """Normalize legacy display tokens without claiming S2 cell membership."""

    family = identity.get("family")
    if family not in {"foundation", "trench"}:
        raise ValueError(f"Unsupported legacy family: {family!r}.")
    source_family, geometry_class = _require_token(
        GEOMETRY_MAP,
        identity.get("geometry"),
        "geometry",
    )
    topology = _require_token(
        TOPOLOGY_MAP,
        identity.get("topology"),
        "topology",
    )
    dump_layout = _require_token(
        DUMP_LAYOUT_MAP,
        identity.get("dump_layout"),
        "dump_layout",
    )
    side_access = identity.get("side_access")
    if side_access not in SIDE_ACCESS:
        raise ValueError(f"Unsupported legacy side_access: {side_access!r}.")

    distance_center = identity.get("distance_center_tiles")
    separation_token = (
        "sep00_02" if distance_center is None else f"sep{int(distance_center):02d}"
    )
    if 3.0 <= single_layer_area_ratio <= 4.0:
        capacity_token = "slcap03_04"
    elif 7.0 <= single_layer_area_ratio <= 10.0:
        capacity_token = "slcap07_10"
    elif 20.0 <= single_layer_area_ratio <= 45.0:
        capacity_token = "slcap20_45"
    else:
        capacity_token = "legacy_outside_frozen_pilot_band"

    return {
        "source_family": source_family,
        "geometry_class": geometry_class,
        "topology": topology,
        "dump_layout": dump_layout,
        "side_access": side_access,
        "dig_dump_separation": {
            "metric": "p50_tiles",
            "legacy_target_token": separation_token,
            "achieved_value": separation_p50_tiles,
        },
        "capacity": {
            "metric": "single_layer_area_ratio",
            "derived_token": capacity_token,
            "achieved_value": single_layer_area_ratio,
        },
        "site_class": "none",
        "required_volume": required_volume,
        "volume_band": "legacy_unfrozen",
        "reset_mode": "full",
    }


def _component_sizes(
    mask: np.ndarray,
    *,
    structure: np.ndarray = EIGHT_CONNECTED,
) -> list[int]:
    labels, count = ndi.label(mask, structure=structure)
    return [int(np.count_nonzero(labels == index)) for index in range(1, count + 1)]


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(
        mask,
        structure=EIGHT_CONNECTED,
        border_value=0,
    )


def _shortest_paths(sources: np.ndarray, traversable: np.ndarray) -> np.ndarray:
    distance = np.full(sources.shape, np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int]] = []
    for row, column in np.argwhere(sources & traversable):
        row_int = int(row)
        column_int = int(column)
        distance[row_int, column_int] = 0.0
        heapq.heappush(queue, (0.0, row_int, column_int))
    while queue:
        current, row, column = heapq.heappop(queue)
        if current != distance[row, column]:
            continue
        for row_step, column_step, cost in SHORTEST_PATH_MOVES:
            next_row = row + row_step
            next_column = column + column_step
            if not (
                0 <= next_row < sources.shape[0]
                and 0 <= next_column < sources.shape[1]
                and traversable[next_row, next_column]
            ):
                continue
            proposed = current + cost
            if proposed < distance[next_row, next_column]:
                distance[next_row, next_column] = proposed
                heapq.heappush(
                    queue,
                    (proposed, next_row, next_column),
                )
    return distance


def _separation_metrics(
    dig: np.ndarray,
    accepted_dump: np.ndarray,
    occupancy: np.ndarray,
    tile_size_m: float,
) -> dict[str, dict[str, float]]:
    values = _shortest_paths(accepted_dump, ~occupancy)[_boundary(dig)]
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Dig boundary cannot reach the exact accepted dump mask.")
    tiles = {
        "p50": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }
    return {
        "tiles": tiles,
        "metres": {key: value * tile_size_m for key, value in tiles.items()},
    }


def _perimeter_4_edges(mask: np.ndarray) -> int:
    padded = np.pad(mask, 1, constant_values=False)
    horizontal = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    vertical = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    return int(horizontal + vertical)


def _foundation_descriptors(dig: np.ndarray) -> dict[str, Any]:
    points = np.argwhere(dig)
    if not len(points):
        raise ValueError("Excavation geometry is empty.")
    area = int(dig.sum())
    perimeter = _perimeter_4_edges(dig)
    height, width = (points.max(axis=0) - points.min(axis=0) + 1).tolist()
    covariance = np.cov(points.astype(np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_index = int(np.argmax(eigenvalues))
    minor = float(max(eigenvalues.min(), np.finfo(np.float64).eps))
    major = float(eigenvalues[major_index])
    major_vector = eigenvectors[:, major_index]
    return {
        "perimeter_4_edges": perimeter,
        "compactness_4pi_area_over_perimeter_squared": float(
            4.0 * math.pi * area / perimeter**2
        ),
        "component_count_4": len(_component_sizes(dig, structure=FOUR_CONNECTED)),
        "hole_cells": int(ndi.binary_fill_holes(dig).sum() - area),
        "bbox_height_cells": int(height),
        "bbox_width_cells": int(width),
        "bbox_aspect_ratio": float(max(height, width) / min(height, width)),
        "moment_aspect_ratio": float(math.sqrt(major / minor)),
        "moment_orientation_degrees": float(
            math.degrees(math.atan2(major_vector[0], major_vector[1])) % 180.0
        ),
    }


def _trench_descriptors(identity: Mapping[str, Any]) -> dict[str, Any]:
    topology = _require_token(
        TOPOLOGY_MAP,
        identity.get("topology"),
        "topology",
    )
    default_segments = {
        "straight": 1,
        "segmented_2": 2,
        "segmented_3": 3,
        "T": 3,
        "X": 4,
        "disconnected": 2,
    }
    default_degrees = {"T": [3], "X": [4]}
    axes = identity.get("axes_ABC") or []
    return {
        "segment_count": int(
            identity.get("trench_segments") or default_segments.get(topology, len(axes))
        ),
        "axis_count": int(identity.get("trench_axes_count") or len(axes)),
        "junction_count": int(identity.get("intersection_junctions") or 0),
        "junction_degrees": list(
            identity.get("junction_degrees") or default_degrees.get(topology, [])
        ),
        "global_angle_degrees": identity.get("trench_global_angle_deg"),
        "width_radius_tiles": identity.get("trench_width_radius_tiles"),
    }


def _trench_side_metrics(
    identity: Mapping[str, Any],
    dig: np.ndarray,
    accepted_dump: np.ndarray,
) -> dict[str, Any]:
    if identity.get("family") != "trench":
        return {
            "side_sign": None,
            "negative_dump_cells": None,
            "positive_dump_cells": None,
            "smaller_side_fraction": None,
            "forbidden_side_dump_cells": None,
        }
    heading_degrees = identity.get("trench_global_angle_deg")
    if not isinstance(heading_degrees, (int, float)) or not np.isfinite(
        heading_degrees
    ):
        raise ValueError("Trench side metrics require a finite global heading.")
    heading = math.radians(float(heading_degrees))
    normal = np.asarray([math.cos(heading), -math.sin(heading)])
    center = np.argwhere(dig).mean(axis=0)
    rows, columns = np.indices(dig.shape)
    projection = (rows - center[0]) * normal[0] + (columns - center[1]) * normal[1]
    negative = int(np.count_nonzero(accepted_dump & (projection <= -1.0)))
    positive = int(np.count_nonzero(accepted_dump & (projection >= 1.0)))
    side_sign = identity.get("side_sign")
    forbidden = None
    if identity.get("side_access") == "one":
        if side_sign not in {-1, 1}:
            raise ValueError("One-side trench requires side_sign in {-1, 1}.")
        forbidden = int(
            np.count_nonzero(accepted_dump & (side_sign * projection < 1.0))
        )
        if forbidden:
            raise ValueError(
                f"One-side accepted dump mask has {forbidden} forbidden cells."
            )
    return {
        "side_sign": side_sign,
        "negative_dump_cells": negative,
        "positive_dump_cells": positive,
        "smaller_side_fraction": float(
            min(negative, positive) / max(1, int(accepted_dump.sum()))
        ),
        "forbidden_side_dump_cells": forbidden,
    }


def _reset_metrics(
    target: np.ndarray,
    occupancy: np.ndarray,
    initial_soil: np.ndarray,
) -> dict[str, Any]:
    if not np.issubdtype(initial_soil.dtype, np.integer):
        raise ValueError("Initial soil must use an integer dtype.")
    if (
        initial_soil.min(initial=0) < np.iinfo(np.int8).min
        or initial_soil.max(initial=0) > np.iinfo(np.int8).max
    ):
        raise ValueError("Initial soil exceeds the Terra int8 range.")
    negative = initial_soil < 0
    positive = initial_soil > 0
    dig = target < 0
    if np.any(negative & ~dig):
        raise ValueError("Initial negative soil lies outside the excavation target.")
    if np.any(positive & (dig | occupancy)):
        raise ValueError("Initial positive soil overlaps excavation or occupancy.")
    negative_volume = -int(initial_soil[negative].astype(np.int64).sum())
    positive_volume = int(initial_soil[positive].astype(np.int64).sum())
    mass_residual = positive_volume - negative_volume
    required_volume = int(np.clip(-target.astype(np.int64), 0, None).sum())
    completion_fraction = negative_volume / required_volume if required_volume else 0.0
    remaining = dig & (initial_soil >= 0)
    return {
        "mode": "full",
        "completion_fraction": float(completion_fraction),
        "initial_negative_volume": negative_volume,
        "initial_positive_volume": positive_volume,
        "remaining_components": len(_component_sizes(remaining)),
        "remaining_component_sizes": _component_sizes(remaining),
        "mass_balance": {
            "residual": mass_residual,
            "conserved": mass_residual == 0,
        },
    }


def _site_metrics(
    occupancy: np.ndarray,
    dumpability: np.ndarray,
    initial_base_position: list[int] | None,
) -> dict[str, Any]:
    free = ~occupancy
    free_count = int(free.sum())
    traversable_sizes = _component_sizes(free)
    spawn_fraction = None
    if initial_base_position is not None:
        row, column = (int(value) for value in initial_base_position)
        labels, _ = ndi.label(free, structure=EIGHT_CONNECTED)
        label = int(labels[row, column])
        if label:
            spawn_fraction = float(
                np.count_nonzero(labels == label) / max(1, free_count)
            )
    return {
        "class": "none",
        "object_count_8": len(_component_sizes(occupancy)),
        "obstacle_fraction": float(occupancy.mean()),
        "nondump_fraction_of_free_cells": float(
            np.count_nonzero((~dumpability) & free) / max(1, free_count)
        ),
        "traversable_components_8": len(traversable_sizes),
        "traversable_component_sizes": traversable_sizes,
        "spawn_component_fraction_8_connected_proxy": spawn_fraction,
        "minimum_access_width_tiles": None,
        "minimum_access_width_status": "deferred_exact_pose_graph",
    }


def _metadata_identity_payload(
    identity: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "axes_ABC",
        "foundation_border_axes_ABC",
        "trench_axes_count",
        "trench_segments",
        "intersection_junctions",
        "intersection_branches",
        "junction_degrees",
        "trench_global_angle_deg",
        "trench_width_radius_tiles",
        "foundation_angle_deg",
        "foundation_wings",
    )
    return {
        "identity": {key: identity.get(key) for key in keys},
        # Loader bookkeeping such as legacy map_id and primary_cell must not
        # make counterfactual dump variants look like different geometries.
        "metadata": {key: metadata.get(key) for key in keys},
    }


def derive_content_ids(
    scenario: LoadedLegacyScenario,
    factor_vector: Mapping[str, Any],
    *,
    state_sha256: str,
    reset_seed_uint32: int,
) -> dict[str, str]:
    """Derive the geometry/map/scenario IDs at their intended boundaries."""

    target = np.asarray(scenario.target)
    dig = np.asarray(target < 0, dtype=np.bool_)
    layer_hashes = {
        "target_sha256": sha256_array(target),
        "occupancy_sha256": sha256_array(scenario.occupancy),
        "dumpability_sha256": sha256_array(scenario.dumpability),
        "initial_soil_sha256": sha256_array(scenario.initial_soil),
        "reward_distance_sha256": sha256_array(scenario.reward_distance),
        "metadata_sha256": canonical_json_sha256(scenario.metadata),
    }
    geometry_payload = {
        "schema": "terra_geometry_identity_v1",
        "family": scenario.identity["family"],
        "geometry_class": factor_vector["geometry_class"],
        "topology": factor_vector["topology"],
        "dig_mask_sha256": sha256_array(dig.astype(np.uint8)),
        "geometry_metadata": _metadata_identity_payload(
            scenario.identity,
            scenario.metadata,
        ),
    }
    geometry_id = _content_id("geometry", geometry_payload)
    map_payload = {
        "schema": "terra_map_identity_v1",
        "geometry_id": geometry_id,
        "target_sha256": layer_hashes["target_sha256"],
        "occupancy_sha256": layer_hashes["occupancy_sha256"],
        "dumpability_sha256": layer_hashes["dumpability_sha256"],
        "metadata_sha256": layer_hashes["metadata_sha256"],
    }
    map_id = _content_id("map", map_payload)
    scenario_payload = {
        "schema": "terra_scenario_identity_v1",
        "map_id": map_id,
        "initial_soil_sha256": layer_hashes["initial_soil_sha256"],
        "initial_agent_state_sha256": state_sha256,
        "environment_reset_seed_uint32": int(reset_seed_uint32),
    }
    return {
        "geometry_id": geometry_id,
        "map_id": map_id,
        "scenario_id": _content_id("scenario", scenario_payload),
        **layer_hashes,
    }


def reward_contract_sha256(
    environment_protocol: Mapping[str, Any],
) -> str:
    episode = environment_protocol.get("episode")
    if not isinstance(episode, Mapping):
        raise ValueError("Environment protocol has no frozen episode contract.")
    payload = {
        "schema": "terra_reward_contract_v1",
        "accepted_dump_contract": environment_protocol.get("accepted_dump_contract"),
        "rewards_type": episode.get("rewards_type"),
        "rewards_sha256": episode.get("rewards_sha256"),
        "apply_trench_rewards": episode.get("apply_trench_rewards"),
        "trench_shaping": episode.get("trench_shaping"),
    }
    if not isinstance(payload["rewards_sha256"], str) or not payload["rewards_sha256"]:
        raise ValueError("Environment protocol has no frozen reward hash.")
    return canonical_json_sha256(payload)


def derive_reward_treatment(
    *,
    scenario_id: str,
    reward_distance_sha256: str,
    frozen_reward_contract_sha256: str,
) -> dict[str, str]:
    payload = {
        "schema": "terra_reward_treatment_identity_v1",
        "scenario_id": scenario_id,
        "reward_distance_sha256": reward_distance_sha256,
        "reward_contract_sha256": frozen_reward_contract_sha256,
    }
    return {
        "treatment_id": _content_id("treatment", payload),
        "reward_distance_sha256": reward_distance_sha256,
        "reward_contract_sha256": frozen_reward_contract_sha256,
    }


def recompute_affordable_audit(
    scenario: LoadedLegacyScenario,
    *,
    tile_size_m: float,
    initial_base_position: list[int] | None,
) -> dict[str, Any]:
    """Recompute fields that do not invoke the exact pose/service search."""

    target = np.asarray(scenario.target)
    occupancy = np.asarray(scenario.occupancy)
    dumpability = np.asarray(scenario.dumpability)
    initial_soil = np.asarray(scenario.initial_soil)
    reward_distance = np.asarray(scenario.reward_distance)
    arrays = {
        "target": target,
        "occupancy": occupancy,
        "dumpability": dumpability,
        "initial_soil": initial_soil,
        "reward_distance": reward_distance,
    }
    for name, array in arrays.items():
        if array.shape != (BENCHMARK_MAP_SIZE, BENCHMARK_MAP_SIZE):
            raise ValueError(
                f"{name} has shape {array.shape}; expected "
                f"{(BENCHMARK_MAP_SIZE,) * 2}."
            )
    if not np.all(np.isin(target, (-1, 0, 1))):
        raise ValueError("Target contains values outside {-1, 0, 1}.")
    if not np.all(np.isin(occupancy, (0, 1))):
        raise ValueError("Occupancy is not binary.")
    if not np.all(np.isin(dumpability, (0, 1))):
        raise ValueError("Dumpability is not binary.")
    if not np.all(np.isfinite(reward_distance)):
        raise ValueError("Reward distance contains non-finite values.")
    if reward_distance.min() < 0.0 or reward_distance.max() > 1.0:
        raise ValueError("Reward distance lies outside [0, 1].")

    occupancy_bool = occupancy.astype(np.bool_)
    dumpability_bool = dumpability.astype(np.bool_)
    dig = target < 0
    declared_dump = target > 0
    accepted_dump = declared_dump & ~occupancy_bool
    if not np.any(dig) or not np.any(declared_dump):
        raise ValueError("B0a migration requires non-empty dig and dump targets.")
    if np.any((dig | declared_dump) & occupancy_bool):
        raise ValueError("Task target overlaps occupancy.")
    if np.any(accepted_dump & ~dumpability_bool):
        raise ValueError("Exact accepted dump mask contains non-dumpable cells.")

    capacity = contained_dump_capacity_sanity_check(
        target,
        occupancy_bool,
        dumpability_bool,
        initial_soil,
    )
    separation = _separation_metrics(
        dig,
        accepted_dump,
        occupancy_bool,
        tile_size_m,
    )
    dig_sizes = _component_sizes(dig)
    dump_sizes = _component_sizes(accepted_dump)
    required_volume = int(capacity["required_dig_volume"])
    reset = _reset_metrics(target, occupancy_bool, initial_soil)
    if reset["mode"] == "full" and (
        reset["initial_negative_volume"] or reset["initial_positive_volume"]
    ):
        raise ValueError("Legacy B0a full resets must start with zero soil work.")
    if not reset["mass_balance"]["conserved"]:
        raise ValueError("Initial soil fails mass conservation.")

    geometry: dict[str, Any] = {
        "source_class": _require_token(
            GEOMETRY_MAP,
            scenario.identity.get("geometry"),
            "geometry",
        )[0],
        "topology": _require_token(
            TOPOLOGY_MAP,
            scenario.identity.get("topology"),
            "topology",
        ),
        "dig_components": len(dig_sizes),
        "dig_component_sizes": dig_sizes,
        "dig_cells": int(dig.sum()),
        "required_volume": required_volume,
    }
    if scenario.identity.get("family") == "foundation":
        geometry["foundation_descriptors"] = _foundation_descriptors(dig)
    else:
        geometry.update(_trench_descriptors(scenario.identity))

    minimum_headroom = int(
        (
            np.iinfo(np.int8).max
            - np.clip(initial_soil.astype(np.int64), 0, None)[accepted_dump]
        ).min()
    )
    dump = {
        "layout": _require_token(
            DUMP_LAYOUT_MAP,
            scenario.identity.get("dump_layout"),
            "dump_layout",
        ),
        "side_access": scenario.identity.get("side_access"),
        **_trench_side_metrics(scenario.identity, dig, accepted_dump),
        "component_count": len(dump_sizes),
        "cells_per_component": dump_sizes,
        "accepted_cells": int(accepted_dump.sum()),
        "legal_free_coverage": float(accepted_dump.sum() / max(1, declared_dump.sum())),
        "dig_dump_separation_tiles": separation["tiles"],
        "dig_dump_separation_m": separation["metres"],
        "single_layer_area_ratio": capacity["single_layer_capacity_ratio"],
        "representable_remaining_volume": capacity["representable_remaining_volume"],
        "minimum_cell_headroom": minimum_headroom,
        "reachable_capacity_ratio": None,
        "reachable_capacity_status": "pending_exact_static",
        "any_direct_transfer_pose_exists_initial": None,
        "direct_service_coverage_initial": None,
        "direct_service_status": DIRECT_SERVICE_STATUS,
    }
    return {
        "geometry": geometry,
        "dump": dump,
        "site": _site_metrics(
            occupancy_bool,
            dumpability_bool,
            initial_base_position,
        ),
        "work": {
            "required_volume": required_volume,
            "separation_work_proxy": (required_volume * separation["tiles"]["p50"]),
            "forced_rehandling": None,
            "forced_rehandling_status": "deferred_direct_service_and_witness",
        },
        "reset": reset,
        "capacity_validation": capacity,
    }


def _intersect_spawn_contract(
    scenarios: list[LoadedLegacyScenario],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not scenarios:
        raise ValueError("At least one source-group variant is required.")
    padding = np.maximum.reduce(
        [np.asarray(row.occupancy, dtype=np.int8) for row in scenarios]
    )
    occupied_soil = np.logical_or.reduce(
        [np.asarray(row.initial_soil) != 0 for row in scenarios]
    ).astype(np.int8)
    dumpability = np.logical_and.reduce(
        [np.asarray(row.dumpability, dtype=np.bool_) for row in scenarios]
    )
    return padding, occupied_soil, dumpability


def canonical_state_materializer(
    source_group_id: str,
    split: str,
    scenarios: list[LoadedLegacyScenario],
    env_config: Any,
) -> MaterializedState:
    padding, occupied_soil, dumpability = _intersect_spawn_contract(scenarios)
    agent, seed_receipt = sample_benchmark_initial_agent(
        release_id=BENCHMARK_RELEASE_ID,
        split=split,
        source_group_id=source_group_id,
        state_index=0,
        env_cfg=env_config,
        padding_mask=padding,
        action_map=occupied_soil,
        dumpability_mask=dumpability,
    )
    for scenario in scenarios:
        validate_benchmark_initial_agent(
            agent,
            env_cfg=env_config,
            padding_mask=scenario.occupancy,
            action_map=scenario.initial_soil,
            dumpability_mask=scenario.dumpability,
        )
    return MaterializedState(
        state_record=agent_to_record(agent),
        state_sha256=seed_receipt["initial_agent_state_sha256"],
        seed_receipt=seed_receipt,
    )


def _source_groups(
    scenarios: Iterable[LoadedLegacyScenario],
) -> dict[str, list[LoadedLegacyScenario]]:
    groups: dict[str, list[LoadedLegacyScenario]] = {}
    for scenario in scenarios:
        source_id = scenario.identity.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                f"{scenario.identity.get('map_id')} has no valid source_id."
            )
        groups.setdefault(source_id, []).append(scenario)
    return groups


def _validate_grouping(
    scenarios: list[LoadedLegacyScenario],
    *,
    expected_identity_count: int,
    expected_source_group_count: int,
    expected_source_group_size_counts: Mapping[int, int],
) -> dict[str, list[LoadedLegacyScenario]]:
    if len(scenarios) != expected_identity_count:
        raise ValueError(
            f"Expected {expected_identity_count} identities, got {len(scenarios)}."
        )
    legacy_ids = [row.identity.get("map_id") for row in scenarios]
    if len(set(legacy_ids)) != len(legacy_ids):
        raise ValueError("Legacy map_id values are not unique.")
    groups = _source_groups(scenarios)
    if len(groups) != expected_source_group_count:
        raise ValueError(
            f"Expected {expected_source_group_count} source groups, got "
            f"{len(groups)}."
        )
    size_counts = Counter(len(rows) for rows in groups.values())
    if dict(size_counts) != dict(expected_source_group_size_counts):
        raise ValueError(
            "Source-group size distribution changed: "
            f"{dict(sorted(size_counts.items()))} != "
            f"{dict(sorted(expected_source_group_size_counts.items()))}."
        )
    for source_group_id, rows in groups.items():
        splits = {row.identity.get("split") for row in rows}
        dig_hashes = {sha256_array((row.target < 0).astype(np.uint8)) for row in rows}
        if len(splits) != 1 or len(dig_hashes) != 1:
            raise ValueError(
                f"Source group {source_group_id!r} crosses split or dig geometry."
            )
    source_splits: dict[str, set[str]] = {}
    for source_group_id, rows in groups.items():
        source_splits.setdefault(source_group_id, set()).update(
            str(row.identity.get("split")) for row in rows
        )
    overlap = {
        source: splits for source, splits in source_splits.items() if len(splits) != 1
    }
    if overlap:
        raise ValueError(f"Source groups cross splits: {overlap}.")
    return groups


def _active_base_position(state_record: Mapping[str, Any]) -> list[int]:
    positions = state_record["agent_states"]["pos_base"]
    return [int(value) for value in positions[0]]


def _legacy_outcome(identity: Mapping[str, Any]) -> dict[str, Any]:
    validation = identity.get("validation")
    if not isinstance(validation, Mapping):
        return {"status": None, "validation": None}
    return {
        "status": validation.get("status"),
        "stale_static_gate": validation.get("static_gate"),
        "stale_distance": validation.get("distance"),
        "stale_capacity": validation.get("capacity"),
    }


def _failure_outcome(
    identity: Mapping[str, Any],
    errors: Iterable[str],
) -> dict[str, Any]:
    return {
        "schema": MIGRATION_SCHEMA,
        "release_id": BENCHMARK_RELEASE_ID,
        "legacy_map_id": identity.get("map_id"),
        "source_group_id": identity.get("source_id"),
        "condition_id": None,
        "condition_status": CONDITION_STATUS,
        "migration_status": "failed",
        "direct_service_status": DIRECT_SERVICE_STATUS,
        "errors": list(errors),
        "legacy_outcome": _legacy_outcome(identity),
        "scenario": None,
        "audit": None,
    }


def migrate_loaded_scenarios(
    scenarios: list[LoadedLegacyScenario],
    *,
    environment_protocol: Mapping[str, Any],
    env_config: Any,
    state_materializer: StateMaterializer = canonical_state_materializer,
    expected_identity_count: int = EXPECTED_IDENTITY_COUNT,
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT,
    expected_source_group_size_counts: Mapping[
        int, int
    ] = EXPECTED_SOURCE_GROUP_SIZE_COUNTS,
) -> list[dict[str, Any]]:
    """Return one fail-closed migration outcome for every loaded identity."""

    groups = _validate_grouping(
        scenarios,
        expected_identity_count=expected_identity_count,
        expected_source_group_count=expected_source_group_count,
        expected_source_group_size_counts=expected_source_group_size_counts,
    )
    protocol_hash = environment_protocol.get("environment_protocol_sha256")
    map_protocol = environment_protocol.get("map")
    if not isinstance(protocol_hash, str) or not protocol_hash:
        raise ValueError("Environment protocol has no canonical hash.")
    if not isinstance(map_protocol, Mapping):
        raise ValueError("Environment protocol has no map receipt.")
    frozen_reward_hash = reward_contract_sha256(environment_protocol)
    tile_size_m = float(map_protocol["tile_size_m_derived_float64"])
    edge_length_m = float(map_protocol["edge_length_m"])
    edge_length_px = int(map_protocol["edge_length_px"])
    if not math.isclose(
        tile_size_m,
        edge_length_m / edge_length_px,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Protocol tile size does not equal edge_length_m / pixels.")

    state_by_group: dict[str, MaterializedState] = {}
    state_errors: dict[str, str] = {}
    for source_group_id in sorted(groups):
        rows = sorted(groups[source_group_id], key=lambda row: row.identity["map_id"])
        legacy_split = rows[0].identity.get("split")
        try:
            split = _require_token(SPLIT_MAP, legacy_split, "split")
            state_by_group[source_group_id] = state_materializer(
                source_group_id,
                split,
                rows,
                env_config,
            )
        except Exception as error:  # continue to name every affected row
            state_errors[source_group_id] = f"{type(error).__name__}: {error}"

    outcomes: list[dict[str, Any]] = []
    for scenario in sorted(scenarios, key=lambda row: row.identity["map_id"]):
        identity = scenario.identity
        source_group_id = str(identity["source_id"])
        if source_group_id in state_errors:
            outcomes.append(_failure_outcome(identity, [state_errors[source_group_id]]))
            continue
        try:
            split = _require_token(SPLIT_MAP, identity.get("split"), "split")
            state = state_by_group[source_group_id]
            base_position = _active_base_position(state.state_record)
            audit = recompute_affordable_audit(
                scenario,
                tile_size_m=tile_size_m,
                initial_base_position=base_position,
            )
            factor_vector = normalize_factor_vector(
                identity,
                separation_p50_tiles=audit["dump"]["dig_dump_separation_tiles"]["p50"],
                single_layer_area_ratio=audit["dump"]["single_layer_area_ratio"],
                required_volume=audit["work"]["required_volume"],
            )
            seed = int(state.seed_receipt["seed_uint32"])
            ids = derive_content_ids(
                scenario,
                factor_vector,
                state_sha256=state.state_sha256,
                reset_seed_uint32=seed,
            )
            reward_treatment = derive_reward_treatment(
                scenario_id=ids["scenario_id"],
                reward_distance_sha256=ids["reward_distance_sha256"],
                frozen_reward_contract_sha256=frozen_reward_hash,
            )
            scenario_record = {
                "schema": "terra_b0a_design_input_scenario_v1",
                "release_id": BENCHMARK_RELEASE_ID,
                "scenario_id": ids["scenario_id"],
                "map_id": ids["map_id"],
                "geometry_id": ids["geometry_id"],
                "source_group_id": source_group_id,
                "source_id": identity["source_id"],
                "legacy_map_id": identity["map_id"],
                "condition_id": None,
                "condition_status": CONDITION_STATUS,
                "split": split,
                "family": identity["family"],
                "factor_vector": factor_vector,
                "paired_source_group_id": identity.get("paired_source_group_id"),
                "topology_match_group_id": identity.get("topology_match_group_id"),
                "layers": {
                    "target_sha256": ids["target_sha256"],
                    "occupancy_sha256": ids["occupancy_sha256"],
                    "dumpability_sha256": ids["dumpability_sha256"],
                    "initial_soil_sha256": ids["initial_soil_sha256"],
                    "metadata_sha256": ids["metadata_sha256"],
                    "shape": [edge_length_px, edge_length_px],
                    "edge_length_m": edge_length_m,
                    "tile_size_m": tile_size_m,
                },
                "initial_condition": {
                    "environment_reset_seed": seed,
                    "seed_receipt": state.seed_receipt,
                    "initial_agent_state": state.state_record,
                    "initial_agent_state_sha256": state.state_sha256,
                    "initial_soil_sha256": ids["initial_soil_sha256"],
                    "initial_negative_volume": audit["reset"][
                        "initial_negative_volume"
                    ],
                    "initial_positive_volume": audit["reset"][
                        "initial_positive_volume"
                    ],
                    "completion_fraction": audit["reset"]["completion_fraction"],
                },
                "reward_treatment": reward_treatment,
                "environment_protocol_sha256": protocol_hash,
            }
            audit_record = {
                "schema": "terra_b0a_design_input_audit_v1",
                "release_id": BENCHMARK_RELEASE_ID,
                "scenario_id": ids["scenario_id"],
                **audit,
                "validation": {
                    "migration_record_valid": True,
                    "benchmark_format_valid": False,
                    "benchmark_format_status": (
                        "partial_design_input_not_canonical_s2_scenario"
                    ),
                    "affordable_semantics_valid": True,
                    "exact_capacity_valid": True,
                    "initial_state_valid": True,
                    "static_valid": None,
                    "static_status": MIGRATION_STATUS,
                    "direct_service_status": DIRECT_SERVICE_STATUS,
                    "witnessed": None,
                    "deferred_exact_static_fields": [
                        "action_reachable_base_pose_graph",
                        "admissible_pose_count_initial",
                        "initial_workspace_coverage",
                        "reachable_capacity_ratio",
                        "any_direct_transfer_pose_exists_initial",
                        "direct_service_coverage_initial",
                    ],
                },
            }
            outcomes.append(
                {
                    "schema": MIGRATION_SCHEMA,
                    "release_id": BENCHMARK_RELEASE_ID,
                    "legacy_map_id": identity["map_id"],
                    "source_group_id": source_group_id,
                    "condition_id": None,
                    "condition_status": CONDITION_STATUS,
                    "migration_status": MIGRATION_STATUS,
                    "direct_service_status": DIRECT_SERVICE_STATUS,
                    "errors": [],
                    "legacy_outcome": _legacy_outcome(identity),
                    "scenario": scenario_record,
                    "audit": audit_record,
                }
            )
        except Exception as error:
            outcomes.append(
                _failure_outcome(
                    identity,
                    [f"{type(error).__name__}: {error}"],
                )
            )
    return outcomes


@contextmanager
def _dataset_size(slot_count: int) -> Iterable[None]:
    previous = os.environ.get("DATASET_SIZE")
    os.environ["DATASET_SIZE"] = str(slot_count)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DATASET_SIZE", None)
        else:
            os.environ["DATASET_SIZE"] = previous


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object.")
        rows.append(value)
    return rows


def _require_verified_paths(
    input_root: Path,
    paths: Iterable[Path],
    verified_relative_paths: frozenset[str],
) -> None:
    missing = []
    for path in paths:
        try:
            relative = path.resolve().relative_to(input_root).as_posix()
        except ValueError as error:
            raise ValueError(f"Consumed B0a path leaves input root: {path}.") from error
        if relative not in verified_relative_paths:
            missing.append(relative)
    if missing:
        raise ValueError(
            "Consumed B0a files are absent from the frozen checksum manifest: "
            f"{sorted(missing)}."
        )


def _load_dataset(
    directory: Path,
    *,
    input_root: Path,
    verified_relative_paths: frozenset[str],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    dataset_path = directory / "dataset.json"
    manifest_path = directory / "manifest.jsonl"
    _require_verified_paths(
        input_root,
        (dataset_path, manifest_path),
        verified_relative_paths,
    )
    dataset = _read_json(dataset_path)
    slot_count = dataset.get("slot_count")
    if not isinstance(slot_count, int) or slot_count <= 0:
        raise ValueError(f"{directory}/dataset.json has invalid slot_count.")
    source_registry = dataset.get("source_registry")
    if not isinstance(source_registry, str) or not source_registry:
        raise ValueError(f"{dataset_path} has no source_registry path.")
    consumed = [
        directory / source_registry,
        *(
            directory / subdirectory / f"{prefix}{index}{suffix}"
            for index in range(1, slot_count + 1)
            for subdirectory, prefix, suffix in (
                ("images", "img_", ".npy"),
                ("occupancy", "img_", ".npy"),
                ("dumpability", "img_", ".npy"),
                ("actions", "img_", ".npy"),
                ("distance", "img_", ".npy"),
                ("metadata", "trench_", ".json"),
            )
        ),
    ]
    _require_verified_paths(
        input_root,
        consumed,
        verified_relative_paths,
    )
    manifest, shape, _ = validate_exact_dataset_contract(directory, slot_count)
    if shape != (BENCHMARK_MAP_SIZE, BENCHMARK_MAP_SIZE):
        raise ValueError(
            f"{directory} has shape {shape}; expected " f"{(BENCHMARK_MAP_SIZE,) * 2}."
        )
    with _dataset_size(slot_count):
        loaded = load_maps_from_disk(
            str(directory),
            require_trench_metadata=False,
            require_exact_contract=True,
        )
    return manifest, [np.asarray(jax.device_get(array)) for array in loaded]


def _validate_legacy_target_identity(
    *,
    legacy_map_id: str,
    identity: Mapping[str, Any],
    raw_target: np.ndarray,
    loaded_target: np.ndarray,
) -> None:
    """Verify the stored identity without conflating storage and loader dtypes."""

    if sha256_array(raw_target) != identity.get("target_identity_sha256"):
        raise ValueError(f"{legacy_map_id} target identity hash changed.")
    if not np.array_equal(loaded_target, raw_target):
        raise ValueError(f"{legacy_map_id} exact loader changed target values.")
    if sha256_array((raw_target < 0).astype(np.uint8)) != identity.get(
        "dig_identity_sha256"
    ):
        raise ValueError(f"{legacy_map_id} dig identity hash changed.")


def load_legacy_scenarios(
    input_root: Path,
    *,
    verified_relative_paths: frozenset[str],
) -> list[LoadedLegacyScenario]:
    """Join every legacy identity to the exact-loader result once per cell."""

    input_root = input_root.resolve()
    identities_path = input_root / "identities.jsonl"
    if not identities_path.is_file():
        raise FileNotFoundError(identities_path)
    _require_verified_paths(
        input_root,
        (identities_path,),
        verified_relative_paths,
    )
    identities = _read_jsonl(identities_path)
    if len(identities) != EXPECTED_IDENTITY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IDENTITY_COUNT} legacy identities, got "
            f"{len(identities)}."
        )

    datasets: dict[
        tuple[str, str],
        tuple[list[dict[str, Any]], list[np.ndarray]],
    ] = {}
    scenarios: list[LoadedLegacyScenario] = []
    for identity in identities:
        legacy_map_id = identity.get("map_id")
        split = identity.get("split")
        primary_cell = identity.get("primary_cell")
        if not all(
            isinstance(value, str) and value
            for value in (legacy_map_id, split, primary_cell)
        ):
            raise ValueError(f"Malformed legacy identity: {identity!r}.")
        key = (split, primary_cell)
        directory = input_root / "cells" / split / primary_cell
        if key not in datasets:
            datasets[key] = _load_dataset(
                directory,
                input_root=input_root,
                verified_relative_paths=verified_relative_paths,
            )
        manifest, arrays = datasets[key]
        matches = [row for row in manifest if row["map_id"] == legacy_map_id]
        if len(matches) != 1:
            raise ValueError(
                f"{directory} has {len(matches)} slots for {legacy_map_id}."
            )
        manifest_row = matches[0]
        for field in (
            "source_id",
            "split",
            "family",
            "stratum",
            "primary_cell",
        ):
            if manifest_row[field] != identity.get(field):
                raise ValueError(
                    f"{legacy_map_id} identity/manifest mismatch for {field}."
                )
        slot_index = int(manifest_row["slot_index"])
        slot = slot_index - 1
        (
            targets,
            occupancy,
            _trench_axes,
            _trench_types,
            _trench_axis_owners,
            _foundation_axes,
            _foundation_types,
            dumpability,
            initial_soil,
            reward_distance,
        ) = arrays
        metadata_path = directory / "metadata" / f"trench_{slot_index}.json"
        metadata = _read_json(metadata_path)
        target = targets[slot]
        raw_target = np.load(
            directory / "images" / f"img_{slot_index}.npy",
            allow_pickle=False,
        )
        _validate_legacy_target_identity(
            legacy_map_id=legacy_map_id,
            identity=identity,
            raw_target=raw_target,
            loaded_target=target,
        )
        source_files = (
            directory / "dataset.json",
            directory / "manifest.jsonl",
            directory / "images" / f"img_{slot_index}.npy",
            directory / "occupancy" / f"img_{slot_index}.npy",
            directory / "dumpability" / f"img_{slot_index}.npy",
            directory / "actions" / f"img_{slot_index}.npy",
            directory / "distance" / f"img_{slot_index}.npy",
            metadata_path,
        )
        scenarios.append(
            LoadedLegacyScenario(
                identity=identity,
                metadata=metadata,
                target=target,
                occupancy=occupancy[slot],
                dumpability=dumpability[slot],
                initial_soil=initial_soil[slot],
                reward_distance=reward_distance[slot],
                dataset_directory=directory,
                slot_index=slot_index,
                source_files=source_files,
            )
        )
    return scenarios


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def validate_checkout_state(
    *,
    requested_revision: str,
    head_revision: str,
    porcelain_status: str,
) -> None:
    if requested_revision != head_revision:
        raise ValueError(
            "Requested Terra revision does not match the executing checkout: "
            f"{requested_revision} != {head_revision}."
        )
    if porcelain_status.strip():
        raise ValueError("Refusing a real B0a migration from a dirty Terra worktree.")


def verify_clean_checkout(repository: Path, requested_revision: str) -> None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Could not verify Terra checkout state: {error.stderr.strip()}"
        ) from error
    validate_checkout_state(
        requested_revision=requested_revision,
        head_revision=head,
        porcelain_status=status,
    )


def run_migration(
    *,
    input_root: Path,
    output: Path,
    terra_revision: str,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output = output.resolve()
    verify_clean_checkout(REPOSITORY_ROOT, terra_revision)
    verified_input = verify_input_integrity(input_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty {output}.")
    output.mkdir(parents=True, exist_ok=True)

    env_config, env_receipt = frozen_benchmark_protocol()
    environment_protocol = frozen_environment_protocol(terra_revision)
    scenarios = load_legacy_scenarios(
        input_root,
        verified_relative_paths=verified_input.verified_relative_paths,
    )
    outcomes = migrate_loaded_scenarios(
        scenarios,
        environment_protocol=environment_protocol,
        env_config=env_config,
    )
    migration_validation_path = output / "migration_validation.jsonl"
    _write_jsonl(migration_validation_path, outcomes)
    migration_validation_sha256 = sha256_file(migration_validation_path)
    status_counts = Counter(row["migration_status"] for row in outcomes)
    group_sizes = Counter(len(rows) for rows in _source_groups(scenarios).values())
    scenario_ids = [
        row["scenario"]["scenario_id"]
        for row in outcomes
        if row["scenario"] is not None
    ]
    map_ids = [
        row["scenario"]["map_id"] for row in outcomes if row["scenario"] is not None
    ]
    geometry_ids = [
        row["scenario"]["geometry_id"]
        for row in outcomes
        if row["scenario"] is not None
    ]
    state_hashes_by_group: dict[str, set[str]] = {}
    for row in outcomes:
        if row["scenario"] is None:
            continue
        state_hashes_by_group.setdefault(row["source_group_id"], set()).add(
            row["scenario"]["initial_condition"]["initial_agent_state_sha256"]
        )
    source_id_by_legacy_map = {
        row.identity["map_id"]: row.identity["source_id"] for row in scenarios
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "release_id": BENCHMARK_RELEASE_ID,
        "status": (MIGRATION_STATUS if not status_counts.get("failed") else "failed"),
        "input_root": str(input_root),
        "identity_count": len(outcomes),
        "source_group_count": len(_source_groups(scenarios)),
        "source_group_size_counts": {
            str(size): count for size, count in sorted(group_sizes.items())
        },
        "source_group_assertions": {
            "canonical_field": "legacy.source_id",
            "source_group_id_equals_source_id": all(
                row["source_group_id"] == source_id_by_legacy_map[row["legacy_map_id"]]
                for row in outcomes
            ),
            "identity_count_is_256": len(outcomes) == EXPECTED_IDENTITY_COUNT,
            "source_group_count_is_144": (
                len(_source_groups(scenarios)) == EXPECTED_SOURCE_GROUP_COUNT
            ),
            "group_size_distribution_matches_frozen_b0a": (
                dict(group_sizes) == EXPECTED_SOURCE_GROUP_SIZE_COUNTS
            ),
            "one_initial_state_per_source_group": all(
                len(hashes) == 1 for hashes in state_hashes_by_group.values()
            )
            and len(state_hashes_by_group) == EXPECTED_SOURCE_GROUP_COUNT,
        },
        "unique_scenario_count": len(set(scenario_ids)),
        "unique_map_count": len(set(map_ids)),
        "unique_geometry_count": len(set(geometry_ids)),
        "condition_id_policy": "null_for_legacy_design_input",
        "condition_status": CONDITION_STATUS,
        "migration_status_counts": dict(sorted(status_counts.items())),
        "direct_service_status": DIRECT_SERVICE_STATUS,
        "static_valid_claimed": False,
        "source": verified_input.receipt,
        "implementation": {
            "migration_script_sha256": sha256_file(Path(__file__).resolve()),
            "environment_protocol_sha256": environment_protocol[
                "environment_protocol_sha256"
            ],
            "env_config_sha256": env_receipt["env_config_sha256"],
            "terra_revision": terra_revision,
        },
        "output": {
            "migration_validation_sha256": migration_validation_sha256,
            "migration_validation_record_count": len(outcomes),
        },
        "output_contract": [
            "migration_validation.jsonl",
            "migration_summary.json",
        ],
    }
    _write_json(output / "migration_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the fail-closed live-geometry B0a migration receipt."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terra-revision", required=True)
    args = parser.parse_args()
    summary = run_migration(
        input_root=args.input_root,
        output=args.output,
        terra_revision=args.terra_revision,
    )
    if summary["status"] == "failed":
        raise SystemExit(
            "B0a migration listed failed identities; inspect "
            "migration_validation.jsonl."
        )


if __name__ == "__main__":
    main()
