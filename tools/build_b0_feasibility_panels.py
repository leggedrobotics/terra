#!/usr/bin/env python3
"""Build B0a paired, source-disjoint Terra feasibility panels.

This builder intentionally does only offline generation.  It changes one map
axis at a time, writes exact-loader datasets, and validates the contained dump
contract before any PPO job can consume the bank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import ndimage as ndi

from terra.maps_buffer import contained_dump_capacity_sanity_check
from terra.maps_buffer import validate_exact_dataset_contract

MAP_SIZE = 64
TILE_SIZE_M = 44.0 / MAP_SIZE
MINIMUM_CAPACITY_RATIO = 3.0
TARGET_CAPACITY_RATIO = 3.25
DISTANCE_TOLERANCE_TILES = 0.75
MAX_WITHIN_CELL_GEOMETRY_IOU = 0.995
SPLITS = ("train", "development")
SPLIT_BASE_SEEDS = {
    "train": 2_026_072_700,
    "development": 2_126_072_700,
}


@dataclass(frozen=True)
class CellSpec:
    name: str
    family: str
    geometry: str
    dump_layout: str
    distance_center_tiles: int | None = None
    side_access: str = "all"
    topology: str | None = None


@dataclass
class Sample:
    target: np.ndarray
    occupancy: np.ndarray
    dumpability: np.ndarray
    action: np.ndarray
    distance: np.ndarray
    metadata: dict[str, Any]


CELLS = {
    "f_osm_all": CellSpec(
        "f_osm_all",
        "foundation",
        "foundation_osm",
        "all_around",
    ),
    "f_procedural_all": CellSpec(
        "f_procedural_all",
        "foundation",
        "foundation_procedural",
        "all_around",
    ),
    **{
        f"f_apron_d{distance:02d}": CellSpec(
            f"f_apron_d{distance:02d}",
            "foundation",
            "foundation_osm",
            "broad_apron",
            distance,
        )
        for distance in (2, 4, 6, 8)
    },
    **{
        f"t_straight_both_d{distance:02d}": CellSpec(
            f"t_straight_both_d{distance:02d}",
            "trench",
            "trench_straight",
            "broad_side_cast",
            distance,
            "both",
            "straight",
        )
        for distance in (2, 4, 6, 8)
    },
    "t_straight_one_d02": CellSpec(
        "t_straight_one_d02",
        "trench",
        "trench_straight",
        "broad_side_cast",
        2,
        "one",
        "straight",
    ),
    "t_segmented2_both_d02": CellSpec(
        "t_segmented2_both_d02",
        "trench",
        "trench_segmented2",
        "broad_side_cast",
        2,
        "both",
        "segmented_end_to_end_2",
    ),
    "t_segmented3_both_d02": CellSpec(
        "t_segmented3_both_d02",
        "trench",
        "trench_segmented3",
        "broad_side_cast",
        2,
        "both",
        "segmented_end_to_end_3",
    ),
    "t_T_both_d02": CellSpec(
        "t_T_both_d02",
        "trench",
        "trench_T",
        "broad_side_cast",
        2,
        "both",
        "T",
    ),
    "t_X_both_d02": CellSpec(
        "t_X_both_d02",
        "trench",
        "trench_X",
        "broad_side_cast",
        2,
        "both",
        "X",
    ),
    "t_disconnected_both_d02": CellSpec(
        "t_disconnected_both_d02",
        "trench",
        "trench_disconnected",
        "broad_side_cast",
        2,
        "both",
        "disconnected_2",
    ),
}

PANELS = {
    "foundation_geometry": (
        "f_osm_all",
        "f_procedural_all",
    ),
    "foundation_distance": tuple(
        f"f_apron_d{distance:02d}" for distance in (2, 4, 6, 8)
    ),
    "trench_distance": tuple(
        f"t_straight_both_d{distance:02d}" for distance in (2, 4, 6, 8)
    ),
    "trench_side": (
        "t_straight_both_d02",
        "t_straight_one_d02",
    ),
    "trench_topology": (
        "t_straight_both_d02",
        "t_segmented2_both_d02",
        "t_segmented3_both_d02",
        "t_T_both_d02",
        "t_X_both_d02",
        "t_disconnected_both_d02",
    ),
}

PRIMARY_EASY_CELLS = {
    "foundation": (
        "f_osm_all",
        "f_procedural_all",
        "f_apron_d02",
        "f_apron_d04",
    ),
    "trench": (
        "t_straight_both_d02",
        "t_straight_one_d02",
        "t_segmented2_both_d02",
        "t_segmented3_both_d02",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {key: json_value(item) for key, item in asdict(value).items()}
    if hasattr(value, "_asdict"):
        return {key: json_value(item) for key, item in value._asdict().items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_value) + "\n"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=json_value) + "\n")


def boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=np.bool_),
        border_value=0,
    )


def shortest_paths(
    sources: np.ndarray,
    traversable: np.ndarray | None = None,
) -> np.ndarray:
    if traversable is None:
        traversable = np.ones(sources.shape, dtype=np.bool_)
    distance = np.full(sources.shape, np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int]] = []
    for y, x in np.argwhere(sources & traversable):
        distance[y, x] = 0.0
        heappush(queue, (0.0, int(y), int(x)))
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    while queue:
        current, y, x = heappop(queue)
        if current != distance[y, x]:
            continue
        for dy, dx, cost in moves:
            next_y = y + dy
            next_x = x + dx
            if not (
                0 <= next_y < sources.shape[0]
                and 0 <= next_x < sources.shape[1]
                and traversable[next_y, next_x]
            ):
                continue
            proposed = current + cost
            if proposed < distance[next_y, next_x]:
                distance[next_y, next_x] = proposed
                heappush(queue, (proposed, next_y, next_x))
    return distance


def centered(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    if not len(points):
        return np.zeros_like(mask, dtype=np.bool_)
    center = points.mean(axis=0)
    requested = np.asarray(mask.shape, dtype=np.float64) / 2.0 - 0.5
    shift = np.rint(requested - center).astype(int)
    return (
        ndi.shift(
            mask.astype(np.uint8),
            shift=tuple(int(value) for value in shift),
            order=0,
            mode="constant",
            cval=0,
        )
        > 0
    )


def maximum_dihedral_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = centered(left)
    candidates = []
    for rotation in range(4):
        rotated = np.rot90(right, rotation)
        candidates.append(rotated)
        candidates.append(np.fliplr(rotated))
    maximum = 0.0
    for candidate in candidates:
        right_centered = centered(candidate)
        union = left_centered | right_centered
        intersection = left_centered & right_centered
        maximum = max(
            maximum,
            float(intersection.sum() / max(1, union.sum())),
        )
    return maximum


def maximum_previous_geometry_iou(
    accepted_digs: dict[tuple[str, str], list[np.ndarray]],
    split: str,
    cell_name: str,
    dig: np.ndarray,
) -> float:
    return max(
        (
            maximum_dihedral_iou(dig, previous)
            for previous in accepted_digs.get((split, cell_name), [])
        ),
        default=0.0,
    )


def line_coefficients(
    point_a_yx: np.ndarray,
    point_b_yx: np.ndarray,
) -> dict[str, float]:
    y1, x1 = (float(value) for value in point_a_yx)
    y2, x2 = (float(value) for value in point_b_yx)
    return {
        "A": y2 - y1,
        "B": x1 - x2,
        "C": x2 * y1 - x1 * y2,
    }


def load_review_generator(generator_root: Path):
    sys.path.insert(0, str(generator_root))
    import generate_prototypes_v5 as v5

    return v5


def derived_rng(seed: int, attempt: int) -> np.random.Generator:
    attempt_seed = int(np.random.SeedSequence([seed, attempt]).generate_state(1)[0])
    return np.random.default_rng(attempt_seed)


def make_segmented_trench(
    v5,
    rng: np.random.Generator,
    segment_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    for _ in range(300):
        heading = float(rng.choice(np.deg2rad(np.arange(0, 180, 15))))
        headings = [heading]
        headings.append(
            heading + float(rng.choice(np.deg2rad(np.asarray([-45, -30, 30, 45]))))
        )
        if segment_count == 3:
            headings.append(
                headings[-1]
                + float(rng.choice(np.deg2rad(np.asarray([-30, -15, 15, 30]))))
            )
        lengths = rng.uniform(8.0, 12.0, size=segment_count)
        points = [np.zeros(2, dtype=np.float64)]
        for local_heading, length in zip(headings, lengths):
            points.append(
                points[-1]
                + length
                * np.asarray([math.sin(local_heading), math.cos(local_heading)])
            )
        points_array = np.asarray(points)
        box_center = (points_array.min(axis=0) + points_array.max(axis=0)) / 2.0
        points_array += rng.uniform(29.0, 35.0, size=2) - box_center
        if not v5.v3.points_inside(points_array, margin=10):
            continue
        dig = v5.base.rasterize_polyline(points_array, radius=1)
        if not 55 <= int(dig.sum()) <= 125:
            continue
        return dig, {
            "trench_axes_count": segment_count,
            "trench_segments": segment_count,
            "intersection_junctions": 0,
            "junction_degrees": [],
            "trench_topology": f"segmented_end_to_end_{segment_count}",
            "trench_width_radius_tiles": 1,
            "trench_global_angle_deg": round(math.degrees(heading), 1),
            "axes_ABC": [
                line_coefficients(start, end)
                for start, end in zip(
                    points_array[:-1],
                    points_array[1:],
                )
            ],
        }
    raise RuntimeError(f"could not generate a {segment_count}-segment trench")


def make_disconnected_trench(
    v5,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    for _ in range(300):
        heading = float(rng.choice(np.deg2rad(np.arange(0, 180, 15))))
        direction = np.asarray([math.sin(heading), math.cos(heading)])
        normal = np.asarray([direction[1], -direction[0]])
        center = rng.uniform(29.0, 35.0, size=2)
        separation = float(rng.uniform(7.0, 10.0))
        length_a, length_b = rng.uniform(11.0, 17.0, size=2)
        centers = (
            center - normal * separation / 2.0,
            center + normal * separation / 2.0,
        )
        line_sets = [
            np.vstack(
                [
                    local_center - direction * length / 2.0,
                    local_center + direction * length / 2.0,
                ]
            )
            for local_center, length in zip(
                centers,
                (length_a, length_b),
            )
        ]
        all_points = np.vstack(line_sets)
        if not v5.v3.points_inside(all_points, margin=10):
            continue
        dig = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        for points in line_sets:
            dig |= v5.base.rasterize_polyline(points, radius=1)
        _, component_count = ndi.label(
            dig,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if component_count != 2 or not 55 <= int(dig.sum()) <= 125:
            continue
        return dig, {
            "trench_axes_count": 2,
            "trench_segments": 2,
            "intersection_junctions": 0,
            "junction_degrees": [],
            "trench_topology": "disconnected_2",
            "trench_width_radius_tiles": 1,
            "trench_global_angle_deg": round(math.degrees(heading), 1),
            "axes_ABC": [
                line_coefficients(points[0], points[1]) for points in line_sets
            ],
        }
    raise RuntimeError("could not generate a disconnected trench")


def make_geometry(
    v5,
    geometry_factory,
    geometry: str,
    seed: int,
    used_osm_sources: set[int],
) -> tuple[np.ndarray, dict[str, Any], str, int]:
    for attempt in range(2_000):
        rng = derived_rng(seed, attempt)
        if geometry == "foundation_osm":
            dig, metadata = geometry_factory.foundation_osm(rng)
            source_index = int(metadata["foundation_source_index"])
            if source_index in used_osm_sources:
                continue
            if not 90 <= int(dig.sum()) <= 180:
                continue
            used_osm_sources.add(source_index)
            return (
                dig,
                metadata,
                f"osm-foundation:{source_index}",
                attempt,
            )
        if geometry == "foundation_procedural":
            dig, metadata = geometry_factory.foundation_procedural(rng)
            if not 110 <= int(dig.sum()) <= 190:
                continue
            return dig, metadata, f"procedural:{seed}:{attempt}", attempt
        if geometry == "trench_straight":
            dig, metadata = geometry_factory.trench_axes_1(rng)
            if not 55 <= int(dig.sum()) <= 125:
                continue
            return dig, metadata, f"procedural:{seed}:{attempt}", attempt
        if geometry == "trench_segmented2":
            dig, metadata = make_segmented_trench(v5, rng, 2)
            return dig, metadata, f"procedural:{seed}:{attempt}", attempt
        if geometry == "trench_segmented3":
            dig, metadata = make_segmented_trench(v5, rng, 3)
            return dig, metadata, f"procedural:{seed}:{attempt}", attempt
        if geometry in {"trench_T", "trench_X"}:
            requested = geometry.removeprefix("trench_")
            dig, metadata = geometry_factory.trench_axes_2(rng)
            if metadata["trench_topology"] != requested:
                continue
            if not 80 <= int(dig.sum()) <= 155:
                continue
            return dig, metadata, f"procedural:{seed}:{attempt}", attempt
        if geometry == "trench_disconnected":
            dig, metadata = make_disconnected_trench(v5, rng)
            return dig, metadata, f"procedural:{seed}:{attempt}", attempt
        raise ValueError(f"unknown geometry {geometry}")
    raise RuntimeError(f"exhausted geometry candidates for {geometry}")


def trench_projection(
    dig: np.ndarray,
    heading_degrees: float,
) -> np.ndarray:
    heading = math.radians(heading_degrees)
    normal = np.asarray([math.cos(heading), -math.sin(heading)])
    center = np.argwhere(dig).mean(axis=0)
    yy, xx = np.indices(dig.shape)
    return (yy - center[0]) * normal[0] + (xx - center[1]) * normal[1]


def nearest_cells(
    allowed: np.ndarray,
    euclidean_distance: np.ndarray,
    count: int,
) -> np.ndarray | None:
    candidates = np.argwhere(allowed)
    if len(candidates) < count:
        return None
    candidate_distance = euclidean_distance[
        candidates[:, 0],
        candidates[:, 1],
    ]
    order = np.lexsort(
        (
            candidates[:, 1],
            candidates[:, 0],
            candidate_distance,
        )
    )
    selected = candidates[order[:count]]
    mask = np.zeros(allowed.shape, dtype=np.bool_)
    mask[selected[:, 0], selected[:, 1]] = True
    return mask


def candidate_dump(
    dig: np.ndarray,
    euclidean_distance: np.ndarray,
    lower_distance: float,
    target_cells: int,
    side_access: str,
    heading_degrees: float | None,
    side_sign: int,
) -> np.ndarray | None:
    allowed = (~dig) & (euclidean_distance >= lower_distance)
    if side_access == "all":
        return nearest_cells(allowed, euclidean_distance, target_cells)
    if heading_degrees is None:
        raise RuntimeError("trench side access requires a global heading")
    projection = trench_projection(dig, heading_degrees)
    if side_access == "one":
        allowed &= side_sign * projection >= 1.0
        return nearest_cells(allowed, euclidean_distance, target_cells)
    if side_access != "both":
        raise ValueError(side_access)
    negative_count = target_cells // 2
    positive_count = target_cells - negative_count
    negative = nearest_cells(
        allowed & (projection <= -1.0),
        euclidean_distance,
        negative_count,
    )
    positive = nearest_cells(
        allowed & (projection >= 1.0),
        euclidean_distance,
        positive_count,
    )
    if negative is None or positive is None:
        return None
    return negative | positive


def dump_distance_statistics(
    dig: np.ndarray,
    dump: np.ndarray,
) -> dict[str, float]:
    values = shortest_paths(dump)[boundary(dig)]
    if not len(values) or not np.all(np.isfinite(values)):
        raise RuntimeError("dig boundary cannot reach accepted dump")
    return {
        "p50_tiles": float(np.median(values)),
        "p95_tiles": float(np.quantile(values, 0.95)),
        "max_tiles": float(values.max()),
        "p50_metres": float(np.median(values) * TILE_SIZE_M),
        "p95_metres": float(np.quantile(values, 0.95) * TILE_SIZE_M),
        "max_metres": float(values.max() * TILE_SIZE_M),
    }


def build_apron_dump(
    dig: np.ndarray,
    distance_center_tiles: int,
    *,
    side_access: str,
    heading_degrees: float | None = None,
    side_sign: int = 1,
    target_capacity_ratio: float = TARGET_CAPACITY_RATIO,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_cells = int(math.ceil(target_capacity_ratio * int(dig.sum())))
    euclidean_distance = ndi.distance_transform_edt(~dig)
    best: tuple[float, np.ndarray, dict[str, float], float] | None = None
    for shift in np.linspace(-0.25, 2.0, 46):
        lower_distance = max(1.0, distance_center_tiles - float(shift))
        dump = candidate_dump(
            dig,
            euclidean_distance,
            lower_distance,
            target_cells,
            side_access,
            heading_degrees,
            side_sign,
        )
        if dump is None or np.any(dump & dig):
            continue
        statistics = dump_distance_statistics(dig, dump)
        error = abs(statistics["p50_tiles"] - distance_center_tiles)
        candidate = (error, dump, statistics, lower_distance)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("could not construct requested broad apron")
    error, dump, statistics, lower_distance = best
    if error > DISTANCE_TOLERANCE_TILES:
        raise RuntimeError(
            "requested distance bin was not achieved: "
            f"center={distance_center_tiles}, "
            f"p50={statistics['p50_tiles']:.5f}"
        )
    return dump, {
        "distance_center_tiles": distance_center_tiles,
        "distance_tolerance_tiles": DISTANCE_TOLERANCE_TILES,
        "euclidean_generation_lower_bound_tiles": lower_distance,
        "target_capacity_ratio": target_capacity_ratio,
        "target_dump_cells": target_cells,
        "side_access": side_access,
        **statistics,
    }


def make_sample(
    v5,
    spec: CellSpec,
    dig: np.ndarray,
    geometry_metadata: dict[str, Any],
    *,
    side_sign: int,
) -> Sample:
    occupancy = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    dumpability = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
    action = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    if spec.dump_layout == "all_around":
        dump = ~dig
        distance_metadata = dump_distance_statistics(dig, dump)
        dump_metadata = {
            "distance_center_tiles": None,
            "distance_tolerance_tiles": None,
            "target_capacity_ratio": float(dump.sum() / max(1, dig.sum())),
            "target_dump_cells": int(dump.sum()),
            "side_access": "all",
            **distance_metadata,
        }
    else:
        dump, dump_metadata = build_apron_dump(
            dig,
            int(spec.distance_center_tiles),
            side_access=spec.side_access,
            heading_degrees=geometry_metadata.get("trench_global_angle_deg"),
            side_sign=side_sign,
        )
    target = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    target[dig] = -1
    target[dump] = 1
    distance = v5.base.compute_geodesic_distance(
        target,
        occupancy.astype(np.bool_),
    ).astype(np.float32)
    return Sample(
        target=target,
        occupancy=occupancy,
        dumpability=dumpability,
        action=action,
        distance=distance,
        metadata={
            **geometry_metadata,
            **dump_metadata,
            "geometry": spec.geometry,
            "dump_layout": spec.dump_layout,
            "side_sign": side_sign if spec.side_access == "one" else None,
        },
    )


def validate_sample(
    v5,
    spec: CellSpec,
    sample: Sample,
) -> dict[str, Any]:
    target = sample.target
    occupancy = sample.occupancy.astype(np.bool_)
    dumpability = sample.dumpability.astype(np.bool_)
    dig = target < 0
    dump = target > 0
    if target.shape != (MAP_SIZE, MAP_SIZE):
        raise RuntimeError("target is not 64 x 64")
    if not np.all(np.isin(target, (-1, 0, 1))):
        raise RuntimeError("target contains values outside {-1, 0, 1}")
    if np.any((dig | dump) & occupancy):
        raise RuntimeError("task target overlaps occupancy")
    if np.any(dump & ~dumpability):
        raise RuntimeError("accepted dump overlaps non-dumpable ground")
    if not np.all(np.isfinite(sample.distance)):
        raise RuntimeError("reward distance is non-finite")
    if sample.distance.min() < 0.0 or sample.distance.max() > 1.0:
        raise RuntimeError("reward distance is outside [0, 1]")

    capacity = contained_dump_capacity_sanity_check(
        target,
        occupancy,
        dumpability,
        sample.action,
        minimum_single_layer_ratio=MINIMUM_CAPACITY_RATIO,
    )
    static_gate = v5.base.static_gate(target, occupancy, dumpability)
    if not static_gate.accepted:
        raise RuntimeError(f"static reachability failed: {static_gate.reason}")
    distance = dump_distance_statistics(dig, dump)
    if spec.distance_center_tiles is not None:
        error = abs(distance["p50_tiles"] - spec.distance_center_tiles)
        if error > DISTANCE_TOLERANCE_TILES:
            raise RuntimeError(
                f"{spec.name} distance error {error:.5f} exceeds "
                f"{DISTANCE_TOLERANCE_TILES:.5f}"
            )

    dig_labels, dig_components = ndi.label(
        dig,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    del dig_labels
    dump_labels, dump_components = ndi.label(
        dump,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    del dump_labels
    expected_dig_components = 2 if spec.topology == "disconnected_2" else 1
    if dig_components != expected_dig_components:
        raise RuntimeError(
            f"{spec.name} has {dig_components} dig components, "
            f"expected {expected_dig_components}"
        )
    if spec.topology is not None:
        if sample.metadata.get("trench_topology") != spec.topology:
            raise RuntimeError(
                f"{spec.name} topology metadata mismatch: "
                f"{sample.metadata.get('trench_topology')}"
            )

    side_metrics: dict[str, Any] = {
        "negative_dump_cells": None,
        "positive_dump_cells": None,
        "smaller_side_fraction": None,
        "forbidden_side_dump_cells": None,
    }
    if spec.family == "trench":
        projection = trench_projection(
            dig,
            float(sample.metadata["trench_global_angle_deg"]),
        )
        negative = int((dump & (projection <= -1.0)).sum())
        positive = int((dump & (projection >= 1.0)).sum())
        side_metrics.update(
            {
                "negative_dump_cells": negative,
                "positive_dump_cells": positive,
                "smaller_side_fraction": float(
                    min(negative, positive) / max(1, dump.sum())
                ),
            }
        )
        if spec.side_access == "both":
            if min(negative, positive) < int(0.40 * dump.sum()):
                raise RuntimeError(f"{spec.name} lacks material capacity on both sides")
        elif spec.side_access == "one":
            side_sign = int(sample.metadata["side_sign"])
            forbidden = int((dump & (side_sign * projection < 1.0)).sum())
            side_metrics["forbidden_side_dump_cells"] = forbidden
            if forbidden:
                raise RuntimeError(f"{spec.name} has {forbidden} forbidden-side cells")

    return {
        "status": "passed",
        "accepted_dump_definition": "(target > 0) & ~occupancy",
        "accepted_dump_contract": "exact_visible_dump_v1",
        "dig_cells": int(dig.sum()),
        "dump_cells": int(dump.sum()),
        "dig_components": int(dig_components),
        "dump_components": int(dump_components),
        "capacity": capacity,
        "static_gate": json_value(static_gate),
        "distance": distance,
        "side": side_metrics,
    }


def add_record(
    records: list[dict[str, Any]],
    samples: dict[str, Sample],
    *,
    v5,
    split: str,
    spec: CellSpec,
    identity_index: int,
    dig: np.ndarray,
    geometry_metadata: dict[str, Any],
    source_id: str,
    generation_seed: int,
    generation_attempt: int,
    paired_source_group_id: str | None,
    topology_match_group_id: str | None,
    side_sign: int,
    accepted_digs: dict[tuple[str, str], list[np.ndarray]],
) -> None:
    key = (split, spec.name)
    previous = accepted_digs.setdefault(key, [])
    similarities = [maximum_dihedral_iou(dig, prior) for prior in previous]
    maximum_similarity = max(similarities, default=0.0)
    if maximum_similarity >= MAX_WITHIN_CELL_GEOMETRY_IOU:
        raise RuntimeError(
            f"{split}/{spec.name} generated a templated duplicate "
            f"(dihedral IoU={maximum_similarity:.6f})"
        )

    sample = make_sample(
        v5,
        spec,
        dig,
        geometry_metadata,
        side_sign=side_sign,
    )
    validation = validate_sample(v5, spec, sample)
    map_id = f"b0a-{split}-{spec.name}-{identity_index:02d}"
    if map_id in samples:
        raise RuntimeError(f"duplicate map ID {map_id}")
    record = {
        "map_id": map_id,
        "source_id": source_id,
        "split": split,
        "family": spec.family,
        "stratum": "B0a",
        "primary_cell": spec.name,
        "geometry": spec.geometry,
        "dump_layout": spec.dump_layout,
        "distance_center_tiles": spec.distance_center_tiles,
        "side_access": spec.side_access,
        "topology": spec.topology,
        "generation_seed": generation_seed,
        "generation_attempt": generation_attempt,
        "paired_source_group_id": paired_source_group_id,
        "topology_match_group_id": topology_match_group_id,
        "dig_identity_sha256": sha256_array(dig.astype(np.uint8)),
        "target_identity_sha256": sha256_array(sample.target),
        "canonical_similarity_metric": (
            "maximum centered dihedral intersection_over_union"
        ),
        "maximum_within_cell_geometry_iou": maximum_similarity,
        "maximum_allowed_within_cell_geometry_iou": (MAX_WITHIN_CELL_GEOMETRY_IOU),
        **{key_name: json_value(value) for key_name, value in sample.metadata.items()},
        "validation": validation,
    }
    previous.append(dig.copy())
    records.append(record)
    samples[map_id] = sample


def generate_bank(
    v5,
    source_foundations: Path,
    count_per_cell: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Sample],
    dict[str, int],
]:
    geometry_factory = v5.v3.GeometryFactoryV3(source_foundations)
    used_osm_sources: set[int] = set()
    accepted_digs: dict[tuple[str, str], list[np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    samples: dict[str, Sample] = {}
    rejections: Counter[str] = Counter()

    for split_index, split in enumerate(SPLITS):
        split_seed = SPLIT_BASE_SEEDS[split]

        for cell_offset, cell_name in enumerate(("f_osm_all", "f_procedural_all")):
            spec = CELLS[cell_name]
            for identity_index in range(count_per_cell):
                base_seed = (
                    split_seed + 100_000 * (cell_offset + 1) + identity_index * 1_000
                )
                for duplicate_attempt in range(200):
                    seed = base_seed + duplicate_attempt
                    dig, geometry_metadata, source_id, attempt = make_geometry(
                        v5,
                        geometry_factory,
                        spec.geometry,
                        seed,
                        used_osm_sources,
                    )
                    try:
                        add_record(
                            records,
                            samples,
                            v5=v5,
                            split=split,
                            spec=spec,
                            identity_index=identity_index,
                            dig=dig,
                            geometry_metadata=geometry_metadata,
                            source_id=source_id,
                            generation_seed=seed,
                            generation_attempt=attempt,
                            paired_source_group_id=None,
                            topology_match_group_id=None,
                            side_sign=1,
                            accepted_digs=accepted_digs,
                        )
                    except RuntimeError as error:
                        if "templated duplicate" not in str(error):
                            raise
                        rejections[f"{split}:{cell_name}:templated_duplicate"] += 1
                        continue
                    break
                else:
                    raise RuntimeError(
                        f"exhausted diverse identities for {split}/{cell_name}"
                    )

        for identity_index in range(count_per_cell):
            base_seed = split_seed + 300_000 + identity_index * 1_000
            for duplicate_attempt in range(200):
                seed = base_seed + duplicate_attempt
                dig, geometry_metadata, source_id, attempt = make_geometry(
                    v5,
                    geometry_factory,
                    "foundation_osm",
                    seed,
                    used_osm_sources,
                )
                similarity = maximum_previous_geometry_iou(
                    accepted_digs,
                    split,
                    "f_apron_d02",
                    dig,
                )
                if similarity >= MAX_WITHIN_CELL_GEOMETRY_IOU:
                    rejections[f"{split}:foundation_distance:templated_duplicate"] += 1
                    continue
                break
            else:
                raise RuntimeError(
                    f"exhausted diverse identities for " f"{split}/foundation_distance"
                )
            pair_group = f"{split}:foundation-distance:{identity_index:02d}"
            for distance in (2, 4, 6, 8):
                spec = CELLS[f"f_apron_d{distance:02d}"]
                add_record(
                    records,
                    samples,
                    v5=v5,
                    split=split,
                    spec=spec,
                    identity_index=identity_index,
                    dig=dig,
                    geometry_metadata=geometry_metadata,
                    source_id=source_id,
                    generation_seed=seed,
                    generation_attempt=attempt,
                    paired_source_group_id=pair_group,
                    topology_match_group_id=None,
                    side_sign=1,
                    accepted_digs=accepted_digs,
                )

        for identity_index in range(count_per_cell):
            base_seed = split_seed + 400_000 + identity_index * 1_000
            for duplicate_attempt in range(200):
                seed = base_seed + duplicate_attempt
                dig, geometry_metadata, source_id, attempt = make_geometry(
                    v5,
                    geometry_factory,
                    "trench_straight",
                    seed,
                    used_osm_sources,
                )
                similarity = maximum_previous_geometry_iou(
                    accepted_digs,
                    split,
                    "t_straight_both_d02",
                    dig,
                )
                if similarity >= MAX_WITHIN_CELL_GEOMETRY_IOU:
                    rejections[f"{split}:trench_straight:templated_duplicate"] += 1
                    continue
                break
            else:
                raise RuntimeError(
                    f"exhausted diverse identities for " f"{split}/trench_straight"
                )
            pair_group = f"{split}:trench-straight:{identity_index:02d}"
            side_sign = -1 if identity_index % 2 else 1
            for distance in (2, 4, 6, 8):
                spec = CELLS[f"t_straight_both_d{distance:02d}"]
                add_record(
                    records,
                    samples,
                    v5=v5,
                    split=split,
                    spec=spec,
                    identity_index=identity_index,
                    dig=dig,
                    geometry_metadata=geometry_metadata,
                    source_id=source_id,
                    generation_seed=seed,
                    generation_attempt=attempt,
                    paired_source_group_id=pair_group,
                    topology_match_group_id=(
                        f"{split}:topology:{identity_index:02d}"
                        if distance == 2
                        else None
                    ),
                    side_sign=side_sign,
                    accepted_digs=accepted_digs,
                )
            add_record(
                records,
                samples,
                v5=v5,
                split=split,
                spec=CELLS["t_straight_one_d02"],
                identity_index=identity_index,
                dig=dig,
                geometry_metadata=geometry_metadata,
                source_id=source_id,
                generation_seed=seed,
                generation_attempt=attempt,
                paired_source_group_id=pair_group,
                topology_match_group_id=None,
                side_sign=side_sign,
                accepted_digs=accepted_digs,
            )

        topology_cells = (
            "t_segmented2_both_d02",
            "t_segmented3_both_d02",
            "t_T_both_d02",
            "t_X_both_d02",
            "t_disconnected_both_d02",
        )
        for topology_offset, cell_name in enumerate(topology_cells):
            spec = CELLS[cell_name]
            for identity_index in range(count_per_cell):
                base_seed = (
                    split_seed
                    + 500_000
                    + topology_offset * 100_000
                    + identity_index * 1_000
                )
                for duplicate_attempt in range(200):
                    seed = base_seed + duplicate_attempt
                    dig, geometry_metadata, source_id, attempt = make_geometry(
                        v5,
                        geometry_factory,
                        spec.geometry,
                        seed,
                        used_osm_sources,
                    )
                    try:
                        add_record(
                            records,
                            samples,
                            v5=v5,
                            split=split,
                            spec=spec,
                            identity_index=identity_index,
                            dig=dig,
                            geometry_metadata=geometry_metadata,
                            source_id=source_id,
                            generation_seed=seed,
                            generation_attempt=attempt,
                            paired_source_group_id=None,
                            topology_match_group_id=(
                                f"{split}:topology:{identity_index:02d}"
                            ),
                            side_sign=1,
                            accepted_digs=accepted_digs,
                        )
                    except RuntimeError as error:
                        if "templated duplicate" not in str(error):
                            raise
                        rejections[f"{split}:{cell_name}:templated_duplicate"] += 1
                        continue
                    break
                else:
                    raise RuntimeError(
                        f"exhausted diverse identities for {split}/{cell_name}"
                    )

        expected = len(CELLS) * count_per_cell
        observed = sum(record["split"] == split for record in records)
        if observed != expected:
            raise RuntimeError(
                f"{split} has {observed} identities, expected {expected}"
            )
        if split_index == 0 and not used_osm_sources:
            raise RuntimeError("no OSM foundation sources were consumed")

    return records, samples, dict(rejections)


def write_dataset(
    directory: Path,
    records: list[dict[str, Any]],
    samples: dict[str, Sample],
    source_registry: Path,
) -> None:
    if directory.exists():
        raise FileExistsError(directory)
    for name in (
        "images",
        "occupancy",
        "dumpability",
        "actions",
        "distance",
        "metadata",
    ):
        (directory / name).mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda row: row["map_id"])
    manifest = []
    for slot_index, record in enumerate(ordered, start=1):
        sample = samples[record["map_id"]]
        stem = f"img_{slot_index}"
        np.save(directory / "images" / f"{stem}.npy", sample.target)
        np.save(
            directory / "occupancy" / f"{stem}.npy",
            sample.occupancy,
        )
        np.save(
            directory / "dumpability" / f"{stem}.npy",
            sample.dumpability,
        )
        np.save(directory / "actions" / f"{stem}.npy", sample.action)
        np.save(
            directory / "distance" / f"{stem}.npy",
            sample.distance,
        )
        write_json(
            directory / "metadata" / f"trench_{slot_index}.json",
            {
                "map_id": record["map_id"],
                "family": record["family"],
                "primary_cell": record["primary_cell"],
                "geometry": record["geometry"],
                "topology": record["topology"],
                "axes_ABC": sample.metadata.get("axes_ABC", []),
                "foundation_border_axes_ABC": [],
            },
        )
        manifest.append(
            {
                "slot_index": slot_index,
                "map_id": record["map_id"],
                "source_id": record["source_id"],
                "split": record["split"],
                "family": record["family"],
                "stratum": record["stratum"],
                "primary_cell": record["primary_cell"],
                "slot_weight": 1.0,
                "identity_slot_multiplicity": 1,
            }
        )
    write_jsonl(directory / "manifest.jsonl", manifest)
    registry_relative = os.path.relpath(source_registry, directory)
    write_json(
        directory / "dataset.json",
        {
            "schema": "terra_exact_map_dataset_v1",
            "slot_count": len(ordered),
            "unique_identity_count": len(ordered),
            "shape": [MAP_SIZE, MAP_SIZE],
            "distance_metric": ("8_connected_cardinal_1_diagonal_sqrt2"),
            "distance_normalization": "per_map_max_to_1",
            "accepted_dump_contract": "exact_visible_dump_v1",
            "minimum_dump_capacity_ratio": MINIMUM_CAPACITY_RATIO,
            "source_registry": registry_relative,
            "source_registry_sha256": sha256_file(source_registry),
        },
    )
    validate_exact_dataset_contract(directory, len(ordered))


def render_cell_gallery(
    path: Path,
    records: list[dict[str, Any]],
    samples: dict[str, Sample],
) -> None:
    columns = 4
    rows = math.ceil(len(records) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    colormap = ListedColormap(["#f3e6c3", "#ef8b23", "#4daa6b", "#111111"])
    for axis, record in zip(axes.ravel(), records):
        sample = samples[record["map_id"]]
        code = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        code[sample.target < 0] = 1
        code[sample.target > 0] = 2
        code[sample.occupancy.astype(np.bool_)] = 3
        axis.imshow(
            code,
            cmap=colormap,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        distance = record["validation"]["distance"]
        axis.set_title(
            f"{record['map_id'].rsplit('-', 1)[-1]}  "
            f"work={record['validation']['dig_cells']}  "
            f"cap={record['validation']['capacity']['single_layer_capacity_ratio']:.2f}x\n"
            f"dump p50={distance['p50_tiles']:.2f}, "
            f"p95={distance['p95_tiles']:.2f}",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(records) :]:
        axis.axis("off")
    figure.suptitle(
        f"{records[0]['split']} / {records[0]['primary_cell']}",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_panel_gallery(
    path: Path,
    panel: str,
    cell_names: tuple[str, ...],
    records: list[dict[str, Any]],
    samples: dict[str, Sample],
) -> None:
    figure, axes = plt.subplots(
        2,
        len(cell_names),
        figsize=(3.2 * len(cell_names), 6.4),
        squeeze=False,
        constrained_layout=True,
    )
    colormap = ListedColormap(["#f3e6c3", "#ef8b23", "#4daa6b", "#111111"])
    for row, identity_index in enumerate((0, 1)):
        for column, cell_name in enumerate(cell_names):
            candidates = [
                record for record in records if record["primary_cell"] == cell_name
            ]
            record = sorted(
                candidates,
                key=lambda item: item["map_id"],
            )[identity_index % len(candidates)]
            sample = samples[record["map_id"]]
            code = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
            code[sample.target < 0] = 1
            code[sample.target > 0] = 2
            axis = axes[row, column]
            axis.imshow(
                code,
                cmap=colormap,
                vmin=0,
                vmax=3,
                interpolation="nearest",
            )
            axis.set_title(
                cell_name.replace("_", " ")
                + "\n"
                + (f"p50 " f"{record['validation']['distance']['p50_tiles']:.2f}"),
                fontsize=8,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(f"train / {panel} paired feasibility panel")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_global(
    records: list[dict[str, Any]],
    count_per_cell: int,
) -> dict[str, Any]:
    expected = len(CELLS) * count_per_cell * len(SPLITS)
    if len(records) != expected:
        raise RuntimeError(f"bank has {len(records)} records, expected {expected}")
    map_ids = [record["map_id"] for record in records]
    if len(map_ids) != len(set(map_ids)):
        raise RuntimeError("map IDs are not unique")
    target_hashes = [record["target_identity_sha256"] for record in records]
    if len(target_hashes) != len(set(target_hashes)):
        raise RuntimeError("target arrays contain an undeclared exact duplicate")

    source_splits: dict[str, set[str]] = {}
    for record in records:
        source_splits.setdefault(record["source_id"], set()).add(record["split"])
    overlap = {
        source: sorted(splits)
        for source, splits in source_splits.items()
        if len(splits) > 1
    }
    if overlap:
        raise RuntimeError(
            f"source identities cross splits: {next(iter(overlap.items()))}"
        )

    paired_dig_hashes: dict[str, set[str]] = {}
    for record in records:
        pair_group = record["paired_source_group_id"]
        if pair_group is not None:
            paired_dig_hashes.setdefault(pair_group, set()).add(
                record["dig_identity_sha256"]
            )
    bad_pairs = {
        group: hashes for group, hashes in paired_dig_hashes.items() if len(hashes) != 1
    }
    if bad_pairs:
        raise RuntimeError(f"paired source groups changed dig geometry: {bad_pairs}")

    counts = Counter((record["split"], record["primary_cell"]) for record in records)
    incorrect_counts = {
        f"{split}/{cell}": count
        for (split, cell), count in counts.items()
        if count != count_per_cell
    }
    if incorrect_counts:
        raise RuntimeError(f"cell identity count mismatch: {incorrect_counts}")

    return {
        "status": "passed",
        "identity_count": len(records),
        "unique_map_ids": len(set(map_ids)),
        "unique_target_arrays": len(set(target_hashes)),
        "train_development_source_disjoint": True,
        "paired_source_groups": len(paired_dig_hashes),
        "paired_groups_preserve_dig_geometry": True,
        "cell_identity_count": count_per_cell,
        "maximum_allowed_within_cell_geometry_iou": (MAX_WITHIN_CELL_GEOMETRY_IOU),
        "distance_tolerance_tiles": DISTANCE_TOLERANCE_TILES,
        "minimum_capacity_ratio": MINIMUM_CAPACITY_RATIO,
    }


def file_manifest(output: Path) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "files.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(output)}" for path in paths]
    (output / "files.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--source-foundations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-per-cell", type=int, default=8)
    args = parser.parse_args()
    if args.count_per_cell < 2:
        raise ValueError("--count-per-cell must be at least 2")
    generator_root = args.generator_root.resolve()
    source_foundations = args.source_foundations.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    v5 = load_review_generator(generator_root)
    records, samples, rejection_counts = generate_bank(
        v5,
        source_foundations,
        args.count_per_cell,
    )
    global_validation = validate_global(records, args.count_per_cell)

    source_registry = output / "source_registry.jsonl"
    registry_rows = [
        {
            "map_id": record["map_id"],
            "source_id": record["source_id"],
            "split": record["split"],
            "paired_source_group_id": record["paired_source_group_id"],
        }
        for record in records
    ]
    write_jsonl(source_registry, registry_rows)
    write_jsonl(output / "identities.jsonl", records)

    scalar_fields = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if not isinstance(value, (dict, list))
        }
    )
    with (output / "identities.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows(
            [
                {key: json_value(record.get(key, "")) for key in scalar_fields}
                for record in records
            ]
        )

    dataset_receipts: dict[str, int] = {}
    for split in SPLITS:
        for cell_name in CELLS:
            selected = [
                record
                for record in records
                if record["split"] == split and record["primary_cell"] == cell_name
            ]
            relative = f"cells/{split}/{cell_name}"
            write_dataset(
                output / relative,
                selected,
                samples,
                source_registry,
            )
            dataset_receipts[relative] = len(selected)
            gallery = output / "galleries" / split
            gallery.mkdir(parents=True, exist_ok=True)
            render_cell_gallery(
                gallery / f"{cell_name}.png",
                selected,
                samples,
            )

        for panel, cell_names in PANELS.items():
            selected = [
                record
                for record in records
                if record["split"] == split and record["primary_cell"] in cell_names
            ]
            relative = f"panels/{split}/{panel}"
            write_dataset(
                output / relative,
                selected,
                samples,
                source_registry,
            )
            dataset_receipts[relative] = len(selected)
            if split == "train":
                panel_gallery = output / "galleries" / "panels"
                panel_gallery.mkdir(parents=True, exist_ok=True)
                render_panel_gallery(
                    panel_gallery / f"{panel}.png",
                    panel,
                    cell_names,
                    selected,
                    samples,
                )

    generator_files = [
        generator_root / f"generate_prototypes{suffix}.py"
        for suffix in ("", "_v2", "_v3", "_v4", "_v5")
    ]
    provenance = {
        "schema": "terra_b0a_paired_feasibility_panels_v1",
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "generator_files": {str(path): sha256_file(path) for path in generator_files},
        "source_foundations": str(source_foundations),
        "split_base_seeds": SPLIT_BASE_SEEDS,
        "count_per_cell": args.count_per_cell,
        "cells": list(CELLS),
        "panels": PANELS,
        "primary_easy_cells": PRIMARY_EASY_CELLS,
        "source_registry_sha256": sha256_file(source_registry),
        "identity_manifest_sha256": sha256_file(output / "identities.jsonl"),
        "dataset_directories": dataset_receipts,
    }
    write_json(output / "provenance.json", provenance)
    write_json(
        output / "validation.json",
        {
            **global_validation,
            "dataset_directories": dataset_receipts,
            "rejection_counts": rejection_counts,
        },
    )
    write_json(
        output / "generation_summary.json",
        {
            "schema": "terra_b0a_generation_summary_v1",
            "accepted_identities": len(records),
            "identities_per_cell_per_split": args.count_per_cell,
            "rejection_counts": rejection_counts,
        },
    )
    (output / "README.md").write_text(
        "# Terra B0a paired feasibility panels\n\n"
        "This immutable bank changes one declared map axis at a time. "
        "It is a static and bounded-dynamic feasibility instrument, not a "
        "production curriculum bank.\n\n"
        "- `cells/{train,development}/`: eight identities per candidate cell.\n"
        "- `panels/{train,development}/`: the five declared B0b run inputs.\n"
        "- `galleries/`: every candidate cell plus five paired panel views.\n"
        "- `identities.jsonl`: quantitative geometry, distance, side, "
        "capacity, source, split, and hash metadata.\n"
        "- `validation.json`: exact-loader, C1a capacity, distance, "
        "similarity, pairing, and split-disjointness gate.\n"
        "- `provenance.json`: frozen builder, generator, seed, and dataset "
        "receipts.\n\n"
        "Static acceptance is not a legal action trajectory. B0b supplies "
        "that separate dynamic witness.\n"
    )
    file_manifest(output)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
