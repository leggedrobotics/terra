#!/usr/bin/env python3
"""Build the frozen local M0-M2 banks from the reviewed v5 generator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi


MAP_SIZE = 64
TILE_SIZE_M = 44.0 / MAP_SIZE
SPLIT_BASE_SEEDS = {
    "train": 1_026_072_400,
    "development": 1_126_072_400,
    "sealed": 1_226_072_400,
}
SPLIT_ORDER = ("train", "development", "sealed")
STRATUM_ORDER = ("M0", "M1", "M2")
FAMILY_ORDER = ("foundation", "trench")


@dataclass(frozen=True)
class Cell:
    stratum: str
    family: str
    name: str
    geometry: str
    dump_style: str
    site_style: str
    volume_bucket: str = "any"
    trench_topology: str | None = None


CELLS = (
    Cell("M0", "foundation", "all_around_low", "foundation_osm", "easy_surround", "light", "low"),
    Cell("M0", "foundation", "all_around_normal", "foundation_osm", "easy_surround", "light", "normal"),
    Cell("M0", "foundation", "large_apron_low", "foundation_osm", "near_apron_large", "light", "low"),
    Cell("M0", "foundation", "large_apron_normal", "foundation_osm", "near_apron_large", "light", "normal"),
    Cell("M0", "trench", "straight_both_low", "trench_axes_1", "easy_surround", "light", "low", "straight"),
    Cell("M0", "trench", "straight_both_normal", "trench_axes_1", "easy_surround", "light", "normal", "straight"),
    Cell("M0", "trench", "straight_one_side_low", "trench_axes_1", "near_apron_large", "light", "low", "straight"),
    Cell("M0", "trench", "straight_one_side_normal", "trench_axes_1", "near_apron_large", "light", "normal", "straight"),
    Cell("M1", "foundation", "procedural_apron_light", "foundation_procedural", "near_apron_large", "light"),
    Cell("M1", "foundation", "procedural_one_side_light", "foundation_procedural", "one_side_near", "light"),
    Cell("M1", "foundation", "procedural_separated_light", "foundation_procedural", "separated_zones", "light"),
    Cell("M1", "foundation", "procedural_one_side_objects", "foundation_procedural", "one_side_near", "scattered_objects"),
    Cell("M1", "trench", "straight_one_side_light", "trench_axes_1", "near_apron_large", "light", trench_topology="straight"),
    Cell("M1", "trench", "segmented_both_light", "trench_axes_segmented", "easy_surround", "light", trench_topology="segmented_end_to_end"),
    Cell("M1", "trench", "segmented_one_side_objects", "trench_axes_segmented", "near_apron_large", "scattered_objects", trench_topology="segmented_end_to_end"),
    Cell("M1", "trench", "segmented_separated_objects", "trench_axes_segmented", "separated_zones", "scattered_objects", trench_topology="segmented_end_to_end"),
    Cell("M2", "foundation", "irregular_apron_objects", "foundation_procedural", "near_apron_large", "scattered_objects"),
    Cell("M2", "foundation", "irregular_one_side_road", "foundation_procedural", "one_side_near", "access_road"),
    Cell("M2", "foundation", "irregular_separated_wall", "foundation_procedural", "separated_zones", "gapped_wall"),
    Cell("M2", "foundation", "irregular_nearby_road", "foundation_procedural", "broad_nearby", "access_road"),
    Cell("M2", "trench", "T_both_objects", "trench_axes_2", "easy_surround", "scattered_objects", trench_topology="T"),
    Cell("M2", "trench", "T_one_side_road", "trench_axes_2", "near_apron_large", "access_road", trench_topology="T"),
    Cell("M2", "trench", "X_both_wall", "trench_axes_2", "easy_surround", "gapped_wall", trench_topology="X"),
    Cell("M2", "trench", "X_one_side_wall", "trench_axes_2", "near_apron_large", "gapped_wall", trench_topology="X"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_generator(generator_root: Path):
    sys.path.insert(0, str(generator_root))
    import generate_prototypes_v5 as v5

    # v5 normally installs these bindings inside main(). This builder calls the
    # accepted sample function directly, so install the same bindings here.
    v5.v4.DumpFactoryV4 = v5.DumpFactoryV5
    v5.v4.FOUNDATION_DUMP_WEIGHTS = v5.FOUNDATION_DUMP_WEIGHTS
    v5.v4.TRENCH_DUMP_WEIGHTS = v5.TRENCH_DUMP_WEIGHTS
    v5.v4.TRENCH_MAX_MEDIAN_DISTANCE_TILES = (
        v5.TRENCH_MAX_MEDIAN_DISTANCE_TILES
    )
    for style in (
        "broad_nearby",
        "near_apron_large",
        "one_side_near",
        "separated_zones",
    ):
        v5.v2.CAPACITY_RANGES[style] = (2.10, 2.60)

    class CurriculumGeometryFactory(v5.v3.GeometryFactoryV3):
        @staticmethod
        def trench_axes_segmented(rng):
            for _ in range(240):
                segment_count = int(rng.integers(2, 4))
                base_heading = float(
                    rng.choice(np.deg2rad(np.arange(0, 180, 15)))
                )
                headings = [base_heading]
                first_bend = float(
                    rng.choice(np.deg2rad(np.array([-45, -30, 30, 45])))
                )
                headings.append(base_heading + first_bend)
                if segment_count == 3:
                    headings.append(
                        headings[-1]
                        + float(
                            rng.choice(
                                np.deg2rad(
                                    np.array([-30, -15, 0, 15, 30])
                                )
                            )
                        )
                    )
                lengths = rng.uniform(9.0, 15.0, size=segment_count)
                points = [np.zeros(2, dtype=np.float64)]
                for heading, length in zip(headings, lengths):
                    points.append(
                        points[-1]
                        + length
                        * np.array(
                            [math.sin(heading), math.cos(heading)]
                        )
                    )
                points = np.asarray(points)
                box_center = (points.min(axis=0) + points.max(axis=0)) / 2
                requested_center = rng.uniform(28.0, 36.0, size=2)
                points += requested_center - box_center
                if not v5.v3.points_inside(points):
                    continue
                radius = int(rng.choice([1, 2], p=[0.7, 0.3]))
                dig = v5.base.rasterize_polyline(points, radius=radius)
                if not 60 <= int(dig.sum()) <= 230:
                    continue
                axes = [
                    v5.v3.line_coefficients(start, end)
                    for start, end in zip(points[:-1], points[1:])
                ]
                return dig, {
                    "geometry_hardness": "trench_easy_segmented",
                    "trench_axes_count": segment_count,
                    "trench_segments": segment_count,
                    "intersection_junctions": 0,
                    "intersection_branches": 2,
                    "junction_degrees": [],
                    "trench_topology": "segmented_end_to_end",
                    "trench_width_radius_tiles": radius,
                    "trench_global_angle_deg": round(
                        math.degrees(base_heading), 1
                    ),
                    "axes_ABC": axes,
                }
            raise RuntimeError(
                "Could not construct an end-to-end segmented trench"
            )

    v5.v3.GEOMETRY_HARDNESS["trench_axes_segmented"] = (
        "trench_easy_segmented"
    )
    v5.v3.TARGET_GEOMETRY_WEIGHTS["trench_axes_segmented"] = 0.0
    return v5, CurriculumGeometryFactory


def shortest_paths(
    sources: np.ndarray, traversable: np.ndarray
) -> np.ndarray:
    distance = np.full(sources.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for y, x in np.argwhere(sources & traversable):
        distance[y, x] = 0.0
        heappush(heap, (0.0, int(y), int(x)))
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
    while heap:
        current, y, x = heappop(heap)
        if current != distance[y, x]:
            continue
        for dy, dx, cost in moves:
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < MAP_SIZE
                and 0 <= nx < MAP_SIZE
                and traversable[ny, nx]
            ):
                proposed = current + cost
                if proposed < distance[ny, nx]:
                    distance[ny, nx] = proposed
                    heappush(heap, (proposed, ny, nx))
    return distance


def validate_sample(sample, cell: Cell) -> dict[str, Any]:
    target = np.asarray(sample.target)
    occupancy = np.asarray(sample.occupancy, dtype=bool)
    dumpability = np.asarray(sample.dumpability, dtype=bool)
    reward_distance = np.asarray(sample.distance)
    if not (
        target.shape
        == occupancy.shape
        == dumpability.shape
        == reward_distance.shape
        == (MAP_SIZE, MAP_SIZE)
    ):
        raise ValueError("map arrays must all be 64 x 64")
    if not np.all(np.isin(target, (-1, 0, 1))):
        raise ValueError("target contains values outside {-1, 0, 1}")
    if np.any((target != 0) & occupancy):
        raise ValueError("target overlaps an obstacle")
    if np.any((target > 0) & ~dumpability):
        raise ValueError("dump target contains non-dumpable cells")
    if not np.issubdtype(reward_distance.dtype, np.floating):
        raise ValueError("reward distance must use a floating dtype")
    if not np.all(np.isfinite(reward_distance)):
        raise ValueError("reward distance contains non-finite values")
    if reward_distance.min() < 0.0 or reward_distance.max() > 1.0:
        raise ValueError("reward distance is not normalized to [0, 1]")

    dig = target < 0
    dump = target > 0
    boundary = dig & ~ndi.binary_erosion(
        dig, structure=np.ones((3, 3), dtype=np.uint8)
    )
    traversable = ~occupancy
    distance_to_dump = shortest_paths(dump, traversable)
    work_distances = distance_to_dump[boundary]
    if work_distances.size == 0 or not np.all(np.isfinite(work_distances)):
        raise ValueError("a dig-boundary work cell cannot reach a dump cell")

    limits = (
        (6.0, 10.0, 12.0)
        if cell.stratum == "M0"
        else (10.0, 14.0, 18.0)
    )
    p50 = float(np.median(work_distances))
    p95 = float(np.quantile(work_distances, 0.95))
    maximum = float(work_distances.max())
    if p50 > limits[0] or p95 > limits[1] or maximum > limits[2]:
        raise ValueError(
            "path-distance gate failed: "
            f"p50={p50:.3f}, p95={p95:.3f}, max={maximum:.3f}"
        )

    distance_from_work = shortest_paths(boundary, traversable)
    nearby_dump = dump & (distance_from_work <= limits[2])
    required_capacity = (
        2.5
        if cell.stratum == "M0"
        and cell.family == "trench"
        and cell.dump_style == "easy_surround"
        else 2.0
    )
    if cell.family == "foundation" and cell.dump_style == "easy_surround":
        required_capacity = 3.0
        legal_free = (~dig) & (~occupancy) & dumpability
        if not np.array_equal(dump, legal_free):
            raise ValueError(
                "foundation all-around target is not all legal free ground"
            )
    nearby_capacity = float(nearby_dump.sum() / max(1, dig.sum()))
    if nearby_capacity + 1e-8 < required_capacity:
        raise ValueError(
            "nearby reachable capacity gate failed: "
            f"{nearby_capacity:.4f} < {required_capacity:.4f}"
        )

    labels, component_count = ndi.label(
        dump, structure=np.ones((3, 3), dtype=np.uint8)
    )
    components = []
    for component in range(1, component_count + 1):
        mask = labels == component
        component_distances = distance_from_work[mask]
        components.append(
            {
                "component": component,
                "cells": int(mask.sum()),
                "nearby_cells": int(
                    (mask & (distance_from_work <= limits[2])).sum()
                ),
                "minimum_path_tiles": round(
                    float(component_distances.min()), 5
                ),
            }
        )

    return {
        "eligibility_path_metric": "8_connected_cardinal_1_diagonal_sqrt2",
        "eligibility_work_cells": int(boundary.sum()),
        "eligibility_distance_p50_tiles": round(p50, 5),
        "eligibility_distance_p95_tiles": round(p95, 5),
        "eligibility_distance_max_tiles": round(maximum, 5),
        "eligibility_distance_p50_m": round(p50 * TILE_SIZE_M, 5),
        "eligibility_distance_p95_m": round(p95 * TILE_SIZE_M, 5),
        "eligibility_distance_max_m": round(maximum * TILE_SIZE_M, 5),
        "nearby_capacity_radius_tiles": limits[2],
        "nearby_reachable_dump_cells": int(nearby_dump.sum()),
        "nearby_reachable_capacity_ratio": round(nearby_capacity, 5),
        "required_capacity_ratio": required_capacity,
        "dump_component_metrics": components,
        "reward_distance_metric": "reviewed_v5_obstacle_aware_geodesic",
        "reward_distance_normalization": "per_map_max_to_1",
        "reward_distance_sha256": hashlib.sha256(
            reward_distance.tobytes()
        ).hexdigest(),
    }


def volume_matches(cell: Cell, dig_cells: int) -> bool:
    if cell.volume_bucket == "any":
        return True
    threshold = 160 if cell.family == "foundation" else 110
    if cell.volume_bucket == "low":
        return dig_cells <= threshold
    return dig_cells > threshold


def count_per_cell(split: str, stratum: str) -> int:
    if split != "train":
        return 8
    return 8 if stratum == "M0" else 12


def generate_identities(
    v5,
    geometry_factory,
    split: str,
    used_osm_sources: set[int],
    used_dig_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    samples: dict[str, Any] = {}
    split_cells = [cell for cell in CELLS]
    for cell_index, cell in enumerate(split_cells):
        requested = count_per_cell(split, cell.stratum)
        accepted = 0
        candidate = 0
        rejection_counts: dict[str, int] = {}
        while accepted < requested:
            if candidate >= 5000:
                raise RuntimeError(
                    f"exhausted candidates for {split}/{cell.stratum}/"
                    f"{cell.family}/{cell.name}: {rejection_counts}"
                )
            seed = (
                SPLIT_BASE_SEEDS[split]
                + cell_index * 100_000
                + candidate
            )
            candidate += 1
            sample, generator_rejections = v5.make_sample_v5(
                geometry_factory,
                cell.geometry,
                cell.dump_style,
                cell.site_style,
                seed,
                240,
            )
            if sample is None:
                rejection_counts["generator"] = (
                    rejection_counts.get("generator", 0) + 1
                )
                for reason, count in generator_rejections.items():
                    rejection_counts[f"generator:{reason}"] = (
                        rejection_counts.get(f"generator:{reason}", 0)
                        + int(count)
                    )
                continue
            metadata = sample.metadata
            if not volume_matches(cell, int(metadata["dig_cells"])):
                rejection_counts["volume_bucket"] = (
                    rejection_counts.get("volume_bucket", 0) + 1
                )
                continue
            if (
                cell.trench_topology is not None
                and metadata.get("trench_topology")
                != cell.trench_topology
            ):
                rejection_counts["trench_topology"] = (
                    rejection_counts.get("trench_topology", 0) + 1
                )
                continue
            source_index = metadata.get("foundation_source_index")
            if source_index is not None and int(source_index) in used_osm_sources:
                rejection_counts["source_overlap"] = (
                    rejection_counts.get("source_overlap", 0) + 1
                )
                continue
            dig_hash = hashlib.sha256(
                np.asarray(sample.target < 0, dtype=np.uint8).tobytes()
            ).hexdigest()
            if dig_hash in used_dig_hashes:
                rejection_counts["dig_identity_overlap"] = (
                    rejection_counts.get("dig_identity_overlap", 0) + 1
                )
                continue
            try:
                eligibility = validate_sample(sample, cell)
            except ValueError as exc:
                reason = str(exc).split(":", 1)[0]
                rejection_counts[reason] = (
                    rejection_counts.get(reason, 0) + 1
                )
                continue

            if source_index is not None:
                used_osm_sources.add(int(source_index))
                source_id = f"osm:{int(source_index)}"
            else:
                source_id = f"procedural_seed:{seed}"
            used_dig_hashes.add(dig_hash)
            map_id = (
                f"{split}-{cell.stratum}-{cell.family}-"
                f"{cell.name}-{accepted:02d}"
            )
            record = {
                "map_id": map_id,
                "split": split,
                "stratum": cell.stratum,
                "family": cell.family,
                "primary_cell": cell.name,
                "source_id": source_id,
                "generation_seed": seed,
                "dig_identity_sha256": dig_hash,
                "geometry": cell.geometry,
                "dump_style": cell.dump_style,
                "site_style": cell.site_style,
                "volume_bucket": cell.volume_bucket,
                **{
                    key: json_value(value)
                    for key, value in metadata.items()
                    if key
                    not in {
                        "seed",
                        "schema",
                    }
                },
                **eligibility,
            }
            identities.append(record)
            samples[map_id] = sample
            accepted += 1
        print(
            f"{split}/{cell.stratum}/{cell.family}/{cell.name}: "
            f"{accepted} accepted from {candidate} candidates"
        )
    return identities, samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, default=json_value) + "\n"
            )


def write_level(
    directory: Path,
    records: list[dict[str, Any]],
    samples: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if directory.exists():
        raise FileExistsError(directory)
    for folder in (
        "images",
        "occupancy",
        "dumpability",
        "actions",
        "distance",
        "metadata",
    ):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(records))
    slot_rows = []
    for slot_index, source_index in enumerate(order, start=1):
        identity = records[int(source_index)]
        sample = samples[identity["map_id"]]
        stem = f"img_{slot_index}"
        np.save(directory / "images" / f"{stem}.npy", sample.target)
        np.save(
            directory / "occupancy" / f"{stem}.npy",
            np.asarray(sample.occupancy, dtype=np.int8),
        )
        np.save(
            directory / "dumpability" / f"{stem}.npy",
            np.asarray(sample.dumpability, dtype=np.int8),
        )
        np.save(directory / "actions" / f"{stem}.npy", sample.action)
        np.save(directory / "distance" / f"{stem}.npy", sample.distance)
        metadata = {
            "map_id": identity["map_id"],
            "split": identity["split"],
            "stratum": identity["stratum"],
            "family": identity["family"],
            "primary_cell": identity["primary_cell"],
            "axes_ABC": identity.get("axes_ABC", []),
            "foundation_border_axes_ABC": identity.get(
                "foundation_border_axes_ABC", []
            ),
        }
        (directory / "metadata" / f"trench_{slot_index}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        slot_rows.append(
            {
                "slot_index": slot_index,
                "map_id": identity["map_id"],
                "split": identity["split"],
                "stratum": identity["stratum"],
                "family": identity["family"],
                "primary_cell": identity["primary_cell"],
                "source_id": identity["source_id"],
            }
        )
    write_jsonl(directory / "manifest.jsonl", slot_rows)
    return slot_rows


def repeated(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    quotient, remainder = divmod(size, len(records))
    return records * quotient + records[:remainder]


def render_gallery(
    path: Path,
    records: list[dict[str, Any]],
    samples: dict[str, Any],
    title: str,
) -> None:
    selected = []
    for family in FAMILY_ORDER:
        for cell in [
            item
            for item in CELLS
            if item.stratum == records[0]["stratum"]
            and item.family == family
        ]:
            selected.append(
                next(
                    record
                    for record in records
                    if record["family"] == family
                    and record["primary_cell"] == cell.name
                )
            )
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), constrained_layout=True)
    for axis, record in zip(axes.ravel(), selected):
        sample = samples[record["map_id"]]
        target = np.asarray(sample.target)
        occupancy = np.asarray(sample.occupancy, dtype=bool)
        dumpability = np.asarray(sample.dumpability, dtype=bool)
        code = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        code[target < 0] = 1
        code[target > 0] = 2
        code[~dumpability & ~occupancy] = 3
        code[occupancy] = 4
        axis.imshow(
            code,
            cmap=matplotlib.colors.ListedColormap(
                ["#f3e6c3", "#ef8b23", "#4daa6b", "#808080", "#111111"]
            ),
            vmin=0,
            vmax=4,
            interpolation="nearest",
        )
        axis.set_title(
            record["primary_cell"].replace("_", " ")
            + "\n"
            + (
                f"p50 {record['eligibility_distance_p50_tiles']:.1f}, "
                f"p95 {record['eligibility_distance_p95_tiles']:.1f}, "
                f"cap {record['nearby_reachable_capacity_ratio']:.1f}x"
            ),
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0, 0].set_ylabel("foundations", fontsize=10)
    axes[1, 0].set_ylabel("trenches", fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate_written_bank(
    output: Path,
    all_identities: list[dict[str, Any]],
    expected_directories: dict[str, int],
) -> dict[str, Any]:
    for relative, expected_count in expected_directories.items():
        directory = output / relative
        rows = [
            json.loads(line)
            for line in (directory / "manifest.jsonl").read_text().splitlines()
        ]
        if len(rows) != expected_count:
            raise RuntimeError(
                f"{relative} has {len(rows)} slots, expected {expected_count}"
            )
        for index in range(1, expected_count + 1):
            shapes = []
            for folder in (
                "images",
                "occupancy",
                "dumpability",
                "actions",
                "distance",
            ):
                path = directory / folder / f"img_{index}.npy"
                if not path.exists():
                    raise RuntimeError(f"missing {path}")
                shapes.append(np.load(path).shape)
            if any(shape != (MAP_SIZE, MAP_SIZE) for shape in shapes):
                raise RuntimeError(
                    f"invalid shapes in {relative}/img_{index}: {shapes}"
                )
            distance = np.load(
                directory / "distance" / f"img_{index}.npy"
            )
            if (
                not np.all(np.isfinite(distance))
                or distance.min() < 0
                or distance.max() > 1
            ):
                raise RuntimeError(
                    f"invalid reward distance in {relative}/img_{index}"
                )

    sources = {
        split: {
            row["source_id"]
            for row in all_identities
            if row["split"] == split
        }
        for split in SPLIT_ORDER
    }
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            overlap = sources[left] & sources[right]
            if overlap:
                raise RuntimeError(
                    f"source overlap between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )
    dig_hashes = [row["dig_identity_sha256"] for row in all_identities]
    if len(dig_hashes) != len(set(dig_hashes)):
        raise RuntimeError("dig identities overlap across frozen splits")
    return {
        "status": "passed",
        "identity_count": len(all_identities),
        "source_disjoint": True,
        "dig_identity_disjoint": True,
        "directories": expected_directories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--source-foundations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generator_root = args.generator_root.resolve()
    source_foundations = args.source_foundations.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"output already exists; choose a fresh frozen bank path: {output}"
        )
    output.mkdir(parents=True)

    v5, geometry_factory_type = load_generator(generator_root)
    geometry_factory = geometry_factory_type(source_foundations)
    used_osm_sources: set[int] = set()
    used_dig_hashes: set[str] = set()
    identities_by_split: dict[str, list[dict[str, Any]]] = {}
    samples: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        identities, split_samples = generate_identities(
            v5,
            geometry_factory,
            split,
            used_osm_sources,
            used_dig_hashes,
        )
        identities_by_split[split] = identities
        samples.update(split_samples)

    manifests = output / "manifests"
    galleries = output / "galleries"
    manifests.mkdir()
    galleries.mkdir()
    all_identities = [
        record
        for split in SPLIT_ORDER
        for record in identities_by_split[split]
    ]
    write_jsonl(manifests / "identities.jsonl", all_identities)
    fieldnames = sorted(
        {
            key
            for record in all_identities
            for key in record
            if not isinstance(record[key], (list, dict))
        }
    )
    with (manifests / "identities.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {key: record.get(key, "") for key in fieldnames}
                for record in all_identities
            ]
        )

    expected_directories: dict[str, int] = {}
    train = identities_by_split["train"]
    train_by_stratum = {
        stratum: [
            record for record in train if record["stratum"] == stratum
        ]
        for stratum in STRATUM_ORDER
    }
    train_levels = {
        "train/local_M0": repeated(train_by_stratum["M0"], 256),
        "train/local_M1": repeated(train_by_stratum["M1"], 256),
        "train/local_M2_terminal": train,
    }
    for index, (relative, records) in enumerate(train_levels.items()):
        write_level(
            output / relative,
            records,
            samples,
            np.random.default_rng(2_026_072_400 + index),
        )
        expected_directories[relative] = 256

    for split in ("development", "sealed"):
        for stratum_index, stratum in enumerate(STRATUM_ORDER):
            records = [
                record
                for record in identities_by_split[split]
                if record["stratum"] == stratum
            ]
            relative = f"{split}/{stratum}"
            write_level(
                output / relative,
                records,
                samples,
                np.random.default_rng(
                    SPLIT_BASE_SEEDS[split] + stratum_index
                ),
            )
            expected_directories[relative] = 64

    for split in SPLIT_ORDER:
        for stratum in STRATUM_ORDER:
            records = [
                record
                for record in identities_by_split[split]
                if record["stratum"] == stratum
            ]
            render_gallery(
                galleries / f"{split}_{stratum}.png",
                records,
                samples,
                f"{split} {stratum} frozen primary cells",
            )

    validation = validate_written_bank(
        output, all_identities, expected_directories
    )
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )

    generator_files = [
        generator_root / f"generate_prototypes{suffix}.py"
        for suffix in ("", "_v2", "_v3", "_v4", "_v5")
    ]
    provenance = {
        "schema": "terra_training_design_v1",
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "generator_files": {
            str(path): sha256_file(path) for path in generator_files
        },
        "source_foundations": str(source_foundations),
        "split_base_seeds": SPLIT_BASE_SEEDS,
        "terminal_universe": "train/local_M2_terminal",
        "flat_training_path": "train/local_M2_terminal",
        "staged_training_paths": [
            "train/local_M0",
            "train/local_M1",
            "train/local_M2_terminal",
        ],
        "identity_manifest_sha256": sha256_file(
            manifests / "identities.jsonl"
        ),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    (output / "README.md").write_text(
        "# Terra training-design v1 frozen banks\n\n"
        "- `train/local_M0`: 256 slots from 64 M0 identities.\n"
        "- `train/local_M1`: 256 slots from 96 M1 identities.\n"
        "- `train/local_M2_terminal`: the common 256-identity terminal "
        "universe and the direct flat-arm path.\n"
        "- `development/{M0,M1,M2}` and `sealed/{M0,M1,M2}`: 64 "
        "source-disjoint identities each.\n"
        "- `galleries/`: one inspectable primary-cell gallery per split and "
        "stratum.\n"
        "- `validation.json`: strict shape, reward-distance, and "
        "source-disjointness result.\n"
        "- `provenance.json`: frozen builder, generator, seed, and manifest "
        "hashes.\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
