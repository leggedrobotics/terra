#!/usr/bin/env python3
"""Review-driven revision of the Terra procedural map prototype.

This revision implements Lorenzo's first visual review:

* easy, large, nearby dump zones dominate;
* far edge and haul-away layouts are explicitly rare/late;
* every dump zone has conservative single-layer capacity tied to dig volume;
* curved trenches are replaced by segmented and N-intersection networks;
* disconnected structural footings/pillar pads are added.

The implementation reuses rendering, Terra-format output, site constraints, and
the footprint-aware static gate from ``generate_prototypes.py``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import generate_prototypes as base


LEGACY_DUMP_FACTORY = base.DumpFactory


GEOMETRIES = (
    "foundation_osm",
    "foundation_procedural",
    "foundation_structural",
    "trench_segmented",
    "trench_intersections",
)

DUMP_STYLES = (
    "broad_nearby",
    "near_apron_large",
    "one_side_near",
    "separated_zones",
    "one_side_far",
    "haul_away_edge",
)

SITE_STYLES = base.SITE_STYLES

TARGET_DUMP_WEIGHTS = {
    "broad_nearby": 0.25,
    "near_apron_large": 0.30,
    "one_side_near": 0.20,
    "separated_zones": 0.20,
    "one_side_far": 0.04,
    "haul_away_edge": 0.01,
}

TARGET_GEOMETRY_WEIGHTS = {
    "foundation_osm": 0.35,
    "foundation_procedural": 0.15,
    "foundation_structural": 0.15,
    "trench_segmented": 0.25,
    "trench_intersections": 0.10,
}

TARGET_SITE_WEIGHTS = {
    "light": 0.45,
    "scattered_objects": 0.25,
    "access_road": 0.15,
    "gapped_wall": 0.10,
    "combined": 0.05,
}

CAPACITY_RANGES = {
    "broad_nearby": (1.75, 2.25),
    "near_apron_large": (1.55, 2.00),
    "one_side_near": (1.30, 1.65),
    "separated_zones": (1.30, 1.65),
    "one_side_far": (1.25, 1.50),
    "haul_away_edge": (1.50, 2.00),
}

DIFFICULTY_TIERS = {
    "broad_nearby": "early",
    "near_apron_large": "early",
    "one_side_near": "early_mid",
    "separated_zones": "mid",
    "one_side_far": "late_rare",
    "haul_away_edge": "late_rare",
}

MAX_MEDIAN_DISTANCE_TILES = {
    "broad_nearby": 13.0,
    "near_apron_large": 12.0,
    "one_side_near": 13.0,
    "separated_zones": 16.0,
}

CURRICULUM_STAGES = {
    "stage_0_easy_nearby": {
        "advance_gate": "fixed easy replay success >= 95%",
        "geometry_weights": {
            "foundation_osm": 0.55,
            "foundation_procedural": 0.15,
            "foundation_structural": 0.00,
            "trench_segmented": 0.30,
            "trench_intersections": 0.00,
        },
        "dump_weights": {
            "broad_nearby": 0.45,
            "near_apron_large": 0.40,
            "one_side_near": 0.15,
            "separated_zones": 0.00,
            "one_side_far": 0.00,
            "haul_away_edge": 0.00,
        },
        "site_weights": {
            "light": 0.80,
            "scattered_objects": 0.20,
            "access_road": 0.00,
            "gapped_wall": 0.00,
            "combined": 0.00,
        },
    },
    "stage_1_nearby_constraints": {
        "advance_gate": "fixed near/mid replay success >= 90%",
        "geometry_weights": TARGET_GEOMETRY_WEIGHTS,
        "dump_weights": {
            "broad_nearby": 0.30,
            "near_apron_large": 0.30,
            "one_side_near": 0.25,
            "separated_zones": 0.15,
            "one_side_far": 0.00,
            "haul_away_edge": 0.00,
        },
        "site_weights": {
            "light": 0.55,
            "scattered_objects": 0.25,
            "access_road": 0.15,
            "gapped_wall": 0.05,
            "combined": 0.00,
        },
    },
    "stage_2_full_distribution": {
        "advance_gate": "terminal training mixture",
        "geometry_weights": TARGET_GEOMETRY_WEIGHTS,
        "dump_weights": TARGET_DUMP_WEIGHTS,
        "site_weights": TARGET_SITE_WEIGHTS,
    },
}


def angle_difference(angles: np.ndarray, target: float) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(angles - target), np.cos(angles - target)))


def seed_near_angle(
    allowed: np.ndarray,
    dig: np.ndarray,
    target_angle: float,
    preferred_distance: float,
) -> np.ndarray:
    yy, xx = np.indices(dig.shape)
    cy, cx = ndi.center_of_mass(dig)
    distance = ndi.distance_transform_edt(~dig)
    angle = np.arctan2(yy - cy, xx - cx)
    score = (
        np.abs(distance - preferred_distance)
        + 2.0 * angle_difference(angle, target_angle)
    )
    score[~allowed] = np.inf
    seed = np.zeros_like(dig, dtype=bool)
    if np.isfinite(score).any():
        y, x = np.unravel_index(np.argmin(score), score.shape)
        seed[y, x] = True
    return seed


class GeometryFactoryV2(base.GeometryFactory):
    @staticmethod
    def foundation_structural(
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Disconnected strip footings, bearing walls, and pillar pads."""
        for _ in range(160):
            dig = np.zeros((base.MAP_SIZE, base.MAP_SIZE), dtype=bool)
            center = np.array(
                [float(rng.uniform(27, 37)), float(rng.uniform(27, 37))]
            )
            global_angle = float(
                rng.choice(np.deg2rad(np.arange(0, 180, 15)))
            )
            tangent = np.array(
                [math.sin(global_angle), math.cos(global_angle)]
            )
            normal = np.array([tangent[1], -tangent[0]])

            n_walls = int(rng.integers(1, 4))
            for _wall in range(n_walls):
                offset = (
                    tangent * float(rng.uniform(-7, 7))
                    + normal * float(rng.uniform(-9, 9))
                )
                wall_center = center + offset
                dig |= base.rotated_rectangle(
                    tuple(wall_center),
                    length=float(rng.uniform(12, 27)),
                    width=float(rng.uniform(2.0, 4.2)),
                    angle=global_angle
                    + float(rng.choice([0, 0, math.pi / 2])),
                )

            n_pads = int(rng.integers(3, 8))
            for _pad in range(n_pads):
                offset = (
                    tangent * float(rng.uniform(-13, 13))
                    + normal * float(rng.uniform(-13, 13))
                )
                pad_center = center + offset
                dig |= base.rotated_rectangle(
                    tuple(pad_center),
                    length=float(rng.uniform(3.5, 6.5)),
                    width=float(rng.uniform(3.5, 6.5)),
                    angle=global_angle,
                )

            dig = ndi.binary_closing(dig, structure=base.binary_disk(1))
            labels, n_components = ndi.label(
                dig, structure=np.ones((3, 3), dtype=np.uint8)
            )
            if n_components < 2:
                continue
            counts = np.bincount(labels.ravel())
            keep = np.zeros_like(dig, dtype=bool)
            for label_idx in range(1, n_components + 1):
                if counts[label_idx] >= 8:
                    keep |= labels == label_idx
            dig = keep
            labels, n_components = ndi.label(
                dig, structure=np.ones((3, 3), dtype=np.uint8)
            )
            if not (2 <= n_components <= 10):
                continue
            ys, xs = np.where(dig)
            if (
                80 <= int(dig.sum()) <= 310
                and ys.min() >= 9
                and xs.min() >= 9
                and ys.max() <= base.MAP_SIZE - 10
                and xs.max() <= base.MAP_SIZE - 10
            ):
                return dig, {
                    "structural_components": int(n_components),
                    "bearing_wall_strips": n_walls,
                    "pillar_or_pad_footings": n_pads,
                    "structural_angle_deg": round(
                        math.degrees(global_angle), 1
                    ),
                }
        raise RuntimeError("Could not construct structural foundation")

    def trench_segmented(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        dig, metadata = self.trench_straight(rng)
        metadata["trench_geometry"] = "straight_or_segmented"
        return dig, metadata

    @staticmethod
    def trench_intersections(
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate realistic T/X and multi-junction trench networks."""
        for _ in range(160):
            center = np.array(
                [float(rng.uniform(28, 36)), float(rng.uniform(28, 36))]
            )
            heading = float(rng.uniform(0, math.pi))
            direction = np.array([math.sin(heading), math.cos(heading)])
            normal = np.array([direction[1], -direction[0]])
            main_length = float(rng.uniform(27, 40))
            main_points = np.vstack(
                [
                    center - direction * main_length / 2,
                    center + direction * main_length / 2,
                ]
            )
            if np.any(main_points < 7) or np.any(
                main_points > base.MAP_SIZE - 8
            ):
                continue

            dig = np.zeros((base.MAP_SIZE, base.MAP_SIZE), dtype=bool)
            main_radius = int(rng.integers(1, 3))
            dig |= base.rasterize_polyline(main_points, radius=main_radius)

            n_crossings = int(rng.integers(1, 4))
            along_positions = np.sort(
                rng.uniform(-0.34, 0.34, size=n_crossings)
            )
            double_sided = 0
            successful = True
            for along in along_positions:
                junction = center + direction * float(along * main_length)
                branch_heading = heading + math.pi / 2 + float(
                    rng.uniform(-0.24, 0.24)
                )
                branch_direction = np.array(
                    [math.sin(branch_heading), math.cos(branch_heading)]
                )
                length_a = float(rng.uniform(9, 18))
                if rng.random() < 0.58:
                    length_b = float(rng.uniform(9, 18))
                    points = np.vstack(
                        [
                            junction - branch_direction * length_a,
                            junction,
                            junction + branch_direction * length_b,
                        ]
                    )
                    double_sided += 1
                else:
                    side_sign = float(rng.choice([-1, 1]))
                    points = np.vstack(
                        [
                            junction,
                            junction
                            + side_sign * branch_direction * length_a,
                        ]
                    )
                if np.any(points < 7) or np.any(points > base.MAP_SIZE - 8):
                    successful = False
                    break
                dig |= base.rasterize_polyline(
                    points, radius=int(rng.integers(1, 3))
                )
            if not successful:
                continue
            dig = base.largest_component(dig)
            if 75 <= int(dig.sum()) <= 360:
                return dig, {
                    "intersecting_trench_paths": 1 + n_crossings,
                    "intersection_junctions": n_crossings,
                    "intersection_double_sided": double_sided,
                    "intersection_branches": n_crossings + double_sided + 2,
                    "trench_geometry": "T_X_multi_junction_network",
                }
        raise RuntimeError("Could not construct intersecting trenches")


class DumpFactoryV2:
    @staticmethod
    def broad_nearby(
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del side
        yy, xx = np.indices(dig.shape)
        cy, cx = ndi.center_of_mass(dig)
        distance = ndi.distance_transform_edt(~dig)
        angles = np.arctan2(yy - cy, xx - cx)
        center_angle = float(rng.uniform(-math.pi, math.pi))
        sector_width = float(rng.uniform(math.radians(250), math.radians(340)))
        allowed = (
            (distance >= 3)
            & (distance <= int(rng.integers(17, 23)))
            & (angle_difference(angles, center_angle) <= sector_width / 2)
        )
        seed = seed_near_angle(allowed, dig, center_angle, preferred_distance=4.5)
        target = base.grow_region(allowed, seed, target_area, rng)
        return target, {
            "dump_side": "broad_nearby",
            "dump_components_requested": 1,
            "dump_sector_degrees": round(math.degrees(sector_width), 1),
        }

    @staticmethod
    def near_apron_large(
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del side
        yy, xx = np.indices(dig.shape)
        cy, cx = ndi.center_of_mass(dig)
        distance = ndi.distance_transform_edt(~dig)
        angles = np.arctan2(yy - cy, xx - cx)
        center_angle = float(rng.uniform(-math.pi, math.pi))
        sector_width = float(rng.uniform(math.radians(170), math.radians(285)))
        allowed = (
            (distance >= 3)
            & (distance <= int(rng.integers(11, 16)))
            & (angle_difference(angles, center_angle) <= sector_width / 2)
        )
        seed = seed_near_angle(allowed, dig, center_angle, preferred_distance=4.5)
        target = base.grow_region(allowed, seed, target_area, rng)
        return target, {
            "dump_side": "near_apron",
            "dump_components_requested": 1,
            "dump_sector_degrees": round(math.degrees(sector_width), 1),
        }

    @staticmethod
    def one_side_near(
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        yy, xx = np.indices(dig.shape)
        cy, cx = ndi.center_of_mass(dig)
        directions = (
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, -1.0]),
        )
        direction = directions[side]
        tangent = np.array([direction[1], -direction[0]])
        rel_y = yy - cy
        rel_x = xx - cx
        projection = rel_y * direction[0] + rel_x * direction[1]
        lateral = np.abs(rel_y * tangent[0] + rel_x * tangent[1])
        distance = ndi.distance_transform_edt(~dig)
        allowed = (
            (projection >= 3)
            & (projection <= 22)
            & (lateral <= int(rng.integers(15, 25)))
            & (distance >= 3)
            & (distance <= 21)
        )
        target_angle = math.atan2(direction[0], direction[1])
        seed = seed_near_angle(allowed, dig, target_angle, preferred_distance=4.5)
        target = base.grow_region(allowed, seed, target_area, rng)
        return target, {
            "dump_side": base.SIDE_NAMES[side],
            "dump_components_requested": 1,
            "distance_bucket": "near",
        }

    @staticmethod
    def separated_zones(
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del side
        yy, xx = np.indices(dig.shape)
        cy, cx = ndi.center_of_mass(dig)
        distance = ndi.distance_transform_edt(~dig)
        n_zones = int(rng.integers(2, 4))
        areas = np.full(n_zones, target_area // n_zones, dtype=int)
        areas[: target_area % n_zones] += 1
        base_angle = float(rng.uniform(-math.pi, math.pi))
        zone_angles = (
            base_angle
            + np.linspace(0, 2 * math.pi, n_zones, endpoint=False)
            + rng.uniform(-0.25, 0.25, size=n_zones)
        )
        target = np.zeros_like(dig, dtype=bool)
        for angle, zone_area in zip(zone_angles, areas):
            angles = np.arctan2(yy - cy, xx - cx)
            allowed = (
                (distance >= 4)
                & (distance <= int(rng.integers(15, 22)))
                & (angle_difference(angles, float(angle)) <= math.radians(38))
            )
            allowed &= ~ndi.binary_dilation(
                target, structure=base.binary_disk(4)
            )
            seed = seed_near_angle(
                allowed, dig, float(angle), preferred_distance=float(rng.uniform(8, 13))
            )
            zone = base.grow_region(allowed, seed, int(zone_area), rng)
            if int(zone.sum()) < int(zone_area):
                return np.zeros_like(dig), {
                    "dump_side": "separated_nearby",
                    "dump_components_requested": n_zones,
                }
            target |= zone
        return target, {
            "dump_side": "separated_nearby",
            "dump_components_requested": n_zones,
            "distance_bucket": "near_medium",
        }

    @staticmethod
    def one_side_far(
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target, metadata = LEGACY_DUMP_FACTORY.continuous_one_side(
            dig, target_area, side, rng
        )
        metadata["distance_bucket"] = "far"
        return target, metadata

    @staticmethod
    def haul_away_edge(
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        depth, tangent = base.side_coordinates(side)
        max_depth = int(rng.integers(8, 14))
        span = int(
            np.clip(math.ceil(target_area / max_depth) + 6, 28, base.MAP_SIZE - 4)
        )
        center = int(rng.integers(18, 47))
        lo = max(1, center - span // 2)
        hi = min(base.MAP_SIZE - 2, lo + span)
        allowed = (
            (depth < max_depth)
            & (tangent >= lo)
            & (tangent <= hi)
            & ~ndi.binary_dilation(dig, structure=base.binary_disk(2))
        )
        seed = (
            (depth <= 1)
            & (tangent >= lo + 1)
            & (tangent <= hi - 1)
        )
        target = base.grow_region(allowed, seed, target_area, rng)
        return target, {
            "dump_side": base.SIDE_NAMES[side],
            "dump_components_requested": 1,
            "distance_bucket": "far",
            "edge_depth_tiles": max_depth,
        }

    def make(
        self,
        style: str,
        dig: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        capacity_low, capacity_high = CAPACITY_RANGES[style]
        capacity_factor = float(rng.uniform(capacity_low, capacity_high))
        target_area = int(math.ceil(int(dig.sum()) * capacity_factor))
        side = int(rng.integers(0, 4))
        method = getattr(self, style)
        target, metadata = method(dig, target_area, side, rng)
        metadata.update(
            {
                "dump_target_cells_requested": target_area,
                "capacity_factor_required": round(capacity_factor, 5),
                "difficulty_tier": DIFFICULTY_TIERS[style],
                "proposed_target_weight": TARGET_DUMP_WEIGHTS[style],
            }
        )
        return target, metadata


def dig_to_dump_distance_metrics(
    dig: np.ndarray, dump: np.ndarray
) -> dict[str, float]:
    distance = ndi.distance_transform_edt(~dump)
    values = distance[dig]
    return {
        "dig_dump_distance_min_tiles": round(float(np.min(values)), 4),
        "dig_dump_distance_median_tiles": round(float(np.median(values)), 4),
        "dig_dump_distance_p95_tiles": round(float(np.quantile(values, 0.95)), 4),
    }


def make_sample_v2(
    geometry_factory: GeometryFactoryV2,
    geometry: str,
    dump_style: str,
    site_style: str,
    seed: int,
    max_attempts: int,
) -> tuple[base.Sample | None, Counter[str]]:
    rejections: Counter[str] = Counter()
    for attempt in range(max_attempts):
        attempt_seed = int(
            np.random.SeedSequence([seed, attempt]).generate_state(1)[0]
        )
        rng = np.random.default_rng(attempt_seed)
        try:
            dig, geometry_meta = geometry_factory.make(geometry, rng)
            dump, dump_meta = DumpFactoryV2().make(dump_style, dig, rng)
            requested_cells = int(dump_meta["dump_target_cells_requested"])
            if int(dump.sum()) < requested_cells:
                rejections["capacity_generation_shortfall"] += 1
                continue
            occupancy, dumpability, corridor, site_meta = base.SiteFactory.make(
                site_style, dig, dump, rng
            )
        except RuntimeError as exc:
            rejections[str(exc)] += 1
            continue

        target = np.zeros((base.MAP_SIZE, base.MAP_SIZE), dtype=np.int8)
        target[dig] = -1
        target[dump & ~dig] = 1
        actual_dump_components = base.target_components(target)
        requested_components = int(dump_meta["dump_components_requested"])
        if actual_dump_components != requested_components:
            rejections["dump_component_contract"] += 1
            continue

        gate = base.static_gate(target, occupancy, dumpability)
        if not gate.accepted:
            rejections[gate.reason] += 1
            continue

        capacity_factor_required = float(
            dump_meta["capacity_factor_required"]
        )
        capacity_ratio = float(dump.sum() / max(1, dig.sum()))
        reachable_capacity_ratio = float(
            gate.reachable_dump_cells_post / max(1, dig.sum())
        )
        if capacity_ratio + 1e-8 < capacity_factor_required:
            rejections["capacity_ratio_contract"] += 1
            continue
        if reachable_capacity_ratio + 1e-8 < capacity_factor_required:
            rejections["reachable_capacity_contract"] += 1
            continue

        distance_metrics = dig_to_dump_distance_metrics(dig, dump)
        max_median_distance = MAX_MEDIAN_DISTANCE_TILES.get(dump_style)
        if (
            max_median_distance is not None
            and distance_metrics["dig_dump_distance_median_tiles"]
            > max_median_distance
        ):
            rejections["near_distance_contract"] += 1
            continue

        distance = base.compute_geodesic_distance(target, occupancy)
        action = np.zeros_like(target, dtype=np.int8)
        _, dig_components = ndi.label(
            dig, structure=np.ones((3, 3), dtype=np.uint8)
        )
        metadata: dict[str, Any] = {
            "schema": "site_constraints_v2_review",
            "seed": seed,
            "attempt": attempt,
            "attempt_seed": attempt_seed,
            "geometry": geometry,
            "dump_style": dump_style,
            "site_style": site_style,
            "dig_cells": int(dig.sum()),
            "dig_components_actual": int(dig_components),
            "dump_cells": int(dump.sum()),
            "dump_to_dig_area_ratio": round(capacity_ratio, 4),
            "reachable_dump_to_dig_ratio": round(
                reachable_capacity_ratio, 4
            ),
            "dump_components_actual": actual_dump_components,
            "obstacle_cells": int(occupancy.sum()),
            "nondump_cells": int((~dumpability & ~occupancy).sum()),
            "tile_size_m": base.TILE_SIZE_M,
            "proposed_geometry_weight": TARGET_GEOMETRY_WEIGHTS[geometry],
            "proposed_site_weight": TARGET_SITE_WEIGHTS[site_style],
            "static_gate_is_action_witness": False,
            **distance_metrics,
            **geometry_meta,
            **dump_meta,
            **site_meta,
        }
        return (
            base.Sample(
                target=target,
                occupancy=occupancy,
                dumpability=dumpability,
                action=action,
                distance=distance,
                service_corridor=corridor,
                metadata=metadata,
                gate=gate,
            ),
            rejections,
        )
    return None, rejections


def render_matrix_v2(
    samples: dict[tuple[str, str, str, int], base.Sample],
    geometry: str,
    variant: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(DUMP_STYLES),
        len(SITE_STYLES),
        figsize=(16, 19),
        constrained_layout=True,
    )
    for row, dump_style in enumerate(DUMP_STYLES):
        for col, site_style in enumerate(SITE_STYLES):
            ax = axes[row, col]
            sample = samples[(geometry, dump_style, site_style, variant)]
            ax.imshow(
                base.render_code(sample),
                cmap=base.COLORS,
                vmin=0,
                vmax=4,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(site_style.replace("_", "\n"), fontsize=11)
            if col == 0:
                tier = sample.metadata["difficulty_tier"]
                weight = int(
                    round(sample.metadata["proposed_target_weight"] * 100)
                )
                ax.set_ylabel(
                    f"{dump_style.replace('_', ' ')}\n{tier}; target {weight}%",
                    fontsize=9.5,
                )
            ax.text(
                1,
                62,
                f"D{sample.metadata['dig_cells']} / Z{sample.metadata['dump_cells']}\n"
                f"cap {sample.metadata['dump_to_dig_area_ratio']:.2f}x / "
                f"dist {sample.metadata['dig_dump_distance_median_tiles']:.1f}",
                fontsize=6.4,
                color="black",
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                },
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"site_constraints_v2 — {geometry} — variant {variant}", fontsize=16
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_variability_v2(
    samples: dict[tuple[str, str, str, int], base.Sample],
    dump_style: str,
    variants: int,
    path: Path,
) -> None:
    columns = len(SITE_STYLES) * variants
    fig, axes = plt.subplots(
        len(GEOMETRIES),
        columns,
        figsize=(3.0 * columns, 14.5),
        constrained_layout=True,
    )
    for row, geometry in enumerate(GEOMETRIES):
        for variant in range(variants):
            for site_idx, site_style in enumerate(SITE_STYLES):
                col = variant * len(SITE_STYLES) + site_idx
                ax = axes[row, col]
                sample = samples[(geometry, dump_style, site_style, variant)]
                ax.imshow(
                    base.render_code(sample),
                    cmap=base.COLORS,
                    vmin=0,
                    vmax=4,
                    interpolation="nearest",
                )
                if row == 0:
                    ax.set_title(
                        f"v{variant} {site_style.replace('_', ' ')}",
                        fontsize=8,
                    )
                if col == 0:
                    ax.set_ylabel(geometry.replace("_", "\n"), fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(
        f"v2 variability — {dump_style.replace('_', ' ')}", fontsize=16
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_readme_v2(output: Path, variants: int, accepted: int) -> None:
    variant_word = "variant" if variants == 1 else "variants"
    lines = [
        "# `site_constraints_v2` review set",
        "",
        f"This folder contains **{accepted} maps** and {variants} {variant_word} of every exact combination.",
        "",
        "This revision reflects Lorenzo's first review:",
        "",
        "- nearby, high-capacity dumping is common and early;",
        "- far-side and haul-away edge cases are explicitly rare and late;",
        "- curved trenches were removed from the primary bank;",
        "- segmented and N-intersection trench networks were added;",
        "- disconnected bearing-wall and pillar/pad foundation targets were added;",
        "- every target passes a conservative reachable single-layer soil-capacity check.",
        "- every nearby style passes a measured excavation-to-dump distance limit.",
        "",
        "## Proposed final mixture",
        "",
        "### Geometry",
        "",
    ]
    for name, weight in TARGET_GEOMETRY_WEIGHTS.items():
        lines.append(f"- `{name}`: {weight:.0%}")
    lines.extend(["", "### Dump layout", ""])
    for name, weight in TARGET_DUMP_WEIGHTS.items():
        lines.append(
            f"- `{name}`: {weight:.0%} ({DIFFICULTY_TIERS[name]})"
        )
    lines.extend(["", "### Site constraints", ""])
    for name, weight in TARGET_SITE_WEIGHTS.items():
        lines.append(f"- `{name}`: {weight:.0%}")
    lines.extend(
        [
            "",
            "## Inspection",
            "",
            "- `matrices/`: dump layouts by rows, site constraints by columns",
            "- `variability/`: two variants grouped by dump algorithm",
            "- `previews/`: one labeled image per map",
            "- `manifest.csv`: exact seed, capacity, distance, geometry, and gate metrics",
            "- `dataset/`: Terra-loadable arrays",
            "- `curriculum_proposal.json`: staged sampling weights and promotion gates",
            "",
            "Colours: orange dig, green terminal dump, grey traversable non-dump road, black obstacle.",
            "",
            "## Capacity meaning",
            "",
            "The generator conservatively requires reachable dump-zone tile area to exceed",
            "excavated tile volume by the style-specific factor. This assumes only one",
            "unit of soil per target tile, even though current Terra dynamics can stack",
            "soil. It therefore gives the edge/road combinations explicit reserve rather",
            "than relying on visual size or unlimited piles.",
            "",
            "The `haul_away_edge` family reserves 1.50-2.00 dump cells per dig cell,",
            "including the access-road cross-product. Far edge and haul-away maps are",
            "absent from stages 0 and 1 and together occupy only 5% of stage 2.",
            "",
            "## Remaining limitation",
            "",
            "The gate is footprint- and workspace-aware but still static. Production",
            "promotion needs a Terra action-level witness under dynamic excavation and",
            "soil-pile blockage.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    # The base main owns deterministic enumeration, Terra-format output, and
    # manifest generation. Replace only the explicitly reviewed components.
    base.GEOMETRIES = GEOMETRIES
    base.DUMP_STYLES = DUMP_STYLES
    base.SITE_STYLES = SITE_STYLES
    base.GeometryFactory = GeometryFactoryV2
    base.DumpFactory = DumpFactoryV2
    base.make_sample = make_sample_v2
    base.render_matrix = render_matrix_v2
    base.render_variability = render_variability_v2
    base.write_readme = write_readme_v2
    args = base.parse_args()
    base.main()

    output = args.output.resolve()
    summary_path = output / "generation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "schema": "site_constraints_v2_review",
            "target_geometry_weights": TARGET_GEOMETRY_WEIGHTS,
            "target_dump_weights": TARGET_DUMP_WEIGHTS,
            "target_site_weights": TARGET_SITE_WEIGHTS,
            "capacity_ranges": CAPACITY_RANGES,
            "max_median_distance_tiles": MAX_MEDIAN_DISTANCE_TILES,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "curriculum_proposal.json").write_text(
        json.dumps(
            {
                "schema": "site_constraints_v2_curriculum_proposal",
                "stages": CURRICULUM_STAGES,
                "held_out_stress_evaluation": {
                    "sampling": "balanced across all dump and site styles",
                    "purpose": "measure rare-tail robustness without making long-haul maps common during early training",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
