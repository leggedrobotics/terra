#!/usr/bin/env python3
"""Generate an inspectable, compositional Terra site-constraints prototype.

This is intentionally an artifact-side review generator. It does not modify the
canonical Terra dataset or active training jobs.

The generator crosses:

* four excavation geometry styles,
* five terminal dump-layout styles, and
* five site-constraint styles.

Every accepted sample passes structural checks plus a footprint-aware static
reachability proxy. The proxy is useful for rejecting obviously impossible
maps, but it is not a substitute for a Terra action-level planner witness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from skimage.draw import line, polygon
from skimage.graph import route_through_array
from skimage.morphology import disk


MAP_SIZE = 64
TILE_SIZE_M = 44.0 / MAP_SIZE
AGENT_WIDTH_TILES = 5
AGENT_HEIGHT_TILES = 9
AGENT_ORIENTATIONS = 12
SPAWN_BORDER_TILES = 8
WORKSPACE_R_MIN_TILES = (
    0.5 + TILE_SIZE_M * max(AGENT_WIDTH_TILES, AGENT_HEIGHT_TILES) / 2
) / TILE_SIZE_M
WORKSPACE_R_MAX_TILES = WORKSPACE_R_MIN_TILES + 5

GEOMETRIES = (
    "foundation_osm",
    "foundation_procedural",
    "trench_straight",
    "trench_curved",
)
DUMP_STYLES = (
    "continuous_one_side",
    "irregular_one_side",
    "separated_zones",
    "haul_away_edge",
    "near_apron",
)
SITE_STYLES = (
    "light",
    "scattered_objects",
    "access_road",
    "gapped_wall",
    "combined",
)
SIDE_NAMES = ("north", "east", "south", "west")


@dataclass
class StaticGate:
    accepted: bool
    reason: str
    spawn_centers: int
    spawn_component_fraction: float
    dig_workspace_coverage_pre: float
    dig_workspace_coverage_post: float
    dump_workspace_coverage_post: float
    reachable_dump_cells_post: int
    pre_base_centers: int
    post_base_centers: int


@dataclass
class Sample:
    target: np.ndarray
    occupancy: np.ndarray
    dumpability: np.ndarray
    action: np.ndarray
    distance: np.ndarray
    service_corridor: np.ndarray
    metadata: dict[str, Any]
    gate: StaticGate


def numeric_key(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 10**9


def binary_disk(radius: int) -> np.ndarray:
    return disk(max(1, radius)).astype(bool)


def smooth_noise(
    rng: np.random.Generator,
    shape: tuple[int, int] = (MAP_SIZE, MAP_SIZE),
    sigma: float = 4.0,
) -> np.ndarray:
    field = ndi.gaussian_filter(rng.normal(size=shape), sigma=sigma, mode="reflect")
    field -= field.min()
    peak = field.max()
    if peak > 0:
        field /= peak
    return field


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()


def rotated_rectangle(
    center_yx: tuple[float, float],
    length: float,
    width: float,
    angle: float,
) -> np.ndarray:
    cy, cx = center_yx
    local = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    xy = local @ rotation.T
    xs = xy[:, 0] + cx
    ys = xy[:, 1] + cy
    rr, cc = polygon(ys, xs, shape=(MAP_SIZE, MAP_SIZE))
    out = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    out[rr, cc] = True
    return out


def rasterize_polyline(points_yx: np.ndarray, radius: int) -> np.ndarray:
    out = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    rounded = np.rint(points_yx).astype(int)
    for p0, p1 in zip(rounded[:-1], rounded[1:]):
        rr, cc = line(
            int(np.clip(p0[0], 0, MAP_SIZE - 1)),
            int(np.clip(p0[1], 0, MAP_SIZE - 1)),
            int(np.clip(p1[0], 0, MAP_SIZE - 1)),
            int(np.clip(p1[1], 0, MAP_SIZE - 1)),
        )
        out[rr, cc] = True
    return ndi.binary_dilation(out, structure=binary_disk(radius))


class GeometryFactory:
    def __init__(self, source_root: Path):
        candidates: list[np.ndarray] = []
        for path in sorted((source_root / "images").glob("img_*.npy"), key=numeric_key):
            target = np.load(path)
            dig = target < 0
            if not np.any(dig):
                continue
            ys, xs = np.where(dig)
            height = int(ys.max() - ys.min() + 1)
            width = int(xs.max() - xs.min() + 1)
            if 90 <= int(dig.sum()) <= 340 and max(height, width) <= 34:
                candidates.append(dig)
        if not candidates:
            raise RuntimeError(f"No usable foundation masks found under {source_root}")
        self.foundation_sources = candidates

    @staticmethod
    def _place_crop(
        crop: np.ndarray, rng: np.random.Generator, margin: int = 11
    ) -> np.ndarray | None:
        ys, xs = np.where(crop)
        if len(ys) == 0:
            return None
        crop = crop[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        h, w = crop.shape
        if h + 2 * margin >= MAP_SIZE or w + 2 * margin >= MAP_SIZE:
            return None
        center_y = int(rng.integers(max(margin + h // 2, 25), min(40, MAP_SIZE - margin - (h + 1) // 2)))
        center_x = int(rng.integers(max(margin + w // 2, 25), min(40, MAP_SIZE - margin - (w + 1) // 2)))
        y0 = center_y - h // 2
        x0 = center_x - w // 2
        out = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
        out[y0 : y0 + h, x0 : x0 + w] = crop
        return out

    def foundation_osm(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for _ in range(40):
            source_idx = int(rng.integers(0, len(self.foundation_sources)))
            dig = self.foundation_sources[source_idx]
            dig = np.rot90(dig, int(rng.integers(0, 4)))
            if rng.random() < 0.5:
                dig = np.fliplr(dig)
            placed = self._place_crop(dig, rng)
            if placed is not None:
                return placed, {"foundation_source_index": source_idx}
        raise RuntimeError("Could not place an OSM foundation source")

    @staticmethod
    def foundation_procedural(
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for _ in range(80):
            center = (float(rng.uniform(27, 37)), float(rng.uniform(27, 37)))
            angle = float(rng.choice(np.deg2rad(np.arange(0, 180, 15))))
            length = float(rng.uniform(15, 24))
            width = float(rng.uniform(8, 15))
            dig = rotated_rectangle(center, length, width, angle)

            n_wings = int(rng.integers(1, 4))
            for _wing in range(n_wings):
                along = float(rng.uniform(-0.35, 0.35) * length)
                across = float(rng.choice([-1, 1]) * rng.uniform(0.25, 0.55) * width)
                dx = along * math.cos(angle) - across * math.sin(angle)
                dy = along * math.sin(angle) + across * math.cos(angle)
                wing_center = (center[0] + dy, center[1] + dx)
                wing = rotated_rectangle(
                    wing_center,
                    float(rng.uniform(6, 13)),
                    float(rng.uniform(5, 10)),
                    angle + float(rng.choice([0, math.pi / 2])),
                )
                dig |= wing

            dig = ndi.binary_closing(dig, structure=binary_disk(1))
            dig = largest_component(dig)
            ys, xs = np.where(dig)
            if (
                110 <= int(dig.sum()) <= 340
                and ys.min() >= 10
                and xs.min() >= 10
                and ys.max() <= MAP_SIZE - 11
                and xs.max() <= MAP_SIZE - 11
            ):
                return dig, {
                    "foundation_angle_deg": round(math.degrees(angle), 1),
                    "foundation_wings": n_wings,
                }
        raise RuntimeError("Could not construct a procedural foundation")

    @staticmethod
    def trench_straight(
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for _ in range(80):
            n_segments = int(rng.integers(1, 4))
            heading = float(rng.uniform(0, 2 * math.pi))
            segment_lengths = rng.uniform(9, 16, size=n_segments)
            points = [np.array([float(rng.uniform(25, 39)), float(rng.uniform(25, 39))])]
            backward = sum(segment_lengths) / 2
            points[0] -= backward * np.array([math.sin(heading), math.cos(heading)])
            current_heading = heading
            for segment_length in segment_lengths:
                if len(points) > 1:
                    current_heading += float(
                        rng.choice(np.deg2rad(np.array([-45, -30, 0, 0, 30, 45])))
                    )
                delta = segment_length * np.array(
                    [math.sin(current_heading), math.cos(current_heading)]
                )
                points.append(points[-1] + delta)
            points_arr = np.asarray(points)
            if np.any(points_arr < 10) or np.any(points_arr > MAP_SIZE - 11):
                continue
            dig = rasterize_polyline(points_arr, int(rng.integers(1, 3)))
            if 45 <= int(dig.sum()) <= 190:
                return dig, {"trench_segments": n_segments}
        raise RuntimeError("Could not construct a straight trench")

    @staticmethod
    def trench_curved(
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for _ in range(80):
            length = float(rng.uniform(26, 38))
            heading = float(rng.uniform(0, 2 * math.pi))
            tangent = np.array([math.sin(heading), math.cos(heading)])
            normal = np.array([tangent[1], -tangent[0]])
            center = np.array([float(rng.uniform(27, 37)), float(rng.uniform(27, 37))])
            p0 = center - tangent * length / 2
            p3 = center + tangent * length / 2
            curvature = float(rng.choice([-1, 1]) * rng.uniform(5, 12))
            s_shape = float(rng.choice([-1, 1]) if rng.random() < 0.45 else 1)
            p1 = p0 + tangent * length / 3 + normal * curvature
            p2 = p0 + tangent * 2 * length / 3 + normal * curvature * s_shape
            control = np.vstack([p0, p1, p2, p3])
            if np.any(control < 8) or np.any(control > MAP_SIZE - 9):
                continue
            t = np.linspace(0.0, 1.0, 120)
            curve = (
                ((1 - t) ** 3)[:, None] * p0
                + (3 * (1 - t) ** 2 * t)[:, None] * p1
                + (3 * (1 - t) * t**2)[:, None] * p2
                + (t**3)[:, None] * p3
            )
            dig = rasterize_polyline(curve, int(rng.integers(1, 3)))
            if 50 <= int(dig.sum()) <= 190:
                return dig, {
                    "trench_curve": "S" if s_shape < 0 else "C",
                    "trench_curvature_tiles": round(abs(curvature), 2),
                }
        raise RuntimeError("Could not construct a curved trench")

    def make(
        self, style: str, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        method = getattr(self, style)
        return method(rng)


def side_coordinates(side: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((MAP_SIZE, MAP_SIZE))
    if side == 0:
        return yy, xx
    if side == 1:
        return MAP_SIZE - 1 - xx, yy
    if side == 2:
        return MAP_SIZE - 1 - yy, MAP_SIZE - 1 - xx
    return xx, MAP_SIZE - 1 - yy


def cells_to_mask(cells: Iterable[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    for y, x in cells:
        if 0 <= y < MAP_SIZE and 0 <= x < MAP_SIZE:
            out[y, x] = True
    return out


def grow_region(
    allowed: np.ndarray,
    seeds: np.ndarray,
    target_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cost-weighted region growth with a smooth random field."""
    target_cells = min(int(target_cells), int(allowed.sum()))
    seeds = seeds & allowed
    if not np.any(seeds) or target_cells <= 0:
        return np.zeros_like(allowed, dtype=bool)

    noise = smooth_noise(rng, sigma=float(rng.uniform(2.5, 5.5)))
    costs = 0.75 + 0.85 * noise
    distance = np.full(allowed.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for y, x in np.argwhere(seeds):
        distance[y, x] = 0.0
        heappush(heap, (0.0, int(y), int(x)))

    selected = np.zeros_like(allowed, dtype=bool)
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    )
    count = 0
    while heap and count < target_cells:
        current, y, x = heappop(heap)
        if current != distance[y, x] or selected[y, x]:
            continue
        selected[y, x] = True
        count += 1
        for dy, dx, step in moves:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < MAP_SIZE and 0 <= nx < MAP_SIZE and allowed[ny, nx]):
                continue
            proposed = current + step * float(costs[ny, nx])
            if proposed < distance[ny, nx]:
                distance[ny, nx] = proposed
                heappush(heap, (proposed, ny, nx))
    return selected


class DumpFactory:
    @staticmethod
    def _target_area(
        dig_cells: int, style: str, rng: np.random.Generator
    ) -> int:
        if style == "haul_away_edge":
            ratio = float(rng.uniform(0.68, 0.95))
        elif style == "separated_zones":
            ratio = float(rng.uniform(0.9, 1.25))
        else:
            ratio = float(rng.uniform(0.82, 1.18))
        return int(np.clip(round(dig_cells * ratio), 80, 300))

    @staticmethod
    def continuous_one_side(
        dig: np.ndarray, target_area: int, side: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        depth, tangent = side_coordinates(side)
        span = int(np.clip(target_area / float(rng.uniform(5.5, 8.0)), 22, 48))
        center = int(rng.integers(18, 47))
        lo = max(3, center - span // 2)
        hi = min(MAP_SIZE - 4, lo + span)
        allowed = (depth <= 15) & (tangent >= lo) & (tangent <= hi)
        allowed &= ~ndi.binary_dilation(dig, structure=binary_disk(2))
        seed = (depth <= 1) & (tangent >= lo + 2) & (tangent <= hi - 2)
        target = grow_region(allowed, seed, target_area, rng)
        return target, {"dump_side": SIDE_NAMES[side], "dump_components_requested": 1}

    @staticmethod
    def irregular_one_side(
        dig: np.ndarray, target_area: int, side: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        depth, tangent = side_coordinates(side)
        allowed = (depth <= int(rng.integers(15, 22))) & (tangent >= 5) & (tangent <= 58)
        allowed &= ~ndi.binary_dilation(dig, structure=binary_disk(2))
        n_lobes = int(rng.integers(2, 5))
        tangents = np.sort(rng.choice(np.arange(10, 54), size=n_lobes, replace=False))
        seed = np.zeros_like(dig, dtype=bool)
        anchor_depth = int(rng.integers(1, 4))
        previous: tuple[int, int] | None = None
        for tangent_value in tangents:
            branch_depth = int(rng.integers(3, 10))
            cells = np.argwhere((depth == branch_depth) & (tangent == tangent_value))
            if len(cells) == 0:
                continue
            y, x = map(int, cells[0])
            seed[y, x] = True
            anchor_cells = np.argwhere(
                (depth == anchor_depth) & (tangent == tangent_value)
            )
            if len(anchor_cells):
                ay, ax = map(int, anchor_cells[0])
                rr, cc = line(ay, ax, y, x)
                seed[rr, cc] = True
                if previous is not None:
                    rr, cc = line(previous[0], previous[1], ay, ax)
                    seed[rr, cc] = True
                previous = (ay, ax)
        seed = ndi.binary_dilation(seed, structure=binary_disk(1))
        target = grow_region(allowed, seed, target_area, rng)
        return target, {
            "dump_side": SIDE_NAMES[side],
            "dump_lobes_requested": n_lobes,
            "dump_components_requested": 1,
        }

    @staticmethod
    def separated_zones(
        dig: np.ndarray, target_area: int, side: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n_zones = int(rng.integers(2, 4))
        areas = np.full(n_zones, target_area // n_zones, dtype=int)
        areas[: target_area % n_zones] += 1
        target = np.zeros_like(dig, dtype=bool)
        chosen: list[tuple[int, int]] = []

        candidate_sides = [side]
        if rng.random() < 0.55:
            candidate_sides.append((side + int(rng.choice([1, 3]))) % 4)
        for zone_idx, zone_area in enumerate(areas):
            placed = False
            for _ in range(80):
                zone_side = int(rng.choice(candidate_sides))
                depth, tangent = side_coordinates(zone_side)
                d = int(rng.integers(3, 11))
                t = int(rng.integers(10, 54))
                cells = np.argwhere((depth == d) & (tangent == t))
                if len(cells) == 0:
                    continue
                y, x = map(int, cells[0])
                if any(math.hypot(y - py, x - px) < 16 for py, px in chosen):
                    continue
                allowed = (depth <= 17) & (np.abs(tangent - t) <= 11)
                allowed &= ~ndi.binary_dilation(dig | target, structure=binary_disk(4))
                seed = np.zeros_like(dig, dtype=bool)
                seed[y, x] = True
                zone = grow_region(allowed, seed, int(zone_area), rng)
                if int(zone.sum()) < int(zone_area * 0.85):
                    continue
                target |= zone
                chosen.append((y, x))
                placed = True
                break
            if not placed:
                return np.zeros_like(dig), {
                    "dump_side": SIDE_NAMES[side],
                    "dump_components_requested": n_zones,
                }
        return target, {
            "dump_side": SIDE_NAMES[side],
            "dump_components_requested": n_zones,
        }

    @staticmethod
    def haul_away_edge(
        dig: np.ndarray, target_area: int, side: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        depth, tangent = side_coordinates(side)
        max_depth = int(rng.integers(4, 7))
        span = int(np.clip(math.ceil(target_area / max_depth) + 4, 24, 55))
        center = int(rng.integers(18, 47))
        lo = max(3, center - span // 2)
        hi = min(MAP_SIZE - 4, lo + span)
        allowed = (depth < max_depth) & (tangent >= lo) & (tangent <= hi)
        allowed &= ~ndi.binary_dilation(dig, structure=binary_disk(2))
        seed = (depth == 0) & (tangent >= lo + 1) & (tangent <= hi - 1)
        target = grow_region(allowed, seed, target_area, rng)
        return target, {
            "dump_side": SIDE_NAMES[side],
            "dump_components_requested": 1,
            "edge_depth_tiles": max_depth,
        }

    @staticmethod
    def near_apron(
        dig: np.ndarray, target_area: int, side: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del side
        yy, xx = np.indices(dig.shape)
        cy, cx = ndi.center_of_mass(dig)
        outside_distance = ndi.distance_transform_edt(~dig)
        angles = np.arctan2(yy - cy, xx - cx)
        chosen_angle = float(rng.uniform(-math.pi, math.pi))
        angular_delta = np.abs(
            np.arctan2(np.sin(angles - chosen_angle), np.cos(angles - chosen_angle))
        )
        sector_width = float(rng.uniform(math.radians(100), math.radians(220)))
        allowed = (
            (outside_distance >= 3)
            & (outside_distance <= int(rng.integers(8, 12)))
            & (angular_delta <= sector_width / 2)
        )
        seed_score = (
            np.abs(outside_distance - 5.5)
            + 0.25 * angular_delta
            + 0.05 * smooth_noise(rng)
        )
        seed_score[~allowed] = np.inf
        if not np.isfinite(seed_score).any():
            return np.zeros_like(dig), {
                "dump_side": "apron",
                "dump_components_requested": 1,
            }
        seed_y, seed_x = np.unravel_index(np.argmin(seed_score), seed_score.shape)
        seed = np.zeros_like(dig, dtype=bool)
        seed[seed_y, seed_x] = True
        target = grow_region(allowed, seed, target_area, rng)
        return target, {
            "dump_side": "apron",
            "dump_sector_degrees": round(math.degrees(sector_width), 1),
            "dump_components_requested": 1,
        }

    def make(
        self, style: str, dig: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target_area = self._target_area(int(dig.sum()), style, rng)
        side = int(rng.integers(0, 4))
        method = getattr(self, style)
        target, metadata = method(dig, target_area, side, rng)
        metadata["dump_target_cells_requested"] = target_area
        return target, metadata


def curve_between(
    start_yx: np.ndarray,
    end_yx: np.ndarray,
    rng: np.random.Generator,
    bend_scale: float,
) -> np.ndarray:
    delta = end_yx - start_yx
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return np.vstack([start_yx, end_yx])
    normal = np.array([delta[1], -delta[0]]) / norm
    midpoint = (start_yx + end_yx) / 2
    control = midpoint + normal * float(rng.uniform(-bend_scale, bend_scale))
    t = np.linspace(0, 1, 100)
    return (
        ((1 - t) ** 2)[:, None] * start_yx
        + (2 * (1 - t) * t)[:, None] * control
        + (t**2)[:, None] * end_yx
    )


def edge_point(side: int, tangent: int) -> np.ndarray:
    if side == 0:
        return np.array([0.0, float(tangent)])
    if side == 1:
        return np.array([float(tangent), MAP_SIZE - 1.0])
    if side == 2:
        return np.array([MAP_SIZE - 1.0, float(MAP_SIZE - 1 - tangent)])
    return np.array([float(MAP_SIZE - 1 - tangent), 0.0])


def place_objects(
    rng: np.random.Generator,
    occupancy: np.ndarray,
    forbidden: np.ndarray,
    count: int,
    large: bool,
) -> tuple[np.ndarray, int]:
    placed = 0
    for _ in range(count * 40):
        if placed >= count:
            break
        cy = float(rng.uniform(5, MAP_SIZE - 5))
        cx = float(rng.uniform(5, MAP_SIZE - 5))
        if large:
            length, width = float(rng.uniform(3, 7)), float(rng.uniform(2, 5))
        else:
            length, width = float(rng.uniform(2, 4)), float(rng.uniform(2, 4))
        candidate = rotated_rectangle(
            (cy, cx), length, width, float(rng.uniform(0, math.pi))
        )
        if np.any(candidate & forbidden):
            continue
        if np.any(
            ndi.binary_dilation(candidate, structure=binary_disk(3)) & occupancy
        ):
            continue
        occupancy |= candidate
        forbidden |= ndi.binary_dilation(candidate, structure=binary_disk(2))
        placed += 1
    return occupancy, placed


def make_access_road(
    rng: np.random.Generator,
    dig: np.ndarray,
    dump: np.ndarray,
) -> np.ndarray:
    """Route a smooth-ish access corridor through a random continuous cost field."""
    work = dig | dump
    protected = ndi.binary_dilation(work, structure=binary_disk(2))
    distance_from_protected = ndi.distance_transform_edt(~protected)
    distance_from_work = ndi.distance_transform_edt(~work)
    boundary = np.zeros_like(work, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True

    for _ in range(80):
        road_radius = int(rng.integers(3, 5))
        center_allowed = distance_from_protected >= road_radius + 0.5
        starts = np.argwhere(boundary & center_allowed)
        goals = np.argwhere(
            center_allowed
            & (distance_from_work >= road_radius + 2)
            & (distance_from_work <= road_radius + 10)
        )
        if len(starts) == 0 or len(goals) == 0:
            continue
        start = starts[int(rng.integers(0, len(starts)))]
        goal = goals[int(rng.integers(0, len(goals)))]
        if float(np.linalg.norm(goal - start)) < 18:
            continue

        noise = smooth_noise(rng, sigma=float(rng.uniform(5, 9)))
        proximity_penalty = np.clip(
            (road_radius + 7 - distance_from_protected) / 7, 0, 1
        )
        cost = 1.0 + 0.6 * noise + 1.2 * proximity_penalty
        cost[~center_allowed] = np.inf
        try:
            path, _ = route_through_array(
                cost,
                tuple(map(int, start)),
                tuple(map(int, goal)),
                fully_connected=True,
                geometric=True,
            )
        except ValueError:
            continue
        centerline = cells_to_mask(path)
        road = ndi.binary_dilation(
            centerline, structure=binary_disk(road_radius)
        )
        if np.any(road & protected):
            continue
        if int(road.sum()) < 90:
            continue
        return road
    return np.zeros_like(dig, dtype=bool)


def line_across_map(point_yx: np.ndarray, direction_yx: np.ndarray) -> np.ndarray:
    ts = np.linspace(-100, 100, 800)
    points = point_yx[None, :] + ts[:, None] * direction_yx[None, :]
    inside = np.all((points >= 0) & (points < MAP_SIZE), axis=1)
    return rasterize_polyline(points[inside], radius=0)


def make_gapped_wall(
    rng: np.random.Generator,
    dig: np.ndarray,
    dump: np.ndarray,
    road: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    protected = ndi.binary_dilation(dig | dump, structure=binary_disk(3))
    dig_center = np.array(ndi.center_of_mass(dig))
    dump_center = np.array(ndi.center_of_mass(dump))
    for _ in range(100):
        connector = dump_center - dig_center
        if float(np.linalg.norm(connector)) < 8:
            angle = float(rng.uniform(0, math.pi))
            direction = np.array([math.sin(angle), math.cos(angle)])
            point = np.array(
                [float(rng.uniform(12, 52)), float(rng.uniform(12, 52))]
            )
        else:
            direction = np.array([connector[1], -connector[0]])
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            point = (dig_center + dump_center) / 2
            point += direction * float(rng.uniform(-8, 8))
        wall_centerline = line_across_map(point, direction)
        wall = ndi.binary_dilation(
            wall_centerline, structure=binary_disk(int(rng.integers(1, 3)))
        )

        if road is not None and np.any(road & wall_centerline):
            intersections = np.argwhere(road & wall_centerline)
            gap_center = intersections[len(intersections) // 2].astype(float)
        else:
            candidates = np.argwhere(wall_centerline & ~protected)
            if len(candidates) == 0:
                continue
            gap_center = candidates[int(rng.integers(0, len(candidates)))].astype(float)
        gap_width = int(rng.integers(12, 17))
        yy, xx = np.indices(wall.shape)
        gap = np.hypot(yy - gap_center[0], xx - gap_center[1]) <= gap_width / 2
        wall &= ~gap
        if np.any(wall & protected):
            continue
        if int(wall.sum()) < 25:
            continue
        return wall, gap_width

    # Deterministic neutral-line fallback. It still spans the site and has one
    # certified footprint-width gap, but is placed tangentially when a
    # dig-to-dump separating wall would intersect an apron-style target.
    candidates: list[tuple[str, int]] = []
    for coordinate in range(8, MAP_SIZE - 8):
        if not np.any(protected[coordinate, :]):
            candidates.append(("row", coordinate))
        if not np.any(protected[:, coordinate]):
            candidates.append(("column", coordinate))
    rng.shuffle(candidates)
    for axis, coordinate in candidates:
        wall_centerline = np.zeros_like(dig, dtype=bool)
        if axis == "row":
            wall_centerline[coordinate, :] = True
        else:
            wall_centerline[:, coordinate] = True
        wall = ndi.binary_dilation(wall_centerline, structure=binary_disk(1))
        if road is not None and np.any(road & wall_centerline):
            intersections = np.argwhere(road & wall_centerline)
            gap_center = intersections[len(intersections) // 2].astype(float)
        else:
            gap_tangent = int(rng.integers(12, MAP_SIZE - 12))
            gap_center = (
                np.array([coordinate, gap_tangent], dtype=float)
                if axis == "row"
                else np.array([gap_tangent, coordinate], dtype=float)
            )
        gap_width = int(rng.integers(12, 17))
        yy, xx = np.indices(wall.shape)
        wall &= np.hypot(yy - gap_center[0], xx - gap_center[1]) > gap_width / 2
        if not np.any(wall & protected) and int(wall.sum()) >= 25:
            return wall, gap_width
    return np.zeros_like(dig, dtype=bool), 0


class SiteFactory:
    @staticmethod
    def make(
        style: str,
        dig: np.ndarray,
        dump: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        occupancy = np.zeros_like(dig, dtype=bool)
        nondump = np.zeros_like(dig, dtype=bool)
        service_corridor = np.zeros_like(dig, dtype=bool)
        forbidden = ndi.binary_dilation(dig | dump, structure=binary_disk(3))
        metadata: dict[str, Any] = {
            "site_style": style,
            "object_count": 0,
            "wall_gap_tiles": 0,
        }

        if style == "light":
            requested = int(rng.integers(0, 3))
            occupancy, placed = place_objects(
                rng, occupancy, forbidden, requested, large=False
            )
            metadata["object_count"] = placed

        elif style == "scattered_objects":
            requested = int(rng.integers(5, 10))
            occupancy, placed = place_objects(
                rng, occupancy, forbidden, requested, large=True
            )
            metadata["object_count"] = placed

        elif style == "access_road":
            road = make_access_road(rng, dig, dump)
            if int(road.sum()) < 90:
                raise RuntimeError("access_road_generation_failed")
            nondump |= road
            service_corridor |= road

        elif style == "gapped_wall":
            wall, gap_width = make_gapped_wall(rng, dig, dump)
            if int(wall.sum()) < 25 or gap_width < 12:
                raise RuntimeError("gapped_wall_generation_failed")
            occupancy |= wall
            metadata["wall_gap_tiles"] = gap_width

        elif style == "combined":
            road = make_access_road(rng, dig, dump)
            if int(road.sum()) < 90:
                raise RuntimeError("combined_road_generation_failed")
            nondump |= road
            service_corridor |= road
            wall, gap_width = make_gapped_wall(rng, dig, dump, road=road)
            if int(wall.sum()) < 25 or gap_width < 12:
                raise RuntimeError("combined_wall_generation_failed")
            occupancy |= wall
            metadata["wall_gap_tiles"] = gap_width
            forbidden |= ndi.binary_dilation(road | wall, structure=binary_disk(2))
            requested = int(rng.integers(3, 7))
            occupancy, placed = place_objects(
                rng, occupancy, forbidden, requested, large=True
            )
            metadata["object_count"] = placed

        else:
            raise ValueError(f"Unknown site style: {style}")

        occupancy &= ~(dig | dump)
        nondump &= ~(dig | dump | occupancy)
        service_corridor &= nondump
        dumpability = ~(nondump | occupancy)
        return occupancy, dumpability, service_corridor, metadata


def footprint_kernels() -> list[np.ndarray]:
    kernels: list[np.ndarray] = []
    kernel_size = 17
    center = (kernel_size - 1) / 2
    for orientation in range(AGENT_ORIENTATIONS):
        angle = orientation * 2 * math.pi / AGENT_ORIENTATIONS
        local = np.array(
            [
                [-AGENT_WIDTH_TILES / 2, -AGENT_HEIGHT_TILES / 2],
                [AGENT_WIDTH_TILES / 2, -AGENT_HEIGHT_TILES / 2],
                [AGENT_WIDTH_TILES / 2, AGENT_HEIGHT_TILES / 2],
                [-AGENT_WIDTH_TILES / 2, AGENT_HEIGHT_TILES / 2],
            ]
        )
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        xy = local @ rotation.T
        xs = xy[:, 0] + center
        ys = xy[:, 1] + center
        rr, cc = polygon(ys, xs, shape=(kernel_size, kernel_size))
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        kernel[rr, cc] = 1
        kernels.append(kernel)
    return kernels


FOOTPRINT_KERNELS = footprint_kernels()


def valid_center_mask(blocked: np.ndarray) -> np.ndarray:
    valid_any = np.zeros_like(blocked, dtype=bool)
    for kernel in FOOTPRINT_KERNELS:
        collisions = ndi.convolve(
            blocked.astype(np.uint8), kernel, mode="constant", cval=1
        )
        valid_any |= collisions == 0
    return valid_any


def workspace_reachable(base_centers: np.ndarray) -> np.ndarray:
    radius = int(math.ceil(WORKSPACE_R_MAX_TILES))
    yy, xx = np.indices((2 * radius + 1, 2 * radius + 1))
    rr = np.hypot(yy - radius, xx - radius)
    annulus = (
        (rr >= WORKSPACE_R_MIN_TILES - 0.5)
        & (rr <= WORKSPACE_R_MAX_TILES + 0.5)
    ).astype(np.uint8)
    return (
        ndi.convolve(
            base_centers.astype(np.uint8), annulus, mode="constant", cval=0
        )
        > 0
    )


def choose_component(
    center_free: np.ndarray, preference: np.ndarray
) -> tuple[np.ndarray, float, int]:
    labels, n = ndi.label(center_free, structure=np.ones((3, 3), dtype=np.uint8))
    total = int(preference.sum())
    if n == 0 or total == 0:
        return np.zeros_like(center_free, dtype=bool), 0.0, total
    preferred_labels = labels[preference]
    counts = np.bincount(preferred_labels, minlength=n + 1)
    counts[0] = 0
    chosen = int(counts.argmax())
    fraction = float(counts[chosen] / max(1, total))
    return labels == chosen, fraction, total


def static_gate(
    target: np.ndarray,
    occupancy: np.ndarray,
    dumpability: np.ndarray,
) -> StaticGate:
    dig = target < 0
    dump = target > 0
    if not np.any(dig):
        return StaticGate(False, "no_dig", 0, 0, 0, 0, 0, 0, 0, 0)
    if not np.any(dump):
        return StaticGate(False, "no_dump", 0, 0, 0, 0, 0, 0, 0, 0)
    if np.any(dig & dump):
        return StaticGate(False, "dig_dump_overlap", 0, 0, 0, 0, 0, 0, 0, 0)
    if np.any((dig | dump) & occupancy):
        return StaticGate(False, "target_obstacle_overlap", 0, 0, 0, 0, 0, 0, 0, 0)
    if np.any(dump & ~dumpability):
        return StaticGate(False, "dump_not_dumpable", 0, 0, 0, 0, 0, 0, 0, 0)
    if int(dump.sum()) < max(70, int(0.58 * dig.sum())):
        return StaticGate(False, "insufficient_dump_area", 0, 0, 0, 0, 0, 0, 0, 0)

    free_pre = valid_center_mask(occupancy)
    spawn_blocked = occupancy | ~dumpability
    spawn_centers_mask = valid_center_mask(spawn_blocked)
    yy, xx = np.indices(target.shape)
    border_distance = np.minimum.reduce(
        [yy, xx, MAP_SIZE - 1 - yy, MAP_SIZE - 1 - xx]
    )
    spawn_centers_mask &= border_distance >= SPAWN_BORDER_TILES

    pre_component, spawn_fraction, n_spawn = choose_component(
        free_pre, spawn_centers_mask
    )
    if n_spawn < 100:
        return StaticGate(False, "too_few_spawn_centers", n_spawn, spawn_fraction, 0, 0, 0, 0, int(pre_component.sum()), 0)
    if spawn_fraction < 0.985:
        return StaticGate(False, "disconnected_spawn_space", n_spawn, spawn_fraction, 0, 0, 0, 0, int(pre_component.sum()), 0)

    reachable_pre = workspace_reachable(pre_component)
    dig_pre = float((reachable_pre & dig).sum() / max(1, dig.sum()))
    if dig_pre < 0.995:
        return StaticGate(False, "dig_not_reachable_pre", n_spawn, spawn_fraction, dig_pre, 0, 0, 0, int(pre_component.sum()), 0)

    free_post = valid_center_mask(occupancy | dig)
    post_component, _, _ = choose_component(free_post, pre_component & free_post)
    reachable_post = workspace_reachable(post_component)
    dig_post = float((reachable_post & dig).sum() / max(1, dig.sum()))
    dump_post = float((reachable_post & dump).sum() / max(1, dump.sum()))
    reachable_dump_cells = int((reachable_post & dump).sum())
    if dig_post < 0.97:
        return StaticGate(False, "dig_not_reachable_post", n_spawn, spawn_fraction, dig_pre, dig_post, dump_post, reachable_dump_cells, int(pre_component.sum()), int(post_component.sum()))
    if dump_post < 0.58:
        return StaticGate(False, "dump_not_reachable_post", n_spawn, spawn_fraction, dig_pre, dig_post, dump_post, reachable_dump_cells, int(pre_component.sum()), int(post_component.sum()))
    if reachable_dump_cells < max(60, int(0.5 * dig.sum())):
        return StaticGate(False, "reachable_dump_capacity_low", n_spawn, spawn_fraction, dig_pre, dig_post, dump_post, reachable_dump_cells, int(pre_component.sum()), int(post_component.sum()))

    return StaticGate(
        True,
        "accepted",
        n_spawn,
        spawn_fraction,
        dig_pre,
        dig_post,
        dump_post,
        reachable_dump_cells,
        int(pre_component.sum()),
        int(post_component.sum()),
    )


def compute_geodesic_distance(
    target: np.ndarray, occupancy: np.ndarray
) -> np.ndarray:
    dump = target > 0
    distance = np.full(target.shape, np.inf, dtype=np.float32)
    heap: list[tuple[float, int, int]] = []
    for y, x in np.argwhere(dump & ~occupancy):
        distance[y, x] = 0
        heappush(heap, (0.0, int(y), int(x)))
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    )
    while heap:
        current, y, x = heappop(heap)
        if current != float(distance[y, x]):
            continue
        for dy, dx, step in moves:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < MAP_SIZE and 0 <= nx < MAP_SIZE):
                continue
            if occupancy[ny, nx]:
                continue
            proposed = current + step
            if proposed < float(distance[ny, nx]):
                distance[ny, nx] = proposed
                heappush(heap, (proposed, ny, nx))
    finite = np.isfinite(distance)
    if finite.any() and float(distance[finite].max()) > 0:
        distance[finite] /= float(distance[finite].max())
    distance[~finite] = 1.0
    return distance


def target_components(target: np.ndarray) -> int:
    _, n = ndi.label(target > 0, structure=np.ones((3, 3), dtype=np.uint8))
    return int(n)


def make_sample(
    geometry_factory: GeometryFactory,
    geometry: str,
    dump_style: str,
    site_style: str,
    seed: int,
    max_attempts: int,
) -> tuple[Sample | None, Counter[str]]:
    rejections: Counter[str] = Counter()
    for attempt in range(max_attempts):
        attempt_seed = int(np.random.SeedSequence([seed, attempt]).generate_state(1)[0])
        rng = np.random.default_rng(attempt_seed)
        try:
            dig, geometry_meta = geometry_factory.make(geometry, rng)
            dump, dump_meta = DumpFactory().make(dump_style, dig, rng)
            if int(dump.sum()) < 70:
                rejections["dump_generation_shortfall"] += 1
                continue
            occupancy, dumpability, corridor, site_meta = SiteFactory.make(
                site_style, dig, dump, rng
            )
        except RuntimeError as exc:
            rejections[str(exc)] += 1
            continue

        target = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
        target[dig] = -1
        target[dump & ~dig] = 1
        actual_components = target_components(target)
        requested_components = int(dump_meta["dump_components_requested"])
        if actual_components != requested_components:
            rejections["dump_component_contract"] += 1
            continue
        gate = static_gate(target, occupancy, dumpability)
        if not gate.accepted:
            rejections[gate.reason] += 1
            continue

        distance = compute_geodesic_distance(target, occupancy)
        action = np.zeros_like(target, dtype=np.int8)
        metadata: dict[str, Any] = {
            "schema": "site_constraints_v1_review",
            "seed": seed,
            "attempt": attempt,
            "attempt_seed": attempt_seed,
            "geometry": geometry,
            "dump_style": dump_style,
            "site_style": site_style,
            "dig_cells": int(dig.sum()),
            "dump_cells": int(dump.sum()),
            "dump_to_dig_area_ratio": round(float(dump.sum() / dig.sum()), 4),
            "dump_components_actual": actual_components,
            "obstacle_cells": int(occupancy.sum()),
            "nondump_cells": int((~dumpability & ~occupancy).sum()),
            "tile_size_m": TILE_SIZE_M,
            "static_gate_is_action_witness": False,
            **geometry_meta,
            **dump_meta,
            **site_meta,
        }
        return (
            Sample(
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


COLORS = ListedColormap(
    [
        "#f0e3c2",  # neutral sand
        "#e68a2e",  # dig target
        "#51a868",  # terminal dump target
        "#9ea3a8",  # non-dumpable road/exclusion
        "#202124",  # obstacle
    ]
)


def render_code(sample: Sample) -> np.ndarray:
    code = np.zeros(sample.target.shape, dtype=np.uint8)
    code[sample.target < 0] = 1
    code[sample.target > 0] = 2
    code[~sample.dumpability & ~sample.occupancy] = 3
    code[sample.occupancy] = 4
    return code


def render_sample(
    sample: Sample,
    path: Path,
    title: str,
    *,
    compact: bool = False,
) -> None:
    fig_size = (3.0, 3.0) if compact else (5.2, 5.5)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(render_code(sample), cmap=COLORS, vmin=0, vmax=4, interpolation="nearest")
    if not compact:
        gate = sample.gate
        subtitle = (
            f"dig={sample.metadata['dig_cells']} dump={sample.metadata['dump_cells']} "
            f"obs={sample.metadata['obstacle_cells']} nondump={sample.metadata['nondump_cells']}\n"
            f"spawn-comp={gate.spawn_component_fraction:.3f} "
            f"dig-post={gate.dig_workspace_coverage_post:.3f} "
            f"dump-post={gate.dump_workspace_coverage_post:.3f}"
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    else:
        ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_matrix(
    samples: dict[tuple[str, str, str, int], Sample],
    geometry: str,
    variant: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(DUMP_STYLES),
        len(SITE_STYLES),
        figsize=(16, 16),
        constrained_layout=True,
    )
    for row, dump_style in enumerate(DUMP_STYLES):
        for col, site_style in enumerate(SITE_STYLES):
            ax = axes[row, col]
            sample = samples[(geometry, dump_style, site_style, variant)]
            ax.imshow(
                render_code(sample),
                cmap=COLORS,
                vmin=0,
                vmax=4,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(site_style.replace("_", "\n"), fontsize=11)
            if col == 0:
                ax.set_ylabel(dump_style.replace("_", "\n"), fontsize=11)
            ax.text(
                1,
                62,
                f"D{sample.metadata['dig_cells']} / Z{sample.metadata['dump_cells']}\n"
                f"O{sample.metadata['obstacle_cells']} / N{sample.metadata['nondump_cells']}",
                fontsize=6.5,
                color="black",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.68, "edgecolor": "none"},
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"site_constraints_v1 review matrix — {geometry} — variant {variant}",
        fontsize=16,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_variability(
    samples: dict[tuple[str, str, str, int], Sample],
    dump_style: str,
    variants: int,
    path: Path,
) -> None:
    columns = len(SITE_STYLES) * variants
    fig, axes = plt.subplots(
        len(GEOMETRIES),
        columns,
        figsize=(3.0 * columns, 11.5),
        constrained_layout=True,
    )
    if len(GEOMETRIES) == 1:
        axes = np.asarray([axes])
    for row, geometry in enumerate(GEOMETRIES):
        for variant in range(variants):
            for site_idx, site_style in enumerate(SITE_STYLES):
                col = variant * len(SITE_STYLES) + site_idx
                ax = axes[row, col]
                sample = samples[(geometry, dump_style, site_style, variant)]
                ax.imshow(
                    render_code(sample),
                    cmap=COLORS,
                    vmin=0,
                    vmax=4,
                    interpolation="nearest",
                )
                if row == 0:
                    ax.set_title(
                        f"v{variant} {site_style.replace('_', ' ')}", fontsize=8
                    )
                if col == 0:
                    ax.set_ylabel(geometry.replace("_", "\n"), fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(
        f"Procedural variability — {dump_style.replace('_', ' ')}", fontsize=16
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_readme(output: Path, variants: int, accepted: int) -> None:
    readme = f"""# `site_constraints_v1` procedural review set

This folder contains **{accepted} generated maps** crossing:

- geometries: {", ".join(GEOMETRIES)}
- dump layouts: {", ".join(DUMP_STYLES)}
- site constraints: {", ".join(SITE_STYLES)}
- variants per exact combination: {variants}

## Start here

- `matrices/`: one 5x5 dump-layout × site-constraint matrix per geometry and variant
- `variability/`: all geometries and variants grouped by dump-layout algorithm
- `previews/`: one labeled PNG per generated map
- `dataset/`: Terra-shaped `images`, `occupancy`, `dumpability`, `actions`, and
  obstacle-aware `distance` arrays
- `review_metadata/`: generation parameters and static-gate diagnostics
- `manifest.csv`: searchable index
- `generation_summary.json`: accepted count and rejection diagnostics

Colours:

- orange: dig target
- green: terminal dump target
- grey: traversable but non-dumpable road/exclusion
- black: obstacle
- sand: neutral dumpable ground

## Important limitation

All maps passed a 12-orientation, 5x9-tile footprint-aware static connectivity
and workspace-annulus gate. That gate rejects obvious disconnected spawns and
unreachable dig/dump regions. It does **not** prove that Terra's discrete action
system can execute a complete plan after dirt dynamically blocks traversal.
These are review candidates, not a production dataset.
"""
    (output / "README.md").write_text(readme)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_source = (
        script_dir.parent / "full_data" / "foundations_dumpzones_v3"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-foundations",
        type=Path,
        default=default_source,
        help="Terra-format foundation bank used only for OSM dig-mask sources",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-attempts", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    dataset = output / "dataset"
    previews = output / "previews"
    metadata_dir = output / "review_metadata"
    matrices = output / "matrices"
    variability = output / "variability"
    for folder in (
        output,
        previews,
        metadata_dir,
        matrices,
        variability,
        dataset / "images",
        dataset / "occupancy",
        dataset / "dumpability",
        dataset / "actions",
        dataset / "distance",
    ):
        folder.mkdir(parents=True, exist_ok=True)

    geometry_factory = GeometryFactory(args.source_foundations)
    samples: dict[tuple[str, str, str, int], Sample] = {}
    rows: list[dict[str, Any]] = []
    rejection_totals: Counter[str] = Counter()
    sample_idx = 0

    total = (
        len(GEOMETRIES)
        * len(DUMP_STYLES)
        * len(SITE_STYLES)
        * args.variants
    )
    for geometry_idx, geometry in enumerate(GEOMETRIES):
        for dump_idx, dump_style in enumerate(DUMP_STYLES):
            for site_idx, site_style in enumerate(SITE_STYLES):
                for variant in range(args.variants):
                    sample_seed = int(
                        np.random.SeedSequence(
                            [
                                args.seed,
                                geometry_idx,
                                dump_idx,
                                site_idx,
                                variant,
                            ]
                        ).generate_state(1)[0]
                    )
                    sample, rejections = make_sample(
                        geometry_factory,
                        geometry,
                        dump_style,
                        site_style,
                        sample_seed,
                        args.max_attempts,
                    )
                    rejection_totals.update(rejections)
                    if sample is None:
                        raise RuntimeError(
                            "Failed to generate "
                            f"{geometry}/{dump_style}/{site_style}/v{variant}; "
                            f"rejections={dict(rejections)}"
                        )

                    sample_idx += 1
                    key = (geometry, dump_style, site_style, variant)
                    samples[key] = sample
                    stem = f"img_{sample_idx}"
                    np.save(dataset / "images" / f"{stem}.npy", sample.target)
                    np.save(dataset / "occupancy" / f"{stem}.npy", sample.occupancy)
                    np.save(
                        dataset / "dumpability" / f"{stem}.npy", sample.dumpability
                    )
                    np.save(dataset / "actions" / f"{stem}.npy", sample.action)
                    np.save(dataset / "distance" / f"{stem}.npy", sample.distance)

                    record = {
                        "sample_index": sample_idx,
                        "variant": variant,
                        **sample.metadata,
                        **{
                            f"gate_{name}": value
                            for name, value in asdict(sample.gate).items()
                        },
                    }
                    with (metadata_dir / f"{stem}.json").open("w") as handle:
                        json.dump(record, handle, indent=2, sort_keys=True)
                    rows.append(record)
                    render_sample(
                        sample,
                        previews / f"{stem}.png",
                        f"{stem}: {geometry} | {dump_style} | {site_style} | v{variant}",
                    )
                    print(
                        f"[{sample_idx:03d}/{total}] {geometry} / {dump_style} / "
                        f"{site_style} / v{variant} accepted after "
                        f"{sample.metadata['attempt'] + 1} attempt(s)"
                    )

    fieldnames = sorted({key for row in rows for key in row})
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for geometry in GEOMETRIES:
        for variant in range(args.variants):
            render_matrix(
                samples,
                geometry,
                variant,
                matrices / f"{geometry}_variant_{variant}.png",
            )
    for dump_style in DUMP_STYLES:
        render_variability(
            samples,
            dump_style,
            args.variants,
            variability / f"{dump_style}.png",
        )

    summary = {
        "schema": "site_constraints_v1_review",
        "accepted_maps": len(samples),
        "requested_maps": total,
        "base_seed": args.seed,
        "variants": args.variants,
        "geometries": list(GEOMETRIES),
        "dump_styles": list(DUMP_STYLES),
        "site_styles": list(SITE_STYLES),
        "static_gate_is_action_witness": False,
        "rejections_before_acceptance": dict(rejection_totals),
    }
    with (output / "generation_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_readme(output, args.variants, len(samples))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
