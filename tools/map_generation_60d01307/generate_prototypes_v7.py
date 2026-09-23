#!/usr/bin/env python3
"""v3.1 curriculum review bank — RC1-RC5 fixes, plus the §8.2 v3.2 amendments.

Spec: ``terra-digging-benchmark-site/docs/CURRICULUM_TAXONOMY_SPEC.md`` §8 + §8.1
+ **§8.2**.

§8.2 touches exactly three conditions (``fnd-strips-split-wall``,
``trn-net-split-wall``, ``fnd-proc-side1-road``); every other code path below is
byte-frozen so the remaining 21 conditions reproduce identically. See the "v3.2"
comment blocks and ``V32_*`` constants for what moved.
Defect list: ``.artifacts/terra_map_distribution_review_v3/review_bank/
AGENT_REVIEW_FINDINGS.md``.

What changes versus v6 (which produced the v3 bank):

RC1 dump-layout variants get genuinely generous capacity. The inherited
    ``generate_prototypes_v2.CAPACITY_RANGES`` (one_side_near/separated_zones
    1.30-1.65x, haul_away_edge 1.50-2.00x) is no longer consulted at all;
    capacity is a function of the taxonomy *capacity level*, not of the dump
    algorithm. 1.30-1.65x is reserved for the explicit ``tight`` level.
RC2 ``make_gapped_wall`` is replaced by ``make_separating_wall``: the wall is a
    band perpendicular to the dig->dump connector, jittered along its own NORMAL
    (the old code jittered along the wall direction, a no-op), and the wall
    conditions get a *clustered* split-pad sampler so a separating wall exists
    at all (rotationally symmetric pads make it impossible - see §8.1 note).
RC3 site constraints stop hard-vetoing the working area. Objects are placed
    inside the 2-8 tile working annulus with a guaranteed free corridor; roads
    run border-to-border through the near-dump band.
RC4 apron azimuth, sector width (fixed 200 deg), dump side sign and standoff are
    drawn from a *layout RNG* keyed by (layout group, map index), so capacity and
    layout siblings are matched. Standoff is randomised 2-6 tiles instead of the
    constant 3.0 of every previous bank.
RC5 the dig bank rejects near-duplicates (centred IoU >= 0.6 inside a geometry
    level) and re-used source indices, and places targets on a spread schedule
    instead of always near the map centre.

Plus the §8.1 misc gates: topology honesty (tee = T only, net >= 50% non-double_T,
branch >= 35% of spine), T0 post-dig workspace coverage >= 0.99, remote min
distance >= 15 tiles, split pads >= 3 tiles from the border with a >= 150 deg
angular gap, and a trench heading schedule that covers 0-165 deg in 15 deg steps.

Dumpability semantics are unchanged and deliberately so (§8.1): uniformly-True
dumpability on a clean map is correct - only physical obstacles and roads
restrict it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import route_through_array
from skimage.morphology import convex_hull_image

import generate_prototypes_v6 as v6

v5 = v6.v5
v4 = v6.v4
v3 = v6.v3
v2 = v6.v2
base = v6.base
tax = v6.tax

SCHEMA = "terra_curriculum_v31_review_bank"
SEED_BASE = 20260728
MAP_ID_PREFIX = "curriculum-v31"
MAPS_PER_CONDITION = 16
PREVIEW_MAPS_PER_CONDITION = 4
SHARED_DIG_ATTEMPTS = 120
REROLL_DUMP_ATTEMPTS = 20
MAX_ATTEMPTS = 320

CONDITIONS = v6.CONDITIONS
OBJECT_BANDS = v6.OBJECT_BANDS
SITE_CLASS_TOKENS = v6.SITE_CLASS_TOKENS
CAPACITY_TOKENS = v6.CAPACITY_TOKENS
GEOMETRY_LEVEL_SOURCE = v6.GEOMETRY_LEVEL_SOURCE
GEOMETRY_LEVEL_INDEX = v6.GEOMETRY_LEVEL_INDEX
GEOMETRY_HARDNESS = v6.GEOMETRY_HARDNESS
ARRAY_FOLDERS = v6.ARRAY_FOLDERS

# --------------------------------------------------------------------------
# RC1 — capacity is a function of the capacity LEVEL, never of the algorithm.

APRON_CAPACITY_BANDS = v6.APRON_CAPACITY_BANDS      # c7x / c3x / c1p6
RING_CAPACITY_BANDS = v6.RING_CAPACITY_BANDS        # c11x
RING_MIN_CAPACITY = v6.RING_MIN_CAPACITY
TIGHT_CAPACITY_BAND = (1.30, 1.65)
# "generous" per dump algorithm. Every one of these is >= 2.0x (§8.1 RC1).
GENEROUS_CAPACITY_BANDS = {
    "easy_surround": (2.50, 3.25),        # trench, both flanks
    "near_apron_large": (2.00, 2.60),     # trench, one flank
    "one_side_near": (2.00, 2.60),        # foundation one side / trench flank
    "separated_zones": (2.00, 2.60),
    "haul_away_edge": (2.00, 2.60),
}
# Wall conditions must fit dig + keep-out + wall band + keep-out + pads into a
# 64x64 map, so their generous band is the bottom of the generous range. Still
# >= 2.0x, i.e. still a real RC1 fix; documented in GENERATION_NOTES.
WALL_SPLIT_CAPACITY_BAND = (2.00, 2.30)
# §8.1 RC1 floor asserted by the validator on every generous non-ring layout.
GENEROUS_REACHABLE_FLOOR = 2.00

# --------------------------------------------------------------------------
# RC4 — layout parameters shared inside a group.

LAYOUT_GROUP = {
    "fnd-slab-apron-c7x": "slab-apron",
    "fnd-slab-apron-c3x": "slab-apron",
    "fnd-slab-apron-c1p6": "slab-apron",
    "trn-straight-side1": "trn-one-side",
    "trn-straight-side1-tight": "trn-one-side",
}
APRON_SECTOR_DEGREES = 200.0
STANDOFF_CHOICES = (2, 3, 4, 5, 6)
WALL_CONDITIONS = {"fnd-strips-split-wall", "trn-net-split-wall"}
# 3 (dig keep-out) + 3 (a 3-tile wall band) + 3 (dump keep-out) + slack for the
# connector/azimuth mismatch. The pad sampler pushes the whole dump past this
# along the connector so a full-width band fits between the two keep-outs.
WALL_CLEARANCE_TILES = 9.5
# v6 capped the slab bbox at 30 tiles; the anisotropic rescale needs a little
# more room or every large-area slab is rejected and the size range collapses.
SLAB_MAX_DIM_V7 = 34

# Near-distance contracts. Inherited limits assumed a constant 3.0 standoff, so
# they move with the drawn standoff; the clustered wall split needs its own.
FOUNDATION_MEDIAN_LIMITS = {
    "capacity_apron": 13.0,
    "one_side_near": 15.0,
    "separated_zones": 17.0,
}
TRENCH_MEDIAN_LIMITS = {
    "easy_surround": 6.0,
    "near_apron_large": 7.0,
    "one_side_near": 7.0,
    "separated_zones": 17.0,
}
WALL_SPLIT_MEDIAN_LIMIT = 26.0
REMOTE_MIN_DISTANCE_TILES = 15.0

# --------------------------------------------------------------------------
# RC3 — site intrusion gates.

OBJECT_ANNULUS_BLOCK_MIN = {"obj": 0.08, "obj1": 0.03}
OBJECT_NEAR_ANNULUS_FREE_MIN = 0.60      # <=4 tile annulus stays >=60% free
ROAD_ANNULUS_STERILIZE_MIN = 0.10
ROAD_ANNULUS_FREE_MIN = 0.50
OBJECT_FOOTPRINT_PROFILE = "shared_v31"  # obj and obj1 draw from the same sizes

# --------------------------------------------------------------------------
# RC2 — wall gates.

WALL_BITE_MIN = 0.40
WALL_MIN_COMPONENT_CELLS = 25
WALL_MIN_COMPONENT_EXTENT = 8
WALL_MIN_COMPONENTS = 2
SPLIT_PAD_BORDER_MARGIN = 3

# --------------------------------------------------------------------------
# §8.2 v3.2 — the three amended conditions and their new gates.
#
# The rest of the bank is frozen; every branch keyed on these sets is a no-op
# for the other 21 conditions, which therefore reproduce byte-identically.

V32_REGENERATED = frozenset(
    {
        "fnd-strips-split-wall",
        "trn-net-split-wall",
        "fnd-proc-side1-road",
        "trn-tee-side1-road",
    }
)
# Zoned layouts where the road is built BEFORE the dump ("as on rings", §8.2).
# trn-tee-side1-road added post-§8.2: same v3 veto, was outside the named scope
# but fails the amended zoned gate on 11/16 maps.
V32_ROAD_FIRST_CONDITIONS = frozenset({"fnd-proc-side1-road", "trn-tee-side1-road"})

# Wall efficacy is a geodesic statement, not a sight-line one. `WALL_BITE_MIN`
# survives as a *reported* metric only.
WALL_DETOUR_MIN = 1.15                # per map
WALL_DETOUR_CONDITION_MEDIAN_MIN = 1.25
WALL_GAP_OFFAXIS_MIN = 8.0            # gap midpoint, perpendicular to the connector
WALL_MIN_THICKNESS_TILES = 3          # 1-tile bands are 4-connectivity-permeable
# The wall is an offset fence around the dig's convex hull: a band at constant
# distance covering the pads' azimuths, with one gate cut off-axis. A straight
# full-width band cannot coexist with >= 80 deg pad separation on 64x64 (see
# GENERATION_NOTES §7).
FENCE_INNER_CHOICES = (4.0, 5.0, 6.0, 7.0)
FENCE_THICKNESS_CHOICES = (3.0, 3.0, 5.0)
FENCE_PAD_CLEARANCE = 3.0             # dilate(dump, disk(3)) keep-out
FENCE_ARC_MARGIN_DEG = (18.0, 34.0)   # arc overhang past the outer pad edge
FENCE_PAD_HALF_WIDTH_DEG = 28.0
FENCE_GAP_MIN_OFFAXIS_DEG = 38.0

# True split on the wall conditions (§8.2): within the realized plain-split
# ranges (edge gap 4.1-31.3 tiles, centroid span 88-135 deg).
SPLIT_PAD_EDGE_GAP_MIN = 6.0
SPLIT_PAD_ANGULAR_SPAN_MIN = 80.0
SPLIT_PAD_SEPARATION_DISK = 6         # forced clearance between pads

# Road-gate contract, §8.2. v3.1 measured the DIG annulus and called it the
# "near-dump band"; the two are only the same thing on a ring layout.
ROAD_RADIUS_RANGE_RING = (3, 5)       # rng.integers(low, high)
ROAD_RADIUS_RANGE_ZONED = (2, 5)
ROAD_DIG_ANNULUS_STERILIZE_RANGE = (0.10, 0.50)   # ring layouts
ROAD_SIGHTLINE_CROSS_MIN = 0.15                    # zoned layouts, branch A
ROAD_DUMP_ANNULUS_STERILIZE_MIN = 0.10             # zoned layouts, branch B

# --------------------------------------------------------------------------
# RC5 / misc gates.

DIG_IOU_MAX = 0.60
T0_DIG_COVERAGE_MIN = 0.99
SPLIT_MAX_ANGULAR_GAP_MIN = 150.0
TRENCH_HEADINGS_DEG = tuple(float(a) for a in range(0, 180, 15))
TRENCH_BRANCH_SPINE_MIN = 0.35
NET_TOPOLOGY_CYCLE = ("H", "double_T", "T_plus_X", "H", "T_plus_X", "double_T")
# Placement schedule: target centroid offset from the map centre, in tiles.
PLACEMENT_RADII = (3.0, 8.0, 12.0, 5.5, 15.0, 10.0, 1.0, 13.5)
FOUNDATION_PLACEMENT_MARGIN = 8
TRENCH_PLACEMENT_MARGIN = 6

MAP_SIZE = base.MAP_SIZE
MAP_CENTRE = (MAP_SIZE - 1) / 2.0

T0_CONDITIONS = frozenset(
    condition.id for condition in CONDITIONS if condition.tier == 0
)
# Geometry levels that feed a T0 condition get the strict coverage prescreen.
T0_GEOMETRY_LEVELS = frozenset(
    condition.geometry_level for condition in CONDITIONS if condition.tier == 0
)


def stable_key(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")))


def sha256_mask(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    ).hexdigest()


def rng_from(*parts: int) -> np.random.Generator:
    seed = int(np.random.SeedSequence(list(parts)).generate_state(1)[0])
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# geometry helpers


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def border_margin(mask: np.ndarray) -> int:
    y0, y1, x0, x1 = _bbox(mask)
    return int(min(y0, x0, MAP_SIZE - 1 - y1, MAP_SIZE - 1 - x1))


def centroid(mask: np.ndarray) -> np.ndarray:
    return np.asarray(ndi.center_of_mass(mask), dtype=float)


def centroid_offset(mask: np.ndarray) -> float:
    cy, cx = centroid(mask)
    return float(math.hypot(cy - MAP_CENTRE, cx - MAP_CENTRE))


def place_at(
    mask: np.ndarray, radius: float, angle: float, margin: int
) -> np.ndarray | None:
    """Translate ``mask`` toward a target centroid offset, clamped to the margin.

    RC5/placement: v6 drew every footprint centre inside a 25-40 box, which put
    all 360 targets within ~9 tiles of the map centre. Here the offset is a
    schedule; the clamp only shrinks it when the footprint would leave the map.
    """
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape
    if height + 2 * margin > MAP_SIZE or width + 2 * margin > MAP_SIZE:
        return None
    local_y, local_x = ndi.center_of_mass(crop)
    y0 = int(
        np.clip(
            round(MAP_CENTRE + radius * math.sin(angle) - local_y),
            margin,
            MAP_SIZE - margin - height,
        )
    )
    x0 = int(
        np.clip(
            round(MAP_CENTRE + radius * math.cos(angle) - local_x),
            margin,
            MAP_SIZE - margin - width,
        )
    )
    out = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    out[y0 : y0 + height, x0 : x0 + width] = crop
    return out


def centred_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU after aligning both masks on their centroids (translation-free)."""
    ay, ax = centroid(a)
    by, bx = centroid(b)
    shifted = np.roll(np.roll(b, int(round(ay - by)), axis=0), int(round(ax - bx)), axis=1)
    union = int((a | shifted).sum())
    return float((a & shifted).sum() / max(1, union))


def dig_only_coverage(dig: np.ndarray) -> float:
    """``static_gate``'s post-dig workspace coverage on an otherwise empty map."""
    empty = np.zeros_like(dig)
    free_pre = base.valid_center_mask(empty)
    pre_component, _, _ = base.choose_component(free_pre, free_pre)
    free_post = base.valid_center_mask(dig)
    post_component, _, _ = base.choose_component(free_post, pre_component & free_post)
    reachable = base.workspace_reachable(post_component)
    return float((reachable & dig).sum() / max(1, dig.sum()))


def annulus(dig: np.ndarray, radius: float) -> np.ndarray:
    distance = ndi.distance_transform_edt(~dig)
    return (distance > 0) & (distance <= radius)


def max_angular_gap_degrees(dig: np.ndarray, dump: np.ndarray) -> float:
    """Largest empty angular sector, seen from the dig centroid."""
    cy, cx = centroid(dig)
    ys, xs = np.where(dump)
    if len(ys) == 0:
        return 360.0
    angles = np.sort(np.degrees(np.arctan2(ys - cy, xs - cx)) % 360.0)
    gaps = np.diff(angles)
    wrap = 360.0 - float(angles[-1]) + float(angles[0])
    return float(max(gaps.max() if len(gaps) else 0.0, wrap))


SIGHT_LINE_SAMPLES = 48
SIGHT_LINE_STEPS = 256


def sight_line_bite(
    dig: np.ndarray,
    dump: np.ndarray,
    wall: np.ndarray,
    samples: int = SIGHT_LINE_SAMPLES,
) -> float:
    """Fraction of straight dig<->dump sight-lines that cross a wall cell.

    Fully deterministic (fixed stride over the sorted cell lists, fixed step
    count along each line) so the generator and the validator measure the same
    number. 128 steps over a <=90 tile diagonal is a <=0.71 tile step, so a
    1-tile-thick wall can never be stepped over.
    """
    if not wall.any():
        return 0.0
    dig_cells = np.argwhere(dig)
    dump_cells = np.argwhere(dump)
    if len(dig_cells) == 0 or len(dump_cells) == 0:
        return 0.0

    def pick(cells: np.ndarray) -> np.ndarray:
        if len(cells) <= samples:
            return cells
        return cells[np.linspace(0, len(cells) - 1, samples).round().astype(int)]

    a = pick(dig_cells).astype(np.float32)
    b = pick(dump_cells).astype(np.float32)
    t = np.linspace(0.0, 1.0, SIGHT_LINE_STEPS, dtype=np.float32)
    ys = a[:, None, 0, None] * (1 - t) + b[None, :, 0, None] * t
    xs = a[:, None, 1, None] * (1 - t) + b[None, :, 1, None] * t
    hit = wall[np.rint(ys).astype(np.int16), np.rint(xs).astype(np.int16)]
    return float(hit.any(axis=2).mean())


DIJKSTRA_MOVES = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def geodesic_from(sources: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """8-connected Dijkstra in tiles from every source cell. Not normalised."""
    distance = np.full(sources.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for y, x in np.argwhere(sources & ~blocked):
        distance[y, x] = 0.0
        heappush(heap, (0.0, int(y), int(x)))
    while heap:
        current, y, x = heappop(heap)
        if current != distance[y, x]:
            continue
        for dy, dx, step in DIJKSTRA_MOVES:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < MAP_SIZE and 0 <= nx < MAP_SIZE) or blocked[ny, nx]:
                continue
            proposed = current + step
            if proposed < distance[ny, nx]:
                distance[ny, nx] = proposed
                heappush(heap, (proposed, ny, nx))
    return distance


def haul_detour(
    dig: np.ndarray, dump: np.ndarray, wall: np.ndarray
) -> tuple[float, float, float, float]:
    """§8.2: mean dig->dump haul length with the wall, divided by without it.

    Multi-source Dijkstra out of the whole dig mask, averaged over the dump
    cells. This is the metric the wall is gated on; the sight-line bite is a
    report-only proxy that v3.1 gamed.
    """
    free = geodesic_from(dig, np.zeros_like(dig, dtype=bool))
    walled = geodesic_from(dig, wall)
    cells = dump & ~wall
    if not cells.any():
        return float("nan"), 0.0, 0.0, 1.0
    open_lengths = free[cells]
    walled_lengths = walled[cells]
    reachable = np.isfinite(walled_lengths)
    unreachable = float(1.0 - reachable.mean())
    if not reachable.any():
        return float("inf"), float(open_lengths.mean()), float("inf"), unreachable
    mean_free = float(open_lengths[reachable].mean())
    mean_walled = float(walled_lengths[reachable].mean())
    return mean_walled / max(mean_free, 1e-9), mean_free, mean_walled, unreachable


def wall_min_thickness(wall: np.ndarray) -> int:
    """Thinnest place in the wall, as ``2 * erosion_radius + 1`` tiles."""
    if not wall.any():
        return 0
    for radius in range(1, 7):
        if not ndi.binary_erosion(wall, structure=base.binary_disk(radius)).any():
            return 2 * radius - 1
    return 13


def gap_offaxis_tiles(
    dig: np.ndarray, dump: np.ndarray, gap_centre: tuple[float, float]
) -> float:
    """Perpendicular distance of the gap midpoint from the dig->dump connector."""
    dig_c = centroid(dig)
    dump_c = centroid(dump)
    connector = dump_c - dig_c
    unit = connector / max(float(np.linalg.norm(connector)), 1e-9)
    tangent = np.array([unit[1], -unit[0]])
    return float(
        abs((gap_centre[0] - dig_c[0]) * tangent[0]
            + (gap_centre[1] - dig_c[1]) * tangent[1])
    )


def pad_separation(dig: np.ndarray, dump: np.ndarray) -> tuple[int, float, float]:
    """(#pads, smallest inter-pad edge gap, largest pad-centroid angular span)."""
    labels, n = ndi.label(dump, structure=np.ones((3, 3), dtype=np.uint8))
    if n < 2:
        return int(n), 0.0, 0.0
    cy, cx = centroid(dig)
    gaps: list[float] = []
    angles: list[float] = []
    for index in range(1, n + 1):
        pad = labels == index
        py, px = centroid(pad)
        angles.append(math.degrees(math.atan2(py - cy, px - cx)) % 360.0)
        distance = ndi.distance_transform_edt(~pad)
        for other in range(index + 1, n + 1):
            gaps.append(float(distance[labels == other].min()))
    span = 0.0
    for a in angles:
        for b in angles:
            delta = abs(a - b) % 360.0
            span = max(span, min(delta, 360.0 - delta))
    return int(n), min(gaps), span


def dump_annulus(dig: np.ndarray, dump: np.ndarray, radius: float) -> np.ndarray:
    """The <= ``radius`` tile band around the DUMP, excluding the dig itself."""
    return annulus(dump, radius) & ~dig


def wall_component_stats(wall: np.ndarray) -> tuple[int, int, int]:
    labels, n = ndi.label(wall, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return 0, 0, 0
    min_cells = 10**9
    min_extent = 10**9
    for index in range(1, n + 1):
        component = labels == index
        cells = int(component.sum())
        y0, y1, x0, x1 = _bbox(component)
        extent = int(max(y1 - y0, x1 - x0) + 1)
        min_cells = min(min_cells, cells)
        min_extent = min(min_extent, extent)
    return int(n), min_cells, min_extent


def axial_coverage(dig: np.ndarray, dump: np.ndarray, heading_deg: float) -> float:
    """Fraction of the trench's axial extent that has dump within 1 tile bins."""
    heading = math.radians(heading_deg)
    direction = np.array([math.sin(heading), math.cos(heading)])
    dig_cells = np.argwhere(dig).astype(float)
    dump_cells = np.argwhere(dump).astype(float)
    if len(dump_cells) == 0:
        return 0.0
    origin = dig_cells.mean(axis=0)
    dig_projection = (dig_cells - origin) @ direction
    dump_projection = (dump_cells - origin) @ direction
    low, high = float(dig_projection.min()), float(dig_projection.max())
    if high - low < 1.0:
        return 1.0
    bins = np.arange(low, high + 1.0, 1.0)
    occupied = np.histogram(dump_projection, bins=bins)[0] > 0
    return float(occupied.mean())


# --------------------------------------------------------------------------
# geometry factory


class GeometryFactoryV7(v6.GeometryFactoryV6):
    """v6 sources, but every target is placed on the spread schedule."""

    def slab(self, rng, radius, angle):
        for _ in range(80):
            index = int(rng.integers(0, len(self.slab_sources)))
            crop = np.rot90(self.slab_sources[index], int(rng.integers(0, 4)))
            if rng.random() < 0.5:
                crop = np.fliplr(crop)
            # RC5: the raw OSM pool saturates at ~11 mutually dissimilar slabs
            # (centred IoU < 0.6), so aspect is a variety knob here as well as
            # on slab-lg. Area stays inside SLAB_DIG_CELLS.
            crop = self._rescale(crop, v6.SLAB_DIG_CELLS, SLAB_MAX_DIM_V7, rng, (0.72, 1.40))
            if crop is None:
                continue
            placed = place_at(crop, radius, angle, FOUNDATION_PLACEMENT_MARGIN)
            if placed is not None:
                return placed, {
                    "foundation_source_index": index,
                    "foundation_size_class": "slab",
                }
        return None, {}

    def slab_lg(self, rng, radius, angle):
        for _ in range(80):
            index = int(rng.integers(0, len(self.scale_sources)))
            crop = np.rot90(self.scale_sources[index], int(rng.integers(0, 4)))
            if rng.random() < 0.5:
                crop = np.fliplr(crop)
            # §8.1 RC5: slab-lg may rescale anisotropically for shape variety.
            scaled = self._rescale(
                crop, v6.LARGE_SLAB_DIG_CELLS, v6.LARGE_SLAB_MAX_DIM, rng, (0.46, 2.15)
            )
            if scaled is None:
                continue
            placed = place_at(scaled, radius, angle, FOUNDATION_PLACEMENT_MARGIN)
            if placed is not None:
                return placed, {
                    "foundation_source_index": index,
                    "foundation_size_class": "slab_large",
                    "foundation_source_cells": int(crop.sum()),
                }
        return None, {}

    @staticmethod
    def _rescale(mask, cell_band, max_dim, rng, aspect_band):
        """Anisotropic rescale of a source footprint into a cell band."""
        ys, xs = np.where(mask)
        crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(float)
        low, high = cell_band
        goal = float(rng.uniform(low + 2, high - 2))
        aspect = float(rng.uniform(*aspect_band))
        ideal = math.sqrt(goal / float(crop.sum()))
        best: tuple[float, np.ndarray] | None = None
        for factor in ideal * np.linspace(0.86, 1.18, 41):
            zoom = (float(factor) * aspect, float(factor) / aspect)
            scaled = ndi.zoom(crop, zoom, order=1) > 0.5
            if not scaled.any():
                continue
            scaled = base.largest_component(scaled)
            cells = int(scaled.sum())
            if not (low <= cells <= high and max(scaled.shape) <= max_dim):
                continue
            error = abs(cells - goal)
            # Closest to the requested area, not merely inside the band: a wide
            # band plus a first-hit loop collapses every slab to the small end.
            if best is None or error < best[0]:
                best = (error, scaled)
        return None if best is None else best[1]

    def proc(self, rng, radius, angle):
        """Wider parameter space than v2's: the old ranges saturated at ~6
        mutually dissimilar footprints, well short of the 16 the bank needs."""
        for _ in range(160):
            heading = float(rng.choice(np.deg2rad(np.arange(0, 180, 15))))
            length = float(rng.uniform(11, 30))
            width = float(rng.uniform(5, 18))
            if not 90 <= length * width <= 400:
                continue
            centre = (MAP_CENTRE, MAP_CENTRE)
            dig = base.rotated_rectangle(centre, length, width, heading)
            wings = int(rng.integers(1, 5))
            for _wing in range(wings):
                along = float(rng.uniform(-0.55, 0.55)) * length
                across = float(rng.choice([-1.0, 1.0])) * float(
                    rng.uniform(0.20, 0.90)
                ) * width
                dy = along * math.sin(heading) + across * math.cos(heading)
                dx = along * math.cos(heading) - across * math.sin(heading)
                dig |= base.rotated_rectangle(
                    (centre[0] + dy, centre[1] + dx),
                    float(rng.uniform(5, 17)),
                    float(rng.uniform(4, 13)),
                    heading + float(rng.choice([0.0, math.pi / 2])),
                )
            dig = base.largest_component(
                ndi.binary_closing(dig, structure=base.binary_disk(1))
            )
            if not dig.any() or not 110 <= int(dig.sum()) <= 340:
                continue
            placed = place_at(dig, radius, angle, FOUNDATION_PLACEMENT_MARGIN)
            if placed is None:
                continue
            return placed, {
                "foundation_angle_deg": round(math.degrees(heading), 1),
                "foundation_wings": wings,
            }
        return None, {}

    def strips(self, rng, radius, angle):
        for _ in range(40):
            try:
                dig, meta = v2.GeometryFactoryV2.foundation_structural(rng)
            except RuntimeError:
                continue
            placed = place_at(dig, radius, angle, FOUNDATION_PLACEMENT_MARGIN)
            if placed is None:
                continue
            _, components = ndi.label(placed, structure=np.ones((3, 3), np.uint8))
            if components < 2:
                continue
            return placed, meta
        return None, {}

    def straight(self, rng, radius, angle, heading_deg, _topology=None):
        heading = math.radians(heading_deg)
        direction = np.array([math.sin(heading), math.cos(heading)])
        for _ in range(120):
            # Wider than v3's 24-38: a single straight axis has very little
            # shape freedom, and the RC5 IoU < 0.6 rule needs more than the
            # ~16 distinct shapes the old range could produce.
            length = float(rng.uniform(20, 42))
            points = np.vstack([-direction * length / 2, direction * length / 2])
            width_radius = int(rng.choice([1, 2], p=[0.55, 0.45]))
            dig = base.rasterize_polyline(points + MAP_CENTRE, radius=width_radius)
            if not 55 <= int(dig.sum()) <= 230:
                continue
            placed = place_at(dig, radius, angle, TRENCH_PLACEMENT_MARGIN)
            if placed is None:
                continue
            return placed, {
                "trench_axes_count": 1,
                "intersection_junctions": 0,
                "intersection_branches": 2,
                "trench_topology": "straight",
                "trench_width_radius_tiles": width_radius,
                "trench_global_angle_deg": heading_deg,
                "trench_spine_length_tiles": round(length, 2),
                "trench_branch_spine_ratio": 1.0,
                "axes_ABC": [
                    v3.line_coefficients(points[0] + MAP_CENTRE, points[1] + MAP_CENTRE)
                ],
            }
        return None, {}

    def tee(self, rng, radius, angle, heading_deg, _topology=None):
        """§8.1 topology honesty: `tee` is exactly one T junction, never an X."""
        heading = math.radians(heading_deg)
        direction = np.array([math.sin(heading), math.cos(heading)])
        normal = np.array([direction[1], -direction[0]])
        for _ in range(160):
            spine = float(rng.uniform(26, 36))
            branch = float(rng.uniform(TRENCH_BRANCH_SPINE_MIN + 0.02, 0.50)) * spine
            junction_along = float(rng.uniform(-0.14, 0.14)) * spine
            side = float(rng.choice([-1.0, 1.0]))
            main = np.vstack([-direction * spine / 2, direction * spine / 2])
            junction = direction * junction_along
            branch_points = np.vstack([junction, junction + side * normal * branch])
            width_radius = int(rng.choice([1, 2], p=[0.6, 0.4]))
            dig = base.rasterize_polyline(main + MAP_CENTRE, radius=width_radius)
            dig |= base.rasterize_polyline(branch_points + MAP_CENTRE, radius=width_radius)
            dig = base.largest_component(dig)
            if not 90 <= int(dig.sum()) <= 300:
                continue
            placed = place_at(dig, radius, angle, TRENCH_PLACEMENT_MARGIN)
            if placed is None:
                continue
            return placed, {
                "trench_axes_count": 2,
                "intersection_junctions": 1,
                "intersection_branches": 3,
                "trench_topology": "T",
                "trench_relative_angle_deg": 90.0,
                "trench_width_radius_tiles": width_radius,
                "trench_global_angle_deg": heading_deg,
                "trench_spine_length_tiles": round(spine, 2),
                "trench_branch_spine_ratio": round(branch / spine, 4),
                "axes_ABC": [
                    v3.line_coefficients(main[0] + MAP_CENTRE, main[1] + MAP_CENTRE),
                    v3.line_coefficients(
                        branch_points[0] + MAP_CENTRE, branch_points[-1] + MAP_CENTRE
                    ),
                ],
            }
        return None, {}

    def net(self, rng, radius, angle, heading_deg, topology):
        """>=2 junctions; the topology is scheduled so >=50% are non-double_T."""
        heading = math.radians(heading_deg)
        direction = np.array([math.sin(heading), math.cos(heading)])
        normal = np.array([direction[1], -direction[0]])
        for _ in range(200):
            # Smaller than v3's 27-39: a 3-axis net plus a keep-out plus a wall
            # band plus generous pads has to fit in 64x64 for the wall condition.
            spine = float(rng.uniform(24, 32))
            main = np.vstack([-direction * spine / 2, direction * spine / 2])
            positions = np.array([-0.26, 0.26]) + rng.uniform(-0.03, 0.03, size=2)
            branch_points_list = []
            axes = [v3.line_coefficients(main[0] + MAP_CENTRE, main[1] + MAP_CENTRE)]
            ratios = []
            for branch_index, along in enumerate(positions):
                junction = direction * float(along * spine)
                length_a = float(rng.uniform(TRENCH_BRANCH_SPINE_MIN + 0.02, 0.46)) * spine
                double_sided = topology == "H" or (
                    topology == "T_plus_X" and branch_index == 1
                )
                if double_sided:
                    length_b = (
                        float(rng.uniform(TRENCH_BRANCH_SPINE_MIN + 0.02, 0.46)) * spine
                    )
                    points = np.vstack(
                        [
                            junction - normal * length_a,
                            junction,
                            junction + normal * length_b,
                        ]
                    )
                    ratios.extend([length_a / spine, length_b / spine])
                else:
                    side = (
                        (-1.0 if branch_index == 0 else 1.0)
                        if topology == "double_T"
                        else float(rng.choice([-1.0, 1.0]))
                    )
                    points = np.vstack([junction, junction + side * normal * length_a])
                    ratios.append(length_a / spine)
                branch_points_list.append(points)
                axes.append(
                    v3.line_coefficients(points[0] + MAP_CENTRE, points[-1] + MAP_CENTRE)
                )
            width_radius = int(rng.choice([1, 2], p=[0.6, 0.4]))
            dig = base.rasterize_polyline(main + MAP_CENTRE, radius=width_radius)
            for points in branch_points_list:
                dig |= base.rasterize_polyline(points + MAP_CENTRE, radius=width_radius)
            dig = base.largest_component(dig)
            if not 120 <= int(dig.sum()) <= 290:
                continue
            placed = place_at(dig, radius, angle, TRENCH_PLACEMENT_MARGIN)
            if placed is None:
                continue
            double_sided_count = sum(len(p) == 3 for p in branch_points_list)
            return placed, {
                "trench_axes_count": 3,
                "intersection_junctions": 2,
                "intersection_double_sided": double_sided_count,
                "intersection_branches": 4 + double_sided_count,
                "trench_topology": topology,
                "trench_relative_angle_deg": 90.0,
                "trench_width_radius_tiles": width_radius,
                "trench_global_angle_deg": heading_deg,
                "trench_spine_length_tiles": round(spine, 2),
                "trench_branch_spine_ratio": round(float(min(ratios)), 4),
                "axes_ABC": axes,
            }
        return None, {}


GEOMETRY_BUILDER = {
    "slab": "slab",
    "slab-lg": "slab_lg",
    "proc": "proc",
    "strips": "strips",
    "straight": "straight",
    "tee": "tee",
    "net": "net",
}
TRENCH_LEVELS = frozenset({"straight", "tee", "net"})


# --------------------------------------------------------------------------
# RC5 — dig bank with dissimilarity + source-uniqueness rejection


class DigBankV7:
    """Excavation targets keyed by (geometry level, map index).

    The salt-0 bank is prebuilt in map-index order so the dedup decision does not
    depend on which condition asks first, and every re-roll (salt > 0) is also
    held to IoU < 0.6 against the whole salt-0 bank. Dissimilarity *inside* a
    condition is enforced one level up, in ``generate_condition``.
    """

    def __init__(self, factory: GeometryFactoryV7, n_maps: int) -> None:
        self.factory = factory
        self.n_maps = n_maps
        self.bank: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = {}
        self.rerolls: dict[
            tuple[str, int, int], tuple[np.ndarray, dict[str, Any]] | None
        ] = {}

    def _headings(self, level: str) -> list[float]:
        """0-165 deg in 15 deg steps, permuted per level; no 0/90 over-weighting."""
        rng = rng_from(SEED_BASE, 4242, GEOMETRY_LEVEL_INDEX[level])
        order = [TRENCH_HEADINGS_DEG[i] for i in rng.permutation(len(TRENCH_HEADINGS_DEG))]
        return [order[k % len(order)] for k in range(self.n_maps)]

    def _sample(
        self, level: str, map_index: int, salt: int, attempt: int
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        seed = int(
            np.random.SeedSequence(
                [SEED_BASE, 7777, GEOMETRY_LEVEL_INDEX[level], map_index, salt, attempt]
            ).generate_state(1)[0]
        )
        rng = np.random.default_rng(seed)
        placement_rng = rng_from(
            SEED_BASE, 3131, GEOMETRY_LEVEL_INDEX[level], map_index, salt
        )
        radius = PLACEMENT_RADII[
            (map_index + attempt // 12) % len(PLACEMENT_RADII)
        ] + float(placement_rng.uniform(-1.0, 1.0))
        angle = float(placement_rng.uniform(-math.pi, math.pi))
        builder = getattr(self.factory, GEOMETRY_BUILDER[level])
        if level in TRENCH_LEVELS:
            # The heading schedule is a property of the *shared* bank slot. A
            # re-roll has already left the shared-dig contract, and a straight
            # trench at one fixed heading cannot supply 17 mutually dissimilar
            # shapes, so re-rolls step to another slot's heading.
            headings = self._headings(level)
            heading = headings[(map_index + salt) % len(headings)]
            topology = NET_TOPOLOGY_CYCLE[
                (map_index + salt) % len(NET_TOPOLOGY_CYCLE)
            ]
            dig, meta = builder(rng, max(0.0, radius), angle, heading, topology)
        else:
            dig, meta = builder(rng, max(0.0, radius), angle)
        if dig is None:
            return None, {}
        meta = {
            **meta,
            "geometry": GEOMETRY_LEVEL_SOURCE[level],
            "geometry_hardness": GEOMETRY_HARDNESS[GEOMETRY_LEVEL_SOURCE[level]],
            "dig_bank_seed": seed,
            "dig_bank_salt": salt,
            "dig_bank_attempt": attempt,
            "dig_centroid_offset_tiles": round(centroid_offset(dig), 3),
            "dig_border_margin_tiles": border_margin(dig),
        }
        return dig, meta

    def _acceptable(
        self,
        level: str,
        dig: np.ndarray,
        meta: dict[str, Any],
        against: list[tuple[np.ndarray, dict[str, Any]]],
    ) -> str:
        if level in T0_GEOMETRY_LEVELS:
            coverage = dig_only_coverage(dig)
            if coverage < T0_DIG_COVERAGE_MIN:
                return "dig_bank_t0_coverage"
            meta["dig_only_workspace_coverage"] = round(coverage, 5)
        source = meta.get("foundation_source_index")
        for other, other_meta in against:
            if source is not None and other_meta.get("foundation_source_index") == source:
                return "dig_bank_source_reuse"
            if centred_iou(dig, other) >= DIG_IOU_MAX:
                return "dig_bank_iou"
        return ""

    def build(self, levels: list[str]) -> Counter[str]:
        rejections: Counter[str] = Counter()
        for level in levels:
            accepted: list[tuple[np.ndarray, dict[str, Any]]] = []
            for map_index in range(self.n_maps):
                for attempt in range(1500):
                    dig, meta = self._sample(level, map_index, 0, attempt)
                    if dig is None:
                        rejections["dig_bank_construction"] += 1
                        continue
                    reason = self._acceptable(level, dig, meta, accepted)
                    if reason:
                        rejections[reason] += 1
                        continue
                    accepted.append((dig, meta))
                    break
                else:
                    raise RuntimeError(
                        f"dig bank exhausted for {level} map {map_index}"
                    )
            self.bank[level] = accepted
        return rejections

    def get(
        self, level: str, map_index: int, salt: int
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """A dig for this slot, or ``None`` when a re-roll cannot be found.

        Exhaustion is a rejection, not a crash: the caller simply moves on to
        the next salt.
        """
        if salt == 0:
            dig, meta = self.bank[level][map_index]
            return dig.copy(), dict(meta)
        key = (level, map_index, salt)
        if key not in self.rerolls:
            self.rerolls[key] = None
            for attempt in range(300):
                dig, meta = self._sample(level, map_index, salt, attempt)
                if dig is None:
                    continue
                if self._acceptable(level, dig, meta, self.bank[level]):
                    continue
                self.rerolls[key] = (dig, meta)
                break
        found = self.rerolls[key]
        if found is None:
            return None, {}
        return found[0].copy(), dict(found[1])


# --------------------------------------------------------------------------
# RC4 — layout parameters


@dataclass(frozen=True)
class Layout:
    group: str
    azimuth: float
    sector_degrees: float
    standoff: int
    side_sign: int
    side_index: int
    n_zones: int
    zone_span_degrees: float

    def metadata(self) -> dict[str, Any]:
        return {
            "layout_group": self.group,
            "layout_azimuth_deg": round(math.degrees(self.azimuth) % 360.0, 3),
            "layout_sector_degrees": round(self.sector_degrees, 3),
            "layout_standoff_tiles": self.standoff,
            "layout_side_sign": self.side_sign,
            "layout_side_index": self.side_index,
            "layout_zones": self.n_zones,
            "layout_zone_span_deg": round(self.zone_span_degrees, 3),
        }


def layout_for(condition: v6.ConditionSpec, map_index: int) -> Layout:
    group = LAYOUT_GROUP.get(condition.id, condition.id)
    rng = rng_from(SEED_BASE, 5150, stable_key(group), map_index)
    wall = condition.id in WALL_CONDITIONS
    standoff = int(rng.choice(STANDOFF_CHOICES))
    if wall:
        n_zones = 2
        # v3.2: this is now the TARGET pad-centroid angular span, not the width
        # of a clustered lobe. v3.1's 45-70 deg collapsed the split (§8.2).
        span = float(rng.uniform(92.0, 124.0))
    else:
        n_zones = int(rng.choice([2, 2, 2, 3]))
        span = float(rng.uniform(90.0, 130.0) if n_zones == 2 else rng.uniform(110.0, 130.0))
    return Layout(
        group=group,
        azimuth=float(rng.uniform(-math.pi, math.pi)),
        sector_degrees=APRON_SECTOR_DEGREES,
        standoff=standoff,
        side_sign=int(rng.choice([-1, 1])),
        side_index=int(rng.integers(0, 4)),
        n_zones=n_zones,
        zone_span_degrees=span,
    )


# --------------------------------------------------------------------------
# dump algorithms


def _polar(dig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices(dig.shape)
    cy, cx = centroid(dig)
    return (
        ndi.distance_transform_edt(~dig),
        np.arctan2(yy - cy, xx - cx),
        np.stack([yy - cy, xx - cx]),
    )


def _interior(margin: int = SPLIT_PAD_BORDER_MARGIN) -> np.ndarray:
    yy, xx = np.indices((MAP_SIZE, MAP_SIZE))
    return (
        (yy >= margin)
        & (xx >= margin)
        & (yy <= MAP_SIZE - 1 - margin)
        & (xx <= MAP_SIZE - 1 - margin)
    )


def apron_sector(
    dig: np.ndarray, layout: Layout, target_area: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """RC4: fixed 200 deg sector, pair-seeded azimuth, capacity radial only.

    The azimuth is the first pair-seeded candidate whose 200 deg sector admits
    10.5x the dig area, so the *whole* c7x/c3x/c1p6 triple sits on one azimuth
    and c7x is feasible there. The choice depends only on the shared dig, so the
    three siblings compute it identically.
    """
    distance, angles, _ = _polar(dig)
    half = math.radians(layout.sector_degrees) / 2
    interior = _interior(2)
    need = 10.5 * int(dig.sum())

    candidates = np.degrees(layout.azimuth) + np.arange(0, 360, 15.0)
    order = np.argsort(
        [stable_key(f"{layout.group}:{value:.1f}") for value in candidates]
    )
    best_azimuth, best_area, chosen = None, -1, None
    for index in order:
        azimuth = math.radians(float(candidates[index]))
        allowed = base.largest_component(
            (distance >= layout.standoff)
            & interior
            & (v2.angle_difference(angles, azimuth) <= half)
        )
        area = int(allowed.sum())
        if area > best_area:
            best_azimuth, best_area, chosen = azimuth, area, allowed
        if area >= need:
            best_azimuth, best_area, chosen = azimuth, area, allowed
            break
    if chosen is None or best_area < target_area:
        raise RuntimeError("apron_capacity_infeasible")

    # Capacity is expressed radially: grow the annulus outward until it fits.
    for radius in range(6, 46, 2):
        allowed = base.largest_component(chosen & (distance <= radius))
        if int(allowed.sum()) < 1.02 * target_area:
            continue
        seed = v2.seed_near_angle(
            allowed, dig, best_azimuth, preferred_distance=layout.standoff + 1.5
        )
        target = base.grow_region(allowed, seed, target_area, rng)
        if int(target.sum()) < target_area:
            continue
        return target, {
            "dump_side": "near_apron",
            "dump_components_requested": 1,
            "dump_sector_degrees": float(layout.sector_degrees),
            "dump_azimuth_deg": round(math.degrees(best_azimuth) % 360.0, 3),
            "apron_max_offset_tiles": radius,
            "distance_bucket": "near",
            "dump_alignment": "foundation_capacity_apron",
        }
    raise RuntimeError("apron_capacity_infeasible")


def foundation_one_side(
    dig: np.ndarray,
    layout: Layout,
    target_area: int,
    rng: np.random.Generator,
    blocked: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    distance, _, rel = _polar(dig)
    direction = np.array([math.sin(layout.azimuth), math.cos(layout.azimuth)])
    tangent = np.array([direction[1], -direction[0]])
    projection = rel[0] * direction[0] + rel[1] * direction[1]
    lateral = np.abs(rel[0] * tangent[0] + rel[1] * tangent[1])
    lateral_limit = int(rng.integers(17, 27))
    room = (
        (projection >= layout.standoff)
        & (distance >= layout.standoff)
        & (distance <= 25)
        & (lateral <= lateral_limit)
        & _interior(2)
    )
    if blocked is not None:
        room &= ~blocked
    allowed = base.largest_component(room)
    if int(allowed.sum()) < target_area:
        raise RuntimeError("one_side_infeasible")
    seed = v2.seed_near_angle(
        allowed, dig, layout.azimuth, preferred_distance=layout.standoff + 1.5
    )
    target = base.grow_region(allowed, seed, target_area, rng)
    cardinal = int(round((math.degrees(layout.azimuth) % 360.0) / 90.0)) % 4
    return target, {
        "dump_side": base.SIDE_NAMES[(cardinal + 3) % 4],
        "dump_azimuth_deg": round(math.degrees(layout.azimuth) % 360.0, 3),
        "dump_components_requested": 1,
        "dump_access_sides": "one",
        "dump_alignment": "foundation_one_side_sector",
        "distance_bucket": "near",
        "one_side_lateral_limit_tiles": lateral_limit,
    }


def trench_flank(
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    target_area: int,
    rng: np.random.Generator,
    *,
    both_sides: bool,
    blocked: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One or both trench flanks.

    ``trn-straight-side1`` and ``trn-straight-side1-tight`` call this with the
    same layout group, so they share side sign and standoff: the only difference
    between them is the capacity band (RC4).

    ``blocked`` is the road on a v3.2 road-first zoned layout. It MUST be
    honoured: without it the flank grows straight over the corridor, which both
    fragments the road and desyncs every road metric from the exported arrays.
    """
    heading = math.radians(float(dig_meta["trench_global_angle_deg"]))
    normal = np.array([math.cos(heading), -math.sin(heading)])
    cells = np.argwhere(dig).astype(float)
    origin = cells.mean(axis=0)
    yy, xx = np.indices(dig.shape)
    projection = (yy - origin[0]) * normal[0] + (xx - origin[1]) * normal[1]
    distance = ndi.distance_transform_edt(~dig)
    reach = int(rng.integers(18, 24))
    band = (distance >= layout.standoff) & (distance <= reach) & _interior(2)
    if blocked is not None:
        band &= ~blocked

    def flank(sign: float, area: int) -> np.ndarray:
        allowed = base.largest_component(band & (sign * projection >= 1.0))
        seeds = allowed & (distance <= layout.standoff + 2.75)
        return base.grow_region(allowed, seeds, area, rng)

    if both_sides:
        areas = (target_area // 2, target_area - target_area // 2)
        parts = [flank(-1.0, areas[0]), flank(1.0, areas[1])]
        target = parts[0] | parts[1]
        side_cells = [int(part.sum()) for part in parts]
        metadata = {
            "dump_side": "trench_both_sides",
            "dump_access_sides": "both",
            "dump_alignment": "trench_both_sides_large",
            "dump_components_requested": base.target_components(np.where(target, 1, 0)),
            "distance_bucket": "immediate_near",
            "both_side_negative_cells": side_cells[0],
            "both_side_positive_cells": side_cells[1],
            "both_side_balance": round(min(side_cells) / max(1, sum(side_cells)), 4),
        }
    else:
        target = flank(float(layout.side_sign), target_area)
        metadata = {
            "dump_side": (
                "trench_main_axis_left"
                if layout.side_sign < 0
                else "trench_main_axis_right"
            ),
            "dump_access_sides": "one",
            "dump_alignment": "trench_one_side_large",
            "dump_components_requested": 1,
            "distance_bucket": "immediate_near",
            "one_side_sign": layout.side_sign,
        }
    metadata["trench_flank_reach_tiles"] = reach
    return target, metadata


def fence_axis(
    dig: np.ndarray, hull_distance: np.ndarray, layout: Layout, pad_min: float
) -> float:
    """Direction with the most room for two pads outside the fence.

    Depends only on the shared dig plus the pair-seeded layout azimuth, so it is
    deterministic per map index.
    """
    yy, xx = np.indices(dig.shape)
    cy, cx = centroid(dig)
    angles = np.arctan2(yy - cy, xx - cx)
    interior = _interior(SPLIT_PAD_BORDER_MARGIN)
    outside = (hull_distance >= pad_min) & interior
    best = (-1.0, float(layout.azimuth))
    for step in np.arange(0, 360.0, 7.5):
        azimuth = float(layout.azimuth) + math.radians(float(step))
        area = float(
            (outside & (v2.angle_difference(angles, azimuth) <= math.radians(75.0))).sum()
        )
        if area > best[0]:
            best = (area, azimuth)
    return best[1]


def wall_split_zones(
    dig: np.ndarray, layout: Layout, target_area: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """v3.2 §8.2: two genuinely separated pads outside an offset fence.

    v3.1 clustered both pads into one lobe (edge gap 2.2-3.0 tiles, centroid
    span 27-52 deg) because a straight full-width band perpendicular to the
    connector forces every pad past ``max projection of the dig + 9.5`` on ONE
    axis. Here the wall is an offset fence instead, so "behind the wall" is a
    radial statement and the pads are free to sit ~+-50 deg apart.

    The fence parameters are chosen here and handed to ``make_offset_fence_wall``
    through the returned metadata: the pads and the wall are one construction.
    """
    hull = convex_hull_image(dig)
    hull_distance = ndi.distance_transform_edt(~hull)
    dig_distance = ndi.distance_transform_edt(~dig)
    yy, xx = np.indices(dig.shape)
    cy, cx = centroid(dig)
    angles = np.arctan2(yy - cy, xx - cx)
    interior = _interior(SPLIT_PAD_BORDER_MARGIN)

    inner = float(rng.choice(FENCE_INNER_CHOICES))
    thickness = float(rng.choice(FENCE_THICKNESS_CHOICES))
    pad_min = inner + thickness + FENCE_PAD_CLEARANCE
    axis = fence_axis(dig, hull_distance, layout, pad_min)
    delta = math.radians(
        layout.zone_span_degrees / 2.0 + float(rng.uniform(-6.0, 6.0))
    )
    half_width = math.radians(FENCE_PAD_HALF_WIDTH_DEG)

    areas = (target_area // 2, target_area - target_area // 2)
    target = np.zeros_like(dig, dtype=bool)
    for sign, zone_area in zip((-1.0, 1.0), areas):
        centre = axis + sign * delta
        allowed = (
            (hull_distance >= pad_min)
            & (dig_distance >= layout.standoff)
            & interior
            & (v2.angle_difference(angles, centre) <= half_width)
        )
        allowed &= ~ndi.binary_dilation(
            target, structure=base.binary_disk(SPLIT_PAD_SEPARATION_DISK)
        )
        allowed = base.largest_component(allowed)
        if int(allowed.sum()) < zone_area:
            raise RuntimeError("separated_zone_shortfall")
        score = np.abs(hull_distance - (pad_min + 2.0)) + 6.0 * np.abs(
            v2.angle_difference(angles, centre)
        )
        score[~allowed] = np.inf
        seed = np.zeros_like(dig, dtype=bool)
        seed[np.unravel_index(np.argmin(score), score.shape)] = True
        zone = base.grow_region(allowed, seed, int(zone_area), rng)
        if int(zone.sum()) < int(zone_area):
            raise RuntimeError("separated_zone_shortfall")
        target |= zone

    pads, edge_gap, span = pad_separation(dig, target)
    if pads != 2:
        raise RuntimeError("split_pad_count")
    if edge_gap < SPLIT_PAD_EDGE_GAP_MIN:
        raise RuntimeError("split_pad_edge_gap")
    if span < SPLIT_PAD_ANGULAR_SPAN_MIN:
        raise RuntimeError("split_pad_angular_span")

    return target, {
        "dump_side": "separated_behind_fence",
        "dump_components_requested": 2,
        "dump_alignment": "separated_outside_offset_fence",
        "distance_bucket": "near_medium",
        "dump_azimuth_deg": round(math.degrees(axis) % 360.0, 3),
        "separated_zone_span_deg": round(layout.zone_span_degrees, 3),
        "separated_zone_half_width_deg": round(FENCE_PAD_HALF_WIDTH_DEG, 3),
        "fence_inner_offset_tiles": inner,
        "fence_band_thickness_tiles": thickness,
        "fence_axis_rad": float(axis),
        "fence_pad_delta_rad": float(delta),
        "fence_pad_half_width_rad": float(half_width),
    }


def separated_zones(
    dig: np.ndarray,
    layout: Layout,
    target_area: int,
    rng: np.random.Generator,
    *,
    clustered: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Split pads.

    ``clustered=True`` (wall conditions) delegates to the v3.2 fence sampler —
    see ``wall_split_zones``. The spread sampler below is the plain-``split``
    path and is byte-frozen.
    """
    if clustered:
        return wall_split_zones(dig, layout, target_area, rng)

    distance, _, rel = _polar(dig)
    yy, xx = np.indices(dig.shape)
    cy, cx = centroid(dig)
    interior = _interior(SPLIT_PAD_BORDER_MARGIN)
    n_zones = layout.n_zones
    areas = np.full(n_zones, target_area // n_zones, dtype=int)
    areas[: target_area % n_zones] += 1
    target = np.zeros_like(dig, dtype=bool)

    # 2 x 38 deg leaves a >= 150 deg empty sector at the 90-130 deg spans the
    # layout draws, and still keeps the pads as separate components.
    half_width = math.radians(38.0 if n_zones == 2 else 24.0)
    span = math.radians(layout.zone_span_degrees)
    offsets = np.linspace(-span / 2, span / 2, n_zones)
    reach = int(rng.integers(17, 27))
    for offset, zone_area in zip(offsets, areas):
        angle = layout.azimuth + float(offset)
        angles = np.arctan2(yy - cy, xx - cx)
        allowed = (
            (distance >= layout.standoff)
            & (distance <= layout.standoff + reach)
            & interior
            & (v2.angle_difference(angles, angle) <= half_width)
        )
        allowed &= ~ndi.binary_dilation(target, structure=base.binary_disk(4))
        allowed = base.largest_component(allowed)
        seed = v2.seed_near_angle(
            allowed, dig, angle, preferred_distance=layout.standoff + 4.0
        )
        zone = base.grow_region(allowed, seed, int(zone_area), rng)
        if int(zone.sum()) < int(zone_area):
            raise RuntimeError("separated_zone_shortfall")
        target |= zone
    return target, {
        "dump_side": "separated_nearby",
        "dump_components_requested": n_zones,
        "dump_alignment": "separated_spread",
        "distance_bucket": "near_medium",
        "dump_azimuth_deg": round(math.degrees(layout.azimuth) % 360.0, 3),
        "separated_zone_span_deg": round(layout.zone_span_degrees, 3),
        "separated_zone_half_width_deg": round(math.degrees(half_width), 3),
    }


def remote_edge(
    dig: np.ndarray, layout: Layout, target_area: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """§8.1: a remote haul must actually be >= 15 tiles from the dig."""
    distance = ndi.distance_transform_edt(~dig)
    order = sorted(
        range(4),
        key=lambda side: (
            -float(distance[base.side_coordinates(side)[0] < 4].max()),
            (side + layout.side_index) % 4,
        ),
    )
    for side in order:
        depth, tangent = base.side_coordinates(side)
        for max_depth in (14, 12, 10):
            span = int(
                np.clip(math.ceil(target_area / max_depth) + 8, 30, MAP_SIZE - 4)
            )
            centre = int(rng.integers(18, 47))
            low = max(1, centre - span // 2)
            high = min(MAP_SIZE - 2, low + span)
            allowed = (
                (depth < max_depth)
                & (tangent >= low)
                & (tangent <= high)
                & (distance >= REMOTE_MIN_DISTANCE_TILES)
            )
            allowed = base.largest_component(allowed)
            if int(allowed.sum()) < target_area:
                continue
            seeds = allowed & (depth <= 1)
            if not seeds.any():
                seeds = allowed & (depth <= 3)
            target = base.grow_region(allowed, seeds, target_area, rng)
            if int(target.sum()) < target_area:
                continue
            return target, {
                "dump_side": base.SIDE_NAMES[side],
                "dump_components_requested": 1,
                "distance_bucket": "far",
                "dump_alignment": "remote_edge_band",
                "edge_depth_tiles": max_depth,
            }
    raise RuntimeError("remote_edge_infeasible")


def build_dump(
    condition: v6.ConditionSpec,
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    rng: np.random.Generator,
    blocked: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    style = condition.dump_style
    if condition.capacity_level in APRON_CAPACITY_BANDS:
        band = APRON_CAPACITY_BANDS[condition.capacity_level]
    elif condition.capacity_level == "tight":
        band = TIGHT_CAPACITY_BAND
    elif condition.id in WALL_CONDITIONS:
        band = WALL_SPLIT_CAPACITY_BAND
    else:
        band = GENEROUS_CAPACITY_BANDS[style]
    factor = float(rng.uniform(*band))
    target_area = int(math.ceil(int(dig.sum()) * factor))

    if style == "capacity_apron":
        target, metadata = apron_sector(dig, layout, target_area, rng)
    elif style == "easy_surround":
        target, metadata = trench_flank(
            dig, dig_meta, layout, target_area, rng, both_sides=True
        )
    elif style in ("near_apron_large", "one_side_near"):
        if condition.family == "trench":
            target, metadata = trench_flank(
                dig, dig_meta, layout, target_area, rng,
                both_sides=False, blocked=blocked,
            )
        else:
            target, metadata = foundation_one_side(
                dig, layout, target_area, rng, blocked
            )
    elif style == "separated_zones":
        target, metadata = separated_zones(
            dig, layout, target_area, rng, clustered=condition.id in WALL_CONDITIONS
        )
    elif style == "haul_away_edge":
        target, metadata = remote_edge(dig, layout, target_area, rng)
    else:
        raise ValueError(f"unknown dump style {style!r}")

    metadata.update(
        {
            "dump_target_cells_requested": target_area,
            "capacity_factor_required": round(factor, 5),
            "capacity_band_low": band[0],
            "capacity_band_high": band[1],
            "difficulty_tier": f"capacity_{condition.capacity_level}",
            **layout.metadata(),
        }
    )
    return target, metadata


# --------------------------------------------------------------------------
# RC2 — separating wall


def make_offset_fence_wall(
    dig: np.ndarray,
    dump: np.ndarray,
    dump_meta: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """v3.2 §8.2: a gapped fence at constant offset from the dig's convex hull.

    v3.1 built a straight full-map band perpendicular to the dig->dump
    connector. That is only a wall if every pad sits past the same projection,
    which is what collapsed the split (§8.2) — and half the resulting walls were
    geodesically inert anyway (9/32 under 10% extra haul, 3 at <=1.007) because
    the sight-line gate can be satisfied by a band the machine simply walks
    around.

    Here the wall follows the work area: a band ``[inner, inner+thickness)``
    tiles outside the hull, covering the pads' azimuths plus an overhang, with
    one gate cut at least ``FENCE_GAP_MIN_OFFAXIS_DEG`` off the connector. The
    geometry is chosen by ``wall_split_zones`` and passed through ``dump_meta``:
    pads and fence are one construction, so "behind the wall" is exact by
    construction and the detour gate is what proves it.
    """
    hull = convex_hull_image(dig)
    hull_distance = ndi.distance_transform_edt(~hull)
    yy, xx = np.indices(dig.shape)
    cy, cx = centroid(dig)
    angles = np.arctan2(yy - cy, xx - cx)

    inner = float(dump_meta["fence_inner_offset_tiles"])
    thickness = float(dump_meta["fence_band_thickness_tiles"])
    axis = float(dump_meta["fence_axis_rad"])
    delta = float(dump_meta["fence_pad_delta_rad"])
    half_width = float(dump_meta["fence_pad_half_width_rad"])

    margin = math.radians(float(rng.uniform(*FENCE_ARC_MARGIN_DEG)))
    cover = delta + half_width + margin
    band = (
        (hull_distance >= inner)
        & (hull_distance < inner + thickness)
        & (v2.angle_difference(angles, axis) <= cover)
    )
    band &= ~ndi.binary_dilation(dump, structure=base.binary_disk(3))
    band &= ~ndi.binary_dilation(dig, structure=base.binary_disk(3))
    if int(band.sum()) < 60:
        raise RuntimeError("fence_band_too_small")

    offaxis_max = max(FENCE_GAP_MIN_OFFAXIS_DEG + 2.0, math.degrees(cover) - 8.0)
    for _ in range(24):
        gap_width = int(rng.integers(12, 17))
        offaxis = math.radians(
            float(rng.uniform(FENCE_GAP_MIN_OFFAXIS_DEG, offaxis_max))
        )
        gate_angle = axis + float(rng.choice([-1.0, 1.0])) * offaxis
        ring = band & (v2.angle_difference(angles, gate_angle) <= math.radians(6.0))
        if not ring.any():
            continue
        gap_centre = centroid(ring)
        gap = np.hypot(yy - gap_centre[0], xx - gap_centre[1]) <= gap_width / 2
        wall = band & ~gap
        components, min_cells, min_extent = wall_component_stats(wall)
        if components < WALL_MIN_COMPONENTS:
            continue
        if min_cells < WALL_MIN_COMPONENT_CELLS:
            continue
        if min_extent < WALL_MIN_COMPONENT_EXTENT:
            continue
        min_thickness = wall_min_thickness(wall)
        if min_thickness < WALL_MIN_THICKNESS_TILES:
            continue
        offaxis_tiles = gap_offaxis_tiles(dig, dump, tuple(gap_centre))
        if offaxis_tiles < WALL_GAP_OFFAXIS_MIN:
            continue
        ratio, mean_free, mean_walled, unreachable = haul_detour(dig, dump, wall)
        if not math.isfinite(ratio) or ratio < WALL_DETOUR_MIN:
            continue
        connector = float(np.linalg.norm(centroid(dump) - centroid(dig)))
        return wall, {
            "wall_gap_tiles": gap_width,
            "wall_thickness_tiles": min_thickness,
            "wall_band_thickness_tiles": int(thickness),
            "wall_style": "offset_fence",
            "wall_fence_inner_offset_tiles": inner,
            "wall_fence_cover_deg": round(math.degrees(cover), 2),
            "wall_gap_offaxis_deg": round(math.degrees(offaxis), 2),
            "wall_gap_offaxis_tiles": round(offaxis_tiles, 3),
            "wall_connector_tiles": round(connector, 3),
            "wall_detour_ratio": round(ratio, 4),
            "wall_haul_open_tiles": round(mean_free, 3),
            "wall_haul_walled_tiles": round(mean_walled, 3),
            "wall_dump_unreachable_fraction": round(unreachable, 4),
            "wall_bite_fraction": round(sight_line_bite(dig, dump, wall), 4),
            "wall_components": components,
            "wall_min_component_cells": min_cells,
            "wall_min_component_extent_tiles": min_extent,
            "wall_cells": int(wall.sum()),
        }
    raise RuntimeError("separating_wall_infeasible")


# --------------------------------------------------------------------------
# RC3 — objects inside the working annulus, roads across the near band


def place_objects_v7(
    dig: np.ndarray, protect: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[int]]:
    """Objects live in the 2-8 tile working annulus, not banished from it."""
    distance = ndi.distance_transform_edt(~dig)
    forbidden = ndi.binary_dilation(protect, structure=base.binary_disk(2)) | ~_interior(3)
    occupancy = np.zeros_like(dig, dtype=bool)
    inner = np.argwhere((distance >= 3) & (distance <= 9) & ~forbidden)
    outer = np.argwhere((distance > 9) & (distance <= 22) & ~forbidden)
    if len(inner) == 0:
        raise RuntimeError("object_annulus_empty")
    # At least half of the objects are seeded inside the working annulus.
    inner_quota = max(1, (count + 1) // 2)
    areas: list[int] = []
    placed = 0
    for index in range(count):
        pool = inner if index < inner_quota or len(outer) == 0 else outer
        for _ in range(160):
            centre = pool[int(rng.integers(0, len(pool)))]
            length = float(rng.uniform(3, 7))
            width = float(rng.uniform(2, 5))
            candidate = base.rotated_rectangle(
                (float(centre[0]), float(centre[1])),
                length,
                width,
                float(rng.uniform(0, math.pi)),
            )
            if not candidate.any() or np.any(candidate & forbidden):
                continue
            if np.any(
                ndi.binary_dilation(candidate, structure=base.binary_disk(3)) & occupancy
            ):
                continue
            occupancy |= candidate
            forbidden |= ndi.binary_dilation(candidate, structure=base.binary_disk(2))
            areas.append(int(candidate.sum()))
            placed += 1
            break
    if placed != count:
        raise RuntimeError("object_band_shortfall")
    return occupancy, areas


def object_intrusion(dig: np.ndarray, occupancy: np.ndarray) -> dict[str, float]:
    near = annulus(dig, 4.0)
    work = annulus(dig, 6.0)
    return {
        "annulus4_free_fraction": round(
            float((near & ~occupancy).sum() / max(1, near.sum())), 4
        ),
        "annulus6_blocked_fraction": round(
            float((work & occupancy).sum() / max(1, work.sum())), 4
        ),
    }


def border_sides_touched(mask: np.ndarray) -> int:
    # bool(...) matters: numpy bools add as logical OR, so a naive sum is 1.
    return sum(
        bool(edge.any())
        for edge in (mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])
    )


def make_two_border_road(
    dig: np.ndarray,
    protect: np.ndarray,
    rng: np.random.Generator,
    radius_range: tuple[int, int] = ROAD_RADIUS_RANGE_RING,
    dig_annulus_sterilize_min: float = ROAD_ANNULUS_STERILIZE_MIN,
) -> tuple[np.ndarray, dict[str, Any]]:
    """A corridor between two different map borders that hugs the near band.

    ``radius_range`` is the only v3.2 knob: zoned layouts allow a 2-tile radius
    so the corridor can slip through the dig->dump standoff. Ring layouts keep
    ``(3, 5)`` and are byte-frozen.
    """
    protected = ndi.binary_dilation(protect, structure=base.binary_disk(2))
    distance_from_protected = ndi.distance_transform_edt(~protected)
    distance_from_dig = ndi.distance_transform_edt(~dig)
    work = annulus(dig, 6.0)

    for _ in range(90):
        road_radius = int(rng.integers(*radius_range))
        allowed = distance_from_protected >= road_radius + 0.5
        near = np.argwhere(
            allowed & (distance_from_dig <= road_radius + 6) & (distance_from_dig >= 2)
        )
        if len(near) == 0:
            continue
        sides = rng.permutation(4)[:2]
        ends = []
        for side in sides:
            if side == 0:
                candidates = np.argwhere(allowed[0:1, :])
                candidates = np.column_stack([np.zeros(len(candidates), int), candidates[:, 1]])
            elif side == 1:
                candidates = np.argwhere(allowed[:, -1:])
                candidates = np.column_stack(
                    [candidates[:, 0], np.full(len(candidates), MAP_SIZE - 1)]
                )
            elif side == 2:
                candidates = np.argwhere(allowed[-1:, :])
                candidates = np.column_stack(
                    [np.full(len(candidates), MAP_SIZE - 1), candidates[:, 1]]
                )
            else:
                candidates = np.argwhere(allowed[:, 0:1])
                candidates = np.column_stack([candidates[:, 0], np.zeros(len(candidates), int)])
            if len(candidates) == 0:
                ends = []
                break
            ends.append(candidates[int(rng.integers(0, len(candidates)))])
        if len(ends) != 2:
            continue
        waypoint = near[int(rng.integers(0, len(near)))]

        noise = base.smooth_noise(rng, sigma=float(rng.uniform(5, 9)))
        cost = 1.0 + 0.6 * noise
        cost[~allowed] = np.inf
        centerline = np.zeros_like(dig, dtype=bool)
        ok = True
        for start, goal in ((ends[0], waypoint), (waypoint, ends[1])):
            try:
                path, _ = route_through_array(
                    cost,
                    tuple(map(int, start)),
                    tuple(map(int, goal)),
                    fully_connected=True,
                    geometric=True,
                )
            except ValueError:
                ok = False
                break
            centerline |= base.cells_to_mask(path)
        if not ok:
            continue
        road = ndi.binary_dilation(centerline, structure=base.binary_disk(road_radius))
        if np.any(road & protected) or int(road.sum()) < 90:
            continue
        if border_sides_touched(road) < 2:
            continue
        sterilized = float((road & work).sum() / max(1, work.sum()))
        free = float((work & ~road).sum() / max(1, work.sum()))
        if sterilized < dig_annulus_sterilize_min or free < ROAD_ANNULUS_FREE_MIN:
            continue
        return road, {
            "road_radius_tiles": road_radius,
            "road_cells": int(road.sum()),
            "road_borders_touched": border_sides_touched(road),
            # §8.2: `road_annulus6_*` has always measured the DIG annulus.
            # Named, not renamed — the 21 frozen conditions carry the old column.
            "road_gate_annulus": "dig",
            "road_annulus6_sterilized_fraction": round(sterilized, 4),
            "road_annulus6_free_fraction": round(free, 4),
            "road_dig_annulus6_sterilized_fraction": round(sterilized, 4),
            "road_dig_annulus6_free_fraction": round(free, 4),
        }
    raise RuntimeError("access_road_generation_failed")


def zoned_road_metrics(
    dig: np.ndarray, dump: np.ndarray, road: np.ndarray
) -> dict[str, Any]:
    """§8.2: on a zoned layout the near-dump band is NOT the dig annulus.

    Every value here is computed from the FINAL corridor — the same mask the
    validator recovers from ``dumpability``/``occupancy`` — not from the road as
    the placer built it. The two differ the moment anything is grown on top of
    the corridor, and a manifest that reports the pre-clip road is a lie the
    validator catches as a mismatch.
    """
    dig_ring = annulus(dig, 6.0)
    dump_ring = dump_annulus(dig, dump, 6.0)
    return {
        "road_gate_annulus": "dump",
        # legacy column names, restated from the final corridor
        "road_annulus6_sterilized_fraction": round(
            float((road & dig_ring).sum() / max(1, dig_ring.sum())), 4
        ),
        "road_annulus6_free_fraction": round(
            float((dig_ring & ~road).sum() / max(1, dig_ring.sum())), 4
        ),
        "road_cells": int(road.sum()),
        "road_borders_touched": border_sides_touched(road),
        "road_components": int(
            ndi.label(road, structure=np.ones((3, 3), dtype=np.uint8))[1]
        ),
        "road_sightline_cross_fraction": round(
            sight_line_bite(dig, dump, road), 4
        ),
        "road_dump_annulus6_sterilized_fraction": round(
            float((road & dump_ring).sum() / max(1, dump_ring.sum())), 4
        ),
        "road_dump_annulus6_free_fraction": round(
            float((dump_ring & ~road).sum() / max(1, dump_ring.sum())), 4
        ),
        "road_dig_annulus6_sterilized_fraction": round(
            float((road & dig_ring).sum() / max(1, dig_ring.sum())), 4
        ),
        "road_dig_annulus6_free_fraction": round(
            float((dig_ring & ~road).sum() / max(1, dig_ring.sum())), 4
        ),
    }


def zoned_road_bites(metrics: dict[str, Any]) -> bool:
    return (
        metrics["road_sightline_cross_fraction"] >= ROAD_SIGHTLINE_CROSS_MIN
        or metrics["road_dump_annulus6_sterilized_fraction"]
        >= ROAD_DUMP_ANNULUS_STERILIZE_MIN
    )


class SiteFactoryV7:
    """Site constraints keyed by taxonomy site level, with intrusion gates."""

    @staticmethod
    def make(
        condition: v6.ConditionSpec,
        dig: np.ndarray,
        dump: np.ndarray,
        protect: np.ndarray,
        rng: np.random.Generator,
        dump_meta: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        level = condition.site_level
        occupancy = np.zeros_like(dig, dtype=bool)
        nondump = np.zeros_like(dig, dtype=bool)
        corridor = np.zeros_like(dig, dtype=bool)
        metadata: dict[str, Any] = {
            "site_level": level,
            "site_class": SITE_CLASS_TOKENS[level],
            "object_count": 0,
            "wall_gap_tiles": 0,
        }

        if level == "clean":
            pass
        elif level in ("obj", "obj1"):
            low, high = OBJECT_BANDS[level]
            requested = int(rng.integers(low, high + 1))
            occupancy, areas = place_objects_v7(dig, protect, requested, rng)
            intrusion = object_intrusion(dig, occupancy)
            if intrusion["annulus4_free_fraction"] < OBJECT_NEAR_ANNULUS_FREE_MIN:
                raise RuntimeError("object_corridor_blocked")
            if intrusion["annulus6_blocked_fraction"] < OBJECT_ANNULUS_BLOCK_MIN[level]:
                raise RuntimeError("object_intrusion_too_low")
            metadata.update(intrusion)
            metadata["object_count"] = requested
            metadata["object_footprint_profile"] = OBJECT_FOOTPRINT_PROFILE
            metadata["object_area_cells_mean"] = round(float(np.mean(areas)), 3)
            metadata["object_area_cells_min"] = int(min(areas))
            metadata["object_area_cells_max"] = int(max(areas))
        elif level == "road":
            road, road_meta = make_two_border_road(dig, protect, rng)
            nondump |= road
            corridor |= road
            metadata.update(road_meta)
        elif level == "wall":
            wall, wall_meta = make_offset_fence_wall(dig, dump, dump_meta or {}, rng)
            occupancy |= wall
            metadata.update(wall_meta)
        else:
            raise ValueError(f"unknown site level {level!r}")

        # `protect` is the mask the site may never eat into: dig | dump for a
        # zoned layout, dig alone on a ring (where the dump is defined *after*
        # the site, as everything legal and free).
        occupancy &= ~protect
        nondump &= ~(protect | occupancy)
        corridor &= nondump
        dumpability = ~(nondump | occupancy)
        return occupancy, dumpability, corridor, metadata


# --------------------------------------------------------------------------
# sample construction


def _common_metadata(
    condition: v6.ConditionSpec,
    dig: np.ndarray,
    dump: np.ndarray,
    occupancy: np.ndarray,
    dumpability: np.ndarray,
    gate: base.StaticGate,
) -> dict[str, Any]:
    _, dig_components = ndi.label(dig, structure=np.ones((3, 3), dtype=np.uint8))
    return {
        "schema": SCHEMA,
        "condition_id": condition.id,
        "tier": condition.tier,
        "preview": condition.preview,
        "anchor_condition_id": condition.anchor or "",
        "family": condition.family,
        "geometry_level": condition.geometry_level,
        "dump_level": condition.dump_level,
        "capacity_level": condition.capacity_level,
        "site_level": condition.site_level,
        "capacity_band_token": CAPACITY_TOKENS[condition.capacity_level],
        "dump_layout": condition.dump_layout,
        "dump_style": condition.dump_style,
        "pair_group": condition.pair_group or "",
        "dig_cells": int(dig.sum()),
        "dig_components_actual": int(dig_components),
        "dump_cells": int(dump.sum()),
        "dump_to_dig_area_ratio": round(float(dump.sum() / max(1, dig.sum())), 4),
        "reachable_dump_to_dig_ratio": round(
            float(gate.reachable_dump_cells_post / max(1, dig.sum())), 4
        ),
        "obstacle_cells": int(occupancy.sum()),
        "nondump_cells": int((~dumpability & ~occupancy).sum()),
        "tile_size_m": base.TILE_SIZE_M,
        "static_gate_is_action_witness": False,
    }


def _split_metrics(dig: np.ndarray, dump: np.ndarray) -> dict[str, Any]:
    pads, edge_gap, span = pad_separation(dig, dump)
    return {
        "dump_max_angular_gap_deg": round(max_angular_gap_degrees(dig, dump), 2),
        "dump_border_margin_tiles": border_margin(dump),
        "dump_pad_count": pads,
        "dump_pad_edge_gap_tiles": round(edge_gap, 3),
        "dump_pad_angular_span_deg": round(span, 2),
    }


def make_ring_sample(
    condition: v6.ConditionSpec,
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    rng: np.random.Generator,
) -> tuple[base.Sample | None, str]:
    """Foundation ring: every legal free cell around the dig is a dump target."""
    planning_area = int(math.ceil(1.25 * int(dig.sum())))
    planning_dump, _ = v2.DumpFactoryV2.broad_nearby(
        dig, planning_area, int(rng.integers(0, 4)), rng
    )
    if int(planning_dump.sum()) < planning_area:
        return None, "planning_dump_generation_shortfall"
    # RC3: on a ring layout the whole legal-free area is the dump, so the site
    # constraint may only protect the DIG. v6 protected `dig | planning_dump`,
    # which is exactly why no road ever crossed the dump ring.
    occupancy, dumpability, corridor, site_meta = SiteFactoryV7.make(
        condition, dig, planning_dump, dig, rng
    )

    dump = (~dig) & (~occupancy) & dumpability
    target = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    target[dig] = -1
    target[dump] = 1
    gate = base.static_gate(target, occupancy, dumpability)
    if not gate.accepted:
        return None, gate.reason
    if condition.id in T0_CONDITIONS and gate.dig_workspace_coverage_post < T0_DIG_COVERAGE_MIN:
        return None, "t0_dig_coverage_contract"

    reachable_ratio = float(gate.reachable_dump_cells_post / max(1, dig.sum()))
    capacity_ratio = float(dump.sum() / max(1, dig.sum()))
    band = RING_CAPACITY_BANDS.get(condition.capacity_level)
    if band is None:
        if reachable_ratio + 1e-8 < RING_MIN_CAPACITY:
            return None, "reachable_capacity_contract"
    elif not band[0] <= capacity_ratio <= band[1]:
        return None, "capacity_band_contract"

    legal_free = (~dig) & (~occupancy) & dumpability
    coverage = float((dump & legal_free).sum() / max(1, legal_free.sum()))
    if not math.isclose(coverage, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return None, "all_around_coverage_contract"

    metadata = {
        **_common_metadata(condition, dig, dump, occupancy, dumpability, gate),
        "dump_components_actual": base.target_components(target),
        "dump_components_requested": base.target_components(target),
        "dump_target_cells_requested": int(dump.sum()),
        "capacity_factor_required": (band[0] if band is not None else RING_MIN_CAPACITY),
        "capacity_band_low": (band[0] if band is not None else RING_MIN_CAPACITY),
        "capacity_band_high": (band[1] if band is not None else -1.0),
        "difficulty_tier": "all_legal_free",
        "dump_side": "foundation_all_around",
        "dump_access_sides": "all",
        "dump_alignment": "all_legal_free_around_foundation",
        "distance_bucket": "immediate",
        "dump_coverage_of_legal_free": round(coverage, 6),
        "planning_dump_cells": int(planning_dump.sum()),
        **layout.metadata(),
        **v2.dig_to_dump_distance_metrics(dig, dump),
        **_split_metrics(dig, dump),
        **dig_meta,
        **site_meta,
    }
    return (
        base.Sample(
            target=target,
            occupancy=occupancy,
            dumpability=dumpability,
            action=np.zeros_like(target, dtype=np.int8),
            distance=base.compute_geodesic_distance(target, occupancy),
            service_corridor=corridor,
            metadata=metadata,
            gate=gate,
        ),
        "",
    )


def make_zoned_sample(
    condition: v6.ConditionSpec,
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    rng: np.random.Generator,
) -> tuple[base.Sample | None, str]:
    """Apron band, one side, split zones, remote edge."""
    # §8.2: on a zoned road layout the road is built FIRST, against
    # `dilate(dig, disk(2))` only — the same ordering rings already use. v3.1
    # protected `dilate(dig | dump, disk(2))`, so the corridor could never enter
    # the haul path and crossed 0.000-0.004 of the dig<->dump sight-lines.
    # The dump is then grown around the road, which keeps every dump cell
    # dumpable (a road over a designated dump cell is a hard static-gate reject).
    road: np.ndarray | None = None
    road_meta: dict[str, Any] = {}
    if condition.id in V32_ROAD_FIRST_CONDITIONS:
        road, road_meta = make_two_border_road(
            dig, dig, rng, ROAD_RADIUS_RANGE_ZONED, dig_annulus_sterilize_min=0.0
        )

    dump, dump_meta = build_dump(condition, dig, dig_meta, layout, rng, road)
    requested = int(dump_meta["dump_target_cells_requested"])
    if int(dump.sum()) < requested:
        return None, "capacity_generation_shortfall"
    if road is None:
        occupancy, dumpability, corridor, site_meta = SiteFactoryV7.make(
            condition, dig, dump, dig | dump, rng, dump_meta
        )
    else:
        occupancy = np.zeros_like(dig, dtype=bool)
        corridor = road & ~(dig | dump)
        dumpability = ~corridor
        site_meta = {
            "site_level": condition.site_level,
            "site_class": SITE_CLASS_TOKENS[condition.site_level],
            "object_count": 0,
            "wall_gap_tiles": 0,
            **road_meta,
            **zoned_road_metrics(dig, dump, corridor),
        }
        # The dump must have been grown AROUND the corridor, not over it.
        if int(corridor.sum()) != int(road.sum()):
            return None, "zoned_road_clipped_contract"
        if site_meta["road_components"] != 1:
            return None, "zoned_road_split_contract"
        if site_meta["road_borders_touched"] < 2:
            return None, "zoned_road_border_contract"
        if not zoned_road_bites(site_meta):
            return None, "zoned_road_bite_contract"
        if site_meta["road_dig_annulus6_free_fraction"] < ROAD_ANNULUS_FREE_MIN:
            return None, "zoned_road_free_contract"

    target = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    target[dig] = -1
    target[dump & ~dig] = 1
    components = base.target_components(target)
    if components != int(dump_meta["dump_components_requested"]):
        return None, "dump_component_contract"

    gate = base.static_gate(target, occupancy, dumpability)
    if not gate.accepted:
        return None, gate.reason
    if condition.id in T0_CONDITIONS and gate.dig_workspace_coverage_post < T0_DIG_COVERAGE_MIN:
        return None, "t0_dig_coverage_contract"

    required = float(dump_meta["capacity_factor_required"])
    capacity_ratio = float(dump.sum() / max(1, dig.sum()))
    reachable_ratio = float(gate.reachable_dump_cells_post / max(1, dig.sum()))
    low = float(dump_meta["capacity_band_low"])
    high = float(dump_meta["capacity_band_high"])
    if condition.capacity_level in APRON_CAPACITY_BANDS or condition.capacity_level == "tight":
        # An explicit band is a statement about reachable single-layer capacity.
        if not low <= reachable_ratio <= high:
            return None, "capacity_band_contract"
    else:
        if capacity_ratio + 1e-8 < required:
            return None, "capacity_ratio_contract"
        if reachable_ratio + 1e-8 < GENEROUS_REACHABLE_FLOOR:
            return None, "generous_capacity_floor"

    distance_metrics = v2.dig_to_dump_distance_metrics(dig, dump)
    limits = (
        TRENCH_MEDIAN_LIMITS if condition.family == "trench" else FOUNDATION_MEDIAN_LIMITS
    )
    limit = limits.get(condition.dump_style)
    if condition.id in WALL_CONDITIONS:
        limit = WALL_SPLIT_MEDIAN_LIMIT
    if limit is not None:
        # The inherited limits assumed the constant 3.0 standoff of v2-v6.
        limit += max(0, layout.standoff - 3)
        if distance_metrics["dig_dump_distance_median_tiles"] > limit:
            return None, "near_distance_contract"
    if condition.dump_level == "remote":
        if distance_metrics["dig_dump_distance_min_tiles"] < REMOTE_MIN_DISTANCE_TILES:
            return None, "remote_distance_contract"
    if condition.dump_level == "split":
        if max_angular_gap_degrees(dig, dump) < SPLIT_MAX_ANGULAR_GAP_MIN:
            return None, "split_angular_gap_contract"
        if border_margin(dump) < SPLIT_PAD_BORDER_MARGIN:
            return None, "split_border_margin_contract"
    if condition.id in WALL_CONDITIONS:
        # §8.2: a wall-condition `split` must be a real split, comparable to the
        # plain-split ranges (edge gap 4.1-31.3 tiles, centroid span 88-135 deg).
        _, edge_gap, span = pad_separation(dig, dump)
        if edge_gap < SPLIT_PAD_EDGE_GAP_MIN:
            return None, "split_pad_edge_gap_contract"
        if span < SPLIT_PAD_ANGULAR_SPAN_MIN:
            return None, "split_pad_angular_span_contract"

    extra: dict[str, Any] = {}
    if condition.geometry_level == "straight" and condition.dump_level == "side1":
        coverage = axial_coverage(dig, dump, float(dig_meta["trench_global_angle_deg"]))
        if coverage < 0.90:
            return None, "trench_axial_coverage_contract"
        extra["trench_axial_coverage"] = round(coverage, 4)

    metadata = {
        **_common_metadata(condition, dig, dump, occupancy, dumpability, gate),
        "dump_components_actual": components,
        **distance_metrics,
        **_split_metrics(dig, dump),
        **extra,
        **dig_meta,
        **dump_meta,
        **site_meta,
    }
    return (
        base.Sample(
            target=target,
            occupancy=occupancy,
            dumpability=dumpability,
            action=np.zeros_like(target, dtype=np.int8),
            distance=base.compute_geodesic_distance(target, occupancy),
            service_corridor=corridor,
            metadata=metadata,
            gate=gate,
        ),
        "",
    )


def make_map(
    condition: v6.ConditionSpec,
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    rng: np.random.Generator,
) -> tuple[base.Sample | None, str]:
    maker = (
        make_ring_sample
        if condition.dump_style == "ring_all_legal_free"
        else make_zoned_sample
    )
    try:
        return maker(condition, dig, dig_meta, layout, rng)
    except RuntimeError as exc:
        return None, str(exc)


def generate_condition(
    condition: v6.ConditionSpec,
    condition_index: int,
    bank: DigBankV7,
    n_maps: int,
    max_attempts: int,
) -> tuple[list[base.Sample], Counter[str], list[str]]:
    samples: list[base.Sample] = []
    rejections: Counter[str] = Counter()
    unsatisfied: list[str] = []
    accepted_digs: list[np.ndarray] = []
    for map_index in range(n_maps):
        layout = layout_for(condition, map_index)
        accepted: base.Sample | None = None
        for attempt in range(max_attempts):
            # Each re-rolled dig gets REROLL_DUMP_ATTEMPTS dump/site tries before
            # the next one is drawn, so a hard map costs a bounded number of dig
            # constructions instead of one per attempt.
            salt = (
                0
                if attempt < SHARED_DIG_ATTEMPTS
                else 1 + (attempt - SHARED_DIG_ATTEMPTS) // REROLL_DUMP_ATTEMPTS
            )
            dig, dig_meta = bank.get(condition.geometry_level, map_index, salt)
            if dig is None:
                rejections["dig_reroll_exhausted"] += 1
                continue
            # RC5: the bank is mutually dissimilar, but a re-roll could still
            # land on top of an earlier map *of this condition*.
            if salt and any(centred_iou(dig, other) >= DIG_IOU_MAX for other in accepted_digs):
                rejections["condition_dig_iou"] += 1
                continue
            seed = int(
                np.random.SeedSequence(
                    [SEED_BASE, condition_index, map_index, attempt]
                ).generate_state(1)[0]
            )
            sample, reason = make_map(
                condition, dig, dig_meta, layout, np.random.default_rng(seed)
            )
            if sample is None:
                rejections[reason] += 1
                continue
            sample.metadata.update(
                {
                    "map_index": map_index,
                    "attempt": attempt,
                    "attempt_seed": seed,
                    "seed_base": SEED_BASE,
                    "condition_index": condition_index,
                    "shared_dig": int(salt == 0),
                    "dig_sha256": sha256_mask(sample.target < 0),
                    "occupancy_sha256": sha256_mask(sample.occupancy),
                }
            )
            accepted = sample
            accepted_digs.append(sample.target < 0)
            break
        if accepted is None:
            unsatisfied.append(
                f"{condition.id} map {map_index}: no accepted sample in "
                f"{max_attempts} attempts; rejections={dict(rejections)}"
            )
            continue
        samples.append(accepted)
    return samples, rejections, unsatisfied


# --------------------------------------------------------------------------
# output


def sample_index_of(condition_index: int, map_index: int) -> int:
    return 100 * condition_index + map_index


def map_id_of(sample_index: int) -> str:
    return f"{MAP_ID_PREFIX}-{sample_index:04d}"


def write_condition(
    output: Path,
    condition: v6.ConditionSpec,
    condition_index: int,
    samples: list[base.Sample],
) -> list[dict[str, Any]]:
    dataset = output / "dataset"
    folder = output / condition.id
    (folder / "previews").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        map_index = int(sample.metadata["map_index"])
        sample_index = sample_index_of(condition_index, map_index)
        map_id = map_id_of(sample_index)
        for folder_name, attribute in ARRAY_FOLDERS.items():
            np.save(
                dataset / folder_name / f"img_{sample_index}.npy",
                getattr(sample, attribute),
            )
        record = {
            "sample_index": sample_index,
            "map_id": map_id,
            **sample.metadata,
            **{f"gate_{k}": v for k, v in asdict(sample.gate).items()},
        }
        rows.append(record)
        (output / "review_metadata" / f"img_{sample_index}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
        )
        base.render_sample(
            sample,
            folder / "previews" / f"{map_index:02d}__{map_id}.png",
            f"{condition.id} #{map_index:02d} ({map_id})",
        )
    if samples:
        v6.render_condition_overview(folder / "overview.png", condition, samples)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "conditionId": condition.id,
                "cellId": condition.id,
                "tier": condition.tier,
                "tierLabel": tax.TIER_LABELS[condition.tier],
                "preview": condition.preview,
                "anchorConditionId": condition.anchor,
                "family": condition.family,
                "factorLevels": condition.levels,
                "factors": {
                    "geometryClass": condition.geometry,
                    "dumpLayout": condition.dump_layout,
                    "siteClass": SITE_CLASS_TOKENS[condition.site_level],
                    "capacityBand": CAPACITY_TOKENS[condition.capacity_level],
                },
                "objectBand": list(OBJECT_BANDS[condition.site_level]),
                "layoutGroup": LAYOUT_GROUP.get(condition.id, condition.id),
                "seedBase": SEED_BASE,
                "mapCount": len(rows),
                "maps": [
                    {
                        "id": row["map_id"],
                        "sampleIndex": row["sample_index"],
                        "mapIndex": row["map_index"],
                        "arrays": {
                            name: f"dataset/{name}/img_{row['sample_index']}.npy"
                            for name in ARRAY_FOLDERS
                        },
                        "objectCount": row["object_count"],
                        "digCells": row["dig_cells"],
                        "dumpCells": row["dump_cells"],
                        "capacityRatio": row["dump_to_dig_area_ratio"],
                        "sharedDig": bool(row["shared_dig"]),
                        "digSha256": row["dig_sha256"],
                    }
                    for row in rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return rows


def write_manifest(output: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_conditions_csv(output: Path, counts: dict[str, int]) -> None:
    with (output / "conditions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tax.CSV_COLUMNS)
        writer.writeheader()
        for condition in CONDITIONS:
            levels = condition.levels
            writer.writerow(
                {
                    "condition_id": condition.id,
                    "old_cell_id": condition.id,
                    "family": condition.family,
                    "geometry": levels["geometry"],
                    "dump": levels["dump"],
                    "capacity": levels["capacity"],
                    "site": levels["site"],
                    "distance": levels["distance"],
                    "tier": condition.tier,
                    "preview": "true" if condition.preview else "false",
                    "anchor_condition_id": condition.anchor or "",
                    "n_maps": counts.get(condition.id, 0),
                }
            )


def write_terra_metadata(output: Path, rows: list[dict[str, Any]]) -> None:
    destination = output / "dataset" / "metadata"
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        payload = {
            "schema": f"{SCHEMA}_axis_metadata",
            "geometry": row["geometry"],
            "trench_axes_count": int(row.get("trench_axes_count", -1) or -1),
            "trench_topology": row.get("trench_topology", ""),
            "axes_ABC": row.get("axes_ABC", []),
        }
        (destination / f"trench_{row['sample_index']}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        )


def write_readme(output: Path, counts: dict[str, int]) -> None:
    lines = [
        "# Terra curriculum v3.1 review bank (v3.2 amended)",
        "",
        "Taxonomy-native bank: one folder per **condition id**, no stage folders.",
        "Difficulty tier is computed from factor levels, never hand-assigned.",
        "See `../../terra-digging-benchmark-site/docs/CURRICULUM_TAXONOMY_SPEC.md`"
        " sections 8, 8.1 and 8.2.",
        "",
        f"- seed base: `{SEED_BASE}` (fully reproducible)",
        f"- conditions: {len(CONDITIONS)}",
        f"- maps: {sum(counts.values())} "
        f"({MAPS_PER_CONDITION} per condition, "
        f"{PREVIEW_MAPS_PER_CONDITION} for the two remote-haul previews)",
        f"- map ids: `{MAP_ID_PREFIX}-NNNN`",
        "",
        "v3.1 fixes RC1-RC5 from the five-agent review of the v3 bank:",
        "generous dump-layout capacity, a wall that actually separates dig from",
        "dump, site constraints that intrude on the working area, pair-seeded",
        "dump layout parameters, and a de-duplicated dig bank.",
        "",
        "## Layout",
        "",
        "- `<condition-id>/manifest.json` — factor levels, tier, per-map metrics",
        "- `<condition-id>/previews/*.png` — one labelled composite per map",
        "- `<condition-id>/overview.png` — the whole condition on one sheet",
        "- `dataset/{images,occupancy,dumpability,actions,distance}/img_N.npy`"
        " — Terra arrays, flat and shared across conditions",
        "- `manifest.csv` — every map, every measured factor",
        "- `conditions.csv` — spec section 4 columns",
        "- `GENERATION_NOTES.md` — fixes, realized ranges, constraint reports",
        "",
        "The static gate is not an action-level completion witness.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-foundations",
        type=Path,
        default=script_dir.parent / "full_data" / "foundations_dumpzones_v3",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maps",
        type=int,
        default=0,
        help="override maps per condition (0 = spec counts; smoke runs use 2)",
    )
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--only", default="", help="comma-separated condition ids")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="with --only: carry the existing manifest rows for every condition "
        "that is not regenerated, so the bank stays complete",
    )
    return parser.parse_args()


def main() -> None:
    v6.assert_conditions_match_taxonomy()
    args = parse_args()
    output = args.output.resolve()
    for folder in (
        output,
        output / "review_metadata",
        *(output / "dataset" / name for name in ARRAY_FOLDERS),
    ):
        folder.mkdir(parents=True, exist_ok=True)

    selected = set(filter(None, args.only.split(",")))
    factory = GeometryFactoryV7(args.source_foundations)
    bank_size = args.maps or MAPS_PER_CONDITION
    bank = DigBankV7(factory, bank_size)
    bank_rejections = bank.build(sorted(GEOMETRY_LEVEL_SOURCE))
    print(f"dig bank built: {bank_size} per level, rejections={dict(bank_rejections)}")

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    rejection_totals: Counter[str] = Counter()
    unsatisfied: list[str] = []

    for condition_index, condition in enumerate(CONDITIONS):
        if selected and condition.id not in selected:
            continue
        n_maps = args.maps or condition.n_maps
        samples, rejections, failures = generate_condition(
            condition, condition_index, bank, n_maps, args.max_attempts
        )
        rejection_totals.update(rejections)
        unsatisfied.extend(failures)
        condition_rows = write_condition(output, condition, condition_index, samples)
        rows.extend(condition_rows)
        counts[condition.id] = len(condition_rows)
        rerolled = sum(1 for row in condition_rows if not row["shared_dig"])
        print(
            f"[{condition_index + 1:02d}/{len(CONDITIONS)}] {condition.id}: "
            f"{len(condition_rows)}/{n_maps} maps"
            + (f" rerolled={rerolled}" if rerolled else "")
            + (f" UNSATISFIED={len(failures)}" if failures else "")
            + (f" rejections={dict(rejections.most_common(4))}" if rejections else ""),
            flush=True,
        )

    write_terra_metadata(output, rows)          # only the rows just generated
    regenerated = sorted(counts)   # conditions built by THIS invocation

    if args.resume:
        # Targeted regeneration (§8.2): everything outside --only is byte-frozen,
        # so its manifest rows are carried over verbatim and its
        # dataset/metadata/*.json are left alone.
        with (output / "manifest.csv").open(newline="") as handle:
            previous = [
                row
                for row in csv.DictReader(handle)
                if row["condition_id"] not in selected
            ]
        for row in previous:
            row["sample_index"] = int(row["sample_index"])
            counts[row["condition_id"]] = counts.get(row["condition_id"], 0) + 1
        rows.extend(previous)
        print(f"resume: carried {len(previous)} rows from the existing manifest")

    rows.sort(key=lambda row: row["sample_index"])
    write_manifest(output, rows)
    write_conditions_csv(output, counts)
    write_readme(output, counts)

    realized = defaultdict(list)
    for row in rows:
        realized[row["condition_id"]].append(float(row["reachable_dump_to_dig_ratio"]))
    summary = {
        "schema": SCHEMA,
        "seed_base": SEED_BASE,
        "spec_path": "docs/CURRICULUM_TAXONOMY_SPEC.md",
        "spec_section": "8 + 8.1 + 8.2",
        # The §8.2 scope is the constant, not whichever --only batch ran last.
        "v32_regenerated_conditions": sorted(V32_REGENERATED),
        "conditions_built_this_run": regenerated,
        "v32_gates": {
            "wall_detour_min": WALL_DETOUR_MIN,
            "wall_detour_condition_median_min": WALL_DETOUR_CONDITION_MEDIAN_MIN,
            "wall_gap_offaxis_min_tiles": WALL_GAP_OFFAXIS_MIN,
            "wall_min_thickness_tiles": WALL_MIN_THICKNESS_TILES,
            "split_pad_edge_gap_min_tiles": SPLIT_PAD_EDGE_GAP_MIN,
            "split_pad_angular_span_min_deg": SPLIT_PAD_ANGULAR_SPAN_MIN,
            "road_sightline_cross_min": ROAD_SIGHTLINE_CROSS_MIN,
            "road_dump_annulus_sterilize_min": ROAD_DUMP_ANNULUS_STERILIZE_MIN,
            "road_dig_annulus_sterilize_range_ring": list(
                ROAD_DIG_ANNULUS_STERILIZE_RANGE
            ),
        },
        "taxonomy_version": tax.TAXONOMY_VERSION,
        "generator": "generate_prototypes_v7.py",
        "map_id_prefix": MAP_ID_PREFIX,
        "source_foundations": str(args.source_foundations),
        "condition_count": len(CONDITIONS),
        "maps_per_condition": counts,
        "accepted_maps": len(rows),
        "resume": bool(args.resume),
        "rerolled_dig_maps": sum(1 for row in rows if int(row["shared_dig"]) == 0),
        "object_bands": {k: list(v) for k, v in OBJECT_BANDS.items()},
        "apron_capacity_bands": {k: list(v) for k, v in APRON_CAPACITY_BANDS.items()},
        "generous_capacity_bands": {
            k: list(v) for k, v in GENEROUS_CAPACITY_BANDS.items()
        },
        "tight_capacity_band": list(TIGHT_CAPACITY_BAND),
        "ring_capacity_bands": {k: list(v) for k, v in RING_CAPACITY_BANDS.items()},
        "ring_min_capacity": RING_MIN_CAPACITY,
        "generous_reachable_floor": GENEROUS_REACHABLE_FLOOR,
        "realized_reachable_capacity": {
            key: [round(min(values), 3), round(max(values), 3)]
            for key, values in sorted(realized.items())
        },
        "shared_dig_attempts": SHARED_DIG_ATTEMPTS,
        "reroll_dump_attempts": REROLL_DUMP_ATTEMPTS,
        "dig_bank_rejections": dict(bank_rejections),
        "unsatisfied_constraints": unsatisfied,
        "rejections_before_acceptance": dict(rejection_totals),
        "static_gate_is_action_witness": False,
    }
    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "maps_per_condition"},
            indent=2,
            sort_keys=True,
        )
    )
    if unsatisfied:
        print("UNSATISFIED CONSTRAINTS:")
        for line in unsatisfied:
            print(f"  {line}")


if __name__ == "__main__":
    main()
