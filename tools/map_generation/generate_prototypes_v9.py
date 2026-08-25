#!/usr/bin/env python3
"""v5 curriculum review banks — spec §8.5 (U8-U11) and §8.6 (turn-dump contract).

Lineage: ``generate_prototypes_v8.py`` (the v4 bank). Everything §8/§8.1/§8.2/
§8.3/§8.4 asked for carries forward except where Lorenzo's second review pass
and the env-validated turn-dump panel supersede it.

U8   Transport is its own track. This script builds TWO datasets from one
     condition table: ``main`` (everything turn-dumpable from natural digging
     poses) and ``transport`` (walls, remote hauls, the d20/d24 bins). They get
     separate output roots, separate map-id namespaces and separate taxonomy
     releases; the transport bank publishes the 22.75-tile single-station budget
     as its defining quantity and gates multi-leg (dig -> stage -> re-dig)
     feasibility instead of single-station reach.
U9   Trench geometry rebalance: `tee` joins T0, and T1 runs up to four arms
     (`net3`, `net4`) with the U4 ordering stressor.
U10  Planning compositions are the composed tier: one-side or locally blocked
     dumping on a richer geometry, with a scripted start-side-sensitivity check
     that must actually bind (near-side-first vs far-side-first plans differ).
U11  Objects get sparser and graded: obj1 = 1-2, obj = 2-5, with non-overlapping
     blockage bands, a total-blockage cap and a cluster-size cap.

§8.6 (env-validated, 540 poses) replaces the v4 trench lane geometry:

* the service annulus is [6.375, 11.375] tiles and the 12 cabin indices cover
  it entirely, so turn-reachable == inside the annulus;
* the reserved lane's inner edge goes in [6.5, 10.0] tiles FROM THE SPINE and
  the map is rejected rather than the lane pushed out (v4: median 11.69, 66% of
  lanes entirely outside the boom);
* ``lane_usable_frac >= 0.5`` and ``turn_dump_cov_strict >= 0.95`` are gates,
  measured with the panel's own code (``turn_dump.py``);
* hugging banks stay, but down-line turn-dumpability is gated per station;
* ``standoff`` is dropped as a trench difficulty axis (meaningless inside the
  dead ring);
* ``tile_size_m`` is the live 0.5714 m, not the stale 0.6875.

Spec: ``terra-digging-benchmark-site/docs/CURRICULUM_TAXONOMY_SPEC.md``
§8, §8.1, §8.2, §8.3, §8.4, §8.5, §8.6.

Axis-contract v2 retains this construction as a private dependency while
emitting exact trench owner sidecars under new schema and map-ID namespaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

import generate_prototypes_v8 as v8
import terra_geom as tgeom
import terra_service as tsvc
import turn_dump as tdump

v7 = v8.v7
v6 = v8.v6
v5 = v8.v5
v3 = v8.v3
v2 = v8.v2
base = v8.base
tax = v8.tax

SCHEMA = "terra_curriculum_v5_axis_bank_v2"
SEED_BASE = 20260729
SHARED_DIG_ATTEMPTS = 120
REROLL_DUMP_ATTEMPTS = 20
MAX_ATTEMPTS = 320

MAP_SIZE = base.MAP_SIZE
MAP_CENTRE = (MAP_SIZE - 1) / 2.0
# §8.6 R7: the manifest's 0.6875 is stale metadata; the live scale is
# edge_length_m / 64. MAP_BENCHMARK_SPEC.md:219-220 makes the live scale
# authoritative, so v5 manifests carry it.
TILE_SIZE_M = tgeom.TILE_SIZE

ARRAY_FOLDERS = v6.ARRAY_FOLDERS
TRENCH_AXIS_OWNERS_FOLDER = "trench_axis_owners"
TRENCH_AXIS_CONTRACT = "generator_owner_bits_v1"
SITE_CLASS_TOKENS = v6.SITE_CLASS_TOKENS

CAPACITY_TOKENS = v8.CAPACITY_TOKENS
DISTANCE_TOKENS = v8.DISTANCE_TOKENS

GEOMETRY_LEVEL_SOURCE = {
    "slab": "foundation_osm",
    "slab-lg": "foundation_osm_large",
    "proc": "foundation_procedural",
    "strips": "foundation_structural",
    "straight": "trench_axes_1",
    "seg2": "trench_segments_2",
    "seg3": "trench_segments_3",
    "tee": "trench_axes_2",
    "net3": "trench_axes_3",
    "net4": "trench_axes_4",
}
GEOMETRY_LEVEL_INDEX = {
    level: index for index, level in enumerate(sorted(GEOMETRY_LEVEL_SOURCE))
}
GEOMETRY_HARDNESS = dict(v8.GEOMETRY_HARDNESS)
GEOMETRY_HARDNESS["trench_axes_4"] = "trench_network_4"

TRENCH_LEVELS = frozenset({"straight", "seg2", "seg3", "tee", "net3", "net4"})
MULTI_ARM_LEVELS = frozenset({"seg2", "seg3", "tee", "net3", "net4"})

# --------------------------------------------------------------------------
# carried forward from §8.3 (U1, U2, U5, U6)

RING_BAND_CAPACITY = v8.RING_BAND_CAPACITY
RING_BAND_DRAW = v8.RING_BAND_DRAW
DIRECT_SERVICE_MIN = v8.DIRECT_SERVICE_MIN
DIRECT_SERVICE_P95_MAX = v8.DIRECT_SERVICE_P95_MAX
APRON_STANDOFF_CHOICES = v8.APRON_STANDOFF_CHOICES
STANDOFF_CHOICES = v8.STANDOFF_CHOICES
APRON_CAPACITY_BANDS = v8.APRON_CAPACITY_BANDS
TIGHT_CAPACITY_BAND = v8.TIGHT_CAPACITY_BAND
GENEROUS_CAPACITY_BANDS = dict(v8.GENEROUS_CAPACITY_BANDS)
WALL_SPLIT_CAPACITY_BAND = v8.WALL_SPLIT_CAPACITY_BAND
TRENCH_WALL_CAPACITY_BAND = v8.WALL_SPLIT_CAPACITY_BAND
GENEROUS_REACHABLE_FLOOR = v8.GENEROUS_REACHABLE_FLOOR
DISTANCE_BINS = {"d12": 12.0, "d16": 16.0, "d20": 20.0, "d24": 24.0}
DISTANCE_BIN_TOLERANCE = v8.DISTANCE_BIN_TOLERANCE

# --------------------------------------------------------------------------
# §8.6 — the turn-dump geometry contract

R_MIN_TILES = tgeom.R_MIN_TILES          # 6.375
R_MAX_TILES = tgeom.R_MAX_TILES          # 11.375
SINGLE_STATION_BUDGET_TILES = 2.0 * R_MAX_TILES   # 22.75

TURN_DUMP_COV_MIN = 0.95                 # main track, every condition
# Down-line dumpability, per station: of the natural stations that can dig, what
# share has a designated dump cell inside the annulus? A continuous hugging bank
# must serve essentially all of them. Alternating banks are discontinuous BY
# DESIGN — that IS the condition — so a station opposite a gap between two banks
# legitimately has to move; its floor is lower and the realised value reported.
# The floor is 0.90, not 0.95: the realised value is a property of the dig + lane
# geometry and does not move at all when the dump is redrawn (measured: 20 dump
# draws on each of 5 `trn-straight-side1` slots give a single value to 3
# decimals), so a 0.95 floor is not a re-roll away — it is a dig rejection, and
# the slots it rejects sit at 0.926-0.947 with turn-dump coverage 0.967-1.000.
# 0.90 still says nine of every ten natural digging stations can turn-dump
# without moving, and v4 had no per-station gate at all.
STATION_DUMP_FRAC_MIN_BY_STYLE = {
    "trench_band": 0.90,
    "trench_flank": 0.90,
    "trench_altsides": 0.70,
}
STATION_DUMP_FRAC_MIN = 0.90
# ... and on a condition whose whole point is that dumping is LOCALLY BLOCKED
# (U10's planning compositions, and any trench carrying a road or objects), a
# station that has to move before it can dump is the difficulty, not a defect.
# Coverage still guarantees every dig cell is serviceable from SOME natural
# station; this floor only keeps the condition from degenerating.
STATION_DUMP_FRAC_MIN_BLOCKED_SITE = 0.70
LANE_INNER_BAND = (6.5, 10.0)            # tiles from the SPINE
LANE_INNER_CHOICES = (6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0)
LANE_USABLE_FRAC_MIN = 0.5
# The lane is picked on a cheap proxy for the usable strip; the margin over the
# gate absorbs the footprint / component tests the proxy does not do.
LANE_USABLE_PROXY_TARGET = 0.58
# A 15-tile lane whose inner edge sits at 9.5 has ~2 workable tiles (panel R2).
# v5 keeps the base-carrying widths but drops the widest ones: the outer half of
# a 15-tile lane is past the boom by construction and can never be usable.
LANE_WIDTH_CHOICES = (9, 11, 7, 13)
LANE_END_MARGIN = 5.0
# The lane must clear every arm, so branches all leave the spine on ONE side and
# the other side carries nothing but the spine itself.
TRENCH_LANE_SIDE_MAX = 3.5               # spine-lateral extent on the lane side
# v4 capped |lateral| at 10 in the CENTROID frame, i.e. a 20-tile total spread.
# The spine frame puts all of that on one side, so the same shapes need 20 here.
TRENCH_MAX_LATERAL_TILES = 20.0          # branch reach off the spine
# §8.6: standoff is dropped as a difficulty axis — inside the dead ring a
# 2-tile and a 6-tile standoff both mean "unreachable sideways". Trench banks
# hug the excavation and are gated on down-line dumpability instead.
TRENCH_BANK_STANDOFF = 1
BACKWARD_DRIFT_PER_TILE_MAX = v8.BACKWARD_DRIFT_PER_TILE_MAX
# U7(d) survives as the drift-rate + footprint-clear contract. The in-lane
# retreat-step count is REPORTED, and only gated where the lane is at least as
# wide as the 7-tile footprint plus a margin (see GENERATION_NOTES §5).
LANE_WIDTH_FOR_RETREAT_GATE = 11
BACKWARD_LANE_STEPS_MIN = 1

# --------------------------------------------------------------------------
# Lorenzo's v5-main review pass — apron proximity (deviation-fix)
#
# `turn_dump_cov_strict` is an EXISTENCE gate: it asks whether SOME natural
# station can dig a cell and turn-dump. It passed at >= 0.95 on the apron family
# while the typical working station was far from the spoil — measured on the
# first v5-main bank (scipy EDT, dig -> nearest designated dump per cell):
#
#   fnd-slab-apron-c3x   p95 per map 9.4 / 12.9 / 16.1 (min/med/max), worst 17.8
#   fnd-slab-apron-near  p95 median 12.1, max 13.8
#   fnd-slab-ring3x      p95 3.0-5.1                    <- what adjacency looks like
#
# Existence is not adjacency. The apron family therefore gains an explicit
# proximity gate on the distribution, not on a witness:
#
#   p95(dig -> nearest designated dump) <= r_max   AND   max <= 14.0
#
# The distance ladder is EXEMPT: `far` is its controlled factor. The ring
# conditions are gated too — they already comply, and the gate keeps them there.
DUMP_PROXIMITY_P95_MAX = R_MAX_TILES     # 11.375 — one boom reach, not two
DUMP_PROXIMITY_MAX_MAX = 14.0
PROXIMITY_GATED_STYLES = frozenset({"capacity_apron", "ring_band"})
# The p95 was driven by two things at once: a 200-260 deg wrap that left the far
# side of the slab unserved, and a standoff drawn as high as 8 on top of it. The
# wrap goes to 320 deg (a 40 deg gap is still a gap, and the dump layout token is
# still `near_apron_large`) and the standoff is clamped low. Capacity stays
# RADIAL-ONLY, so c1p2 -> c3x remains a pure capacity delta on a shared dig.
APRON_PROXIMITY_SECTOR_DEGREES = 320.0
APRON_PROXIMITY_STANDOFF_MAX = 5


def dump_proximity(dig: np.ndarray, dump: np.ndarray) -> dict[str, float]:
    """Distance from each dig cell to the nearest DESIGNATED dump cell."""
    values = ndi.distance_transform_edt(~dump)[dig]
    return {
        "dump_proximity_median_tiles": round(float(np.median(values)), 4),
        "dump_proximity_p95_tiles": round(float(np.quantile(values, 0.95)), 4),
        "dump_proximity_max_tiles": round(float(values.max()), 4),
    }


# --------------------------------------------------------------------------
# §8.5 U10 — planning compositions

PLAN_STEPS = 4
PLAN_DELTA_MIN = 0.10        # near-side-first must be materially worse
PLAN_FAR_COST_MAX = 0.45     # ... and the far-side-first plan must be workable

# --------------------------------------------------------------------------
# §8.5 U11 — objects sparser and gradual

OBJECT_BANDS = {
    "clean": (0, 0),
    "obj1": (1, 2),
    "obj": (2, 5),
    "road": (0, 0),
    "wall": (0, 0),
}
# Non-overlapping blockage bands: the two rungs must be genuinely graded, not
# just differently labelled. Measured on the <= 6-tile working annulus.
# Measured over 48 draws per geometry level with the v3.1 footprint profile
# (3-7 x 2-5 tiles): 1-2 objects block 0.000-0.045 of the <= 6-tile annulus
# (median 0.013), 2-5 objects block 0.000-0.066 (median 0.022). The two counts
# therefore OVERLAP on blockage, which is what §8.5 U11 forbids, so the rungs are
# separated on the measured quantity: obj1 takes the lower half, obj the upper.
OBJECT_ANNULUS_BLOCK_BAND = {"obj1": (0.004, 0.022), "obj": (0.030, 0.075)}
OBJECT_NEAR_ANNULUS_FREE_MIN = v8.OBJECT_NEAR_ANNULUS_FREE_MIN
# No merged obstacle fields: one connected obstacle may not exceed one object's
# largest legal footprint (7 x 5 rectangle = 35 cells) plus a rounding tile.
OBJECT_MAX_CLUSTER_CELLS = 36
OBJECT_TOTAL_BLOCK_MAX = 0.10
OBJECT_FOOTPRINT_PROFILE = "shared_v5"

# --------------------------------------------------------------------------
# carried forward: U4 overlap, U7 lattice, walls, roads, splits

SPOIL_FLANK_TILES = v8.SPOIL_FLANK_TILES
ARM_OVERLAP_MIN = v8.ARM_OVERLAP_MIN
ARM_MERGE_MAX = v8.ARM_MERGE_MAX
ARM_FILL_MAX = v8.ARM_FILL_MAX
ALTSIDES_MIN_SPAN = v8.ALTSIDES_MIN_SPAN
TRENCH_AXES_DEG = v8.TRENCH_AXES_DEG
DIG_IOU_MAX = v8.DIG_IOU_MAX
TRENCH_DIG_IOU_MAX = v8.TRENCH_DIG_IOU_MAX
T0_DIG_COVERAGE_MIN = v8.T0_DIG_COVERAGE_MIN
SPLIT_MAX_ANGULAR_GAP_MIN = v8.SPLIT_MAX_ANGULAR_GAP_MIN
SPLIT_PAD_BORDER_MARGIN = v8.SPLIT_PAD_BORDER_MARGIN
SPLIT_PAD_EDGE_GAP_MIN = v8.SPLIT_PAD_EDGE_GAP_MIN
SPLIT_PAD_ANGULAR_SPAN_MIN = v8.SPLIT_PAD_ANGULAR_SPAN_MIN
TRENCH_BRANCH_SPINE_MIN = v8.TRENCH_BRANCH_SPINE_MIN
NET_TURN_CYCLE = v8.NET_TURN_CYCLE
NET_TOPOLOGY_CYCLE = v8.NET_TOPOLOGY_CYCLE
# Branch turns must be NON-INCREASING along the spine or two branches point at
# each other and the comb closes into a ring (measured on the first net4 smoke:
# a 60 deg branch behind a 120 deg branch merged into a solid triangle with a
# hole). Leaning the rear branches back and the front branches forward keeps the
# teeth divergent, which is what makes them separately workable.
NET3_TURN_CYCLE = {
    "double_T": (90.0, 90.0),
    "skew_open": (120.0, 60.0),
    "skew_lead": (90.0, 60.0),
    "skew_trail": (120.0, 90.0),
    "skew_pair": (120.0, 60.0),
}
NET4_TURN_CYCLE = {
    "double_T": (90.0, 90.0, 90.0),
    "skew_open": (120.0, 90.0, 60.0),
    "skew_lead": (90.0, 90.0, 60.0),
    "skew_trail": (120.0, 90.0, 90.0),
    "skew_pair": (120.0, 120.0, 60.0),
}
REMOTE_MIN_DISTANCE_TILES = v8.REMOTE_MIN_DISTANCE_TILES
WALL_DETOUR_MIN = v8.WALL_DETOUR_MIN
WALL_DETOUR_CONDITION_MEDIAN_MIN = v8.WALL_DETOUR_CONDITION_MEDIAN_MIN
WALL_GAP_OFFAXIS_MIN = v8.WALL_GAP_OFFAXIS_MIN
WALL_MIN_THICKNESS_TILES = v8.WALL_MIN_THICKNESS_TILES
WALL_MIN_COMPONENTS = v8.WALL_MIN_COMPONENTS
WALL_MIN_COMPONENT_CELLS = v8.WALL_MIN_COMPONENT_CELLS
WALL_MIN_COMPONENT_EXTENT = v8.WALL_MIN_COMPONENT_EXTENT
ROAD_ANNULUS_FREE_MIN = v8.ROAD_ANNULUS_FREE_MIN
ROAD_DIG_ANNULUS_STERILIZE_RANGE = v8.ROAD_DIG_ANNULUS_STERILIZE_RANGE
ROAD_RADIUS_RANGE_RING = v8.ROAD_RADIUS_RANGE_RING
ROAD_RADIUS_RANGE_ZONED = v8.ROAD_RADIUS_RANGE_ZONED
ROAD_BAND_STERILIZE_MIN = v8.ROAD_BAND_STERILIZE_MIN

FOUNDATION_MEDIAN_LIMITS = v8.FOUNDATION_MEDIAN_LIMITS
TRENCH_MEDIAN_LIMITS = {
    "trench_band": 8.0,
    "trench_flank": 10.0,
    "trench_altsides": 12.0,
}
WALL_SPLIT_MEDIAN_LIMIT = v8.WALL_SPLIT_MEDIAN_LIMIT
TRENCH_WALL_MEDIAN_LIMIT = 26.0

PLACEMENT_RADII = v8.PLACEMENT_RADII
TRENCH_PLACEMENT_RADII = v8.TRENCH_PLACEMENT_RADII
TRENCH_PLACEMENT_RADII_WIDE = v8.TRENCH_PLACEMENT_RADII_WIDE
WIDE_TRENCH_LEVELS = frozenset({"net3", "net4", "tee"})
FOUNDATION_PLACEMENT_MARGIN = v8.FOUNDATION_PLACEMENT_MARGIN
TRENCH_PLACEMENT_MARGIN = v8.TRENCH_PLACEMENT_MARGIN
APRON_SECTOR_DEGREES = v8.APRON_SECTOR_DEGREES
APRON_WRAP_SECTOR_DEGREES = v8.APRON_WRAP_SECTOR_DEGREES

# §8.5 U8: the transport track may waive the lane gate (a wall is precisely what
# removes a straight corridor) and is measured on staging, not single-station.
STAGING_MAX_HOPS = 4

# reused helpers
stable_key = v8.stable_key
sha256_mask = v8.sha256_mask
rng_from = v8.rng_from
border_margin = v8.border_margin
centroid = v8.centroid
centroid_offset = v8.centroid_offset
place_at = v8.place_at
centred_iou = v8.centred_iou
dig_only_coverage = v8.dig_only_coverage
max_angular_gap_degrees = v8.max_angular_gap_degrees
haul_detour = v8.haul_detour
pad_separation = v8.pad_separation
object_intrusion = v8.object_intrusion
place_objects_v7 = v8.place_objects_v7
make_two_border_road = v8.make_two_border_road
zoned_road_metrics = v8.zoned_road_metrics
zoned_road_bites = v8.zoned_road_bites
make_offset_fence_wall = v8.make_offset_fence_wall
sight_line_bite = v8.sight_line_bite
wall_min_thickness = v8.wall_min_thickness
wall_component_stats = v8.wall_component_stats
gap_offaxis_tiles = v8.gap_offaxis_tiles
_interior = v8._interior
_polar = v8._polar
rasterize_segments = v8.rasterize_segments
edge_irregularity = v8.edge_irregularity
regularise_edges = v8.regularise_edges
arm_masks = v8.arm_masks
arm_overlap_fraction = v8.arm_overlap_fraction
arm_merge_fraction = v8.arm_merge_fraction
ring_band = v8.ring_band
apron_sector = v8.apron_sector
distance_apron = v8.distance_apron
_as_arms = v8._as_arms


def _distance_to_polyline(points_yx: np.ndarray, cells_yx: np.ndarray) -> np.ndarray:
    """Return each cell-centre distance to one generated trench arm."""

    if points_yx.ndim != 2 or points_yx.shape[0] < 2 or points_yx.shape[1] != 2:
        raise RuntimeError(f"Invalid trench arm shape: {points_yx.shape}.")
    best = np.full(cells_yx.shape[0], np.inf, dtype=np.float64)
    for start, end in zip(points_yx[:-1], points_yx[1:]):
        delta = end - start
        denominator = float(delta @ delta)
        if denominator <= 1e-12:
            continue
        fraction = np.clip(((cells_yx - start) @ delta) / denominator, 0.0, 1.0)
        nearest = start + fraction[:, None] * delta
        best = np.minimum(best, np.linalg.norm(cells_yx - nearest, axis=1))
    if not np.all(np.isfinite(best)):
        raise RuntimeError("Trench arm has no non-degenerate segment.")
    return best


def trench_axis_owners(target: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Emit exact axis-owner bits for every trench target cell.

    Arm rasters define junction ownership. Edge regularisation can create a few
    cells outside those raw rasters; those cells are assigned to the nearest
    finite generated arm, with exact ties retaining multiple owners. This work
    happens once in the generator. Terra consumes the resulting uint8 map and
    never reconstructs ownership from geometry.
    """

    target = np.asarray(target)
    owners = np.zeros(target.shape, dtype=np.uint8)
    axes = list(metadata.get("axes_ABC", []) or [])
    raw_arms = metadata.get("trench_arms", []) or []
    arms = _as_arms(raw_arms) if raw_arms else []
    if not axes and not arms:
        return owners
    if len(axes) != len(arms):
        raise RuntimeError(
            f"Trench axes/arms disagree: {len(axes)} axes, {len(arms)} arms."
        )
    if not 0 < len(axes) <= 8:
        raise RuntimeError(f"uint8 owner maps support 1..8 axes, got {len(axes)}.")

    dig = target < 0
    half_width = float(metadata["trench_half_width_tiles"])
    arm_points = [np.asarray(arm, dtype=np.float64) for arm in arms]
    for axis_index, points in enumerate(arm_points):
        mask = rasterize_segments(points, half_width) & dig
        owners[mask] |= np.uint8(1 << axis_index)

    missing = dig & (owners == 0)
    if np.any(missing):
        cells = np.argwhere(missing).astype(np.float64)
        distances = np.stack(
            [_distance_to_polyline(points, cells) for points in arm_points]
        )
        nearest = np.min(distances, axis=0)
        for axis_index, axis_distances in enumerate(distances):
            selected = axis_distances <= nearest + 1e-9
            rows, columns = cells[selected].astype(np.int32).T
            owners[rows, columns] |= np.uint8(1 << axis_index)

    if np.any(dig & (owners == 0)) or np.any((~dig) & (owners != 0)):
        raise RuntimeError(
            "Generated trench owner map does not partition the dig target."
        )
    return owners


# --------------------------------------------------------------------------
# condition table


@dataclass(frozen=True)
class ConditionSpec:
    id: str
    anchor: str | None
    dataset: str
    family: str
    geometry_level: str
    dump_level: str
    capacity_level: str
    site_level: str
    distance_level: str
    dump_style: str
    dump_layout: str
    pair_group: str | None = None
    layout_group: str | None = None
    planning: bool = False

    @property
    def geometry(self) -> str:
        return GEOMETRY_LEVEL_SOURCE[self.geometry_level]

    @property
    def release(self):
        return tax.RELEASES[f"v5-{self.dataset}"]

    @property
    def levels(self) -> dict[str, str]:
        return {
            "family": "fnd" if self.family == "foundation" else "trn",
            "geometry": self.geometry_level,
            "dump": self.dump_level,
            "capacity": self.capacity_level,
            "site": self.site_level,
            "distance": self.distance_level,
        }

    @property
    def tier(self) -> int:
        return tax.tier(self.levels, self.release)

    @property
    def preview(self) -> bool:
        return False

    @property
    def group(self) -> str:
        return self.layout_group or self.id


def _fnd(condition_id, anchor, dataset, **kwargs) -> ConditionSpec:
    return ConditionSpec(condition_id, anchor, dataset, family="foundation", **kwargs)


def _trn(condition_id, anchor, dataset, **kwargs) -> ConditionSpec:
    return ConditionSpec(condition_id, anchor, dataset, family="trench", **kwargs)


APRON_CAPACITY_GROUP = "slab-apron-capacity"
APRON_DISTANCE_GROUP = "slab-apron-distance"
TRENCH_ONE_SIDE_GROUP = "trn-one-side"

MAIN_CONDITIONS: tuple[ConditionSpec, ...] = (
    # ---- foundation T0: capped rings (U1) + direct-service apron (U2) ------
    _fnd("fnd-slab-ring3x", None, "main", geometry_level="slab",
         dump_level="ring3x", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-proc-ring3x", None, "main", geometry_level="proc",
         dump_level="ring3x", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-lg-ring3x", None, "main", geometry_level="slab-lg",
         dump_level="ring3x", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-apron-near", None, "main", geometry_level="slab",
         dump_level="apron", capacity_level="generous", site_level="clean",
         distance_level="near", dump_style="capacity_apron",
         dump_layout="near_apron_large",
         pair_group=APRON_DISTANCE_GROUP, layout_group=APRON_DISTANCE_GROUP),
    _fnd("fnd-slab-apron-c3x", None, "main", geometry_level="slab",
         dump_level="apron", capacity_level="c3x", site_level="clean",
         distance_level="unspec", dump_style="capacity_apron",
         dump_layout="near_apron_large",
         pair_group=APRON_CAPACITY_GROUP, layout_group=APRON_CAPACITY_GROUP),
    # ---- U5 capacity ladder (shared digs, one azimuth per index) -----------
    *[
        _fnd(f"fnd-slab-apron-{level}", "fnd-slab-apron-c3x", "main",
             geometry_level="slab", dump_level="apron", capacity_level=level,
             site_level="clean", distance_level="unspec",
             dump_style="capacity_apron", dump_layout="near_apron_large",
             pair_group=APRON_CAPACITY_GROUP, layout_group=APRON_CAPACITY_GROUP)
        for level in ("c2x", "c1p6", "c1p2")
    ],
    # ---- U6 distance ladder, main keeps only the turn-dumpable bins --------
    *[
        _fnd(f"fnd-slab-apron-{bin_name}", "fnd-slab-apron-near", "main",
             geometry_level="slab", dump_level="apron",
             capacity_level="generous", site_level="clean",
             distance_level=bin_name, dump_style="distance_apron",
             dump_layout="near_apron_large",
             pair_group=APRON_DISTANCE_GROUP, layout_group=APRON_DISTANCE_GROUP)
        for bin_name in ("d12", "d16")
    ],
    # ---- T1 layout / geometry / site off the capped ring -------------------
    _fnd("fnd-slab-side1", "fnd-slab-ring3x", "main", geometry_level="slab",
         dump_level="side1", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="one_side_near",
         dump_layout="one_side_near", layout_group="fnd-slab-side1"),
    _fnd("fnd-slab-split", "fnd-slab-ring3x", "main", geometry_level="slab",
         dump_level="split", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="separated_zones",
         dump_layout="separated_zones"),
    _fnd("fnd-strips-ring3x", "fnd-slab-ring3x", "main", geometry_level="strips",
         dump_level="ring3x", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-ring3x-obj1", "fnd-slab-ring3x", "main", geometry_level="slab",
         dump_level="ring3x", capacity_level="generous", site_level="obj1",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-ring3x-obj", "fnd-slab-ring3x", "main", geometry_level="slab",
         dump_level="ring3x", capacity_level="generous", site_level="obj",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-ring3x-road", "fnd-slab-ring3x", "main", geometry_level="slab",
         dump_level="ring3x", capacity_level="generous", site_level="road",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    # ---- T2 planning compositions (U10) ------------------------------------
    _fnd("fnd-proc-side1-road", "fnd-proc-ring3x", "main", geometry_level="proc",
         dump_level="side1", capacity_level="generous", site_level="road",
         distance_level="unspec", dump_style="one_side_near",
         dump_layout="one_side_near", planning=True),
    _fnd("fnd-slab-side1-obj", "fnd-slab-ring3x", "main", geometry_level="slab",
         dump_level="side1", capacity_level="generous", site_level="obj",
         distance_level="unspec", dump_style="one_side_near",
         dump_layout="one_side_near", layout_group="fnd-slab-side1",
         planning=True),
    # ---- trench T0 (U9: tee joins the baseline) ----------------------------
    _trn("trn-straight-side2", None, "main", geometry_level="straight",
         dump_level="side2", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="trench_band",
         dump_layout="easy_surround"),
    _trn("trn-straight-side1", None, "main", geometry_level="straight",
         dump_level="side1", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="trench_flank",
         dump_layout="near_apron_large",
         pair_group=TRENCH_ONE_SIDE_GROUP, layout_group=TRENCH_ONE_SIDE_GROUP),
    _trn("trn-tee-side2", None, "main", geometry_level="tee",
         dump_level="side2", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="trench_band",
         dump_layout="easy_surround"),
    # ---- trench T1 ---------------------------------------------------------
    _trn("trn-straight-side1-tight", "trn-straight-side1", "main",
         geometry_level="straight", dump_level="side1", capacity_level="tight",
         site_level="clean", distance_level="unspec", dump_style="trench_flank",
         dump_layout="near_apron_large",
         pair_group=TRENCH_ONE_SIDE_GROUP, layout_group=TRENCH_ONE_SIDE_GROUP),
    _trn("trn-straight-altsides", "trn-straight-side2", "main",
         geometry_level="straight", dump_level="altsides",
         capacity_level="generous", site_level="clean", distance_level="unspec",
         dump_style="trench_altsides", dump_layout="alternating_sides"),
    *[
        _trn(f"trn-{level}-side2", "trn-straight-side2", "main",
             geometry_level=level, dump_level="side2",
             capacity_level="generous", site_level="clean",
             distance_level="unspec", dump_style="trench_band",
             dump_layout="easy_surround")
        for level in ("seg2", "seg3", "net3", "net4")
    ],
    # ---- trench T2 planning compositions (U10) -----------------------------
    # Both trench planning rungs use a road as the local blocker. `obj` was tried
    # first (`trn-seg3-side1-obj`) and does not fit: the reserved lane already
    # sterilises the flank an object would have to intrude into, so 2-5 objects
    # reach only 0.000-0.056 of the <= 6-tile annulus (median 0.019) against the
    # 0.030-0.075 rung U11 defines, and the condition exhausts. See
    # GENERATION_NOTES "deviations".
    _trn("trn-net3-side1-road", "trn-straight-side1", "main",
         geometry_level="net3", dump_level="side1", capacity_level="generous",
         site_level="road", distance_level="unspec", dump_style="trench_flank",
         dump_layout="near_apron_large", planning=True),
    _trn("trn-net4-side1-road", "trn-straight-side1", "main",
         geometry_level="net4", dump_level="side1", capacity_level="generous",
         site_level="road", distance_level="unspec", dump_style="trench_flank",
         dump_layout="near_apron_large", planning=True),
)

TRANSPORT_CONDITIONS: tuple[ConditionSpec, ...] = (
    *[
        _fnd(f"fnd-slab-apron-{bin_name}", "fnd-slab-apron-near", "transport",
             geometry_level="slab", dump_level="apron",
             capacity_level="generous", site_level="clean",
             distance_level=bin_name, dump_style="distance_apron",
             dump_layout="near_apron_large",
             pair_group=APRON_DISTANCE_GROUP, layout_group=APRON_DISTANCE_GROUP)
        for bin_name in ("d20", "d24")
    ],
    _fnd("fnd-slab-remote", "fnd-slab-ring3x", "transport", geometry_level="slab",
         dump_level="remote", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="haul_away_edge",
         dump_layout="haul_away_edge"),
    _fnd("fnd-strips-split-wall", "fnd-slab-ring3x", "transport",
         geometry_level="strips", dump_level="split", capacity_level="generous",
         site_level="wall", distance_level="unspec",
         dump_style="separated_zones", dump_layout="separated_zones"),
    _trn("trn-straight-remote", "trn-straight-side2", "transport",
         geometry_level="straight", dump_level="remote",
         capacity_level="generous", site_level="clean", distance_level="unspec",
         dump_style="haul_away_edge", dump_layout="haul_away_edge"),
    _trn("trn-net3-split-wall", "trn-straight-side2", "transport",
         geometry_level="net3", dump_level="split", capacity_level="generous",
         site_level="wall", distance_level="unspec",
         dump_style="separated_zones", dump_layout="separated_zones"),
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    conditions: tuple[ConditionSpec, ...]
    maps_per_condition: int
    map_id_prefix: str
    release: str
    # §8.6 gates apply to the main track; the transport track is measured on
    # staging feasibility instead (U8) and its numbers are reported, not gated.
    gate_turn_dump: bool
    gate_lane_band: bool
    gate_staging: bool


DATASETS = {
    "main": DatasetSpec(
        name="main",
        conditions=MAIN_CONDITIONS,
        maps_per_condition=16,
        map_id_prefix="curriculum-v5m-axis-v2",
        release="v5-main",
        gate_turn_dump=True,
        gate_lane_band=True,
        gate_staging=False,
    ),
    "transport": DatasetSpec(
        name="transport",
        conditions=TRANSPORT_CONDITIONS,
        maps_per_condition=8,
        map_id_prefix="curriculum-v5t-axis-v2",
        release="v5-transport",
        gate_turn_dump=False,
        gate_lane_band=False,
        gate_staging=True,
    ),
}

WALL_CONDITIONS = frozenset({"fnd-strips-split-wall", "trn-net3-split-wall"})
PLANNING_CONDITIONS = frozenset(
    c.id for c in MAIN_CONDITIONS + TRANSPORT_CONDITIONS if c.planning
)


def assert_conditions_match_taxonomy(dataset: DatasetSpec) -> None:
    """The generator table and `scripts/curriculum_taxonomy.py` must agree."""
    spec = tax.RELEASES[dataset.release]
    table = spec.condition_table
    conditions = dataset.conditions
    assert {c.id for c in conditions} == set(table), (
        f"{dataset.name}: condition set mismatch: "
        f"{sorted({c.id for c in conditions} ^ set(table))}"
    )
    for condition in conditions:
        derived = tax.condition_id(condition.levels, spec)
        assert derived == condition.id, f"{condition.id}: id grammar gives {derived}"
        expected_tier, expected_anchor = table[condition.id]
        assert condition.tier == expected_tier, (
            f"{condition.id}: tier {condition.tier} != table {expected_tier}"
        )
        assert condition.anchor == expected_anchor, (
            f"{condition.id}: anchor {condition.anchor} != table {expected_anchor}"
        )
        assert spec.dump_levels[condition.family][condition.dump_layout] == (
            condition.dump_level
        )
        assert spec.site_levels[SITE_CLASS_TOKENS[condition.site_level]] == (
            condition.site_level
        )
        token = CAPACITY_TOKENS[condition.capacity_level]
        if token:
            assert spec.capacity_levels[token] == condition.capacity_level
        assert spec.distance_levels[DISTANCE_TOKENS[condition.distance_level]] == (
            condition.distance_level
        )
        assert spec.geometry_levels[condition.geometry] == condition.geometry_level
    tax.build_conditions(
        [
            {
                "cellId": c.id,
                "family": c.family,
                "id": c.id,
                "factors": {
                    "geometryClass": c.geometry,
                    "dumpLayout": c.dump_layout,
                    "siteClass": SITE_CLASS_TOKENS[c.site_level],
                    "capacityBand": CAPACITY_TOKENS[c.capacity_level] or None,
                    "distanceBand": DISTANCE_TOKENS[c.distance_level] or None,
                },
            }
            for c in conditions
        ],
        spec,
    )
    # U8: the two datasets must not collide, on ids or on map-id namespaces.
    other = DATASETS["transport" if dataset.name == "main" else "main"]
    assert not ({c.id for c in conditions} & {c.id for c in other.conditions})
    assert dataset.map_id_prefix != other.map_id_prefix


# --------------------------------------------------------------------------
# §8.6 — the spine frame
#
# v4 measured the lane offset from the dig CENTROID, which on a comb sits
# between the spine and the branch tips: an 8-tile "offset" was then 3 tiles
# from the spine on one side and 13 on the other. §8.6 states the band in
# spine coordinates, so v5 measures it there. The centroid-frame number is kept
# in the manifest for panel comparability.


def spine_frame(dig_meta: dict[str, Any]):
    arms = _as_arms(dig_meta["trench_arms"])
    p0 = np.asarray(arms[0][0], dtype=float)
    p1 = np.asarray(arms[0][-1], dtype=float)
    heading = float(dig_meta["trench_global_angle_deg"])
    direction = np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
    normal = np.array([direction[1], -direction[0]])
    origin = (p0 + p1) / 2.0
    return origin, direction, normal


def spine_fields(shape, dig_meta: dict[str, Any]):
    origin, direction, normal = spine_frame(dig_meta)
    yy, xx = np.indices(shape)
    axial = (yy - origin[0]) * direction[0] + (xx - origin[1]) * direction[1]
    lateral = (yy - origin[0]) * normal[0] + (xx - origin[1]) * normal[1]
    return origin, direction, normal, axial, lateral


# --------------------------------------------------------------------------
# geometry factory


class GeometryFactoryV9(v8.GeometryFactoryV8):
    """v8 geometries, re-gated in the spine frame, plus net3 / net4."""

    @staticmethod
    def _finish_trench(dig, segments, half_width, heading_deg, radius, angle,
                       cells, extra):
        dig = regularise_edges(dig)
        if not dig.any() or not cells[0] <= int(dig.sum()) <= cells[1]:
            return None, {}
        spurs, notches = edge_irregularity(dig)
        if spurs or notches:
            return None, {}
        placed = place_at(dig, radius, angle, TRENCH_PLACEMENT_MARGIN)
        if placed is None:
            return None, {}
        offset = np.array(centroid(placed)) - np.array(centroid(dig))
        arms = [
            [[round(float(p[0] + offset[0]), 3), round(float(p[1] + offset[1]), 3)]
             for p in segment]
            for segment in segments
        ]
        meta = {
            "trench_global_angle_deg": heading_deg,
            "trench_half_width_tiles": half_width,
            "trench_width_radius_tiles": int(round(half_width)),
            "trench_arms": arms,
            "trench_arm_count": len(arms),
            "axes_ABC": [
                v3.line_coefficients(
                    np.array(arm[0], dtype=float), np.array(arm[-1], dtype=float)
                )
                for arm in arms
            ],
            **extra,
        }
        # §8.6: the lane goes in [6.5, 10.0] tiles from the SPINE, so one side of
        # the spine must carry nothing but the spine itself. Every branch leaves
        # on the other side by construction; this asserts it on the raster.
        _, _, _, _, lateral = spine_fields(placed.shape, meta)
        extents = {sign: float((lateral[placed] * sign).max()) for sign in (-1.0, 1.0)}
        lane_side = min(extents, key=lambda s: extents[s])
        if extents[lane_side] > TRENCH_LANE_SIDE_MAX:
            return None, {}
        if extents[-lane_side] > TRENCH_MAX_LATERAL_TILES:
            return None, {}
        meta["trench_lane_side"] = int(lane_side)
        meta["trench_lane_side_extent_tiles"] = round(extents[lane_side], 3)
        meta["trench_branch_side_extent_tiles"] = round(extents[-lane_side], 3)
        return placed, meta

    def _net(self, rng, radius, angle, heading_deg, topology, n_branches):
        """A comb: a spine with `n_branches` junctions, all on one side.

        U9 extends the v4 two-branch comb to three. The branch angles carry the
        junction variety (the SIDE cannot: §8.6 needs a clear spine flank for the
        lane), so `double_T` stays a minority of the level.
        """
        heading = math.radians(heading_deg)
        direction = np.array([math.sin(heading), math.cos(heading)])
        turns_table = NET3_TURN_CYCLE if n_branches == 2 else NET4_TURN_CYCLE
        spine_band = (20.0, 27.0) if n_branches == 2 else (26.0, 34.0)
        cell_band = (100, 205) if n_branches == 2 else (120, 250)
        positions_base = (
            np.array([-0.26, 0.26]) if n_branches == 2 else np.array([-0.32, 0.0, 0.32])
        )
        for _ in range(160):
            spine = float(rng.uniform(*spine_band))
            main = np.vstack([-direction * spine / 2, direction * spine / 2]) + MAP_CENTRE
            positions = positions_base + rng.uniform(-0.03, 0.03, size=n_branches)
            half_width = float(rng.choice([1.0, 2.0], p=[0.7, 0.3]))
            dig = rasterize_segments(main, half_width)
            segments = [main]
            ratios = []
            turns = turns_table[topology]
            for branch_index, along in enumerate(positions):
                junction = direction * float(along * spine) + MAP_CENTRE
                turn = float(turns[branch_index])
                branch_heading = (heading_deg + turn) % 360.0
                branch_direction = np.array(
                    [
                        math.sin(math.radians(branch_heading)),
                        math.cos(math.radians(branch_heading)),
                    ]
                )
                length = (
                    float(rng.uniform(TRENCH_BRANCH_SPINE_MIN + 0.02, 0.42)) * spine
                )
                points = np.vstack([junction, junction + branch_direction * length])
                ratios.append(length / spine)
                dig |= rasterize_segments(points, half_width)
                segments.append(points)
            placed, meta = self._finish_trench(
                dig, segments, half_width, heading_deg, radius, angle, cell_band,
                {
                    "trench_axes_count": n_branches + 1,
                    "intersection_junctions": n_branches,
                    "intersection_double_sided": 0,
                    "intersection_branches": n_branches + 2,
                    "trench_topology": topology,
                    "trench_branch_turns_deg": [float(t) for t in turns],
                    "trench_relative_angle_deg": float(min(turns)),
                    "trench_spine_length_tiles": round(spine, 2),
                    "trench_branch_spine_ratio": round(float(min(ratios)), 4),
                },
            )
            if placed is None:
                continue
            if arm_overlap_fraction(placed, meta)[0] < ARM_OVERLAP_MIN:
                continue
            # The comb must still READ as separate teeth: notch-filling between
            # two branches that lean toward each other silently turns the net
            # into a solid wedge with a hole, and no per-arm statistic notices.
            if arm_merge_fraction(placed, meta) > ARM_MERGE_MAX:
                continue
            union = np.zeros_like(placed)
            for arm in arm_masks(placed, meta):
                union |= arm
            if int(placed.sum()) > ARM_FILL_MAX * max(1, int(union.sum())):
                continue
            return placed, meta
        return None, {}

    def net3(self, rng, radius, angle, heading_deg, topology):
        return self._net(rng, radius, angle, heading_deg, topology, 2)

    def net4(self, rng, radius, angle, heading_deg, topology):
        return self._net(rng, radius, angle, heading_deg, topology, 3)


GEOMETRY_BUILDER = {
    "slab": "slab",
    "slab-lg": "slab_lg",
    "proc": "proc",
    "strips": "strips",
    "straight": "straight",
    "seg2": "seg2",
    "seg3": "seg3",
    "tee": "tee",
    "net3": "net3",
    "net4": "net4",
}


# --------------------------------------------------------------------------
# dig bank


class DigBankV9(v8.DigBankV8):
    def __init__(self, factory, n_maps, t0_levels: frozenset[str]) -> None:
        super().__init__(factory, n_maps)
        self.t0_levels = t0_levels

    def _headings(self, level: str) -> list[float]:
        rng = rng_from(SEED_BASE, 4242, GEOMETRY_LEVEL_INDEX[level])
        order = [TRENCH_AXES_DEG[i] for i in rng.permutation(len(TRENCH_AXES_DEG))]
        return [order[k % len(order)] for k in range(self.n_maps)]

    def _sample(self, level, map_index, salt, attempt):
        seed = int(
            np.random.SeedSequence(
                [SEED_BASE, 7777, GEOMETRY_LEVEL_INDEX[level], map_index, salt, attempt]
            ).generate_state(1)[0]
        )
        rng = np.random.default_rng(seed)
        placement_rng = rng_from(
            SEED_BASE, 3131, GEOMETRY_LEVEL_INDEX[level], map_index, salt
        )
        if level in WIDE_TRENCH_LEVELS:
            schedule = TRENCH_PLACEMENT_RADII_WIDE
        elif level in TRENCH_LEVELS:
            schedule = TRENCH_PLACEMENT_RADII
        else:
            schedule = PLACEMENT_RADII
        radius = schedule[
            (map_index + attempt // 12) % len(schedule)
        ] + float(placement_rng.uniform(-1.0, 1.0))
        angle = float(placement_rng.uniform(-math.pi, math.pi))
        builder = getattr(self.factory, GEOMETRY_BUILDER[level])
        if level in TRENCH_LEVELS:
            headings = self._headings(level)
            heading = headings[(map_index + salt) % len(headings)]
            topology = NET_TOPOLOGY_CYCLE[(map_index + salt) % len(NET_TOPOLOGY_CYCLE)]
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
        if level in MULTI_ARM_LEVELS:
            worst, per_arm = arm_overlap_fraction(dig, meta)
            meta["arm_overlap_fraction"] = round(worst, 4)
            meta["arm_overlap_per_arm"] = per_arm
        return dig, meta

    def _acceptable(self, level, dig, meta, against):
        if level in self.t0_levels:
            coverage = dig_only_coverage(dig)
            if coverage < T0_DIG_COVERAGE_MIN:
                return "dig_bank_t0_coverage"
            meta["dig_only_workspace_coverage"] = round(coverage, 5)
        source = meta.get("foundation_source_index")
        for other, other_meta in against:
            if source is not None and other_meta.get("foundation_source_index") == source:
                return "dig_bank_source_reuse"
            limit = TRENCH_DIG_IOU_MAX if level in TRENCH_LEVELS else DIG_IOU_MAX
            if centred_iou(dig, other) >= limit:
                return "dig_bank_iou"
        return ""


# --------------------------------------------------------------------------
# layout


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
    lane_sign: int
    lane_inner: float
    alternations: int

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
            "layout_lane_sign": self.lane_sign,
            "layout_lane_inner_tiles": self.lane_inner,
            "layout_alternations": self.alternations,
        }


def layout_for(condition: ConditionSpec, map_index: int) -> Layout:
    group = condition.group
    rng = rng_from(SEED_BASE, 5150, stable_key(group), map_index)
    wall = condition.id in WALL_CONDITIONS
    if condition.family == "trench":
        # §8.6: standoff is not a trench difficulty axis any more.
        standoff = TRENCH_BANK_STANDOFF
    elif condition.dump_style in ("capacity_apron", "distance_apron"):
        standoff = int(rng.choice(APRON_STANDOFF_CHOICES))
    else:
        standoff = int(rng.choice(STANDOFF_CHOICES))
    if condition.dump_style == "capacity_apron":
        # The DRAW is deliberately left untouched and the VALUE clamped: the same
        # rng stream then produces the same azimuth, side sign and zone span, so
        # `fnd-slab-apron-near` keeps the pair-seeded azimuth it shares with the
        # frozen d12 / d16 rungs, and the four capacity siblings keep sharing
        # theirs. Re-drawing here would silently re-key the whole layout group.
        standoff = min(standoff, APRON_PROXIMITY_STANDOFF_MAX)
    if wall:
        n_zones = 2
        span = float(rng.uniform(92.0, 124.0))
    else:
        n_zones = int(rng.choice([2, 2, 2, 3]))
        span = float(rng.uniform(90.0, 130.0) if n_zones == 2 else rng.uniform(110.0, 130.0))
    if condition.dump_style == "capacity_apron":
        sector = APRON_PROXIMITY_SECTOR_DEGREES
    elif group == APRON_DISTANCE_GROUP:
        # the distance ladder is exempt from the proximity gate, so its wrap is
        # untouched and d12 / d16 stay byte-identical
        sector = APRON_WRAP_SECTOR_DEGREES
    else:
        sector = APRON_SECTOR_DEGREES
    return Layout(
        group=group,
        azimuth=float(rng.uniform(-math.pi, math.pi)),
        sector_degrees=sector,
        standoff=standoff,
        side_sign=int(rng.choice([-1, 1])),
        side_index=int(rng.integers(0, 4)),
        n_zones=n_zones,
        zone_span_degrees=span,
        lane_sign=int(rng.choice([-1, 1])),
        # §8.6 anti-overfit: the lane offset is randomised INSIDE the compliant
        # band, per map, instead of being a constant.
        lane_inner=float(rng.choice(LANE_INNER_CHOICES)),
        alternations=int(rng.choice([2, 3, 3, 4])),
    )


# --------------------------------------------------------------------------
# §8.6 — the working lane, in the boom band


def build_lane(dig, dig_meta, layout: Layout):
    """A straight base-width corridor whose inner edge is INSIDE the boom.

    §8.6 replaces v4's ``inner = global lateral extent + gap`` (which put 66% of
    lanes entirely outside the 11.375-tile envelope) with a hard band measured
    from the spine: ``LANE_INNER_BAND``. If no compliant lane exists on this dig
    the map is REJECTED — the lane's position is a machine constraint, the spoil
    band's is not (panel R3).
    """
    _, direction, normal, axial, lateral = spine_fields(dig.shape, dig_meta)
    origin, _, _ = spine_frame(dig_meta)
    dig_axial = axial[dig]
    extent = float(dig_axial.max() - dig_axial.min())
    axial_centre = float((dig_axial.max() + dig_axial.min()) / 2.0)
    heading_deg = float(dig_meta["trench_global_angle_deg"])

    extents = {sign: float((lateral[dig] * sign).max()) for sign in (-1.0, 1.0)}
    signs = sorted(
        (-1.0, 1.0),
        key=lambda sign: (round(extents[sign], 3), -sign * float(layout.lane_sign)),
    )
    # The usable-strip proxy: a lane cell can dig iff at least one excavation
    # cell falls inside its annulus. That is the whole of `lane_usable_frac`
    # minus the footprint/component tests, and it costs one correlation — so the
    # lane is CHOSEN on it instead of being gated on it after the fact.
    diggable = (
        ndi.correlate(
            dig.astype(np.int32), tgeom.ANNULUS.astype(np.int32),
            mode="constant", cval=0,
        )
        > 0
    )
    distance = ndi.distance_transform_edt(~dig)

    candidates: list[tuple[tuple, np.ndarray, dict[str, Any]]] = []
    failure = "lane_band_infeasible"
    for width in LANE_WIDTH_CHOICES:
        for sign in signs:
            for inner in LANE_INNER_CHOICES:
                if inner <= extents[sign] + 0.5:
                    failure = "lane_hits_dig"
                    continue
                centre = (
                    origin
                    + normal * sign * (inner + width / 2.0)
                    + direction * axial_centre
                )
                lane = base.rotated_rectangle(
                    (float(centre[0]), float(centre[1])),
                    extent + 2 * LANE_END_MARGIN,
                    float(width),
                    math.radians(heading_deg),
                )
                if not lane.any() or (lane & dig).any():
                    failure = "lane_hits_dig"
                    continue
                if border_margin(lane) < 1:
                    failure = "lane_off_map"
                    continue
                lane_axial = axial[lane]
                if lane_axial.min() > dig_axial.min() or lane_axial.max() < dig_axial.max():
                    failure = "lane_too_short"
                    continue
                realised = float((lateral[lane] * sign).min())
                if not (
                    LANE_INNER_BAND[0] - 1e-6 <= realised <= LANE_INNER_BAND[1] + 1e-6
                ):
                    failure = "lane_inner_band_infeasible"
                    continue
                proxy = float((lane & diggable).sum() / max(1, int(lane.sum())))
                meta = {
                    "lane_sign": int(sign),
                    "lane_inner_offset_tiles": round(realised, 3),
                    "lane_inner_requested_tiles": inner,
                    "lane_inner_dig_gap_tiles": round(float(distance[lane].min()), 3),
                    "lane_usable_proxy": round(proxy, 5),
                    "lane_width_tiles": width,
                    "lane_cells": int(lane.sum()),
                    "lane_centre_y": round(float(centre[0]), 3),
                    "lane_centre_x": round(float(centre[1]), 3),
                    "trench_working_length_tiles": round(extent, 3),
                }
                # rank: compliant strip first, then the seeded offset (variety),
                # then the widest lane that still complies (room for the base)
                key = (
                    proxy >= LANE_USABLE_PROXY_TARGET,
                    -abs(inner - layout.lane_inner),
                    width,
                    proxy,
                )
                candidates.append((key, lane, meta))
    if not candidates:
        raise RuntimeError(failure)
    _, lane, meta = max(candidates, key=lambda item: item[0])
    return lane, meta


def lane_metrics(dig, occupancy, dump, lane, lane_meta, heading_deg):
    blocked = dig | occupancy | dump
    free = float((lane & ~blocked).sum() / max(1, int(lane.sum())))
    drive = tsvc.backward_drive_check(
        dig,
        dig | occupancy,
        heading_deg,
        (float(lane_meta["lane_centre_y"]), float(lane_meta["lane_centre_x"])),
        float(lane_meta["trench_working_length_tiles"]),
        lane,
        BACKWARD_LANE_STEPS_MIN,
    )
    return {"lane_free_fraction": round(free, 5), **drive}


def lane_metrics_no_lane(dig, occupancy, heading_deg, dig_meta):
    """The retreat check for a trench with no RESERVED lane (U8 transport).

    The drive still has to happen beside the excavation, not on it — the machine
    cannot back up along the middle of its own trench. The track therefore starts
    at the centre of the §8.6 working band on the clear side of the spine, which
    is where a lane would have been reserved if the condition had one.
    """
    origin, direction, normal, axial, _ = spine_fields(dig.shape, dig_meta)
    extent = float(axial[dig].ptp())
    axial_centre = float((axial[dig].max() + axial[dig].min()) / 2.0)
    sign = float(dig_meta.get("trench_lane_side", 1))
    offset = sum(LANE_INNER_BAND) / 2.0
    centre = origin + normal * sign * offset + direction * axial_centre
    drive = tsvc.backward_drive_check(
        dig, dig | occupancy, heading_deg,
        (float(centre[0]), float(centre[1])), extent, None, 0,
    )
    return {
        "lane_free_fraction": 1.0,
        "retreat_band_offset_tiles": round(offset, 3),
        "trench_working_length_tiles": round(extent, 3),
        **drive,
    }


# --------------------------------------------------------------------------
# dump algorithms — trench banks hug the excavation (§8.6)


def trench_band(dig, dig_meta, layout, target_area, rng, blocked=None):
    """An adjacent spoil band hugging the whole trench, both flanks."""
    _, _, _, _, lateral = spine_fields(dig.shape, dig_meta)
    distance = ndi.distance_transform_edt(~dig)
    reach = int(rng.integers(12, 20))
    band = (distance >= layout.standoff) & (distance <= reach) & _interior(2)
    if blocked is not None:
        band &= ~blocked
    allowed = base.largest_component(band)
    if int(allowed.sum()) < target_area:
        raise RuntimeError("trench_band_infeasible")
    seeds = allowed & (distance <= layout.standoff + 2.75)
    if not seeds.any():
        seeds = allowed
    target = base.grow_region(allowed, seeds, target_area, rng)
    if int(target.sum()) < target_area:
        raise RuntimeError("trench_band_infeasible")
    negative = int((target & (lateral < 0)).sum())
    positive = int((target & (lateral >= 0)).sum())
    return target, {
        "dump_side": "trench_both_sides",
        "dump_access_sides": "both",
        "dump_alignment": "trench_adjacent_band",
        "dump_components_requested": base.target_components(np.where(target, 1, 0)),
        "distance_bucket": "immediate_near",
        "both_side_negative_cells": negative,
        "both_side_positive_cells": positive,
        "both_side_balance": round(
            min(negative, positive) / max(1, negative + positive), 4
        ),
        "trench_flank_reach_tiles": reach,
    }


def trench_flank(dig, dig_meta, layout, target_area, rng, blocked=None, side=1):
    """One adjacent flank, on the lane side.

    §8.6 R3: the compliant lane leaves 4.5-9 tiles between the trench and the
    corridor, which is not enough room for a 2x flank on a 30-tile trench. The
    band therefore keeps its near strip (that is what a lane station turn-dumps
    into, down-line) and takes the rest of its capacity by wrapping the trench
    ENDS and continuing past the lane. Reachability is gated on the measured
    ``turn_dump_cov_strict``, not on where the last designated cell sits.
    """
    _, _, _, _, lateral = spine_fields(dig.shape, dig_meta)
    distance = ndi.distance_transform_edt(~dig)
    reach = int(rng.integers(16, 26))
    sign = float(side)
    band = (
        (distance >= layout.standoff)
        & (distance <= reach)
        & (sign * lateral >= -2.0)
        & _interior(2)
    )
    if blocked is not None:
        band &= ~blocked
    allowed = base.largest_component(band)
    if int(allowed.sum()) < target_area:
        raise RuntimeError("one_side_infeasible")
    seeds = allowed & (distance <= layout.standoff + 2.75)
    if not seeds.any():
        seeds = allowed
    target = base.grow_region(allowed, seeds, target_area, rng)
    if int(target.sum()) < target_area:
        raise RuntimeError("one_side_infeasible")
    return target, {
        "dump_side": "trench_main_axis_left" if sign < 0 else "trench_main_axis_right",
        "dump_access_sides": "one",
        "dump_alignment": "trench_one_side_adjacent",
        "dump_components_requested": 1,
        "distance_bucket": "immediate_near",
        "one_side_sign": int(sign),
        "one_side_purity": round(
            float((target & (sign * lateral >= 1.0)).sum() / max(1, int(target.sum()))),
            4,
        ),
        "trench_flank_reach_tiles": reach,
    }


def trench_altsides(dig, dig_meta, layout, target_area, rng, blocked=None):
    """Alternating banks along the trench, always adjacent (U3)."""
    _, _, _, axial, lateral = spine_fields(dig.shape, dig_meta)
    distance = ndi.distance_transform_edt(~dig)
    dig_axial = axial[dig]
    low, high = float(dig_axial.min()), float(dig_axial.max())
    extent = high - low
    gap = max(2.5, extent * 0.06)
    feasible = int((extent + gap) // (ALTSIDES_MIN_SPAN + gap))
    requested = max(2, min(int(layout.alternations), feasible))
    failure = "altsides_span_too_short"
    for n in range(requested, 1, -1):
        span = (extent - gap * (n - 1)) / n
        if span < ALTSIDES_MIN_SPAN:
            continue
        try:
            return v8._altsides_pads(
                dig, layout, target_area, rng, blocked, 0, 0.0,
                axial, lateral, distance, low, high, extent, gap, n, span,
            )
        except RuntimeError as exc:
            failure = str(exc)
    raise RuntimeError(failure)


def offset_fence_wall(dig, dump, dump_meta, rng, from_hull: bool = True):
    """v3.2's gapped offset fence, with the offset reference made explicit.

    Identical to ``generate_prototypes_v7.make_offset_fence_wall`` except that
    the band may be offset from the EXCAVATION instead of from its convex hull.
    Foundations keep the hull (a slab's hull is the slab); trenches must not use
    it — see the call site.
    """
    from skimage.morphology import convex_hull_image

    reference = convex_hull_image(dig) if from_hull else dig
    reference_distance = ndi.distance_transform_edt(~reference)
    yy, xx = np.indices(dig.shape)
    cy, cx = centroid(dig)
    angles = np.arctan2(yy - cy, xx - cx)

    inner = float(dump_meta["fence_inner_offset_tiles"])
    thickness = float(dump_meta["fence_band_thickness_tiles"])
    axis = float(dump_meta["fence_axis_rad"])
    delta = float(dump_meta["fence_pad_delta_rad"])
    half_width = float(dump_meta["fence_pad_half_width_rad"])

    margin = math.radians(float(rng.uniform(*v7.FENCE_ARC_MARGIN_DEG)))
    cover = delta + half_width + margin
    band = (
        (reference_distance >= inner)
        & (reference_distance < inner + thickness)
        & (v2.angle_difference(angles, axis) <= cover)
    )
    band &= ~ndi.binary_dilation(dump, structure=base.binary_disk(3))
    band &= ~ndi.binary_dilation(dig, structure=base.binary_disk(3))
    if int(band.sum()) < 60:
        raise RuntimeError("fence_band_too_small")

    offaxis_max = max(v7.FENCE_GAP_MIN_OFFAXIS_DEG + 2.0, math.degrees(cover) - 8.0)
    for _ in range(24):
        gap_width = int(rng.integers(12, 17))
        offaxis = math.radians(
            float(rng.uniform(v7.FENCE_GAP_MIN_OFFAXIS_DEG, offaxis_max))
        )
        gate_angle = axis + float(rng.choice([-1.0, 1.0])) * offaxis
        ring = band & (v2.angle_difference(angles, gate_angle) <= math.radians(6.0))
        if not ring.any():
            continue
        gap_centre = centroid(ring)
        gap = np.hypot(yy - gap_centre[0], xx - gap_centre[1]) <= gap_width / 2
        wall = band & ~gap
        components, min_cells, min_extent = wall_component_stats(wall)
        if components < WALL_MIN_COMPONENTS or min_cells < WALL_MIN_COMPONENT_CELLS:
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
            "wall_style": "offset_fence_hull" if from_hull else "offset_fence_dig",
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


def build_dump(condition, dig, dig_meta, layout, rng, blocked=None, lane_sign=1):
    style = condition.dump_style
    if condition.capacity_level in APRON_CAPACITY_BANDS:
        band = APRON_CAPACITY_BANDS[condition.capacity_level]
    elif condition.capacity_level == "tight":
        band = TIGHT_CAPACITY_BAND
    elif condition.id in WALL_CONDITIONS:
        band = WALL_SPLIT_CAPACITY_BAND
    elif style == "ring_band":
        band = RING_BAND_DRAW
    else:
        band = GENEROUS_CAPACITY_BANDS[style]
    factor = float(rng.uniform(*band))
    target_area = int(math.ceil(int(dig.sum()) * factor))

    if style == "ring_band":
        target, metadata = ring_band(dig, target_area, rng, blocked)
    elif style == "capacity_apron":
        target, metadata = apron_sector(dig, layout, target_area, rng)
    elif style == "distance_apron":
        target, metadata = distance_apron(
            dig, layout, target_area, rng, DISTANCE_BINS[condition.distance_level]
        )
    elif style == "one_side_near":
        target, metadata = v7.foundation_one_side(dig, layout, target_area, rng, blocked)
    elif style == "separated_zones":
        target, metadata = v7.separated_zones(
            dig, layout, target_area, rng, clustered=condition.id in WALL_CONDITIONS
        )
    elif style == "haul_away_edge":
        target, metadata = v7.remote_edge(dig, layout, target_area, rng)
    elif style == "trench_band":
        target, metadata = trench_band(dig, dig_meta, layout, target_area, rng, blocked)
    elif style == "trench_flank":
        target, metadata = trench_flank(
            dig, dig_meta, layout, target_area, rng, blocked, lane_sign
        )
    elif style == "trench_altsides":
        target, metadata = trench_altsides(
            dig, dig_meta, layout, target_area, rng, blocked
        )
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
# sample construction


def _common_metadata(condition, dig, dump, occupancy, dumpability, gate):
    _, dig_components = ndi.label(dig, structure=np.ones((3, 3), dtype=np.uint8))
    return {
        "schema": SCHEMA,
        "dataset": condition.dataset,
        "condition_id": condition.id,
        "tier": condition.tier,
        "preview": condition.preview,
        "anchor_condition_id": condition.anchor or "",
        "anchor_is_external": condition.anchor in tax.RELEASES[
            f"v5-{condition.dataset}"
        ].external_anchors,
        "family": condition.family,
        "geometry_level": condition.geometry_level,
        "dump_level": condition.dump_level,
        "capacity_level": condition.capacity_level,
        "site_level": condition.site_level,
        "distance_level": condition.distance_level,
        "capacity_band_token": CAPACITY_TOKENS[condition.capacity_level],
        "distance_band_token": DISTANCE_TOKENS[condition.distance_level],
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
        # §8.6 R7 — live scale, not the stale 0.6875
        "tile_size_m": TILE_SIZE_M,
        "single_station_budget_tiles": round(SINGLE_STATION_BUDGET_TILES, 4),
        "workspace_r_min_tiles": round(R_MIN_TILES, 4),
        "workspace_r_max_tiles": round(R_MAX_TILES, 4),
        "static_gate_is_action_witness": False,
    }


def make_sample(condition, dataset, dig, dig_meta, layout, rng):
    is_trench = condition.family == "trench"
    heading = float(dig_meta.get("trench_global_angle_deg", 0.0))

    blocked = np.zeros_like(dig, dtype=bool)
    lane = np.zeros_like(dig, dtype=bool)
    lane_meta: dict[str, Any] = {}
    # A reserved working lane only means something when the spoil goes beside
    # the trench. U8's transport trenches (remote haul, split + wall) put the
    # spoil elsewhere on purpose, and the lane gate is waived for them.
    wants_lane = is_trench and condition.dump_style.startswith("trench_")
    if wants_lane:
        try:
            lane, lane_meta = build_lane(dig, dig_meta, layout)
        except RuntimeError as exc:
            if dataset.gate_lane_band:
                return None, str(exc)
            wants_lane = False
        if wants_lane:
            blocked |= lane
            probe = tsvc.backward_drive_check(
                dig, dig, heading,
                (float(lane_meta["lane_centre_y"]), float(lane_meta["lane_centre_x"])),
                float(lane_meta["trench_working_length_tiles"]),
                lane, BACKWARD_LANE_STEPS_MIN,
            )
            if probe["backward_drive_drift_per_tile"] > BACKWARD_DRIFT_PER_TILE_MAX:
                return None, "backward_drive_drift_contract"
            if not probe["backward_drive_footprint_clear"]:
                return None, "backward_drive_blocked_contract"

    is_ring = condition.dump_style == "ring_band"
    road: np.ndarray | None = None
    road_meta: dict[str, Any] = {}
    objects: np.ndarray | None = None
    object_meta: dict[str, Any] = {}
    band_probe_seed = 0
    if condition.site_level == "road":
        band_probe_seed = int(rng.integers(0, 2**31))
        radius_range = ROAD_RADIUS_RANGE_RING if is_ring else ROAD_RADIUS_RANGE_ZONED
        floor = ROAD_DIG_ANNULUS_STERILIZE_RANGE[0] if is_ring else 0.0
        protect = dig | lane if is_trench else dig
        road, road_meta = make_two_border_road(
            dig, protect, rng, radius_range, dig_annulus_sterilize_min=floor
        )
        blocked |= road
    elif condition.site_level in ("obj", "obj1"):
        low, high = OBJECT_BANDS[condition.site_level]
        requested_objects = int(rng.integers(low, high + 1))
        protect = dig | lane if is_trench else dig
        objects, areas = place_objects_v7(dig, protect, requested_objects, rng)
        intrusion = object_intrusion(dig, objects)
        if intrusion["annulus4_free_fraction"] < OBJECT_NEAR_ANNULUS_FREE_MIN:
            return None, "object_corridor_blocked"
        # U11: graded, non-overlapping blockage bands and no merged fields
        lo, hi = OBJECT_ANNULUS_BLOCK_BAND[condition.site_level]
        if not lo <= intrusion["annulus6_blocked_fraction"] <= hi:
            return None, "object_blockage_band_contract"
        if intrusion["annulus6_blocked_fraction"] > OBJECT_TOTAL_BLOCK_MAX:
            return None, "object_total_blockage_contract"
        labels, n_clusters = ndi.label(objects, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = ndi.sum(objects, labels, range(1, n_clusters + 1)) if n_clusters else [0]
        if n_clusters and max(sizes) > OBJECT_MAX_CLUSTER_CELLS:
            return None, "object_cluster_size_contract"
        object_meta = {
            **intrusion,
            "object_count": requested_objects,
            "object_clusters": int(n_clusters),
            "object_max_cluster_cells": int(max(sizes)) if n_clusters else 0,
            "object_footprint_profile": OBJECT_FOOTPRINT_PROFILE,
            "object_area_cells_mean": round(float(np.mean(areas)), 3),
            "object_area_cells_min": int(min(areas)),
            "object_area_cells_max": int(max(areas)),
        }
        blocked |= objects

    lane_sign = int(lane_meta.get("lane_sign", layout.lane_sign))
    dump, dump_meta = build_dump(
        condition, dig, dig_meta, layout, rng,
        blocked if blocked.any() else None, lane_sign,
    )
    requested = int(dump_meta["dump_target_cells_requested"])
    if int(dump.sum()) < requested:
        return None, "capacity_generation_shortfall"
    dump &= ~dig

    occupancy = np.zeros_like(dig, dtype=bool)
    dumpability = np.ones_like(dig, dtype=bool)
    corridor = np.zeros_like(dig, dtype=bool)
    site_meta: dict[str, Any] = {
        "site_level": condition.site_level,
        "site_class": SITE_CLASS_TOKENS[condition.site_level],
        "object_count": 0,
        "wall_gap_tiles": 0,
    }
    if condition.site_level == "road":
        corridor = road & ~(dig | dump)
        dumpability = ~corridor
        site_meta.update(road_meta)
        site_meta.update(zoned_road_metrics(dig, dump, corridor))
        if is_ring:
            site_meta["road_gate_annulus"] = "band"
        if int(corridor.sum()) != int(road.sum()):
            return None, "road_clipped_contract"
        if site_meta["road_components"] != 1:
            return None, "road_split_contract"
        if site_meta["road_borders_touched"] < 2:
            return None, "road_border_contract"
    elif objects is not None:
        occupancy = objects & ~(dig | dump)
        if int(occupancy.sum()) != int(objects.sum()):
            return None, "site_overlaps_dump"
        dumpability = ~occupancy
        site_meta.update(object_meta)
    elif condition.site_level == "wall":
        # A comb trench's convex hull swallows the space between its branches, so
        # a hull-offset fence lands inside the work area and seals the base out
        # (measured: 310/640 `dig_not_reachable_pre` on the first attempt). For
        # trenches the offset is taken from the excavation itself.
        wall, wall_meta = offset_fence_wall(
            dig, dump, dump_meta, rng, from_hull=not is_trench
        )
        occupancy = wall & ~(dig | dump)
        if (occupancy & dump).any() or int(occupancy.sum()) != int(wall.sum()):
            return None, "site_overlaps_dump"
        dumpability = ~occupancy
        site_meta.update(wall_meta)

    target = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    target[dig] = -1
    target[dump] = 1
    components = base.target_components(target)
    if components != int(dump_meta["dump_components_requested"]):
        return None, "dump_component_contract"

    gate = base.static_gate(target, occupancy, dumpability)
    if not gate.accepted:
        return None, gate.reason
    if condition.tier == 0 and gate.dig_workspace_coverage_post < T0_DIG_COVERAGE_MIN:
        return None, "t0_dig_coverage_contract"

    # ---- capacity -------------------------------------------------------
    required = float(dump_meta["capacity_factor_required"])
    capacity_ratio = float(dump.sum() / max(1, dig.sum()))
    reachable_ratio = float(gate.reachable_dump_cells_post / max(1, dig.sum()))
    low = float(dump_meta["capacity_band_low"])
    high = float(dump_meta["capacity_band_high"])
    if condition.capacity_level in APRON_CAPACITY_BANDS or condition.capacity_level == "tight":
        if not low <= reachable_ratio <= high:
            return None, "capacity_band_contract"
    elif condition.dump_style == "ring_band":
        lo, hi = RING_BAND_CAPACITY
        if not lo <= capacity_ratio <= hi:
            return None, "ring_band_capacity_contract"
        if not lo <= reachable_ratio <= hi:
            return None, "ring_band_reachable_contract"
    else:
        if capacity_ratio + 1e-8 < required:
            return None, "capacity_ratio_contract"
        if reachable_ratio + 1e-8 < GENEROUS_REACHABLE_FLOOR:
            return None, "generous_capacity_floor"

    # ---- distance -------------------------------------------------------
    distance_metrics = v2.dig_to_dump_distance_metrics(dig, dump)
    if condition.dump_style == "distance_apron":
        target_median = DISTANCE_BINS[condition.distance_level]
        realized = distance_metrics["dig_dump_distance_median_tiles"]
        if abs(realized - target_median) > DISTANCE_BIN_TOLERANCE:
            return None, "distance_bin_contract"
    else:
        limits = TRENCH_MEDIAN_LIMITS if is_trench else FOUNDATION_MEDIAN_LIMITS
        limit = limits.get(condition.dump_style)
        if condition.id in WALL_CONDITIONS:
            limit = TRENCH_WALL_MEDIAN_LIMIT if is_trench else WALL_SPLIT_MEDIAN_LIMIT
        if limit is not None:
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
        _, edge_gap, span = pad_separation(dig, dump)
        if edge_gap < SPLIT_PAD_EDGE_GAP_MIN:
            return None, "split_pad_edge_gap_contract"
        if span < SPLIT_PAD_ANGULAR_SPAN_MIN:
            return None, "split_pad_angular_span_contract"
        if float(site_meta.get("wall_detour_ratio", 0.0)) < WALL_DETOUR_MIN:
            return None, "wall_detour_contract"

    extra: dict[str, Any] = {}

    # ---- apron / ring proximity (Lorenzo's v5-main review) --------------
    proximity = dump_proximity(dig, dump)
    extra.update(proximity)
    if condition.dump_style in PROXIMITY_GATED_STYLES:
        if proximity["dump_proximity_p95_tiles"] > DUMP_PROXIMITY_P95_MAX:
            return None, "dump_proximity_p95_contract"
        if proximity["dump_proximity_max_tiles"] > DUMP_PROXIMITY_MAX_MAX:
            return None, "dump_proximity_max_contract"

    # ---- U2 direct service ---------------------------------------------
    coverage = tsvc.direct_service_coverage(target, occupancy, dumpability)
    extra["direct_service_coverage"] = round(coverage, 5)
    if condition.tier == 0 and coverage < DIRECT_SERVICE_MIN:
        return None, "direct_service_contract"
    if condition.distance_level == "near":
        if distance_metrics["dig_dump_distance_p95_tiles"] > DIRECT_SERVICE_P95_MAX:
            return None, "direct_service_p95_contract"

    # ---- U1 road-on-a-finite-band --------------------------------------
    if condition.site_level == "road" and is_ring:
        open_band, _ = ring_band(dig, requested, np.random.default_rng(band_probe_seed))
        sterilized = float((corridor & open_band).sum() / max(1, int(open_band.sum())))
        extra["road_band_sterilized_fraction"] = round(sterilized, 4)
        if sterilized < ROAD_BAND_STERILIZE_MIN:
            return None, "ring_road_band_contract"
        dig_share = site_meta["road_dig_annulus6_sterilized_fraction"]
        lo, hi = ROAD_DIG_ANNULUS_STERILIZE_RANGE
        if not lo <= dig_share <= hi:
            return None, "ring_road_dig_annulus_contract"
        if site_meta["road_dig_annulus6_free_fraction"] < ROAD_ANNULUS_FREE_MIN:
            return None, "ring_road_free_contract"
    elif condition.site_level == "road":
        if not zoned_road_bites(site_meta):
            return None, "zoned_road_bite_contract"
        if site_meta["road_dig_annulus6_free_fraction"] < ROAD_ANNULUS_FREE_MIN:
            return None, "zoned_road_free_contract"

    # ---- U4 overlap stressor -------------------------------------------
    if condition.geometry_level in MULTI_ARM_LEVELS:
        worst, per_arm = arm_overlap_fraction(dig, dig_meta)
        extra["arm_overlap_fraction"] = round(worst, 4)
        extra["arm_overlap_per_arm"] = per_arm
        if worst < ARM_OVERLAP_MIN:
            return None, "arm_overlap_contract"

    # ---- U7 lane geometry, lattice, backward drive ----------------------
    if is_trench:
        extra["lane_present"] = int(wants_lane)
        if wants_lane:
            metrics = lane_metrics(dig, occupancy, dump, lane, lane_meta, heading)
        else:
            metrics = lane_metrics_no_lane(dig, occupancy, heading, dig_meta)
        extra.update(lane_meta)
        extra.update(metrics)
        if wants_lane and metrics["lane_free_fraction"] < 1.0:
            return None, "lane_not_free_contract"
        # A lane-less trench (U8 transport: remote haul, split + wall) has no
        # reserved corridor by design, so the retreat is checked in the §8.6
        # working band and REPORTED rather than gated — a wall is precisely what
        # removes a straight corridor.
        if (wants_lane or dataset.gate_lane_band) and not metrics[
            "backward_drive_footprint_clear"
        ]:
            return None, "backward_drive_blocked_contract"
        if metrics["backward_drive_drift_per_tile"] > BACKWARD_DRIFT_PER_TILE_MAX:
            return None, "backward_drive_drift_contract"
        if (
            wants_lane
            and int(lane_meta["lane_width_tiles"]) >= LANE_WIDTH_FOR_RETREAT_GATE
            and metrics["backward_drive_lane_steps"] < BACKWARD_LANE_STEPS_MIN
        ):
            return None, "backward_drive_lane_contract"
        spurs, notches = edge_irregularity(dig)
        extra["trench_edge_spurs"] = spurs
        extra["trench_edge_notches"] = notches
        if spurs or notches:
            return None, "trench_edge_regularity_contract"
        if dataset.gate_lane_band and wants_lane:
            inner = float(lane_meta["lane_inner_offset_tiles"])
            if not LANE_INNER_BAND[0] - 1e-6 <= inner <= LANE_INNER_BAND[1] + 1e-6:
                return None, "lane_inner_band_contract"

    # ---- §8.6 turn-only dumpability from the natural pose ----------------
    trench_probe = None
    if is_trench:
        trench_probe = {
            "heading_deg": heading,
            "arms": _as_arms(dig_meta["trench_arms"]),
            "half_width": float(dig_meta["trench_half_width_tiles"]),
            "lane": lane if wants_lane else None,
        }
    # screen on the gated numbers only; the per-subset breakdown is re-measured
    # once at the bottom, on the maps that survive
    extra.update(tdump.measure(target, occupancy, dumpability, trench_probe, full=False))
    if dataset.gate_turn_dump:
        if extra["turn_dump_cov_strict"] < TURN_DUMP_COV_MIN:
            return None, "turn_dump_cov_contract"
        if is_trench:
            # §8.6: the per-station gate exists for HUGGING banks — a trench
            # bank sits inside the dead ring, so a station can only dump into it
            # down-line. It is not a foundation gate: an apron leaves plenty of
            # legal stations on the far side of the slab that can dig but not
            # reach the apron, and coverage (which is what a plan needs) is the
            # right measure there.
            floor = STATION_DUMP_FRAC_MIN_BY_STYLE.get(
                condition.dump_style, STATION_DUMP_FRAC_MIN
            )
            if condition.site_level != "clean":
                floor = min(floor, STATION_DUMP_FRAC_MIN_BLOCKED_SITE)
            if extra["turn_dump_station_dump_frac"] < floor:
                return None, "station_dump_frac_contract"
            if wants_lane and extra["lane_usable_frac"] < LANE_USABLE_FRAC_MIN:
                return None, "lane_usable_frac_contract"

    # ---- §8.5 U10 start-side sensitivity ---------------------------------
    if condition.planning:
        plan = tdump.plan_sensitivity(target, occupancy, dumpability, PLAN_STEPS)
        extra.update(plan)
        if plan["plan_delta"] < PLAN_DELTA_MIN:
            return None, "plan_start_side_contract"
        if plan["plan_cost_far"] > PLAN_FAR_COST_MAX:
            return None, "plan_far_cost_contract"

    # ---- §8.5 U8 transport: multi-leg staging feasibility ----------------
    if dataset.gate_staging:
        staging = tdump.staging_feasibility(
            target, occupancy, dumpability, max_hops=STAGING_MAX_HOPS + 2
        )
        extra.update(staging)
        if staging["staging_infeasible_frac"] > 0.0:
            return None, "staging_infeasible_contract"
        if staging["staging_max_hops"] > STAGING_MAX_HOPS:
            return None, "staging_hops_contract"

    # accepted: pay for the full §8.6 breakdown once
    extra.update(tdump.measure(target, occupancy, dumpability, trench_probe, full=True))

    metadata = {
        **_common_metadata(condition, dig, dump, occupancy, dumpability, gate),
        "dump_components_actual": components,
        **distance_metrics,
        **v8._split_metrics(dig, dump),
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


def make_map(condition, dataset, dig, dig_meta, layout, rng):
    try:
        return make_sample(condition, dataset, dig, dig_meta, layout, rng)
    except RuntimeError as exc:
        return None, str(exc)


def generate_condition(condition, dataset, condition_index, bank, n_maps, max_attempts):
    samples: list[base.Sample] = []
    rejections: Counter[str] = Counter()
    unsatisfied: list[str] = []
    accepted_digs: list[np.ndarray] = []
    for map_index in range(n_maps):
        layout = layout_for(condition, map_index)
        accepted: base.Sample | None = None
        for attempt in range(max_attempts):
            salt = (
                0
                if attempt < SHARED_DIG_ATTEMPTS
                else 1 + (attempt - SHARED_DIG_ATTEMPTS) // REROLL_DUMP_ATTEMPTS
            )
            dig, dig_meta = bank.get(condition.geometry_level, map_index, salt)
            if dig is None:
                rejections["dig_reroll_exhausted"] += 1
                continue
            iou_limit = (
                TRENCH_DIG_IOU_MAX
                if condition.geometry_level in TRENCH_LEVELS
                else DIG_IOU_MAX
            )
            if salt and any(
                centred_iou(dig, other) >= iou_limit for other in accepted_digs
            ):
                rejections["condition_dig_iou"] += 1
                continue
            seed = int(
                np.random.SeedSequence(
                    [SEED_BASE, condition_index, map_index, attempt]
                ).generate_state(1)[0]
            )
            sample, reason = make_map(
                condition, dataset, dig, dig_meta, layout, np.random.default_rng(seed)
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


def write_condition(output, dataset, condition, condition_index, samples):
    data = output / "dataset"
    folder = output / condition.id
    (folder / "previews").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        map_index = int(sample.metadata["map_index"])
        sample_index = sample_index_of(condition_index, map_index)
        map_id = f"{dataset.map_id_prefix}-{sample_index:04d}"
        for folder_name, attribute in ARRAY_FOLDERS.items():
            np.save(
                data / folder_name / f"img_{sample_index}.npy",
                getattr(sample, attribute),
            )
        owners = trench_axis_owners(sample.target, sample.metadata)
        np.save(
            data / TRENCH_AXIS_OWNERS_FOLDER / f"img_{sample_index}.npy",
            owners,
        )
        record = {
            "sample_index": sample_index,
            "map_id": map_id,
            "trench_axis_owners_sha256": sha256_mask(owners),
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
                "dataset": dataset.name,
                "tier": condition.tier,
                "tierLabel": tax.TIER_LABELS[condition.tier],
                "preview": condition.preview,
                "anchorConditionId": condition.anchor,
                "anchorIsExternal": condition.anchor
                in tax.RELEASES[dataset.release].external_anchors,
                "family": condition.family,
                "factorLevels": condition.levels,
                "factors": {
                    "geometryClass": condition.geometry,
                    "dumpLayout": condition.dump_layout,
                    "siteClass": SITE_CLASS_TOKENS[condition.site_level],
                    "capacityBand": CAPACITY_TOKENS[condition.capacity_level],
                    "distanceBand": DISTANCE_TOKENS[condition.distance_level],
                },
                "objectBand": list(OBJECT_BANDS[condition.site_level]),
                "layoutGroup": condition.group,
                "pairGroup": condition.pair_group,
                "seedBase": SEED_BASE,
                "tileSizeM": TILE_SIZE_M,
                "singleStationBudgetTiles": round(SINGLE_STATION_BUDGET_TILES, 4),
                "mapCount": len(rows),
                "maps": [
                    {
                        "id": row["map_id"],
                        "sampleIndex": row["sample_index"],
                        "mapIndex": row["map_index"],
                        "arrays": {
                            name: f"dataset/{name}/img_{row['sample_index']}.npy"
                            for name in ARRAY_FOLDERS
                        }
                        | {
                            TRENCH_AXIS_OWNERS_FOLDER:
                                f"dataset/{TRENCH_AXIS_OWNERS_FOLDER}/"
                                f"img_{row['sample_index']}.npy"
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


def write_conditions_csv(output: Path, dataset, counts: dict[str, int]) -> None:
    with (output / "conditions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tax.CSV_COLUMNS)
        writer.writeheader()
        for condition in dataset.conditions:
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
        axes = row.get("axes_ABC", [])
        payload = {
            "schema": f"{SCHEMA}_axis_metadata",
            "geometry": row["geometry"],
            "trench_axes_count": len(axes) if axes else -1,
            "trench_topology": row.get("trench_topology", ""),
            "axes_ABC": axes,
            "trench_axis_contract": TRENCH_AXIS_CONTRACT,
            "trench_axis_owners_sha256": row["trench_axis_owners_sha256"],
        }
        (destination / f"trench_{row['sample_index']}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        )


def write_readme(output: Path, dataset, counts: dict[str, int]) -> None:
    if dataset.name == "main":
        purpose = [
            "The **main** curriculum: every condition is turn-dumpable from the",
            "natural digging poses (`turn_dump_cov_strict >= 0.95`), so the",
            "machine can put the spoil away by rotating the cabin, without a",
            "base move between dig and dump.",
        ]
    else:
        purpose = [
            "The **transport** probe set (spec 8.5 U8): walls, remote hauls and",
            "the top distance bins. These test long-range soil transport, which",
            "is a different capability, and they are deliberately NOT",
            "single-station solvable:",
            "",
            f"> **Single-station budget: {SINGLE_STATION_BUDGET_TILES:.2f} tiles.**",
            "> A dig->dump separation beyond `2 * r_max` cannot be closed from",
            "> any one station, so every map here is solved by staging.",
            "",
            "Each map is validated for multi-leg feasibility: a",
            "dig -> stage -> re-dig chain to the designated dump exists within",
            f"at most {STAGING_MAX_HOPS} reach hops (`staging_max_hops`).",
        ]
    lines = [
        f"# Terra curriculum v5 axis-contract v2 bank — `{dataset.name}`",
        "",
        "Taxonomy-native bank: one folder per **condition id**, no stage folders.",
        "Difficulty tier is computed from factor levels, never hand-assigned.",
        "See `../../../terra-digging-benchmark-site/docs/CURRICULUM_TAXONOMY_SPEC.md`"
        " sections 8, 8.1, 8.2, 8.3, 8.4, **8.5 and 8.6**.",
        "",
        *purpose,
        "",
        f"- seed base: `{SEED_BASE}` (fully reproducible)",
        f"- taxonomy release: `{dataset.release}`",
        f"- conditions: {len(dataset.conditions)}",
        f"- maps: {sum(counts.values())} ({dataset.maps_per_condition} per condition)",
        f"- map ids: `{dataset.map_id_prefix}-NNNN`",
        f"- tile size: {TILE_SIZE_M:.10f} m (live scale; the 0.6875 in v2-v4"
        " manifests was stale)",
        f"- service annulus: [{R_MIN_TILES:.3f}, {R_MAX_TILES:.3f}] tiles",
        "",
        "## Layout",
        "",
        "- `<condition-id>/manifest.json` — factor levels, tier, per-map metrics",
        "- `<condition-id>/previews/*.png` — one labelled composite per map",
        "- `<condition-id>/overview.png` — the whole condition on one sheet",
        "- `dataset/{images,occupancy,dumpability,actions,distance,"
        "trench_axis_owners}/img_N.npy`"
        " — Terra arrays, flat and shared across conditions",
        "- `manifest.csv` — every map, every measured factor",
        "- `conditions.csv` — spec section 4 columns",
        "- `GENERATION_NOTES.md` — per-U implementation, realized ranges, deviations",
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
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--only", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DATASETS[args.dataset]
    assert_conditions_match_taxonomy(dataset)
    output = args.output.resolve()
    for folder in (
        output,
        output / "review_metadata",
        *(output / "dataset" / name for name in ARRAY_FOLDERS),
        output / "dataset" / TRENCH_AXIS_OWNERS_FOLDER,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    selected = set(filter(None, args.only.split(",")))
    factory = GeometryFactoryV9(args.source_foundations)
    bank_size = args.maps or dataset.maps_per_condition
    t0_levels = frozenset(c.geometry_level for c in dataset.conditions if c.tier == 0)
    bank = DigBankV9(factory, bank_size, t0_levels)
    wanted = selected or {c.id for c in dataset.conditions}
    levels = sorted({c.geometry_level for c in dataset.conditions if c.id in wanted})
    bank_rejections = bank.build(levels)
    print(f"dig bank built: {bank_size} per level, rejections={dict(bank_rejections)}")

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    rejection_totals: Counter[str] = Counter()
    unsatisfied: list[str] = []

    for condition_index, condition in enumerate(dataset.conditions):
        if selected and condition.id not in selected:
            continue
        n_maps = args.maps or dataset.maps_per_condition
        samples, rejections, failures = generate_condition(
            condition, dataset, condition_index, bank, n_maps, args.max_attempts
        )
        rejection_totals.update(rejections)
        unsatisfied.extend(failures)
        condition_rows = write_condition(
            output, dataset, condition, condition_index, samples
        )
        rows.extend(condition_rows)
        counts[condition.id] = len(condition_rows)
        rerolled = sum(1 for row in condition_rows if not row["shared_dig"])
        print(
            f"[{condition_index + 1:02d}/{len(dataset.conditions)}] {condition.id}: "
            f"{len(condition_rows)}/{n_maps} maps"
            + (f" rerolled={rerolled}" if rerolled else "")
            + (f" UNSATISFIED={len(failures)}" if failures else "")
            + (f" rejections={dict(rejections.most_common(4))}" if rejections else ""),
            flush=True,
        )

    write_terra_metadata(output, rows)
    regenerated = sorted(counts)

    if args.resume:
        with (output / "manifest.csv").open(newline="") as handle:
            previous = [
                row
                for row in csv.DictReader(handle)
                if row["condition_id"] not in selected
            ]
        backfilled = 0
        for row in previous:
            row["sample_index"] = int(row["sample_index"])
            counts[row["condition_id"]] = counts.get(row["condition_id"], 0) + 1
            # A carried row predates any column this run added. Rather than leave
            # the manifest half-populated, the missing proximity columns are
            # RECOMPUTED from that map's exported arrays — the same function the
            # gate uses, on the same bytes the reviewer will look at.
            if not row.get("dump_proximity_p95_tiles"):
                target = np.load(
                    output / "dataset" / "images" / f"img_{row['sample_index']}.npy"
                )
                row.update(dump_proximity(target < 0, target > 0))
                backfilled += 1
        rows.extend(previous)
        print(
            f"resume: carried {len(previous)} rows from the existing manifest"
            + (f", backfilled proximity on {backfilled}" if backfilled else "")
        )

    rows.sort(key=lambda row: row["sample_index"])
    write_manifest(output, rows)
    write_conditions_csv(output, dataset, counts)
    write_readme(output, dataset, counts)

    realized = defaultdict(list)
    realized_capacity = defaultdict(list)
    turn_dump = defaultdict(list)
    for row in rows:
        realized[row["condition_id"]].append(float(row["reachable_dump_to_dig_ratio"]))
        realized_capacity[row["condition_id"]].append(float(row["dump_to_dig_area_ratio"]))
        if row.get("turn_dump_cov_strict") not in (None, ""):
            turn_dump[row["condition_id"]].append(float(row["turn_dump_cov_strict"]))
    summary = {
        "schema": SCHEMA,
        "dataset": dataset.name,
        "seed_base": SEED_BASE,
        "spec_path": "docs/CURRICULUM_TAXONOMY_SPEC.md",
        "spec_section": "8 + 8.1 + 8.2 + 8.3 + 8.4 + 8.5 + 8.6",
        "taxonomy_version": tax.TAXONOMY_VERSION,
        "taxonomy_release": dataset.release,
        "generator": "generate_prototypes_v9.py",
        "map_id_prefix": dataset.map_id_prefix,
        "conditions_built_this_run": regenerated,
        "source_foundations": str(args.source_foundations),
        "condition_count": len(dataset.conditions),
        "maps_per_condition": counts,
        "accepted_maps": len(rows),
        "resume": bool(args.resume),
        "rerolled_dig_maps": sum(1 for row in rows if int(row["shared_dig"]) == 0),
        "tile_size_m": TILE_SIZE_M,
        "workspace_annulus_tiles": [round(R_MIN_TILES, 4), round(R_MAX_TILES, 4)],
        "single_station_budget_tiles": round(SINGLE_STATION_BUDGET_TILES, 4),
        "u_gates": {
            "u1_ring_band_capacity": list(RING_BAND_CAPACITY),
            "u2_direct_service_min": DIRECT_SERVICE_MIN,
            "u4_arm_overlap_min": ARM_OVERLAP_MIN,
            "u5_apron_capacity_bands": {k: list(v) for k, v in APRON_CAPACITY_BANDS.items()},
            "u6_distance_bins": DISTANCE_BINS,
            "u7_trench_axes_deg": list(TRENCH_AXES_DEG),
            "u8_staging_max_hops": STAGING_MAX_HOPS,
            "u10_plan_delta_min": PLAN_DELTA_MIN,
            "u10_plan_far_cost_max": PLAN_FAR_COST_MAX,
            "u11_object_bands": {k: list(v) for k, v in OBJECT_BANDS.items()},
            "u11_object_blockage_bands": {
                k: list(v) for k, v in OBJECT_ANNULUS_BLOCK_BAND.items()
            },
            "u11_object_max_cluster_cells": OBJECT_MAX_CLUSTER_CELLS,
            "s86_turn_dump_cov_min": TURN_DUMP_COV_MIN,
            "s86_station_dump_frac_min": STATION_DUMP_FRAC_MIN,
            "s86_lane_inner_band_tiles": list(LANE_INNER_BAND),
            "s86_lane_usable_frac_min": LANE_USABLE_FRAC_MIN,
            "s86_trench_bank_standoff": TRENCH_BANK_STANDOFF,
            "review_dump_proximity_p95_max_tiles": DUMP_PROXIMITY_P95_MAX,
            "review_dump_proximity_max_max_tiles": DUMP_PROXIMITY_MAX_MAX,
            "review_proximity_gated_styles": sorted(PROXIMITY_GATED_STYLES),
            "review_apron_proximity_sector_deg": APRON_PROXIMITY_SECTOR_DEGREES,
            "review_apron_proximity_standoff_max": APRON_PROXIMITY_STANDOFF_MAX,
            "wall_detour_min": WALL_DETOUR_MIN,
            "wall_detour_condition_median_min": WALL_DETOUR_CONDITION_MEDIAN_MIN,
        },
        "gates_enforced": {
            "turn_dump": dataset.gate_turn_dump,
            "lane_band": dataset.gate_lane_band,
            "staging": dataset.gate_staging,
        },
        "generous_capacity_bands": {k: list(v) for k, v in GENEROUS_CAPACITY_BANDS.items()},
        "tight_capacity_band": list(TIGHT_CAPACITY_BAND),
        "realized_reachable_capacity": {
            key: [round(min(v), 3), round(max(v), 3)] for key, v in sorted(realized.items())
        },
        "realized_designated_capacity": {
            key: [round(min(v), 3), round(max(v), 3)]
            for key, v in sorted(realized_capacity.items())
        },
        "realized_turn_dump_cov_strict": {
            key: [round(min(v), 4), round(float(np.median(v)), 4), round(max(v), 4)]
            for key, v in sorted(turn_dump.items())
        },
        "shared_dig_attempts": SHARED_DIG_ATTEMPTS,
        "reroll_dump_attempts": REROLL_DUMP_ATTEMPTS,
        "dig_bank_rejections": dict(bank_rejections),
        "unsatisfied_constraints": unsatisfied,
        "rejections_before_acceptance": dict(rejection_totals),
        "static_gate_is_action_witness": False,
    }
    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "maps_per_condition"},
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    if unsatisfied:
        print("UNSATISFIED CONSTRAINTS:")
        for line in unsatisfied:
            print(f"  {line}")


if __name__ == "__main__":
    main()
