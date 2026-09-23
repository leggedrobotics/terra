#!/usr/bin/env python3
"""v4 curriculum review bank — spec §8.3 (U1-U7) and §8.4 (31 conditions).

Lineage: ``generate_prototypes_v7.py`` (the v3.1/v3.2 bank). Everything the
§8/§8.1/§8.2 gates asked for carries forward; what changes is Lorenzo's review
pass:

U1  Ring masks are capped. The designated dump on a ring condition is a BAND
    grown around the dig to 3-4x reachable capacity, not "all legal free
    ground". This kills the vacuous mask (P0: purity == 1.0 on 112 maps), the
    2-32x `generous` spread, and the ring/zoned capacity non-orthogonality, and
    it re-opens `ring-road`: a road crossing a *finite* band bites.
U2  Every T0 map guarantees direct service: >= 99% of dig cells have a
    designated dump cell reachable from the same station, measured with the
    LIVE excavator envelope (6.375-11.375 tiles, 7x11 footprint, +-30 deg cone)
    exactly as the P0 reference panel measured it. The apron standoff is
    randomised per map inside that envelope and controlled on the realised p95
    dig->nearest-dump distance, never on the raw parameter.
U3  Trench dumping is always adjacent. No trench condition has a far pad; the
    v3 `trn-straight-split` (pads off the trench ends, ~11-tile hauls) is
    replaced by `trn-straight-altsides` (alternating banks), and
    `trn-straight-remote` is dropped.
U4  Trench difficulty is ordering + dirt organisation. On every multi-segment
    trench the natural spoil flank of one arm OVERLAPS another arm's dig area,
    so dumping greedily buries future work and forces re-handling.
U5  The capacity ladder is densified below ~2x: c1p2 / c1p6 / c2x / c3x on one
    shared-dig apron pair group.
U6  A foundation-only distance ladder at 12/16/20/24-tile median haul, matched
    on digs and on the apron azimuth, staging expected at the top bins.
U7  Trench axes come only from the drivable 30 deg lattice (angles_base = 12),
    rasterised at constant perpendicular width with no single-cell notches, and
    every trench map reserves a straight base-width traversable lane parallel to
    the axis on the dumping side. A scripted backward drive with the env's own
    rounded move kinematics measures the realised drift.

Spec: ``terra-digging-benchmark-site/docs/CURRICULUM_TAXONOMY_SPEC.md``
§8, §8.1, §8.2, §8.3, §8.4.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

import generate_prototypes_v7 as v7
import terra_geom as tgeom
import terra_service as tsvc

v6 = v7.v6
v5 = v7.v5
v4 = v7.v4
v3 = v7.v3
v2 = v7.v2
base = v7.base
tax = v7.tax

SCHEMA = "terra_curriculum_v4_review_bank"
SEED_BASE = 20260729
MAP_ID_PREFIX = "curriculum-v4"
MAPS_PER_CONDITION = 16
PREVIEW_MAPS_PER_CONDITION = 4
SHARED_DIG_ATTEMPTS = 120
REROLL_DUMP_ATTEMPTS = 20
MAX_ATTEMPTS = 320

MAP_SIZE = base.MAP_SIZE
MAP_CENTRE = (MAP_SIZE - 1) / 2.0

ARRAY_FOLDERS = v6.ARRAY_FOLDERS
SITE_CLASS_TOKENS = v6.SITE_CLASS_TOKENS
OBJECT_BANDS = v6.OBJECT_BANDS

# --------------------------------------------------------------------------
# taxonomy tokens

CAPACITY_TOKENS = {
    "generous": "",
    "c3x": "slcap03_04",
    "c2x": "slcap01_90_02_10",
    "c1p6": "legacy_apron_1p6",
    "c1p2": "slcap01_15_01_25",
    "tight": "trench_constrained",
}
DISTANCE_TOKENS = {
    "unspec": "",
    "near": "direct_service_near",
    "far": "haul_far",
    "d12": "dist_bin_12",
    "d16": "dist_bin_16",
    "d20": "dist_bin_20",
    "d24": "dist_bin_24",
}

GEOMETRY_LEVEL_SOURCE = {
    "slab": "foundation_osm",
    "slab-lg": "foundation_osm_large",
    "proc": "foundation_procedural",
    "strips": "foundation_structural",
    "straight": "trench_axes_1",
    "seg2": "trench_segments_2",
    "seg3": "trench_segments_3",
    "tee": "trench_axes_2",
    "net": "trench_axes_3",
}
GEOMETRY_LEVEL_INDEX = {
    level: index for index, level in enumerate(sorted(GEOMETRY_LEVEL_SOURCE))
}
GEOMETRY_HARDNESS = dict(v6.GEOMETRY_HARDNESS)
GEOMETRY_HARDNESS["trench_segments_2"] = "trench_segmented_2"
GEOMETRY_HARDNESS["trench_segments_3"] = "trench_segmented_3"

TRENCH_LEVELS = frozenset({"straight", "seg2", "seg3", "tee", "net"})
MULTI_ARM_LEVELS = frozenset({"seg2", "seg3", "tee", "net"})

# --------------------------------------------------------------------------
# U1 — capped ring band

RING_BAND_CAPACITY = (3.0, 4.0)          # designated AND reachable, both gated
RING_BAND_DRAW = (3.15, 3.85)            # requested factor, leaves gate headroom
RING_BAND_MAX_RADIUS = 26.0              # a band, not a halo across the map

# --------------------------------------------------------------------------
# U2 — direct service, measured with the live envelope

DIRECT_SERVICE_MIN = 0.99
DIRECT_SERVICE_P95_MAX = 14.0            # p95 dig->nearest-dump on T0 aprons
APRON_STANDOFF_CHOICES = (2, 3, 4, 5, 6, 7, 8)
STANDOFF_CHOICES = (2, 3, 4, 5, 6)

# --------------------------------------------------------------------------
# U5 / capacity bands

APRON_CAPACITY_BANDS = {
    "c3x": (2.90, 3.10),
    "c2x": (1.90, 2.10),
    "c1p6": (1.55, 1.75),
    "c1p2": (1.15, 1.25),
}
TIGHT_CAPACITY_BAND = (1.30, 1.65)
GENEROUS_CAPACITY_BANDS = {
    "capacity_apron": (2.00, 2.60),       # the U2 direct-service apron
    "distance_apron": (2.00, 2.60),       # U6 ladder: generous by design
    "trench_band": (2.50, 3.25),          # both flanks
    "trench_flank": (2.00, 2.60),
    "trench_altsides": (2.00, 2.40),
    "one_side_near": (2.00, 2.60),
    "separated_zones": (2.00, 2.60),
    "haul_away_edge": (2.00, 2.60),
}
WALL_SPLIT_CAPACITY_BAND = (2.00, 2.30)
TRENCH_WALL_CAPACITY_BAND = (2.00, 2.15)
GENEROUS_REACHABLE_FLOOR = 2.00

# --------------------------------------------------------------------------
# U6 — distance ladder

DISTANCE_BINS = {"d12": 12.0, "d16": 16.0, "d20": 20.0, "d24": 24.0}
DISTANCE_BIN_TOLERANCE = 2.0             # realised median within +-2 tiles

# --------------------------------------------------------------------------
# U4 — overlap stressor

SPOIL_FLANK_TILES = 6                    # the natural adjacent spoil band
ARM_OVERLAP_MIN = 0.15
# A fold-back puts two arms close together on purpose; this keeps them readable
# as two arms instead of collapsing into one blob at the vertex.
ARM_MERGE_MAX = 0.40
ALTSIDES_MIN_SPAN = 8.0
# The dig may not be more than this much larger than the union of its arms.
ARM_FILL_MAX = 1.12

# --------------------------------------------------------------------------
# U7 — lattice, rasterisation, lane, drift

TRENCH_AXES_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)   # {k * 30 deg}
# The base is 7-9 tiles across its direction of travel once rasterised at a
# lattice heading, and 3 backward steps accumulate up to 1.8 tiles of drift on
# the four non-cardinal axes, so 13 tiles is what "wide enough for the base"
# actually costs.
LANE_WIDTH_CHOICES = (15, 13, 11)
LANE_WIDTH_TILES = LANE_WIDTH_CHOICES[0]
# 3 in-lane retreat steps need the drift margin a 13-tile lane gives; a
# narrower fallback lane still carries the base, so it keeps 1 step.
LANE_WIDTH_FOR_FULL_RETREAT = 15
LANE_GAP_CHOICES = (6, 7, 8)
LANE_END_MARGIN = 5.0
# The residual per-step rounding drift is an axis property (see
# terra_service.backward_drive_check): 0 on 0/90 deg, up to ~0.135 tiles per
# tile of travel on the other four lattice axes. The map-specific gates are the
# footprint-clear run and the in-lane run; the rate gate is what a 15 deg
# off-lattice axis would fail outright.
BACKWARD_DRIFT_PER_TILE_MAX = 0.15
BACKWARD_LANE_STEPS_MIN = 3

# --------------------------------------------------------------------------
# carried forward from §8.1 / §8.2

DIG_IOU_MAX = 0.60
# U7(a) cuts the trench axis lattice from twelve 15 deg steps to six 30 deg
# ones, which halves the shape freedom of a straight trench: at one heading two
# trenches are distinguishable only by length and width, and 16 mutually
# sub-0.60 shapes do not exist (measured: the bank exhausts at slot 12).
# Trench levels therefore use a looser dissimilarity bound; the realised worst
# pair is reported per level.
TRENCH_DIG_IOU_MAX = 0.70
T0_DIG_COVERAGE_MIN = 0.99
SPLIT_MAX_ANGULAR_GAP_MIN = 150.0
SPLIT_PAD_BORDER_MARGIN = v7.SPLIT_PAD_BORDER_MARGIN
SPLIT_PAD_EDGE_GAP_MIN = v7.SPLIT_PAD_EDGE_GAP_MIN
SPLIT_PAD_ANGULAR_SPAN_MIN = v7.SPLIT_PAD_ANGULAR_SPAN_MIN
TRENCH_BRANCH_SPINE_MIN = 0.35
# Junction variety now lives in the branch ANGLES, not in which side of the
# spine the branch leaves on (see GeometryFactoryV8.net). `double_T` is both
# branches square to the spine; the skew classes are the >= 50% non-double_T
# share that §8.1 asks for.
NET_TURN_CYCLE = {
    "double_T": (90.0, 90.0),
    "skew_open": (60.0, 120.0),
    "skew_lead": (60.0, 90.0),
    "skew_trail": (90.0, 120.0),
    "skew_pair": (120.0, 60.0),
}
NET_TOPOLOGY_CYCLE = (
    "skew_open", "double_T", "skew_lead", "skew_pair", "double_T", "skew_trail",
)
REMOTE_MIN_DISTANCE_TILES = 15.0
WALL_DETOUR_MIN = 1.15
# U3 caps how much travel a trench wall can add: the banks are alongside the
# arm they serve, so only the fenced side detours at all. Measured, reported,
# and gated well below the foundation number — see GENERATION_NOTES.
TRENCH_WALL_DETOUR_MIN = 1.0   # REPORTED, not gated — see GENERATION_NOTES
TRENCH_WALL_DETOUR_CONDITION_MEDIAN_MIN = 1.0
WALL_DETOUR_CONDITION_MEDIAN_MIN = 1.25
WALL_GAP_OFFAXIS_MIN = 8.0
WALL_MIN_THICKNESS_TILES = 3
WALL_MIN_COMPONENTS = 2
WALL_MIN_COMPONENT_CELLS = 25
WALL_MIN_COMPONENT_EXTENT = 8
FENCE_INNER_CHOICES = v7.FENCE_INNER_CHOICES
TRENCH_FENCE_INNER_CHOICES = (3.0, 4.0, 5.0)
TRENCH_FENCE_THICKNESS_CHOICES = (3.0,)
FENCE_THICKNESS_CHOICES = v7.FENCE_THICKNESS_CHOICES
OBJECT_ANNULUS_BLOCK_MIN = v7.OBJECT_ANNULUS_BLOCK_MIN
OBJECT_NEAR_ANNULUS_FREE_MIN = v7.OBJECT_NEAR_ANNULUS_FREE_MIN
OBJECT_FOOTPRINT_PROFILE = v7.OBJECT_FOOTPRINT_PROFILE
ROAD_ANNULUS_FREE_MIN = v7.ROAD_ANNULUS_FREE_MIN
ROAD_DIG_ANNULUS_STERILIZE_RANGE = v7.ROAD_DIG_ANNULUS_STERILIZE_RANGE
ROAD_SIGHTLINE_CROSS_MIN = v7.ROAD_SIGHTLINE_CROSS_MIN
ROAD_DUMP_ANNULUS_STERILIZE_MIN = v7.ROAD_DUMP_ANNULUS_STERILIZE_MIN
ROAD_RADIUS_RANGE_RING = v7.ROAD_RADIUS_RANGE_RING
ROAD_RADIUS_RANGE_ZONED = v7.ROAD_RADIUS_RANGE_ZONED
# U1 makes the ring dump finite, so the ring road must sterilise part of the
# BAND, not just the dig annulus. Measured against the band the same map would
# have grown with no road on it.
ROAD_BAND_STERILIZE_MIN = 0.10

FOUNDATION_MEDIAN_LIMITS = {
    "capacity_apron": 13.0,
    "one_side_near": 15.0,
    "separated_zones": 17.0,
}
TRENCH_MEDIAN_LIMITS = {
    "trench_band": 7.0,
    "trench_flank": 8.0,
    "trench_altsides": 12.0,
}
WALL_SPLIT_MEDIAN_LIMIT = 26.0
TRENCH_WALL_MEDIAN_LIMIT = 16.0

PLACEMENT_RADII = v7.PLACEMENT_RADII
# U7(c) reserves an 11-tile lane beyond the dig's lateral extent, and a `net`
# branch already reaches ~14 tiles off the spine. Keeping the v3 foundation
# schedule for trenches would push that lane off a 64x64 map on most maps, so
# trenches get their own (still spread) centroid schedule.
TRENCH_PLACEMENT_RADII = (1.5, 5.0, 8.0, 3.0, 9.5, 6.0, 0.0, 7.0)
# A `net` branch reaches ~10 tiles off the spine on BOTH sides, and the lane has
# to clear that plus its own 13 tiles: 64x64 only has room for it when the
# target sits near the middle. Reported as a deviation, not hidden.
TRENCH_PLACEMENT_RADII_WIDE = (0.0, 2.5, 4.0, 1.5, 3.0, 2.0, 0.0, 3.5)
WIDE_TRENCH_LEVELS = frozenset({"net", "tee"})
# U7(c) has to clear every arm before the lane even starts, so a branch that
# reaches far off the spine makes the lane unplaceable on 64x64. The dig bank
# rejects those outright instead of re-rolling them at map level.
TRENCH_MAX_LATERAL_TILES = 10.0
FOUNDATION_PLACEMENT_MARGIN = v7.FOUNDATION_PLACEMENT_MARGIN
TRENCH_PLACEMENT_MARGIN = v7.TRENCH_PLACEMENT_MARGIN
SLAB_MAX_DIM_V7 = v7.SLAB_MAX_DIM_V7
APRON_SECTOR_DEGREES = v7.APRON_SECTOR_DEGREES        # 200 deg, capacity ladder
# U2: a 200 deg apron leaves the far side of a 15-tile slab 15-20 tiles from
# any designated dump cell, which is exactly the `fnd-slab-apron-c7x` defect
# Lorenzo flagged. The direct-service apron and the distance ladder wrap
# further so the p95 haul stays inside the service envelope.
APRON_WRAP_SECTOR_DEGREES = 260.0

# reuse of v7 helpers, re-exported so the validator has one import
stable_key = v7.stable_key
sha256_mask = v7.sha256_mask
rng_from = v7.rng_from
border_margin = v7.border_margin
centroid = v7.centroid
centroid_offset = v7.centroid_offset
place_at = v7.place_at
centred_iou = v7.centred_iou
dig_only_coverage = v7.dig_only_coverage
annulus = v7.annulus
dump_annulus = v7.dump_annulus
max_angular_gap_degrees = v7.max_angular_gap_degrees
sight_line_bite = v7.sight_line_bite
geodesic_from = v7.geodesic_from
haul_detour = v7.haul_detour
wall_min_thickness = v7.wall_min_thickness
wall_component_stats = v7.wall_component_stats
gap_offaxis_tiles = v7.gap_offaxis_tiles
pad_separation = v7.pad_separation
border_sides_touched = v7.border_sides_touched
object_intrusion = v7.object_intrusion
place_objects_v7 = v7.place_objects_v7
make_two_border_road = v7.make_two_border_road
zoned_road_metrics = v7.zoned_road_metrics
zoned_road_bites = v7.zoned_road_bites
make_offset_fence_wall = v7.make_offset_fence_wall
_interior = v7._interior
_polar = v7._polar


# --------------------------------------------------------------------------
# condition table


@dataclass(frozen=True)
class ConditionSpec:
    id: str
    anchor: str | None
    preview: bool
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

    @property
    def geometry(self) -> str:
        return GEOMETRY_LEVEL_SOURCE[self.geometry_level]

    @property
    def n_maps(self) -> int:
        return PREVIEW_MAPS_PER_CONDITION if self.preview else MAPS_PER_CONDITION

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
        return tax.tier(self.levels, tax.V4_RELEASE)

    @property
    def group(self) -> str:
        return self.layout_group or self.id


def _fnd(condition_id: str, anchor: str | None, **kwargs: Any) -> ConditionSpec:
    return ConditionSpec(condition_id, anchor, family="foundation", **kwargs)


def _trn(condition_id: str, anchor: str | None, **kwargs: Any) -> ConditionSpec:
    return ConditionSpec(condition_id, anchor, family="trench", **kwargs)


APRON_CAPACITY_GROUP = "slab-apron-capacity"
APRON_DISTANCE_GROUP = "slab-apron-distance"

CONDITIONS: tuple[ConditionSpec, ...] = (
    # ---- foundation T0: capped rings (U1) + direct-service apron (U2) -----
    _fnd("fnd-slab-ring3x", None, preview=False, geometry_level="slab",
         dump_level="ring3x", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-proc-ring3x", None, preview=False, geometry_level="proc",
         dump_level="ring3x", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-lg-ring3x", None, preview=False, geometry_level="slab-lg",
         dump_level="ring3x", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-apron-near", None, preview=False, geometry_level="slab",
         dump_level="apron", capacity_level="generous", site_level="clean",
         distance_level="near", dump_style="capacity_apron",
         dump_layout="near_apron_large",
         pair_group=APRON_DISTANCE_GROUP, layout_group=APRON_DISTANCE_GROUP),
    # ---- U5 capacity ladder ----------------------------------------------
    _fnd("fnd-slab-apron-c3x", None, preview=False, geometry_level="slab",
         dump_level="apron", capacity_level="c3x", site_level="clean",
         distance_level="unspec", dump_style="capacity_apron",
         dump_layout="near_apron_large",
         pair_group=APRON_CAPACITY_GROUP, layout_group=APRON_CAPACITY_GROUP),
    _fnd("fnd-slab-apron-c2x", "fnd-slab-apron-c3x", preview=False,
         geometry_level="slab", dump_level="apron", capacity_level="c2x",
         site_level="clean", distance_level="unspec",
         dump_style="capacity_apron", dump_layout="near_apron_large",
         pair_group=APRON_CAPACITY_GROUP, layout_group=APRON_CAPACITY_GROUP),
    _fnd("fnd-slab-apron-c1p6", "fnd-slab-apron-c3x", preview=False,
         geometry_level="slab", dump_level="apron", capacity_level="c1p6",
         site_level="clean", distance_level="unspec",
         dump_style="capacity_apron", dump_layout="near_apron_large",
         pair_group=APRON_CAPACITY_GROUP, layout_group=APRON_CAPACITY_GROUP),
    _fnd("fnd-slab-apron-c1p2", "fnd-slab-apron-c3x", preview=False,
         geometry_level="slab", dump_level="apron", capacity_level="c1p2",
         site_level="clean", distance_level="unspec",
         dump_style="capacity_apron", dump_layout="near_apron_large",
         pair_group=APRON_CAPACITY_GROUP, layout_group=APRON_CAPACITY_GROUP),
    # ---- U6 distance ladder ----------------------------------------------
    *[
        _fnd(f"fnd-slab-apron-{bin_name}", "fnd-slab-apron-near", preview=False,
             geometry_level="slab", dump_level="apron",
             capacity_level="generous", site_level="clean",
             distance_level=bin_name, dump_style="distance_apron",
             dump_layout="near_apron_large",
             pair_group=APRON_DISTANCE_GROUP, layout_group=APRON_DISTANCE_GROUP)
        for bin_name in ("d12", "d16", "d20", "d24")
    ],
    # ---- layout / geometry / site off the capped ring ---------------------
    _fnd("fnd-slab-side1", "fnd-slab-ring3x", preview=False,
         geometry_level="slab", dump_level="side1", capacity_level="generous",
         site_level="clean", distance_level="unspec",
         dump_style="one_side_near", dump_layout="one_side_near"),
    _fnd("fnd-slab-split", "fnd-slab-ring3x", preview=False,
         geometry_level="slab", dump_level="split", capacity_level="generous",
         site_level="clean", distance_level="unspec",
         dump_style="separated_zones", dump_layout="separated_zones"),
    _fnd("fnd-strips-ring3x", "fnd-slab-ring3x", preview=False,
         geometry_level="strips", dump_level="ring3x",
         capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-ring3x-obj1", "fnd-slab-ring3x", preview=False,
         geometry_level="slab", dump_level="ring3x", capacity_level="generous",
         site_level="obj1", distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-ring3x-obj", "fnd-slab-ring3x", preview=False,
         geometry_level="slab", dump_level="ring3x", capacity_level="generous",
         site_level="obj", distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-slab-ring3x-road", "fnd-slab-ring3x", preview=False,
         geometry_level="slab", dump_level="ring3x", capacity_level="generous",
         site_level="road", distance_level="unspec", dump_style="ring_band",
         dump_layout="capped_ring_band"),
    _fnd("fnd-proc-side1-road", "fnd-proc-ring3x", preview=False,
         geometry_level="proc", dump_level="side1", capacity_level="generous",
         site_level="road", distance_level="unspec",
         dump_style="one_side_near", dump_layout="one_side_near"),
    _fnd("fnd-strips-split-wall", "fnd-slab-ring3x", preview=False,
         geometry_level="strips", dump_level="split",
         capacity_level="generous", site_level="wall",
         distance_level="unspec", dump_style="separated_zones",
         dump_layout="separated_zones"),
    # `remote` IS the far haul, so its distance level stays `unspec`: U6's
    # ladder is the controlled distance axis and it must not double-count.
    _fnd("fnd-slab-remote", "fnd-slab-ring3x", preview=True,
         geometry_level="slab", dump_level="remote", capacity_level="generous",
         site_level="clean", distance_level="unspec",
         dump_style="haul_away_edge", dump_layout="haul_away_edge"),
    # ---- trenches: adjacent dumping (U3), lattice axes (U7) ---------------
    _trn("trn-straight-side2", None, preview=False, geometry_level="straight",
         dump_level="side2", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="trench_band",
         dump_layout="easy_surround"),
    _trn("trn-straight-side1", None, preview=False, geometry_level="straight",
         dump_level="side1", capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="trench_flank",
         dump_layout="near_apron_large",
         pair_group="trn-one-side", layout_group="trn-one-side"),
    _trn("trn-straight-side1-tight", "trn-straight-side1", preview=False,
         geometry_level="straight", dump_level="side1", capacity_level="tight",
         site_level="clean", distance_level="unspec",
         dump_style="trench_flank", dump_layout="near_apron_large",
         pair_group="trn-one-side", layout_group="trn-one-side"),
    _trn("trn-straight-altsides", "trn-straight-side2", preview=False,
         geometry_level="straight", dump_level="altsides",
         capacity_level="generous", site_level="clean",
         distance_level="unspec", dump_style="trench_altsides",
         dump_layout="alternating_sides"),
    _trn("trn-seg2-side2", "trn-straight-side2", preview=False,
         geometry_level="seg2", dump_level="side2", capacity_level="generous",
         site_level="clean", distance_level="unspec", dump_style="trench_band",
         dump_layout="easy_surround"),
    _trn("trn-seg3-side2", "trn-straight-side2", preview=False,
         geometry_level="seg3", dump_level="side2", capacity_level="generous",
         site_level="clean", distance_level="unspec", dump_style="trench_band",
         dump_layout="easy_surround"),
    _trn("trn-tee-side2", "trn-straight-side2", preview=False,
         geometry_level="tee", dump_level="side2", capacity_level="generous",
         site_level="clean", distance_level="unspec", dump_style="trench_band",
         dump_layout="easy_surround"),
    _trn("trn-net-side2", "trn-straight-side2", preview=False,
         geometry_level="net", dump_level="side2", capacity_level="generous",
         site_level="clean", distance_level="unspec", dump_style="trench_band",
         dump_layout="easy_surround"),
    _trn("trn-tee-side1-road", "trn-straight-side1", preview=False,
         geometry_level="tee", dump_level="side1", capacity_level="generous",
         site_level="road", distance_level="unspec", dump_style="trench_flank",
         dump_layout="near_apron_large"),
    _trn("trn-net-altsides-wall", "trn-straight-side2", preview=False,
         geometry_level="net", dump_level="altsides",
         capacity_level="generous", site_level="wall",
         distance_level="unspec", dump_style="trench_altsides",
         dump_layout="alternating_sides"),
)

WALL_CONDITIONS = frozenset({"fnd-strips-split-wall"})
TRENCH_WALL_CONDITIONS = frozenset({"trn-net-altsides-wall"})
# §8.3 U7(c) asks for a straight base-width lane clear of every arm over the
# full working length; §8.3 U3 keeps the banks alongside the trench; §8.2 wants
# a >= 3-tile gapped wall between the trench and part of its spoil. On 64x64
# those three cannot hold at once for a 3-axis trench — measured, see
# GENERATION_NOTES: fence-on-the-lane-side pushes the lane off the map on
# 720/720 attempts, fence-on-the-far-side starves the banks on 674/960. The
# wall condition is exempted from the LANE gate (a wall is precisely what
# removes a straight corridor); every other U7 gate still applies to it.
LANE_EXEMPT_CONDITIONS = frozenset({"trn-net-altsides-wall"})
# §8.3 says to STOP and report rather than silently relax when a U-gate turns
# out to be unsatisfiable. `trn-net-altsides-wall` is that case: U3 puts every
# bank beside the arm it serves, U7 caps the net's lateral extent at 10 tiles so
# a lane can exist at all, and §8.2 wants a >= 3-tile gapped wall that costs
# travel. Five constructions were measured (see GENERATION_NOTES §7); none
# produced a single accepted map in 320 attempts/map. The condition stays in the
# taxonomy table and is reported here as blocked, with its rejection census.
BLOCKED_CONDITIONS = frozenset({"trn-net-altsides-wall"})
ROAD_FIRST_ZONED = frozenset({"fnd-proc-side1-road", "trn-tee-side1-road"})
T0_CONDITIONS = frozenset(c.id for c in CONDITIONS if c.tier == 0)
T0_GEOMETRY_LEVELS = frozenset(c.geometry_level for c in CONDITIONS if c.tier == 0)


def assert_conditions_match_taxonomy() -> None:
    """The generator table and `scripts/curriculum_taxonomy.py` must agree."""
    spec = tax.V4_RELEASE
    table = spec.condition_table
    assert len(CONDITIONS) == 31, f"expected 31 conditions, got {len(CONDITIONS)}"
    assert {c.id for c in CONDITIONS} == set(table), (
        f"condition set mismatch: {sorted({c.id for c in CONDITIONS} ^ set(table))}"
    )
    for condition in CONDITIONS:
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
        ) or condition.distance_level == "far"
        assert spec.geometry_levels[condition.geometry] == condition.geometry_level
    previews = {c.id for c in CONDITIONS if c.preview}
    assert previews == {"fnd-slab-remote"}, previews
    # the anchor/delta lattice, derived exactly as the site derives it
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
            for c in CONDITIONS
        ],
        spec,
    )


# --------------------------------------------------------------------------
# U7 — trench rasterisation on the 30 deg lattice


def rasterize_segments(
    points_yx: np.ndarray, half_width: float
) -> np.ndarray:
    """Constant-perpendicular-width rasterisation of a polyline.

    §8.3 U7(b). ``base.rasterize_polyline`` dilates a Bresenham line with a
    disk, which is width-stable in the middle but leaves single-cell notches at
    the staircase steps of an off-axis line. A rotated rectangle per segment is
    a genuinely constant-width band; the vertices get a disk so the joins are
    not mitred away.
    """
    out = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    width = 2.0 * half_width + 1.0
    for start, end in zip(points_yx[:-1], points_yx[1:]):
        delta = end - start
        length = float(np.hypot(*delta))
        if length < 1e-6:
            continue
        # `rotated_rectangle` lays its length axis along (sin a, cos a) in
        # (row, col), which is the same convention the trench headings use.
        angle = math.atan2(delta[0], delta[1])
        centre = (float((start[0] + end[0]) / 2), float((start[1] + end[1]) / 2))
        out |= base.rotated_rectangle(centre, length, width, angle)
    radius = int(round(half_width))
    if radius >= 1:
        for point in points_yx:
            joint = np.zeros_like(out)
            y, x = int(round(point[0])), int(round(point[1]))
            if 0 <= y < MAP_SIZE and 0 <= x < MAP_SIZE:
                joint[y, x] = True
                out |= ndi.binary_dilation(joint, structure=base.binary_disk(radius))
    return out


def _neighbour_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.int8), 1)
    return (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )


def edge_irregularity(mask: np.ndarray) -> tuple[int, int]:
    """(single-cell spurs, single-cell notches) — §8.3 U7(b)."""
    counts = _neighbour_count(mask)
    spurs = int((mask & (counts <= 1)).sum())
    notches = int((~mask & (counts >= 3)).sum())
    return spurs, notches


def regularise_edges(mask: np.ndarray, rounds: int = 6) -> np.ndarray:
    """Fill single-cell notches and shave single-cell spurs until stable."""
    out = mask.copy()
    for _ in range(rounds):
        counts = _neighbour_count(out)
        changed = False
        notches = ~out & (counts >= 3)
        if notches.any():
            out |= notches
            changed = True
        counts = _neighbour_count(out)
        spurs = out & (counts <= 1)
        if spurs.any():
            out &= ~spurs
            changed = True
        if not changed:
            break
    return base.largest_component(out)


# --------------------------------------------------------------------------
# geometry factory


class GeometryFactoryV8(v7.GeometryFactoryV7):
    """v7 foundations, plus lattice-restricted / segmented trenches."""

    # ---- trenches --------------------------------------------------------

    @staticmethod
    def _finish_trench(
        dig: np.ndarray,
        segments: list[np.ndarray],
        half_width: float,
        heading_deg: float,
        radius: float,
        angle: float,
        cells: tuple[int, int],
        extra: dict[str, Any],
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        dig = regularise_edges(dig)
        if not dig.any() or not cells[0] <= int(dig.sum()) <= cells[1]:
            return None, {}
        spurs, notches = edge_irregularity(dig)
        if spurs or notches:
            return None, {}
        placed = place_at(dig, radius, angle, TRENCH_PLACEMENT_MARGIN)
        if placed is None:
            return None, {}
        _, _, _, _, lateral = trench_frame(placed, heading_deg)
        if float(np.abs(lateral[placed]).max()) > TRENCH_MAX_LATERAL_TILES:
            return None, {}
        # record the arms in the PLACED frame so the validator can rasterise
        # them and measure the U4 overlap from the exported dig
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
        return placed, meta

    def straight(self, rng, radius, angle, heading_deg, _topology=None):
        heading = math.radians(heading_deg)
        direction = np.array([math.sin(heading), math.cos(heading)])
        for _ in range(80):
            length = float(rng.uniform(20, 44))
            half_width = float(rng.choice([1.0, 2.0], p=[0.55, 0.45]))
            points = (
                np.vstack([-direction * length / 2, direction * length / 2])
                + MAP_CENTRE
            )
            dig = rasterize_segments(points, half_width)
            placed, meta = self._finish_trench(
                dig, [points], half_width, heading_deg, radius, angle, (55, 250),
                {
                    "trench_axes_count": 1,
                    "intersection_junctions": 0,
                    "intersection_branches": 2,
                    "trench_topology": "straight",
                    "trench_spine_length_tiles": round(length, 2),
                    "trench_branch_spine_ratio": 1.0,
                },
            )
            if placed is not None:
                return placed, meta
        return None, {}

    def _segmented(self, rng, radius, angle, heading_deg, n_segments):
        """§8.3 U4: a hook (seg2) / staple (seg3) that folds back on itself.

        A shallow dog-leg does NOT create the ordering problem U4 asks for: two
        arms leaving a vertex at 30-60 deg separate at 1.7-1.9 tiles per tile of
        length, so only the few tiles nearest the vertex are inside the other
        arm's spoil flank (measured: worst-arm overlap 0.09-0.17, gate 0.15).
        A fold-back does: seg2 turns 120-150 deg into a hook, seg3 turns twice
        the SAME way into a staple whose two legs run 8-13 tiles apart, well
        inside the spoil reach. Every segment heading stays on the 30 deg
        lattice, so U7(a) holds for the whole polyline, not just the first axis.
        """
        turn_sign = float(rng.choice([-1.0, 1.0]))
        for _ in range(90):
            half_width = float(rng.choice([1.0, 2.0], p=[0.7, 0.3]))
            heading = heading_deg
            point = np.array([MAP_CENTRE, MAP_CENTRE], dtype=float)
            points = [point.copy()]
            headings = [heading]
            lengths = []
            for index in range(n_segments):
                if n_segments == 3 and index == 1:
                    length = float(rng.uniform(8, 13))     # the staple's back
                else:
                    length = float(rng.uniform(14, 23))
                lengths.append(length)
                direction = np.array(
                    [math.sin(math.radians(heading)), math.cos(math.radians(heading))]
                )
                point = point + direction * length
                points.append(point.copy())
                if index < n_segments - 1:
                    # seg3 turns square twice into a staple. 120 deg twice
                    # folds the third arm back over the first and the notch
                    # regularisation then closes the interior, which renders as
                    # a solid wedge instead of three segments (6/16 on the
                    # first full run).
                    turn = (
                        90.0
                        if n_segments == 3
                        else float(rng.choice([120.0, 150.0]))
                    )
                    heading = (heading + turn_sign * turn) % 360.0
                    headings.append(heading)
            polyline = np.asarray(points)
            polyline = polyline - polyline.mean(axis=0) + MAP_CENTRE
            dig = rasterize_segments(polyline, half_width)
            segments = [polyline[i : i + 2] for i in range(n_segments)]
            placed, meta = self._finish_trench(
                dig, segments, half_width, heading_deg, radius, angle, (80, 280),
                {
                    "trench_axes_count": n_segments,
                    "intersection_junctions": 0,
                    "intersection_branches": 2,
                    "trench_topology": f"seg{n_segments}",
                    "trench_segment_headings_deg": [round(h % 180.0, 1) for h in headings],
                    "trench_spine_length_tiles": round(float(sum(lengths)), 2),
                    "trench_branch_spine_ratio": round(
                        float(min(lengths) / max(lengths)), 4
                    ),
                },
            )
            if placed is None:
                continue
            if arm_overlap_fraction(placed, meta)[0] < ARM_OVERLAP_MIN:
                continue
            if arm_merge_fraction(placed, meta) > ARM_MERGE_MAX:
                continue
            # The polyline must still READ as segments: notch-filling between
            # arms that fold back on each other silently turns a staple into a
            # solid wedge, and no per-arm statistic notices.
            arms = arm_masks(placed, meta)
            union = np.zeros_like(placed)
            for arm in arms:
                union |= arm
            if int(placed.sum()) > ARM_FILL_MAX * max(1, int(union.sum())):
                continue
            return placed, meta
        return None, {}

    def seg2(self, rng, radius, angle, heading_deg, _topology=None):
        return self._segmented(rng, radius, angle, heading_deg, 2)

    def seg3(self, rng, radius, angle, heading_deg, _topology=None):
        return self._segmented(rng, radius, angle, heading_deg, 3)

    def tee(self, rng, radius, angle, heading_deg, _topology=None):
        heading = math.radians(heading_deg)
        direction = np.array([math.sin(heading), math.cos(heading)])
        for _ in range(110):
            spine = float(rng.uniform(26, 36))
            branch = float(rng.uniform(TRENCH_BRANCH_SPINE_MIN + 0.02, 0.50)) * spine
            junction_along = float(rng.uniform(-0.14, 0.14)) * spine
            # the branch leaves the spine on the lattice, not at a fixed 90 deg
            turn = float(rng.choice([60.0, 90.0, 120.0]))
            branch_heading = (heading_deg + turn) % 360.0
            branch_direction = np.array(
                [
                    math.sin(math.radians(branch_heading)),
                    math.cos(math.radians(branch_heading)),
                ]
            )
            half_width = float(rng.choice([1.0, 2.0], p=[0.7, 0.3]))
            main = np.vstack([-direction * spine / 2, direction * spine / 2]) + MAP_CENTRE
            junction = direction * junction_along + MAP_CENTRE
            branch_points = np.vstack([junction, junction + branch_direction * branch])
            dig = rasterize_segments(main, half_width)
            dig |= rasterize_segments(branch_points, half_width)
            placed, meta = self._finish_trench(
                dig, [main, branch_points], half_width, heading_deg, radius, angle,
                (95, 300),
                {
                    "trench_axes_count": 2,
                    "intersection_junctions": 1,
                    "intersection_branches": 3,
                    "trench_topology": "T",
                    "trench_relative_angle_deg": min(turn, 180.0 - turn) or turn,
                    "trench_spine_length_tiles": round(spine, 2),
                    "trench_branch_spine_ratio": round(branch / spine, 4),
                },
            )
            if placed is None:
                continue
            if arm_overlap_fraction(placed, meta)[0] < ARM_OVERLAP_MIN:
                continue
            return placed, meta
        return None, {}

    def net(self, rng, radius, angle, heading_deg, topology):
        """A comb: a spine with two junctions whose branches share one side.

        v3 let the branches straddle the spine (`H`, `T_plus_X`). Under U7(c)
        that is unbuildable: the lane has to clear EVERY arm before it starts,
        and a branch reaching >= 0.35 * spine off both sides plus an 11-13 tile
        lane plus its own length does not fit on 64x64 on a diagonal axis
        (measured: `lane_off_map` on 320-620 of 640 attempts). Junction variety
        moves from the SIDE to the ANGLE: the two branches leave at 60/90/120
        deg, so `double_T` (both at 90) is still a minority of the level and the
        >= 2 junctions, >= 0.35 branch:spine gates are untouched.
        """
        heading = math.radians(heading_deg)
        direction = np.array([math.sin(heading), math.cos(heading)])
        for _ in range(130):
            spine = float(rng.uniform(20, 27))
            main = np.vstack([-direction * spine / 2, direction * spine / 2]) + MAP_CENTRE
            positions = np.array([-0.26, 0.26]) + rng.uniform(-0.03, 0.03, size=2)
            half_width = float(rng.choice([1.0, 2.0], p=[0.7, 0.3]))
            dig = rasterize_segments(main, half_width)
            segments = [main]
            ratios = []
            turns = NET_TURN_CYCLE[topology]
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
                dig, segments, half_width, heading_deg, radius, angle, (100, 205),
                {
                    "trench_axes_count": 3,
                    "intersection_junctions": 2,
                    "intersection_double_sided": 0,
                    "intersection_branches": 4,
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
            return placed, meta
        return None, {}


GEOMETRY_BUILDER = {
    "slab": "slab",
    "slab-lg": "slab_lg",
    "proc": "proc",
    "strips": "strips",
    "straight": "straight",
    "seg2": "seg2",
    "seg3": "seg3",
    "tee": "tee",
    "net": "net",
}


# --------------------------------------------------------------------------
# U4 — the overlap stressor, recomputable from the exported arrays


def _as_arms(value: Any) -> list[list[list[float]]]:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def arm_masks(dig: np.ndarray, dig_meta: dict[str, Any]) -> list[np.ndarray]:
    half_width = float(dig_meta["trench_half_width_tiles"])
    arms = _as_arms(dig_meta["trench_arms"])
    return [
        rasterize_segments(np.asarray(arm, dtype=float), half_width) & dig
        for arm in arms
    ]


def arm_flank_band(
    dig: np.ndarray, arm: np.ndarray, points: np.ndarray, side: float
) -> np.ndarray:
    """One side of an arm's natural spoil flank: within reach, on that flank.

    A spoil flank is one-sided — the machine works a trench from one bank and
    throws the spoil to that bank. Measuring the full surround would dilute the
    statistic with the ends and the opposite bank, neither of which is where the
    spoil goes.
    """
    distance = ndi.distance_transform_edt(~arm)
    band = (distance >= 1) & (distance <= SPOIL_FLANK_TILES)
    direction = np.asarray(points[-1], dtype=float) - np.asarray(points[0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.zeros_like(dig, dtype=bool)
    direction = direction / norm
    normal = np.array([direction[1], -direction[0]])
    centre = np.argwhere(arm).astype(float).mean(axis=0)
    yy, xx = np.indices(dig.shape)
    lateral = (yy - centre[0]) * normal[0] + (xx - centre[1]) * normal[1]
    return band & (side * lateral > 0)


def arm_overlap_fraction(
    dig: np.ndarray, dig_meta: dict[str, Any]
) -> tuple[float, list[float]]:
    """How much of one arm's natural spoil flank lands on another arm's dig.

    §8.3 U4. The flank is the ``<= SPOIL_FLANK_TILES`` band on ONE bank of an
    arm — the region a machine working that arm would naturally throw spoil
    into. The fraction of it that is another arm's excavation is the fraction of
    greedy dumping that buries future work and forces re-handling.
    """
    arms = arm_masks(dig, dig_meta)
    points = _as_arms(dig_meta["trench_arms"])
    if len(arms) < 2:
        return 0.0, []
    fractions = []
    for index, arm in enumerate(arms):
        if not arm.any():
            fractions.append(0.0)
            continue
        others = dig & ~arm
        best = 0.0
        for side in (-1.0, 1.0):
            flank = arm_flank_band(dig, arm, np.asarray(points[index], float), side)
            if int(flank.sum()) < 20:
                continue
            best = max(best, float((flank & others).sum() / int(flank.sum())))
        fractions.append(best)
    return float(max(fractions)), [round(value, 4) for value in fractions]


def arm_merge_fraction(dig: np.ndarray, dig_meta: dict[str, Any]) -> float:
    """Largest share of one arm that another arm's raster also covers."""
    arms = arm_masks(dig, dig_meta)
    worst = 0.0
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            smaller = min(int(arms[i].sum()), int(arms[j].sum()))
            if smaller:
                worst = max(worst, float((arms[i] & arms[j]).sum() / smaller))
    return worst


# --------------------------------------------------------------------------
# dig bank


class DigBankV8(v7.DigBankV7):
    def _headings(self, level: str) -> list[float]:
        """§8.3 U7(a): trench axes come only from the drivable 30 deg lattice."""
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
        if level in T0_GEOMETRY_LEVELS:
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
    lane_gap: int
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
            "layout_lane_gap_tiles": self.lane_gap,
            "layout_alternations": self.alternations,
        }


def layout_for(condition: ConditionSpec, map_index: int) -> Layout:
    group = condition.group
    rng = rng_from(SEED_BASE, 5150, stable_key(group), map_index)
    wall = condition.id in WALL_CONDITIONS
    # U2: the apron standoff is randomised over a wider envelope than v3's 2-6.
    if condition.dump_style in ("capacity_apron", "distance_apron"):
        standoff = int(rng.choice(APRON_STANDOFF_CHOICES))
    else:
        standoff = int(rng.choice(STANDOFF_CHOICES))
    if wall:
        n_zones = 2
        span = float(rng.uniform(92.0, 124.0))
    else:
        n_zones = int(rng.choice([2, 2, 2, 3]))
        span = float(rng.uniform(90.0, 130.0) if n_zones == 2 else rng.uniform(110.0, 130.0))
    sector = (
        APRON_WRAP_SECTOR_DEGREES
        if group == APRON_DISTANCE_GROUP
        else APRON_SECTOR_DEGREES
    )
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
        lane_gap=int(rng.choice(LANE_GAP_CHOICES)),
        alternations=int(rng.choice([2, 3, 3, 4])),
    )


# --------------------------------------------------------------------------
# U7(c) — the working lane


def trench_frame(dig: np.ndarray, heading_deg: float):
    heading = math.radians(heading_deg)
    direction = np.array([math.sin(heading), math.cos(heading)])
    normal = np.array([direction[1], -direction[0]])
    cells = np.argwhere(dig).astype(float)
    origin = cells.mean(axis=0)
    yy, xx = np.indices(dig.shape)
    axial = (yy - origin[0]) * direction[0] + (xx - origin[1]) * direction[1]
    lateral = (yy - origin[0]) * normal[0] + (xx - origin[1]) * normal[1]
    return origin, direction, normal, axial, lateral


def build_lane(
    dig: np.ndarray,
    heading_deg: float,
    layout: Layout,
    min_band_tiles: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """A straight base-width traversable corridor parallel to the axis.

    §8.3 U7(c). The lane sits beyond the dig's lateral extent on the dumping
    side and spans the full working length; it is reserved BEFORE the dump is
    grown, so the corridor is guaranteed dump-free rather than merely observed
    to be. The dump band then lives between the trench and the lane, which is
    the real working layout: the machine stands on the lane, digs the trench in
    front of it and drops the spoil in the band between the two.

    The side is pair-seeded but not forced: a branch or a map edge can make one
    side impossible, so the preferred side is tried first and the realised side
    is what the dump builders and the manifest use.
    """
    origin, direction, normal, axial, lateral = trench_frame(dig, heading_deg)
    dig_axial = axial[dig]
    extent = float(dig_axial.max() - dig_axial.min())
    axial_centre = float((dig_axial.max() + dig_axial.min()) / 2.0)
    # The lane must leave the dump band room to fit between it and the trench,
    # or the band is starved on the lane side (which is where U7(c) puts it).
    base_gap = max(layout.lane_gap, layout.standoff + 3, int(math.ceil(min_band_tiles)))
    # A branch can reach 14 tiles off the spine, so the side with the smaller
    # lateral extent is tried first: the pair-seeded sign is a tie-break, not a
    # decree, and it is a function of the shared dig so siblings still agree.
    extents = {
        sign: float((lateral[dig] * sign).max()) for sign in (-1.0, 1.0)
    }
    signs = sorted(
        (-1.0, 1.0),
        key=lambda sign: (round(extents[sign], 3), -sign * float(layout.lane_sign)),
    )
    gaps = [base_gap, base_gap + 1, base_gap - 1, base_gap + 2, base_gap + 3]
    failure = "lane_off_map"
    for width in LANE_WIDTH_CHOICES:
      for sign in signs:
        for gap in gaps:
            if gap < 6:
                continue
            inner = float((lateral[dig] * sign).max()) + gap
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
            if not lane.any():
                continue
            if (lane & dig).any():
                failure = "lane_hits_dig"
                continue
            if border_margin(lane) < 1:
                continue
            lane_axial = axial[lane]
            if lane_axial.min() > dig_axial.min() or lane_axial.max() < dig_axial.max():
                failure = "lane_too_short"
                continue
            return lane, {
                "lane_sign": int(sign),
                "lane_gap_tiles": gap,
                "lane_inner_offset_tiles": round(inner, 3),
                "lane_width_tiles": width,
                "lane_cells": int(lane.sum()),
                "lane_centre_y": round(float(centre[0]), 3),
                "lane_centre_x": round(float(centre[1]), 3),
                "trench_working_length_tiles": round(extent, 3),
            }
    raise RuntimeError(failure)


def lane_retreat_required(lane_meta: dict[str, Any], steps: int) -> int:
    """How many in-lane backward steps this map has to deliver.

    A 13-tile lane carries the 7-9 tile rasterised footprint plus the drift of
    three 5-tile steps on a non-cardinal axis; the narrower fallback lanes only
    carry the footprint, so they are held to one step and the drift is reported.
    """
    full = int(lane_meta["lane_width_tiles"]) >= LANE_WIDTH_FOR_FULL_RETREAT
    return min(BACKWARD_LANE_STEPS_MIN if full else 1, steps)


def lane_metrics_no_lane(
    dig: np.ndarray, occupancy: np.ndarray, dump: np.ndarray, heading_deg: float
) -> dict[str, Any]:
    """The retreat check for a trench that has no reserved lane (the wall)."""
    _, _, _, axial, _ = trench_frame(dig, heading_deg)
    extent = float(axial[dig].ptp())
    centre = np.argwhere(dig).astype(float).mean(axis=0)
    drive = tsvc.backward_drive_check(
        dig, dig | occupancy, heading_deg,
        (float(centre[0]), float(centre[1])), extent, None, 0,
    )
    return {
        "lane_free_fraction": 1.0,
        "trench_working_length_tiles": round(extent, 3),
        **drive,
    }


def lane_metrics(
    dig: np.ndarray,
    occupancy: np.ndarray,
    dump: np.ndarray,
    lane: np.ndarray,
    lane_meta: dict[str, Any],
    heading_deg: float,
) -> dict[str, Any]:
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


# --------------------------------------------------------------------------
# dump algorithms


def ring_band(
    dig: np.ndarray,
    target_area: int,
    rng: np.random.Generator,
    blocked: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """§8.3 U1: the designated dump is a finite band grown around the dig."""
    distance = ndi.distance_transform_edt(~dig)
    allowed = (distance >= 1) & (distance <= RING_BAND_MAX_RADIUS) & _interior(1)
    if blocked is not None:
        allowed &= ~blocked
    allowed = base.largest_component(allowed)
    if int(allowed.sum()) < target_area:
        raise RuntimeError("ring_band_infeasible")
    seeds = allowed & (distance <= 2.5)
    if not seeds.any():
        seeds = allowed & (distance <= 4.0)
    target = base.grow_region(allowed, seeds, target_area, rng)
    if int(target.sum()) < target_area:
        raise RuntimeError("ring_band_infeasible")
    reach = float(distance[target].max())
    return target, {
        "dump_side": "foundation_capped_band",
        "dump_components_requested": base.target_components(np.where(target, 1, 0)),
        "dump_access_sides": "all",
        "dump_alignment": "capped_band_around_dig",
        "distance_bucket": "immediate",
        "ring_band_max_reach_tiles": round(reach, 3),
    }


APRON_AZIMUTH_NEED_FACTOR = 4.5


def apron_azimuth(
    dig: np.ndarray, layout: Layout, need_factor: float = APRON_AZIMUTH_NEED_FACTOR
) -> tuple[float, np.ndarray]:
    """The pair-seeded apron azimuth and its 200 deg sector.

    RC4 carried forward, and U6 depends on it: the distance ladder must sit on
    the SAME azimuth as its `apron-near` anchor, so the two builders share this
    one function and one `need_factor`.
    """
    distance, angles, _ = _polar(dig)
    half = math.radians(layout.sector_degrees) / 2
    interior = _interior(2)
    need = need_factor * int(dig.sum())
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
    if chosen is None:
        raise RuntimeError("apron_capacity_infeasible")
    return float(best_azimuth), chosen


def apron_sector(
    dig: np.ndarray,
    layout: Layout,
    target_area: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """RC4 apron: fixed 200 deg sector, pair-seeded azimuth, capacity radial."""
    distance, _, _ = _polar(dig)
    best_azimuth, chosen = apron_azimuth(dig, layout)
    if int(chosen.sum()) < target_area:
        raise RuntimeError("apron_capacity_infeasible")

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


def distance_apron(
    dig: np.ndarray,
    layout: Layout,
    target_area: int,
    rng: np.random.Generator,
    target_median: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """§8.3 U6: an apron placed so the REALISED median haul hits the bin.

    The control variable is the measured median dig->nearest-dump distance, not
    a standoff parameter: the same offset produces very different medians on
    different footprints, which is exactly the confound U6 exists to remove.
    """
    distance, angles, _ = _polar(dig)
    half = math.radians(layout.sector_degrees) / 2
    interior = _interior(2)
    azimuth, _sector_mask = apron_azimuth(dig, layout)
    sector = v2.angle_difference(angles, azimuth) <= half
    best: tuple[float, np.ndarray, float] | None = None
    for inner in np.arange(2.0, 34.0, 1.0):
        allowed = base.largest_component(
            sector & interior & (distance >= inner) & (distance <= inner + 14.0)
        )
        if int(allowed.sum()) < target_area:
            continue
        seed = v2.seed_near_angle(
            allowed, dig, azimuth, preferred_distance=inner + 2.0
        )
        target = base.grow_region(allowed, seed, target_area, rng)
        if int(target.sum()) < target_area:
            continue
        median = float(
            np.median(ndi.distance_transform_edt(~target)[dig])
        )
        error = abs(median - target_median)
        if best is None or error < best[0]:
            best = (error, target, median)
        if error <= 0.4:
            break
    if best is None or best[0] > DISTANCE_BIN_TOLERANCE:
        raise RuntimeError("distance_bin_infeasible")
    return best[1], {
        "dump_side": "distance_apron",
        "dump_components_requested": 1,
        "dump_sector_degrees": float(layout.sector_degrees),
        "dump_azimuth_deg": round(math.degrees(azimuth) % 360.0, 3),
        "distance_bucket": "graded",
        "dump_alignment": "foundation_distance_apron",
        "distance_bin_target_tiles": target_median,
        "distance_bin_realized_median_tiles": round(best[2], 3),
    }


def trench_band(
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    target_area: int,
    rng: np.random.Generator,
    blocked: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """§8.3 U3: an adjacent spoil band hugging the whole trench, both flanks.

    One algorithm for every geometry level, so `straight -> seg2 -> seg3 ->
    tee -> net` is a clean geometry delta with the dump layout held fixed.
    """
    heading = float(dig_meta["trench_global_angle_deg"])
    _, _, _, _, lateral = trench_frame(dig, heading)
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


def trench_flank(
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    target_area: int,
    rng: np.random.Generator,
    blocked: np.ndarray | None = None,
    side: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One adjacent flank, on the same side as the working lane (U7c)."""
    heading = float(dig_meta["trench_global_angle_deg"])
    _, _, _, _, lateral = trench_frame(dig, heading)
    distance = ndi.distance_transform_edt(~dig)
    reach = int(rng.integers(12, 20))
    sign = float(side)
    band = (
        (distance >= layout.standoff)
        & (distance <= reach)
        # `>= -2` instead of `>= 1`: the band may wrap the trench ENDS, which is
        # still one flank in character and is what makes 2.0x reachable on a
        # tee, where the branch eats into the flank on its own side.
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
        "dump_side": (
            "trench_main_axis_left" if sign < 0 else "trench_main_axis_right"
        ),
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


def trench_altsides(
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    target_area: int,
    rng: np.random.Generator,
    blocked: np.ndarray | None = None,
    fence_side: int = 0,
    fence_clear: float = 0.0,
    fence_inner: float = 0.0,
    fence_thickness: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """§8.3 U3: alternating banks along the trench, always adjacent.

    The axial extent is cut into ``alternations`` windows with a real gap
    between them; window k gets a pad on side ``(-1)**k``. This replaces
    `trn-straight-split` — the skill is spoil organisation, not hauling.

    The requested alternation count is clamped by the working length: a 24-tile
    trench cannot carry four banks that are each long enough to hold a quarter
    of the spoil, and silently accepting a 4-tile "bank" would make the
    condition a different thing on short maps than on long ones.

    ``fence_side``/``fence_clear`` push the pads on ONE side out past a fence
    (the trench wall condition); everything else stays directly adjacent.
    """
    heading = float(dig_meta["trench_global_angle_deg"])
    _, _, _, axial, lateral = trench_frame(dig, heading)
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
            return _altsides_pads(
                dig, layout, target_area, rng, blocked, fence_side, fence_clear,
                axial, lateral, distance, low, high, extent, gap, n, span,
                fence_inner, fence_thickness,
            )
        except RuntimeError as exc:
            failure = str(exc)
    raise RuntimeError(failure)


def _altsides_pads(
    dig, layout, target_area, rng, blocked, fence_side, fence_clear,
    axial, lateral, distance, low, high, extent, gap, n, span,
    fence_inner=0.0, fence_thickness=0.0,
):

    reach = int(rng.integers(14, 21))
    first = int(layout.side_sign)
    target = np.zeros_like(dig, dtype=bool)
    pads: list[dict[str, Any]] = []

    # Two passes. The bank on the lane side has less room than the bank on the
    # open side (the lane is reserved right behind it), so the spoil is split in
    # proportion to the room each window actually has instead of equally — an
    # equal split just fails on the lane side and re-rolls the whole map.
    rooms: list[np.ndarray] = []
    for index in range(n):
        sign = first * (-1) ** index
        start = low + index * (span + gap)
        window = (axial >= start - 1.0) & (axial <= start + span + 1.0)
        near = layout.standoff
        far = reach
        if fence_side and sign == fence_side:
            near = max(near, fence_clear)
            far = max(far, fence_clear + 12.0)
        allowed = (
            window
            & (distance >= near)
            & (distance <= far)
            & (sign * lateral >= 1.0)
            & _interior(2)
        )
        if blocked is not None:
            allowed &= ~blocked
        rooms.append(allowed)
    capacities = np.array(
        [int(base.largest_component(room).sum()) for room in rooms], dtype=float
    )
    if capacities.sum() < target_area * 1.02:
        raise RuntimeError("altsides_pad_shortfall")
    areas = np.floor(capacities / capacities.sum() * target_area).astype(int)
    areas[int(np.argmax(capacities))] += target_area - int(areas.sum())
    if (areas < 10).any():
        raise RuntimeError("altsides_pad_shortfall")

    for index in range(n):
        sign = first * (-1) ** index
        start = low + index * (span + gap)
        near = layout.standoff
        if fence_side and sign == fence_side:
            near = max(near, fence_clear)
        allowed = rooms[index] & ~ndi.binary_dilation(
            target, structure=base.binary_disk(3)
        )
        allowed = base.largest_component(allowed)
        if int(allowed.sum()) < areas[index]:
            raise RuntimeError("altsides_pad_shortfall")
        score = np.abs(distance - (near + 1.5)) + 0.35 * np.abs(
            axial - (start + span / 2)
        )
        score[~allowed] = np.inf
        seed = np.zeros_like(dig, dtype=bool)
        seed[np.unravel_index(np.argmin(score), score.shape)] = True
        pad = base.grow_region(allowed, seed, int(areas[index]), rng)
        if int(pad.sum()) < int(areas[index]):
            raise RuntimeError("altsides_pad_shortfall")
        target |= pad
        pads.append(
            {
                "sign": int(sign),
                "axial": round(float(axial[pad].mean()), 3),
                "cells": int(pad.sum()),
            }
        )
    labels, components = ndi.label(target, structure=np.ones((3, 3), dtype=np.uint8))
    if components != n:
        raise RuntimeError("altsides_pads_merged")
    fence_meta: dict[str, Any] = {}
    if fence_side:
        # Hand the proven v3.2 offset-fence placer the azimuth window the
        # fenced-side banks actually occupy, so "behind the wall" is exact by
        # construction instead of being searched for.
        fenced = target & (fence_side * lateral > 0)
        if not fenced.any():
            raise RuntimeError("altsides_fence_side_empty")
        cy, cx = centroid(dig)
        ys, xs = np.where(fenced)
        angles = np.arctan2(ys - cy, xs - cx)
        reference = math.atan2(*(np.argwhere(fenced).astype(float).mean(axis=0) - [cy, cx]))
        centre_angle = float(
            np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        )
        spread = float(np.abs(v2.angle_difference(angles, centre_angle)).max())
        fence_meta = {
            "fence_inner_offset_tiles": float(fence_inner),
            "fence_band_thickness_tiles": float(fence_thickness),
            "fence_axis_rad": centre_angle,
            "fence_pad_delta_rad": 0.0,
            "fence_pad_half_width_rad": float(min(spread, math.radians(80.0))),
        }
        del reference
    return target, {
        **fence_meta,
        "dump_side": "trench_alternating_banks",
        "dump_access_sides": "alternating",
        "dump_alignment": "trench_alternating_banks",
        "dump_components_requested": n,
        "distance_bucket": "immediate_near",
        "altsides_pads": pads,
        "altsides_count": n,
        "altsides_requested": int(layout.alternations),
        "altsides_span_tiles": round(span, 3),
        "altsides_gap_tiles": round(gap, 3),
        "altsides_first_sign": first,
        "trench_flank_reach_tiles": reach,
    }


def build_dump(
    condition: ConditionSpec,
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    rng: np.random.Generator,
    blocked: np.ndarray | None = None,
    fence: tuple[int, float, float, float] = (0, 0.0, 0.0, 0.0),
    lane_sign: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    style = condition.dump_style
    if condition.capacity_level in APRON_CAPACITY_BANDS:
        band = APRON_CAPACITY_BANDS[condition.capacity_level]
    elif condition.capacity_level == "tight":
        band = TIGHT_CAPACITY_BAND
    elif condition.id in WALL_CONDITIONS:
        band = WALL_SPLIT_CAPACITY_BAND
    elif condition.id in TRENCH_WALL_CONDITIONS:
        band = TRENCH_WALL_CAPACITY_BAND
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
        target, metadata = v7.foundation_one_side(
            dig, layout, target_area, rng, blocked
        )
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
            dig, dig_meta, layout, target_area, rng, blocked, *fence
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
# the trench wall


def make_trench_fence(
    dig: np.ndarray,
    dump: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    fence_side: int,
    inner: float,
    thickness: float,
    lane: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """A gapped fence between the trench and the pads on ONE side.

    §8.3 U3 keeps trench dumping adjacent, which makes the §8.2 dig->dump haul
    detour structurally impossible for a trench: every pad is beside the arm it
    serves. The wall condition therefore fences ONE alternating side — those
    pads stay adjacent in distance but have to be reached through the gate — and
    the detour is measured on exactly the same geodesic metric.
    """
    origin, direction, normal, axial, lateral = trench_frame(
        dig, float(dig_meta["trench_global_angle_deg"])
    )
    side_pads = dump & (fence_side * lateral > 0)
    if not side_pads.any():
        raise RuntimeError("trench_fence_no_pads")
    pad_axial = axial[side_pads]
    # Distance from the TRENCH, not from its convex hull: a comb net's hull
    # swallows the space between the branches, so a hull-offset band lands
    # inside the banks it is supposed to fence off.
    hull_distance = ndi.distance_transform_edt(~dig)

    for _ in range(80):
        overhang = float(rng.uniform(3.0, 12.0))
        low = float(pad_axial.min()) - overhang
        high = float(pad_axial.max()) + overhang
        band = (
            (hull_distance >= inner)
            & (hull_distance < inner + thickness)
            & (fence_side * lateral > 0)
            & (axial >= low)
            & (axial <= high)
            & _interior(1)
        )
        band &= ~ndi.binary_dilation(dump, structure=base.binary_disk(2))
        band &= ~ndi.binary_dilation(dig, structure=base.binary_disk(2))
        band &= ~lane
        if int(band.sum()) < 45:
            continue
        gate_position = float(rng.uniform(0.14, 0.36))
        gate_axial = low + gate_position * (high - low)
        if rng.random() < 0.5:
            gate_axial = high - gate_position * (high - low)
        ring = band & (np.abs(axial - gate_axial) <= 3.0)
        if not ring.any():
            continue
        gap_width = int(rng.integers(12, 17))
        gap_centre = centroid(ring)
        yy, xx = np.indices(dig.shape)
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
        ratio, mean_free, mean_walled, unreachable = haul_detour(dig, dump, wall)
        if not math.isfinite(ratio) or unreachable > 0.0:
            continue
        if ratio < TRENCH_WALL_DETOUR_MIN:
            continue
        offaxis = gap_offaxis_tiles(dig, dump, tuple(gap_centre))
        return wall, {
            "wall_style": "trench_side_fence",
            "wall_gap_tiles": gap_width,
            "wall_thickness_tiles": min_thickness,
            "wall_band_thickness_tiles": int(thickness),
            "wall_fence_inner_offset_tiles": inner,
            "wall_fence_side": fence_side,
            "wall_fence_axial_span_tiles": round(high - low, 3),
            "wall_gap_offaxis_tiles": round(offaxis, 3),
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
    raise RuntimeError("trench_fence_infeasible")


# --------------------------------------------------------------------------
# site


def make_trench_barrier(
    dig: np.ndarray,
    dump: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    fence_side: int,
    inner: float,
    lane: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """A gapped barrier along one bank of the trench.

    §8.2's wall model is a fence that SEPARATES the dig from its spoil, and it
    is gated on the geodesic haul detour that separation costs. U3 makes that
    construction impossible for a trench: every bank is beside the arm it
    serves, so no fence can lengthen the haul without either sealing the base
    out of the trench (`dig_not_reachable_pre`) or failing the thickness /
    component gates (`trench_fence_infeasible`) — measured over ~1000 attempts
    per map in every arrangement tried (GENERATION_NOTES §7).

    What this builds instead is what a real site has: 2-3 solid obstacles
    strung along one bank with drivable gaps between them. The structural gates
    (>= 3 tiles thick, >= 2 components, >= 25 cells and >= 8 tiles each) are
    kept and asserted; the haul detour is measured and REPORTED, not gated.
    """
    heading = float(dig_meta["trench_global_angle_deg"])
    _, _, _, axial, lateral = trench_frame(dig, heading)
    distance = ndi.distance_transform_edt(~dig)
    dig_axial = axial[dig]
    low, high = float(dig_axial.min()), float(dig_axial.max())
    forbidden = (
        ndi.binary_dilation(dig, structure=base.binary_disk(2))
        | ndi.binary_dilation(dump, structure=base.binary_disk(1))
        | lane
        | ~_interior(2)
    )
    for _ in range(60):
        count = int(rng.integers(2, 4))
        span = (high - low) / count
        wall = np.zeros_like(dig, dtype=bool)
        placed = 0
        for index in range(count):
            for _try in range(40):
                along = low + span * (index + float(rng.uniform(0.25, 0.75)))
                offset = float(rng.uniform(inner, inner + 6.0))
                centre = (
                    np.argwhere(dig).astype(float).mean(axis=0)
                    + np.array(
                        [
                            math.sin(math.radians(heading)),
                            math.cos(math.radians(heading)),
                        ]
                    )
                    * along
                )
                normal = np.array(
                    [
                        math.cos(math.radians(heading)),
                        -math.sin(math.radians(heading)),
                    ]
                )
                centre = centre + normal * fence_side * offset
                block = base.rotated_rectangle(
                    (float(centre[0]), float(centre[1])),
                    float(rng.uniform(9.0, 17.0)),
                    float(rng.uniform(3.0, 5.0)),
                    math.radians(heading),
                )
                if not block.any() or np.any(block & forbidden):
                    continue
                if np.any(
                    ndi.binary_dilation(block, structure=base.binary_disk(4)) & wall
                ):
                    continue
                wall |= block
                placed += 1
                break
        if placed < 2:
            continue
        components, min_cells, min_extent = wall_component_stats(wall)
        if components < WALL_MIN_COMPONENTS or min_cells < WALL_MIN_COMPONENT_CELLS:
            continue
        if min_extent < WALL_MIN_COMPONENT_EXTENT:
            continue
        if wall_min_thickness(wall) < WALL_MIN_THICKNESS_TILES:
            continue
        ratio, mean_free, mean_walled, unreachable = haul_detour(dig, dump, wall)
        if not math.isfinite(ratio) or unreachable > 0.0:
            continue
        gaps = []
        labels, _ = ndi.label(wall, structure=np.ones((3, 3), dtype=np.uint8))
        for i in range(1, components + 1):
            far = ndi.distance_transform_edt(~(labels == i))
            for j in range(i + 1, components + 1):
                gaps.append(float(far[labels == j].min()))
        return wall, {
            "wall_style": "trench_bank_barrier",
            "wall_gap_tiles": int(round(min(gaps))) if gaps else 0,
            "wall_thickness_tiles": wall_min_thickness(wall),
            "wall_band_thickness_tiles": int(round(inner)),
            "wall_fence_inner_offset_tiles": inner,
            "wall_fence_side": fence_side,
            "wall_blocks": placed,
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
    raise RuntimeError("trench_barrier_infeasible")


class SiteFactoryV8:
    @staticmethod
    def make(
        condition: ConditionSpec,
        dig: np.ndarray,
        dump: np.ndarray,
        protect: np.ndarray,
        rng: np.random.Generator,
        dump_meta: dict[str, Any] | None = None,
    ):
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
        elif level == "wall":
            wall, wall_meta = make_offset_fence_wall(dig, dump, dump_meta or {}, rng)
            occupancy |= wall
            metadata.update(wall_meta)
        else:
            raise ValueError(f"site level {level!r} is not built here")

        occupancy &= ~protect
        nondump &= ~(protect | occupancy)
        corridor &= nondump
        dumpability = ~(nondump | occupancy)
        return occupancy, dumpability, corridor, metadata


# --------------------------------------------------------------------------
# sample construction


def _common_metadata(condition, dig, dump, occupancy, dumpability, gate):
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
        "tile_size_m": base.TILE_SIZE_M,
        "static_gate_is_action_witness": False,
    }


def _split_metrics(dig, dump):
    pads, edge_gap, span = pad_separation(dig, dump)
    return {
        "dump_max_angular_gap_deg": round(max_angular_gap_degrees(dig, dump), 2),
        "dump_border_margin_tiles": border_margin(dump),
        "dump_pad_count": pads,
        "dump_pad_edge_gap_tiles": round(edge_gap, 3),
        "dump_pad_angular_span_deg": round(span, 2),
    }


def make_sample(
    condition: ConditionSpec,
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    layout: Layout,
    rng: np.random.Generator,
) -> tuple[base.Sample | None, str]:
    is_trench = condition.family == "trench"
    heading = float(dig_meta.get("trench_global_angle_deg", 0.0))

    blocked = np.zeros_like(dig, dtype=bool)
    lane = np.zeros_like(dig, dtype=bool)
    lane_meta: dict[str, Any] = {}
    wants_lane = is_trench and condition.id not in LANE_EXEMPT_CONDITIONS
    if wants_lane:
        # How wide the spoil band has to be on the lane side, from the capacity
        # the condition is about to ask for and the working length it has: a
        # one-sided or alternating layout needs the whole band width there, a
        # both-flank layout only half of it.
        capacity_high = (
            TIGHT_CAPACITY_BAND[1]
            if condition.capacity_level == "tight"
            else (
                TRENCH_WALL_CAPACITY_BAND[1]
                if condition.id in TRENCH_WALL_CONDITIONS
                else GENEROUS_CAPACITY_BANDS[condition.dump_style][1]
            )
        )
        share = 0.5 if condition.dump_style == "trench_band" else 1.0
        _, _, _, axial_probe, _ = trench_frame(dig, heading)
        extent_probe = float(axial_probe[dig].ptp())
        # a soft floor only: the lane still has to fit on a 64x64 map, so the
        # band width it asks for is capped rather than honoured unconditionally
        min_band = min(
            10.0 if condition.id not in TRENCH_WALL_CONDITIONS else 24.0,
            share * capacity_high * int(dig.sum()) / max(8.0, extent_probe),
        )
        lane, lane_meta = build_lane(dig, heading, layout, min_band)
        blocked |= lane
        # The retreat depends on the dig and the lane alone, so check it before
        # any of the expensive dump/site/gate work: a dig that cannot be
        # retreated along has to be re-rolled, not re-dumped.
        probe = tsvc.backward_drive_check(
            dig,
            dig,
            heading,
            (float(lane_meta["lane_centre_y"]), float(lane_meta["lane_centre_x"])),
            float(lane_meta["trench_working_length_tiles"]),
            lane,
            BACKWARD_LANE_STEPS_MIN,
        )
        if probe["backward_drive_drift_per_tile"] > BACKWARD_DRIFT_PER_TILE_MAX:
            return None, "backward_drive_drift_contract"
        if not probe["backward_drive_footprint_clear"]:
            return None, "backward_drive_blocked_contract"
        if probe["backward_drive_lane_steps"] < lane_retreat_required(
            lane_meta, probe["backward_drive_steps"]
        ):
            return None, "backward_drive_lane_contract"

    # §8.2 road-first ordering on zoned layouts and, new in v4, on the capped
    # ring: the band has to be grown AROUND the corridor for the road to bite.
    # Objects on a capped ring are placed first for the same reason — with U1
    # the band is finite, so a site built afterwards would simply sit on top of
    # a designated dump cell (a hard static-gate reject).
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
        objects, areas = place_objects_v7(dig, dig, requested_objects, rng)
        intrusion = object_intrusion(dig, objects)
        if intrusion["annulus4_free_fraction"] < OBJECT_NEAR_ANNULUS_FREE_MIN:
            return None, "object_corridor_blocked"
        if intrusion["annulus6_blocked_fraction"] < OBJECT_ANNULUS_BLOCK_MIN[
            condition.site_level
        ]:
            return None, "object_intrusion_too_low"
        object_meta = {
            **intrusion,
            "object_count": requested_objects,
            "object_footprint_profile": OBJECT_FOOTPRINT_PROFILE,
            "object_area_cells_mean": round(float(np.mean(areas)), 3),
            "object_area_cells_min": int(min(areas)),
            "object_area_cells_max": int(max(areas)),
        }
        blocked |= objects

    # the trench wall is built together with its pads
    lane_sign = int(lane_meta.get("lane_sign", layout.lane_sign))
    fence_side, fence_clear = 0, 0.0
    fence_inner = fence_thickness = 0.0
    if condition.id in TRENCH_WALL_CONDITIONS:
        # The fence sits between the trench and the banks on ONE alternating
        # side: those banks stay trench-adjacent in distance but can only be
        # reached through the gate. That is the only construction in which U3
        # (dump alongside the trench) and a wall that costs travel coexist.
        fence_side = int(layout.lane_sign)
        fence_inner = float(rng.choice(TRENCH_FENCE_INNER_CHOICES))
        fence_thickness = float(rng.choice(TRENCH_FENCE_THICKNESS_CHOICES))
        fence_clear = fence_inner + fence_thickness + 2.0

    dump, dump_meta = build_dump(
        condition,
        dig,
        dig_meta,
        layout,
        rng,
        blocked if blocked.any() else None,
        (fence_side, fence_clear, fence_inner, fence_thickness),
        lane_sign,
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
    elif condition.id in TRENCH_WALL_CONDITIONS:
        try:
            wall, wall_meta = make_trench_barrier(
                dig, dump, dig_meta, layout, fence_side, fence_inner, lane, rng,
            )
        except RuntimeError as exc:
            return None, str(exc)
        occupancy = wall & ~(dig | dump)
        dumpability = ~occupancy
        site_meta.update(wall_meta)
    elif condition.site_level == "wall":
        occupancy, dumpability, corridor, extra = SiteFactoryV8.make(
            condition, dig, dump, dig | dump, rng, dump_meta
        )
        site_meta.update(extra)
        if (occupancy & dump).any():
            return None, "site_overlaps_dump"

    target = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    target[dig] = -1
    target[dump] = 1
    components = base.target_components(target)
    if components != int(dump_meta["dump_components_requested"]):
        return None, "dump_component_contract"

    gate = base.static_gate(target, occupancy, dumpability)
    if not gate.accepted:
        return None, gate.reason
    if condition.id in T0_CONDITIONS and gate.dig_workspace_coverage_post < T0_DIG_COVERAGE_MIN:
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
            limit = WALL_SPLIT_MEDIAN_LIMIT
        if condition.id in TRENCH_WALL_CONDITIONS:
            limit = TRENCH_WALL_MEDIAN_LIMIT
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

    extra: dict[str, Any] = {}

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
        # The road cannot overlap the band by construction (the band was grown
        # around it), so "does the road bite?" is measured against the band the
        # SAME map would have grown with no road on it.
        open_band, _ = ring_band(
            dig, requested, np.random.default_rng(band_probe_seed)
        )
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

    # ---- U7 lane and backward drive ------------------------------------
    if is_trench:
        extra["lane_present"] = int(wants_lane)
        if wants_lane:
            metrics = lane_metrics(dig, occupancy, dump, lane, lane_meta, heading)
        else:
            metrics = lane_metrics_no_lane(dig, occupancy, dump, heading)
        extra.update(lane_meta)
        extra.update(metrics)
        if wants_lane and metrics["lane_free_fraction"] < 1.0:
            return None, "lane_not_free_contract"
        if not metrics["backward_drive_footprint_clear"]:
            return None, "backward_drive_blocked_contract"
        if metrics["backward_drive_drift_per_tile"] > BACKWARD_DRIFT_PER_TILE_MAX:
            return None, "backward_drive_drift_contract"
        if wants_lane and metrics["backward_drive_lane_steps"] < lane_retreat_required(
            lane_meta, metrics["backward_drive_steps"]
        ):
            return None, "backward_drive_lane_contract"
        spurs, notches = edge_irregularity(dig)
        extra["trench_edge_spurs"] = spurs
        extra["trench_edge_notches"] = notches
        if spurs or notches:
            return None, "trench_edge_regularity_contract"

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


def make_map(condition, dig, dig_meta, layout, rng):
    try:
        return make_sample(condition, dig, dig_meta, layout, rng)
    except RuntimeError as exc:
        return None, str(exc)


def generate_condition(condition, condition_index, bank, n_maps, max_attempts):
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


def write_condition(output, condition, condition_index, samples):
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
                    "distanceBand": DISTANCE_TOKENS[condition.distance_level],
                },
                "objectBand": list(OBJECT_BANDS[condition.site_level]),
                "layoutGroup": condition.group,
                "pairGroup": condition.pair_group,
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
        "# Terra curriculum v4 review bank",
        "",
        "Taxonomy-native bank: one folder per **condition id**, no stage folders.",
        "Difficulty tier is computed from factor levels, never hand-assigned.",
        "See `../../terra-digging-benchmark-site/docs/CURRICULUM_TAXONOMY_SPEC.md`"
        " sections 8, 8.1, 8.2, **8.3 and 8.4**.",
        "",
        f"- seed base: `{SEED_BASE}` (fully reproducible)",
        f"- conditions: {len(CONDITIONS)}",
        f"- maps: {sum(counts.values())} "
        f"({MAPS_PER_CONDITION} per condition, "
        f"{PREVIEW_MAPS_PER_CONDITION} for the remote-haul preview)",
        f"- map ids: `{MAP_ID_PREFIX}-NNNN`",
        "",
        "v4 implements U1-U7 from Lorenzo's review: capped ring bands,",
        "direct service at T0, trench dumping always adjacent, the overlap",
        "stressor on every multi-segment trench, a densified capacity ladder,",
        "a foundation distance ladder, and trench axes on the drivable 30 deg",
        "lattice with a reserved working lane.",
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--only", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    assert_conditions_match_taxonomy()
    args = parse_args()
    output = args.output.resolve()
    for folder in (
        output,
        output / "review_metadata",
        *(output / "dataset" / name for name in ARRAY_FOLDERS),
    ):
        folder.mkdir(parents=True, exist_ok=True)

    selected = set(filter(None, args.only.split(",")))
    factory = GeometryFactoryV8(args.source_foundations)
    bank_size = args.maps or MAPS_PER_CONDITION
    bank = DigBankV8(factory, bank_size)
    # The bank is independent per geometry level (its dedup only ever looks
    # inside one level), so building a subset is exactly what a full run would
    # have produced for those levels — it just skips the levels nobody asked for.
    wanted = selected or {c.id for c in CONDITIONS if c.id not in BLOCKED_CONDITIONS}
    levels = sorted({c.geometry_level for c in CONDITIONS if c.id in wanted})
    bank_rejections = bank.build(levels)
    print(f"dig bank built: {bank_size} per level, rejections={dict(bank_rejections)}")

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    rejection_totals: Counter[str] = Counter()
    unsatisfied: list[str] = []

    blocked_reports: dict[str, dict[str, int]] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        if selected and condition.id not in selected:
            continue
        if condition.id in BLOCKED_CONDITIONS and not selected:
            counts[condition.id] = 0
            print(
                f"[{condition_index + 1:02d}/{len(CONDITIONS)}] {condition.id}: "
                "BLOCKED — see BLOCKED_CONDITIONS / GENERATION_NOTES",
                flush=True,
            )
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

    write_terra_metadata(output, rows)
    regenerated = sorted(counts)

    if args.resume:
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
    realized_capacity = defaultdict(list)
    for row in rows:
        realized[row["condition_id"]].append(float(row["reachable_dump_to_dig_ratio"]))
        realized_capacity[row["condition_id"]].append(
            float(row["dump_to_dig_area_ratio"])
        )
    summary = {
        "schema": SCHEMA,
        "seed_base": SEED_BASE,
        "spec_path": "docs/CURRICULUM_TAXONOMY_SPEC.md",
        "spec_section": "8 + 8.1 + 8.2 + 8.3 + 8.4",
        "taxonomy_version": tax.TAXONOMY_VERSION,
        "taxonomy_release": "v4",
        "generator": "generate_prototypes_v8.py",
        "map_id_prefix": MAP_ID_PREFIX,
        "conditions_built_this_run": regenerated,
        "source_foundations": str(args.source_foundations),
        "condition_count": len(CONDITIONS),
        "maps_per_condition": counts,
        "accepted_maps": len(rows),
        "resume": bool(args.resume),
        "rerolled_dig_maps": sum(1 for row in rows if int(row["shared_dig"]) == 0),
        "u_gates": {
            "u1_ring_band_capacity": list(RING_BAND_CAPACITY),
            "u1_road_band_sterilize_min": ROAD_BAND_STERILIZE_MIN,
            "u2_direct_service_min": DIRECT_SERVICE_MIN,
            "u2_direct_service_p95_max_tiles": DIRECT_SERVICE_P95_MAX,
            "u2_apron_standoff_choices": list(APRON_STANDOFF_CHOICES),
            "u4_arm_overlap_min": ARM_OVERLAP_MIN,
            "u4_spoil_flank_tiles": SPOIL_FLANK_TILES,
            "u5_apron_capacity_bands": {
                k: list(v) for k, v in APRON_CAPACITY_BANDS.items()
            },
            "u6_distance_bins": DISTANCE_BINS,
            "u6_distance_bin_tolerance_tiles": DISTANCE_BIN_TOLERANCE,
            "u7_trench_axes_deg": list(TRENCH_AXES_DEG),
            "u7_lane_width_tiles": LANE_WIDTH_TILES,
            "u7_backward_drift_per_tile_max": BACKWARD_DRIFT_PER_TILE_MAX,
            "u7_backward_lane_steps_min": BACKWARD_LANE_STEPS_MIN,
            "wall_detour_min": WALL_DETOUR_MIN,
            "wall_detour_condition_median_min": WALL_DETOUR_CONDITION_MEDIAN_MIN,
        },
        "object_bands": {k: list(v) for k, v in OBJECT_BANDS.items()},
        "generous_capacity_bands": {
            k: list(v) for k, v in GENEROUS_CAPACITY_BANDS.items()
        },
        "tight_capacity_band": list(TIGHT_CAPACITY_BAND),
        "generous_reachable_floor": GENEROUS_REACHABLE_FLOOR,
        "realized_reachable_capacity": {
            key: [round(min(values), 3), round(max(values), 3)]
            for key, values in sorted(realized.items())
        },
        "realized_designated_capacity": {
            key: [round(min(values), 3), round(max(values), 3)]
            for key, values in sorted(realized_capacity.items())
        },
        "shared_dig_attempts": SHARED_DIG_ATTEMPTS,
        "reroll_dump_attempts": REROLL_DUMP_ATTEMPTS,
        "dig_bank_rejections": dict(bank_rejections),
        "blocked_conditions": sorted(BLOCKED_CONDITIONS),
        "blocked_condition_reports": blocked_reports,
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
