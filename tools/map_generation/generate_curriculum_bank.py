#!/usr/bin/env python3
"""Generate the Terra curriculum bank with exact trench-axis ownership.

This is the one supported generator entry point for the diversity pilot. It
preserves the reviewed v6 condition, capacity, proximity, lane, obstacle, and
static-validity rules. The old all-pairs centred-IoU novelty gate is deliberately
not an admission rule: exact duplicate dig rasters are rejected, while
centred-IoU is reported later as a distribution diagnostic.

Axis-contract v2 changes trench generation only: global trench headings use a
15 degree lattice, every target cell carries exact generated owner bits, and
the schema/map IDs are new. Foundation geometry remains on the reviewed v6
construction and seeds.

The implementation retains the reviewed v6 construction gates. Historical
modules are private implementation dependencies; callers use this file only.

Lineage: ``generate_prototypes_v9.py`` (the v5 bank). The historical v6 deltas
below retain ``SEED_BASE`` so carried foundation conditions reproduce from the
same seeds and paired conditions retain their shared geometry slots.

D1 -- **gapped ring masks**.  All seven ring conditions regenerate.  The capped
     3-4x band gains 1-3 forbidden angular sectors of 15-40 deg each (total
     15-90 deg), drawn per map.  The gap sectors are NON-DESIGNATED GROUND, not
     obstacles: ``occupancy`` and ``dumpability`` are untouched, only the
     designated dump (``target > 0``) has notches.  Dumping stays trivially easy
     -- the band still surrounds the dig -- but the designated mask must be READ
     to avoid the notches, so the channel carries gradient from update zero.
     The capacity gate (designated AND reachable in [3.0, 4.0]x) and the
     proximity gate (p95 <= 11.375, max <= 14.0) are unchanged, and the band is
     grown INSIDE the gapped annulus so the gaps cost coverage, not capacity.

D2 -- **mini-junction conditions** (+3): ``trn-tee-side2-s``,
     ``trn-net3-side2-s``, ``trn-net4-side2-s``.  Short spine (12-18 tiles),
     arms scaled, dig volume <= 60% of the standard variant's median, generous
     both-side adjacent banks, own dig-bank entries.  Every U7 / turn-dump /
     lane gate applies unchanged.

     One spec gate is NOT satisfiable and is reported instead of relaxed
     silently -- see ``MINI_OVERLAP_SPEC_BAND`` below and GENERATION_NOTES.

v6.1 -- **the ADJACENT proximity class is tightened** (Lorenzo's v6 visual
     review, 2026-07-30). The v5.1 gate (p95 <= 11.375, max <= 14.0) still let
     typical apron stations haul. Adjacent conditions -- all seven rings, the
     apron capacity family, all trenches -- now gate on **per map max <= 11.375
     AND p95 <= 9.0**: every dig cell one arm swing from a designated dump. The
     bounded-asymmetric compositions (side1 / split) and the U6 distance ladder
     are unchanged, and named as such. The only lever needed was the apron inner
     edge: ``APRON_PROXIMITY_STANDOFF_MAX`` 5 -> 2, clamped (never re-drawn) so
     the pair-seeded azimuths survive. Three trench conditions do not clear the
     gate and are held out by name -- see ``ADJACENT_PROXIMITY_DEFERRED``.

v6.2 -- **the trench debt from TRENCH_WIDTH_AUDIT.md, closed** (2026-07-30).
     Three changes, all on `trn-*` maps:

     R1  the reserved lane is made WIDTH-AWARE. `LANE_INNER_BAND` is measured
         from the spine, so at ``half_width`` 2 an inner offset of 9.0-9.5 put
         the trench's far row 11.50-12.00 tiles from the lane's inner edge --
         outside ``r_max`` = 11.375. ``build_lane`` now gates
         ``lane_inner + half_width + 0.5 <= 10.375`` (one tile of slack under the
         boom) and publishes ``lane_far_reach_tiles`` per map. See
         ``LANE_FAR_REACH_MAX_TILES``.

     R4  ``seg2`` may no longer fold shallowly at ``half_width`` 2. A 30 deg
         relative turn merged two 5-tile legs into an 8-12 tile solid wedge
         (``curriculum-v6m-2311``: inscribed thickness 10.0 tiles = 5.71 m) --
         a slab, not a segmented trench. ``SEG2_MIN_RELATIVE_TURN_DEG``.

     R-prox  the ADJACENT proximity gate is now ENFORCED by CLASS at generation
         time, not only measured by the validator afterwards, and
         ``ADJACENT_PROXIMITY_DEFERRED`` is EMPTY. ``trn-straight-altsides``
         re-draws its one offending map under the gate. The two road combs are
         reclassified: every U10 planning rung is BOUNDED-ASYMMETRIC, because
         measured on a bank that clears the adjacent gate their ``plan_delta``
         collapses from 0.100-0.194 to 0.000 -- on those conditions the haul IS
         the planning pressure. Numbers in the block above
         ``ADJACENT_PROXIMITY_DEFERRED``.

Current taxonomy and bank contract: ``CURRICULUM_TAXONOMY.md``. Historical v6
construction lineage is recorded in this module's D1/D2 and carried-gate notes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import dataclasses
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import ndimage as ndi

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import generate_prototypes_v9 as v9
import terra_geom as tgeom  # noqa: F401  (kept for parity with v9's namespace)
import terra_service as tsvc  # noqa: F401
import turn_dump as tdump  # noqa: F401

v8 = v9.v8
v7 = v9.v7
v6 = v9.v6
v5 = v9.v5
v3 = v9.v3
v2 = v9.v2
base = v9.base
tax = v9.tax

DatasetSpec = v9.DatasetSpec

SCHEMA = "terra_curriculum_diverse_axis_bank_v2"
CURRENT_TAXONOMY_PATH = tax.SPEC_PATH
# NOT bumped: byte-identity of the carried conditions depends on it, and so does
# the shared dig between the ring conditions and their capacity/layout siblings.
SEED_BASE = v9.SEED_BASE

MAP_SIZE = v9.MAP_SIZE
MAP_CENTRE = v9.MAP_CENTRE
TILE_SIZE_M = v9.TILE_SIZE_M
ARRAY_FOLDERS = v9.ARRAY_FOLDERS
SITE_CLASS_TOKENS = v9.SITE_CLASS_TOKENS
CAPACITY_TOKENS = v9.CAPACITY_TOKENS
DISTANCE_TOKENS = v9.DISTANCE_TOKENS


def reset_array_scenario_sha256(arrays: dict[str, Any]) -> str:
    """Hash reset arrays without importing the JAX runtime into this CPU tool."""
    expected = tuple(ARRAY_FOLDERS)
    if set(arrays) != set(expected):
        raise ValueError(
            f"Scenario identity requires exactly {expected}; got {tuple(arrays)}."
        )
    digest = hashlib.sha256()
    for name in expected:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()

R_MIN_TILES = v9.R_MIN_TILES
R_MAX_TILES = v9.R_MAX_TILES
SINGLE_STATION_BUDGET_TILES = v9.SINGLE_STATION_BUDGET_TILES
PLANNING_LAYOUT_ROUNDS = 5

# --------------------------------------------------------------------------
# D1 -- gapped ring masks (spec v6 section 2)

RING_GAP_COUNT_CHOICES = (1, 2, 3)
RING_GAP_WIDTH_DEG = (15.0, 40.0)
RING_GAP_TOTAL_DEG = (15.0, 90.0)
# Two gaps separated by a 5 deg arc read as one wide gap; the arcs between them
# have to be wide enough to be a place you would actually dump.
RING_GAP_EDGE_SEPARATION_DEG = 25.0
RING_GAP_DRAW_ATTEMPTS = 64
# "the mask must be READ": the rim is the 1-3 tile ring around the excavation,
# i.e. where a station standing at the dig would swing to. On an ungapped v5
# ring the whole rim is designated; a gapped ring must leave a measurable,
# obstacle-free notch there or the gap is cosmetic.
RING_GAP_RIM_TILES = 3.0
RING_GAP_MIN_NOTCH_CELLS = 24
RING_GAP_MIN_NOTCH_CELLS_PER_GAP = 8
# reported (not gated): rim non-designated share should track total_gap / 360.
RING_GAP_CONDITIONS = frozenset(
    c.id for c in v9.MAIN_CONDITIONS if c.dump_style == "ring_band"
)

# --------------------------------------------------------------------------
# D2 -- mini-junction conditions (spec v6 section 2)
#
# Realised medians of the standard variants in the v5-main bank, from
# manifest.csv (dig_cells): tee 161, net3 132, net4 203. The 60% cap is per map,
# against the standard condition's MEDIAN, so the cap is a fixed integer rather
# than a moving target.
STANDARD_DIG_CELLS_MEDIAN = {"tee": 161, "net3": 132, "net4": 203}
MINI_VOLUME_FRACTION_MAX = 0.60
MINI_DIG_CELLS_MAX = {
    level: int(math.floor(MINI_VOLUME_FRACTION_MAX * median))
    for level, median in STANDARD_DIG_CELLS_MEDIAN.items()
}
MINI_SPINE_SPEC_BAND = (12.0, 18.0)   # spec v6 §2 D2, verbatim
MINI_HALF_WIDTH = 1.0
# Per level, chosen INSIDE the spec band by measurement: the binding constraint
# is the volume cap, and the cap is reached at a shorter spine on the two
# cardinal lattice headings (axis-aligned rasterisation is ~15% denser for the
# same nominal polyline length, measured: net3 median 90 cells at 0 deg vs 78 at
# 30 deg for identical draws).
MINI_SPINE_BAND = {
    "tee": (12.0, 15.0),
    "net3": (12.0, 15.0),
    "net4": (12.0, 18.0),
}
assert all(
    MINI_SPINE_SPEC_BAND[0] <= lo and hi <= MINI_SPINE_SPEC_BAND[1]
    for lo, hi in MINI_SPINE_BAND.values()
)
# Arms are scaled so the polyline fits under the volume cap: total rasterised
# cells ~= 3 * (spine + sum(arms)) at half_width 1. `tee` gets LONG arms on
# purpose — for a one-junction trench the branch is the binding arm and a longer
# branch dilutes the junction's share of its spoil flank, which is what puts the
# spec's mild band within reach (measured: worst-arm overlap 0.198 at ratio 0.37,
# 0.140 at ratio 0.9). For the combs the binding arm is the SPINE, whose overlap
# is ratio-insensitive (0.202 flat from ratio 0.4 to 2.5), so their arms are kept
# short to buy volume headroom instead.
MINI_BRANCH_RATIO = {
    "tee": (1.05, 1.50),
    "net3": (0.26, 0.46),
    "net4": (0.30, 0.55),
}
# The junction SPREAD along the spine is drawn per map, not fixed. A fixed
# spread plus a narrow ratio band makes every comb at a given heading nearly the
# same shape, and the dig bank's centred-IoU rule (< 0.70) then cannot fill 16
# slots: measured on the first full run, `net4@s` exhausted at map 9. The spread
# is the cheapest real shape axis a short comb has.
MINI_JUNCTION_SPREAD = (0.28, 0.48)
MINI_JUNCTION_JITTER = 0.06
MINI_CELL_BAND = {
    "tee": (40, MINI_DIG_CELLS_MAX["tee"]),
    "net3": (40, MINI_DIG_CELLS_MAX["net3"]),
    "net4": (55, MINI_DIG_CELLS_MAX["net4"]),
}
# Axis orientations use a 15 degree generation lattice. Terra's 30 degree base
# headings plus the inclusive 15 degree tolerance make every generated axis
# locally alignable, including the exact half-bin cases.
TRENCH_AXES_DEG = tuple(float(value) for value in range(0, 180, 15))

# `net3` cannot use the two
# CARDINAL headings under the 60% volume cap: its minimum realisable volume at
# 0 / 90 deg is 79 cells against a cap of 79 (measured over 1 200 draws per
# heading; yield 0.0008 at 90 deg, 0.0175 at 0 deg — the accepted tail is a
# single shape and the dig-bank IoU gate then rejects it against itself). Rather
# than raise the cap to 85 (= 64% of the standard median, i.e. break the spec
# clause) or let the bank exhaust, `net3` omits the two cardinal headings.
# `tee` and `net4` retain all 12 headings on the 15 degree lattice.
MINI_HEADINGS = {
    "tee": TRENCH_AXES_DEG,
    "net3": tuple(value for value in TRENCH_AXES_DEG if value not in (0.0, 90.0)),
    "net4": TRENCH_AXES_DEG,
}
assert all(
    set(headings) <= set(TRENCH_AXES_DEG)
    for headings in MINI_HEADINGS.values()
), "Mini headings must stay on the generation lattice"

# ---- the one unsatisfiable spec gate, stated in full -----------------------
#
# Spec v6 section 2 D2 asks for the U4 overlap stressor "at mild dose
# (0.05-0.15 -- below the standard >= 0.15 gate, deliberately)".
#
# `arm_overlap_fraction` is a RATIO: (another arm's dig inside this arm's 6-tile
# spoil flank) / (that flank's area). The numerator is fixed by the junction --
# each branch crosses the spine's flank over ~6 tiles of its own root, whatever
# its length -- while the denominator grows with the SPINE LENGTH. Shrinking a
# comb therefore RAISES the overlap ratio. Measured (400 draws per cell, the
# sampler reproduces the v5 standard values to within noise: net3
# 0.173/0.198/0.214 vs v5's realised 0.178/0.193/0.301, net4 0.208/0.236/0.294
# vs 0.208/0.234/0.292):
#
#   worst-arm overlap, min / median, branch ratio 0.70-0.95, half_width 1
#   net3  spine 12 -> 0.227/0.243   18 -> 0.188/0.202   30 -> 0.133/0.147
#         spine 38 -> 0.112/0.122   46 -> 0.105/0.123
#   net4  spine 12 -> 0.347/0.389   18 -> 0.285/0.318   30 -> 0.203/0.224
#         spine 38 -> 0.171/0.186   46 -> 0.150/0.164
#
# Branch length has NO effect on the binding (spine) arm: at spine 18 the
# spine-arm overlap is 0.202 (net3) / 0.312 (net4) for every branch ratio from
# 0.4 to 2.5. Clustering the junctions makes it worse (net3 same-junction
# 0.245-0.346, net4 0.408-0.577), because the branches then sit in each other's
# flanks too.
#
# The contradiction is arithmetic, and it does not need the 12-18 spine clause:
#   * volume cap:   <= 79 cells (net3) => total polyline length <= ~26 tiles
#     <= 121 cells (net4) => <= ~40 tiles
#   * mild overlap: needs spine >= ~34 (net3) / >= ~50 (net4) tiles ON ITS OWN
# so the two clauses of D2 are mutually exclusive for net3 and net4. For `tee`
# (one junction, one crossing) they are jointly satisfiable and the spec band is
# ENFORCED.
#
# Nothing is relaxed silently: the spec band is enforced where it is reachable,
# and where it is not the condition is gated at the measured achievable floor
# (so the sampler still pushes toward the mild end) with
# `mini_overlap_spec_band_satisfied = False` written on every map.
#
# Realised gates, set to the lowest ceiling that keeps every admitted lattice
# heading feasible (measured minima across headings, 250-1200 draws each):
#   tee   0.110 (diagonal) / 0.144 (cardinal)  -> ceiling 0.16
#   net3  0.168 (diagonal)                     -> ceiling 0.26
#   net4  0.240 (diagonal) / 0.256 (cardinal)  -> ceiling 0.34
# The measurements above describe the historical 30 degree lattice. Generation
# still enforces the same gates for every admitted 15 degree heading; the combs
# do not approach the requested band, per the argument above.
MINI_OVERLAP_SPEC_BAND = (0.05, 0.15)
MINI_OVERLAP_GATE = {
    "tee": (0.05, 0.16),
    "net3": (0.00, 0.26),
    "net4": (0.00, 0.34),
}
MINI_OVERLAP_SPEC_SATISFIABLE = {"tee": "diagonal_headings_only", "net3": False,
                                 "net4": False}

MINI_BANK_LEVELS = ("tee@s", "net3@s", "net4@s")
MINI_GEOMETRY_OF = {"tee@s": "tee", "net3@s": "net3", "net4@s": "net4"}

# The dig-bank seeds are keyed by GEOMETRY_LEVEL_INDEX, so the ten v5 levels must
# keep the indices they had. New levels are appended, never re-sorted in.
# The v9 dict is mutated IN PLACE (not copied): v9's `ConditionSpec.geometry`
# closes over v9's global, and the carried conditions still use it.
v9.GEOMETRY_LEVEL_SOURCE.update(
    {"tee@s": "trench_axes_2", "net3@s": "trench_axes_3", "net4@s": "trench_axes_4"}
)
GEOMETRY_LEVEL_SOURCE = v9.GEOMETRY_LEVEL_SOURCE
GEOMETRY_LEVEL_INDEX = dict(v9.GEOMETRY_LEVEL_INDEX)
for _offset, _level in enumerate(MINI_BANK_LEVELS):
    GEOMETRY_LEVEL_INDEX[_level] = len(v9.GEOMETRY_LEVEL_INDEX) + _offset
# Pinned: these ten indices seed every carried condition's dig. If a future edit
# re-sorts the level table, every carried array changes and byte-identity dies.
V5_GEOMETRY_LEVEL_INDEX = {
    "net3": 0, "net4": 1, "proc": 2, "seg2": 3, "seg3": 4,
    "slab": 5, "slab-lg": 6, "straight": 7, "strips": 8, "tee": 9,
}
assert {k: GEOMETRY_LEVEL_INDEX[k] for k in V5_GEOMETRY_LEVEL_INDEX} == (
    V5_GEOMETRY_LEVEL_INDEX
), "v5 geometry-level indices moved; dig-bank seeds would change"

GEOMETRY_HARDNESS = {
    **v9.GEOMETRY_HARDNESS,
}
GEOMETRY_BUILDER = {
    **v9.GEOMETRY_BUILDER,
    "tee@s": "tee_s",
    "net3@s": "net3_s",
    "net4@s": "net4_s",
}
TRENCH_LEVELS = frozenset(v9.TRENCH_LEVELS) | frozenset(MINI_BANK_LEVELS)
MULTI_ARM_LEVELS = frozenset(v9.MULTI_ARM_LEVELS) | frozenset(MINI_BANK_LEVELS)
WIDE_TRENCH_LEVELS = frozenset(v9.WIDE_TRENCH_LEVELS)

# reused, unchanged
stable_key = v9.stable_key
sha256_mask = v9.sha256_mask
rng_from = v9.rng_from
centroid = v9.centroid
_interior = v9._interior
_polar = v9._polar
rasterize_segments = v9.rasterize_segments
regularise_edges = v9.regularise_edges
edge_irregularity = v9.edge_irregularity
arm_masks = v9.arm_masks
arm_overlap_fraction = v9.arm_overlap_fraction
arm_merge_fraction = v9.arm_merge_fraction
centred_iou = v9.centred_iou
place_at = v9.place_at
spine_fields = v9.spine_fields
dump_proximity = v9.dump_proximity
TRENCH_PLACEMENT_MARGIN = v9.TRENCH_PLACEMENT_MARGIN
TRENCH_LANE_SIDE_MAX = v9.TRENCH_LANE_SIDE_MAX
TRENCH_MAX_LATERAL_TILES = v9.TRENCH_MAX_LATERAL_TILES
ARM_MERGE_MAX = v9.ARM_MERGE_MAX
ARM_FILL_MAX = v9.ARM_FILL_MAX
ARM_OVERLAP_MIN = v9.ARM_OVERLAP_MIN
NET_TOPOLOGY_CYCLE = v9.NET_TOPOLOGY_CYCLE
RING_BAND_CAPACITY = v9.RING_BAND_CAPACITY
RING_BAND_MAX_RADIUS = v8.RING_BAND_MAX_RADIUS
PROXIMITY_GATED_STYLES = v9.PROXIMITY_GATED_STYLES
DUMP_PROXIMITY_P95_MAX = v9.DUMP_PROXIMITY_P95_MAX
DUMP_PROXIMITY_MAX_MAX = v9.DUMP_PROXIMITY_MAX_MAX

# --------------------------------------------------------------------------
# v6.1 -- the ADJACENT proximity class (Lorenzo's v6 visual review, 2026-07-30)
#
# The v5.1 gate (p95 <= 11.375, max <= 14.0, §8.6) still let typical stations
# haul: on the capacity aprons the realised p95 ran 7.2-11.4 tiles and the max
# to 13.0, and Lorenzo flagged `fnd-slab-apron-c1p6` map `curriculum-v6m-0612`
# (dig `af435783f345...`, p95 9.85 / max 10.44) as visibly too far. The
# proximity classes are now explicit, and the adjacent one is tightened so that
# EVERY dig cell is within one arm swing of a designated dump cell:
#
#   ADJACENT             per map max <= 11.375 AND p95 <= 9.0
#   BOUNDED-ASYMMETRIC   unchanged -- the distance IS the planning pressure
#   EXEMPT               the distance ladder -- far IS its factor
ADJACENT_PROXIMITY_MAX_MAX = R_MAX_TILES   # 11.375: one arm swing, no hauling
ADJACENT_PROXIMITY_P95_MAX = 9.0

# Bounded-asymmetric: side1 / split. Named so the class assignment is data, not
# a style-string coincidence.
BOUNDED_ASYMMETRIC_STYLES = frozenset({"one_side_near", "separated_zones"})
# Exempt: the U6 distance ladder.
PROXIMITY_EXEMPT_STYLES = frozenset({"distance_apron", "haul_away_edge"})

# v6.2: and every U10 planning rung, whatever its dump style. v6.1 already had
# this rule in effect for the three foundation rungs, because all three happen to
# use `one_side_near`; it classified the two TRENCH rungs as adjacent only
# because they carry a trench dump style. `plan_delta` measured on a bank that
# clears the adjacent gate is 0.000 (max 0.065 over 70 draws) against U10's 0.10
# floor -- the distance IS the pressure. See the deferral block below.
PROXIMITY_BOUNDED_BY_PLANNING = True

# v6.1 enforced the adjacent gate at generation time on `capacity_apron` +
# `ring_band` only and left the trench styles to the validator; v6.2 enforces it
# on every adjacent-CLASS condition, so a violating draw is rejected where it is
# drawn instead of being discovered afterwards. The 11 trench conditions that
# already cleared it did so by a wide margin (worst p95 6.00 against 9.0), so
# this costs them nothing; `trn-straight-altsides` re-draws its one offending map.

# Adjacent-class conditions that do NOT clear the tightened gate. v6.1 deferred
# three trench conditions here (`trn-net4-side1-road` 15/16 maps, worst p95
# 12.72 / max 15.00; `trn-net3-side1-road` 6/16, 10.30 / 12.08;
# `trn-straight-altsides` 1/16, 11.76 / 13.45) because a concurrent trench-width
# audit owned every `trn-*` map. v6.2 owns them and the list is now EMPTY:
# `trn-straight-altsides` re-draws under the enforced gate, and the two road
# combs are not adjacent-class at all -- they are U10 rungs. The set is kept
# rather than deleted: the validator fails loudly if a name in here turns out to
# be clean, and the next condition that misses the gate must be NAMED here to
# ship at all, never silently relaxed.
ADJACENT_PROXIMITY_DEFERRED: frozenset[str] = frozenset()

# The apron's inner edge is the dominant term in dig->dump proximity: measured
# over the 80 capacity-apron maps, the realised p50 tracks the standoff almost
# one-for-one (standoff 5 -> p50 6.3, standoff 2 -> p50 4.0). v5.1 clamped the
# draw to 5; the tightened gate needs 2. A CLAMP, not a re-draw, exactly as in
# v9: the same rng stream must still produce the same azimuth, side sign and
# zone span, or `fnd-slab-apron-near` loses the pair-seeded azimuth it shares
# with the frozen d12 / d16 rungs. Verified over all 80 maps: the apron azimuth
# is bit-identical at clamps 5, 4, 3 and 2.
#
# The capacity family therefore lands on a single inner offset. That is the
# point: capacity is the axis, and the standoff was an uncontrolled nuisance
# variable that produced exactly the "too far" complaint. The band still varies
# radially -- c3x is a thick band, c1p2 a thin one hugging the slab.
APRON_PROXIMITY_STANDOFF_MAX = 2

# Per-draw acceptance of the tightened gate, 24 dump draws x 80 maps: clamp 3
# leaves 6 maps with ZERO accepting draws (they would be forced onto a re-rolled
# dig and lose their shared excavation), clamp 2 leaves none.
APRON_PROXIMITY_STANDOFF_MAX_REJECTED = {3: 6, 2: 0}

# --------------------------------------------------------------------------
# v6.2 -- the trench debt (TRENCH_WIDTH_AUDIT.md, R1 / R4 / deferred proximity)
#
# R1. The reserved lane's inner edge must reach the trench's FAR edge. For a
# straight trench the geometry is exact:
#
#     far-edge reach from the lane's inner edge = lane_inner + (half_width + 0.5)
#
# `LANE_INNER_CHOICES` / `LANE_INNER_BAND` are measured from the SPINE, so the
# band was never width-aware, and `build_lane`'s only lateral tests are the band
# itself and `lane_hits_dig` (which protects the NEAR edge). The predictor is
# perfect bank-wide: all 26 v6.1 maps with `lane_inner + half_width + 0.5 >
# 11.375` have lane-only completability < 1, all 26 are `half_width` 2, and no
# map inside the boom fails. The gate is taken with one tile of slack, which
# leaves `half_width` 2 the rungs {6.5, 7.0, 7.5} (`lane_hits_dig` needs only
# `inner > half_width + 1`, so it stays satisfiable) and costs `half_width` 1
# nothing below 8.875.
#
# `half_width + 0.5` is the trench's OWN lateral extent, deliberately NOT
# `trench_branch_side_extent_tiles`: on a comb that is the branch reach (up to 20
# tiles) and would reject every comb.
LANE_FAR_REACH_MARGIN_TILES = 1.0
LANE_FAR_REACH_MAX_TILES = R_MAX_TILES - LANE_FAR_REACH_MARGIN_TILES  # 10.375

# R4. `seg2` draws its fold as a 120 or 150 deg heading change, i.e. a 60 or 30
# deg RELATIVE turn between the two legs. At `half_width` 2 the shallow fold
# merges two 5-tile legs into a single 8-12 tile solid wedge -- `curriculum-v6m-
# 2311` reaches an inscribed thickness of 10.0 tiles (5.71 m) with 10.7% of its
# cells wider than nominal, and reads as a blob rather than a segmented trench.
# It was always completable (`marginC` >= +1.38), so this is a taxonomy rule, not
# a feasibility one: at the wide half_width the fold must be a real corner.
# `half_width` 1 keeps both turns -- a 3-tile corridor cannot fill its own fold.
SEG2_MIN_RELATIVE_TURN_DEG = 60.0
SEG2_WIDE_HALF_WIDTH = 2.0
# The builder draws inside its own retry loop, so a violating draw is re-drawn by
# calling it again (which advances the same rng) rather than by reaching into it.
SEG2_TURN_REDRAW_ATTEMPTS = 24

# The deferred proximity debt, and why two of the three names come off the list
# by being RECLASSIFIED rather than regenerated.
#
# `trench_flank` keeps its bulk on the lane side and wraps the trench ENDS (v9
# docstring). On a straight trench that serves every dig cell -- realised max
# 6.00 tiles, the gate is 11.375. On a comb it cannot: §8.6 puts every branch on
# the side AWAY from the lane, so a branch tip 7-17 tiles off the spine is 12-15
# tiles from a strictly one-side bank. `trn-net4-side1-road` missed the gate on
# 15 of 16 maps (worst p95 12.72 / max 15.00), `trn-net3-side1-road` on 6 of 16.
#
# The v6.1 sweep named the fix: a down-line bank segment on the far (road) side.
# It was BUILT and MEASURED in this pass, then removed rather than shipped -- the
# band admitted cells hugging a BRANCH (nearest-arm assignment, each branch's
# strip kept as its own connected segment, because the spine's far flank stays
# out). It works on the metric it was aimed at -- over 3 maps x 60 attempts of
# `trn-net4-side1-road` the `adjacent_proximity_*` rejections go to ZERO -- and it
# destroys the condition:
#
#   bank                       plan_cost_near   plan_cost_far   plan_delta
#   v6.1, one-side (hauling)   0.132            0.013           0.100-0.194
#   v6.2 probe, down-line      0.004            0.002           0.000 (max 0.065)
#
# U10's gate is `plan_delta >= 0.10`: starting from the near side must be
# materially worse. On these two conditions that pressure IS the haul. Give every
# dig cell a designated dump cell within one arm swing and BOTH plans cost ~0.4%
# -- there is no ordering left to get wrong, and 70 of 70 draws are rejected on
# `plan_start_side_contract`. The two gates are mutually exclusive on a
# `trench_flank` comb; the down-line bank is therefore NOT shipped.
#
# What was actually wrong is the class. v6.1 already exempts every OTHER U10
# planning rung from the proximity gate -- `fnd-slab-side1`, `fnd-slab-side1-obj`
# and `fnd-proc-side1-road` are BOUNDED-ASYMMETRIC "because the distance IS the
# planning pressure" -- and classified these two as ADJACENT only because they
# carry a trench dump style. The measurement above is the direct evidence for the
# general rule: **every U10 rung is bounded-asymmetric**. See
# `adjacent_proximity_class`.
#
# The way to have both would be to move the ordering pressure off distance and
# onto CAPACITY (a tight adjacent bank self-blocks when spoil is consumed
# nearest-first). That is a capacity-level change to a T2 rung, i.e. a taxonomy
# edit, and is left for the spec rather than smuggled in here.


def layout_for(condition, map_index: int):
    """v9's layout, with the v6.1 apron standoff clamp applied on top.

    v9 is called first and unmodified so every draw it makes is consumed in the
    same order; only the standoff VALUE is then clamped.
    """
    layout = v9.layout_for(condition, map_index)
    if condition.dump_style == "capacity_apron":
        layout = dataclasses.replace(
            layout, standoff=min(layout.standoff, APRON_PROXIMITY_STANDOFF_MAX)
        )
    return layout


def adjacent_proximity_class(condition) -> str:
    """ADJACENT / BOUNDED-ASYMMETRIC / EXEMPT for one condition."""
    if condition.dump_style in PROXIMITY_EXEMPT_STYLES:
        return "exempt"
    if condition.dump_style in BOUNDED_ASYMMETRIC_STYLES:
        return "bounded_asymmetric"
    # v6.2: a U10 rung's dig->dump distance is the variable under test. Measured,
    # not assumed: on a bank that clears the adjacent gate, `plan_delta` collapses
    # from 0.100-0.194 to 0.000 because both plans cost ~0.4%.
    if PROXIMITY_BOUNDED_BY_PLANNING and condition.planning:
        return "bounded_asymmetric"
    return "adjacent"


# --------------------------------------------------------------------------
# D1 -- gapped ring band


def arm_overlap_absolute(dig: np.ndarray, dig_meta: dict[str, Any]) -> dict[str, Any]:
    """U4 in ABSOLUTE units: how much future work a greedy dump would bury.

    `arm_overlap_fraction` is a RATIO whose denominator is one arm's 6-tile spoil
    flank. That denominator shrinks with the arm, so the ratio rises as a comb
    gets smaller even though there is strictly LESS work to bury — which is what
    makes it the wrong instrument for the D2 mini rungs (spec v6 §2; see
    GENERATION_NOTES "the mild-overlap band"). The design intent behind "mild
    ordering pressure" is the absolute amount of buried future work, so it is
    measured here in cells and as a share of the excavation:

    * ``arm_overlap_cells_worst`` -- the numerator of `arm_overlap_fraction`,
      on the same (arm, side) pair the ratio selects, so the two are consistent;
    * ``arm_overlap_union_cells`` -- DISTINCT excavation cells lying in some
      arm's natural spoil bank, i.e. the work actually at risk of burial;
    * ``arm_overlap_union_fraction_of_dig`` -- that union over the excavation.
      Scale-free, and its denominator shrinks WITH the trench, so a small comb
      reads mild when it is mild.
    """
    arms = arm_masks(dig, dig_meta)
    points = v9._as_arms(dig_meta["trench_arms"])
    if len(arms) < 2:
        return {
            "arm_overlap_cells_worst": 0,
            "arm_overlap_cells_per_arm": [],
            "arm_overlap_union_cells": 0,
            "arm_overlap_union_fraction_of_dig": 0.0,
        }
    per_arm_cells: list[int] = []
    union = np.zeros_like(dig, dtype=bool)
    worst_cells, worst_fraction = 0, -1.0
    for index, arm in enumerate(arms):
        if not arm.any():
            per_arm_cells.append(0)
            continue
        others = dig & ~arm
        best_cells, best_fraction, best_mask = 0, -1.0, None
        for side in (-1.0, 1.0):
            flank = v8.arm_flank_band(
                dig, arm, np.asarray(points[index], float), side
            )
            if int(flank.sum()) < 20:
                continue
            hit = flank & others
            fraction = float(int(hit.sum()) / int(flank.sum()))
            if fraction > best_fraction:
                best_fraction, best_cells, best_mask = fraction, int(hit.sum()), hit
        per_arm_cells.append(best_cells)
        if best_mask is not None:
            union |= best_mask
        if best_fraction > worst_fraction:
            worst_fraction, worst_cells = best_fraction, best_cells
    dig_cells = max(1, int(dig.sum()))
    return {
        "arm_overlap_cells_worst": int(worst_cells),
        "arm_overlap_cells_per_arm": per_arm_cells,
        "arm_overlap_union_cells": int(union.sum()),
        "arm_overlap_union_fraction_of_dig": round(int(union.sum()) / dig_cells, 5),
    }


def ring_gap_sectors(
    dig: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """1-3 forbidden angular sectors of 15-40 deg each, total 15-90 deg.

    Drawn per map, in the dig's own polar frame. Returns the sector mask over
    the whole grid plus the metadata that describes it; the caller subtracts the
    mask from the DESIGNATED dump only -- obstacles and dumpability are not
    touched (spec v6 section 2 D1: "gap sectors count as non-designated, not
    obstacles").
    """
    _, angles, _ = _polar(dig)
    count = int(rng.choice(RING_GAP_COUNT_CHOICES))
    widths: list[float] = []
    for _ in range(RING_GAP_DRAW_ATTEMPTS):
        draw = [float(rng.uniform(*RING_GAP_WIDTH_DEG)) for _ in range(count)]
        if RING_GAP_TOTAL_DEG[0] <= sum(draw) <= RING_GAP_TOTAL_DEG[1]:
            widths = draw
            break
    if not widths:
        # 3 x [15,40] can exceed 90; fall back to the widest legal equal split
        # rather than re-drawing forever.
        widths = [min(RING_GAP_TOTAL_DEG[1] / count, RING_GAP_WIDTH_DEG[1])] * count
        widths = [max(w, RING_GAP_WIDTH_DEG[0]) for w in widths]

    azimuths: list[float] = []
    for _ in range(RING_GAP_DRAW_ATTEMPTS):
        candidate = [float(rng.uniform(0.0, 360.0)) for _ in range(count)]
        ok = True
        for i in range(count):
            for j in range(i + 1, count):
                separation = abs((candidate[i] - candidate[j] + 180.0) % 360.0 - 180.0)
                needed = (widths[i] + widths[j]) / 2.0 + RING_GAP_EDGE_SEPARATION_DEG
                if separation < needed:
                    ok = False
        if ok:
            azimuths = candidate
            break
    if not azimuths:
        start = float(rng.uniform(0.0, 360.0))
        azimuths = [(start + k * 360.0 / count) % 360.0 for k in range(count)]

    per_sector = []
    sectors = np.zeros(dig.shape, dtype=bool)
    for azimuth, width in zip(azimuths, widths):
        half = math.radians(width) / 2.0
        mask = v2.angle_difference(angles, math.radians(azimuth)) <= half
        per_sector.append(mask)
        sectors |= mask
    meta = {
        "ring_gap_count": count,
        "ring_gap_widths_deg": [round(w, 3) for w in widths],
        "ring_gap_azimuths_deg": [round(a % 360.0, 3) for a in azimuths],
        "ring_gap_total_deg": round(float(sum(widths)), 3),
    }
    return sectors, meta, per_sector


def gapped_ring_band(
    dig: np.ndarray,
    target_area: int,
    rng: np.random.Generator,
    blocked: np.ndarray | None = None,
    gaps: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """v8.ring_band with D1's forbidden sectors removed from the ALLOWED set.

    Two differences from v8, both deliberate:

    * ``largest_component`` is taken on the UNGAPPED annulus and the gaps are
      subtracted afterwards. Taking it after would collapse the ring to a single
      arc, which is a one-sided apron, not a notched ring.
    * the band grows to the same ``target_area`` inside the remaining arcs, so
      the capacity gate ([3.0, 4.0]x designated AND reachable) is unchanged and
      the gaps cost angular coverage rather than capacity.
    """
    distance = ndi.distance_transform_edt(~dig)
    allowed = (distance >= 1) & (distance <= RING_BAND_MAX_RADIUS) & _interior(1)
    if blocked is not None:
        allowed &= ~blocked
    allowed = base.largest_component(allowed)
    if gaps is not None:
        allowed = allowed & ~gaps
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


def ring_gap_metrics(
    dig: np.ndarray,
    dump: np.ndarray,
    occupancy: np.ndarray,
    gaps: np.ndarray | None,
    per_sector: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Is the gap a real, readable notch in the designated mask?

    Measured on the RIM -- the 1-3 tile ring around the excavation, which is
    where a station at the dig swings to. Blockers (road corridor, objects) are
    excluded from both numerator and denominator so the statistic isolates D1.
    """
    distance = ndi.distance_transform_edt(~dig)
    rim = (distance >= 1) & (distance <= RING_GAP_RIM_TILES) & _interior(1)
    rim &= ~occupancy
    rim_cells = int(rim.sum())
    non_designated = int((rim & ~dump).sum())
    metrics = {
        "ring_rim_cells": rim_cells,
        "ring_rim_nondesignated_fraction": round(
            non_designated / max(1, rim_cells), 5
        ),
    }
    if gaps is None:
        metrics["ring_gap_notch_cells"] = 0
        metrics["ring_gap_notch_cells_per_gap"] = []
        metrics["ring_gap_designated_leak_cells"] = 0
        return metrics
    # a notch cell: inside a gap sector, on the rim, free ground, not designated
    notch = rim & gaps & ~dump
    metrics["ring_gap_notch_cells"] = int(notch.sum())
    # Per sector, not just the union: on a spread-out dig (strips, proc) an
    # angular sector taken from the CENTROID can be swallowed by the excavation
    # and clip the band only far out, which is a wedge in the middle of nowhere
    # rather than a notch the machine has to read. Every sector must bite the rim.
    metrics["ring_gap_notch_cells_per_gap"] = (
        [int((rim & sector & ~dump).sum()) for sector in per_sector]
        if per_sector
        else []
    )
    # by construction zero -- asserted, not assumed
    metrics["ring_gap_designated_leak_cells"] = int((dump & gaps).sum())
    return metrics


# --------------------------------------------------------------------------
# D2 -- mini-junction geometries


class GeometryFactoryV10(v9.GeometryFactoryV9):
    """v9 geometries plus the three short-extent mini junctions."""

    @staticmethod
    def _source_sha256(source: np.ndarray) -> str:
        source = np.ascontiguousarray(source)
        digest = hashlib.sha256()
        digest.update(source.dtype.str.encode())
        digest.update(np.asarray(source.shape, dtype=np.int64).tobytes())
        digest.update(source.tobytes())
        return digest.hexdigest()

    def slab(self, rng, radius, angle):
        placed, metadata = super().slab(rng, radius, angle)
        if placed is not None:
            source = self.slab_sources[metadata["foundation_source_index"]]
            metadata["foundation_source_sha256"] = self._source_sha256(source)
        return placed, metadata

    def slab_lg(self, rng, radius, angle):
        placed, metadata = super().slab_lg(rng, radius, angle)
        if placed is not None:
            source = self.scale_sources[metadata["foundation_source_index"]]
            metadata["foundation_source_sha256"] = self._source_sha256(source)
        return placed, metadata

    # ---- v6.2 R4: no shallow seg2 fold at the wide half_width ------------

    @staticmethod
    def _seg2_relative_turn_deg(meta: dict[str, Any]) -> float:
        """The angle between the two legs, from the recorded segment headings.

        The builder stores headings mod 180, so a 150 deg heading change reads as
        a 30 deg relative turn and a 120 deg change as 60 -- which is exactly the
        quantity R4 is about.
        """
        headings = [float(h) for h in meta["trench_segment_headings_deg"]]
        delta = abs(headings[0] - headings[1]) % 180.0
        return min(delta, 180.0 - delta)

    def seg2(self, rng, radius, angle, heading_deg, _topology=None):
        """v8's hook, re-drawn while the fold is a shallow wedge at half_width 2.

        `_segmented` draws the half_width and the turn together inside its own
        retry loop, so the honest way to add a rule about their COMBINATION is to
        re-draw the whole geometry -- which advances the same rng -- rather than
        to reach into the loop and re-roll one variable.
        """
        for _ in range(SEG2_TURN_REDRAW_ATTEMPTS):
            placed, meta = self._segmented(rng, radius, angle, heading_deg, 2)
            if placed is None:
                return None, {}
            wide = float(meta["trench_half_width_tiles"]) >= SEG2_WIDE_HALF_WIDTH
            relative = self._seg2_relative_turn_deg(meta)
            meta["trench_relative_angle_deg"] = round(relative, 1)
            if not wide or relative >= SEG2_MIN_RELATIVE_TURN_DEG - 1e-6:
                return placed, meta
        return None, {}

    def _mini_common(self, dig, segments, heading_deg, radius, angle, level, extra):
        placed, meta = self._finish_trench(
            dig, segments, MINI_HALF_WIDTH, heading_deg, radius, angle,
            MINI_CELL_BAND[level], extra,
        )
        if placed is None:
            return None, {}
        worst, per_arm = arm_overlap_fraction(placed, meta)
        low, high = MINI_OVERLAP_GATE[level]
        if not low <= worst <= high:
            return None, {}
        if arm_merge_fraction(placed, meta) > ARM_MERGE_MAX:
            return None, {}
        union = np.zeros_like(placed)
        for arm in arm_masks(placed, meta):
            union |= arm
        if int(placed.sum()) > ARM_FILL_MAX * max(1, int(union.sum())):
            return None, {}
        meta["arm_overlap_fraction"] = round(worst, 4)
        meta["arm_overlap_per_arm"] = per_arm
        meta["mini_scale"] = "s"
        meta["mini_dig_cells_cap"] = MINI_DIG_CELLS_MAX[level]
        meta["mini_volume_fraction_of_standard"] = round(
            int(placed.sum()) / STANDARD_DIG_CELLS_MEDIAN[level], 4
        )
        meta["mini_overlap_spec_band"] = list(MINI_OVERLAP_SPEC_BAND)
        meta["mini_overlap_gate"] = list(MINI_OVERLAP_GATE[level])
        meta["mini_overlap_spec_band_satisfied"] = bool(
            MINI_OVERLAP_SPEC_BAND[0] <= worst <= MINI_OVERLAP_SPEC_BAND[1]
        )
        return placed, meta

    def tee_s(self, rng, radius, angle, heading_deg, _topology=None):
        heading = math.radians(heading_deg)
        direction = np.array([math.sin(heading), math.cos(heading)])
        for _ in range(160):
            spine = float(rng.uniform(*MINI_SPINE_BAND["tee"]))
            branch = float(rng.uniform(*MINI_BRANCH_RATIO["tee"])) * spine
            junction_along = float(rng.uniform(-0.14, 0.14)) * spine
            turn = float(rng.choice([60.0, 90.0, 120.0]))
            branch_heading = (heading_deg + turn) % 360.0
            branch_direction = np.array([
                math.sin(math.radians(branch_heading)),
                math.cos(math.radians(branch_heading)),
            ])
            main = (
                np.vstack([-direction * spine / 2, direction * spine / 2]) + MAP_CENTRE
            )
            junction = direction * junction_along + MAP_CENTRE
            branch_points = np.vstack([junction, junction + branch_direction * branch])
            dig = rasterize_segments(main, MINI_HALF_WIDTH)
            dig |= rasterize_segments(branch_points, MINI_HALF_WIDTH)
            placed, meta = self._mini_common(
                dig, [main, branch_points], heading_deg, radius, angle, "tee",
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
            if placed is not None:
                return placed, meta
        return None, {}

    def _net_s(self, rng, radius, angle, heading_deg, topology, n_branches, level):
        heading = math.radians(heading_deg)
        direction = np.array([math.sin(heading), math.cos(heading)])
        turns_table = v9.NET3_TURN_CYCLE if n_branches == 2 else v9.NET4_TURN_CYCLE
        for _ in range(200):
            spine = float(rng.uniform(*MINI_SPINE_BAND[level]))
            spread = float(rng.uniform(*MINI_JUNCTION_SPREAD))
            positions_base = (
                np.array([-spread, spread])
                if n_branches == 2
                else np.array([-spread, 0.0, spread])
            )
            main = (
                np.vstack([-direction * spine / 2, direction * spine / 2]) + MAP_CENTRE
            )
            positions = positions_base + rng.uniform(
                -MINI_JUNCTION_JITTER, MINI_JUNCTION_JITTER, size=n_branches
            )
            dig = rasterize_segments(main, MINI_HALF_WIDTH)
            segments = [main]
            ratios = []
            turns = turns_table[topology]
            for branch_index, along in enumerate(positions):
                junction = direction * float(along * spine) + MAP_CENTRE
                turn = float(turns[branch_index])
                branch_heading = (heading_deg + turn) % 360.0
                branch_direction = np.array([
                    math.sin(math.radians(branch_heading)),
                    math.cos(math.radians(branch_heading)),
                ])
                length = float(rng.uniform(*MINI_BRANCH_RATIO[level])) * spine
                points = np.vstack([junction, junction + branch_direction * length])
                ratios.append(length / spine)
                dig |= rasterize_segments(points, MINI_HALF_WIDTH)
                segments.append(points)
            placed, meta = self._mini_common(
                dig, segments, heading_deg, radius, angle, level,
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
            if placed is not None:
                return placed, meta
        return None, {}

    def net3_s(self, rng, radius, angle, heading_deg, topology):
        return self._net_s(rng, radius, angle, heading_deg, topology, 2, "net3")

    def net4_s(self, rng, radius, angle, heading_deg, topology):
        return self._net_s(rng, radius, angle, heading_deg, topology, 3, "net4")


# --------------------------------------------------------------------------
# condition table -- v5 verbatim, the three minis APPENDED
#
# Appending matters: `condition_index` seeds every map's rng and indexes
# `sample_index_of`, so inserting anywhere but the end would re-key the whole
# bank and break byte-identity on the carried conditions.

@dataclass(frozen=True)
class ConditionSpec:
    """v9's spec plus the v6 `scale` axis and a separable dig-bank key.

    `geometry_level` stays a real taxonomy level (`tee`), so tier computation is
    unaffected; `bank_level` is what the dig bank is keyed by, so the minis get
    their OWN dig entries (spec v6 §2 D2) without colliding with the standard
    variants that share the same geometry.
    """

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
    scale_level: str = "std"
    bank_level: str | None = None

    @property
    def geometry(self) -> str:
        return GEOMETRY_LEVEL_SOURCE[self.geometry_level]

    @property
    def dig_bank_level(self) -> str:
        return self.bank_level or self.geometry_level

    @property
    def release(self):
        return tax.RELEASES["v6-main"]

    @property
    def levels(self) -> dict[str, str]:
        return {
            "family": "fnd" if self.family == "foundation" else "trn",
            "geometry": self.geometry_level,
            "dump": self.dump_level,
            "capacity": self.capacity_level,
            "site": self.site_level,
            "distance": self.distance_level,
            "scale": self.scale_level,
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


_V9_FIELDS = tuple(f.name for f in fields(v9.ConditionSpec))


def _carry(spec) -> ConditionSpec:
    return ConditionSpec(**{name: getattr(spec, name) for name in _V9_FIELDS})


MINI_CONDITIONS: tuple[ConditionSpec, ...] = tuple(
    ConditionSpec(
        id=f"trn-{level}-side2-s",
        anchor=None if level == "tee" else "trn-tee-side2-s",
        dataset="main",
        family="trench",
        geometry_level=level,
        dump_level="side2",
        capacity_level="generous",
        site_level="clean",
        distance_level="unspec",
        dump_style="trench_band",
        dump_layout="easy_surround",
        scale_level="s",
        bank_level=f"{level}@s",
    )
    for level in ("tee", "net3", "net4")
)

MAIN_CONDITIONS: tuple[ConditionSpec, ...] = (
    *(_carry(c) for c in v9.MAIN_CONDITIONS),
    *MINI_CONDITIONS,
)

DATASETS = {
    "main": DatasetSpec(
        name="main",
        conditions=MAIN_CONDITIONS,
        maps_per_condition=16,
        map_id_prefix="curriculum-v6m-axis-v2",
        release="v6-main",
        gate_turn_dump=True,
        gate_lane_band=True,
        gate_staging=False,
    ),
}

MINI_CONDITION_IDS = frozenset(c.id for c in MINI_CONDITIONS)
# v6.1: the five capacity aprons are no longer carried from v5 either -- the
# tightened adjacent proximity gate moved their dump masks. Kept as its own set
# so `check_byte_identity_v6.py` stops claiming a v5 identity that is now false,
# and so the two regeneration reasons stay distinguishable in the receipts.
APRON_PROXIMITY_CONDITIONS = frozenset(
    c.id for c in v9.MAIN_CONDITIONS if c.dump_style == "capacity_apron"
)
# v6.2: every trench condition is regenerated -- the width-aware lane (R1) is a
# lane-placement change that applies to all 14, and the seg2 fold rule (R4) plus
# the enforced adjacent-proximity gate touch four of them. Named as its own set
# for the same reason as the aprons: the regeneration REASONS stay separable in
# the receipts.
TRENCH_CONDITIONS = frozenset(c.id for c in MAIN_CONDITIONS if c.family == "trench")
REGENERATED_CONDITIONS = (
    RING_GAP_CONDITIONS
    | MINI_CONDITION_IDS
    | APRON_PROXIMITY_CONDITIONS
    | TRENCH_CONDITIONS
)
CARRIED_CONDITIONS = (
    frozenset(c.id for c in v9.MAIN_CONDITIONS)
    - RING_GAP_CONDITIONS
    - APRON_PROXIMITY_CONDITIONS
    - TRENCH_CONDITIONS
)
assert len(MAIN_CONDITIONS) == 32, len(MAIN_CONDITIONS)
assert len(RING_GAP_CONDITIONS) == 7, sorted(RING_GAP_CONDITIONS)
assert len(APRON_PROXIMITY_CONDITIONS) == 5, sorted(APRON_PROXIMITY_CONDITIONS)
assert len(TRENCH_CONDITIONS) == 14, sorted(TRENCH_CONDITIONS)
assert len(CARRIED_CONDITIONS) == 6, sorted(CARRIED_CONDITIONS)

WALL_CONDITIONS = v9.WALL_CONDITIONS
PLANNING_CONDITIONS = frozenset(c.id for c in MAIN_CONDITIONS if c.planning)


def _levels_of(condition: ConditionSpec) -> dict[str, str]:
    return condition.levels


def _scale_token(condition: ConditionSpec) -> str:
    return "short_arm_mini" if condition.scale_level == "s" else ""


def assert_conditions_match_taxonomy(dataset: DatasetSpec) -> None:
    spec = tax.RELEASES[dataset.release]
    table = spec.condition_table
    conditions = dataset.conditions
    assert {c.id for c in conditions} == set(table), (
        f"{dataset.name}: condition set mismatch: "
        f"{sorted({c.id for c in conditions} ^ set(table))}"
    )
    for condition in conditions:
        levels = _levels_of(condition)
        derived = tax.condition_id(levels, spec)
        assert derived == condition.id, f"{condition.id}: id grammar gives {derived}"
        expected_tier, expected_anchor = table[condition.id]
        assert tax.tier(levels, spec) == expected_tier, (
            f"{condition.id}: tier {tax.tier(levels, spec)} != table {expected_tier}"
        )
        assert condition.anchor == expected_anchor, (
            f"{condition.id}: anchor {condition.anchor} != table {expected_anchor}"
        )
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
                    "scaleBand": _scale_token(c) or None,
                },
            }
            for c in conditions
        ],
        spec,
    )
    # U8 carries forward: the transport track is unchanged and disjoint.
    transport = tax.RELEASES["v5-transport"].condition_table
    assert not ({c.id for c in conditions} & set(transport))


# --------------------------------------------------------------------------
# dig bank -- mini levels get their OWN entries


class DigBankV10(v9.DigBankV9):
    def _headings(self, level: str) -> list[float]:
        """Return the 15 degree trench-axis lattice, permuted per level.

        The mini levels may use a SUBSET of the lattice — see MINI_HEADINGS for
        why `net3@s` drops the two cardinal headings — so the axis set is looked
        up per level instead of being the global lattice.
        """
        axes = MINI_HEADINGS.get(MINI_GEOMETRY_OF.get(level, ""), TRENCH_AXES_DEG)
        rng = rng_from(SEED_BASE, 4242, GEOMETRY_LEVEL_INDEX[level])
        order = [axes[i] for i in rng.permutation(len(axes))]
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
            schedule = v9.TRENCH_PLACEMENT_RADII_WIDE
        elif level in TRENCH_LEVELS:
            schedule = v9.TRENCH_PLACEMENT_RADII
        else:
            schedule = v9.PLACEMENT_RADII
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
        source = GEOMETRY_LEVEL_SOURCE[level]
        meta = {
            **meta,
            "geometry": source,
            "geometry_hardness": GEOMETRY_HARDNESS[source],
            "dig_bank_seed": seed,
            "dig_bank_salt": salt,
            "dig_bank_attempt": attempt,
            "dig_centroid_offset_tiles": round(v9.centroid_offset(dig), 3),
            "dig_border_margin_tiles": v9.border_margin(dig),
        }
        if level in MULTI_ARM_LEVELS and "arm_overlap_fraction" not in meta:
            worst, per_arm = arm_overlap_fraction(dig, meta)
            meta["arm_overlap_fraction"] = round(worst, 4)
            meta["arm_overlap_per_arm"] = per_arm
        return dig, meta

    def _acceptable(self, level, dig, meta, against):
        """Keep physical/source gates and reject only exact dig duplicates."""
        if level in self.t0_levels:
            coverage = v9.dig_only_coverage(dig)
            if coverage < v9.T0_DIG_COVERAGE_MIN:
                return "dig_bank_t0_coverage"
            meta["dig_only_workspace_coverage"] = round(coverage, 5)
        source = meta.get("foundation_source_index")
        for other, other_meta in against:
            if source is not None and other_meta.get("foundation_source_index") == source:
                return "dig_bank_source_reuse"
            if np.array_equal(dig, other):
                return "dig_bank_exact_duplicate"
        return ""


# --------------------------------------------------------------------------
# v6.2 R1 -- the width-aware lane


def build_lane(dig, dig_meta, layout):
    """v9's lane, with the far-edge reach gate the audit's R1 asks for.

    v9's version is copied rather than wrapped: the gate belongs INSIDE the
    candidate loop (a rejected offset must let a closer one win, not fail the
    map), and the candidate ranking is the thing being changed. Everything else
    -- the width/side/offset grid, the usable-strip proxy, the ranking key -- is
    v9's, verbatim.

    The added test is one line:

        lane_inner + half_width + 0.5 <= LANE_FAR_REACH_MAX_TILES

    i.e. from the lane's inner edge the machine must still reach the far row of
    the excavation, with a tile of slack under `r_max`.
    """
    _, direction, normal, axial, lateral = spine_fields(dig.shape, dig_meta)
    origin, _, _ = v9.spine_frame(dig_meta)
    dig_axial = axial[dig]
    extent = float(dig_axial.max() - dig_axial.min())
    axial_centre = float((dig_axial.max() + dig_axial.min()) / 2.0)
    heading_deg = float(dig_meta["trench_global_angle_deg"])
    # The trench's own lateral half-extent, NOT the branch reach -- see
    # LANE_FAR_REACH_MAX_TILES.
    spine_far_extent = float(dig_meta["trench_half_width_tiles"]) + 0.5

    extents = {sign: float((lateral[dig] * sign).max()) for sign in (-1.0, 1.0)}
    signs = sorted(
        (-1.0, 1.0),
        key=lambda sign: (round(extents[sign], 3), -sign * float(layout.lane_sign)),
    )
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
    for width in v9.LANE_WIDTH_CHOICES:
        for sign in signs:
            for inner in v9.LANE_INNER_CHOICES:
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
                    extent + 2 * v9.LANE_END_MARGIN,
                    float(width),
                    math.radians(heading_deg),
                )
                if not lane.any() or (lane & dig).any():
                    failure = "lane_hits_dig"
                    continue
                if v9.border_margin(lane) < 1:
                    failure = "lane_off_map"
                    continue
                lane_axial = axial[lane]
                if lane_axial.min() > dig_axial.min() or lane_axial.max() < dig_axial.max():
                    failure = "lane_too_short"
                    continue
                realised = float((lateral[lane] * sign).min())
                if not (
                    v9.LANE_INNER_BAND[0] - 1e-6
                    <= realised
                    <= v9.LANE_INNER_BAND[1] + 1e-6
                ):
                    failure = "lane_inner_band_infeasible"
                    continue
                # v6.2 R1: can the far row of the trench be dug from the lane?
                far_reach = realised + spine_far_extent
                if far_reach > LANE_FAR_REACH_MAX_TILES + 1e-6:
                    failure = "lane_far_edge_out_of_reach"
                    continue
                proxy = float((lane & diggable).sum() / max(1, int(lane.sum())))
                meta = {
                    "lane_sign": int(sign),
                    "lane_inner_offset_tiles": round(realised, 3),
                    "lane_inner_requested_tiles": inner,
                    "lane_inner_dig_gap_tiles": round(float(distance[lane].min()), 3),
                    "lane_far_reach_tiles": round(far_reach, 3),
                    "lane_usable_proxy": round(proxy, 5),
                    "lane_width_tiles": width,
                    "lane_cells": int(lane.sum()),
                    "lane_centre_y": round(float(centre[0]), 3),
                    "lane_centre_x": round(float(centre[1]), 3),
                    "trench_working_length_tiles": round(extent, 3),
                }
                key = (
                    proxy >= v9.LANE_USABLE_PROXY_TARGET,
                    -abs(inner - layout.lane_inner),
                    width,
                    proxy,
                )
                candidates.append((key, lane, meta))
    if not candidates:
        raise RuntimeError(failure)
    _, lane, meta = max(candidates, key=lambda item: item[0])
    return lane, meta


# --------------------------------------------------------------------------
# dump construction -- only the ring branch changes


def build_dump(condition, dig, dig_meta, layout, rng, blocked=None, lane_sign=1,
               gaps=None):
    style = condition.dump_style
    if condition.capacity_level in v9.APRON_CAPACITY_BANDS:
        band = v9.APRON_CAPACITY_BANDS[condition.capacity_level]
    elif condition.capacity_level == "tight":
        band = v9.TIGHT_CAPACITY_BAND
    elif condition.id in WALL_CONDITIONS:
        band = v9.WALL_SPLIT_CAPACITY_BAND
    elif style == "ring_band":
        band = v9.RING_BAND_DRAW
    else:
        band = v9.GENEROUS_CAPACITY_BANDS[style]
    factor = float(rng.uniform(*band))
    target_area = int(math.ceil(int(dig.sum()) * factor))

    if style == "ring_band":
        target, metadata = gapped_ring_band(dig, target_area, rng, blocked, gaps)
    elif style == "capacity_apron":
        target, metadata = v9.apron_sector(dig, layout, target_area, rng)
    elif style == "distance_apron":
        target, metadata = v9.distance_apron(
            dig, layout, target_area, rng, v9.DISTANCE_BINS[condition.distance_level]
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
        target, metadata = v9.trench_band(dig, dig_meta, layout, target_area, rng, blocked)
    elif style == "trench_flank":
        target, metadata = v9.trench_flank(
            dig, dig_meta, layout, target_area, rng, blocked, lane_sign
        )
    elif style == "trench_altsides":
        target, metadata = v9.trench_altsides(
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
#
# `make_sample` is v9's, with the ring branch threaded through the gap mask and
# the D1/D2 metadata appended. It is copied rather than monkey-patched so the
# whole accept/reject chain stays readable in one place.


def make_sample(condition, dataset, dig, dig_meta, layout, rng):
    is_trench = condition.family == "trench"
    heading = float(dig_meta.get("trench_global_angle_deg", 0.0))

    blocked = np.zeros_like(dig, dtype=bool)
    lane = np.zeros_like(dig, dtype=bool)
    lane_meta: dict[str, Any] = {}
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
                lane, v9.BACKWARD_LANE_STEPS_MIN,
            )
            if probe["backward_drive_drift_per_tile"] > v9.BACKWARD_DRIFT_PER_TILE_MAX:
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
        radius_range = (
            v9.ROAD_RADIUS_RANGE_RING if is_ring else v9.ROAD_RADIUS_RANGE_ZONED
        )
        floor = v9.ROAD_DIG_ANNULUS_STERILIZE_RANGE[0] if is_ring else 0.0
        protect = dig | lane if is_trench else dig
        road, road_meta = v9.make_two_border_road(
            dig, protect, rng, radius_range, dig_annulus_sterilize_min=floor
        )
        blocked |= road
    elif condition.site_level in ("obj", "obj1"):
        low, high = v9.OBJECT_BANDS[condition.site_level]
        requested_objects = int(rng.integers(low, high + 1))
        protect = dig | lane if is_trench else dig
        objects, areas = v9.place_objects_v7(dig, protect, requested_objects, rng)
        intrusion = v9.object_intrusion(dig, objects)
        if intrusion["annulus4_free_fraction"] < v9.OBJECT_NEAR_ANNULUS_FREE_MIN:
            return None, "object_corridor_blocked"
        lo, hi = v9.OBJECT_ANNULUS_BLOCK_BAND[condition.site_level]
        if not lo <= intrusion["annulus6_blocked_fraction"] <= hi:
            return None, "object_blockage_band_contract"
        if intrusion["annulus6_blocked_fraction"] > v9.OBJECT_TOTAL_BLOCK_MAX:
            return None, "object_total_blockage_contract"
        labels, n_clusters = ndi.label(objects, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = ndi.sum(objects, labels, range(1, n_clusters + 1)) if n_clusters else [0]
        if n_clusters and max(sizes) > v9.OBJECT_MAX_CLUSTER_CELLS:
            return None, "object_cluster_size_contract"
        object_meta = {
            **intrusion,
            "object_count": requested_objects,
            "object_clusters": int(n_clusters),
            "object_max_cluster_cells": int(max(sizes)) if n_clusters else 0,
            "object_footprint_profile": v9.OBJECT_FOOTPRINT_PROFILE,
            "object_area_cells_mean": round(float(np.mean(areas)), 3),
            "object_area_cells_min": int(min(areas)),
            "object_area_cells_max": int(max(areas)),
        }
        blocked |= objects

    # ---- D1: the forbidden sectors -------------------------------------
    gaps: np.ndarray | None = None
    gap_sectors: list[np.ndarray] | None = None
    gap_meta: dict[str, Any] = {}
    if is_ring:
        gaps, gap_meta, gap_sectors = ring_gap_sectors(dig, rng)

    lane_sign = int(lane_meta.get("lane_sign", layout.lane_sign))
    dump, dump_meta = build_dump(
        condition, dig, dig_meta, layout, rng,
        blocked if blocked.any() else None, lane_sign, gaps,
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
        site_meta.update(v9.zoned_road_metrics(dig, dump, corridor))
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
        wall, wall_meta = v9.offset_fence_wall(
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
    if condition.tier == 0 and gate.dig_workspace_coverage_post < v9.T0_DIG_COVERAGE_MIN:
        return None, "t0_dig_coverage_contract"

    # ---- capacity (unchanged) -------------------------------------------
    required = float(dump_meta["capacity_factor_required"])
    capacity_ratio = float(dump.sum() / max(1, dig.sum()))
    reachable_ratio = float(gate.reachable_dump_cells_post / max(1, dig.sum()))
    low = float(dump_meta["capacity_band_low"])
    high = float(dump_meta["capacity_band_high"])
    if (
        condition.capacity_level in v9.APRON_CAPACITY_BANDS
        or condition.capacity_level == "tight"
    ):
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
        if reachable_ratio + 1e-8 < v9.GENEROUS_REACHABLE_FLOOR:
            return None, "generous_capacity_floor"

    # ---- distance (unchanged) -------------------------------------------
    distance_metrics = v2.dig_to_dump_distance_metrics(dig, dump)
    if condition.dump_style == "distance_apron":
        target_median = v9.DISTANCE_BINS[condition.distance_level]
        realized = distance_metrics["dig_dump_distance_median_tiles"]
        if abs(realized - target_median) > v9.DISTANCE_BIN_TOLERANCE:
            return None, "distance_bin_contract"
    else:
        limits = v9.TRENCH_MEDIAN_LIMITS if is_trench else v9.FOUNDATION_MEDIAN_LIMITS
        limit = limits.get(condition.dump_style)
        if condition.id in WALL_CONDITIONS:
            limit = (
                v9.TRENCH_WALL_MEDIAN_LIMIT if is_trench else v9.WALL_SPLIT_MEDIAN_LIMIT
            )
        if limit is not None:
            limit += max(0, layout.standoff - 3)
            if distance_metrics["dig_dump_distance_median_tiles"] > limit:
                return None, "near_distance_contract"
    if condition.dump_level == "remote":
        if distance_metrics["dig_dump_distance_min_tiles"] < v9.REMOTE_MIN_DISTANCE_TILES:
            return None, "remote_distance_contract"
    if condition.dump_level == "split":
        if v9.max_angular_gap_degrees(dig, dump) < v9.SPLIT_MAX_ANGULAR_GAP_MIN:
            return None, "split_angular_gap_contract"
        if v9.border_margin(dump) < v9.SPLIT_PAD_BORDER_MARGIN:
            return None, "split_border_margin_contract"
    if condition.id in WALL_CONDITIONS:
        _, edge_gap, span = v9.pad_separation(dig, dump)
        if edge_gap < v9.SPLIT_PAD_EDGE_GAP_MIN:
            return None, "split_pad_edge_gap_contract"
        if span < v9.SPLIT_PAD_ANGULAR_SPAN_MIN:
            return None, "split_pad_angular_span_contract"
        if float(site_meta.get("wall_detour_ratio", 0.0)) < v9.WALL_DETOUR_MIN:
            return None, "wall_detour_contract"

    extra: dict[str, Any] = {}

    # ---- D1 gap metrics --------------------------------------------------
    if is_ring:
        extra.update(gap_meta)
        gap_metrics = ring_gap_metrics(dig, dump, occupancy, gaps, gap_sectors)
        extra.update(gap_metrics)
        if gap_metrics["ring_gap_designated_leak_cells"] != 0:
            return None, "ring_gap_leak_contract"
        if gap_metrics["ring_gap_notch_cells"] < RING_GAP_MIN_NOTCH_CELLS:
            return None, "ring_gap_notch_contract"
        if min(gap_metrics["ring_gap_notch_cells_per_gap"]) < (
            RING_GAP_MIN_NOTCH_CELLS_PER_GAP
        ):
            return None, "ring_gap_per_gap_notch_contract"

    # ---- proximity (v6.1: the adjacent class is held to one arm swing) ----
    proximity = dump_proximity(dig, dump)
    extra.update(proximity)
    # The class is a CONDITION property; it is published per condition in
    # generation_summary.json and VALIDATION.json, not per map, so `--resume`
    # never leaves a half-populated manifest column.
    if condition.dump_style in PROXIMITY_GATED_STYLES:
        if proximity["dump_proximity_p95_tiles"] > DUMP_PROXIMITY_P95_MAX:
            return None, "dump_proximity_p95_contract"
        if proximity["dump_proximity_max_tiles"] > DUMP_PROXIMITY_MAX_MAX:
            return None, "dump_proximity_max_contract"
    if (
        adjacent_proximity_class(condition) == "adjacent"
        and condition.id not in ADJACENT_PROXIMITY_DEFERRED
    ):
        if proximity["dump_proximity_p95_tiles"] > ADJACENT_PROXIMITY_P95_MAX:
            return None, "adjacent_proximity_p95_contract"
        if proximity["dump_proximity_max_tiles"] > ADJACENT_PROXIMITY_MAX_MAX:
            return None, "adjacent_proximity_max_contract"

    # ---- U2 direct service ----------------------------------------------
    coverage = tsvc.direct_service_coverage(target, occupancy, dumpability)
    extra["direct_service_coverage"] = round(coverage, 5)
    if condition.tier == 0 and coverage < v9.DIRECT_SERVICE_MIN:
        return None, "direct_service_contract"
    if condition.distance_level == "near":
        if distance_metrics["dig_dump_distance_p95_tiles"] > v9.DIRECT_SERVICE_P95_MAX:
            return None, "direct_service_p95_contract"

    # ---- U1 road-on-a-finite-band ---------------------------------------
    if condition.site_level == "road" and is_ring:
        open_band, _ = gapped_ring_band(
            dig, requested, np.random.default_rng(band_probe_seed), None, gaps
        )
        sterilized = float((corridor & open_band).sum() / max(1, int(open_band.sum())))
        extra["road_band_sterilized_fraction"] = round(sterilized, 4)
        if sterilized < v9.ROAD_BAND_STERILIZE_MIN:
            return None, "ring_road_band_contract"
        dig_share = site_meta["road_dig_annulus6_sterilized_fraction"]
        lo, hi = v9.ROAD_DIG_ANNULUS_STERILIZE_RANGE
        if not lo <= dig_share <= hi:
            return None, "ring_road_dig_annulus_contract"
        if site_meta["road_dig_annulus6_free_fraction"] < v9.ROAD_ANNULUS_FREE_MIN:
            return None, "ring_road_free_contract"
    elif condition.site_level == "road":
        if not v9.zoned_road_bites(site_meta):
            return None, "zoned_road_bite_contract"
        if site_meta["road_dig_annulus6_free_fraction"] < v9.ROAD_ANNULUS_FREE_MIN:
            return None, "zoned_road_free_contract"

    # ---- U4 overlap stressor --------------------------------------------
    if condition.geometry_level in MULTI_ARM_LEVELS:
        worst, per_arm = arm_overlap_fraction(dig, dig_meta)
        extra["arm_overlap_fraction"] = round(worst, 4)
        extra["arm_overlap_per_arm"] = per_arm
        # the same stressor in absolute units -- see arm_overlap_absolute()
        extra.update(arm_overlap_absolute(dig, dig_meta))
        if condition.scale_level == "s":
            level = condition.geometry_level
            lo, hi = MINI_OVERLAP_GATE[level]
            if not lo <= worst <= hi:
                return None, "mini_arm_overlap_contract"
            extra["mini_overlap_spec_band_satisfied"] = bool(
                MINI_OVERLAP_SPEC_BAND[0] <= worst <= MINI_OVERLAP_SPEC_BAND[1]
            )
        elif worst < ARM_OVERLAP_MIN:
            return None, "arm_overlap_contract"

    # ---- D2 volume cap ---------------------------------------------------
    if condition.scale_level == "s":
        level = condition.geometry_level
        cells = int(dig.sum())
        extra["mini_dig_cells_cap"] = MINI_DIG_CELLS_MAX[level]
        extra["mini_volume_fraction_of_standard"] = round(
            cells / STANDARD_DIG_CELLS_MEDIAN[level], 4
        )
        extra["mini_standard_variant_condition_id"] = tax.MINI_STANDARD_VARIANT[
            condition.id
        ]
        if cells > MINI_DIG_CELLS_MAX[level]:
            return None, "mini_volume_cap_contract"

    # ---- U7 lane geometry / lattice / backward drive (unchanged) ---------
    if is_trench:
        extra["lane_present"] = int(wants_lane)
        if wants_lane:
            metrics = v9.lane_metrics(dig, occupancy, dump, lane, lane_meta, heading)
        else:
            metrics = v9.lane_metrics_no_lane(dig, occupancy, heading, dig_meta)
        extra.update(lane_meta)
        extra.update(metrics)
        if wants_lane and metrics["lane_free_fraction"] < 1.0:
            return None, "lane_not_free_contract"
        if (wants_lane or dataset.gate_lane_band) and not metrics[
            "backward_drive_footprint_clear"
        ]:
            return None, "backward_drive_blocked_contract"
        if metrics["backward_drive_drift_per_tile"] > v9.BACKWARD_DRIFT_PER_TILE_MAX:
            return None, "backward_drive_drift_contract"
        if (
            wants_lane
            and int(lane_meta["lane_width_tiles"]) >= v9.LANE_WIDTH_FOR_RETREAT_GATE
            and metrics["backward_drive_lane_steps"] < v9.BACKWARD_LANE_STEPS_MIN
        ):
            return None, "backward_drive_lane_contract"
        spurs, notches = edge_irregularity(dig)
        extra["trench_edge_spurs"] = spurs
        extra["trench_edge_notches"] = notches
        if spurs or notches:
            return None, "trench_edge_regularity_contract"
        if dataset.gate_lane_band and wants_lane:
            inner = float(lane_meta["lane_inner_offset_tiles"])
            if not v9.LANE_INNER_BAND[0] - 1e-6 <= inner <= v9.LANE_INNER_BAND[1] + 1e-6:
                return None, "lane_inner_band_contract"
            # v6.2 R1: re-asserted here as well as in `build_lane`, because this
            # is the number the manifest publishes and the validator gates.
            if (
                float(lane_meta["lane_far_reach_tiles"])
                > LANE_FAR_REACH_MAX_TILES + 1e-6
            ):
                return None, "lane_far_reach_contract"

    # ---- turn-dump (unchanged) -------------------------------------------
    trench_probe = None
    if is_trench:
        trench_probe = {
            "heading_deg": heading,
            "arms": v9._as_arms(dig_meta["trench_arms"]),
            "half_width": float(dig_meta["trench_half_width_tiles"]),
            "lane": lane if wants_lane else None,
        }
    extra.update(tdump.measure(target, occupancy, dumpability, trench_probe, full=False))
    if dataset.gate_turn_dump:
        if extra["turn_dump_cov_strict"] < v9.TURN_DUMP_COV_MIN:
            return None, "turn_dump_cov_contract"
        if is_trench:
            floor = v9.STATION_DUMP_FRAC_MIN_BY_STYLE.get(
                condition.dump_style, v9.STATION_DUMP_FRAC_MIN
            )
            if condition.site_level != "clean":
                floor = min(floor, v9.STATION_DUMP_FRAC_MIN_BLOCKED_SITE)
            if extra["turn_dump_station_dump_frac"] < floor:
                return None, "station_dump_frac_contract"
            if wants_lane and extra["lane_usable_frac"] < v9.LANE_USABLE_FRAC_MIN:
                return None, "lane_usable_frac_contract"

    # ---- U10 start-side sensitivity (unchanged) --------------------------
    if condition.planning:
        plan = tdump.plan_sensitivity(target, occupancy, dumpability, v9.PLAN_STEPS)
        extra.update(plan)
        if plan["plan_delta"] < v9.PLAN_DELTA_MIN:
            return None, "plan_start_side_contract"
        if plan["plan_cost_far"] > v9.PLAN_FAR_COST_MAX:
            return None, "plan_far_cost_contract"

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


def _common_metadata(condition, dig, dump, occupancy, dumpability, gate):
    meta = v9._common_metadata(condition, dig, dump, occupancy, dumpability, gate)
    levels = _levels_of(condition)
    meta.update(
        {
            "schema": SCHEMA,
            "geometry_level": levels["geometry"],
            "scale_level": levels["scale"],
            "scale_band_token": _scale_token(condition),
            "anchor_is_external": condition.anchor
            in tax.RELEASES["v6-main"].external_anchors,
            "dig_bank_level": condition.dig_bank_level,
        }
    )
    return meta


def make_map(condition, dataset, dig, dig_meta, layout, rng):
    try:
        return make_sample(condition, dataset, dig, dig_meta, layout, rng)
    except RuntimeError as exc:
        return None, str(exc)


def _generate_map(
    condition, dataset, condition_index, bank, n_maps, max_attempts, map_index
):
    """Generate one deterministic map, with bounded planning-layout fallbacks."""
    rejections: Counter[str] = Counter()
    layout_rounds = PLANNING_LAYOUT_ROUNDS if condition.planning else 1
    for layout_round in range(layout_rounds):
        layout_map_index = map_index + layout_round * n_maps
        layout = layout_for(condition, layout_map_index)
        if layout_round:
            rejections["layout_reroll_after_exhaustion"] += 1
        for local_attempt in range(max_attempts):
            salt = (
                0
                if local_attempt < v9.SHARED_DIG_ATTEMPTS
                else 1
                + (local_attempt - v9.SHARED_DIG_ATTEMPTS)
                // v9.REROLL_DUMP_ATTEMPTS
            )
            dig, dig_meta = bank.get(condition.dig_bank_level, map_index, salt)
            if dig is None:
                rejections["dig_reroll_exhausted"] += 1
                continue
            attempt = layout_round * max_attempts + local_attempt
            seed = int(
                np.random.SeedSequence(
                    (
                        [SEED_BASE, condition_index, map_index, local_attempt]
                        if layout_round == 0
                        else [
                            SEED_BASE,
                            condition_index,
                            map_index,
                            layout_round,
                            local_attempt,
                        ]
                    )
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
                    "layout_reroll_round": layout_round,
                    "layout_map_index": layout_map_index,
                    "dig_sha256": sha256_mask(sample.target < 0),
                    "occupancy_sha256": sha256_mask(sample.occupancy),
                }
            )
            return map_index, sample, rejections
    return map_index, None, rejections


_WORKER_BANK = None


def _initialize_map_worker(bank):
    global _WORKER_BANK
    _WORKER_BANK = bank


def _generate_map_worker(task):
    if _WORKER_BANK is None:
        raise RuntimeError("map worker started without a dig bank")
    return _generate_map(*task[:3], _WORKER_BANK, *task[3:])


def generate_condition(
    condition,
    dataset,
    condition_index,
    bank,
    n_maps,
    max_attempts,
    executor=None,
):
    tasks = [
        (condition, dataset, condition_index, n_maps, max_attempts, map_index)
        for map_index in range(n_maps)
    ]
    if executor is None:
        results = (
            _generate_map(*task[:3], bank, *task[3:])
            for task in tasks
        )
    else:
        results = executor.map(_generate_map_worker, tasks)

    samples: list[base.Sample] = []
    rejections: Counter[str] = Counter()
    unsatisfied: list[str] = []
    accepted_digs: dict[str, int] = {}
    for map_index, accepted, map_rejections in results:
        rejections.update(map_rejections)
        if accepted is None:
            total_attempts = max_attempts * (
                PLANNING_LAYOUT_ROUNDS if condition.planning else 1
            )
            unsatisfied.append(
                f"{condition.id} map {map_index}: no accepted sample in "
                f"{total_attempts} attempts; rejections={dict(map_rejections)}"
            )
            continue
        dig_identity = sha256_mask(accepted.target < 0)
        if dig_identity in accepted_digs:
            rejections["condition_dig_exact_duplicate"] += 1
            unsatisfied.append(
                f"{condition.id} map {map_index}: exact dig duplicate of map "
                f"{accepted_digs[dig_identity]}"
            )
            continue
        accepted_digs[dig_identity] = map_index
        samples.append(accepted)
        completed = map_index + 1
        if executor is not None and (completed % 40 == 0 or completed == n_maps):
            print(
                f"  {condition.id}: evaluated {completed}/{n_maps} map slots",
                flush=True,
            )
    return samples, rejections, unsatisfied


# --------------------------------------------------------------------------
# output


def sample_index_of(condition_index: int, map_index: int) -> int:
    """Stable collision-free index for banks with up to 999 maps/condition."""
    if condition_index < 0 or not 0 <= map_index < 1000:
        raise ValueError(
            f"invalid condition/map index: {condition_index}/{map_index}"
        )
    return 1000 * condition_index + map_index


def scenario_sha256(sample: base.Sample) -> str:
    return reset_array_scenario_sha256(
        {
            name: getattr(sample, attribute)
            for name, attribute in ARRAY_FOLDERS.items()
        }
    )


def source_group_id(sample: base.Sample) -> str:
    """Identity shared by every transform or counterfactual of one source."""
    source_identity = sample.metadata.get("foundation_source_sha256")
    if source_identity:
        return f"foundation-source:{source_identity}"
    return f"dig:{sample.metadata['dig_sha256']}"


def assert_unique_scenario_rows(rows: list[dict[str, Any]]) -> None:
    """Fail if two manifest rows describe the same reset-consumed arrays."""
    first_by_identity: dict[str, str] = {}
    for row in rows:
        identity = row["scenario_sha256"]
        map_id = row["map_id"]
        previous = first_by_identity.get(identity)
        if previous is not None:
            raise RuntimeError(
                f"duplicate full scenario: {previous} and {map_id}: {identity}"
            )
        first_by_identity[identity] = map_id


def review_map_indices(map_count: int, example_count: int) -> frozenset[int]:
    """Choose a deterministic, evenly spaced provisional review subset."""
    if map_count <= 0 or example_count <= 0:
        return frozenset()
    count = min(map_count, example_count)
    return frozenset(
        int(index)
        for index in np.linspace(0, map_count - 1, num=count, dtype=np.int32)
    )


def write_condition(
    output,
    dataset,
    condition,
    condition_index,
    samples,
    review_examples,
):
    data = output / "dataset"
    folder = output / condition.id
    (folder / "previews").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen_scenarios: set[str] = set()
    review_indices = review_map_indices(len(samples), review_examples)
    for sample in samples:
        map_index = int(sample.metadata["map_index"])
        sample_index = sample_index_of(condition_index, map_index)
        map_id = f"{dataset.map_id_prefix}-{sample_index:04d}"
        scenario_identity = scenario_sha256(sample)
        if scenario_identity in seen_scenarios:
            raise RuntimeError(
                f"{condition.id}: duplicate full scenario at map {map_index}: "
                f"{scenario_identity}"
            )
        seen_scenarios.add(scenario_identity)
        for folder_name, attribute in ARRAY_FOLDERS.items():
            np.save(
                data / folder_name / f"img_{sample_index}.npy",
                getattr(sample, attribute),
            )
        owners = v9.trench_axis_owners(sample.target, sample.metadata)
        np.save(
            data / v9.TRENCH_AXIS_OWNERS_FOLDER / f"img_{sample_index}.npy",
            owners,
        )
        record = {
            "sample_index": sample_index,
            "map_id": map_id,
            "pair_slot_id": f"{condition.dig_bank_level}:{map_index}",
            "source_group_id": source_group_id(sample),
            "scenario_sha256": scenario_identity,
            "trench_axis_owners_sha256": v9.sha256_mask(owners),
            **sample.metadata,
            **{f"gate_{k}": v for k, v in asdict(sample.gate).items()},
        }
        rows.append(record)
        (output / "review_metadata" / f"img_{sample_index}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
        )
        if map_index in review_indices:
            base.render_sample(
                sample,
                folder / "previews" / f"{map_index:02d}__{map_id}.png",
                f"{condition.id} #{map_index:02d} ({map_id})",
            )
    review_samples = [
        sample
        for sample in samples
        if int(sample.metadata["map_index"]) in review_indices
    ]
    if review_samples:
        v6.render_condition_overview(
            folder / "overview.png",
            condition,
            review_samples,
        )
    levels = _levels_of(condition)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "conditionId": condition.id,
                "cellId": condition.id,
                "dataset": dataset.name,
                "tier": tax.RELEASES[dataset.release].condition_table[condition.id][0],
                "tierLabel": tax.TIER_LABELS[
                    tax.RELEASES[dataset.release].condition_table[condition.id][0]
                ],
                "preview": condition.preview,
                "anchorConditionId": condition.anchor,
                "anchorIsExternal": condition.anchor
                in tax.RELEASES[dataset.release].external_anchors,
                # spec v6 section 2 D2 names the standard variant as the anchor;
                # the taxonomy anchor must be strictly easier, so the shrink
                # relation is recorded here instead (see GENERATION_NOTES).
                "standardVariantConditionId": tax.MINI_STANDARD_VARIANT.get(
                    condition.id
                ),
                "family": condition.family,
                "factorLevels": levels,
                "factors": {
                    "geometryClass": condition.geometry,
                    "dumpLayout": condition.dump_layout,
                    "siteClass": SITE_CLASS_TOKENS[condition.site_level],
                    "capacityBand": CAPACITY_TOKENS[condition.capacity_level],
                    "distanceBand": DISTANCE_TOKENS[condition.distance_level],
                    "scaleBand": _scale_token(condition),
                },
                "objectBand": list(v9.OBJECT_BANDS[condition.site_level]),
                "ringGapped": condition.id in RING_GAP_CONDITIONS,
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
                            v9.TRENCH_AXIS_OWNERS_FOLDER:
                                f"dataset/{v9.TRENCH_AXIS_OWNERS_FOLDER}/"
                                f"img_{row['sample_index']}.npy"
                        },
                        "objectCount": row["object_count"],
                        "digCells": row["dig_cells"],
                        "dumpCells": row["dump_cells"],
                        "capacityRatio": row["dump_to_dig_area_ratio"],
                        "sharedDig": bool(row["shared_dig"]),
                        "digSha256": row["dig_sha256"],
                        "pairSlotId": row["pair_slot_id"],
                        "sourceGroupId": row["source_group_id"],
                        "scenarioSha256": row["scenario_sha256"],
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


def write_conditions_csv(output: Path, dataset, counts: dict[str, int]) -> None:
    spec = tax.RELEASES[dataset.release]
    with (output / "conditions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tax.csv_columns(spec))
        writer.writeheader()
        for condition in dataset.conditions:
            levels = _levels_of(condition)
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
                    "scale": levels["scale"],
                    "tier": spec.condition_table[condition.id][0],
                    "preview": "true" if condition.preview else "false",
                    "anchor_condition_id": condition.anchor or "",
                    "n_maps": counts.get(condition.id, 0),
                }
            )


def write_readme(output: Path, dataset, counts: dict[str, int]) -> None:
    lines = [
        f"# Terra curriculum axis-contract v2 bank — `{dataset.name}`",
        "",
        f"Current taxonomy and bank contract: `{CURRENT_TAXONOMY_PATH}`.",
        "",
        "Construction lineage: v5-main plus the two v6 deltas:",
        "",
        "- **D1** — the seven ring conditions carry 1–3 forbidden sectors of",
        "  15–40° each in the capped 3–4× band. The sectors are **non-designated",
        "  ground**, not obstacles: `occupancy` and `dumpability` are untouched and",
        "  only `images` (the target map) has notches.",
        "- **D2** — three mini-junction conditions (`trn-tee-side2-s`,",
        "  `trn-net3-side2-s`, `trn-net4-side2-s`): short spine (12–18 tiles),",
        "  dig volume ≤ 60% of the standard variant's median, generous both-side",
        "  adjacent banks, own dig-bank entries.",
        "",
        "",
        "Historical in-place revisions since v6.0:",
        "",
        "- **v6.1** — the ADJACENT proximity class is tightened to per map",
        "  `max <= 11.375 AND p95 <= 9.0` tiles, and the five capacity-apron",
        "  conditions are regenerated onto it.",
        "- **v6.2** — the trench debt of `TRENCH_WIDTH_AUDIT.md`: the reserved",
        "  lane is width-aware (`lane_far_reach_tiles <= 10.375`), `seg2` may not",
        "  fold below 60° at `half_width` 2, and the adjacent gate is enforced at",
        "  generation time on every adjacent-class condition. All 14 trench",
        "  conditions regenerate; `ADJACENT_PROXIMITY_DEFERRED` is empty.",
        "- **axis-contract v2** — trench headings use a 15° lattice and every",
        "  target cell has exact generated owner bits in `trench_axis_owners`.",
        "- Planning slots deterministically re-draw the layout at most four times",
        "  only after the preceding layout exhausts every candidate; all admission",
        "  gates stay unchanged.",
        "",
        "The 6 conditions listed as carried in `generation_summary.json` are",
        "byte-for-byte from the v5-main bank; the per-array identity is asserted",
        "in `BYTE_IDENTITY.json`.",
        "",
        f"- seed base: `{SEED_BASE}` (unchanged from v5 — byte-identity depends on it)",
        f"- taxonomy release: `{dataset.release}`",
        f"- conditions: {len(dataset.conditions)}",
        f"- maps: {sum(counts.values())} ({dataset.maps_per_condition} per condition)",
        f"- map ids: `{dataset.map_id_prefix}-NNNN`",
        f"- tile size: {TILE_SIZE_M:.10f} m",
        f"- service annulus: [{R_MIN_TILES:.3f}, {R_MAX_TILES:.3f}] tiles",
        "",
        "## Layout",
        "",
        "- `<condition-id>/manifest.json` — factor levels, tier, per-map metrics",
        "- `<condition-id>/previews/*.png` — one labelled composite per map",
        "- `<condition-id>/overview.png` — the whole condition on one sheet",
        "- `dataset/{images,occupancy,dumpability,actions,distance,"
        "trench_axis_owners}/img_N.npy`",
        "- `manifest.csv` — every map, every measured factor",
        "- `conditions.csv` — spec §4 columns plus `scale`",
        "- `GENERATION_NOTES.md` — per-delta results and honest deviations",
        "",
        "The static gate is not an action-level completion witness.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the axis-contract v2 bank without a hard IoU gate."
    )
    parser.add_argument(
        "--source-foundations",
        type=Path,
        required=True,
        help="source pool containing images/img_*.npy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maps",
        type=int,
        required=True,
        help="number of maps to generate per selected condition",
    )
    parser.add_argument("--max-attempts", type=int, default=v9.MAX_ATTEMPTS)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="spawned CPU workers for independent map slots",
    )
    parser.add_argument("--only", default="")
    parser.add_argument(
        "--review-examples",
        type=int,
        default=16,
        help="evenly spaced provisional previews per condition; 0 disables rendering",
    )
    return parser.parse_args()


def source_pool_sha256(source: Path) -> tuple[str, int]:
    images = sorted((source / "images").glob("img_*.npy"))
    if not images:
        raise ValueError(f"no source images found under {source / 'images'}")
    digest = hashlib.sha256()
    for image in images:
        digest.update(image.name.encode())
        with image.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(images)


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _cropped_mask(mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return np.zeros((0, 0), dtype=np.bool_)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0) + 1
    return np.ascontiguousarray(mask[low[0] : high[0], low[1] : high[1]])


def _mask_identity(mask: np.ndarray) -> str:
    mask = np.ascontiguousarray(mask, dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    digest.update(mask.tobytes())
    return digest.hexdigest()


def _translation_normalized_identity(mask: np.ndarray) -> str:
    return _mask_identity(_cropped_mask(mask))


def _dihedral_normalized_identity(mask: np.ndarray) -> str:
    cropped = _cropped_mask(mask)
    variants = []
    for rotation in range(4):
        rotated = np.rot90(cropped, rotation)
        variants.append(_mask_identity(rotated))
        variants.append(_mask_identity(np.fliplr(rotated)))
    return min(variants)


def _digital_perimeter(mask: np.ndarray) -> int:
    """Four-neighbour perimeter in pixel-edge units."""
    mask = np.asarray(mask, dtype=np.bool_)
    return int(
        np.count_nonzero(mask[:, 1:] != mask[:, :-1])
        + np.count_nonzero(mask[1:, :] != mask[:-1, :])
        + np.count_nonzero(mask[:, 0])
        + np.count_nonzero(mask[:, -1])
        + np.count_nonzero(mask[0, :])
        + np.count_nonzero(mask[-1, :])
    )


def write_diversity_report(output: Path, rows: list[dict[str, Any]]) -> None:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition_id"]].append(row)

    conditions: dict[str, Any] = {}
    for condition_id, condition_rows in sorted(by_condition.items()):
        digs = []
        perimeters = []
        compactness = []
        aspects = []
        placed_identities = []
        translation_identities = []
        dihedral_identities = []
        for row in condition_rows:
            target = np.load(
                output / "dataset" / "images" / f"img_{row['sample_index']}.npy"
            )
            dig = np.asarray(target) < 0
            digs.append(dig)
            perimeter = max(1, _digital_perimeter(dig))
            area = int(dig.sum())
            perimeters.append(float(perimeter))
            compactness.append(float(4.0 * math.pi * area / (perimeter**2)))
            placed_identities.append(_mask_identity(dig))
            translation_identities.append(_translation_normalized_identity(dig))
            dihedral_identities.append(_dihedral_normalized_identity(dig))
            coordinates = np.argwhere(dig)
            height, width = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
            aspects.append(float(max(height, width) / max(1, min(height, width))))

        nearest = []
        for index, dig in enumerate(digs):
            similarities = [
                centred_iou(dig, other)
                for other_index, other in enumerate(digs)
                if other_index != index
            ]
            nearest.append(max(similarities, default=0.0))

        scenario_ids = [row["scenario_sha256"] for row in condition_rows]
        conditions[condition_id] = {
            "scenarios": len(condition_rows),
            "source_groups": len(
                {row["source_group_id"] for row in condition_rows}
            ),
            "pair_slots": len({row["pair_slot_id"] for row in condition_rows}),
            "unique_placed_dig_rasters": len(set(placed_identities)),
            "unique_translation_normalized_digs": len(
                set(translation_identities)
            ),
            "unique_dihedral_normalized_digs": len(set(dihedral_identities)),
            "exact_scenario_duplicates": len(scenario_ids)
            - len(set(scenario_ids)),
            "nearest_neighbor_centered_iou": _distribution(nearest),
            "dig_cells": _distribution(
                [float(row["dig_cells"]) for row in condition_rows]
            ),
            "dig_perimeter_cells": _distribution(perimeters),
            "dig_compactness": _distribution(compactness),
            "dig_bbox_aspect": _distribution(aspects),
            "dig_components": _distribution(
                [float(row["dig_components_actual"]) for row in condition_rows]
            ),
            "dump_to_dig_area_ratio": _distribution(
                [
                    float(row["dump_to_dig_area_ratio"])
                    for row in condition_rows
                ]
            ),
            "dig_dump_distance_p50_tiles": _distribution(
                [
                    float(row["dig_dump_distance_median_tiles"])
                    for row in condition_rows
                ]
            ),
            "object_count": _distribution(
                [float(row["object_count"]) for row in condition_rows]
            ),
        }

    payload = {
        "schema": "terra_map_bank_diversity_v1",
        "novelty_rules": {
            "dig_bank": "reject_exact_dig_duplicates",
            "full_bank": "reject_exact_full_scenario_duplicates",
            "centered_iou": "diagnostic_only",
        },
        "physical_and_source_gates": (
            "reviewed-v6 capacity, proximity, lane, obstacle, workspace, and "
            "within-level source-reuse gates retained on the axis-v2 bank"
        ),
        "centered_iou_role": "diagnostic_only",
        "conditions": conditions,
    }
    (output / "diversity_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.maps <= 0 or args.maps >= 1000:
        raise SystemExit("--maps must be in [1, 999]")
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.workers > 1:
        for variable in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"
    if args.review_examples < 0:
        raise SystemExit("--review-examples must be nonnegative")
    source = args.source_foundations.resolve()
    try:
        source_sha256, source_count = source_pool_sha256(source)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    dataset = dataclasses.replace(
        DATASETS["main"],
        maps_per_condition=args.maps,
        map_id_prefix=f"curriculum-axis-v2-{args.maps}",
    )
    assert_conditions_match_taxonomy(dataset)
    selected = set(filter(None, args.only.split(",")))
    known = {condition.id for condition in dataset.conditions}
    unknown = selected - known
    if unknown:
        raise SystemExit(f"unknown --only conditions: {sorted(unknown)}")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"--output must be empty: {output}")
    for folder in (
        output,
        output / "review_metadata",
        *(output / "dataset" / name for name in ARRAY_FOLDERS),
        output / "dataset" / v9.TRENCH_AXIS_OWNERS_FOLDER,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    factory = GeometryFactoryV10(source)

    bank_size = args.maps
    spec = tax.RELEASES[dataset.release]
    t0_levels = frozenset(
        c.dig_bank_level
        for c in dataset.conditions
        if spec.condition_table[c.id][0] == 0
    )
    bank = DigBankV10(factory, bank_size, t0_levels)
    wanted = selected or {c.id for c in dataset.conditions}
    levels = sorted({c.dig_bank_level for c in dataset.conditions if c.id in wanted})
    bank_rejections = bank.build(levels)
    print(f"dig bank built: {bank_size} per level, rejections={dict(bank_rejections)}")

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    rejection_totals: Counter[str] = Counter()
    unsatisfied: list[str] = []

    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_map_worker,
            initargs=(bank,),
        )
    try:
        for condition_index, condition in enumerate(dataset.conditions):
            if selected and condition.id not in selected:
                continue
            n_maps = args.maps
            samples, rejections, failures = generate_condition(
                condition,
                dataset,
                condition_index,
                bank,
                n_maps,
                args.max_attempts,
                executor,
            )
            rejection_totals.update(rejections)
            unsatisfied.extend(failures)
            condition_rows = write_condition(
                output,
                dataset,
                condition,
                condition_index,
                samples,
                args.review_examples,
            )
            rows.extend(condition_rows)
            counts[condition.id] = len(condition_rows)
            rerolled = sum(1 for row in condition_rows if not row["shared_dig"])
            print(
                f"[{condition_index + 1:02d}/{len(dataset.conditions)}] "
                f"{condition.id}: {len(condition_rows)}/{n_maps} maps"
                + (f" rerolled={rerolled}" if rerolled else "")
                + (f" UNSATISFIED={len(failures)}" if failures else "")
                + (
                    f" rejections={dict(rejections.most_common(4))}"
                    if rejections
                    else ""
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    v9.write_terra_metadata(output, rows)

    built_this_run = sorted(counts)

    rows.sort(key=lambda row: row["sample_index"])
    assert_unique_scenario_rows(rows)
    v9.write_manifest(output, rows)
    write_conditions_csv(output, dataset, counts)
    write_readme(output, dataset, counts)
    write_diversity_report(output, rows)

    realized = defaultdict(list)
    realized_capacity = defaultdict(list)
    turn_dump = defaultdict(list)
    ring_gaps = defaultdict(list)
    mini_overlap = defaultdict(list)
    for row in rows:
        realized[row["condition_id"]].append(float(row["reachable_dump_to_dig_ratio"]))
        realized_capacity[row["condition_id"]].append(
            float(row["dump_to_dig_area_ratio"])
        )
        if row.get("turn_dump_cov_strict") not in (None, ""):
            turn_dump[row["condition_id"]].append(float(row["turn_dump_cov_strict"]))
        if row.get("ring_gap_total_deg") not in (None, ""):
            ring_gaps[row["condition_id"]].append(float(row["ring_gap_total_deg"]))
        if row["condition_id"] in MINI_CONDITION_IDS:
            mini_overlap[row["condition_id"]].append(float(row["arm_overlap_fraction"]))

    summary = {
        "schema": SCHEMA,
        "dataset": dataset.name,
        "seed_base": SEED_BASE,
        "spec_path": CURRENT_TAXONOMY_PATH,
        "spec_section": "3-5 (registry, counterfactuals, and bank contract)",
        "taxonomy_version": tax.TAXONOMY_VERSION,
        "taxonomy_release": dataset.release,
        "generator": "tools/map_generation/generate_curriculum_bank.py",
        "map_id_prefix": dataset.map_id_prefix,
        "conditions_built_this_run": built_this_run,
        "regenerated_conditions": sorted(REGENERATED_CONDITIONS),
        "carried_conditions": sorted(CARRIED_CONDITIONS),
        "proximity_classes": {
            c.id: adjacent_proximity_class(c) for c in dataset.conditions
        },
        "adjacent_proximity_gate": {
            "p95_max": ADJACENT_PROXIMITY_P95_MAX,
            "max_max": ADJACENT_PROXIMITY_MAX_MAX,
            # v6.2: enforced by CLASS, not by dump style -- every adjacent-class
            # condition is gated where it is drawn.
            "enforced_conditions": sorted(
                c.id
                for c in dataset.conditions
                if adjacent_proximity_class(c) == "adjacent"
                and c.id not in ADJACENT_PROXIMITY_DEFERRED
            ),
            "bounded_by_planning": PROXIMITY_BOUNDED_BY_PLANNING,
            "deferred_conditions": sorted(ADJACENT_PROXIMITY_DEFERRED),
            "apron_standoff_clamp": APRON_PROXIMITY_STANDOFF_MAX,
        },
        "v6_2_trench_debt": {
            "lane_far_reach_max_tiles": LANE_FAR_REACH_MAX_TILES,
            "lane_far_reach_margin_tiles": LANE_FAR_REACH_MARGIN_TILES,
            "seg2_min_relative_turn_deg": SEG2_MIN_RELATIVE_TURN_DEG,
            "seg2_applies_at_half_width": SEG2_WIDE_HALF_WIDTH,
            "regenerated_trench_conditions": sorted(TRENCH_CONDITIONS),
        },
        "source_foundations": str(source),
        "source_foundations_images": source_count,
        "source_foundations_sha256": source_sha256,
        "workers": args.workers,
        "worker_library_threads": 1 if args.workers > 1 else None,
        "max_attempts_per_layout": args.max_attempts,
        "planning_layout_rounds": PLANNING_LAYOUT_ROUNDS,
        "novelty_rules": {
            "dig_bank": "reject_exact_dig_duplicates",
            "full_bank": "reject_exact_full_scenario_duplicates",
            "centered_iou": "diagnostic_only",
        },
        "centred_iou_role": "diagnostic_only",
        "condition_count": len(dataset.conditions),
        "maps_per_condition": counts,
        "accepted_maps": len(rows),
        "rerolled_dig_maps": sum(1 for row in rows if int(row["shared_dig"]) == 0),
        "tile_size_m": TILE_SIZE_M,
        "d1_ring_gaps": {
            "conditions": sorted(RING_GAP_CONDITIONS),
            "count_choices": list(RING_GAP_COUNT_CHOICES),
            "width_deg": list(RING_GAP_WIDTH_DEG),
            "total_deg": list(RING_GAP_TOTAL_DEG),
            "edge_separation_deg": RING_GAP_EDGE_SEPARATION_DEG,
            "min_notch_cells": RING_GAP_MIN_NOTCH_CELLS,
            "min_notch_cells_per_gap": RING_GAP_MIN_NOTCH_CELLS_PER_GAP,
            "realized_total_deg": {
                key: [round(min(v), 2), round(float(np.median(v)), 2), round(max(v), 2)]
                for key, v in sorted(ring_gaps.items())
            },
        },
        "d2_minis": {
            "conditions": sorted(MINI_CONDITION_IDS),
            "spine_spec_band_tiles": list(MINI_SPINE_SPEC_BAND),
            "spine_band_tiles": {k: list(v) for k, v in MINI_SPINE_BAND.items()},
            "branch_ratio": {k: list(v) for k, v in MINI_BRANCH_RATIO.items()},
            "headings_deg": {k: list(v) for k, v in MINI_HEADINGS.items()},
            "volume_fraction_max": MINI_VOLUME_FRACTION_MAX,
            "dig_cells_cap": MINI_DIG_CELLS_MAX,
            "standard_dig_cells_median": STANDARD_DIG_CELLS_MEDIAN,
            "overlap_spec_band": list(MINI_OVERLAP_SPEC_BAND),
            "overlap_gate_applied": {k: list(v) for k, v in MINI_OVERLAP_GATE.items()},
            "overlap_spec_band_satisfiable": MINI_OVERLAP_SPEC_SATISFIABLE,
            "realized_overlap": {
                key: [round(min(v), 4), round(float(np.median(v)), 4), round(max(v), 4)]
                for key, v in sorted(mini_overlap.items())
            },
        },
        "realized_reachable_capacity": {
            key: [round(min(v), 3), round(max(v), 3)]
            for key, v in sorted(realized.items())
        },
        "realized_designated_capacity": {
            key: [round(min(v), 3), round(max(v), 3)]
            for key, v in sorted(realized_capacity.items())
        },
        "realized_turn_dump_cov_strict": {
            key: [round(min(v), 4), round(float(np.median(v)), 4), round(max(v), 4)]
            for key, v in sorted(turn_dump.items())
        },
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
        raise SystemExit(
            "unsatisfied constraints:\n" + "\n".join(f"  {line}" for line in unsatisfied)
        )


if __name__ == "__main__":
    main()
