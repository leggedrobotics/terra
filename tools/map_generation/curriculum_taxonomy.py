#!/usr/bin/env python3
"""Factor-based curriculum taxonomy for the map curriculum candidate release.

Spec: docs/CURRICULUM_TAXONOMY_SPEC.md (v1, 2026-07-28).

Difficulty tier is computed from factor levels, never hand-assigned. Levels are
derived from the manifest fields already present on every scenario; the explicit
23-row cell table from spec section 2 is the source of truth for anchors and is
asserted against the derived condition ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SPEC_PATH = "docs/CURRICULUM_TAXONOMY_SPEC.md"
TAXONOMY_VERSION = 1

# Releases are scoped. The v3/v3.1 tables below are FROZEN: the published
# `map-curriculum-v3_1-20260728` release derives its condition ids, tiers and
# anchors from them and its review decisions are keyed by scenario id. The v4
# tables (spec §8.3/§8.4) are additive and live in their own ReleaseSpec; every
# public function takes an optional `release=` and defaults to v3.
DEFAULT_RELEASE = "v3"

FAMILY_LEVELS = {"foundation": "fnd", "trench": "trn"}

# Factor order is fixed: it drives the condition id, the csv, and the delta list.
# `scale` is a v6 addition (spec v6 §2 D2, the mini-junction rungs); releases that
# declare no scale vocabulary never emit it and are byte-unchanged.
FACTORS = ("family", "geometry", "dump", "capacity", "site", "distance", "scale")
# Distance is implied by the dump level (remote => far), so it is not an
# independent axis: it neither counts toward the tier nor shows up as a delta.
TIER_FACTORS = ("geometry", "dump", "capacity", "site")

GEOMETRY_LEVELS = {
    "foundation_osm": "slab",
    # v3 banks generate the large slab class natively instead of importing the
    # frozen large-foundation bank, so it gets its own manifest token.
    "foundation_osm_large": "slab-lg",
    "foundation_procedural": "proc",
    "foundation_structural": "strips",
    "trench_axes_1": "straight",
    "trench_axes_2": "tee",
    "trench_axes_3": "net",
}
# The large-foundation cell is procedural but reviewed as its own slab size class.
GEOMETRY_CELL_OVERRIDES = {"a_foundation_large_allaround": "slab-lg"}

DUMP_LEVELS = {
    "foundation": {
        "all_around": "ring",
        "easy_surround": "ring",
        "near_apron": "apron",
        "near_apron_large": "apron",
        "one_side_near": "side1",
        "separated_zones": "split",
        "haul_away_edge": "remote",
    },
    # A trench has two natural dump flanks: surround means both sides, and any
    # apron/one-side layout is a single flank.
    "trench": {
        "all_around": "side2",
        "easy_surround": "side2",
        "near_apron": "side1",
        "near_apron_large": "side1",
        "one_side_near": "side1",
        "separated_zones": "split",
        "haul_away_edge": "remote",
    },
}

CAPACITY_LEVELS = {
    "slcap07_10": "c7x",
    "slcap03_04": "c3x",
    "all_around_11_08_11_49x": "c11x",
    # v3 banks state these two in the manifest; v2 could only express them as a
    # property of the cell (see CAPACITY_CELL_OVERRIDES).
    "legacy_apron_1p6": "c1p6",
    "trench_constrained": "tight",
}
# Two cells carry their capacity in the cell definition rather than in a band
# field: the legacy ~1.6x apron and the constrained one-side trench.
CAPACITY_CELL_OVERRIDES = {
    "b_foundation_legacy_near_apron_light": "c1p6",
    "b_trench_straight_one_constrained": "tight",
}

SITE_LEVELS = {
    "none": "clean",
    # v2's `light` style placed 0-2 small objects, so `light` is only clean by
    # label. v3 banks emit `none` for genuinely empty sites and put the light
    # object band in its own `obj1` level.
    "light": "clean",
    "scattered_objects": "obj",
    "scattered_objects_light": "obj1",
    "access_road": "road",
    "gapped_wall": "wall",
}

EASY_GEOMETRY = {"fnd": {"slab", "slab-lg", "proc"}, "trn": {"straight"}}
EASY_DUMP = {"fnd": {"ring", "apron"}, "trn": {"side2", "side1"}}
EASY_CAPACITY = {"generous", "c7x", "c11x"}
EASY_SITE = {"clean"}

FACTOR_LABELS = {
    "geometry": "Geometry",
    "dump": "Dump layout",
    "capacity": "Dump capacity",
    "site": "Site",
    "distance": "Haul distance",
    "scale": "Extent",
}

LEVEL_LABELS = {
    "geometry": {
        "slab": "source-bank slab",
        "slab-lg": "large slab",
        "proc": "procedural slab",
        "strips": "structural strips + pads",
        "straight": "straight",
        # v4 (spec §8.3 U4): segmented polylines are the first ordering rung.
        "seg2": "2-segment polyline",
        "seg3": "3-segment polyline",
        "tee": "T-junction",
        "net": "3-axis network",
        # v5 (spec §8.5 U9): richer topologies up to 4 arms.
        "net3": "3-arm network",
        "net4": "4-arm network",
    },
    "dump": {
        "ring": "all-around",
        # v4 (spec §8.3 U1): the ring mask is capped at 3-4x as a band.
        "ring3x": "capped ring band (3–4×)",
        "apron": "nearby apron",
        "side1": "one side",
        "side2": "both sides",
        # v4 (spec §8.3 U3): alternating banks along the trench, always adjacent.
        "altsides": "alternating banks",
        "split": "separated zones",
        "remote": "remote haul",
    },
    "capacity": {
        "generous": "generous",
        "c7x": "7–10×",
        "c11x": "11×",
        "c3x": "3–4×",
        # v4 (spec §8.3 U5): the ladder is densified below ~2x.
        "c2x": "1.9–2.1×",
        "c1p6": "~1.6×",
        "c1p2": "~1.2×",
        "tight": "tight",
    },
    "site": {
        "clean": "clean",
        "obj": "scattered objects",
        "obj1": "1–3 scattered objects",
        "road": "access road",
        "wall": "gapped wall",
    },
    "distance": {
        "near": "direct service",
        "far": "far haul",
        # v4 (spec §8.3 U6): the distance ladder, foundation only.
        "unspec": "not controlled",
        "d12": "12-tile median haul",
        "d16": "16-tile median haul",
        "d20": "20-tile median haul",
        "d24": "24-tile median haul",
    },
    # v6 (spec v6 §2 D2): the mini-junction rungs are the same topology at a
    # short extent — the junction/ordering representation without the volume.
    "scale": {
        "std": "standard extent",
        "s": "short-arm mini",
    },
}

TIER_LABELS = {
    0: "T0 · baseline",
    1: "T1 · single factor",
    2: "T2 · two factors",
    3: "T3 · three factors",
}

PREVIEW_CELLS = {"c_foundation_remote_haul_preview", "c_trench_remote_haul_preview"}
# Taxonomy-native banks name their cells by condition id.
PREVIEW_CONDITIONS = {"fnd-slab-remote", "trn-straight-remote"}

# Spec section 2, verbatim: (cellId, conditionId, tier, anchorConditionId).
SPEC_TABLE = (
    ("a_foundation_apron_capacity_07_10", "fnd-slab-apron-c7x", 0, None),
    ("a_foundation_large_allaround", "fnd-slab-lg-ring-c11x", 0, None),
    ("a_foundation_osm_allaround_light", "fnd-slab-ring", 0, None),
    ("a_foundation_procedural_allaround_light", "fnd-proc-ring", 0, None),
    ("b_foundation_apron_capacity_03_04", "fnd-slab-apron-c3x", 1, "fnd-slab-apron-c7x"),
    ("b_foundation_legacy_near_apron_light", "fnd-slab-apron-c1p6", 1, "fnd-slab-apron-c7x"),
    ("b_foundation_one_side_light", "fnd-slab-side1", 1, "fnd-slab-ring"),
    ("b_foundation_separated_light", "fnd-slab-split", 1, "fnd-slab-ring"),
    ("b_foundation_structural_allaround", "fnd-strips-ring", 1, "fnd-slab-ring"),
    ("b_foundation_allaround_objects", "fnd-slab-ring-obj", 1, "fnd-slab-ring"),
    # v3-native light-object band (spec section 8.2). It has no v2 cell, so its
    # cell id is its condition id; releases without it simply skip the row.
    ("fnd-slab-ring-obj1", "fnd-slab-ring-obj1", 1, "fnd-slab-ring"),
    ("b_foundation_allaround_road", "fnd-slab-ring-road", 1, "fnd-slab-ring"),
    ("c_foundation_one_side_road", "fnd-slab-side1-road", 2, "fnd-slab-ring"),
    ("c_foundation_structural_separated_wall", "fnd-strips-split-wall", 3, "fnd-slab-ring"),
    ("c_foundation_remote_haul_preview", "fnd-slab-remote", 1, "fnd-slab-ring"),
    ("a_trench_straight_both_light", "trn-straight-side2", 0, None),
    ("a_trench_straight_one_light", "trn-straight-side1", 0, None),
    ("b_trench_straight_one_constrained", "trn-straight-side1-tight", 1, "trn-straight-side1"),
    ("b_trench_straight_separated", "trn-straight-split", 1, "trn-straight-side2"),
    ("b_trench_two_axis_both", "trn-tee-side2", 1, "trn-straight-side2"),
    ("b_trench_three_axis_both", "trn-net-side2", 1, "trn-straight-side2"),
    ("c_trench_two_axis_one_side_road", "trn-tee-side1-road", 2, "trn-straight-side1"),
    ("c_trench_three_axis_separated_wall", "trn-net-split-wall", 3, "trn-straight-side2"),
    ("c_trench_remote_haul_preview", "trn-straight-remote", 1, "trn-straight-side2"),
)

# Known spec-table bug: section 2 spells this cell `fnd-slab-side1-road` with the
# `fnd-slab-ring` anchor, but its manifest geometryClass is foundation_procedural,
# which derives to `proc`. The manifest wins and the anchor moves to the matching
# procedural baseline; the tier (2 = dump + site) is the same either way.
SPEC_TABLE_CORRECTIONS = {
    "c_foundation_one_side_road": ("fnd-proc-side1-road", "fnd-proc-ring"),
}

def _resolve(row: tuple[str, str, int, str | None]) -> tuple[str, str, int, str | None]:
    cell_id, condition, tier_, anchor = row
    condition, anchor = SPEC_TABLE_CORRECTIONS.get(cell_id, (condition, anchor))
    return cell_id, condition, tier_, anchor


# The spec table with corrections applied: (cellId, conditionId, tier, anchor).
RESOLVED_TABLE = tuple(_resolve(row) for row in SPEC_TABLE)

# v3 banks are taxonomy-native: cellId == conditionId. Both spellings resolve.
TABLE_BY_CELL: dict[str, tuple[str, int, str | None]] = {}
for _cell_id, _condition_id, _tier, _anchor in RESOLVED_TABLE:
    TABLE_BY_CELL[_cell_id] = (_condition_id, _tier, _anchor)
    TABLE_BY_CELL[_condition_id] = (_condition_id, _tier, _anchor)

# conditionId -> (tier, anchorConditionId); the generator asserts against this.
CONDITION_TABLE: dict[str, tuple[int, str | None]] = {
    condition_id: (tier_, anchor)
    for _cell, condition_id, tier_, anchor in RESOLVED_TABLE
}


# --------------------------------------------------------------------------
# Release scoping (spec §8.3 / §8.4). v3 rows above are never edited.


@dataclass(frozen=True)
class ReleaseSpec:
    """Everything a release needs to derive levels, ids, tiers and deltas.

    v3 and v4 differ in three ways that all live here: the level vocabularies,
    which levels count as easy, and whether `distance` is part of the id and of
    the tier count. Nothing else in this module is release-specific.
    """

    name: str
    table: tuple[tuple[str, str, int, str | None], ...]
    geometry_levels: dict[str, str]
    dump_levels: dict[str, dict[str, str]]
    capacity_levels: dict[str, str]
    site_levels: dict[str, str]
    distance_levels: dict[str, str]
    easy_geometry: dict[str, frozenset[str]]
    easy_dump: dict[str, frozenset[str]]
    easy_capacity: frozenset[str]
    easy_site: frozenset[str]
    easy_distance: frozenset[str]
    tier_factors: tuple[str, ...]
    # Distance levels that appear in the condition id. v3 emits none, so its
    # ids are byte-identical to the published release.
    emit_distance: frozenset[str] = frozenset()
    # v6 (spec v6 §2 D2). `scale` is the mini-junction axis: the same topology at
    # a short extent. It is a MITIGATING axis — both levels are easy, so it never
    # scores toward a tier — but it is a real factor, so it shows up in the id and
    # in the delta against the anchor. A release that declares no scale vocabulary
    # (v3, v4, v5) never carries the key at all and is byte-unchanged.
    scale_levels: dict[str, str] = field(default_factory=dict)
    easy_scale: frozenset[str] = frozenset()
    emit_scale: frozenset[str] = frozenset()
    preview_conditions: frozenset[str] = frozenset()
    # Anchor condition ids that live in a DIFFERENT release. v5 splits the bank
    # into `main` and `transport` (spec §8.5 U8); a transport condition's anchor
    # is a main-track baseline, which is not present in the transport bank. The
    # anchor is still recorded and exported, it is simply not resolved (and not
    # delta-checked) inside this release. v3/v4 declare none, so nothing changes.
    external_anchors: frozenset[str] = frozenset()
    resolved: tuple[tuple[str, str, int, str | None], ...] = field(
        init=False, repr=False, default=()
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved", tuple(_resolve(row) for row in self.table))

    @property
    def condition_table(self) -> dict[str, tuple[int, str | None]]:
        return {row[1]: (row[2], row[3]) for row in self.resolved}

    @property
    def table_by_cell(self) -> dict[str, tuple[str, int, str | None]]:
        out: dict[str, tuple[str, int, str | None]] = {}
        for cell_id, condition, tier_, anchor in self.resolved:
            out[cell_id] = (condition, tier_, anchor)
            out[condition] = (condition, tier_, anchor)
        return out


V3_RELEASE = ReleaseSpec(
    name="v3",
    table=SPEC_TABLE,
    geometry_levels=GEOMETRY_LEVELS,
    dump_levels=DUMP_LEVELS,
    capacity_levels=CAPACITY_LEVELS,
    site_levels=SITE_LEVELS,
    # v3 never reads a distance token: `far` is implied by dump == remote.
    distance_levels={},
    easy_geometry={k: frozenset(v) for k, v in EASY_GEOMETRY.items()},
    easy_dump={k: frozenset(v) for k, v in EASY_DUMP.items()},
    easy_capacity=frozenset(EASY_CAPACITY),
    easy_site=frozenset(EASY_SITE),
    easy_distance=frozenset({"near", "far"}),
    tier_factors=TIER_FACTORS,
    emit_distance=frozenset(),
    preview_conditions=frozenset(PREVIEW_CONDITIONS),
)

# --------------------------------------------------------------------------
# v4 — spec §8.3 (U1-U7) and §8.4 (the 31-condition set).

GEOMETRY_LEVELS_V4 = {
    **GEOMETRY_LEVELS,
    "trench_segments_2": "seg2",
    "trench_segments_3": "seg3",
}
DUMP_LEVELS_V4 = {
    "foundation": {**DUMP_LEVELS["foundation"], "capped_ring_band": "ring3x"},
    "trench": {
        **DUMP_LEVELS["trench"],
        "capped_ring_band": "ring3x",
        "alternating_sides": "altsides",
    },
}
CAPACITY_LEVELS_V4 = {
    **CAPACITY_LEVELS,
    "slcap01_15_01_25": "c1p2",
    "slcap01_90_02_10": "c2x",
}
DISTANCE_LEVELS_V4 = {
    "": "unspec",
    "not_controlled": "unspec",
    "direct_service_near": "near",
    "haul_far": "far",
    "dist_bin_12": "d12",
    "dist_bin_16": "d16",
    "dist_bin_20": "d20",
    "dist_bin_24": "d24",
}

# §8.3: the capped 3-4x ring band is the new T0 baseline, so `c3x` stops being
# a hard rung (U1/U5) and the ladder that matters is c2x and below. `d12` sits
# inside the direct-service envelope (P0: unbroken to ~14 tiles) but the whole
# distance ladder is a controlled non-baseline axis, so every bin scores.
EASY_GEOMETRY_V4 = {
    "fnd": frozenset({"slab", "slab-lg", "proc"}),
    "trn": frozenset({"straight"}),
}
EASY_DUMP_V4 = {
    "fnd": frozenset({"ring3x", "apron"}),
    "trn": frozenset({"side2", "side1"}),
}
EASY_CAPACITY_V4 = frozenset({"generous", "c3x", "c7x", "c11x"})
EASY_SITE_V4 = frozenset({"clean"})
EASY_DISTANCE_V4 = frozenset({"unspec", "near", "far"})
TIER_FACTORS_V4 = ("geometry", "dump", "capacity", "site", "distance")
EMIT_DISTANCE_V4 = frozenset({"near", "d12", "d16", "d20", "d24"})

# Spec §8.4, verbatim: (cellId, conditionId, tier, anchorConditionId).
# v4 is taxonomy-native, so cellId == conditionId on every row.
SPEC_TABLE_V4 = (
    # T0 foundations: capped rings (U1) + the direct-service apron (U2).
    ("fnd-slab-ring3x", "fnd-slab-ring3x", 0, None),
    ("fnd-proc-ring3x", "fnd-proc-ring3x", 0, None),
    ("fnd-slab-lg-ring3x", "fnd-slab-lg-ring3x", 0, None),
    ("fnd-slab-apron-near", "fnd-slab-apron-near", 0, None),
    # U5 capacity ladder. c3x is the T0 ceiling rung (3-4x is the new baseline
    # generosity), c2x and below are the rungs that bite.
    ("fnd-slab-apron-c3x", "fnd-slab-apron-c3x", 0, None),
    ("fnd-slab-apron-c2x", "fnd-slab-apron-c2x", 1, "fnd-slab-apron-c3x"),
    ("fnd-slab-apron-c1p6", "fnd-slab-apron-c1p6", 1, "fnd-slab-apron-c3x"),
    ("fnd-slab-apron-c1p2", "fnd-slab-apron-c1p2", 1, "fnd-slab-apron-c3x"),
    # U6 distance ladder, foundation only, anchored on the direct-service apron.
    ("fnd-slab-apron-d12", "fnd-slab-apron-d12", 1, "fnd-slab-apron-near"),
    ("fnd-slab-apron-d16", "fnd-slab-apron-d16", 1, "fnd-slab-apron-near"),
    ("fnd-slab-apron-d20", "fnd-slab-apron-d20", 1, "fnd-slab-apron-near"),
    ("fnd-slab-apron-d24", "fnd-slab-apron-d24", 1, "fnd-slab-apron-near"),
    # layout / geometry / site deltas off the capped ring
    ("fnd-slab-side1", "fnd-slab-side1", 1, "fnd-slab-ring3x"),
    ("fnd-slab-split", "fnd-slab-split", 1, "fnd-slab-ring3x"),
    ("fnd-strips-ring3x", "fnd-strips-ring3x", 1, "fnd-slab-ring3x"),
    ("fnd-slab-ring3x-obj1", "fnd-slab-ring3x-obj1", 1, "fnd-slab-ring3x"),
    ("fnd-slab-ring3x-obj", "fnd-slab-ring3x-obj", 1, "fnd-slab-ring3x"),
    # U1 re-opens ring-road: a road crossing a FINITE band bites.
    ("fnd-slab-ring3x-road", "fnd-slab-ring3x-road", 1, "fnd-slab-ring3x"),
    ("fnd-proc-side1-road", "fnd-proc-side1-road", 2, "fnd-proc-ring3x"),
    ("fnd-strips-split-wall", "fnd-strips-split-wall", 3, "fnd-slab-ring3x"),
    ("fnd-slab-remote", "fnd-slab-remote", 1, "fnd-slab-ring3x"),
    # Trenches: dumping is always adjacent (U3), axes on the 30 deg lattice
    # (U7), overlap stressor on every multi-segment condition (U4).
    ("trn-straight-side2", "trn-straight-side2", 0, None),
    ("trn-straight-side1", "trn-straight-side1", 0, None),
    ("trn-straight-side1-tight", "trn-straight-side1-tight", 1, "trn-straight-side1"),
    ("trn-straight-altsides", "trn-straight-altsides", 1, "trn-straight-side2"),
    ("trn-seg2-side2", "trn-seg2-side2", 1, "trn-straight-side2"),
    ("trn-seg3-side2", "trn-seg3-side2", 1, "trn-straight-side2"),
    ("trn-tee-side2", "trn-tee-side2", 1, "trn-straight-side2"),
    ("trn-net-side2", "trn-net-side2", 1, "trn-straight-side2"),
    ("trn-tee-side1-road", "trn-tee-side1-road", 2, "trn-straight-side1"),
    ("trn-net-altsides-wall", "trn-net-altsides-wall", 3, "trn-straight-side2"),
)

V4_RELEASE = ReleaseSpec(
    name="v4",
    table=SPEC_TABLE_V4,
    geometry_levels=GEOMETRY_LEVELS_V4,
    dump_levels=DUMP_LEVELS_V4,
    capacity_levels=CAPACITY_LEVELS_V4,
    site_levels=SITE_LEVELS,
    distance_levels=DISTANCE_LEVELS_V4,
    easy_geometry=EASY_GEOMETRY_V4,
    easy_dump=EASY_DUMP_V4,
    easy_capacity=EASY_CAPACITY_V4,
    easy_site=EASY_SITE_V4,
    easy_distance=EASY_DISTANCE_V4,
    tier_factors=TIER_FACTORS_V4,
    emit_distance=EMIT_DISTANCE_V4,
    preview_conditions=frozenset({"fnd-slab-remote"}),
)

# --------------------------------------------------------------------------
# v5 — spec §8.5 (U8-U11) and §8.6 (the turn-dump geometry contract).
#
# U8 splits the bank in two. `v5-main` is everything that is turn-dumpable from
# natural digging poses; `v5-transport` is the long-range soil-transport probe
# set (walls, remote hauls, the top distance bins). They are separate releases
# with separate scenario-id namespaces, so a condition never appears in both.

GEOMETRY_LEVELS_V5 = {
    "foundation_osm": "slab",
    "foundation_osm_large": "slab-lg",
    "foundation_procedural": "proc",
    "foundation_structural": "strips",
    "trench_axes_1": "straight",
    "trench_segments_2": "seg2",
    "trench_segments_3": "seg3",
    "trench_axes_2": "tee",
    # U9: `net` splits into an explicit arm count. The bare `net` token is not a
    # v5 level, so `net3`/`net4` parse without a longest-match ambiguity.
    "trench_axes_3": "net3",
    "trench_axes_4": "net4",
}
DUMP_LEVELS_V5 = DUMP_LEVELS_V4
CAPACITY_LEVELS_V5 = CAPACITY_LEVELS_V4
DISTANCE_LEVELS_V5 = DISTANCE_LEVELS_V4

# U9: two-axis (tee) trenches join T0 — measurement says they are not harder
# than straight when the dumping is adjacent. Geometry alone never carries
# difficulty; geometry x dump interaction does.
EASY_GEOMETRY_V5 = {
    "fnd": frozenset({"slab", "slab-lg", "proc"}),
    "trn": frozenset({"straight", "tee"}),
}
EASY_DUMP_V5 = {
    "fnd": frozenset({"ring3x", "apron"}),
    "trn": frozenset({"side2", "side1"}),
}
EASY_CAPACITY_V5 = EASY_CAPACITY_V4
EASY_SITE_V5 = EASY_SITE_V4
EASY_DISTANCE_V5 = EASY_DISTANCE_V4
TIER_FACTORS_V5 = TIER_FACTORS_V4
EMIT_DISTANCE_V5 = EMIT_DISTANCE_V4

# Spec §8.5, the `main` dataset: T0 capped rings + apron, T1 the single-factor
# ladders, T2 the planning compositions (U10).
SPEC_TABLE_V5_MAIN = (
    # ---- T0 --------------------------------------------------------------
    ("fnd-slab-ring3x", "fnd-slab-ring3x", 0, None),
    ("fnd-proc-ring3x", "fnd-proc-ring3x", 0, None),
    ("fnd-slab-lg-ring3x", "fnd-slab-lg-ring3x", 0, None),
    ("fnd-slab-apron-near", "fnd-slab-apron-near", 0, None),
    ("fnd-slab-apron-c3x", "fnd-slab-apron-c3x", 0, None),
    # ---- T1 capacity ladder (shared digs) --------------------------------
    ("fnd-slab-apron-c2x", "fnd-slab-apron-c2x", 1, "fnd-slab-apron-c3x"),
    ("fnd-slab-apron-c1p6", "fnd-slab-apron-c1p6", 1, "fnd-slab-apron-c3x"),
    ("fnd-slab-apron-c1p2", "fnd-slab-apron-c1p2", 1, "fnd-slab-apron-c3x"),
    # ---- T1 distance ladder: only the bins that stay turn-dumpable (§8.6) --
    ("fnd-slab-apron-d12", "fnd-slab-apron-d12", 1, "fnd-slab-apron-near"),
    ("fnd-slab-apron-d16", "fnd-slab-apron-d16", 1, "fnd-slab-apron-near"),
    # ---- T1 layout / geometry / site off the capped ring -------------------
    ("fnd-slab-side1", "fnd-slab-side1", 1, "fnd-slab-ring3x"),
    ("fnd-slab-split", "fnd-slab-split", 1, "fnd-slab-ring3x"),
    ("fnd-strips-ring3x", "fnd-strips-ring3x", 1, "fnd-slab-ring3x"),
    ("fnd-slab-ring3x-obj1", "fnd-slab-ring3x-obj1", 1, "fnd-slab-ring3x"),
    ("fnd-slab-ring3x-obj", "fnd-slab-ring3x-obj", 1, "fnd-slab-ring3x"),
    ("fnd-slab-ring3x-road", "fnd-slab-ring3x-road", 1, "fnd-slab-ring3x"),
    # ---- T2 planning compositions (U10) -----------------------------------
    ("fnd-proc-side1-road", "fnd-proc-side1-road", 2, "fnd-proc-ring3x"),
    ("fnd-slab-side1-obj", "fnd-slab-side1-obj", 2, "fnd-slab-ring3x"),
    # ---- trench T0 (U9: tee joins the baseline) ---------------------------
    ("trn-straight-side2", "trn-straight-side2", 0, None),
    ("trn-straight-side1", "trn-straight-side1", 0, None),
    ("trn-tee-side2", "trn-tee-side2", 0, None),
    # ---- trench T1 --------------------------------------------------------
    ("trn-straight-side1-tight", "trn-straight-side1-tight", 1, "trn-straight-side1"),
    ("trn-straight-altsides", "trn-straight-altsides", 1, "trn-straight-side2"),
    ("trn-seg2-side2", "trn-seg2-side2", 1, "trn-straight-side2"),
    ("trn-seg3-side2", "trn-seg3-side2", 1, "trn-straight-side2"),
    ("trn-net3-side2", "trn-net3-side2", 1, "trn-straight-side2"),
    ("trn-net4-side2", "trn-net4-side2", 1, "trn-straight-side2"),
    # ---- trench T2 planning compositions (U10) ----------------------------
    ("trn-net3-side1-road", "trn-net3-side1-road", 2, "trn-straight-side1"),
    ("trn-net4-side1-road", "trn-net4-side1-road", 2, "trn-straight-side1"),
)

# Spec §8.5 U8, the `transport` dataset: long-range soil transport. Every anchor
# is a main-track baseline, i.e. external to this release.
SPEC_TABLE_V5_TRANSPORT = (
    ("fnd-slab-apron-d20", "fnd-slab-apron-d20", 1, "fnd-slab-apron-near"),
    ("fnd-slab-apron-d24", "fnd-slab-apron-d24", 1, "fnd-slab-apron-near"),
    ("fnd-slab-remote", "fnd-slab-remote", 1, "fnd-slab-ring3x"),
    ("fnd-strips-split-wall", "fnd-strips-split-wall", 3, "fnd-slab-ring3x"),
    # U8 revives the trench remote haul: it is a transport test, not a trench
    # difficulty axis (U3 keeps main-track trench dumping adjacent).
    ("trn-straight-remote", "trn-straight-remote", 1, "trn-straight-side2"),
    # U3 (dumping alongside the trench) is a MAIN-curriculum property. On the
    # transport track the trench wall condition may separate the spoil from the
    # excavation, which is what makes a >= 1.15 haul detour constructible at all
    # — v4 measured five arrangements of an adjacent-bank trench wall and none
    # produced a single accepted map (§8.4 BLOCKED_CONDITIONS).
    ("trn-net3-split-wall", "trn-net3-split-wall", 3, "trn-straight-side2"),
)

V5_MAIN_RELEASE = ReleaseSpec(
    name="v5-main",
    table=SPEC_TABLE_V5_MAIN,
    geometry_levels=GEOMETRY_LEVELS_V5,
    dump_levels=DUMP_LEVELS_V5,
    capacity_levels=CAPACITY_LEVELS_V5,
    site_levels=SITE_LEVELS,
    distance_levels=DISTANCE_LEVELS_V5,
    easy_geometry=EASY_GEOMETRY_V5,
    easy_dump=EASY_DUMP_V5,
    easy_capacity=EASY_CAPACITY_V5,
    easy_site=EASY_SITE_V5,
    easy_distance=EASY_DISTANCE_V5,
    tier_factors=TIER_FACTORS_V5,
    emit_distance=EMIT_DISTANCE_V5,
    preview_conditions=frozenset(),
)

V5_TRANSPORT_RELEASE = ReleaseSpec(
    name="v5-transport",
    table=SPEC_TABLE_V5_TRANSPORT,
    geometry_levels=GEOMETRY_LEVELS_V5,
    dump_levels=DUMP_LEVELS_V5,
    capacity_levels=CAPACITY_LEVELS_V5,
    site_levels=SITE_LEVELS,
    distance_levels=DISTANCE_LEVELS_V5,
    easy_geometry=EASY_GEOMETRY_V5,
    easy_dump=EASY_DUMP_V5,
    easy_capacity=EASY_CAPACITY_V5,
    easy_site=EASY_SITE_V5,
    easy_distance=EASY_DISTANCE_V5,
    tier_factors=TIER_FACTORS_V5,
    emit_distance=EMIT_DISTANCE_V5,
    preview_conditions=frozenset(),
    external_anchors=frozenset(
        {"fnd-slab-apron-near", "fnd-slab-ring3x", "trn-straight-side2"}
    ),
)

# --------------------------------------------------------------------------
# v6 — CURRICULUM_SPEC_V6.md §2 (constant support, annealed dose).
#
# v6-main = v5-main + two deltas. D1 (gapped ring masks) changes the MAPS of the
# seven ring conditions, not the taxonomy — the gap sectors are non-designated
# ground, so `dump` stays `ring3x` and every level is unchanged. D2 adds three
# mini-junction conditions on a new `scale` axis.
#
# The `scale` axis is mitigating: `s` (short-arm mini) is an EASY level, so it
# never scores. That is deliberate — §2 D2's measured basis is that junction
# geometry is volume-not-skill, so a small multi-axis trench is legitimately easy.
#
# Anchor deviation (recorded in GENERATION_NOTES): spec v6 §2 says the minis
# anchor on "their standard variant". The taxonomy invariant is that an anchor is
# strictly EASIER than the condition it anchors (`build_conditions` asserts
# `anchor.tier < condition.tier` and `len(delta) == tier difference`), and a mini
# is easier than its standard variant, not harder — so `trn-net3-side2-s` cannot
# anchor on `trn-net3-side2`. The minis therefore form their own ladder off
# `trn-tee-side2-s` (the T0 mini), and the standard variant is recorded
# separately, per condition, as `standardVariantConditionId` in the bank manifest.
GEOMETRY_LEVELS_V6 = GEOMETRY_LEVELS_V5
DUMP_LEVELS_V6 = DUMP_LEVELS_V5
CAPACITY_LEVELS_V6 = CAPACITY_LEVELS_V5
DISTANCE_LEVELS_V6 = DISTANCE_LEVELS_V5
SCALE_LEVELS_V6 = {
    "": "std",
    "standard_extent": "std",
    "short_arm_mini": "s",
}
EASY_SCALE_V6 = frozenset({"std", "s"})
EMIT_SCALE_V6 = frozenset({"s"})
TIER_FACTORS_V6 = ("geometry", "dump", "capacity", "site", "distance", "scale")

SPEC_TABLE_V6_MAIN = (
    *SPEC_TABLE_V5_MAIN,
    # ---- D2 mini-junction rungs (spec v6 §2) ------------------------------
    # T0: the same tee baseline at a short extent.
    ("trn-tee-side2-s", "trn-tee-side2-s", 0, None),
    # T1: one geometry step off the T0 mini, at the same short extent.
    ("trn-net3-side2-s", "trn-net3-side2-s", 1, "trn-tee-side2-s"),
    ("trn-net4-side2-s", "trn-net4-side2-s", 1, "trn-tee-side2-s"),
)

# The standard variant each mini shrinks (spec v6 §2 "anchor = their standard
# variant"), kept as an explicit relation rather than as the taxonomy anchor.
MINI_STANDARD_VARIANT = {
    "trn-tee-side2-s": "trn-tee-side2",
    "trn-net3-side2-s": "trn-net3-side2",
    "trn-net4-side2-s": "trn-net4-side2",
}

V6_MAIN_RELEASE = ReleaseSpec(
    name="v6-main",
    table=SPEC_TABLE_V6_MAIN,
    geometry_levels=GEOMETRY_LEVELS_V6,
    dump_levels=DUMP_LEVELS_V6,
    capacity_levels=CAPACITY_LEVELS_V6,
    site_levels=SITE_LEVELS,
    distance_levels=DISTANCE_LEVELS_V6,
    easy_geometry=EASY_GEOMETRY_V5,
    easy_dump=EASY_DUMP_V5,
    easy_capacity=EASY_CAPACITY_V5,
    easy_site=EASY_SITE_V5,
    easy_distance=EASY_DISTANCE_V5,
    tier_factors=TIER_FACTORS_V6,
    emit_distance=EMIT_DISTANCE_V5,
    preview_conditions=frozenset(),
    scale_levels=SCALE_LEVELS_V6,
    easy_scale=EASY_SCALE_V6,
    emit_scale=EMIT_SCALE_V6,
)

RELEASES: dict[str, ReleaseSpec] = {
    "v3": V3_RELEASE,
    "v4": V4_RELEASE,
    "v5-main": V5_MAIN_RELEASE,
    "v5-transport": V5_TRANSPORT_RELEASE,
    "v6-main": V6_MAIN_RELEASE,
}
CONDITION_TABLE_V4 = V4_RELEASE.condition_table
CONDITION_TABLE_V5_MAIN = V5_MAIN_RELEASE.condition_table
CONDITION_TABLE_V5_TRANSPORT = V5_TRANSPORT_RELEASE.condition_table
CONDITION_TABLE_V6_MAIN = V6_MAIN_RELEASE.condition_table

# v6-main is v5-main plus the three minis and nothing else: the D1 gapped rings
# are a MAP delta, not a taxonomy delta, so every v5 row must survive verbatim.
assert set(CONDITION_TABLE_V5_MAIN) < set(CONDITION_TABLE_V6_MAIN)
assert set(CONDITION_TABLE_V6_MAIN) - set(CONDITION_TABLE_V5_MAIN) == set(
    MINI_STANDARD_VARIANT
), sorted(set(CONDITION_TABLE_V6_MAIN) - set(CONDITION_TABLE_V5_MAIN))
for _mini, _standard in MINI_STANDARD_VARIANT.items():
    assert _standard in CONDITION_TABLE_V5_MAIN, _standard
# v6 keeps the U8 split: transport is unchanged and still shares no condition.
assert not (set(CONDITION_TABLE_V6_MAIN) & set(CONDITION_TABLE_V5_TRANSPORT))

# The two v5 datasets must not share a condition (U8: transport leaves the main
# curriculum) — asserted here so a table edit cannot quietly re-merge them.
assert not (
    set(CONDITION_TABLE_V5_MAIN) & set(CONDITION_TABLE_V5_TRANSPORT)
), sorted(set(CONDITION_TABLE_V5_MAIN) & set(CONDITION_TABLE_V5_TRANSPORT))


def spec_for(release: str | ReleaseSpec | None = None) -> ReleaseSpec:
    if isinstance(release, ReleaseSpec):
        return release
    return RELEASES[release or DEFAULT_RELEASE]


def parse_condition_id(
    condition: str, release: str | ReleaseSpec | None = None
) -> dict[str, str]:
    """Invert the id grammar back into factor levels.

    The five level vocabularies are disjoint, so `<fam>-<geo>-<dump>[-cap]
    [-dist][-site]` parses without ambiguity (geometry is longest-match, which
    is what `slab-lg` needs). Round-tripping through `condition_id` is asserted
    here, so a vocabulary collision fails loudly instead of silently mis-parsing.
    """
    spec = spec_for(release)
    family, rest = condition.split("-", 1)
    assert family in {"fnd", "trn"}, condition
    geometries = sorted(set(spec.geometry_levels.values()), key=len, reverse=True)
    geometry = next(
        (level for level in geometries if rest == level or rest.startswith(level + "-")),
        None,
    )
    assert geometry is not None, f"no geometry level in {condition!r}"
    tokens = rest[len(geometry) :].strip("-").split("-") if rest != geometry else []
    dumps = set(spec.dump_levels["foundation"].values()) | set(
        spec.dump_levels["trench"].values()
    )
    capacities = set(spec.capacity_levels.values())
    distances = set(spec.distance_levels.values())
    sites = set(spec.site_levels.values()) - {"clean"}
    scales = set(spec.scale_levels.values()) - {"std"}
    levels = {
        "family": family,
        "geometry": geometry,
        "dump": "",
        "capacity": "generous",
        "site": "clean",
        "distance": "unspec" if spec.distance_levels else "near",
    }
    if spec.scale_levels:
        levels["scale"] = "std"
    for token in tokens:
        if token in dumps and not levels["dump"]:
            levels["dump"] = token
        elif token in capacities:
            levels["capacity"] = token
        elif token in distances:
            levels["distance"] = token
        elif token in sites:
            levels["site"] = token
        elif token in scales:
            levels["scale"] = token
        else:
            raise AssertionError(f"token {token!r} in {condition!r} is not a level")
    assert levels["dump"], f"no dump level in {condition!r}"
    if not spec.distance_levels and levels["dump"] == "remote":
        levels["distance"] = "far"
    assert condition_id(levels, spec) == condition, (condition, levels)
    return levels


@dataclass(frozen=True)
class Condition:
    cell_id: str
    condition_id: str
    family: str
    levels: dict[str, str]
    tier: int
    preview: bool
    anchor_condition_id: str | None
    map_count: int


def scenario_levels(
    scenario: dict[str, Any], release: str | ReleaseSpec | None = None
) -> dict[str, str]:
    """Factor levels for one scenario, derived from its manifest fields."""
    spec = spec_for(release)
    cell_id = str(scenario["cellId"])
    family = str(scenario["family"])
    factors = scenario.get("factors") or {}
    assert family in FAMILY_LEVELS, f"unknown family {family!r} in {cell_id}"

    geometry = GEOMETRY_CELL_OVERRIDES.get(
        cell_id, spec.geometry_levels[str(factors["geometryClass"])]
    )
    dump = spec.dump_levels[family][str(factors["dumpLayout"])]
    capacity_band = factors.get("capacityBand")
    capacity = CAPACITY_CELL_OVERRIDES.get(
        cell_id,
        spec.capacity_levels[str(capacity_band)] if capacity_band else "generous",
    )
    site = spec.site_levels[str(factors.get("siteClass") or "none")]
    if spec.distance_levels:
        # v4 makes `distance` a real, scoring factor, so it may not double-count
        # what the dump level already says: a `remote` dump IS the far haul, and
        # its distance level stays `unspec` (see §8.3 U6 — the ladder is the
        # controlled distance axis, and it is foundation-apron only).
        distance = spec.distance_levels[str(factors.get("distanceBand") or "")]
    else:
        distance = "far" if dump == "remote" else "near"
    levels = {
        "family": FAMILY_LEVELS[family],
        "geometry": geometry,
        "dump": dump,
        "capacity": capacity,
        "site": site,
        "distance": distance,
    }
    if spec.scale_levels:
        levels["scale"] = spec.scale_levels[str(factors.get("scaleBand") or "")]
    return levels


def condition_id(levels: dict[str, str], release: str | ReleaseSpec | None = None) -> str:
    """`<fam>-<geometry>-<dump>[-<capacity>][-<distance>][-<site>]`.

    Generous capacity and clean sites are always omitted. Distance only appears
    for the levels the release declares in `emit_distance` — v3 emits none, so
    v3 ids are unchanged.
    """
    spec = spec_for(release)
    parts = [levels["family"], levels["geometry"], levels["dump"]]
    if levels["capacity"] != "generous":
        parts.append(levels["capacity"])
    if levels.get("distance") in spec.emit_distance:
        parts.append(levels["distance"])
    if levels["site"] != "clean":
        parts.append(levels["site"])
    if levels.get("scale") in spec.emit_scale:
        parts.append(levels["scale"])
    return "-".join(parts)


def is_easy(
    factor: str, levels: dict[str, str], release: str | ReleaseSpec | None = None
) -> bool:
    spec = spec_for(release)
    family = levels["family"]
    level = levels[factor]
    if factor == "geometry":
        return level in spec.easy_geometry[family]
    if factor == "dump":
        return level in spec.easy_dump[family]
    if factor == "capacity":
        return level in spec.easy_capacity
    if factor == "site":
        return level in spec.easy_site
    if factor == "distance" and "distance" in spec.tier_factors:
        return level in spec.easy_distance
    if factor == "scale" and "scale" in spec.tier_factors:
        # Both scale levels are easy by construction: shrinking a trench cannot
        # make it harder. `scale` is in `tier_factors` only so that it appears in
        # the delta against the anchor — it contributes 0 to every tier.
        return level in spec.easy_scale
    raise AssertionError(f"{factor} is not a difficulty factor")


def tier(levels: dict[str, str], release: str | ReleaseSpec | None = None) -> int:
    """Number of factors sitting at a non-easy level."""
    spec = spec_for(release)
    return sum(0 if is_easy(factor, levels, spec) else 1 for factor in spec.tier_factors)


def delta(
    levels: dict[str, str],
    anchor_levels: dict[str, str] | None,
    release: str | ReleaseSpec | None = None,
) -> list[dict[str, str]]:
    """Factor-wise diff against the anchor condition (empty for anchors)."""
    if anchor_levels is None:
        return []
    spec = spec_for(release)
    entries = []
    for factor in spec.tier_factors:
        if levels[factor] == anchor_levels[factor]:
            continue
        old = LEVEL_LABELS[factor][anchor_levels[factor]]
        new = LEVEL_LABELS[factor][levels[factor]]
        entries.append(
            {
                "factor": factor,
                "from": anchor_levels[factor],
                "to": levels[factor],
                "label": f"{FACTOR_LABELS[factor]}: {old} → {new}",
            }
        )
    return entries


def release_cells(
    scenarios: list[dict[str, Any]], release: str | ReleaseSpec | None = None
) -> list[tuple[str, str, int, str | None]]:
    """Table rows this release actually covers, in spec section 2 order.

    A release may cover a subset of the table (v2 predates `obj1`), and a
    taxonomy-native v3 release keys its cells by condition id rather than by the
    old cell id. Both are resolved here; unknown cells are a hard error.
    """
    spec = spec_for(release)
    by_cell = spec.table_by_cell
    present = {str(scenario["cellId"]) for scenario in scenarios}
    assert present <= set(by_cell), (
        f"cells not covered by the {spec.name} spec table: "
        f"{sorted(present - set(by_cell))}"
    )
    rows = []
    for cell_id, condition, tier_, anchor in spec.resolved:
        key = cell_id if cell_id in present else condition
        if key in present:
            rows.append((key, condition, tier_, anchor))
    return rows


def build_conditions(
    scenarios: list[dict[str, Any]], release: str | ReleaseSpec | None = None
) -> dict[str, Condition]:
    """One Condition per cell, in spec section 2 order, asserted against the table."""
    spec = spec_for(release)
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        by_cell.setdefault(str(scenario["cellId"]), []).append(scenario)

    conditions: dict[str, Condition] = {}
    for cell_id, expected_id, spec_tier, anchor in release_cells(scenarios, spec):
        rows = by_cell[cell_id]
        levels = scenario_levels(rows[0], spec)
        for row in rows[1:]:
            assert scenario_levels(row, spec) == levels, (
                f"{cell_id} mixes factor levels: {row['id']}"
            )
        derived_id = condition_id(levels, spec)
        derived_tier = tier(levels, spec)
        assert derived_id == expected_id, (
            f"{cell_id}: derived {derived_id} != spec {expected_id}"
        )
        assert derived_tier == spec_tier, (
            f"{cell_id}: derived tier {derived_tier} != spec tier {spec_tier}"
        )
        conditions[cell_id] = Condition(
            cell_id=cell_id,
            condition_id=derived_id,
            family=str(rows[0]["family"]),
            levels=levels,
            tier=derived_tier,
            preview=cell_id in PREVIEW_CELLS or derived_id in spec.preview_conditions,
            anchor_condition_id=anchor,
            map_count=len(rows),
        )

    by_id = {condition.condition_id: condition for condition in conditions.values()}
    assert len(by_id) == len(conditions), "condition ids are not unique"
    for condition in conditions.values():
        if condition.anchor_condition_id is None:
            assert condition.tier == 0, f"{condition.condition_id} has no anchor but tier > 0"
            continue
        if condition.anchor_condition_id in spec.external_anchors:
            # cross-release anchor (spec §8.5 U8): recorded, resolved elsewhere
            assert condition.anchor_condition_id not in by_id, (
                f"{condition.condition_id}: external anchor "
                f"{condition.anchor_condition_id} is also in this release"
            )
            continue
        anchor = by_id[condition.anchor_condition_id]
        assert anchor.tier < condition.tier, (
            f"{condition.condition_id} anchor {anchor.condition_id} is not easier"
        )
        assert len(
            delta(condition.levels, anchor.levels, spec)
        ) == condition.tier - anchor.tier, (
            f"{condition.condition_id} delta size does not match its tier"
        )
    return conditions


def taxonomy_block(
    condition: Condition,
    by_id: dict[str, Condition],
    release: str | ReleaseSpec | None = None,
) -> dict[str, Any]:
    """The per-scenario `taxonomy` value (spec section 3).

    The anchor lookup is tolerant on purpose: `build_conditions` already proved
    that every in-release anchor resolves, so the only way to miss here is a
    cross-release anchor (`ReleaseSpec.external_anchors`, spec §8.5 U8). Such an
    anchor is still named in `anchorConditionId`; its factor delta simply cannot
    be computed inside this release. v3/v4 declare no external anchors, so their
    payloads are unchanged.
    """
    anchor = by_id.get(condition.anchor_condition_id) if condition.anchor_condition_id else None
    return {
        "conditionId": condition.condition_id,
        "tier": condition.tier,
        "tierLabel": TIER_LABELS[condition.tier],
        "preview": condition.preview,
        "anchorConditionId": condition.anchor_condition_id,
        "factorLevels": dict(condition.levels),
        "delta": delta(condition.levels, anchor.levels if anchor else None, release),
    }


def annotate_scenarios(
    scenarios: list[dict[str, Any]], release: str | ReleaseSpec | None = None
) -> dict[str, Condition]:
    """Add the `taxonomy` key to every scenario in place."""
    conditions = build_conditions(scenarios, release)
    by_id = {condition.condition_id: condition for condition in conditions.values()}
    for scenario in scenarios:
        scenario["taxonomy"] = taxonomy_block(
            conditions[str(scenario["cellId"])], by_id, release
        )
    return conditions


def release_taxonomy(
    scenarios: list[dict[str, Any]], release: str | ReleaseSpec | None = None
) -> dict[str, Any]:
    """The top-level `taxonomy` value (spec section 3)."""
    conditions = build_conditions(scenarios, release)
    return {
        "version": TAXONOMY_VERSION,
        "specPath": SPEC_PATH,
        "conditions": [
            {
                "id": condition.condition_id,
                "oldCellId": condition.cell_id,
                "family": condition.family,
                "tier": condition.tier,
                "preview": condition.preview,
                "anchorConditionId": condition.anchor_condition_id,
                "factorLevels": dict(condition.levels),
                "mapCount": condition.map_count,
            }
            for condition in (
                conditions[row[0]] for row in release_cells(scenarios, release)
            )
        ],
    }


CSV_COLUMNS = (
    "condition_id",
    "old_cell_id",
    "family",
    "geometry",
    "dump",
    "capacity",
    "site",
    "distance",
    "tier",
    "preview",
    "anchor_condition_id",
    "n_maps",
)


def csv_columns(release: str | ReleaseSpec | None = None) -> tuple[str, ...]:
    """Spec §4 columns, plus `scale` on releases that declare the axis (v6)."""
    spec = spec_for(release)
    return (*CSV_COLUMNS, "scale") if spec.scale_levels else CSV_COLUMNS


def conditions_csv_rows(
    scenarios: list[dict[str, Any]], release: str | ReleaseSpec | None = None
) -> list[dict[str, Any]]:
    """One row per condition, in spec section 2 order (spec section 4 columns)."""
    spec = spec_for(release)
    conditions = build_conditions(scenarios, spec)
    rows = []
    for cell_id, *_ in release_cells(scenarios, spec):
        condition = conditions[cell_id]
        row = {
            "condition_id": condition.condition_id,
            "old_cell_id": condition.cell_id,
            "family": condition.family,
            "geometry": condition.levels["geometry"],
            "dump": condition.levels["dump"],
            "capacity": condition.levels["capacity"],
            "site": condition.levels["site"],
            "distance": condition.levels["distance"],
            "tier": condition.tier,
            "preview": "true" if condition.preview else "false",
            "anchor_condition_id": condition.anchor_condition_id or "",
            "n_maps": condition.map_count,
        }
        if spec.scale_levels:
            row["scale"] = condition.levels["scale"]
        rows.append(row)
    return rows
