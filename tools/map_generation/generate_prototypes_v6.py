#!/usr/bin/env python3
"""Taxonomy-native v3 curriculum review bank.

Spec: ``terra-digging-benchmark-site/docs/CURRICULUM_TAXONOMY_SPEC.md`` section 8.

What changes versus v5:

* the generation unit is a **condition** (24 of them), not a geometry x dump x
  site grid cell. ``cellId == conditionId``;
* ``clean`` really means zero objects. The v5 ``light`` site style placed
  ``rng.integers(0, 3)`` small objects and the taxonomy maps ``light -> clean``,
  which is the entire v2 object leak;
* ``obj`` is an enforced 4-8 object band and the new ``obj1`` condition is an
  enforced 1-3 object band;
* excavation targets are drawn from a per-geometry **dig bank** indexed by map
  number, so every condition sharing a geometry level sees the same 16 digs.
  Capacity siblings therefore share targets and sites by construction;
* explicit dump-capacity bands (7-10x, 3-4x, ~1.6x, ~11x) are generated
  procedurally instead of being imported from the frozen pilot banks.

Everything else - geometry factories, dump algorithms, site obstacles, the
static gate, Terra array layout - is reused from the v2/v3/v4/v5 chain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import generate_prototypes_v5 as v5

v4 = v5.v4
v3 = v5.v3
v2 = v5.v2
base = v5.base

import curriculum_taxonomy as tax


SCHEMA = "terra_curriculum_v3_review_bank"
SEED_BASE = 20260728
MAPS_PER_CONDITION = 16
PREVIEW_MAPS_PER_CONDITION = 4
# Attempts that keep the shared dig from the bank before the dig itself is
# re-rolled. A re-roll is recorded per map as shared_dig=0, never hidden.
SHARED_DIG_ATTEMPTS = 40
MAX_ATTEMPTS = 160

OBJECT_BANDS = {
    "clean": (0, 0),
    "obj": (4, 8),
    "obj1": (1, 3),
    "road": (0, 0),
    "wall": (0, 0),
}
SITE_CLASS_TOKENS = {
    "clean": "none",
    "obj": "scattered_objects",
    "obj1": "scattered_objects_light",
    "road": "access_road",
    "wall": "gapped_wall",
}
CAPACITY_TOKENS = {
    "generous": "",
    "c7x": "slcap07_10",
    "c3x": "slcap03_04",
    "c11x": "all_around_11_08_11_49x",
    "c1p6": "legacy_apron_1p6",
    "tight": "trench_constrained",
}

# Source-bank slabs are held to a narrow footprint so that an 7-10x apron is
# geometrically feasible on the same dig that the ring conditions use.
SLAB_DIG_CELLS = (120, 250)
SLAB_MAX_DIM = 30
# ring capacity is exactly (4096 - dig) / dig, so the 11.08-11.49x band of the
# frozen large-foundation cell pins the dig area to 328-339 cells (8.0-8.3%).
LARGE_SLAB_DIG_CELLS = (328, 339)
LARGE_SLAB_SOURCE_CELLS = (150, 300)
LARGE_SLAB_MAX_DIM = 40

RING_MIN_CAPACITY = 3.0
RING_CAPACITY_BANDS = {"c11x": (11.00, 11.55)}
APRON_CAPACITY_BANDS = {
    "c7x": (7.0, 10.0),
    "c3x": (3.0, 4.0),
    "c1p6": (1.55, 1.75),
}

GEOMETRY_HARDNESS = dict(v3.GEOMETRY_HARDNESS)
GEOMETRY_HARDNESS["foundation_osm_large"] = "foundation_real_large"

GEOMETRY_LEVEL_SOURCE = {
    "slab": "foundation_osm",
    "slab-lg": "foundation_osm_large",
    "proc": "foundation_procedural",
    "strips": "foundation_structural",
    "straight": "trench_axes_1",
    "tee": "trench_axes_2",
    "net": "trench_axes_3",
}
GEOMETRY_LEVEL_INDEX = {
    level: index for index, level in enumerate(sorted(GEOMETRY_LEVEL_SOURCE))
}


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
    dump_style: str
    dump_layout: str
    pair_group: str | None

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
            "distance": "far" if self.dump_level == "remote" else "near",
        }

    @property
    def tier(self) -> int:
        return tax.tier(self.levels)


def _fnd(
    condition_id: str,
    anchor: str | None,
    geometry_level: str,
    dump_level: str,
    capacity_level: str,
    site_level: str,
    dump_style: str,
    dump_layout: str,
    *,
    preview: bool = False,
    pair_group: str | None = None,
) -> ConditionSpec:
    return ConditionSpec(
        condition_id,
        anchor,
        preview,
        "foundation",
        geometry_level,
        dump_level,
        capacity_level,
        site_level,
        dump_style,
        dump_layout,
        pair_group,
    )


def _trn(
    condition_id: str,
    anchor: str | None,
    geometry_level: str,
    dump_level: str,
    capacity_level: str,
    site_level: str,
    dump_style: str,
    dump_layout: str,
    *,
    preview: bool = False,
    pair_group: str | None = None,
) -> ConditionSpec:
    return ConditionSpec(
        condition_id,
        anchor,
        preview,
        "trench",
        geometry_level,
        dump_level,
        capacity_level,
        site_level,
        dump_style,
        dump_layout,
        pair_group,
    )


# Spec section 2 plus the new obj1 row. `dump_style` is the generator algorithm,
# `dump_layout` the manifest token the taxonomy module maps to a dump level.
CONDITIONS: tuple[ConditionSpec, ...] = (
    _fnd("fnd-slab-apron-c7x", None, "slab", "apron", "c7x", "clean",
         "capacity_apron", "near_apron_large", pair_group="slab-apron"),
    _fnd("fnd-slab-lg-ring-c11x", None, "slab-lg", "ring", "c11x", "clean",
         "ring_all_legal_free", "easy_surround"),
    _fnd("fnd-slab-ring", None, "slab", "ring", "generous", "clean",
         "ring_all_legal_free", "easy_surround"),
    _fnd("fnd-proc-ring", None, "proc", "ring", "generous", "clean",
         "ring_all_legal_free", "easy_surround"),
    _fnd("fnd-slab-apron-c3x", "fnd-slab-apron-c7x", "slab", "apron", "c3x", "clean",
         "capacity_apron", "near_apron_large", pair_group="slab-apron"),
    _fnd("fnd-slab-apron-c1p6", "fnd-slab-apron-c7x", "slab", "apron", "c1p6", "clean",
         "capacity_apron", "near_apron_large", pair_group="slab-apron"),
    _fnd("fnd-slab-side1", "fnd-slab-ring", "slab", "side1", "generous", "clean",
         "one_side_near", "one_side_near"),
    _fnd("fnd-slab-split", "fnd-slab-ring", "slab", "split", "generous", "clean",
         "separated_zones", "separated_zones"),
    _fnd("fnd-strips-ring", "fnd-slab-ring", "strips", "ring", "generous", "clean",
         "ring_all_legal_free", "easy_surround"),
    _fnd("fnd-slab-ring-obj", "fnd-slab-ring", "slab", "ring", "generous", "obj",
         "ring_all_legal_free", "easy_surround"),
    _fnd("fnd-slab-ring-obj1", "fnd-slab-ring", "slab", "ring", "generous", "obj1",
         "ring_all_legal_free", "easy_surround"),
    _fnd("fnd-slab-ring-road", "fnd-slab-ring", "slab", "ring", "generous", "road",
         "ring_all_legal_free", "easy_surround"),
    _fnd("fnd-proc-side1-road", "fnd-proc-ring", "proc", "side1", "generous", "road",
         "one_side_near", "one_side_near"),
    _fnd("fnd-strips-split-wall", "fnd-slab-ring", "strips", "split", "generous", "wall",
         "separated_zones", "separated_zones"),
    _fnd("fnd-slab-remote", "fnd-slab-ring", "slab", "remote", "generous", "clean",
         "haul_away_edge", "haul_away_edge", preview=True),
    _trn("trn-straight-side2", None, "straight", "side2", "generous", "clean",
         "easy_surround", "easy_surround"),
    _trn("trn-straight-side1", None, "straight", "side1", "generous", "clean",
         "near_apron_large", "near_apron_large"),
    _trn("trn-straight-side1-tight", "trn-straight-side1", "straight", "side1", "tight", "clean",
         "one_side_near", "one_side_near"),
    _trn("trn-straight-split", "trn-straight-side2", "straight", "split", "generous", "clean",
         "separated_zones", "separated_zones"),
    _trn("trn-tee-side2", "trn-straight-side2", "tee", "side2", "generous", "clean",
         "easy_surround", "easy_surround"),
    _trn("trn-net-side2", "trn-straight-side2", "net", "side2", "generous", "clean",
         "easy_surround", "easy_surround"),
    _trn("trn-tee-side1-road", "trn-straight-side1", "tee", "side1", "generous", "road",
         "near_apron_large", "near_apron_large"),
    _trn("trn-net-split-wall", "trn-straight-side2", "net", "split", "generous", "wall",
         "separated_zones", "separated_zones"),
    _trn("trn-straight-remote", "trn-straight-side2", "straight", "remote", "generous", "clean",
         "haul_away_edge", "haul_away_edge", preview=True),
)


def assert_conditions_match_taxonomy() -> None:
    """The generator table and scripts/curriculum_taxonomy.py must agree exactly."""
    table = tax.CONDITION_TABLE
    assert len(CONDITIONS) == 24, f"expected 24 conditions, got {len(CONDITIONS)}"
    assert {c.id for c in CONDITIONS} == set(table), (
        f"condition set mismatch: {sorted({c.id for c in CONDITIONS} ^ set(table))}"
    )
    for condition in CONDITIONS:
        derived = tax.condition_id(condition.levels)
        assert derived == condition.id, f"{condition.id}: id grammar gives {derived}"
        expected_tier, expected_anchor = table[condition.id]
        assert condition.tier == expected_tier, (
            f"{condition.id}: tier {condition.tier} != table {expected_tier}"
        )
        assert condition.anchor == expected_anchor, (
            f"{condition.id}: anchor {condition.anchor} != table {expected_anchor}"
        )
        assert tax.DUMP_LEVELS[condition.family][condition.dump_layout] == condition.dump_level
        assert tax.SITE_LEVELS[SITE_CLASS_TOKENS[condition.site_level]] == condition.site_level
        token = CAPACITY_TOKENS[condition.capacity_level]
        if token:
            assert tax.CAPACITY_LEVELS[token] == condition.capacity_level
        assert tax.GEOMETRY_LEVELS[condition.geometry] == condition.geometry_level
    previews = {c.id for c in CONDITIONS if c.preview}
    assert previews == {"fnd-slab-remote", "trn-straight-remote"}, previews


# --------------------------------------------------------------------------
# geometry


def _max_dim(mask: np.ndarray) -> int:
    ys, xs = np.where(mask)
    return max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))


def _scale_to_band(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """Rescale an OSM footprint into the large-slab cell band."""
    ys, xs = np.where(mask)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    low, high = LARGE_SLAB_DIG_CELLS
    goal = float(rng.uniform(low + 2, high - 2))
    ideal = math.sqrt(goal / float(crop.sum()))
    for factor in ideal * np.linspace(0.94, 1.10, 25):
        scaled = ndi.zoom(crop.astype(float), float(factor), order=1) > 0.5
        if not scaled.any():
            continue
        scaled = base.largest_component(scaled)
        cells = int(scaled.sum())
        if low <= cells <= high and max(scaled.shape) <= LARGE_SLAB_MAX_DIM:
            return scaled
    return None


class GeometryFactoryV6(v3.GeometryFactoryV3):
    def __init__(self, source_root: Path) -> None:
        super().__init__(source_root)
        self.slab_sources = [
            dig
            for dig in self.foundation_sources
            if SLAB_DIG_CELLS[0] <= int(dig.sum()) <= SLAB_DIG_CELLS[1]
            and _max_dim(dig) <= SLAB_MAX_DIM
        ]
        self.scale_sources = [
            dig
            for dig in self.foundation_sources
            if LARGE_SLAB_SOURCE_CELLS[0] <= int(dig.sum()) <= LARGE_SLAB_SOURCE_CELLS[1]
        ]
        assert len(self.slab_sources) >= 100, len(self.slab_sources)
        assert len(self.scale_sources) >= 100, len(self.scale_sources)

    def foundation_osm(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for _ in range(60):
            index = int(rng.integers(0, len(self.slab_sources)))
            dig = np.rot90(self.slab_sources[index], int(rng.integers(0, 4)))
            if rng.random() < 0.5:
                dig = np.fliplr(dig)
            placed = self._place_crop(dig, rng)
            if placed is not None:
                return placed, {
                    "foundation_source_index": index,
                    "foundation_size_class": "slab",
                }
        raise RuntimeError("Could not place an OSM foundation source")

    def foundation_osm_large(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for _ in range(60):
            index = int(rng.integers(0, len(self.scale_sources)))
            dig = np.rot90(self.scale_sources[index], int(rng.integers(0, 4)))
            if rng.random() < 0.5:
                dig = np.fliplr(dig)
            scaled = _scale_to_band(dig, rng)
            if scaled is None:
                continue
            placed = self._place_crop(scaled, rng)
            if placed is not None:
                return placed, {
                    "foundation_source_index": index,
                    "foundation_size_class": "slab_large",
                    "foundation_source_cells": int(dig.sum()),
                }
        raise RuntimeError("Could not place a large OSM foundation source")


class DigBank:
    """Excavation targets keyed by (geometry level, map index).

    Every condition on the same geometry level sees the same dig for map k, so
    dump-layout, capacity, and site siblings are controlled comparisons.
    """

    def __init__(self, factory: GeometryFactoryV6) -> None:
        self.factory = factory
        self.cache: dict[tuple[str, int, int], tuple[np.ndarray, dict[str, Any]]] = {}

    def get(
        self, geometry_level: str, map_index: int, salt: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        key = (geometry_level, map_index, salt)
        if key not in self.cache:
            geometry = GEOMETRY_LEVEL_SOURCE[geometry_level]
            seed = int(
                np.random.SeedSequence(
                    [
                        SEED_BASE,
                        7777,
                        GEOMETRY_LEVEL_INDEX[geometry_level],
                        map_index,
                        salt,
                    ]
                ).generate_state(1)[0]
            )
            rng = np.random.default_rng(seed)
            dig, meta = self.factory.make(geometry, rng)
            meta = {
                **meta,
                "geometry": geometry,
                "geometry_hardness": GEOMETRY_HARDNESS[geometry],
                "dig_bank_seed": seed,
                "dig_bank_salt": salt,
            }
            self.cache[key] = (dig, meta)
        dig, meta = self.cache[key]
        return dig.copy(), dict(meta)


# --------------------------------------------------------------------------
# dump


class DumpFactoryV6(v5.DumpFactoryV5):
    def __init__(
        self,
        geometry: str,
        geometry_metadata: dict[str, Any],
        capacity_band: tuple[float, float] | None,
    ) -> None:
        super().__init__(geometry, geometry_metadata)
        self.capacity_band = capacity_band

    def capacity_apron(
        self,
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Nearby apron grown outward until the requested capacity band fits."""
        del side
        distance = ndi.distance_transform_edt(~dig)
        yy, xx = np.indices(dig.shape)
        cy, cx = ndi.center_of_mass(dig)
        angles = np.arctan2(yy - cy, xx - cx)
        center_angle = float(rng.uniform(-math.pi, math.pi))
        for sector_degrees in (200, 260, 320, 360):
            half = math.radians(sector_degrees) / 2
            for radius in range(8, 27, 2):
                allowed = (
                    (distance >= 3)
                    & (distance <= radius)
                    & (v2.angle_difference(angles, center_angle) <= half)
                )
                allowed = base.largest_component(allowed)
                if int(allowed.sum()) < 1.12 * target_area:
                    continue
                seed = v2.seed_near_angle(
                    allowed, dig, center_angle, preferred_distance=4.5
                )
                target = base.grow_region(allowed, seed, target_area, rng)
                return target, {
                    "dump_side": "near_apron",
                    "dump_components_requested": 1,
                    "dump_sector_degrees": float(sector_degrees),
                    "apron_max_offset_tiles": radius,
                    "distance_bucket": "near",
                }
        raise RuntimeError("apron_capacity_infeasible")

    def make(
        self,
        style: str,
        dig: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if style != "capacity_apron":
            return super().make(style, dig, rng)
        assert self.capacity_band is not None
        capacity_factor = float(rng.uniform(*self.capacity_band))
        target_area = int(math.ceil(int(dig.sum()) * capacity_factor))
        target, metadata = self.capacity_apron(dig, target_area, 0, rng)
        metadata.update(
            {
                "dump_target_cells_requested": target_area,
                "capacity_factor_required": round(capacity_factor, 5),
                "difficulty_tier": "explicit_capacity_band",
                "dump_alignment": "foundation_capacity_apron",
            }
        )
        return target, metadata


# --------------------------------------------------------------------------
# site


class SiteFactoryV6:
    """Site constraints keyed by taxonomy site level, with hard object bands."""

    @staticmethod
    def make(
        level: str,
        dig: np.ndarray,
        dump: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        occupancy = np.zeros_like(dig, dtype=bool)
        nondump = np.zeros_like(dig, dtype=bool)
        corridor = np.zeros_like(dig, dtype=bool)
        forbidden = ndi.binary_dilation(dig | dump, structure=base.binary_disk(3))
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
            occupancy, placed = base.place_objects(
                rng, occupancy, forbidden, requested, large=level == "obj"
            )
            if placed != requested:
                raise RuntimeError("object_band_shortfall")
            metadata["object_count"] = placed
        elif level == "road":
            road = base.make_access_road(rng, dig, dump)
            if int(road.sum()) < 90:
                raise RuntimeError("access_road_generation_failed")
            nondump |= road
            corridor |= road
        elif level == "wall":
            wall, gap = base.make_gapped_wall(rng, dig, dump)
            if int(wall.sum()) < 25 or gap < 12:
                raise RuntimeError("gapped_wall_generation_failed")
            occupancy |= wall
            metadata["wall_gap_tiles"] = gap
        else:
            raise ValueError(f"unknown site level {level!r}")

        occupancy &= ~(dig | dump)
        nondump &= ~(dig | dump | occupancy)
        corridor &= nondump
        dumpability = ~(nondump | occupancy)
        return occupancy, dumpability, corridor, metadata


# --------------------------------------------------------------------------
# sample construction


def _common_metadata(
    condition: ConditionSpec,
    dig: np.ndarray,
    dump: np.ndarray,
    occupancy: np.ndarray,
    dumpability: np.ndarray,
    gate: base.StaticGate,
) -> dict[str, Any]:
    capacity_ratio = float(dump.sum() / max(1, dig.sum()))
    reachable_ratio = float(gate.reachable_dump_cells_post / max(1, dig.sum()))
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
        "dump_to_dig_area_ratio": round(capacity_ratio, 4),
        "reachable_dump_to_dig_ratio": round(reachable_ratio, 4),
        "obstacle_cells": int(occupancy.sum()),
        "nondump_cells": int((~dumpability & ~occupancy).sum()),
        "tile_size_m": base.TILE_SIZE_M,
        "static_gate_is_action_witness": False,
    }


def make_ring_sample(
    condition: ConditionSpec,
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[base.Sample | None, str]:
    """Foundation ring: every legal free cell around the dig is a dump target."""
    planning_area = int(math.ceil(1.25 * int(dig.sum())))
    planning_dump, _ = v2.DumpFactoryV2.broad_nearby(
        dig, planning_area, int(rng.integers(0, 4)), rng
    )
    if int(planning_dump.sum()) < planning_area:
        return None, "planning_dump_generation_shortfall"
    occupancy, dumpability, corridor, site_meta = SiteFactoryV6.make(
        condition.site_level, dig, planning_dump, rng
    )

    dump = (~dig) & (~occupancy) & dumpability
    target = np.zeros((base.MAP_SIZE, base.MAP_SIZE), dtype=np.int8)
    target[dig] = -1
    target[dump] = 1
    gate = base.static_gate(target, occupancy, dumpability)
    if not gate.accepted:
        return None, gate.reason

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
        "capacity_factor_required": (
            band[0] if band is not None else RING_MIN_CAPACITY
        ),
        "difficulty_tier": "all_legal_free",
        "dump_side": "foundation_all_around",
        "dump_access_sides": "all",
        "dump_alignment": "all_legal_free_around_foundation",
        "distance_bucket": "immediate",
        "dump_coverage_of_legal_free": round(coverage, 6),
        "planning_dump_cells": int(planning_dump.sum()),
        **v2.dig_to_dump_distance_metrics(dig, dump),
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
    condition: ConditionSpec,
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[base.Sample | None, str]:
    """Every non-ring dump layout: apron band, one side, split zones, remote."""
    factory = DumpFactoryV6(
        condition.geometry,
        dig_meta,
        APRON_CAPACITY_BANDS.get(condition.capacity_level),
    )
    dump, dump_meta = factory.make(condition.dump_style, dig, rng)
    requested = int(dump_meta["dump_target_cells_requested"])
    if int(dump.sum()) < requested:
        return None, "capacity_generation_shortfall"
    occupancy, dumpability, corridor, site_meta = SiteFactoryV6.make(
        condition.site_level, dig, dump, rng
    )

    target = np.zeros((base.MAP_SIZE, base.MAP_SIZE), dtype=np.int8)
    target[dig] = -1
    target[dump & ~dig] = 1
    components = base.target_components(target)
    if components != int(dump_meta["dump_components_requested"]):
        return None, "dump_component_contract"

    gate = base.static_gate(target, occupancy, dumpability)
    if not gate.accepted:
        return None, gate.reason

    required = float(dump_meta["capacity_factor_required"])
    capacity_ratio = float(dump.sum() / max(1, dig.sum()))
    reachable_ratio = float(gate.reachable_dump_cells_post / max(1, dig.sum()))
    band = APRON_CAPACITY_BANDS.get(condition.capacity_level)
    if band is not None:
        # An explicit capacity band is a statement about reachable single-layer
        # capacity, so the band - not the requested factor - is the contract.
        if not band[0] <= reachable_ratio <= band[1]:
            return None, "capacity_band_contract"
    else:
        if capacity_ratio + 1e-8 < required:
            return None, "capacity_ratio_contract"
        if reachable_ratio + 1e-8 < required:
            return None, "reachable_capacity_contract"

    distance_metrics = v2.dig_to_dump_distance_metrics(dig, dump)
    limits = (
        v5.TRENCH_MAX_MEDIAN_DISTANCE_TILES
        if condition.family == "trench"
        else v2.MAX_MEDIAN_DISTANCE_TILES
    )
    limit = limits.get(condition.dump_style)
    if (
        limit is not None
        and distance_metrics["dig_dump_distance_median_tiles"] > limit
    ):
        return None, "near_distance_contract"

    metadata = {
        **_common_metadata(condition, dig, dump, occupancy, dumpability, gate),
        "dump_components_actual": components,
        **distance_metrics,
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
    condition: ConditionSpec,
    dig: np.ndarray,
    dig_meta: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[base.Sample | None, str]:
    maker = (
        make_ring_sample
        if condition.dump_style == "ring_all_legal_free"
        else make_zoned_sample
    )
    try:
        return maker(condition, dig, dig_meta, rng)
    except RuntimeError as exc:
        return None, str(exc)


def generate_condition(
    condition: ConditionSpec,
    condition_index: int,
    bank: DigBank,
    n_maps: int,
    max_attempts: int,
) -> tuple[list[base.Sample], Counter[str], list[str]]:
    samples: list[base.Sample] = []
    rejections: Counter[str] = Counter()
    unsatisfied: list[str] = []
    for map_index in range(n_maps):
        accepted: base.Sample | None = None
        for attempt in range(max_attempts):
            salt = 0 if attempt < SHARED_DIG_ATTEMPTS else attempt
            dig, dig_meta = bank.get(condition.geometry_level, map_index, salt)
            seed = int(
                np.random.SeedSequence(
                    [SEED_BASE, condition_index, map_index, attempt]
                ).generate_state(1)[0]
            )
            sample, reason = make_map(
                condition, dig, dig_meta, np.random.default_rng(seed)
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
            break
        if accepted is None:
            unsatisfied.append(
                f"{condition.id} map {map_index}: no accepted sample in "
                f"{max_attempts} attempts; rejections={dict(rejections)}"
            )
            continue
        samples.append(accepted)
    return samples, rejections, unsatisfied


def sha256_mask(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    ).hexdigest()


# --------------------------------------------------------------------------
# output


def sample_index_of(condition_index: int, map_index: int) -> int:
    return 100 * condition_index + map_index


def map_id_of(sample_index: int) -> str:
    return f"curriculum-v3-{sample_index:04d}"


ARRAY_FOLDERS = {
    "images": "target",
    "occupancy": "occupancy",
    "dumpability": "dumpability",
    "actions": "action",
    "distance": "distance",
}


def write_condition(
    output: Path,
    condition: ConditionSpec,
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
    render_condition_overview(folder / "overview.png", condition, samples)
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


def render_condition_overview(
    path: Path, condition: ConditionSpec, samples: list[base.Sample]
) -> None:
    count = len(samples)
    columns = 4 if count > 4 else max(1, count)
    rows = int(math.ceil(count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.9 * columns, 3.1 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for index in range(rows * columns):
        ax = axes[index // columns][index % columns]
        ax.set_xticks([])
        ax.set_yticks([])
        if index >= count:
            ax.axis("off")
            continue
        sample = samples[index]
        ax.imshow(
            base.render_code(sample),
            cmap=base.COLORS,
            vmin=0,
            vmax=4,
            interpolation="nearest",
        )
        ax.set_title(
            f"#{sample.metadata['map_index']:02d} "
            f"dig {sample.metadata['dig_cells']} "
            f"cap {sample.metadata['dump_to_dig_area_ratio']:.2f}x "
            f"obj {sample.metadata['object_count']}",
            fontsize=8,
        )
    fig.suptitle(
        f"{condition.id} — T{condition.tier}"
        f"{' · preview' if condition.preview else ''}",
        fontsize=15,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_manifest(output: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
        "# Terra curriculum v3 review bank",
        "",
        "Taxonomy-native bank: one folder per **condition id**, no stage folders.",
        "Difficulty tier is computed from factor levels, never hand-assigned.",
        "See `../../terra-digging-benchmark-site/docs/CURRICULUM_TAXONOMY_SPEC.md`.",
        "",
        f"- seed base: `{SEED_BASE}` (fully reproducible)",
        f"- conditions: {len(CONDITIONS)}",
        f"- maps: {sum(counts.values())} "
        f"({MAPS_PER_CONDITION} per condition, "
        f"{PREVIEW_MAPS_PER_CONDITION} for the two remote-haul previews)",
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
        "- `GENERATION_NOTES.md` — leak diagnosis, seeds, constraint reports",
        "",
        "## Object control",
        "",
        "- `clean` conditions have **zero** objects, asserted at generation",
        "- `obj` conditions carry 4-8 objects, `obj1` carries 1-3",
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
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated condition ids to generate (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip conditions whose manifest.json already has the expected count",
    )
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
    factory = GeometryFactoryV6(args.source_foundations)
    bank = DigBank(factory)

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    rejection_totals: Counter[str] = Counter()
    unsatisfied: list[str] = []

    for condition_index, condition in enumerate(CONDITIONS):
        if selected and condition.id not in selected:
            continue
        n_maps = args.maps or condition.n_maps
        manifest_path = output / condition.id / "manifest.json"
        if args.resume and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text())
            if existing.get("mapCount") == n_maps:
                rows.extend(
                    json.loads(
                        (
                            output / "review_metadata" / f"img_{entry['sampleIndex']}.json"
                        ).read_text()
                    )
                    for entry in existing["maps"]
                )
                counts[condition.id] = n_maps
                print(f"[skip] {condition.id}: {n_maps} maps already present")
                continue
        samples, rejections, failures = generate_condition(
            condition, condition_index, bank, n_maps, args.max_attempts
        )
        rejection_totals.update(rejections)
        unsatisfied.extend(failures)
        condition_rows = write_condition(output, condition, condition_index, samples)
        rows.extend(condition_rows)
        counts[condition.id] = len(condition_rows)
        print(
            f"[{condition_index + 1:02d}/{len(CONDITIONS)}] {condition.id}: "
            f"{len(condition_rows)}/{n_maps} maps"
            + (f" UNSATISFIED={len(failures)}" if failures else "")
        )

    rows.sort(key=lambda row: row["sample_index"])
    write_manifest(output, rows)
    write_conditions_csv(output, counts)
    write_terra_metadata(output, rows)
    write_readme(output, counts)

    summary = {
        "schema": SCHEMA,
        "seed_base": SEED_BASE,
        "spec_path": "docs/CURRICULUM_TAXONOMY_SPEC.md",
        "taxonomy_version": tax.TAXONOMY_VERSION,
        "generator": "generate_prototypes_v6.py",
        "source_foundations": str(args.source_foundations),
        "condition_count": len(CONDITIONS),
        "maps_per_condition": counts,
        "accepted_maps": len(rows),
        "object_bands": {k: list(v) for k, v in OBJECT_BANDS.items()},
        "apron_capacity_bands": {k: list(v) for k, v in APRON_CAPACITY_BANDS.items()},
        "ring_capacity_bands": {k: list(v) for k, v in RING_CAPACITY_BANDS.items()},
        "ring_min_capacity": RING_MIN_CAPACITY,
        "shared_dig_attempts": SHARED_DIG_ATTEMPTS,
        "unsatisfied_constraints": unsatisfied,
        "rejections_before_acceptance": dict(rejection_totals),
        "static_gate_is_action_witness": False,
    }
    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "maps_per_condition"}, indent=2, sort_keys=True))
    if unsatisfied:
        print("UNSATISFIED CONSTRAINTS:")
        for line in unsatisfied:
            print(f"  {line}")


if __name__ == "__main__":
    main()
