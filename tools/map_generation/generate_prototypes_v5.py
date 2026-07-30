#!/usr/bin/env python3
"""Add current-training-like easy dumping to the structured review bank.

The new ``easy_surround`` stratum is geometry-conditioned:

* foundations: every legal free cell around the excavation is a dump target;
* trenches: two deliberately populated, high-capacity sides of the trench.

The existing ``near_apron_large`` trench stratum is tightened into an explicit
high-capacity one-side case. More constrained v4 rows remain available later in
the curriculum.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import ndimage as ndi

import generate_prototypes_v4 as v4


v3 = v4.v3
v2 = v4.v2
base = v4.base

EASY_STYLE = "easy_surround"
DUMP_STYLES = (
    "broad_nearby",
    "near_apron_large",
    "one_side_near",
    "separated_zones",
    "one_side_far",
    "haul_away_edge",
    EASY_STYLE,
)

FOUNDATION_DUMP_WEIGHTS = {
    EASY_STYLE: 0.25,
    "broad_nearby": 0.18,
    "near_apron_large": 0.22,
    "one_side_near": 0.15,
    "separated_zones": 0.15,
    "one_side_far": 0.04,
    "haul_away_edge": 0.01,
}
TRENCH_DUMP_WEIGHTS = {
    EASY_STYLE: 0.25,
    "broad_nearby": 0.10,
    "near_apron_large": 0.30,
    "one_side_near": 0.20,
    "separated_zones": 0.10,
    "one_side_far": 0.04,
    "haul_away_edge": 0.01,
}

FOUNDATION_EASY_MIN_CAPACITY = 3.0
TRENCH_BOTH_SIDES_CAPACITY_RANGE = (2.50, 3.25)
TRENCH_ONE_SIDE_LARGE_CAPACITY_RANGE = (2.00, 2.60)
TRENCH_MAX_MEDIAN_DISTANCE_TILES = {
    EASY_STYLE: 5.0,
    "broad_nearby": 5.0,
    "near_apron_large": 6.0,
    "one_side_near": 8.0,
}

CURRICULUM_STAGES = {
    "stage_0_easy_dump_access": {
        "advance_gate": "fixed all-around/both-side/one-side replay success >= 95%",
        "geometry_weights": {
            "foundation_osm": 0.55,
            "foundation_procedural": 0.15,
            "foundation_structural": 0.00,
            "trench_axes_1": 0.30,
            "trench_axes_2": 0.00,
            "trench_axes_3": 0.00,
        },
        "dump_weights_foundations": {
            EASY_STYLE: 0.65,
            "broad_nearby": 0.20,
            "near_apron_large": 0.15,
            "one_side_near": 0.00,
            "separated_zones": 0.00,
            "one_side_far": 0.00,
            "haul_away_edge": 0.00,
        },
        "dump_weights_trenches": {
            EASY_STYLE: 0.45,
            "broad_nearby": 0.10,
            "near_apron_large": 0.45,
            "one_side_near": 0.00,
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
    "stage_1_constraints_and_two_axes": {
        "advance_gate": "fixed easy and mid replay success >= 90%",
        "geometry_weights": {
            "foundation_osm": 0.35,
            "foundation_procedural": 0.15,
            "foundation_structural": 0.10,
            "trench_axes_1": 0.25,
            "trench_axes_2": 0.15,
            "trench_axes_3": 0.00,
        },
        "dump_weights_foundations": {
            EASY_STYLE: 0.35,
            "broad_nearby": 0.20,
            "near_apron_large": 0.20,
            "one_side_near": 0.15,
            "separated_zones": 0.10,
            "one_side_far": 0.00,
            "haul_away_edge": 0.00,
        },
        "dump_weights_trenches": {
            EASY_STYLE: 0.30,
            "broad_nearby": 0.10,
            "near_apron_large": 0.35,
            "one_side_near": 0.15,
            "separated_zones": 0.10,
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
        "geometry_weights": v3.TARGET_GEOMETRY_WEIGHTS,
        "dump_weights_foundations": FOUNDATION_DUMP_WEIGHTS,
        "dump_weights_trenches": TRENCH_DUMP_WEIGHTS,
        "site_weights": v2.TARGET_SITE_WEIGHTS,
    },
}


class DumpFactoryV5(v4.DumpFactoryV4):
    """Explicit large both-side and one-side trench dump generators."""

    def _main_axis_projection(
        self, dig: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        heading = math.radians(
            float(self.geometry_metadata["trench_global_angle_deg"])
        )
        normal = np.array([math.cos(heading), -math.sin(heading)])
        center = np.mean(np.argwhere(dig), axis=0)
        yy, xx = np.indices(dig.shape)
        projection = (
            (yy - center[0]) * normal[0]
            + (xx - center[1]) * normal[1]
        )
        return projection, normal

    def easy_surround(
        self,
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del side
        if not self.is_trench:
            raise RuntimeError(
                "foundation_easy_surround_requires_post_site_generation"
            )
        distance = ndi.distance_transform_edt(~dig)
        projection, _ = self._main_axis_projection(dig)
        part_areas = (target_area // 2, target_area - target_area // 2)
        parts: list[np.ndarray] = []
        for sign, part_area in zip((-1.0, 1.0), part_areas):
            allowed = (
                (distance >= 3)
                & (distance <= int(rng.integers(16, 21)))
                & (sign * projection >= 1.0)
            )
            seeds = allowed & (distance <= 5.25)
            part = self._grow_connected(
                allowed, seeds, int(part_area), rng
            )
            parts.append(part)
        target = parts[0] | parts[1]
        side_cells = [int(part.sum()) for part in parts]
        balance = min(side_cells) / max(1, sum(side_cells))
        return target, {
            "dump_side": "trench_both_sides",
            "dump_access_sides": "both",
            "dump_alignment": "trench_both_sides_large",
            "dump_components_requested": base.target_components(
                np.where(target, 1, 0)
            ),
            "distance_bucket": "immediate_near",
            "both_side_negative_cells": side_cells[0],
            "both_side_positive_cells": side_cells[1],
            "both_side_balance": round(float(balance), 4),
        }

    def near_apron_large(
        self,
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.is_trench:
            return v2.DumpFactoryV2.near_apron_large(
                dig, target_area, side, rng
            )
        projection, _ = self._main_axis_projection(dig)
        side_sign = float(rng.choice([-1, 1]))
        distance = ndi.distance_transform_edt(~dig)
        allowed = (
            (distance >= 3)
            & (distance <= int(rng.integers(17, 22)))
            & (side_sign * projection >= 1.0)
        )
        seeds = allowed & (distance <= 5.25)
        target = self._grow_connected(
            allowed, seeds, target_area, rng
        )
        return target, {
            "dump_side": (
                "trench_main_axis_left"
                if side_sign < 0
                else "trench_main_axis_right"
            ),
            "dump_access_sides": "one",
            "dump_alignment": "trench_one_side_large",
            "dump_components_requested": 1,
            "distance_bucket": "immediate_near",
            "one_side_sign": int(side_sign),
        }

    def make(
        self,
        style: str,
        dig: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.is_trench and style in (EASY_STYLE, "near_apron_large"):
            capacity_range = (
                TRENCH_BOTH_SIDES_CAPACITY_RANGE
                if style == EASY_STYLE
                else TRENCH_ONE_SIDE_LARGE_CAPACITY_RANGE
            )
            capacity_factor = float(rng.uniform(*capacity_range))
            target_area = int(
                math.ceil(int(dig.sum()) * capacity_factor)
            )
            side = int(rng.integers(0, 4))
            target, metadata = getattr(self, style)(
                dig, target_area, side, rng
            )
            metadata.update(
                {
                    "dump_target_cells_requested": target_area,
                    "capacity_factor_required": round(
                        capacity_factor, 5
                    ),
                    "difficulty_tier": (
                        "starter_both_sides"
                        if style == EASY_STYLE
                        else "starter_one_side"
                    ),
                    "proposed_target_weight": TRENCH_DUMP_WEIGHTS[
                        style
                    ],
                }
            )
            return target, metadata
        return super().make(style, dig, rng)


def make_foundation_easy_sample(
    geometry_factory: v3.GeometryFactoryV3,
    geometry: str,
    site_style: str,
    seed: int,
    max_attempts: int,
) -> tuple[base.Sample | None, Counter[str]]:
    """Generate a foundation whose entire legal free area is a dump target."""
    rejections: Counter[str] = Counter()
    for attempt in range(max_attempts):
        attempt_seed = int(
            np.random.SeedSequence([seed, attempt]).generate_state(1)[0]
        )
        rng = np.random.default_rng(attempt_seed)
        try:
            dig, geometry_meta = geometry_factory.make(geometry, rng)
            planning_area = int(math.ceil(1.25 * int(dig.sum())))
            planning_dump, _ = v2.DumpFactoryV2.broad_nearby(
                dig,
                planning_area,
                int(rng.integers(0, 4)),
                rng,
            )
            if int(planning_dump.sum()) < planning_area:
                rejections["planning_dump_generation_shortfall"] += 1
                continue
            occupancy, dumpability, corridor, site_meta = (
                base.SiteFactory.make(
                    site_style, dig, planning_dump, rng
                )
            )
        except RuntimeError as exc:
            rejections[str(exc)] += 1
            continue

        dump = (~dig) & (~occupancy) & dumpability
        target = np.zeros((base.MAP_SIZE, base.MAP_SIZE), dtype=np.int8)
        target[dig] = -1
        target[dump] = 1
        gate = base.static_gate(target, occupancy, dumpability)
        if not gate.accepted:
            rejections[gate.reason] += 1
            continue

        capacity_ratio = float(dump.sum() / max(1, dig.sum()))
        reachable_capacity_ratio = float(
            gate.reachable_dump_cells_post / max(1, dig.sum())
        )
        if reachable_capacity_ratio + 1e-8 < FOUNDATION_EASY_MIN_CAPACITY:
            rejections["reachable_capacity_contract"] += 1
            continue

        legal_free = (~dig) & (~occupancy) & dumpability
        legal_coverage = float(
            (dump & legal_free).sum() / max(1, legal_free.sum())
        )
        if not math.isclose(
            legal_coverage, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            rejections["all_around_coverage_contract"] += 1
            continue

        distance_metrics = v2.dig_to_dump_distance_metrics(dig, dump)
        distance = base.compute_geodesic_distance(target, occupancy)
        action = np.zeros_like(target, dtype=np.int8)
        _, dig_components = ndi.label(
            dig, structure=np.ones((3, 3), dtype=np.uint8)
        )
        dump_components = base.target_components(target)
        metadata: dict[str, Any] = {
            "schema": "site_constraints_v5_easy_access_review",
            "seed": seed,
            "attempt": attempt,
            "attempt_seed": attempt_seed,
            "geometry": geometry,
            "geometry_hardness": v3.GEOMETRY_HARDNESS[geometry],
            "dump_style": EASY_STYLE,
            "site_style": site_style,
            "dig_cells": int(dig.sum()),
            "dig_components_actual": int(dig_components),
            "dump_cells": int(dump.sum()),
            "dump_to_dig_area_ratio": round(capacity_ratio, 4),
            "reachable_dump_to_dig_ratio": round(
                reachable_capacity_ratio, 4
            ),
            "dump_components_actual": dump_components,
            "dump_components_requested": dump_components,
            "dump_target_cells_requested": int(dump.sum()),
            "capacity_factor_required": FOUNDATION_EASY_MIN_CAPACITY,
            "difficulty_tier": "starter_all_around",
            "dump_side": "foundation_all_around",
            "dump_access_sides": "all",
            "dump_alignment": "all_legal_free_around_foundation",
            "distance_bucket": "immediate",
            "distribution_conditioning": "foundation_all_around",
            "dump_coverage_of_legal_free": round(
                legal_coverage, 6
            ),
            "active_training_analogue": (
                "100_percent_legal_free_dump_coverage"
            ),
            "target_obstacle_policy": "exclude_obstacles_from_target",
            "planning_dump_cells": int(planning_dump.sum()),
            "obstacle_cells": int(occupancy.sum()),
            "nondump_cells": int(
                (~dumpability & ~occupancy).sum()
            ),
            "tile_size_m": base.TILE_SIZE_M,
            "proposed_geometry_weight": v3.TARGET_GEOMETRY_WEIGHTS[
                geometry
            ],
            "proposed_site_weight": v2.TARGET_SITE_WEIGHTS[
                site_style
            ],
            "proposed_target_weight": FOUNDATION_DUMP_WEIGHTS[
                EASY_STYLE
            ],
            "static_gate_is_action_witness": False,
            **distance_metrics,
            **geometry_meta,
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


def make_sample_v5(
    geometry_factory: v3.GeometryFactoryV3,
    geometry: str,
    dump_style: str,
    site_style: str,
    seed: int,
    max_attempts: int,
) -> tuple[base.Sample | None, Counter[str]]:
    if dump_style == EASY_STYLE and geometry.startswith("foundation_"):
        return make_foundation_easy_sample(
            geometry_factory,
            geometry,
            site_style,
            seed,
            max_attempts,
        )

    sample, rejections = v4.make_sample_v4(
        geometry_factory,
        geometry,
        dump_style,
        site_style,
        seed,
        max_attempts,
    )
    if sample is None:
        return None, rejections
    sample.metadata["schema"] = "site_constraints_v5_easy_access_review"
    sample.metadata["proposed_target_weight"] = (
        TRENCH_DUMP_WEIGHTS[dump_style]
        if geometry.startswith("trench_axes_")
        else FOUNDATION_DUMP_WEIGHTS[dump_style]
    )
    if geometry.startswith("trench_axes_"):
        sample.metadata["distribution_conditioning"] = (
            "trench_explicit_side_access"
        )
    legal_free = (
        (sample.target >= 0)
        & (~sample.occupancy)
        & sample.dumpability
    )
    sample.metadata["dump_coverage_of_legal_free"] = round(
        float(
            ((sample.target > 0) & legal_free).sum()
            / max(1, legal_free.sum())
        ),
        6,
    )
    return sample, rejections


def write_terra_metadata(output: Path) -> None:
    destination = output / "dataset" / "metadata"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(
        (output / "review_metadata").glob("img_*.json"),
        key=lambda path: int(path.stem.split("_")[1]),
    ):
        record = json.loads(source.read_text())
        sample_index = int(record["sample_index"])
        payload = {
            "schema": "site_constraints_v5_axis_metadata",
            "geometry": record["geometry"],
            "trench_axes_count": int(
                record.get("trench_axes_count", -1)
            ),
            "trench_topology": record.get("trench_topology", ""),
            "axes_ABC": record.get("axes_ABC", []),
        }
        (destination / f"trench_{sample_index}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


def write_readme_v5(output: Path, variants: int, accepted: int) -> None:
    lines = [
        "# `site_constraints_v5` easy-access review set",
        "",
        f"This folder contains **{accepted} maps** with {variants} independent variants",
        "of every geometry × dump-access × site-constraint condition.",
        "",
        "The new easy anchor deliberately resembles current Terra training:",
        "",
        "- foundations can dump on 100% of legal free ground around the dig;",
        "- trench `easy_surround` maps have large targets on both sides;",
        "- trench `near_apron_large` maps have one large permitted side;",
        "- obstacles and non-dump roads are excluded from the target instead of",
        "  being hidden underneath an everywhere-green target;",
        "- compact, separated, far, and haul layouts remain later difficulty.",
        "",
        "## Start here",
        "",
        "- `REVIEW_ORDER.md`: recommended visual-review sequence",
        "- `foundation_all_around_gallery.png`: easy foundations across site styles",
        "- `trench_side_options_gallery.png`: explicit both-side versus one-side cases",
        "- `trench_hardness_catalog.png`: 1/2/3-axis geometry progression",
        "- `matrices/`: the complete seven-row review grid",
        "- `curriculum_proposal.json`: geometry-conditioned staged weights",
        "- `manifest.csv`: coverage, capacity, side access, and topology metadata",
        "- `VALIDATION.md`: saved-array contracts and remaining promotion gate",
        "- `../generate_prototypes_v5.py`: procedural algorithms and weights",
        "- `../validate_review_set_v5.py`: reproducible strict validator",
        "",
        "The static gate is not an action-level completion witness. Promotion still",
        "requires successful dynamic-soil replay within the selected horizon.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def _manifest_index(output: Path) -> dict[tuple[str, str, str, int], dict]:
    with (output / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return {
        (
            row["geometry"],
            row["dump_style"],
            row["site_style"],
            int(row["variant"]),
        ): row
        for row in rows
    }


def _display_code(
    target: np.ndarray, occupancy: np.ndarray, dumpability: np.ndarray
) -> np.ndarray:
    code = np.zeros_like(target, dtype=np.uint8)
    code[target < 0] = 1
    code[target > 0] = 2
    code[~dumpability & ~occupancy] = 3
    code[occupancy] = 4
    return code


def render_foundation_all_around_gallery(output: Path) -> None:
    index = _manifest_index(output)
    geometries = (
        "foundation_osm",
        "foundation_procedural",
        "foundation_structural",
    )
    sites = tuple(v2.SITE_STYLES)
    fig, axes = plt.subplots(
        len(geometries),
        len(sites),
        figsize=(15.5, 9.2),
        constrained_layout=True,
    )
    cmap = ListedColormap(
        ["#f3e6c3", "#ef8b23", "#4daa6b", "#808080", "#111111"]
    )
    for row_index, geometry in enumerate(geometries):
        for column, site in enumerate(sites):
            record = index[(geometry, EASY_STYLE, site, 0)]
            sample_index = int(record["sample_index"])
            target = np.load(
                output / "dataset" / "images" / f"img_{sample_index}.npy"
            )
            occupancy = np.load(
                output
                / "dataset"
                / "occupancy"
                / f"img_{sample_index}.npy"
            )
            dumpability = np.load(
                output
                / "dataset"
                / "dumpability"
                / f"img_{sample_index}.npy"
            )
            ax = axes[row_index, column]
            ax.imshow(
                _display_code(target, occupancy, dumpability),
                cmap=cmap,
                vmin=0,
                vmax=4,
                interpolation="nearest",
            )
            if row_index == 0:
                ax.set_title(site.replace("_", " "), fontsize=10)
            if column == 0:
                ax.set_ylabel(geometry.replace("_", " "), fontsize=10)
            ax.text(
                1,
                62,
                (
                    f"legal coverage {float(record['dump_coverage_of_legal_free']):.0%}\n"
                    f"reachable cap {float(record['reachable_dump_to_dig_ratio']):.1f}x"
                ),
                fontsize=7,
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.78,
                    "edgecolor": "none",
                },
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Easy foundations — every legal free cell can receive soil",
        fontsize=16,
    )
    fig.savefig(output / "foundation_all_around_gallery.png", dpi=180)
    plt.close(fig)


def render_trench_side_options_gallery(output: Path) -> None:
    index = _manifest_index(output)
    geometries = ("trench_axes_1", "trench_axes_2", "trench_axes_3")
    styles = (EASY_STYLE, "near_apron_large")
    variants = tuple(sorted({key[3] for key in index}))
    fig, axes = plt.subplots(
        len(geometries),
        len(styles) * len(variants),
        figsize=(13.0, 9.0),
        constrained_layout=True,
    )
    cmap = ListedColormap(
        ["#f3e6c3", "#ef8b23", "#4daa6b", "#808080", "#111111"]
    )
    for row_index, geometry in enumerate(geometries):
        for style_index, style in enumerate(styles):
            for variant_index, variant in enumerate(variants):
                column = style_index * len(variants) + variant_index
                record = index[(geometry, style, "light", variant)]
                sample_index = int(record["sample_index"])
                target = np.load(
                    output
                    / "dataset"
                    / "images"
                    / f"img_{sample_index}.npy"
                )
                occupancy = np.load(
                    output
                    / "dataset"
                    / "occupancy"
                    / f"img_{sample_index}.npy"
                )
                dumpability = np.load(
                    output
                    / "dataset"
                    / "dumpability"
                    / f"img_{sample_index}.npy"
                )
                ax = axes[row_index, column]
                ax.imshow(
                    _display_code(target, occupancy, dumpability),
                    cmap=cmap,
                    vmin=0,
                    vmax=4,
                    interpolation="nearest",
                )
                if row_index == 0:
                    access = (
                        "both sides"
                        if style == EASY_STYLE
                        else "one side"
                    )
                    ax.set_title(f"{access} · v{variant}", fontsize=10)
                if column == 0:
                    ax.set_ylabel(
                        geometry.replace("_", " "), fontsize=10
                    )
                ax.text(
                    1,
                    62,
                    (
                        f"{record['dump_access_sides']} allowed\n"
                        f"cap {float(record['reachable_dump_to_dig_ratio']):.2f}x"
                    ),
                    fontsize=7,
                    va="bottom",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.78,
                        "edgecolor": "none",
                    },
                )
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(
        "Easy trench access — large both-side and one-side dump zones",
        fontsize=16,
    )
    fig.savefig(output / "trench_side_options_gallery.png", dpi=180)
    plt.close(fig)


def main() -> None:
    v2.GEOMETRIES = v3.GEOMETRIES
    v2.DUMP_STYLES = DUMP_STYLES
    v2.TARGET_GEOMETRY_WEIGHTS = v3.TARGET_GEOMETRY_WEIGHTS
    v3.RENDER_SCHEMA_LABEL = "site_constraints_v5"

    v4.FOUNDATION_DUMP_WEIGHTS = FOUNDATION_DUMP_WEIGHTS
    v4.TRENCH_DUMP_WEIGHTS = TRENCH_DUMP_WEIGHTS
    v4.TRENCH_MAX_MEDIAN_DISTANCE_TILES = (
        TRENCH_MAX_MEDIAN_DISTANCE_TILES
    )
    v4.DumpFactoryV4 = DumpFactoryV5

    base.GEOMETRIES = v3.GEOMETRIES
    base.DUMP_STYLES = DUMP_STYLES
    base.SITE_STYLES = v2.SITE_STYLES
    base.GeometryFactory = v3.GeometryFactoryV3
    base.DumpFactory = DumpFactoryV5
    base.make_sample = make_sample_v5
    base.render_matrix = v3.render_matrix_v3
    base.render_variability = v3.render_variability_v3
    base.write_readme = write_readme_v5

    args = base.parse_args()
    base.main()

    output = args.output.resolve()
    summary_path = output / "generation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "schema": "site_constraints_v5_easy_access_review",
            "target_geometry_weights": v3.TARGET_GEOMETRY_WEIGHTS,
            "target_dump_weights_foundations": (
                FOUNDATION_DUMP_WEIGHTS
            ),
            "target_dump_weights_trenches": TRENCH_DUMP_WEIGHTS,
            "target_site_weights": v2.TARGET_SITE_WEIGHTS,
            "foundation_easy_contract": {
                "dump_coverage_of_legal_free": 1.0,
                "minimum_reachable_capacity": (
                    FOUNDATION_EASY_MIN_CAPACITY
                ),
                "active_training_analogue": (
                    "100_percent_legal_free_dump_coverage"
                ),
            },
            "trench_easy_contract": {
                "both_sides_style": EASY_STYLE,
                "both_sides_capacity_range": (
                    TRENCH_BOTH_SIDES_CAPACITY_RANGE
                ),
                "one_side_style": "near_apron_large",
                "one_side_capacity_range": (
                    TRENCH_ONE_SIDE_LARGE_CAPACITY_RANGE
                ),
                "max_median_distance_tiles": {
                    EASY_STYLE: (
                        TRENCH_MAX_MEDIAN_DISTANCE_TILES[EASY_STYLE]
                    ),
                    "near_apron_large": (
                        TRENCH_MAX_MEDIAN_DISTANCE_TILES[
                            "near_apron_large"
                        ]
                    ),
                },
            },
            "trench_hardness_contract": {
                "trench_axes_1": {"axes": 1, "junctions": 0},
                "trench_axes_2": {"axes": 2, "junctions": 1},
                "trench_axes_3": {"axes": 3, "junctions": 2},
            },
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    curriculum = {
        "schema": "site_constraints_v5_curriculum_proposal",
        "stages": CURRICULUM_STAGES,
        "terminal_geometry_conditioned_dump_weights": {
            "foundations": FOUNDATION_DUMP_WEIGHTS,
            "trenches": TRENCH_DUMP_WEIGHTS,
        },
        "held_out_stress_evaluation": {
            "sampling": (
                "balanced by geometry, side access, dump distance, and site"
            ),
            "purpose": (
                "separate remote hauling from the dominant easy/local workflow"
            ),
        },
    }
    (output / "curriculum_proposal.json").write_text(
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n"
    )
    write_terra_metadata(output)
    v3.render_trench_hardness_catalog(output)
    v4.render_trench_dump_alignment_gallery(output)
    render_foundation_all_around_gallery(output)
    render_trench_side_options_gallery(output)


if __name__ == "__main__":
    main()
