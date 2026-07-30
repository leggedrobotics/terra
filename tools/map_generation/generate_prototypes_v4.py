#!/usr/bin/env python3
"""Trench-aligned dump-zone revision of the axis-classified review bank."""

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

import generate_prototypes_v3 as v3


v2 = v3.v2
base = v3.base

FOUNDATION_DUMP_WEIGHTS = dict(v2.TARGET_DUMP_WEIGHTS)
TRENCH_DUMP_WEIGHTS = {
    "broad_nearby": 0.25,
    "near_apron_large": 0.35,
    "one_side_near": 0.25,
    "separated_zones": 0.10,
    "one_side_far": 0.04,
    "haul_away_edge": 0.01,
}

TRENCH_MAX_MEDIAN_DISTANCE_TILES = {
    "broad_nearby": 5.0,
    "near_apron_large": 6.0,
    "one_side_near": 8.0,
}


class DumpFactoryV4(v2.DumpFactoryV2):
    def __init__(
        self, geometry: str, geometry_metadata: dict[str, Any]
    ) -> None:
        self.geometry = geometry
        self.geometry_metadata = geometry_metadata
        self.is_trench = geometry.startswith("trench_axes_")

    @staticmethod
    def _grow_connected(
        allowed: np.ndarray,
        seeds: np.ndarray,
        target_area: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        allowed = base.largest_component(allowed)
        seeds &= allowed
        return base.grow_region(allowed, seeds, target_area, rng)

    def broad_nearby(
        self,
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.is_trench:
            return v2.DumpFactoryV2.broad_nearby(
                dig, target_area, side, rng
            )
        distance = ndi.distance_transform_edt(~dig)
        max_distance = int(rng.integers(10, 14))
        allowed = (distance >= 3) & (distance <= max_distance)
        seeds = (distance >= 3) & (distance <= 4.25)
        target = self._grow_connected(
            allowed, seeds, target_area, rng
        )
        return target, {
            "dump_side": "trench_local_apron",
            "dump_components_requested": 1,
            "dump_alignment": "trench_bilateral_local_apron",
            "distance_bucket": "near",
            "apron_max_offset_tiles": max_distance,
        }

    def _lateral_trench_zone(
        self,
        dig: np.ndarray,
        target_area: int,
        rng: np.random.Generator,
        *,
        wide: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        heading = math.radians(
            float(self.geometry_metadata["trench_global_angle_deg"])
        )
        main_direction = np.array(
            [math.sin(heading), math.cos(heading)]
        )
        normal = np.array([main_direction[1], -main_direction[0]])
        side_sign = float(rng.choice([-1, 1]))
        offset = int(rng.integers(5, 8) if wide else rng.integers(5, 9))
        shift = np.rint(side_sign * normal * offset).astype(int)
        shifted = (
            ndi.shift(
                dig.astype(np.uint8),
                shift=(int(shift[0]), int(shift[1])),
                order=0,
                mode="constant",
                cval=0,
            )
            > 0
        )
        radius = int(rng.integers(6, 9) if wide else rng.integers(4, 7))
        distance = ndi.distance_transform_edt(~dig)
        allowed = ndi.binary_dilation(
            shifted, structure=base.binary_disk(radius)
        )
        allowed &= (distance >= 3) & (distance <= (14 if wide else 12))
        seeds = ndi.binary_dilation(
            shifted, structure=base.binary_disk(1)
        )
        seeds &= allowed
        target = self._grow_connected(
            allowed, seeds, target_area, rng
        )
        return target, {
            "dump_side": (
                "trench_lateral_left"
                if side_sign < 0
                else "trench_lateral_right"
            ),
            "dump_components_requested": 1,
            "dump_alignment": (
                "trench_lateral_wide"
                if wide
                else "trench_lateral_compact"
            ),
            "distance_bucket": "near",
            "lateral_offset_tiles": offset,
            "lateral_growth_radius_tiles": radius,
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
        return self._lateral_trench_zone(
            dig, target_area, rng, wide=True
        )

    def one_side_near(
        self,
        dig: np.ndarray,
        target_area: int,
        side: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.is_trench:
            return v2.DumpFactoryV2.one_side_near(
                dig, target_area, side, rng
            )
        return self._lateral_trench_zone(
            dig, target_area, rng, wide=False
        )


def make_sample_v4(
    geometry_factory: v3.GeometryFactoryV3,
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
            dump, dump_meta = DumpFactoryV4(
                geometry, geometry_meta
            ).make(dump_style, dig, rng)
            dump_weights = (
                TRENCH_DUMP_WEIGHTS
                if geometry.startswith("trench_axes_")
                else FOUNDATION_DUMP_WEIGHTS
            )
            dump_meta["proposed_target_weight"] = dump_weights[dump_style]
            dump_meta["distribution_conditioning"] = (
                "trench_aligned"
                if geometry.startswith("trench_axes_")
                else "foundation_default"
            )
            requested_cells = int(
                dump_meta["dump_target_cells_requested"]
            )
            if int(dump.sum()) < requested_cells:
                rejections["capacity_generation_shortfall"] += 1
                continue
            occupancy, dumpability, corridor, site_meta = (
                base.SiteFactory.make(
                    site_style, dig, dump, rng
                )
            )
        except RuntimeError as exc:
            rejections[str(exc)] += 1
            continue

        target = np.zeros((base.MAP_SIZE, base.MAP_SIZE), dtype=np.int8)
        target[dig] = -1
        target[dump & ~dig] = 1
        actual_dump_components = base.target_components(target)
        requested_components = int(
            dump_meta["dump_components_requested"]
        )
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

        distance_metrics = v2.dig_to_dump_distance_metrics(dig, dump)
        if geometry.startswith("trench_axes_"):
            max_median_distance = (
                TRENCH_MAX_MEDIAN_DISTANCE_TILES.get(
                    dump_style,
                    v2.MAX_MEDIAN_DISTANCE_TILES.get(dump_style),
                )
            )
        else:
            max_median_distance = v2.MAX_MEDIAN_DISTANCE_TILES.get(
                dump_style
            )
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
            "schema": "site_constraints_v4_trench_aligned_review",
            "seed": seed,
            "attempt": attempt,
            "attempt_seed": attempt_seed,
            "geometry": geometry,
            "geometry_hardness": v3.GEOMETRY_HARDNESS[geometry],
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
            "nondump_cells": int(
                (~dumpability & ~occupancy).sum()
            ),
            "tile_size_m": base.TILE_SIZE_M,
            "proposed_geometry_weight": v3.TARGET_GEOMETRY_WEIGHTS[
                geometry
            ],
            "proposed_site_weight": v2.TARGET_SITE_WEIGHTS[site_style],
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


def write_readme_v4(output: Path, variants: int, accepted: int) -> None:
    variant_word = "variant" if variants == 1 else "variants"
    lines = [
        "# `site_constraints_v4` trench-aligned review set",
        "",
        f"This folder contains **{accepted} maps** and {variants} {variant_word} of every exact combination.",
        "The variants are independent procedural draws of the same condition.",
        "",
        "This revision combines explicit 1/2/3-axis trench hardness with",
        "geometry-conditioned dump placement:",
        "",
        "- `broad_nearby` becomes a close bilateral apron around trenches;",
        "- `near_apron_large` becomes a wide lateral strip following one side;",
        "- `one_side_near` becomes a compact lateral strip following one side;",
        "- foundations retain their v2 nearby-dump algorithms;",
        "- far/haul remains a 5% terminal stress tail for both geometry groups.",
        "",
        "The terminal trench mixture is 85% adjacent apron/lateral dumping,",
        "10% nearby separated zones, 4% far-side, and 1% haul-away.",
        "",
        "## Start here",
        "",
        "- `REVIEW_ORDER.md`: recommended visual-review sequence and matrix semantics",
        "- `trench_hardness_catalog.png`: dig geometry only by 1/2/3 axes",
        "- `trench_dump_alignment_gallery.png`: trench and adjacent spoil zones only",
        "- `matrices/`: full dump-layout × site-constraint review",
        "- `manifest.csv`: exact axes, topology, alignment, capacity, and distance",
        "- `dataset/metadata/`: Terra-compatible 1-3-axis line metadata",
        "- `curriculum_proposal.json`: staged and geometry-conditioned weights",
        "- `VALIDATION.md`: quantitative contracts, loader check, and remaining gate",
        "- `../generate_prototypes_v3.py`: structured trench topology algorithms",
        "- `../generate_prototypes_v4.py`: geometry-conditioned dump algorithms",
        "- `../validate_review_set_v4.py`: reproducible integrity audit",
        "",
        "Axis count is a hardness proxy, not the sole difficulty measure.",
        "Dump distance, obstacle load, dig volume, and horizon remain separately",
        "stratified. Production promotion still requires dynamic action-level",
        "completion witnesses.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def render_trench_dump_alignment_gallery(output: Path) -> None:
    with (output / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    geometries = ("trench_axes_1", "trench_axes_2", "trench_axes_3")
    dump_styles = (
        "broad_nearby",
        "near_apron_large",
        "one_side_near",
    )
    variants = sorted({int(row["variant"]) for row in rows})
    columns = len(dump_styles) * len(variants)
    fig, axes = plt.subplots(
        len(geometries),
        columns,
        figsize=(3.1 * columns, 9.0),
        constrained_layout=True,
    )
    cmap = ListedColormap(["#f3e6c3", "#ef8b23", "#4daa6b"])
    index = {
        (
            row["geometry"],
            row["dump_style"],
            row["site_style"],
            int(row["variant"]),
        ): row
        for row in rows
    }
    for row_idx, geometry in enumerate(geometries):
        for dump_idx, dump_style in enumerate(dump_styles):
            for variant_idx, variant in enumerate(variants):
                col = dump_idx * len(variants) + variant_idx
                record = index[
                    (geometry, dump_style, "light", variant)
                ]
                sample_index = int(record["sample_index"])
                target = np.load(
                    output
                    / "dataset"
                    / "images"
                    / f"img_{sample_index}.npy"
                )
                display = np.zeros_like(target, dtype=np.uint8)
                display[target < 0] = 1
                display[target > 0] = 2
                ax = axes[row_idx, col]
                ax.imshow(
                    display,
                    cmap=cmap,
                    vmin=0,
                    vmax=2,
                    interpolation="nearest",
                )
                if row_idx == 0:
                    ax.set_title(
                        f"{dump_style.replace('_', ' ')}\nv{variant}",
                        fontsize=9,
                    )
                if col == 0:
                    ax.set_ylabel(
                        geometry.replace("_", " "), fontsize=10
                    )
                ax.text(
                    1,
                    62,
                    (
                        f"{record.get('dump_alignment', '')}\n"
                        f"cap {float(record['reachable_dump_to_dig_ratio']):.2f}x "
                        f"| dist {float(record['dig_dump_distance_median_tiles']):.1f}"
                    ),
                    fontsize=6.4,
                    va="bottom",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.76,
                        "edgecolor": "none",
                    },
                )
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(
        "Trench-aligned spoil zones — orange dig, green terminal dump",
        fontsize=16,
    )
    fig.savefig(output / "trench_dump_alignment_gallery.png", dpi=180)
    plt.close(fig)


def main() -> None:
    v2.GEOMETRIES = v3.GEOMETRIES
    v2.TARGET_GEOMETRY_WEIGHTS = v3.TARGET_GEOMETRY_WEIGHTS
    v3.RENDER_SCHEMA_LABEL = "site_constraints_v4"

    base.GEOMETRIES = v3.GEOMETRIES
    base.DUMP_STYLES = v2.DUMP_STYLES
    base.SITE_STYLES = v2.SITE_STYLES
    base.GeometryFactory = v3.GeometryFactoryV3
    base.DumpFactory = DumpFactoryV4
    base.make_sample = make_sample_v4
    base.render_matrix = v3.render_matrix_v3
    base.render_variability = v3.render_variability_v3
    base.write_readme = write_readme_v4

    args = base.parse_args()
    base.main()

    output = args.output.resolve()
    summary_path = output / "generation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "schema": "site_constraints_v4_trench_aligned_review",
            "target_geometry_weights": v3.TARGET_GEOMETRY_WEIGHTS,
            "target_dump_weights_foundations": FOUNDATION_DUMP_WEIGHTS,
            "target_dump_weights_trenches": TRENCH_DUMP_WEIGHTS,
            "target_site_weights": v2.TARGET_SITE_WEIGHTS,
            "capacity_ranges": v2.CAPACITY_RANGES,
            "max_median_distance_tiles": v2.MAX_MEDIAN_DISTANCE_TILES,
            "trench_max_median_distance_tiles": (
                TRENCH_MAX_MEDIAN_DISTANCE_TILES
            ),
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
    curriculum_stages = json.loads(
        json.dumps(v3.CURRICULUM_STAGES)
    )
    terminal_stage = curriculum_stages[
        "stage_2_three_axis_full_distribution"
    ]
    terminal_stage.pop("dump_weights", None)
    terminal_stage["dump_weights_foundations"] = (
        FOUNDATION_DUMP_WEIGHTS
    )
    terminal_stage["dump_weights_trenches"] = TRENCH_DUMP_WEIGHTS
    curriculum = {
        "schema": "site_constraints_v4_curriculum_proposal",
        "stages": curriculum_stages,
        "terminal_geometry_conditioned_dump_weights": {
            "foundations": FOUNDATION_DUMP_WEIGHTS,
            "trenches": TRENCH_DUMP_WEIGHTS,
        },
        "held_out_stress_evaluation": {
            "sampling": "balanced by trench axes, dump distance, and site style",
            "purpose": "measure rare hauling separately from the dominant local spoil workflow",
        },
    }
    (output / "curriculum_proposal.json").write_text(
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n"
    )
    v3.write_terra_axis_metadata(output)
    v3.render_trench_hardness_catalog(output)
    render_trench_dump_alignment_gallery(output)


if __name__ == "__main__":
    main()
