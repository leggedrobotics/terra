#!/usr/bin/env python3
"""Axis-classified trench revision of the Terra map review bank.

The v2 trench families mixed branch counts and topologies inside broad labels.
This revision makes trench hardness legible and curriculum-addressable:

* one axis: a single straight trench;
* two axes: one controlled T or X junction;
* three axes: two controlled junctions in construction-like networks.

Dump layouts, site constraints, capacity gates, and nearby-distance gates are
reused from v2.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import generate_prototypes_v2 as v2


base = v2.base

RENDER_SCHEMA_LABEL = "site_constraints_v3"

GEOMETRIES = (
    "foundation_osm",
    "foundation_procedural",
    "foundation_structural",
    "trench_axes_1",
    "trench_axes_2",
    "trench_axes_3",
)

TARGET_GEOMETRY_WEIGHTS = {
    "foundation_osm": 0.30,
    "foundation_procedural": 0.10,
    "foundation_structural": 0.10,
    "trench_axes_1": 0.25,
    "trench_axes_2": 0.15,
    "trench_axes_3": 0.10,
}

GEOMETRY_HARDNESS = {
    "foundation_osm": "foundation_real",
    "foundation_procedural": "foundation_connected",
    "foundation_structural": "foundation_disconnected",
    "trench_axes_1": "trench_easy",
    "trench_axes_2": "trench_medium",
    "trench_axes_3": "trench_hard",
}

CURRICULUM_STAGES = {
    "stage_0_easy_nearby": {
        "advance_gate": "fixed easy replay success >= 95%",
        "geometry_weights": {
            "foundation_osm": 0.55,
            "foundation_procedural": 0.15,
            "foundation_structural": 0.00,
            "trench_axes_1": 0.30,
            "trench_axes_2": 0.00,
            "trench_axes_3": 0.00,
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
    "stage_1_two_axis_and_constraints": {
        "advance_gate": "fixed one/two-axis and near/mid replay success >= 90%",
        "geometry_weights": {
            "foundation_osm": 0.35,
            "foundation_procedural": 0.15,
            "foundation_structural": 0.10,
            "trench_axes_1": 0.25,
            "trench_axes_2": 0.15,
            "trench_axes_3": 0.00,
        },
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
    "stage_2_three_axis_full_distribution": {
        "advance_gate": "terminal training mixture",
        "geometry_weights": TARGET_GEOMETRY_WEIGHTS,
        "dump_weights": v2.TARGET_DUMP_WEIGHTS,
        "site_weights": v2.TARGET_SITE_WEIGHTS,
    },
}


def discrete_heading(rng: np.random.Generator) -> float:
    """Global rotation changes while relative trench geometry stays structured."""
    return float(rng.choice(np.deg2rad(np.arange(0, 180, 15))))


def line_coefficients(
    point_a_yx: np.ndarray, point_b_yx: np.ndarray
) -> dict[str, float]:
    """Return A, B, C for A*x + B*y + C = 0."""
    y1, x1 = map(float, point_a_yx)
    y2, x2 = map(float, point_b_yx)
    return {
        "A": float(y2 - y1),
        "B": float(x1 - x2),
        "C": float(x2 * y1 - x1 * y2),
    }


def points_inside(points: np.ndarray, margin: int = 8) -> bool:
    return bool(
        np.all(points >= margin)
        and np.all(points <= base.MAP_SIZE - margin - 1)
    )


class GeometryFactoryV3(v2.GeometryFactoryV2):
    @staticmethod
    def trench_axes_1(
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """One straight construction axis with no junction."""
        for _ in range(180):
            center = np.array(
                [float(rng.uniform(27, 37)), float(rng.uniform(27, 37))]
            )
            heading = discrete_heading(rng)
            direction = np.array([math.sin(heading), math.cos(heading)])
            length = float(rng.uniform(24, 38))
            points = np.vstack(
                [
                    center - direction * length / 2,
                    center + direction * length / 2,
                ]
            )
            if not points_inside(points):
                continue
            radius = int(rng.choice([1, 2], p=[0.65, 0.35]))
            dig = base.rasterize_polyline(points, radius=radius)
            if 60 <= int(dig.sum()) <= 210:
                return dig, {
                    "geometry_hardness": GEOMETRY_HARDNESS["trench_axes_1"],
                    "trench_axes_count": 1,
                    "intersection_junctions": 0,
                    "intersection_branches": 2,
                    "trench_topology": "straight",
                    "trench_width_radius_tiles": radius,
                    "trench_global_angle_deg": round(
                        math.degrees(heading), 1
                    ),
                    "axes_ABC": [
                        line_coefficients(points[0], points[1])
                    ],
                }
        raise RuntimeError("Could not construct one-axis trench")

    @staticmethod
    def trench_axes_2(
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Two axes with one controlled T or X junction."""
        for _ in range(220):
            center = np.array(
                [float(rng.uniform(28, 36)), float(rng.uniform(28, 36))]
            )
            heading = discrete_heading(rng)
            direction = np.array([math.sin(heading), math.cos(heading)])
            main_length = float(rng.uniform(27, 39))
            main_points = np.vstack(
                [
                    center - direction * main_length / 2,
                    center + direction * main_length / 2,
                ]
            )
            junction = center + direction * float(
                rng.uniform(-0.12, 0.12) * main_length
            )
            relative_angle = math.pi / 2
            branch_heading = heading + relative_angle
            branch_direction = np.array(
                [math.sin(branch_heading), math.cos(branch_heading)]
            )
            topology = str(rng.choice(["T", "T", "X"]))
            branch_a = float(rng.uniform(10, 17))
            if topology == "X":
                branch_b = float(rng.uniform(10, 17))
                branch_points = np.vstack(
                    [
                        junction - branch_direction * branch_a,
                        junction,
                        junction + branch_direction * branch_b,
                    ]
                )
            else:
                side = float(rng.choice([-1, 1]))
                branch_points = np.vstack(
                    [junction, junction + side * branch_direction * branch_a]
                )
            all_points = np.vstack([main_points, branch_points])
            if not points_inside(all_points):
                continue
            radius = int(rng.choice([1, 2], p=[0.65, 0.35]))
            dig = base.rasterize_polyline(main_points, radius=radius)
            dig |= base.rasterize_polyline(branch_points, radius=radius)
            dig = base.largest_component(dig)
            if 90 <= int(dig.sum()) <= 300:
                return dig, {
                    "geometry_hardness": GEOMETRY_HARDNESS["trench_axes_2"],
                    "trench_axes_count": 2,
                    "intersection_junctions": 1,
                    "intersection_branches": 4 if topology == "X" else 3,
                    "trench_topology": topology,
                    "trench_relative_angle_deg": round(
                        math.degrees(relative_angle), 1
                    ),
                    "trench_width_radius_tiles": radius,
                    "trench_global_angle_deg": round(
                        math.degrees(heading), 1
                    ),
                    "axes_ABC": [
                        line_coefficients(main_points[0], main_points[1]),
                        line_coefficients(
                            branch_points[0], branch_points[-1]
                        ),
                    ],
                }
        raise RuntimeError("Could not construct two-axis trench")

    @staticmethod
    def trench_axes_3(
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Three axes with two separated construction-like junctions."""
        for _ in range(260):
            center = np.array(
                [float(rng.uniform(28, 36)), float(rng.uniform(28, 36))]
            )
            heading = discrete_heading(rng)
            direction = np.array([math.sin(heading), math.cos(heading)])
            normal = np.array([direction[1], -direction[0]])
            main_length = float(rng.uniform(29, 39))
            main_points = np.vstack(
                [
                    center - direction * main_length / 2,
                    center + direction * main_length / 2,
                ]
            )
            positions = np.array([-0.24, 0.24]) + rng.uniform(
                -0.025, 0.025, size=2
            )
            topology = str(
                rng.choice(["double_T", "double_T", "H", "T_plus_X"])
            )
            branch_points_list: list[np.ndarray] = []
            axis_records = [
                line_coefficients(main_points[0], main_points[1])
            ]
            for branch_index, along in enumerate(positions):
                junction = center + direction * float(along * main_length)
                branch_length_a = float(rng.uniform(9, 15))
                double_sided = topology == "H" or (
                    topology == "T_plus_X" and branch_index == 1
                )
                if double_sided:
                    branch_length_b = float(rng.uniform(9, 15))
                    branch_points = np.vstack(
                        [
                            junction - normal * branch_length_a,
                            junction,
                            junction + normal * branch_length_b,
                        ]
                    )
                else:
                    if topology == "double_T":
                        side = -1.0 if branch_index == 0 else 1.0
                    else:
                        side = float(rng.choice([-1, 1]))
                    branch_points = np.vstack(
                        [junction, junction + side * normal * branch_length_a]
                    )
                branch_points_list.append(branch_points)
                axis_records.append(
                    line_coefficients(
                        branch_points[0], branch_points[-1]
                    )
                )
            all_points = np.vstack([main_points, *branch_points_list])
            if not points_inside(all_points):
                continue
            radius = int(rng.choice([1, 2], p=[0.65, 0.35]))
            dig = base.rasterize_polyline(main_points, radius=radius)
            for branch_points in branch_points_list:
                dig |= base.rasterize_polyline(
                    branch_points, radius=radius
                )
            dig = base.largest_component(dig)
            if 125 <= int(dig.sum()) <= 360:
                double_sided_count = sum(
                    len(points) == 3 for points in branch_points_list
                )
                return dig, {
                    "geometry_hardness": GEOMETRY_HARDNESS["trench_axes_3"],
                    "trench_axes_count": 3,
                    "intersection_junctions": 2,
                    "intersection_double_sided": double_sided_count,
                    "intersection_branches": 4 + double_sided_count,
                    "trench_topology": topology,
                    "trench_relative_angle_deg": 90.0,
                    "trench_width_radius_tiles": radius,
                    "trench_global_angle_deg": round(
                        math.degrees(heading), 1
                    ),
                    "axes_ABC": axis_records,
                }
        raise RuntimeError("Could not construct three-axis trench")


def render_matrix_v3(
    samples: dict[tuple[str, str, str, int], base.Sample],
    geometry: str,
    variant: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(v2.DUMP_STYLES),
        len(v2.SITE_STYLES),
        figsize=(16, 19),
        constrained_layout=True,
    )
    for row, dump_style in enumerate(v2.DUMP_STYLES):
        for col, site_style in enumerate(v2.SITE_STYLES):
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
        f"{RENDER_SCHEMA_LABEL} — {geometry} — variant {variant}",
        fontsize=16,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_variability_v3(
    samples: dict[tuple[str, str, str, int], base.Sample],
    dump_style: str,
    variants: int,
    path: Path,
) -> None:
    columns = len(v2.SITE_STYLES) * variants
    fig, axes = plt.subplots(
        len(GEOMETRIES),
        columns,
        figsize=(3.0 * columns, 17),
        constrained_layout=True,
    )
    for row, geometry in enumerate(GEOMETRIES):
        for variant in range(variants):
            for site_idx, site_style in enumerate(v2.SITE_STYLES):
                col = variant * len(v2.SITE_STYLES) + site_idx
                ax = axes[row, col]
                sample = samples[
                    (geometry, dump_style, site_style, variant)
                ]
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
                    ax.set_ylabel(
                        geometry.replace("_", "\n"), fontsize=9
                    )
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(
        f"{RENDER_SCHEMA_LABEL} axis-classified variability — "
        f"{dump_style.replace('_', ' ')}",
        fontsize=16,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_readme_v3(output: Path, variants: int, accepted: int) -> None:
    variant_word = "variant" if variants == 1 else "variants"
    lines = [
        "# `site_constraints_v3` axis-classified review set",
        "",
        f"This folder contains **{accepted} maps** and {variants} {variant_word} of every exact combination.",
        "",
        "The trench distribution is now classified by an explicit hardness proxy:",
        "",
        "- `trench_axes_1`: one straight axis, zero junctions (easy);",
        "- `trench_axes_2`: one controlled T/X junction (medium);",
        "- `trench_axes_3`: two controlled H/double-T/T+X junctions (hard).",
        "",
        "Global rotation and dimensions vary, but relative topology is selected",
        "from a small construction-like catalogue rather than unconstrained random branches.",
        "",
        "## Proposed final geometry mixture",
        "",
    ]
    for name, weight in TARGET_GEOMETRY_WEIGHTS.items():
        lines.append(f"- `{name}`: {weight:.0%}")
    lines.extend(
        [
            "",
            "This is 50% foundations and 50% trenches. Within trenches, the",
            "conditional mixture is 50% one-axis, 30% two-axis, and 20% three-axis.",
            "",
            "## Inspection",
            "",
            "- `trench_hardness_catalog.png`: dig geometry only, ordered by axis count",
            "- `matrices/`: dump layouts by rows, site constraints by columns",
            "- `variability/`: both variants grouped by dump algorithm",
            "- `previews/`: one labeled image per map",
            "- `manifest.csv`: exact axis count, topology, capacity, distance, and gate metrics",
            "- `dataset/metadata/`: Terra-compatible 1-3-axis line metadata",
            "- `curriculum_proposal.json`: staged geometry/dump/site weights",
            "",
            "Colours: orange dig, green terminal dump, grey traversable non-dump road, black obstacle.",
            "",
            "## Important interpretation",
            "",
            "Axis count is a useful hardness stratum, not a complete difficulty",
            "measure. Dig volume, dump distance, obstacles, and episode horizon still",
            "need separate stratified reporting.",
            "",
            "All maps pass the v2 reachable single-layer capacity, nearby-distance,",
            "and footprint-aware static gates. Production promotion still requires a",
            "dynamic Terra action-level completion witness.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))


def write_terra_axis_metadata(output: Path) -> None:
    """Materialize loader-compatible metadata for every review map."""
    destination = output / "dataset" / "metadata"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(
        (output / "review_metadata").glob("img_*.json"),
        key=lambda path: int(path.stem.split("_")[1]),
    ):
        record = json.loads(source.read_text())
        sample_index = int(record["sample_index"])
        payload = {
            "schema": "site_constraints_v3_axis_metadata",
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


def render_trench_hardness_catalog(output: Path) -> None:
    """Render only dig masks so the three topology strata are easy to compare."""
    with (output / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    trench_geometries = (
        "trench_axes_1",
        "trench_axes_2",
        "trench_axes_3",
    )
    labels = (
        "1 axis | easy | 0 junctions",
        "2 axes | medium | 1 junction",
        "3 axes | hard | 2 junctions",
    )
    available_per_geometry = [
        sum(
            row["geometry"] == geometry and row["site_style"] == "light"
            for row in rows
        )
        for geometry in trench_geometries
    ]
    columns = min(10, *available_per_geometry)
    if columns <= 0:
        raise RuntimeError("No light-site trench samples available for catalog")
    fig, axes = plt.subplots(
        len(trench_geometries),
        columns,
        figsize=(20, 6.7),
        constrained_layout=True,
    )
    cmap = ListedColormap(["#f3e6c3", "#ef8b23"])
    for row_index, (geometry, label) in enumerate(
        zip(trench_geometries, labels)
    ):
        candidates = [
            row
            for row in rows
            if row["geometry"] == geometry and row["site_style"] == "light"
        ][:columns]
        if len(candidates) != columns:
            raise RuntimeError(
                f"Expected {columns} catalog samples for {geometry}"
            )
        for column_index, record in enumerate(candidates):
            sample_index = int(record["sample_index"])
            target = np.load(
                output
                / "dataset"
                / "images"
                / f"img_{sample_index}.npy"
            )
            dig_only = (target < 0).astype(np.uint8)
            ax = axes[row_index, column_index]
            ax.imshow(
                dig_only,
                cmap=cmap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(f"sample {column_index + 1}", fontsize=9)
            if column_index == 0:
                ax.set_ylabel(label, fontsize=10)
            topology = record.get("trench_topology", "")
            ax.text(
                1,
                62,
                topology,
                fontsize=6.7,
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                },
            )
    fig.suptitle(
        "Trench hardness catalogue — controlled topology by axis count",
        fontsize=16,
    )
    fig.savefig(output / "trench_hardness_catalog.png", dpi=180)
    plt.close(fig)


def main() -> None:
    # v2 functions resolve these module globals at runtime.
    v2.GEOMETRIES = GEOMETRIES
    v2.TARGET_GEOMETRY_WEIGHTS = TARGET_GEOMETRY_WEIGHTS
    v2.CURRICULUM_STAGES = CURRICULUM_STAGES

    base.GEOMETRIES = GEOMETRIES
    base.DUMP_STYLES = v2.DUMP_STYLES
    base.SITE_STYLES = v2.SITE_STYLES
    base.GeometryFactory = GeometryFactoryV3
    base.DumpFactory = v2.DumpFactoryV2
    base.make_sample = v2.make_sample_v2
    base.render_matrix = render_matrix_v3
    base.render_variability = render_variability_v3
    base.write_readme = write_readme_v3

    args = base.parse_args()
    base.main()

    output = args.output.resolve()
    summary_path = output / "generation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "schema": "site_constraints_v3_axis_classified_review",
            "target_geometry_weights": TARGET_GEOMETRY_WEIGHTS,
            "target_dump_weights": v2.TARGET_DUMP_WEIGHTS,
            "target_site_weights": v2.TARGET_SITE_WEIGHTS,
            "capacity_ranges": v2.CAPACITY_RANGES,
            "max_median_distance_tiles": v2.MAX_MEDIAN_DISTANCE_TILES,
            "trench_hardness_contract": {
                "trench_axes_1": {
                    "axes": 1,
                    "junctions": 0,
                    "tier": "easy",
                },
                "trench_axes_2": {
                    "axes": 2,
                    "junctions": 1,
                    "tier": "medium",
                },
                "trench_axes_3": {
                    "axes": 3,
                    "junctions": 2,
                    "tier": "hard",
                },
            },
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output / "curriculum_proposal.json").write_text(
        json.dumps(
            {
                "schema": "site_constraints_v3_curriculum_proposal",
                "stages": CURRICULUM_STAGES,
                "held_out_stress_evaluation": {
                    "sampling": "balanced by trench axis count, dump style, and site style",
                    "purpose": "separate topology hardness from rare dump-distance and obstacle tails",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_terra_axis_metadata(output)
    render_trench_hardness_catalog(output)


if __name__ == "__main__":
    main()
