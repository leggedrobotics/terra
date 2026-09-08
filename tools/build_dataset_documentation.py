#!/usr/bin/env python3
"""Render documented Terra banks and count their stored maps without running JAX.

Inputs are read only. Outputs are figures and CSV summaries for docs/DATASET.md.
NumPy and Matplotlib are the only dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


COLORS = ["#F0EBE1", "#B86934", "#75B6AB", "#C6ADC6", "#354052"]
LABELS = [
    "Neutral / temporary staging", "Required excavation", "Accepted final dumping",
    "No dumping (unoccupied)", "Static obstacle",
]
GEOMETRIES = [
    ("v7-fnd-slab-adjacent", "Slab"),
    ("v7-fnd-irregular-adjacent", "Irregular footprint"),
    ("v7-fnd-courtyard-adjacent", "Courtyard"),
    ("v7-fnd-bearing-walls-adjacent", "Perimeter + bearing walls"),
    ("v7-fnd-pads-adjacent", "Disconnected pads"),
    ("v7-fnd-courtyard-pads-adjacent", "Courtyard + pads"),
    ("v7-trn-straight-adjacent", "Straight"),
    ("v7-trn-dogleg-adjacent", "Dog-leg"),
    ("v7-trn-tee-adjacent", "T junction"),
    ("v7-trn-cross-adjacent", "Cross junction"),
    ("v7-trn-double-t-adjacent", "Two-junction network"),
    ("v7-trn-network3-adjacent", "Three-junction network"),
    ("v7-trn-disconnected-pair-adjacent", "Disconnected pair"),
]
CONSTRAINTS = [
    ("fnd-slab-ring3x", "Surrounding dump support"),
    ("fnd-slab-side1", "One-sided dumping"),
    ("fnd-slab-split", "Split dump zones"),
    ("fnd-slab-apron-d16", "Distant dump zone"),
    ("fnd-slab-ring3x-obj1", "Sparse obstacles"),
    ("fnd-slab-ring3x-obj", "Multiple obstacles"),
    ("fnd-slab-ring3x-road", "Traversable no-dump road"),
    ("fnd-slab-side1-obj", "One-sided dumping + obstacles"),
]


def rows_at(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def layers(path: Path, slot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    name = f"img_{slot}.npy"
    return tuple(np.load(path / folder / name, allow_pickle=False)
                 for folder in ("images", "occupancy", "dumpability"))


def summarize(path: Path, row: dict) -> dict:
    target, occupancy, dumpable = layers(path, row["slot_index"])
    assert target.shape == occupancy.shape == dumpable.shape == (64, 64), path
    occupied = occupancy != 0
    return {
        **row,
        "dig_cells": int((target < 0).sum()),
        "accepted_cells": int(((target > 0) & ~occupied).sum()),
        "obstacle_cells": int(occupied.sum()),
        "no_dump_exterior_cells": int(((target >= 0) & ~occupied & ~dumpable.astype(bool)).sum()),
    }


def median_map(rows: list[dict]) -> dict:
    return sorted(rows, key=lambda r: (r["dig_cells"], r["slot_index"]))[len(rows) // 2]


def draw(axis, directory: Path, row: dict, title: str, tile: float, fontsize: float = 10) -> None:
    target, occupancy, dumpable = layers(directory, row["slot_index"])
    categories = np.zeros_like(target, dtype=np.int8)
    categories[target < 0] = 1
    categories[target > 0] = 2
    categories[(target >= 0) & ~dumpable.astype(bool)] = 3
    categories[occupancy != 0] = 4
    edge = target.shape[0] * tile
    axis.imshow(categories, cmap=ListedColormap(COLORS), vmin=0, vmax=4,
                origin="lower", interpolation="nearest", extent=(0, edge, 0, edge))
    axis.set_title(title, fontsize=fontsize, pad=6)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#CFD3D6")
        spine.set_linewidth(0.6)


def save(figure, output: Path, name: str, title: str, subtitle: str) -> None:
    figure.suptitle(title, fontsize=15, fontweight="bold", y=0.988)
    subtitle_y = 0.988 - 0.36 / figure.get_figheight()
    figure.text(0.5, subtitle_y, subtitle, ha="center", fontsize=9, color="#46515B")
    handles = [Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS)]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=9, bbox_to_anchor=(0.5, 0.007))
    figure.text(0.5, 0.66 / figure.get_figheight(),
                "Grid display: columns increase right; rows increase up.",
                ha="center", fontsize=8, color="#46515B")
    svg_path = output / f"{name}.svg"
    figure.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True, help="Full 47-condition finite-enriched bank")
    parser.add_argument("--foundation-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/dataset"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

    metadata = json.loads((args.bank / "dataset.json").read_text())
    tile = metadata["tile_size_m"]
    conditions = {}
    summary = []
    for condition in metadata["train"]:
        directory = args.bank / condition["maps_path"]
        records = [summarize(directory, row) for row in rows_at(directory / "manifest.jsonl")]
        actual_slots = {int(p.stem.split("_")[-1]) for p in (directory / "images").glob("img_*.npy")}
        assert actual_slots == {row["slot_index"] for row in records}, directory
        assert len(records) == condition["map_count"], directory
        conditions[condition["condition_id"]] = (directory, records)
        summary.append({
            "condition": condition["condition_id"], "family": condition["family"],
            "training_maps": len(records), "unique_sources_within_condition": len({r["source_id"] for r in records}),
            "unique_scenario_ids_within_condition": len({r["scenario_id"] for r in records}),
            "dig_cells_min": min(r["dig_cells"] for r in records),
            "dig_cells_mean": round(float(np.mean([r["dig_cells"] for r in records])), 3),
            "dig_cells_max": max(r["dig_cells"] for r in records),
            "accepted_cells_mean": round(float(np.mean([r["accepted_cells"] for r in records])), 3),
            "maps_with_obstacles": sum(r["obstacle_cells"] > 0 for r in records),
            "obstacle_cells_mean": round(float(np.mean([r["obstacle_cells"] for r in records])), 3),
            "maps_with_no_dump_exterior": sum(r["no_dump_exterior_cells"] > 0 for r in records),
        })
    assert len(conditions) == 47 and sum(r["training_maps"] for r in summary) == 4512
    write_csv(args.output / "condition_counts.csv", summary)
    samples = []

    def add_sample(name: str, panel: str, directory: Path, row: dict, rule: str) -> None:
        samples.append({"figure": name, "panel": panel, "dataset_directory": str(directory),
                        "condition": row["primary_cell"], "slot": row["slot_index"],
                        "map_id": row["map_id"], "source_id": row["source_id"], "selection": rule})

    fig, axes = plt.subplots(2, 7, figsize=(16, 6))
    fig.subplots_adjust(left=0.025, right=0.98, top=0.855, bottom=0.145, wspace=0.12, hspace=0.31)
    for index, (condition, title) in enumerate(GEOMETRIES):
        position = index if index < 6 else index + 1
        directory, records = conditions[condition]
        row = median_map(records)
        draw(axes.flat[position], directory, row, title, tile, 9.5)
        add_sample("terrain_geometry", title, directory, row, "upper median dig area within training condition; slot breaks ties")
    axes[0, 6].axis("off")
    axes[0, 6].text(0.08, 0.70, "V7 additions within V8\n6 foundation conditions\n7 trench conditions\n\n96 training maps each\nAdjacent dumping support\n\nAll panels: 64 x 64 cells\n36.57 x 36.57 m", fontsize=10, va="top")
    save(fig, args.output, "terrain_geometry", "Terra task geometry",
         "The 13 V7 geometry conditions within V8. Top: foundations. Bottom: trenches. Actual training maps.")

    fig, axes = plt.subplots(2, 4, figsize=(12, 7.6))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.865, bottom=0.13, wspace=0.16, hspace=0.25)
    anchor_dir, anchor_rows = conditions["fnd-slab-ring3x"]
    anchor = median_map(anchor_rows)
    anchor_dig = layers(anchor_dir, anchor["slot_index"])[0] < 0
    for axis, (condition, title) in zip(axes.flat, CONSTRAINTS):
        directory, records = conditions[condition]
        row = next(r for r in records if r["source_id"] == anchor["source_id"])
        assert np.array_equal(layers(directory, row["slot_index"])[0] < 0, anchor_dig), condition
        draw(axis, directory, row, title, tile)
        add_sample("site_constraints", title, directory, row, "shared foundation source; anchor has median training dig area")
    save(fig, args.output, "site_constraints", "Terra site constraints",
         "The excavation footprint is identical in all eight panels. Each map is 36.57 x 36.57 m.")

    fig, axes = plt.subplots(8, 6, figsize=(15, 21))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.93, bottom=0.065, wspace=0.10, hspace=0.38)
    for axis, (condition, (directory, records)) in zip(axes.flat, conditions.items()):
        row = median_map(records)
        draw(axis, directory, row, condition, tile, 8)
        add_sample("all_conditions", condition, directory, row, "upper median dig area within training condition; slot breaks ties")
    axes.flat[-1].axis("off")
    save(fig, args.output, "all_conditions", "Terra V8: all 47 training conditions",
         "96 stored training maps per condition; 4,512 total. Capability controls are included here. All panels use the same scale.")

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.8))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.81, bottom=0.20, wspace=0.16)
    directory = args.foundation_bank / "train/all"
    rows = [summarize(directory, row) for row in rows_at(directory / "manifest.jsonl")]
    for axis, shape, title in zip(axes, ("square", "rectangle", "l"), ("Square", "Rectangle", "L shape")):
        row = median_map([r for r in rows if r["geometry"]["shape"] == shape])
        draw(axis, directory, row, title, tile, 11)
        add_sample("foundation_suite", title, directory, row, "upper median dig area within training shape; slot breaks ties")
    save(fig, args.output, "foundation_suite", "Foundation efficiency suite",
         "Separate 256 / 64 / 64 train / validation / test bank; no obstacles, broad final dumping. Same 64 x 64 grid.")
    foundation_counts = []
    for split in ("train", "validation", "test"):
        rows = rows_at(args.foundation_bank / split / "all/manifest.jsonl")
        for shape in ("square", "rectangle", "l"):
            foundation_counts.append({"split": split, "shape": shape,
                                      "maps": sum(r["geometry"]["shape"] == shape for r in rows)})
    assert sum(r["maps"] for r in foundation_counts) == 384
    write_csv(args.output / "foundation_counts.csv", foundation_counts)
    write_csv(args.output / "figure_samples.csv", samples)
    print(f"Rendered 4 SVG/PNG figure pairs; counted 4,512 V8 training maps and 384 foundation maps in {args.output}")


if __name__ == "__main__":
    main()
