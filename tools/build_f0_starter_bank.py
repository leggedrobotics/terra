#!/usr/bin/env python3
"""Build the two obstacle-free exact-mask identities used by the F0 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from heapq import heappop, heappush
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from terra.maps_buffer import contained_dump_capacity_sanity_check
from terra.maps_buffer import validate_exact_dataset_contract

MAP_SIZE = 64
TILE_SIZE_M = 0.5


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shortest_paths(sources: np.ndarray) -> np.ndarray:
    distance = np.full(sources.shape, np.inf, dtype=np.float64)
    queue = []
    for y, x in np.argwhere(sources):
        distance[y, x] = 0.0
        heappush(queue, (0.0, int(y), int(x)))
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    while queue:
        current, y, x = heappop(queue)
        if current != distance[y, x]:
            continue
        for dy, dx, cost in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < MAP_SIZE and 0 <= nx < MAP_SIZE:
                proposed = current + cost
                if proposed < distance[ny, nx]:
                    distance[ny, nx] = proposed
                    heappush(queue, (proposed, ny, nx))
    return distance


def boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            interior &= padded[
                dy : dy + mask.shape[0],
                dx : dx + mask.shape[1],
            ]
    return mask & ~interior


def starter_target(family: str) -> np.ndarray:
    target = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    if family == "foundation":
        target[28:36, 28:36] = -1
        target[target == 0] = 1
    elif family == "trench":
        target[30:33, 21:43] = -1
        target[20:29, 15:49] = 1
        target[34:43, 15:49] = 1
    else:
        raise ValueError(family)
    return target


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def render_gallery(output: Path, targets: dict[str, np.ndarray]) -> None:
    colors = np.array(
        [
            [0.82, 0.47, 0.24],
            [0.94, 0.94, 0.90],
            [0.31, 0.63, 0.34],
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, family in zip(axes, ("foundation", "trench")):
        axis.imshow(colors[targets[family] + 1])
        axis.set_title(
            "foundation: all-around"
            if family == "foundation"
            else "trench: broad both-side"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("F0 exact-mask starter identities")
    figure.tight_layout()
    figure.savefig(output / "gallery.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    identities = [
        {
            "map_id": "f0-foundation-all-around-000",
            "source_id": "procedural:f0-foundation-seed-2026072501",
            "split": "train",
            "family": "foundation",
            "stratum": "F0",
            "primary_cell": "all_around_low_volume",
        },
        {
            "map_id": "f0-trench-both-side-000",
            "source_id": "procedural:f0-trench-seed-2026072502",
            "split": "train",
            "family": "trench",
            "stratum": "F0",
            "primary_cell": "straight_both_side_low_volume",
        },
    ]
    registry = output / "source_registry.jsonl"
    write_jsonl(registry, identities)
    registry_sha256 = sha256_file(registry)

    targets = {}
    receipts = []
    for identity in identities:
        family = identity["family"]
        directory = output / family
        for subdirectory in (
            "images",
            "occupancy",
            "dumpability",
            "actions",
            "distance",
            "metadata",
        ):
            (directory / subdirectory).mkdir(parents=True)

        target = starter_target(family)
        occupancy = np.zeros_like(target, dtype=np.int8)
        dumpability = np.ones_like(target, dtype=np.bool_)
        action = np.zeros_like(target, dtype=np.int8)
        raw_distance = shortest_paths(target > 0)
        distance = (raw_distance / raw_distance.max()).astype(np.float32)
        targets[family] = target
        layers = {
            "images": target,
            "occupancy": occupancy,
            "dumpability": dumpability,
            "actions": action,
            "distance": distance,
        }
        for subdirectory, array in layers.items():
            np.save(directory / subdirectory / "img_1.npy", array)

        metadata = {
            **identity,
            "axes_ABC": (
                [{"A": 1.0, "B": 0.0, "C": -31.0}] if family == "trench" else []
            ),
            "foundation_border_axes_ABC": [],
        }
        write_json(directory / "metadata" / "trench_1.json", metadata)
        manifest_row = {
            **identity,
            "slot_index": 1,
            "slot_weight": 1.0,
            "identity_slot_multiplicity": 1,
        }
        write_jsonl(directory / "manifest.jsonl", [manifest_row])
        write_json(
            directory / "dataset.json",
            {
                "schema": "terra_exact_map_dataset_v1",
                "slot_count": 1,
                "unique_identity_count": 1,
                "shape": [MAP_SIZE, MAP_SIZE],
                "distance_metric": ("8_connected_cardinal_1_diagonal_sqrt2"),
                "distance_normalization": "per_map_max_to_1",
                "accepted_dump_contract": "exact_visible_dump_v1",
                "scenario_identity_contract": "terra_legacy_map_id_v0",
                "minimum_dump_capacity_ratio": 3.0,
                "source_registry": "../source_registry.jsonl",
                "source_registry_sha256": registry_sha256,
            },
        )

        capacity = contained_dump_capacity_sanity_check(
            target,
            occupancy,
            dumpability,
            action,
            minimum_single_layer_ratio=3.0,
        )
        work_distances = shortest_paths(target > 0)[boundary(target < 0)]
        validate_exact_dataset_contract(directory, 1)
        receipt = {
            **identity,
            "accepted_dump_definition": "(target > 0) & ~occupancy",
            "obstacle_cells": int(occupancy.sum()),
            "dig_cells": int((target < 0).sum()),
            "dump_cells": int((target > 0).sum()),
            "capacity": capacity,
            "dig_to_dump_path": {
                "metric": "8_connected_cardinal_1_diagonal_sqrt2",
                "p50_tiles": float(np.median(work_distances)),
                "p95_tiles": float(np.quantile(work_distances, 0.95)),
                "max_tiles": float(work_distances.max()),
                "p95_metres": float(np.quantile(work_distances, 0.95) * TILE_SIZE_M),
            },
            "layer_sha256": {
                subdirectory: sha256_file(directory / subdirectory / "img_1.npy")
                for subdirectory in layers
            },
            "dataset_sha256": sha256_file(directory / "dataset.json"),
            "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
        }
        receipts.append(receipt)
        write_json(directory / "validation.json", receipt)

    render_gallery(output, targets)
    write_json(
        output / "validation.json",
        {
            "schema": "terra_f0_starter_bank_v1",
            "status": "passed",
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "source_registry_sha256": registry_sha256,
            "identities": receipts,
        },
    )
    print(json.dumps(receipts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
