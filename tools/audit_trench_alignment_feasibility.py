#!/usr/bin/env python3
"""Audit only local physical feasibility of the strict trench dig rule.

A0 validates generator-owned axis bits. A1 asks whether every trench cell is
reachable by some obstacle-clear, yaw/standoff-valid pose. A2 applies Terra's
atomic DO semantics: all still-fresh trench cells in the selected cone must be
compatible with that pose. Ordering, navigation, dumping, and episode-horizon
success are policy questions and are intentionally outside this audit.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.env import TerraEnvBatch
from terra.maps_buffer import _trench_records_from_metadata
from terra.maps_buffer import trench_axis_contract_sanity_check
from terra.maps_buffer import trench_axis_owners_sanity_check
from terra.state import State
from terra.utils import compute_polygon_mask


MAP_SIZE = 64
MAX_AXES = 4


def env_config() -> EnvConfig:
    batch_env = object.__new__(TerraEnvBatch)
    batch_env.batch_cfg = BatchConfig()._replace(
        maps_dims=MapsDimsConfig(maps_edge_length=MAP_SIZE)
    )
    base = EnvConfig()
    updated = batch_env.update_env_cfgs(
        base._replace(
            agent=base.agent._replace(dig_depth=jnp.ones((1,), dtype=jnp.int32))
        )
    )
    return base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(edge_length_px=MAP_SIZE),
        agent_types=(0,),
        action_types=(0,),
        enforce_trench_dig_alignment=True,
    )


def terra_geometry(cfg: EnvConfig) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
    """Materialize Terra's exact cone and footprint masks once."""

    shape = (MAP_SIZE, MAP_SIZE)
    state = State.new(
        jax.random.PRNGKey(0),
        cfg,
        np.zeros(shape, dtype=np.int8),
        np.zeros(shape, dtype=np.int8),
        -97.0 * np.ones((MAX_AXES, 3), dtype=np.float32),
        np.int32(-1),
        np.zeros(shape, dtype=np.uint8),
        -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1),
        np.ones(shape, dtype=np.bool_),
        np.zeros(shape, dtype=np.int8),
        distance_map_override=np.ones(shape, dtype=np.float32),
    )
    center = np.asarray([MAP_SIZE // 2, MAP_SIZE // 2], dtype=np.int16)
    cones: list[list[np.ndarray]] = []
    footprints: list[np.ndarray] = []
    for base_heading in range(cfg.agent.angles_base):
        by_cabin = []
        for cabin_heading in range(cfg.agent.angles_cabin):
            current = state._get_current_agent_state()._replace(
                pos_base=jnp.asarray(center),
                angle_base=jnp.asarray([base_heading], dtype=jnp.int8),
                angle_cabin=jnp.asarray([cabin_heading], dtype=jnp.int8),
            )
            posed = state._set_current_agent_state(current)
            cone = np.argwhere(
                np.asarray(posed._build_dig_dump_cone()).reshape(shape)
            )
            by_cabin.append(cone.astype(np.int16) - center)
        cones.append(by_cabin)

        current = state._get_current_agent_state()._replace(
            pos_base=jnp.asarray(center),
            angle_base=jnp.asarray([base_heading], dtype=jnp.int8),
        )
        corners = state._get_agent_corners(
            current.pos_base,
            current.angle_base,
            cfg.agent.width,
            cfg.agent.height,
        )
        footprint = np.argwhere(
            np.asarray(compute_polygon_mask(corners, MAP_SIZE, MAP_SIZE))
        )
        footprints.append(footprint.astype(np.int16) - center)
    return cones, footprints


def translated(position: tuple[int, int], offsets: np.ndarray) -> np.ndarray:
    return offsets + np.asarray(position, dtype=np.int16)


def clear_positions(occupancy: np.ndarray, footprint: np.ndarray) -> list[tuple[int, int]]:
    positions = []
    for row in range(MAP_SIZE):
        for column in range(MAP_SIZE):
            cells = translated((row, column), footprint)
            inside = (
                (cells[:, 0] >= 0)
                & (cells[:, 0] < MAP_SIZE)
                & (cells[:, 1] >= 0)
                & (cells[:, 1] < MAP_SIZE)
            )
            if np.all(inside) and not np.any(occupancy[cells[:, 0], cells[:, 1]]):
                positions.append((row, column))
    return positions


def pose_axis_bits(
    row: int,
    column: int,
    heading: int,
    axes: np.ndarray,
    cfg: EnvConfig,
) -> int:
    theta = 2.0 * math.pi * heading / cfg.agent.angles_base
    forward = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    tangents = np.stack([-axes[:, 0], axes[:, 1]], axis=1)
    norms = np.linalg.norm(tangents, axis=1)
    parallel = np.clip(np.abs(tangents @ forward) / norms, 0.0, 1.0)
    standoff = (
        np.abs(axes[:, 0] * column + axes[:, 1] * row + axes[:, 2])
        / np.linalg.norm(axes[:, :2], axis=1)
        * cfg.tile_size
    )
    valid = (
        (parallel + 1e-6 >= math.cos(cfg.trench_dig_yaw_tolerance_rad))
        & (standoff >= cfg.trench_dig_standoff_min_m)
        & (standoff <= cfg.trench_dig_standoff_max_m)
    )
    bits = 0
    for axis_index in np.flatnonzero(valid):
        bits |= 1 << int(axis_index)
    return bits


def analyze_case(case: dict, cfg: EnvConfig, cones, footprints) -> dict:
    target = np.load(case["target"], allow_pickle=False)
    occupancy = np.load(case["occupancy"], allow_pickle=False).astype(np.bool_)
    owners = np.load(case["owners"], allow_pickle=False)
    metadata = json.loads(Path(case["metadata"]).read_text())
    if target.shape != (MAP_SIZE, MAP_SIZE) or occupancy.shape != target.shape:
        raise RuntimeError(f"{case['label']}: only 64 x 64 maps are supported.")
    records, axis_count = _trench_records_from_metadata(metadata, MAX_AXES)
    if axis_count <= 0:
        raise RuntimeError(f"{case['label']}: manifest says trench but has no axes.")
    owners = trench_axis_owners_sanity_check(
        target, owners, axis_count, MAX_AXES
    )
    trench_axis_contract_sanity_check(metadata, owners, axis_count)
    axes = np.asarray(records[:axis_count], dtype=np.float64)

    target_mask = target < 0
    a1 = np.zeros_like(target_mask)
    a2 = np.zeros_like(target_mask)
    pose_count = 0
    action_count = 0
    for heading, footprint in enumerate(footprints):
        for position in clear_positions(occupancy, footprint):
            bits = pose_axis_bits(*position, heading, axes, cfg)
            if bits == 0:
                continue
            pose_count += 1
            for cone_offsets in cones[heading]:
                cells = translated(position, cone_offsets)
                inside = (
                    (cells[:, 0] >= 0)
                    & (cells[:, 0] < MAP_SIZE)
                    & (cells[:, 1] >= 0)
                    & (cells[:, 1] < MAP_SIZE)
                )
                cells = cells[inside]
                if np.any(occupancy[cells[:, 0], cells[:, 1]]):
                    continue
                selected = target_mask[cells[:, 0], cells[:, 1]]
                fresh = cells[selected]
                if fresh.size == 0:
                    continue
                compatible = (
                    owners[fresh[:, 0], fresh[:, 1]] & np.uint8(bits)
                ) != 0
                valid_cells = fresh[compatible]
                a1[valid_cells[:, 0], valid_cells[:, 1]] = True
                if np.all(compatible):
                    action_count += 1
                    a2[fresh[:, 0], fresh[:, 1]] = True

    target_cells = int(target_mask.sum())
    a1_cells = int((a1 & target_mask).sum())
    a2_cells = int((a2 & target_mask).sum())
    return {
        "label": case["label"],
        "dataset": case["dataset"],
        "map_id": case["map_id"],
        "condition": case["condition"],
        "axes": axis_count,
        "target_cells": target_cells,
        "a0_exact_ownership": True,
        "a1_pose_cover_cells": a1_cells,
        "a1_pose_cover_fraction": a1_cells / max(target_cells, 1),
        "a1_pass": a1_cells == target_cells,
        "a2_atomic_cover_cells": a2_cells,
        "a2_atomic_cover_fraction": a2_cells / max(target_cells, 1),
        "a2_pass": a2_cells == target_cells,
        "axis_valid_pose_count": pose_count,
        "admissible_atomic_action_count": action_count,
        "designated_dump_cells": int((target > 0).sum()),
    }


def exact_datasets(root: Path) -> list[Path]:
    if (root / "dataset.json").is_file() and (root / "manifest.jsonl").is_file():
        return [root]
    return sorted(
        path.parent
        for path in root.rglob("dataset.json")
        if (path.parent / "manifest.jsonl").is_file()
    )


def cases(root: Path, requested: list[str] | None) -> list[dict]:
    datasets = (
        [(root / relative).resolve() for relative in requested]
        if requested
        else exact_datasets(root)
    )
    result = []
    for dataset in datasets:
        relative = str(dataset.relative_to(root)) if dataset != root else "."
        for line in (dataset / "manifest.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("family") != "trench":
                continue
            slot = int(row["slot_index"])
            result.append(
                {
                    "label": f"{relative}:{slot}:{row['map_id']}",
                    "dataset": relative,
                    "map_id": row["map_id"],
                    "condition": row.get("primary_cell", "unknown"),
                    "target": dataset / "images" / f"img_{slot}.npy",
                    "occupancy": dataset / "occupancy" / f"img_{slot}.npy",
                    "owners": dataset / "trench_axis_owners" / f"img_{slot}.npy",
                    "metadata": dataset / "metadata" / f"trench_{slot}.json",
                }
            )
    return result


def aggregate(results: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        groups[result[key]].append(result)
    summary = []
    for name, rows in sorted(groups.items()):
        target_cells = sum(row["target_cells"] for row in rows)
        summary.append(
            {
                key: name,
                "maps": len(rows),
                "target_cells": target_cells,
                "a1_pass_maps": sum(row["a1_pass"] for row in rows),
                "a1_pose_cover_fraction": sum(
                    row["a1_pose_cover_cells"] for row in rows
                )
                / max(target_cells, 1),
                "a2_pass_maps": sum(row["a2_pass"] for row in rows),
                "a2_atomic_cover_fraction": sum(
                    row["a2_atomic_cover_cells"] for row in rows
                )
                / max(target_cells, 1),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.bank.resolve()
    selected = cases(root, args.dataset)
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError(f"No exact-layout trench maps found under {root}.")
    cfg = env_config()
    cones, footprints = terra_geometry(cfg)
    results = []
    for index, case in enumerate(selected, start=1):
        result = analyze_case(case, cfg, cones, footprints)
        results.append(result)
        print(
            f"[{index}/{len(selected)}] {result['label']} "
            f"A1={result['a1_pose_cover_cells']}/{result['target_cells']} "
            f"A2={result['a2_atomic_cover_cells']}/{result['target_cells']}",
            flush=True,
        )

    report = {
        "contract": {
            "bank": str(root),
            "maps": len(results),
            "owner_representation": "uint8 generator-owned axis bits",
            "axis_representation": "A*x + B*y + C = 0",
            "yaw_tolerance_deg": math.degrees(cfg.trench_dig_yaw_tolerance_rad),
            "standoff_m": [
                cfg.trench_dig_standoff_min_m,
                cfg.trench_dig_standoff_max_m,
            ],
            "headings": cfg.agent.angles_base,
            "a0": "every trench target cell has only declared owner bits",
            "a1": "per-cell cover by an obstacle-clear yaw/standoff-valid pose",
            "a2": "per-cell cover by a complete Terra atomic DO action",
            "out_of_scope": [
                "navigation and re-approach ordering",
                "spoil placement, rehandling, and dump capacity",
                "episode horizon and learned-policy success",
            ],
        },
        "passed": all(row["a1_pass"] and row["a2_pass"] for row in results),
        "summary_by_dataset": aggregate(results, "dataset"),
        "summary_by_condition": aggregate(results, "condition"),
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
