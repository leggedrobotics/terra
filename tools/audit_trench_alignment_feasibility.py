#!/usr/bin/env python3
"""Strict finite-section feasibility witness for Terra trench alignment.

The probe uses Terra's JAX cone, footprint, movement, finite-section
membership, and dynamic-dumpability implementations.  A fresh DO is
all-or-nothing: every still-fresh target cell in its cone must belong to at
least one section for which the base pose satisfies yaw and standoff.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig, EnvConfig, MapsDimsConfig
from terra.env import TerraEnvBatch
from terra.map import compute_dynamic_dumpability, compute_trench_axis_membership
from terra.state import State
from terra.utils import compute_polygon_mask


SHAPE = (64, 64)
N_HEADINGS = 12
YAW_TOLERANCE_DEG = 15.001
STANDOFF_MIN_M = 3.5
STANDOFF_MAX_M = 7.0
MAX_AXES = 4


def env_config() -> EnvConfig:
    batch_env = object.__new__(TerraEnvBatch)
    batch_env.batch_cfg = BatchConfig()._replace(
        maps_dims=MapsDimsConfig(maps_edge_length=SHAPE[0])
    )
    base = EnvConfig()
    batched = base._replace(
        agent=base.agent._replace(dig_depth=jnp.ones((1,), dtype=jnp.int32))
    )
    updated = batch_env.update_env_cfgs(batched)
    return base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(edge_length_px=SHAPE[0]),
        agent_types=(0,),
        action_types=(0,),
    )


def reference_state(cfg: EnvConfig) -> State:
    state = State.new(
        jax.random.PRNGKey(0),
        cfg,
        np.zeros(SHAPE, dtype=np.int8),
        np.zeros(SHAPE, dtype=np.int8),
        -97.0 * np.ones((MAX_AXES, 8), dtype=np.float32),
        np.int32(-1),
        -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1),
        np.ones(SHAPE, dtype=np.bool_),
        np.zeros(SHAPE, dtype=np.int8),
        distance_map_override=np.ones(SHAPE, dtype=np.float32),
    )
    cur = state._get_current_agent_state()._replace(
        pos_base=jnp.array([32, 32], dtype=jnp.int16),
        angle_base=jnp.array([0], dtype=jnp.int8),
        angle_cabin=jnp.array([0], dtype=jnp.int8),
        loaded=jnp.array([0], dtype=jnp.int8),
    )
    return state._set_current_agent_state(cur)


def terra_geometry(
    cfg: EnvConfig,
) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[tuple[int, int]], list[tuple[int, int]]]:
    """Return exact 12x12 cones, footprints, and forward/backward deltas."""
    state = reference_state(cfg)
    center = np.array([32, 32], dtype=np.int16)
    cones: list[list[np.ndarray]] = []
    footprints: list[np.ndarray] = []
    forward_deltas: list[tuple[int, int]] = []
    backward_deltas: list[tuple[int, int]] = []
    for base_heading in range(N_HEADINGS):
        base_cones: list[np.ndarray] = []
        for cabin_heading in range(N_HEADINGS):
            cur = state._get_current_agent_state()._replace(
                pos_base=jnp.array(center, dtype=jnp.int16),
                angle_base=jnp.array([base_heading], dtype=jnp.int8),
                angle_cabin=jnp.array([cabin_heading], dtype=jnp.int8),
                loaded=jnp.array([0], dtype=jnp.int8),
            )
            posed = state._set_current_agent_state(cur)
            cone = np.argwhere(
                np.asarray(posed._build_dig_dump_cone()).reshape(SHAPE)
            )
            base_cones.append(cone.astype(np.int16) - center)
        cones.append(base_cones)

        cur = state._get_current_agent_state()._replace(
            pos_base=jnp.array(center, dtype=jnp.int16),
            angle_base=jnp.array([base_heading], dtype=jnp.int8),
            angle_cabin=jnp.array([0], dtype=jnp.int8),
            loaded=jnp.array([0], dtype=jnp.int8),
        )
        posed = state._set_current_agent_state(cur)
        corners = posed._get_agent_corners(
            cur.pos_base,
            base_orientation=cur.angle_base,
            agent_width=cfg.agent.width,
            agent_height=cfg.agent.height,
        )
        footprint = np.argwhere(
            np.asarray(compute_polygon_mask(corners, SHAPE[1], SHAPE[0]))
        )
        footprints.append(footprint.astype(np.int16) - center)

        moved_forward = posed._handle_move_forward()._get_current_agent_state().pos_base
        moved_backward = posed._handle_move_backward()._get_current_agent_state().pos_base
        fd = np.asarray(moved_forward, dtype=np.int16) - center
        bd = np.asarray(moved_backward, dtype=np.int16) - center
        forward_deltas.append((int(fd[0]), int(fd[1])))
        backward_deltas.append((int(bd[0]), int(bd[1])))
    return cones, footprints, forward_deltas, backward_deltas


def translated(position: tuple[int, int], offsets: np.ndarray) -> np.ndarray:
    return offsets + np.asarray(position, dtype=np.int16)


def footprint_clear(
    position: tuple[int, int], offsets: np.ndarray, blocked: np.ndarray
) -> bool:
    cells = translated(position, offsets)
    inside = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < SHAPE[0])
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < SHAPE[1])
    )
    return bool(np.all(inside) and not np.any(blocked[cells[:, 0], cells[:, 1]]))


def valid_pose_graph(
    blocked: np.ndarray,
    footprints: list[np.ndarray],
    forward_deltas: list[tuple[int, int]],
    backward_deltas: list[tuple[int, int]],
) -> tuple[set[tuple[int, int, int]], dict[tuple[int, int, int], int]]:
    valid: set[tuple[int, int, int]] = set()
    for heading in range(N_HEADINGS):
        for row in range(SHAPE[0]):
            for col in range(SHAPE[1]):
                if footprint_clear((row, col), footprints[heading], blocked):
                    valid.add((row, col, heading))

    components: dict[tuple[int, int, int], int] = {}
    component_id = 0
    for root in valid:
        if root in components:
            continue
        components[root] = component_id
        queue = deque([root])
        while queue:
            row, col, heading = queue.popleft()
            neighbors = [
                (row, col, (heading - 1) % N_HEADINGS),
                (row, col, (heading + 1) % N_HEADINGS),
            ]
            for dr, dc in (forward_deltas[heading], backward_deltas[heading]):
                neighbors.append((row + dr, col + dc, heading))
            for neighbor in neighbors:
                if neighbor in valid and neighbor not in components:
                    components[neighbor] = component_id
                    queue.append(neighbor)
        component_id += 1
    return valid, components


def records_and_membership(
    target: np.ndarray, metadata: dict
) -> tuple[np.ndarray, np.ndarray, int]:
    axes = metadata["axes_ABC"]
    # Generator review records publish ``trench_arms``; enriched runtime
    # sidecars publish the same finite sections as ``trench_segments_yx``.
    arms = metadata.get("trench_segments_yx")
    if arms is None:
        arms = metadata["trench_arms"]
    half_width = float(metadata["trench_half_width_tiles"])
    if not len(axes) == len(arms) <= MAX_AXES:
        raise RuntimeError(
            f"axis/arm mismatch: axes={len(axes)}, arms={len(arms)}"
        )
    records = []
    for axis, arm in zip(axes, arms):
        start = arm[0]
        end = arm[-1]
        records.append(
            [
                float(axis["A"]),
                float(axis["B"]),
                float(axis["C"]),
                float(start[0]),
                float(start[1]),
                float(end[0]),
                float(end[1]),
                half_width,
            ]
        )
    while len(records) < MAX_AXES:
        records.append([-97.0] * 8)
    # Match MapsBuffer.new: generator metadata records are stored as float16 and
    # converted back to float32 inside GridWorld membership/alignment routines.
    runtime_records = np.asarray(records, dtype=np.float16).astype(np.float32)
    membership = np.asarray(
        compute_trench_axis_membership(
            jnp.asarray(target),
            jnp.asarray(runtime_records),
            jnp.int32(len(axes)),
        )
    )
    return runtime_records, membership, len(axes)


def base_axis_bits(
    valid_poses: set[tuple[int, int, int]],
    records: np.ndarray,
    axis_count: int,
    tile_size: float,
) -> dict[tuple[int, int, int], int]:
    result: dict[tuple[int, int, int], int] = {}
    axes = records[:axis_count, :3].astype(np.float32)
    denominators = np.maximum(np.linalg.norm(axes[:, :2], axis=1), np.float32(1e-6))
    for pose in valid_poses:
        row, col, heading = pose
        theta = np.float32(2.0 * np.pi) * np.float32(heading) / np.float32(N_HEADINGS)
        forward = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float32)
        tangents = np.stack([-axes[:, 0], axes[:, 1]], axis=1)
        cosine = np.clip(
            np.abs(tangents @ forward)
            / np.maximum(np.linalg.norm(tangents, axis=1), np.float32(1e-6)),
            np.float32(0.0),
            np.float32(1.0),
        )
        yaw = np.degrees(np.arccos(cosine))
        standoff = (
            np.abs(axes[:, 0] * np.float32(col) + axes[:, 1] * np.float32(row) + axes[:, 2])
            / denominators
            * np.float32(tile_size)
        )
        valid = (
            (yaw <= np.float32(YAW_TOLERANCE_DEG))
            & (standoff >= np.float32(STANDOFF_MIN_M))
            & (standoff <= np.float32(STANDOFF_MAX_M))
        )
        bits = 0
        for axis_index in np.flatnonzero(valid):
            bits |= 1 << int(axis_index)
        result[pose] = bits
    return result


def backward_chain_qualified(
    pose: tuple[int, int, int],
    axis_bits: dict[tuple[int, int, int], int],
    backward_delta: tuple[int, int],
) -> bool:
    row, col, heading = pose
    current = axis_bits.get(pose, 0)
    if not current:
        return False
    dr, dc = backward_delta
    # Require the selected dig station itself to have a legal next BACKWARD
    # move that stays yaw/standoff-valid for at least one shared section.
    neighbor = (row + dr, col + dc, heading)
    return bool(current & axis_bits.get(neighbor, 0))


def dump_reachable_bases(
    *,
    target: np.ndarray,
    padding: np.ndarray,
    dumpability_init: np.ndarray,
    valid_poses: set[tuple[int, int, int]],
    cones: list[list[np.ndarray]],
) -> set[tuple[int, int, int]]:
    complete_action = np.zeros(SHAPE, dtype=np.int8)
    complete_action[target < 0] = -1
    dynamic = np.asarray(
        compute_dynamic_dumpability(
            jnp.asarray(dumpability_init, dtype=jnp.bool_),
            jnp.asarray(complete_action),
            kernel_size=5,
        )
    )
    accepted = (target > 0) & (~padding.astype(bool)) & dynamic
    reachable: set[tuple[int, int, int]] = set()
    for pose in valid_poses:
        row, col, heading = pose
        for cabin in range(N_HEADINGS):
            cells = translated((row, col), cones[heading][cabin])
            inside = (
                (cells[:, 0] >= 0)
                & (cells[:, 0] < SHAPE[0])
                & (cells[:, 1] >= 0)
                & (cells[:, 1] < SHAPE[1])
            )
            cells = cells[inside]
            if np.any(accepted[cells[:, 0], cells[:, 1]]):
                reachable.add(pose)
                break
    return reachable


def action_table(
    *,
    target: np.ndarray,
    padding: np.ndarray,
    membership: np.ndarray,
    valid_poses: set[tuple[int, int, int]],
    components: dict[tuple[int, int, int], int],
    axis_bits: dict[tuple[int, int, int], int],
    backward_deltas: list[tuple[int, int]],
    cones: list[list[np.ndarray]],
    dump_reachable: set[tuple[int, int, int]],
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    target_cells = np.argwhere(target < 0)
    cell_index = {tuple(cell): index for index, cell in enumerate(target_cells.tolist())}
    actions: list[dict] = []
    for pose in valid_poses:
        row, col, heading = pose
        valid_bits = axis_bits.get(pose, 0)
        if not valid_bits or not backward_chain_qualified(
            pose, axis_bits, backward_deltas[heading]
        ):
            continue
        for cabin in range(N_HEADINGS):
            cells = translated((row, col), cones[heading][cabin])
            inside = (
                (cells[:, 0] >= 0)
                & (cells[:, 0] < SHAPE[0])
                & (cells[:, 1] >= 0)
                & (cells[:, 1] < SHAPE[1])
            )
            cells = cells[inside]
            # Terra rejects a dig if any physical workspace cell intersects a
            # static obstacle, even when another part of the cone reaches soil.
            if np.any(padding[cells[:, 0], cells[:, 1]] != 0):
                continue
            all_mask = 0
            bad_mask = 0
            for cell in cells:
                key = (int(cell[0]), int(cell[1]))
                index = cell_index.get(key)
                if index is None:
                    continue
                bit = 1 << index
                all_mask |= bit
                if int(membership[key]) & valid_bits == 0:
                    bad_mask |= bit
            if all_mask:
                actions.append(
                    {
                        "pose": pose,
                        "cabin": cabin,
                        "component": components[pose],
                        "all_mask": all_mask,
                        "bad_mask": bad_mask,
                        "dump_reachable": pose in dump_reachable,
                    }
                )
    return actions, cell_index


def monotone_closure(
    actions: list[dict], target_count: int, *, require_dump: bool
) -> dict:
    full = (1 << target_count) - 1
    by_component: dict[int, list[dict]] = defaultdict(list)
    for action in actions:
        if require_dump and not action["dump_reachable"]:
            continue
        by_component[action["component"]].append(action)

    best = {
        "covered": 0,
        "remaining": target_count,
        "component": None,
        "actions": [],
        "rounds": 0,
    }
    for component, candidates in by_component.items():
        cleared = 0
        used: list[dict] = []
        rounds = 0
        while cleared != full:
            remaining = full ^ cleared
            available = []
            for action in candidates:
                removal = action["all_mask"] & remaining
                if removal and not (action["bad_mask"] & remaining):
                    available.append((removal.bit_count(), removal, action))
            if not available:
                break
            _, removal, selected = max(available, key=lambda item: item[0])
            cleared |= removal
            used.append(
                {
                    "pose": list(selected["pose"]),
                    "cabin": selected["cabin"],
                    "removed": removal.bit_count(),
                    "dump_reachable": selected["dump_reachable"],
                }
            )
            rounds += 1
        covered = cleared.bit_count()
        if covered > best["covered"]:
            best = {
                "covered": covered,
                "remaining": target_count - covered,
                "component": component,
                "actions": used,
                "rounds": rounds,
            }
    best["coverage"] = best["covered"] / max(target_count, 1)
    best["complete"] = best["remaining"] == 0
    return best


def load_case(root: Path, sample_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    dataset = root / "dataset"
    target = np.load(dataset / "images" / f"img_{sample_index}.npy")
    padding = np.load(dataset / "occupancy" / f"img_{sample_index}.npy")
    dumpability = np.load(dataset / "dumpability" / f"img_{sample_index}.npy")
    metadata = json.loads(
        (root / "review_metadata" / f"img_{sample_index}.json").read_text()
    )
    return target, padding, dumpability, metadata


REVIEW_BANK = Path(
    "/home/lorenzo/moleworks/.artifacts/terra_map_distribution_review_v4/review_bank"
)
REVIEW_CASES = {
    "straight": "trn-straight-side2",
    "tee": "trn-tee-side2",
    "network": "trn-net-side2",
    "road": "trn-tee-side1-road",
}


def note_family(condition: str, topology: str) -> str:
    """Group a bank condition into the note's trench families."""
    if condition.endswith("-road") or "-road-" in condition:
        return "road"
    normalized = (topology or "").lower()
    if normalized == "straight":
        return "straight"
    if normalized in ("t", "tee"):
        return "tee"
    if normalized in ("seg2", "seg3"):
        return "segmented"
    return "network"


def review_cases(root: Path) -> list[dict]:
    cases = []
    for family, condition in REVIEW_CASES.items():
        manifest = json.loads((root / condition / "manifest.json").read_text())
        for entry in manifest["maps"]:
            sample_index = int(entry["sampleIndex"])
            cases.append(
                {
                    "label": f"{family}:{condition}:{sample_index}",
                    "family": family,
                    "condition": condition,
                    "images": str(root / "dataset" / "images" / f"img_{sample_index}.npy"),
                    "occupancy": str(
                        root / "dataset" / "occupancy" / f"img_{sample_index}.npy"
                    ),
                    "dumpability": str(
                        root / "dataset" / "dumpability" / f"img_{sample_index}.npy"
                    ),
                    "metadata": str(
                        root / "review_metadata" / f"img_{sample_index}.json"
                    ),
                }
            )
    return cases


def exact_datasets(root: Path) -> list[str]:
    found = []
    for path in sorted(root.rglob("dataset.json")):
        directory = path.parent
        if directory != root and (directory / "manifest.jsonl").is_file():
            found.append(str(directory.relative_to(root)))
    return found


def exact_cases(root: Path, relatives: list[str]) -> tuple[list[dict], list[dict]]:
    """Every trench map, split into analyzable cases and metadata failures.

    A declared trench map without finite sections is a preflight failure to
    report, never a map to drop.
    """
    cases = []
    missing: list[dict] = []
    for relative in relatives:
        directory = root / relative
        for line in (directory / "manifest.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("family") != "trench":
                continue
            slot = int(row["slot_index"])
            metadata_path = directory / "metadata" / f"trench_{slot}.json"
            metadata = json.loads(metadata_path.read_text())
            if not metadata.get("axes_ABC"):
                continue
            condition = str(row.get("primary_cell", relative))
            family = note_family(condition, str(metadata.get("trench_topology", "")))
            segments = metadata.get("trench_segments_yx") or metadata.get("trench_arms")
            half_width = metadata.get("trench_half_width_tiles")
            if not segments or half_width is None:
                missing.append(
                    {
                        "dataset": relative,
                        "slot_index": slot,
                        "map_id": str(row.get("map_id", "")),
                        "condition": condition,
                        "family": family,
                        "declared_axes": len(metadata["axes_ABC"]),
                        "reason": "declared trench axes without finite sections "
                        "or generated half width",
                    }
                )
                continue
            cases.append(
                {
                    "label": f"{family}:{condition}:{relative}:{slot}",
                    "family": family,
                    "condition": condition,
                    "dataset": relative,
                    "map_id": str(row.get("map_id", "")),
                    "images": str(directory / "images" / f"img_{slot}.npy"),
                    "occupancy": str(directory / "occupancy" / f"img_{slot}.npy"),
                    "dumpability": str(directory / "dumpability" / f"img_{slot}.npy"),
                    "metadata": str(metadata_path),
                }
            )
    return cases, missing


_WORKER: dict = {}


def _init_worker(cfg, cones, footprints, forward_deltas, backward_deltas) -> None:
    _WORKER.update(
        cfg=cfg,
        cones=cones,
        footprints=footprints,
        forward_deltas=forward_deltas,
        backward_deltas=backward_deltas,
    )


def _run_case(case: dict) -> dict:
    metadata = json.loads(Path(case["metadata"]).read_text())
    result = analyze_map(
        label=case["label"],
        target=np.load(case["images"], allow_pickle=False),
        padding=np.load(case["occupancy"], allow_pickle=False),
        dumpability=np.load(case["dumpability"], allow_pickle=False),
        metadata=metadata,
        cfg=_WORKER["cfg"],
        cones=_WORKER["cones"],
        footprints=_WORKER["footprints"],
        forward_deltas=_WORKER["forward_deltas"],
        backward_deltas=_WORKER["backward_deltas"],
    )
    result["family"] = case["family"]
    result["condition"] = case["condition"]
    if "dataset" in case:
        result["dataset"] = case["dataset"]
        result["map_id"] = case["map_id"]
    return result


def compact_result(result: dict) -> dict:
    """Drop the per-map witness action lists, keep every decision number."""
    compacted = dict(result)
    for key in ("unrestricted", "same_base_accepted_dump"):
        block = dict(compacted[key])
        block["actions"] = len(block["actions"])
        compacted[key] = block
    return compacted


def summarize_by(results: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        grouped[str(result[key])].append(result)
    rows = []
    for name in sorted(grouped):
        group = grouped[name]
        target_cells = sum(item["target_cells"] for item in group)
        rows.append(
            {
                key: name,
                "maps": len(group),
                "target_cells": target_cells,
                "fresh_complete_maps": sum(
                    item["unrestricted"]["complete"] for item in group
                ),
                "fresh_covered_cells": sum(
                    item["unrestricted"]["covered"] for item in group
                ),
                "incomplete_labels": [
                    item["label"] for item in group if not item["unrestricted"]["complete"]
                ][:16],
                "same_base_accepted_dump_complete_maps": sum(
                    item["same_base_accepted_dump"]["complete"] for item in group
                ),
                "same_base_accepted_dump_covered_cells": sum(
                    item["same_base_accepted_dump"]["covered"] for item in group
                ),
            }
        )
    return rows


def analyze_map(
    *,
    label: str,
    target: np.ndarray,
    padding: np.ndarray,
    dumpability: np.ndarray,
    metadata: dict,
    cfg: EnvConfig,
    cones: list[list[np.ndarray]],
    footprints: list[np.ndarray],
    forward_deltas: list[tuple[int, int]],
    backward_deltas: list[tuple[int, int]],
) -> dict:
    records, membership, axis_count = records_and_membership(target, metadata)
    target_count = int(np.count_nonzero(target < 0))
    persistent_blocked = padding.astype(bool) | (target < 0)
    valid_poses, components = valid_pose_graph(
        persistent_blocked, footprints, forward_deltas, backward_deltas
    )
    axis_bits = base_axis_bits(valid_poses, records, axis_count, cfg.tile_size)
    dump_reachable = dump_reachable_bases(
        target=target,
        padding=padding,
        dumpability_init=dumpability,
        valid_poses=valid_poses,
        cones=cones,
    )
    actions, _ = action_table(
        target=target,
        padding=padding,
        membership=membership,
        valid_poses=valid_poses,
        components=components,
        axis_bits=axis_bits,
        backward_deltas=backward_deltas,
        cones=cones,
        dump_reachable=dump_reachable,
    )
    unrestricted = monotone_closure(actions, target_count, require_dump=False)
    same_base_dump = monotone_closure(actions, target_count, require_dump=True)
    return {
        "label": label,
        "target_cells": target_count,
        "axes": axis_count,
        "persistent_pose_count": len(valid_poses),
        "persistent_components": len(set(components.values())),
        "candidate_macro_actions": len(actions),
        "candidate_dump_reachable_macro_actions": sum(
            bool(action["dump_reachable"]) for action in actions
        ),
        "unrestricted": unrestricted,
        "same_base_accepted_dump": same_base_dump,
    }


def summarize(results: list[dict]) -> list[dict]:
    """Return the compact per-family decision table."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        grouped[result["label"].split(":", 1)[0]].append(result)
    rows = []
    for family in sorted(grouped):
        family_results = grouped[family]
        target_cells = sum(result["target_cells"] for result in family_results)
        fresh_cells = sum(
            result["unrestricted"]["covered"] for result in family_results
        )
        accepted_dump_cells = sum(
            result["same_base_accepted_dump"]["covered"]
            for result in family_results
        )
        rows.append(
            {
                "family": family,
                "maps": len(family_results),
                "target_cells": target_cells,
                "fresh_complete_maps": sum(
                    result["unrestricted"]["complete"]
                    for result in family_results
                ),
                "fresh_covered_cells": fresh_cells,
                "same_base_accepted_dump_complete_maps": sum(
                    result["same_base_accepted_dump"]["complete"]
                    for result in family_results
                ),
                "same_base_accepted_dump_covered_cells": accepted_dump_cells,
                "same_base_accepted_dump_coverage": (
                    accepted_dump_cells / target_cells
                ),
                "minimum_map_same_base_accepted_dump_coverage": min(
                    result["same_base_accepted_dump"]["coverage"]
                    for result in family_results
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--bank",
        type=Path,
        default=REVIEW_BANK,
        help="bank root; defaults to the review-v4 generated bank",
    )
    parser.add_argument(
        "--layout",
        choices=("review", "exact"),
        default="review",
        help="'review' reads dataset/ + review_metadata/; 'exact' reads the "
        "runtime dataset layout with enriched metadata/trench_<slot>.json",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="exact-layout dataset relative path; repeatable, defaults to all",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="analyze only the first N maps (timing slice)",
    )
    parser.add_argument(
        "--witness-actions",
        choices=("full", "counts"),
        default="full",
        help="'counts' drops the per-map witness action lists from the report",
    )
    args = parser.parse_args()
    root = args.bank
    missing_metadata: list[dict] = []
    if args.layout == "review":
        cases = review_cases(root)
    else:
        relatives = args.dataset or exact_datasets(root)
        cases, missing_metadata = exact_cases(root, relatives)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases and not missing_metadata:
        raise RuntimeError(f"No trench maps found under {root}.")

    cfg = env_config()
    cones, footprints, forward_deltas, backward_deltas = terra_geometry(cfg)
    print(
        "geometry",
        json.dumps(
            {
                "tile_size_m": cfg.tile_size,
                "agent_width_tiles": cfg.agent.width,
                "agent_height_tiles": cfg.agent.height,
                "forward_deltas": forward_deltas,
                "backward_deltas": backward_deltas,
                "maps": len(cases),
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    started = time.time()
    if args.workers > 1:
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(cfg, cones, footprints, forward_deltas, backward_deltas),
        ) as pool:
            results = []
            for index, result in enumerate(
                pool.imap_unordered(_run_case, cases, chunksize=4), start=1
            ):
                results.append(result)
                if index % 50 == 0 or index == len(cases):
                    print(
                        f"[{index}/{len(cases)}] {time.time() - started:.0f}s",
                        flush=True,
                    )
        results.sort(key=lambda item: item["label"])
    else:
        _init_worker(cfg, cones, footprints, forward_deltas, backward_deltas)
        results = []
        for case in cases:
            result = _run_case(case)
            results.append(result)
            print(
                result["label"],
                f"fresh={result['unrestricted']['covered']}/{result['target_cells']}",
                "same_base_dump="
                f"{result['same_base_accepted_dump']['covered']}/"
                f"{result['target_cells']}",
                flush=True,
            )
    wall_seconds = time.time() - started
    incomplete = [
        result["label"] for result in results if not result["unrestricted"]["complete"]
    ]
    reproduction_command = (
        "JAX_PLATFORMS=cpu "
        "PYTHONPATH=/home/lorenzo/moleworks/.worktrees/"
        "terra_trench_fresh_dig_alignment_20260818 "
        "/home/lorenzo/moleworks/.venv-terra-uv/bin/python "
        "tools/audit_trench_alignment_feasibility.py "
        + " ".join(sys.argv[1:])
    )
    if args.witness_actions == "counts":
        results = [compact_result(result) for result in results]
    report = {
        "contract": {
            "terra_revision": "25f855db3d913fd638c4e56b1740437a2b7122ca",
            "bank": str(root),
            "layout": args.layout,
            "datasets": args.dataset,
            "maps": len(results),
            "workers": args.workers,
            "wall_seconds": round(wall_seconds, 2),
            "incomplete_fresh_cover_maps": incomplete[:64],
            "incomplete_fresh_cover_count": len(incomplete),
            "missing_finite_metadata_count": len(missing_metadata),
            "missing_finite_metadata": missing_metadata[:64],
            "preflight_passed": not incomplete and not missing_metadata,
            "metadata_source": (
                "review_metadata trench_arms and half-width"
                if args.layout == "review"
                else "enriched metadata/trench_<slot>.json trench_segments_yx "
                "and trench_half_width_tiles"
            ),
            "runtime_metadata_rounding": "float16 in MapsBuffer, float32 in geometry",
            "tile_size_m": cfg.tile_size,
            "agent_width_tiles": cfg.agent.width,
            "agent_height_tiles": cfg.agent.height,
            "headings": N_HEADINGS,
            "dig_cones": (
                "exact State._build_dig_dump_cone table for all 12 base x 12 "
                "cabin headings"
            ),
            "yaw_tolerance_deg": YAW_TOLERANCE_DEG,
            "standoff_min_m": STANDOFF_MIN_M,
            "standoff_max_m": STANDOFF_MAX_M,
            "pose_graph": (
                "complete target holes plus padding, endpoint-only Terra "
                "movement/rotation"
            ),
            "movement": (
                "deltas obtained through State._handle_move_forward/backward "
                "using Terra's float32 path"
            ),
            "macro_dig": (
                "each action removes its full selected remaining-fresh cone and "
                "is admitted only when every such cell has a pose-valid "
                "finite-section owner"
            ),
            "section_membership": (
                "finite segment plus generated half-width; nearest-only raster "
                "fringe bounded to +1.5 tiles"
            ),
            "chain": (
                "each used dig pose has one exact Terra BACKWARD successor "
                "sharing a yaw/standoff-valid section"
            ),
            "dump_probe": (
                "same base, any cabin, existence of an accepted target>0 cell "
                "after complete-hole 5x5 dynamic dumpability; not a "
                "pile/capacity workflow"
            ),
            "limitations": [
                "Terra endpoint movement only; no physical swept-path check",
                "no accumulated dumped piles, soil-mechanics trajectory, finite "
                "dump capacity proof, or episode-horizon proof",
                "one generated 16-map bank per representative family",
            ],
            "reproduction_command": reproduction_command,
            "forward_deltas": forward_deltas,
            "backward_deltas": backward_deltas,
        },
        "summary": summarize(results),
        "summary_by_family": summarize_by(results, "family"),
        "summary_by_condition": summarize_by(results, "condition"),
        "results": results,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
