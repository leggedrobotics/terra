#!/usr/bin/env python3
"""Fail closed unless every trench partial sidecar has aligned dig coverage."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from audit_trench_alignment_feasibility import action_table
from audit_trench_alignment_feasibility import base_axis_bits
from audit_trench_alignment_feasibility import env_config
from audit_trench_alignment_feasibility import monotone_closure
from audit_trench_alignment_feasibility import records_and_membership
from audit_trench_alignment_feasibility import terra_geometry
from audit_trench_alignment_feasibility import valid_pose_graph
from terra.env_generation.partial_completion import _load_source_layers
from terra.maps_buffer import PARTIAL_COMPLETION_MANIFEST
from terra.maps_buffer import PARTIAL_RESET_BANK_INDEX
from terra.maps_buffer import PARTIAL_RESET_BANK_SCHEMA
from terra.maps_buffer import partial_reset_bank_sha256
from terra.state import State


_WORKER: dict = {}


def _json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _init_worker(cfg, cones, footprints, forward_deltas, backward_deltas) -> None:
    _WORKER.update(
        cfg=cfg,
        cones=cones,
        footprints=footprints,
        forward_deltas=forward_deltas,
        backward_deltas=backward_deltas,
    )


def _audit_triplet(task: dict) -> list[dict]:
    source_directory = Path(task["source_directory"])
    partial_directory = Path(task["partial_directory"])
    source_index = int(task["source_index"])
    target, occupancy, dumpability = _load_source_layers(
        source_directory,
        source_index,
    )
    del dumpability
    metadata = json.loads(
        (
            source_directory
            / "metadata"
            / f"trench_{source_index}.json"
        ).read_text()
    )
    results = []
    for row in task["rows"]:
        action = np.load(
            partial_directory / "actions" / f"img_{row['sidecar_index']}.npy",
            allow_pickle=False,
        )
        remaining = (target < 0) & (action >= 0)
        partial_target = target.copy()
        partial_target[(target < 0) & ~remaining] = 0
        runtime_blocked = np.asarray(
            State._build_traversability_mask(
                jnp.asarray(action),
                jnp.asarray(occupancy),
            )
        ).astype(bool)
        # A service station is persistent only if it remains footprint-clear
        # after all currently fresh cells have become holes.
        persistent_blocked = runtime_blocked | remaining
        valid_poses, components = valid_pose_graph(
            persistent_blocked,
            _WORKER["footprints"],
            _WORKER["forward_deltas"],
            _WORKER["backward_deltas"],
        )
        records, membership, axis_count = records_and_membership(
            partial_target,
            metadata,
        )
        axis_bits = base_axis_bits(
            valid_poses,
            records,
            axis_count,
            _WORKER["cfg"].tile_size,
        )
        actions, _ = action_table(
            target=partial_target,
            padding=occupancy,
            membership=membership,
            valid_poses=valid_poses,
            components=components,
            axis_bits=axis_bits,
            backward_deltas=_WORKER["backward_deltas"],
            cones=_WORKER["cones"],
            dump_reachable=set(),
        )
        closure = monotone_closure(
            actions,
            int(np.count_nonzero(remaining)),
            require_dump=False,
        )
        results.append(
            {
                "maps_path": task["maps_path"],
                "condition_id": task["condition_id"],
                "source_index": source_index,
                "source_map_id": row["source_map_id"],
                "source_scenario_id": row["source_scenario_id"],
                "reset_tier": row["reset_tier"],
                "pile_mode": row["pile_mode"],
                "completion_fraction": row["achieved_completion_fraction"],
                "remaining_cells": int(np.count_nonzero(remaining)),
                "persistent_pose_count": len(valid_poses),
                "aligned_pose_count": sum(bool(bits) for bits in axis_bits.values()),
                "candidate_aligned_actions": len(actions),
                "remaining_cells_covered": closure["covered"],
                "alignment_chain_complete": bool(closure["complete"]),
            }
        )
    return results


def _tasks(canonical_root: Path, partial_root: Path) -> list[dict]:
    index = json.loads((partial_root / PARTIAL_RESET_BANK_INDEX).read_text())
    if index.get("schema") != PARTIAL_RESET_BANK_SCHEMA:
        raise RuntimeError("partial-reset bank has the wrong schema")
    tasks = []
    for maps_path in index["supported_maps_paths"]:
        source_directory = canonical_root / maps_path
        canonical_rows = _json_lines(source_directory / "manifest.jsonl")
        partial_directory = partial_root / maps_path
        partial_rows = _json_lines(
            partial_directory / PARTIAL_COMPLETION_MANIFEST
        )
        by_source: dict[int, list[dict]] = {}
        for row in partial_rows:
            by_source.setdefault(int(row["source_index"]), []).append(row)
        for source_index, rows in sorted(by_source.items()):
            canonical = canonical_rows[source_index - 1]
            if canonical.get("family") != "trench":
                continue
            if [row["reset_tier"] for row in rows] != [1, 2, 3]:
                raise RuntimeError(
                    f"{maps_path}:{source_index} is not one ordered reset triplet"
                )
            tasks.append(
                {
                    "maps_path": maps_path,
                    "condition_id": canonical["primary_cell"],
                    "source_directory": str(source_directory),
                    "partial_directory": str(partial_directory),
                    "source_index": source_index,
                    "rows": rows,
                }
            )
    return tasks


def audit(
    canonical_root: Path,
    partial_root: Path,
    *,
    workers: int,
) -> dict:
    canonical_root = canonical_root.resolve()
    partial_root = partial_root.resolve()
    digest = partial_reset_bank_sha256(partial_root)
    index = json.loads((partial_root / PARTIAL_RESET_BANK_INDEX).read_text())
    if digest != index.get("bank_sha256"):
        raise RuntimeError("partial-reset bank digest mismatch")
    cfg = env_config()
    cones, footprints, forward_deltas, backward_deltas = terra_geometry(cfg)
    tasks = _tasks(canonical_root, partial_root)
    if workers <= 1:
        _init_worker(cfg, cones, footprints, forward_deltas, backward_deltas)
        grouped = [_audit_triplet(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(cfg, cones, footprints, forward_deltas, backward_deltas),
        ) as pool:
            grouped = pool.map(_audit_triplet, tasks)
    rows = [row for group in grouped for row in group]
    failures = [row for row in rows if not row["alignment_chain_complete"]]
    return {
        "schema": "terra_partial_trench_alignment_audit_v1",
        "partial_reset_bank_sha256": digest,
        "canonical_loader_registry_sha256": index.get(
            "canonical_loader_registry_sha256"
        ),
        "yaw_tolerance_degrees": 15.0,
        "standoff_metres": [3.5, 7.0],
        "persistent_pose_semantics": (
            "footprint clear after all remaining fresh cells become holes"
        ),
        "claim_limit": (
            "Necessary persistent aligned dig-chain coverage; not an exact "
            "full-episode motion/dump plan."
        ),
        "trench_source_triplets": len(tasks),
        "audited_sidecars": len(rows),
        "passed_sidecars": len(rows) - len(failures),
        "failed_sidecars": len(failures),
        "accepted": not failures,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    receipt = audit(
        args.canonical_root,
        args.partial_root,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )
    if not receipt["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
