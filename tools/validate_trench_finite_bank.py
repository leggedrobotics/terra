#!/usr/bin/env python3
"""Run the fail-closed trench-alignment load path over a bank, dataset by dataset.

This is the exact contract training uses: the canonical loader with
``require_trench_alignment_metadata=True`` followed by the batch-reset finite
section predicate from ``TerraEnvBatch._validate_trench_alignment_metadata_requirements``.
A dataset either loads with every declared axis carrying a finite segment and a
positive width, or it is reported as a rejection with its reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra.maps_buffer import (  # noqa: E402
    LEGACY_DISTANCE_PROTOCOL_ID,
    REWARD_V2_DISTANCE_PROTOCOL_ID,
    load_maps_from_disk,
)


def batch_reset_predicate(trench_axes: np.ndarray, trench_types: np.ndarray) -> list[list[int]]:
    """Reproduce terra/env.py:_validate_trench_alignment_metadata_requirements."""
    records = np.asarray(trench_axes)
    types = np.asarray(trench_types).reshape(-1)
    if records.shape[-1] < 8:
        raise RuntimeError("Loaded records are not [A,B,C,y0,x0,y1,x1,half_width].")
    axis_indices = np.arange(records.shape[-2])
    declared = axis_indices[None, :] < types[:, None]
    endpoints = records[..., 3:7].astype(np.float64)
    starts, ends = endpoints[..., :2], endpoints[..., 2:]
    finite = (
        np.all(np.isfinite(endpoints), axis=-1)
        & np.all(endpoints > -96.0, axis=-1)
        & (np.linalg.norm(ends - starts, axis=-1) > 1e-6)
        & np.isfinite(records[..., 7].astype(np.float64))
        & (records[..., 7].astype(np.float64) > 0.0)
    )
    return np.argwhere(declared & ~finite).tolist()


def datasets_under(root: Path) -> list[str]:
    found = []
    for path in sorted(root.rglob("dataset.json")):
        directory = path.parent
        if directory != root and (directory / "manifest.jsonl").is_file():
            found.append(str(directory.relative_to(root)))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument(
        "--distance-protocol",
        choices=("legacy", "reward_v2"),
        default="reward_v2",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.bank.resolve()
    protocol = (
        REWARD_V2_DISTANCE_PROTOCOL_ID
        if args.distance_protocol == "reward_v2"
        else LEGACY_DISTANCE_PROTOCOL_ID
    )
    relatives = args.dataset or datasets_under(root)

    results = []
    started = time.time()
    for relative in relatives:
        directory = root / relative
        slot_count = int(json.loads((directory / "dataset.json").read_text())["slot_count"])
        os.environ["DATASET_SIZE"] = str(slot_count)
        entry: dict[str, object] = {"dataset": relative, "slot_count": slot_count}
        try:
            loaded = load_maps_from_disk(
                str(directory),
                require_trench_metadata=False,
                require_trench_alignment_metadata=True,
                require_exact_contract=True,
                required_distance_protocol_id=protocol,
            )
            trench_axes = np.asarray(loaded[2])
            trench_types = np.asarray(loaded[3])
            missing = batch_reset_predicate(trench_axes, trench_types)
            entry["loaded"] = True
            entry["maps_with_trench_axes"] = int(np.count_nonzero(trench_types > 0))
            entry["declared_axes"] = int(np.sum(np.maximum(trench_types, 0)))
            entry["batch_reset_missing_finite_segments"] = len(missing)
            entry["accepted"] = not missing
            if missing:
                entry["first_missing_map_axis_indices"] = missing[:8]
        except Exception as exc:  # noqa: BLE001 - the rejection reason is the result
            entry["loaded"] = False
            entry["accepted"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
        print(
            f"{'PASS' if entry['accepted'] else 'REJECT'} {relative} "
            f"({slot_count} slots)"
            + ("" if entry["accepted"] else f" :: {str(entry.get('error', ''))[:160]}"),
            flush=True,
        )

    report = {
        "schema": "terra_trench_finite_loader_validation_v1",
        "bank": str(root),
        "distance_protocol_id": protocol,
        "require_trench_alignment_metadata": True,
        "datasets": results,
        "accepted_datasets": sum(1 for item in results if item["accepted"]),
        "rejected_datasets": sum(1 for item in results if not item["accepted"]),
        "accepted_slots": sum(
            int(item["slot_count"]) for item in results if item["accepted"]
        ),
        "wall_seconds": round(time.time() - started, 2),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(
        f"accepted {report['accepted_datasets']}/{len(results)} datasets, "
        f"{report['accepted_slots']} accepted slots, {report['wall_seconds']}s"
    )


if __name__ == "__main__":
    main()
