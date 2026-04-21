#!/usr/bin/env python3
"""
Thin CLI wrapper around `terra.env_generation.distance` for the obstacle-aware
(geodesic BFS) distance metric.

Keeps the pre-existing behavior: output is written to `<dataset>/distance_geodesic/`
(not `distance/`), so an existing Manhattan dataset isn't overwritten.
"""
import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from terra.env_generation.distance import (
    compute_geodesic_distance_map,  # re-exported for any external callers
    write_distance_maps,
)


def _walk_and_run(root: Path, connectivity: int) -> None:
    for dirpath, _dirnames, _filenames in os.walk(root):
        p = Path(dirpath)
        if (p / "images").exists() and (p / "occupancy").exists():
            try:
                write_distance_maps(
                    p,
                    metric="geodesic",
                    connectivity=connectivity,
                    distance_subfolder="distance_geodesic",
                )
            except Exception as e:
                print(f"Skipping {p} due to error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate geodesic (obstacle-aware) relocation distance maps for Terra datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a dataset folder containing images/ and occupancy/, or a root folder when --recursive is set",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, walk subfolders and process all datasets discovered",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=4,
        help="Grid neighbor connectivity (default: 4)",
    )
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        raise RuntimeError(f"Dataset path does not exist: {root}")

    if args.recursive:
        _walk_and_run(root, args.connectivity)
    else:
        write_distance_maps(
            root,
            metric="geodesic",
            connectivity=args.connectivity,
            distance_subfolder="distance_geodesic",
        )


if __name__ == "__main__":
    main()
