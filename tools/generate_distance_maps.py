#!/usr/bin/env python3
"""
Thin CLI wrapper around `terra.env_generation.distance`.

Kept for backwards compatibility with the workflow documented in the README
(`python tools/generate_distance_maps.py --dataset ...`). The actual logic
lives in `terra/env_generation/distance.py` and is invoked automatically by
`convert_to_terra._convert_all_imgs_to_terra`, so new datasets no longer need
to run this script manually.
"""
import argparse
import sys
from pathlib import Path

# Allow running this file directly without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from terra.env_generation.distance import (
    DEFAULT_REALISTIC_MAX_DISTANCE,
    compute_distance_map_taxicab,  # re-exported for any external callers
    write_distance_maps,
    write_distance_maps_recursive,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Manhattan (taxicab) relocation distance maps for Terra datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a dataset folder containing images/, or a root folder when --recursive is set",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, walk subfolders and process all datasets discovered",
    )
    parser.add_argument(
        "--realistic-max-distance",
        type=int,
        default=DEFAULT_REALISTIC_MAX_DISTANCE,
        help="Normalization constant for the Manhattan metric (default: %(default)s)",
    )
    parser.add_argument(
        "--obstacle-proximity-cost",
        action="store_true",
        help="If set, add extra cost near obstacles on non-dump tiles.",
    )
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        raise RuntimeError(f"Dataset path does not exist: {root}")

    if args.recursive:
        write_distance_maps_recursive(
            root,
            metric="manhattan",
            realistic_max_distance=args.realistic_max_distance,
            obstacle_proximity_cost=args.obstacle_proximity_cost,
        )
    else:
        write_distance_maps(
            root,
            metric="manhattan",
            realistic_max_distance=args.realistic_max_distance,
            obstacle_proximity_cost=args.obstacle_proximity_cost,
        )


if __name__ == "__main__":
    main()
