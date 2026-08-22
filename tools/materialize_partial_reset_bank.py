#!/usr/bin/env python3
"""Build the sparse action-only partial reset bank used by training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from terra.env_generation.partial_reset_bank import (
    materialize_sparse_partial_reset_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts-per-variant", type=int, default=100)
    parser.add_argument("--min-spawn-centers", type=int, default=16)
    parser.add_argument(
        "--pile-mode",
        action="append",
        dest="pile_modes",
        help=(
            "Ordered per-source pile policy. Repeat for fallbacks; defaults to "
            "relay_corridor."
        ),
    )
    parser.add_argument(
        "--include-maps-path-file",
        type=Path,
        help="Optional newline-delimited list of declared training paths to process.",
    )
    args = parser.parse_args()
    include_maps_paths = None
    if args.include_maps_path_file is not None:
        include_maps_paths = tuple(
            line.strip()
            for line in args.include_maps_path_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    receipt = materialize_sparse_partial_reset_bank(
        args.input_root,
        args.output_root,
        seed=args.seed,
        max_attempts_per_variant=args.max_attempts_per_variant,
        min_spawn_centers=args.min_spawn_centers,
        pile_modes=tuple(args.pile_modes or ("relay_corridor",)),
        include_maps_paths=include_maps_paths,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
