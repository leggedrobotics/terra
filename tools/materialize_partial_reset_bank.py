#!/usr/bin/env python3
"""Build the sparse action-only relay reset bank used by training."""

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
    args = parser.parse_args()
    receipt = materialize_sparse_partial_reset_bank(
        args.input_root,
        args.output_root,
        seed=args.seed,
        max_attempts_per_variant=args.max_attempts_per_variant,
        min_spawn_centers=args.min_spawn_centers,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
