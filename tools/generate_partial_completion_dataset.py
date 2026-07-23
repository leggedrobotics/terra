#!/usr/bin/env python3
"""Generate a Terra-format dataset of partial-completion reset states."""

from __future__ import annotations

import argparse
from pathlib import Path

from terra.env_generation.partial_completion import PartialCompletionConfig
from terra.env_generation.partial_completion import generate_partial_dataset


def _parse_fractions(value: str) -> tuple[float, ...]:
    try:
        fractions = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not fractions:
        raise argparse.ArgumentTypeError(
            "At least one completion fraction is required."
        )
    return fractions


def _parse_mode_weights(value: str) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    try:
        for item in value.split(","):
            name, weight = item.split("=", maxsplit=1)
            result.append((name.strip(), float(weight)))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Mode weights must look like in_zone=0.65,near_zone=0.25,mixed=0.10."
        ) from error
    return tuple(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create physically plausible, mass-conserving partial Terra reset maps "
            "without planning an excavation solution."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--completion-fractions",
        type=_parse_fractions,
        default=(0.25, 0.50, 0.75, 0.90),
    )
    parser.add_argument("--variants-per-fraction", type=int, default=1)
    parser.add_argument(
        "--mode-weights",
        type=_parse_mode_weights,
        default=(("in_zone", 1.0),),
    )
    parser.add_argument("--min-piles", type=int, default=1)
    parser.add_argument("--max-piles", type=int, default=3)
    parser.add_argument("--min-center-separation", type=int, default=4)
    parser.add_argument("--near-distance-min", type=int, default=2)
    parser.add_argument("--near-distance-max", type=int, default=8)
    parser.add_argument("--max-pile-height", type=int, default=32)
    parser.add_argument("--max-workspace-load", type=int, default=127)
    parser.add_argument("--min-spawn-centers", type=int, default=16)
    parser.add_argument("--max-attempts-per-variant", type=int, default=100)
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = PartialCompletionConfig(
        completion_fractions=args.completion_fractions,
        variants_per_fraction=args.variants_per_fraction,
        mode_weights=args.mode_weights,
        min_piles=args.min_piles,
        max_piles=args.max_piles,
        min_center_separation=args.min_center_separation,
        near_distance_min=args.near_distance_min,
        near_distance_max=args.near_distance_max,
        max_pile_height=args.max_pile_height,
        max_workspace_load=args.max_workspace_load,
        min_spawn_centers=args.min_spawn_centers,
        max_attempts_per_variant=args.max_attempts_per_variant,
        include_full=args.include_full,
        limit=args.limit,
        seed=args.seed,
    )
    generate_partial_dataset(args.input, args.output, config=config)


if __name__ == "__main__":
    main()
