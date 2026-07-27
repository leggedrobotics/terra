#!/usr/bin/env python3
"""Audit one supported volume band for the S1 trench pilot.

This is a train-only generator audit. It samples one fixed-width straight
generator and two candidate segmented length ranges, then selects one
10-cell closed band by a deterministic support rule. It does not build maps,
touch held-out seed namespaces, or claim static/dynamic feasibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_b0_feasibility_panels import (  # noqa: E402
    load_review_generator,
    sha256_array,
    sha256_file,
)
from pilot_map_generation import sample_segmented_trench  # noqa: E402

SCHEMA = "terra_pilot_trench_volume_support_v1"
DEFAULT_SEED = 2_026_072_701
DEFAULT_PROPOSALS = 20_000
DEFAULT_BAND_WIDTH = 10
MINIMUM_SUPPORT_RATE = 0.10
INTERIOR_QUANTILES = (0.10, 0.90)
CANDIDATE_LENGTH_RANGES = {
    "segmented_2": (10.0, 13.0),
    "segmented_3": (7.0, 9.5),
}
TOPOLOGIES = ("straight", "segmented_2", "segmented_3")
CANDIDATE_SAMPLER_REVISION = "s1_trench_volume_match_v1"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def sample_rng(seed: int, topology_index: int, sample_index: int):
    sequence = np.random.SeedSequence([seed, topology_index, sample_index])
    return np.random.default_rng(sequence)


def line_length(axis: dict[str, float]) -> float:
    return math.hypot(float(axis["A"]), float(axis["B"]))


def sample_straight(v5, rng: np.random.Generator):
    rejections = {
        "straight_width_radius_not_1": 0,
        "straight_volume_outside_prefilter": 0,
    }
    for _ in range(1_000):
        dig, metadata = v5.v3.GeometryFactoryV3.trench_axes_1(rng)
        if int(metadata["trench_width_radius_tiles"]) != 1:
            rejections["straight_width_radius_not_1"] += 1
            continue
        volume = int(dig.sum())
        if not 55 <= volume <= 125:
            rejections["straight_volume_outside_prefilter"] += 1
            continue
        metadata = dict(metadata)
        metadata["segment_lengths_tiles"] = [line_length(metadata["axes_ABC"][0])]
        metadata["turn_angles_deg"] = []
        metadata["audit_generator_draw_count"] = 1 + sum(rejections.values())
        metadata["audit_generator_rejections"] = rejections
        return dig, metadata
    raise RuntimeError("could not sample a fixed-width straight trench")


def sample_topology(
    v5,
    topology: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if topology == "straight":
        return sample_straight(v5, rng)
    if topology == "segmented_2":
        return sample_segmented_trench(
            v5,
            rng,
            2,
            CANDIDATE_LENGTH_RANGES[topology],
        )
    if topology == "segmented_3":
        return sample_segmented_trench(
            v5,
            rng,
            3,
            CANDIDATE_LENGTH_RANGES[topology],
        )
    raise ValueError(f"unsupported topology: {topology}")


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def eligible_bands(
    volumes_by_topology: dict[str, list[int]],
    band_width: int,
    minimum_support_rate: float = MINIMUM_SUPPORT_RATE,
) -> list[dict[str, Any]]:
    if band_width < 1:
        raise ValueError("band_width must be positive")
    if set(volumes_by_topology) != set(TOPOLOGIES):
        raise ValueError(f"expected topology keys {TOPOLOGIES}")
    if any(not values for values in volumes_by_topology.values()):
        raise ValueError("each topology requires at least one volume")

    lower_limit = min(min(values) for values in volumes_by_topology.values())
    upper_limit = max(max(values) for values in volumes_by_topology.values())
    interior = {
        topology: np.quantile(values, INTERIOR_QUANTILES)
        for topology, values in volumes_by_topology.items()
    }
    candidates = []
    for lower in range(lower_limit, upper_limit - band_width + 2):
        upper = lower + band_width - 1
        midpoint = (lower + upper) / 2.0
        support = {
            topology: sum(lower <= value <= upper for value in values) / len(values)
            for topology, values in volumes_by_topology.items()
        }
        if min(support.values()) < minimum_support_rate:
            continue
        if any(
            not float(bounds[0]) <= midpoint <= float(bounds[1])
            for bounds in interior.values()
        ):
            continue
        candidates.append(
            {
                "lower_inclusive": lower,
                "upper_inclusive": upper,
                "midpoint": midpoint,
                "support_rate": support,
                "minimum_support_rate": min(support.values()),
                "mean_support_rate": sum(support.values()) / len(support),
            }
        )
    return sorted(
        candidates,
        key=lambda row: (
            -row["minimum_support_rate"],
            -row["mean_support_rate"],
            row["lower_inclusive"],
        ),
    )


def summarize_rows(
    rows: list[dict[str, Any]],
    lower: int,
    upper: int,
) -> dict[str, Any]:
    selected = [row for row in rows if lower <= row["required_volume"] <= upper]
    raw_geometry_hashes = {row["geometry_sha256"] for row in rows}
    selected_geometry_hashes = {row["geometry_sha256"] for row in selected}
    raw_lengths = [value for row in rows for value in row["segment_lengths_tiles"]]
    selected_lengths = [
        value for row in selected for value in row["segment_lengths_tiles"]
    ]
    raw_turns = [abs(value) for row in rows for value in row["turn_angles_deg"]]
    selected_turns = [
        abs(value) for row in selected for value in row["turn_angles_deg"]
    ]
    generator_rejection_keys = sorted(
        {key for row in rows for key in row["generator_rejections_before_proposal"]}
    )
    return {
        "proposal_count": len(rows),
        "generator_draw_count": sum(row["generator_draw_count"] for row in rows),
        "generator_rejections_before_proposal": {
            key: sum(
                row["generator_rejections_before_proposal"].get(key, 0) for row in rows
            )
            for key in generator_rejection_keys
        },
        "accepted_count": len(selected),
        "support_rate": len(selected) / len(rows),
        "unique_geometry_count": len(raw_geometry_hashes),
        "accepted_unique_geometry_count": len(selected_geometry_hashes),
        "rejections": {
            "outside_volume_band": len(rows) - len(selected),
            "duplicate_raster": len(rows) - len(raw_geometry_hashes),
            "accepted_duplicate_raster": len(selected) - len(selected_geometry_hashes),
        },
        "required_volume_raw": quantiles(
            [float(row["required_volume"]) for row in rows]
        ),
        "required_volume_accepted": quantiles(
            [float(row["required_volume"]) for row in selected]
        ),
        "segment_length_raw": quantiles(raw_lengths),
        "segment_length_accepted": quantiles(selected_lengths),
        "absolute_turn_angle_raw": quantiles(raw_turns) if raw_turns else None,
        "absolute_turn_angle_accepted": (
            quantiles(selected_turns) if selected_turns else None
        ),
    }


def generator_hashes(generator_root: Path) -> dict[str, str]:
    files = sorted(generator_root.glob("generate_prototypes*.py"))
    if not files:
        raise FileNotFoundError(
            f"no generate_prototypes*.py files under {generator_root}"
        )
    return {path.name: sha256_file(path) for path in files}


def file_manifest(output: Path) -> None:
    paths = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "files.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    (output / "files.sha256").write_text("\n".join(lines) + "\n")


def run_audit(
    generator_root: Path,
    output: Path,
    seed: int,
    proposal_count: int,
    band_width: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if proposal_count < 1:
        raise ValueError("proposal_count must be positive")
    output.mkdir(parents=True)
    v5 = load_review_generator(generator_root)

    rows = []
    volumes_by_topology: dict[str, list[int]] = {}
    for topology_index, topology in enumerate(TOPOLOGIES):
        topology_rows = []
        for sample_index in range(proposal_count):
            rng = sample_rng(seed, topology_index, sample_index)
            dig, metadata = sample_topology(v5, topology, rng)
            topology_rows.append(
                {
                    "topology": topology,
                    "sample_index": sample_index,
                    "required_volume": int(dig.sum()),
                    "segment_lengths_tiles": [
                        float(value) for value in metadata["segment_lengths_tiles"]
                    ],
                    "turn_angles_deg": [
                        float(value) for value in metadata["turn_angles_deg"]
                    ],
                    "trench_width_radius_tiles": int(
                        metadata["trench_width_radius_tiles"]
                    ),
                    "generator_draw_count": int(metadata["audit_generator_draw_count"]),
                    "generator_rejections_before_proposal": dict(
                        metadata["audit_generator_rejections"]
                    ),
                    "geometry_sha256": sha256_array(dig),
                }
            )
        rows.extend(topology_rows)
        volumes_by_topology[topology] = [
            row["required_volume"] for row in topology_rows
        ]

    candidates = eligible_bands(volumes_by_topology, band_width)
    if not candidates:
        raise RuntimeError("no volume band satisfies the frozen support gate")
    selected = candidates[0]
    lower = int(selected["lower_inclusive"])
    upper = int(selected["upper_inclusive"])
    for row in rows:
        row["accepted_volume_band"] = lower <= row["required_volume"] <= upper

    write_jsonl(output / "samples.jsonl", rows)
    summary = {
        "schema": SCHEMA,
        "status": "passed",
        "seed_namespace": "train_only",
        "seed": seed,
        "proposal_count_per_topology": proposal_count,
        "band_width_cells": band_width,
        "minimum_support_rate": MINIMUM_SUPPORT_RATE,
        "interior_quantiles": list(INTERIOR_QUANTILES),
        "selected_band": selected,
        "eligible_band_count": len(candidates),
        "top_eligible_bands": candidates[:20],
        "candidate_length_ranges": {
            key: list(value) for key, value in CANDIDATE_LENGTH_RANGES.items()
        },
        "candidate_sampler_revision": CANDIDATE_SAMPLER_REVISION,
        "fixed_trench_width_radius_tiles": 1,
        "topologies": {
            topology: summarize_rows(
                [row for row in rows if row["topology"] == topology],
                lower,
                upper,
            )
            for topology in TOPOLOGIES
        },
        "inputs": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "pilot_map_generator_sha256": sha256_file(
                TOOLS_DIR / "pilot_map_generation.py"
            ),
            "b0_builder_sha256": sha256_file(
                TOOLS_DIR / "build_b0_feasibility_panels.py"
            ),
            "generator_files": generator_hashes(generator_root),
        },
    }
    write_json(output / "support_summary.json", summary)
    file_manifest(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--proposal-count", type=int, default=DEFAULT_PROPOSALS)
    parser.add_argument("--band-width", type=int, default=DEFAULT_BAND_WIDTH)
    args = parser.parse_args()
    summary = run_audit(
        args.generator_root.resolve(),
        args.output.resolve(),
        args.seed,
        args.proposal_count,
        args.band_width,
    )
    print(json.dumps(summary["selected_band"], sort_keys=True))


if __name__ == "__main__":
    main()
