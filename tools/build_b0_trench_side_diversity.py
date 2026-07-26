#!/usr/bin/env python3
"""Build the diversity-only B0 close trench-side repair bank."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import build_b0_feasibility_panels as b0

CELLS = ("t_straight_both_d02", "t_straight_one_d02")
SPLIT_COUNTS = {"train": 64, "development": 8}
REFERENCE_FIELDS = (
    "map_id",
    "source_id",
    "split",
    "family",
    "primary_cell",
    "geometry",
    "dump_layout",
    "distance_center_tiles",
    "side_access",
    "topology",
    "generation_seed",
    "generation_attempt",
    "paired_source_group_id",
    "topology_match_group_id",
    "dig_identity_sha256",
    "target_identity_sha256",
    "maximum_within_cell_geometry_iou",
    "validation",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_reference_subset(
    records: list[dict],
    reference_records: list[dict],
) -> dict:
    generated = {record["map_id"]: record for record in records}
    reference = {
        record["map_id"]: record
        for record in reference_records
        if record["split"] in SPLIT_COUNTS
        and record["primary_cell"] in CELLS
        and int(record["map_id"].rsplit("-", 1)[-1]) < 8
    }
    expected = len(CELLS) * 8 * len(SPLIT_COUNTS)
    if len(reference) != expected:
        raise RuntimeError(
            f"reference close-side subset has {len(reference)} rows, expected {expected}"
        )
    mismatches = {}
    for map_id, reference_record in reference.items():
        generated_record = generated.get(map_id)
        if generated_record is None:
            mismatches[map_id] = "missing"
            continue
        differing = [
            field
            for field in REFERENCE_FIELDS
            if generated_record[field] != reference_record[field]
        ]
        if differing:
            mismatches[map_id] = differing
    if mismatches:
        first = next(iter(mismatches.items()))
        raise RuntimeError(f"B0 reference subset mismatch: {first}")
    return {
        "passed": True,
        "reference_rows": expected,
        "exact_fields": list(REFERENCE_FIELDS),
        "reference_identity_manifest_sha256": b0.sha256_file(
            Path(reference_records[0]["_identity_manifest"])
        ),
    }


def validate_records(records: list[dict]) -> dict:
    expected = sum(SPLIT_COUNTS.values()) * len(CELLS)
    if len(records) != expected:
        raise RuntimeError(f"generated {len(records)} rows, expected {expected}")
    counts = Counter((record["split"], record["primary_cell"]) for record in records)
    expected_counts = {
        (split, cell): count for split, count in SPLIT_COUNTS.items() for cell in CELLS
    }
    if counts != expected_counts:
        raise RuntimeError(f"diversity-bank counts differ: {counts}")
    map_ids = [record["map_id"] for record in records]
    target_hashes = [record["target_identity_sha256"] for record in records]
    if len(map_ids) != len(set(map_ids)):
        raise RuntimeError("diversity-bank map IDs are not unique")
    if len(target_hashes) != len(set(target_hashes)):
        raise RuntimeError("diversity-bank target arrays are not unique")
    source_splits = {}
    for record in records:
        source_splits.setdefault(record["source_id"], set()).add(record["split"])
    overlap = {
        source: splits for source, splits in source_splits.items() if len(splits) > 1
    }
    if overlap:
        raise RuntimeError(f"diversity-bank sources cross splits: {overlap}")

    pair_hashes = {}
    for record in records:
        pair_hashes.setdefault(record["paired_source_group_id"], set()).add(
            record["dig_identity_sha256"]
        )
    bad_pairs = {
        pair: values for pair, values in pair_hashes.items() if len(values) != 1
    }
    if bad_pairs:
        raise RuntimeError(f"side pair changed dig geometry: {bad_pairs}")
    if any(record["validation"]["status"] != "passed" for record in records):
        raise RuntimeError("diversity-bank static validation failed")
    return {
        "passed": True,
        "identity_count": len(records),
        "counts": {
            f"{split}/{cell}": counts[(split, cell)]
            for split in SPLIT_COUNTS
            for cell in CELLS
        },
        "unique_map_ids": len(set(map_ids)),
        "unique_target_arrays": len(set(target_hashes)),
        "source_disjoint_splits": True,
        "paired_dig_geometry_preserved": True,
    }


def generate_bank(v5, source_foundations: Path):
    geometry_factory = v5.v3.GeometryFactoryV3(source_foundations)
    used_osm_sources: set[int] = set()
    accepted_digs: dict[tuple[str, str], list[np.ndarray]] = {}
    records = []
    samples = {}
    rejections: Counter[str] = Counter()

    for split, count in SPLIT_COUNTS.items():
        split_seed = b0.SPLIT_BASE_SEEDS[split]
        for identity_index in range(count):
            base_seed = split_seed + 400_000 + identity_index * 1_000
            for duplicate_attempt in range(200):
                seed = base_seed + duplicate_attempt
                dig, metadata, source_id, attempt = b0.make_geometry(
                    v5,
                    geometry_factory,
                    "trench_straight",
                    seed,
                    used_osm_sources,
                )
                similarity = b0.maximum_previous_geometry_iou(
                    accepted_digs,
                    split,
                    CELLS[0],
                    dig,
                )
                if similarity >= b0.MAX_WITHIN_CELL_GEOMETRY_IOU:
                    rejections[f"{split}:trench_straight:templated_duplicate"] += 1
                    continue
                break
            else:
                raise RuntimeError(
                    f"exhausted diverse identities for {split}/trench_side"
                )

            pair_group = f"{split}:trench-straight:{identity_index:02d}"
            side_sign = -1 if identity_index % 2 else 1
            for cell in CELLS:
                b0.add_record(
                    records,
                    samples,
                    v5=v5,
                    split=split,
                    spec=b0.CELLS[cell],
                    identity_index=identity_index,
                    dig=dig,
                    geometry_metadata=metadata,
                    source_id=source_id,
                    generation_seed=seed,
                    generation_attempt=attempt,
                    paired_source_group_id=pair_group,
                    topology_match_group_id=(
                        f"{split}:topology:{identity_index:02d}"
                        if cell == CELLS[0]
                        else None
                    ),
                    side_sign=side_sign,
                    accepted_digs=accepted_digs,
                )
                records[-1]["stratum"] = "B0D"
    return records, samples, dict(rejections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--source-foundations", type=Path, required=True)
    parser.add_argument("--reference-b0a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    generator_root = args.generator_root.resolve()
    source_foundations = args.source_foundations.resolve()
    reference_b0a = args.reference_b0a.resolve()

    v5 = b0.load_review_generator(generator_root)
    records, samples, rejections = generate_bank(v5, source_foundations)
    static_validation = validate_records(records)
    reference_path = reference_b0a / "identities.jsonl"
    reference_records = load_jsonl(reference_path)
    for record in reference_records:
        record["_identity_manifest"] = str(reference_path)
    reference_validation = validate_reference_subset(records, reference_records)

    source_registry = output / "source_registry.jsonl"
    b0.write_jsonl(
        source_registry,
        [
            {
                "map_id": record["map_id"],
                "source_id": record["source_id"],
                "split": record["split"],
                "paired_source_group_id": record["paired_source_group_id"],
            }
            for record in records
        ],
    )
    b0.write_jsonl(output / "identities.jsonl", records)
    scalar_fields = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if not isinstance(value, (dict, list))
        }
    )
    with (output / "identities.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows(
            [
                {key: b0.json_value(record.get(key, "")) for key in scalar_fields}
                for record in records
            ]
        )

    datasets = {}
    for split in SPLIT_COUNTS:
        split_records = [record for record in records if record["split"] == split]
        for cell in CELLS:
            selected = [
                record for record in split_records if record["primary_cell"] == cell
            ]
            relative = f"cells/{split}/{cell}"
            b0.write_dataset(output / relative, selected, samples, source_registry)
            datasets[relative] = len(selected)
            gallery_dir = output / "galleries" / split
            gallery_dir.mkdir(parents=True, exist_ok=True)
            b0.render_cell_gallery(
                gallery_dir / f"{cell}.png",
                selected[:16],
                samples,
            )
        relative = f"panels/{split}/trench_side"
        b0.write_dataset(output / relative, split_records, samples, source_registry)
        datasets[relative] = len(split_records)
    gallery_dir = output / "galleries" / "panels"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    b0.render_panel_gallery(
        gallery_dir / "trench_side.png",
        "trench_side_diversity",
        CELLS,
        [record for record in records if record["split"] == "train"],
        samples,
    )

    generator_files = [
        generator_root / f"generate_prototypes{suffix}.py"
        for suffix in ("", "_v2", "_v3", "_v4", "_v5")
    ]
    provenance = {
        "schema": "terra_b0_trench_side_diversity_v1",
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": b0.sha256_file(Path(__file__).resolve()),
        },
        "base_builder": {
            "path": str(Path(b0.__file__).resolve()),
            "sha256": b0.sha256_file(Path(b0.__file__).resolve()),
        },
        "generator_files": {
            str(path): b0.sha256_file(path) for path in generator_files
        },
        "source_foundations": str(source_foundations),
        "reference_b0a": str(reference_b0a),
        "split_base_seeds": b0.SPLIT_BASE_SEEDS,
        "split_counts_per_cell": SPLIT_COUNTS,
        "cells": CELLS,
        "dataset_directories": datasets,
        "identity_manifest_sha256": b0.sha256_file(output / "identities.jsonl"),
    }
    b0.write_json(output / "provenance.json", provenance)
    b0.write_json(
        output / "validation.json",
        {
            "status": "passed",
            "static": static_validation,
            "reference_subset": reference_validation,
            "rejection_counts": rejections,
            "dataset_directories": datasets,
        },
    )
    b0.write_json(
        output / "generation_summary.json",
        {
            "schema": "terra_b0_trench_side_diversity_summary_v1",
            "accepted_identities": len(records),
            "split_counts_per_cell": SPLIT_COUNTS,
            "rejection_counts": rejections,
        },
    )
    (output / "README.md").write_text(
        "# B0 close trench-side diversity repair\n\n"
        "This bank changes only the number of unique training geometries for "
        "`t_straight_both_d02` and `t_straight_one_d02`: 64 per cell instead "
        "of eight. The first eight train identities and all eight development "
        "identities per cell match the frozen B0a bank. Reward, dynamics, "
        "distance, side constraints, and evaluation identities are unchanged.\n"
    )
    b0.file_manifest(output)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
