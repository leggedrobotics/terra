#!/usr/bin/env python3
"""Build the diversity-only B0 foundation-distance repair bank."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from terra.maps_buffer import load_maps_from_disk
from tools import build_b0_feasibility_panels as b0

CELLS = tuple(f"f_apron_d{distance:02d}" for distance in (2, 4, 6, 8))
SPLIT_COUNTS = {"train": 64, "development": 8}
REFERENCE_GROUP_COUNT = 8
REFERENCE_IDENTITIES_SHA256 = (
    "911b6e3a453d6d9e1aeaebfe5fcef33406c89aae0180e1c4eb8739efc1fd5b4e"
)
REFERENCE_FILES_MANIFEST_SHA256 = (
    "89a5b5325e4e6872f7899b087ac5d0a8cd444dac30315feee4f342f8e532a347"
)
SOURCE_FOUNDATIONS_SHA256 = (
    "77cd2983fa9e6c49fa524b2ad7d04587633a4052e212a49ba16822c2b689cc63"
)
BASE_BUILDER_SHA256 = "3a1bb66798f6a4bfc1dc5b3515c5a4485eb9a28d6ffe7c9e8413a548492b79a9"
GENERATOR_SHA256 = {
    "generate_prototypes.py": (
        "94c92eec14b27885c2f9743602178de1cff307e2a41be13a6a88f791c75a6284"
    ),
    "generate_prototypes_v2.py": (
        "74aed6460419ea0b16f426808a0057f8a6c3644765d01fbb0388aa84692430fc"
    ),
    "generate_prototypes_v3.py": (
        "9249ccc81e224949bf68f3b7197ab5efbdc5ef5c5bfb3c77e1490a0ce8adc2ce"
    ),
    "generate_prototypes_v4.py": (
        "39c24511bdc282a49c4b0729484aadb9661f76e29563ebaa9f6b4c29ad7543ee"
    ),
    "generate_prototypes_v5.py": (
        "725fd73f19a033bb31e92d4da7ba0f5dc7d1bfbbb9f46a136c0f478c9016d8ba"
    ),
}
MAX_DIVERSITY_GEOMETRY_IOU = 0.95
TENSOR_FIELDS = ("target", "occupancy", "dumpability", "action", "distance")
LOADER_FIELDS = (
    "target",
    "occupancy",
    "trench_axes",
    "trench_types",
    "trench_axis_owners",
    "foundation_border_axes",
    "foundation_border_types",
    "dumpability",
    "action",
    "distance",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_file_manifest(root: Path, manifest: Path) -> int:
    lines = manifest.read_text().splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        observed = b0.sha256_file(root / relative)
        if observed != expected:
            raise RuntimeError(f"reference manifest mismatch: {relative}")
    return len(lines)


def source_foundations_sha256(source_foundations: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((source_foundations / "images").glob("*.npy"))
    if not paths:
        raise RuntimeError("source foundation corpus is empty")
    for path in paths:
        relative = path.relative_to(source_foundations).as_posix()
        digest.update(f"{b0.sha256_file(path)}  {relative}\n".encode())
    return digest.hexdigest()


def verify_generator_files(generator_root: Path) -> dict[str, str]:
    observed = {
        name: b0.sha256_file(generator_root / name) for name in GENERATOR_SHA256
    }
    if observed != GENERATOR_SHA256:
        changed = [
            name
            for name, expected in GENERATOR_SHA256.items()
            if observed[name] != expected
        ]
        raise RuntimeError(f"unexpected map generator files: {changed}")
    return {str(generator_root / name): digest for name, digest in observed.items()}


def verify_base_builder() -> dict[str, str]:
    path = Path(b0.__file__).resolve()
    observed = b0.sha256_file(path)
    if observed != BASE_BUILDER_SHA256:
        raise RuntimeError("unexpected B0a base builder")
    return {"path": str(path), "sha256": observed}


def reference_index(record: dict) -> int:
    return int(record["map_id"].rsplit("-", 1)[-1])


def load_reference_panel(
    reference_b0a: Path,
) -> tuple[list[dict], dict[str, b0.Sample], list[dict], dict]:
    identities_path = reference_b0a / "identities.jsonl"
    files_manifest = reference_b0a / "files.sha256"
    if b0.sha256_file(identities_path) != REFERENCE_IDENTITIES_SHA256:
        raise RuntimeError("unexpected B0a identity manifest")
    if b0.sha256_file(files_manifest) != REFERENCE_FILES_MANIFEST_SHA256:
        raise RuntimeError("unexpected B0a file manifest")
    verified_files = verify_file_manifest(reference_b0a, files_manifest)

    all_records = load_jsonl(identities_path)
    selected = [
        copy.deepcopy(record)
        for record in all_records
        if record["split"] in SPLIT_COUNTS
        and record["primary_cell"] in CELLS
        and reference_index(record) < REFERENCE_GROUP_COUNT
    ]
    expected = len(CELLS) * REFERENCE_GROUP_COUNT * len(SPLIT_COUNTS)
    if len(selected) != expected:
        raise RuntimeError(
            f"reference foundation-distance panel has {len(selected)} rows, "
            f"expected {expected}"
        )
    by_map_id = {record["map_id"]: record for record in selected}
    samples: dict[str, b0.Sample] = {}
    for split in SPLIT_COUNTS:
        dataset = reference_b0a / "panels" / split / "foundation_distance"
        manifest = load_jsonl(dataset / "manifest.jsonl")
        if len(manifest) != len(CELLS) * REFERENCE_GROUP_COUNT:
            raise RuntimeError(f"unexpected {split} reference panel size")
        for row in manifest:
            map_id = row["map_id"]
            record = by_map_id.get(map_id)
            if record is None:
                raise RuntimeError(f"unknown reference map {map_id}")
            slot = int(row["slot_index"])
            stem = f"img_{slot}.npy"
            samples[map_id] = b0.Sample(
                target=np.load(dataset / "images" / stem, allow_pickle=False),
                occupancy=np.load(
                    dataset / "occupancy" / stem,
                    allow_pickle=False,
                ),
                dumpability=np.load(
                    dataset / "dumpability" / stem,
                    allow_pickle=False,
                ),
                action=np.load(dataset / "actions" / stem, allow_pickle=False),
                distance=np.load(
                    dataset / "distance" / stem,
                    allow_pickle=False,
                ),
                metadata=copy.deepcopy(record),
            )
    if set(samples) != set(by_map_id):
        raise RuntimeError("reference panel tensors are incomplete")
    for record in selected:
        record["stratum"] = "B0D"
    return (
        selected,
        samples,
        all_records,
        {
            "identities_sha256": REFERENCE_IDENTITIES_SHA256,
            "files_manifest_sha256": REFERENCE_FILES_MANIFEST_SHA256,
            "verified_manifest_files": verified_files,
        },
    )


def validate_reference_subset(
    records: list[dict],
    samples: dict[str, b0.Sample],
    reference_records: list[dict],
    reference_samples: dict[str, b0.Sample],
) -> dict:
    generated = {
        record["map_id"]: record
        for record in records
        if reference_index(record) < REFERENCE_GROUP_COUNT
    }
    reference = {record["map_id"]: record for record in reference_records}
    if set(generated) != set(reference):
        raise RuntimeError("retained foundation-distance map IDs changed")
    for map_id, reference_record in reference.items():
        generated_record = generated[map_id]
        left = {
            key: value for key, value in generated_record.items() if key != "stratum"
        }
        right = {
            key: value for key, value in reference_record.items() if key != "stratum"
        }
        if left != right:
            raise RuntimeError(f"retained identity record changed: {map_id}")
        for field in TENSOR_FIELDS:
            if not np.array_equal(
                getattr(samples[map_id], field),
                getattr(reference_samples[map_id], field),
            ):
                raise RuntimeError(f"retained {field} tensor changed: {map_id}")
    return {
        "passed": True,
        "retained_records": len(reference),
        "record_fields": "all_except_stratum",
        "exact_tensor_fields": list(TENSOR_FIELDS),
    }


def paired_group_validation(
    group_records: list[dict],
    samples: dict[str, b0.Sample],
) -> None:
    if {record["primary_cell"] for record in group_records} != set(CELLS):
        raise RuntimeError("foundation-distance pair lacks all four cells")
    for field in (
        "source_id",
        "dig_identity_sha256",
        "generation_seed",
        "generation_attempt",
    ):
        if len({record[field] for record in group_records}) != 1:
            raise RuntimeError(f"foundation-distance pair changed {field}")
    if len({record["target_identity_sha256"] for record in group_records}) != 4:
        raise RuntimeError("foundation-distance pair reused a target")
    if len({record["validation"]["dig_cells"] for record in group_records}) != 1:
        raise RuntimeError("foundation-distance pair changed dig volume")
    if len({record["validation"]["dump_cells"] for record in group_records}) != 1:
        raise RuntimeError("foundation-distance pair changed dump capacity")

    ordered = sorted(group_records, key=lambda record: record["distance_center_tiles"])
    medians = [record["validation"]["distance"]["p50_tiles"] for record in ordered]
    if any(left >= right for left, right in zip(medians, medians[1:])):
        raise RuntimeError("foundation-distance medians are not strictly ordered")
    first = samples[ordered[0]["map_id"]]
    first_dig = first.target < 0
    for record in ordered[1:]:
        sample = samples[record["map_id"]]
        if not np.array_equal(sample.target < 0, first_dig):
            raise RuntimeError("foundation-distance pair changed dig raster")
        for field in ("occupancy", "dumpability", "action"):
            if not np.array_equal(getattr(sample, field), getattr(first, field)):
                raise RuntimeError(f"foundation-distance pair changed {field}")


def validate_records(
    records: list[dict],
    samples: dict[str, b0.Sample],
    all_reference_osm_sources: set[str],
) -> dict:
    expected = sum(SPLIT_COUNTS.values()) * len(CELLS)
    if len(records) != expected:
        raise RuntimeError(f"generated {len(records)} rows, expected {expected}")
    counts = Counter((record["split"], record["primary_cell"]) for record in records)
    expected_counts = {
        (split, cell): count for split, count in SPLIT_COUNTS.items() for cell in CELLS
    }
    if counts != expected_counts:
        raise RuntimeError(f"foundation-distance diversity counts differ: {counts}")
    if set(samples) != {record["map_id"] for record in records}:
        raise RuntimeError("foundation-distance samples are incomplete")
    map_ids = [record["map_id"] for record in records]
    target_hashes = [record["target_identity_sha256"] for record in records]
    if len(map_ids) != len(set(map_ids)):
        raise RuntimeError("foundation-distance map IDs are not unique")
    if len(target_hashes) != len(set(target_hashes)):
        raise RuntimeError("foundation-distance targets are not unique")

    source_splits: dict[str, set[str]] = {}
    for record in records:
        source_splits.setdefault(record["source_id"], set()).add(record["split"])
    overlap = {
        source: splits for source, splits in source_splits.items() if len(splits) > 1
    }
    if overlap:
        raise RuntimeError(f"foundation-distance sources cross splits: {overlap}")
    reused_added_sources = {
        record["source_id"]
        for record in records
        if record["split"] == "train"
        and reference_index(record) >= REFERENCE_GROUP_COUNT
        and record["source_id"] in all_reference_osm_sources
    }
    if reused_added_sources:
        raise RuntimeError(
            f"added foundation sources reuse B0a: {sorted(reused_added_sources)}"
        )
    unique_sources_by_split = {
        split: len(
            {record["source_id"] for record in records if record["split"] == split}
        )
        for split in SPLIT_COUNTS
    }
    if unique_sources_by_split != SPLIT_COUNTS:
        raise RuntimeError(
            f"unexpected foundation source counts: {unique_sources_by_split}"
        )

    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["paired_source_group_id"], []).append(record)
        validation = record["validation"]
        if validation["status"] != "passed":
            raise RuntimeError("foundation-distance static validation failed")
        if validation["accepted_dump_contract"] != "exact_visible_dump_v1":
            raise RuntimeError("foundation-distance dump contract changed")
        if (
            validation["capacity"]["single_layer_capacity_ratio"]
            < b0.MINIMUM_CAPACITY_RATIO
        ):
            raise RuntimeError("foundation-distance capacity is too small")
        distance_error = abs(
            validation["distance"]["p50_tiles"] - record["distance_center_tiles"]
        )
        if distance_error > b0.DISTANCE_TOLERANCE_TILES:
            raise RuntimeError("foundation-distance cell left its distance bin")
    if len(groups) != sum(SPLIT_COUNTS.values()):
        raise RuntimeError("unexpected foundation-distance pair count")
    for group_records in groups.values():
        paired_group_validation(group_records, samples)

    representatives = [
        (group, samples[rows[0]["map_id"]].target < 0)
        for group, rows in sorted(groups.items())
    ]
    maximum_cross_group_iou = 0.0
    for index, (_, left) in enumerate(representatives):
        for _, right in representatives[index + 1 :]:
            similarity = b0.maximum_dihedral_iou(left, right)
            maximum_cross_group_iou = max(maximum_cross_group_iou, similarity)
            if similarity >= MAX_DIVERSITY_GEOMETRY_IOU:
                raise RuntimeError(
                    "foundation-distance bank contains a templated duplicate"
                )

    return {
        "passed": True,
        "identity_count": len(records),
        "paired_geometry_groups": len(groups),
        "counts": {
            f"{split}/{cell}": counts[(split, cell)]
            for split in SPLIT_COUNTS
            for cell in CELLS
        },
        "unique_map_ids": len(set(map_ids)),
        "unique_target_arrays": len(set(target_hashes)),
        "unique_sources_by_split": unique_sources_by_split,
        "source_disjoint_splits": True,
        "added_sources_absent_from_b0a": True,
        "maximum_cross_group_geometry_iou": maximum_cross_group_iou,
        "maximum_allowed_geometry_iou": MAX_DIVERSITY_GEOMETRY_IOU,
    }


def generate_bank(
    v5,
    source_foundations: Path,
    reference_records: list[dict],
    reference_samples: dict[str, b0.Sample],
    all_reference_records: list[dict],
):
    records = copy.deepcopy(reference_records)
    samples = {
        map_id: b0.Sample(
            target=sample.target.copy(),
            occupancy=sample.occupancy.copy(),
            dumpability=sample.dumpability.copy(),
            action=sample.action.copy(),
            distance=sample.distance.copy(),
            metadata=copy.deepcopy(sample.metadata),
        )
        for map_id, sample in reference_samples.items()
    }
    all_reference_osm_sources = {
        record["source_id"]
        for record in all_reference_records
        if record["source_id"].startswith("osm-foundation:")
    }
    used_osm_sources = {
        int(source.rsplit(":", 1)[-1]) for source in all_reference_osm_sources
    }
    geometry_factory = v5.v3.GeometryFactoryV3(source_foundations)
    accepted_digs: dict[tuple[str, str], list[np.ndarray]] = {}
    for record in records:
        accepted_digs.setdefault(
            (record["split"], record["primary_cell"]),
            [],
        ).append(samples[record["map_id"]].target < 0)
    rejections: Counter[str] = Counter()

    for identity_index in range(REFERENCE_GROUP_COUNT, SPLIT_COUNTS["train"]):
        base_seed = b0.SPLIT_BASE_SEEDS["train"] + 300_000 + identity_index * 1_000
        for duplicate_attempt in range(200):
            seed = base_seed + duplicate_attempt
            dig, metadata, source_id, attempt = b0.make_geometry(
                v5,
                geometry_factory,
                "foundation_osm",
                seed,
                used_osm_sources,
            )
            similarity = max(
                b0.maximum_previous_geometry_iou(
                    accepted_digs,
                    split,
                    CELLS[0],
                    dig,
                )
                for split in SPLIT_COUNTS
            )
            if similarity >= MAX_DIVERSITY_GEOMETRY_IOU:
                rejections["train:foundation_distance:templated_duplicate"] += 1
                continue
            break
        else:
            raise RuntimeError("exhausted diverse foundation-distance identities")

        pair_group = f"train:foundation-distance:{identity_index:02d}"
        for cell in CELLS:
            b0.add_record(
                records,
                samples,
                v5=v5,
                split="train",
                spec=b0.CELLS[cell],
                identity_index=identity_index,
                dig=dig,
                geometry_metadata=metadata,
                source_id=source_id,
                generation_seed=seed,
                generation_attempt=attempt,
                paired_source_group_id=pair_group,
                topology_match_group_id=None,
                side_sign=1,
                accepted_digs=accepted_digs,
            )
            records[-1]["stratum"] = "B0D"
    return records, samples, all_reference_osm_sources, dict(rejections)


def selected_gallery_records(records: list[dict], split: str, cell: str) -> list[dict]:
    selected = sorted(
        (
            record
            for record in records
            if record["split"] == split and record["primary_cell"] == cell
        ),
        key=lambda record: reference_index(record),
    )
    if split == "development":
        return selected
    indices = set(range(8)) | {8, 16, 24, 32, 40, 48, 56, 63}
    return [record for record in selected if reference_index(record) in indices]


def validate_full_loader(
    output: Path,
    datasets: dict[str, int],
) -> dict[str, dict[str, object]]:
    """Load every emitted dataset through the exact runtime loader."""
    previous_dataset_size = os.environ.get("DATASET_SIZE")
    loaded_datasets = {}
    try:
        for relative, expected_count in sorted(datasets.items()):
            os.environ["DATASET_SIZE"] = str(expected_count)
            arrays = load_maps_from_disk(
                str(output / relative),
                require_trench_metadata=False,
                require_exact_contract=True,
            )
            if len(arrays) != len(LOADER_FIELDS):
                raise RuntimeError(
                    f"runtime loader returned {len(arrays)} fields for "
                    f"{relative}, expected {len(LOADER_FIELDS)}"
                )
            field_shapes = {}
            field_dtypes = {}
            for field, array in zip(LOADER_FIELDS, arrays):
                shape = tuple(int(value) for value in array.shape)
                if not shape or shape[0] != expected_count:
                    raise RuntimeError(
                        f"runtime loader returned shape {shape} for "
                        f"{relative}/{field}, expected leading dimension "
                        f"{expected_count}"
                    )
                field_shapes[field] = list(shape)
                field_dtypes[field] = str(array.dtype)
            loaded_datasets[relative] = {
                "expected_count": expected_count,
                "field_shapes": field_shapes,
                "field_dtypes": field_dtypes,
            }
    finally:
        if previous_dataset_size is None:
            os.environ.pop("DATASET_SIZE", None)
        else:
            os.environ["DATASET_SIZE"] = previous_dataset_size
    return loaded_datasets


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
    generator_root = args.generator_root.resolve()
    source_foundations = args.source_foundations.resolve()
    reference_b0a = args.reference_b0a.resolve()
    base_builder = verify_base_builder()
    generator_files = verify_generator_files(generator_root)
    observed_source_digest = source_foundations_sha256(source_foundations)
    if observed_source_digest != SOURCE_FOUNDATIONS_SHA256:
        raise RuntimeError("unexpected foundation source corpus")

    reference_records, reference_samples, all_reference_records, reference_gate = (
        load_reference_panel(reference_b0a)
    )
    v5 = b0.load_review_generator(generator_root)
    records, samples, all_reference_osm_sources, rejections = generate_bank(
        v5,
        source_foundations,
        reference_records,
        reference_samples,
        all_reference_records,
    )
    static_validation = validate_records(
        records,
        samples,
        all_reference_osm_sources,
    )
    reference_validation = validate_reference_subset(
        records,
        samples,
        reference_records,
        reference_samples,
    )
    output.mkdir(parents=True)

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
            cell_records = [
                record for record in split_records if record["primary_cell"] == cell
            ]
            relative = f"cells/{split}/{cell}"
            b0.write_dataset(output / relative, cell_records, samples, source_registry)
            datasets[relative] = len(cell_records)
            gallery_dir = output / "galleries" / split
            gallery_dir.mkdir(parents=True, exist_ok=True)
            b0.render_cell_gallery(
                gallery_dir / f"{cell}.png",
                selected_gallery_records(records, split, cell),
                samples,
            )
        relative = f"panels/{split}/foundation_distance"
        b0.write_dataset(output / relative, split_records, samples, source_registry)
        datasets[relative] = len(split_records)

    full_loader_validation = validate_full_loader(output, datasets)

    paired_gallery_dir = output / "galleries" / "paired"
    paired_gallery_dir.mkdir(parents=True)
    train_records = [record for record in records if record["split"] == "train"]
    for name, indices in (
        ("retained_00_07", {0, 7}),
        ("added_08_63", {8, 63}),
    ):
        selected = [
            record for record in train_records if reference_index(record) in indices
        ]
        b0.render_panel_gallery(
            paired_gallery_dir / f"{name}.png",
            f"foundation_distance_{name}",
            CELLS,
            selected,
            samples,
        )

    provenance = {
        "schema": "terra_b0_foundation_distance_diversity_v1",
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": b0.sha256_file(Path(__file__).resolve()),
        },
        "base_builder": base_builder,
        "generator_files": generator_files,
        "source_foundations": str(source_foundations),
        "source_foundations_sha256": observed_source_digest,
        "reference_b0a": str(reference_b0a),
        "reference_gate": reference_gate,
        "split_base_seeds": b0.SPLIT_BASE_SEEDS,
        "split_counts_per_cell": SPLIT_COUNTS,
        "retained_groups_per_split": REFERENCE_GROUP_COUNT,
        "maximum_diversity_geometry_iou": MAX_DIVERSITY_GEOMETRY_IOU,
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
            "full_loader": full_loader_validation,
        },
    )
    b0.write_json(
        output / "generation_summary.json",
        {
            "schema": "terra_b0_foundation_distance_diversity_summary_v1",
            "accepted_identities": len(records),
            "paired_geometry_groups": sum(SPLIT_COUNTS.values()),
            "split_counts_per_cell": SPLIT_COUNTS,
            "rejection_counts": rejections,
        },
    )
    (output / "README.md").write_text(
        "# B0 foundation-distance diversity repair\n\n"
        "This bank changes only foundation source-geometry diversity. It keeps "
        "the exact eight B0a training and eight development source groups, "
        "then adds 56 training groups shared across d02/d04/d06/d08. Reward, "
        "dynamics, dump-distance bins, capacity, PPO, and evaluation identities "
        "are unchanged.\n"
    )
    b0.file_manifest(output)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
