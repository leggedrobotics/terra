#!/usr/bin/env python3
"""Pool the 12 preflight-covered enriched trench train conditions into one dataset.

The fresh-trench dig-alignment pilot (C0/T1) trains on a single curriculum level
so both arms see the identical map distribution with no per-condition ratchet.
The frozen V8 R2 bank stores one 96-map dataset per condition, so this merges the
12 pilot conditions into one contiguous 1,152-slot dataset that still satisfies
``terra.maps_buffer.validate_exact_dataset_contract`` and the reward-v2 distance
contract.

Determinism: conditions are taken in sorted directory-name order, maps within a
condition in numeric slot order, so slot ranges are a pure function of the input
set. Array sidecars are symlinked at their fully resolved targets (the enriched
bank already symlinks into the original R2 bank); only the enriched
``metadata/trench_*.json`` files are copied, because those carry the finite
``trench_segments_yx`` / ``trench_half_width_tiles`` keys the pilot needs.

Nothing in the source bank is mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# The 12 fully-preflight-covered trench train conditions: the 15 enriched train
# conditions minus the three net4 conditions (excluded by the yaw-tolerance
# probe). v7-trn conditions are unenriched and are never eligible.
PILOT_CONDITIONS: tuple[str, ...] = (
    "018__trn-net3-side1-road",
    "019__trn-net3-side2",
    "020__trn-net3-side2-s",
    "024__trn-seg2-side2",
    "025__trn-seg3-side2",
    "026__trn-straight-altsides",
    "027__trn-straight-side1",
    "028__trn-straight-side1-tight",
    "029__trn-straight-side2",
    "030__trn-tee-side2",
    "031__trn-tee-side2-s",
    "033__trn-straight-allfree",
)

ARRAY_FOLDERS: tuple[str, ...] = (
    "images",
    "occupancy",
    "dumpability",
    "actions",
    "distance",
)
METADATA_FOLDER = "metadata"
POOLING_SCHEMA = "terra_trench_pilot_pooled_train_v1"

# Fields every pooled condition must agree on; they define the loader contract.
UNIFORM_FIELDS: tuple[str, ...] = (
    "accepted_dump_contract",
    "distance_bound",
    "distance_metric",
    "distance_normalization",
    "distance_protocol_id",
    "distance_ref_m",
    "lineage",
    "scenario_identity_contract",
    "schema",
    "shape",
    "source_registry_sha256",
    "tile_size_m",
)
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "slot_index",
    "map_id",
    "source_id",
    "split",
    "family",
    "stratum",
    "primary_cell",
    "slot_weight",
    "identity_slot_multiplicity",
    "scenario_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{number} is not a JSON object")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank-root",
        required=True,
        help="Enriched frozen bank root (contains train/, source_registry.jsonl).",
    )
    parser.add_argument(
        "--pooled-name",
        default="train_pilot_pooled_12cond",
        help="Pooled dataset folder name, created directly under the bank root.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Pooling manifest path (default: <bank-root>/<pooled-name>_manifest.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing pooled folder before rebuilding.",
    )
    args = parser.parse_args()

    started = time.time()
    bank_root = Path(args.bank_root).resolve()
    pooled_dir = bank_root / args.pooled_name
    manifest_out = (
        Path(args.manifest_out).resolve()
        if args.manifest_out
        else bank_root / f"{args.pooled_name}_manifest.json"
    )
    tool_path = Path(__file__).resolve()
    tool_sha = sha256_file(tool_path)

    registry_path = bank_root / "source_registry.jsonl"
    if not registry_path.is_file():
        raise RuntimeError(f"Missing bank source registry: {registry_path}")
    registry_sha = sha256_file(registry_path)

    conditions = tuple(sorted(PILOT_CONDITIONS))
    if conditions != tuple(PILOT_CONDITIONS):
        raise RuntimeError("PILOT_CONDITIONS must already be in sorted order")

    # ---- read and cross-check every source condition -------------------------
    sources: list[dict] = []
    for condition in conditions:
        source_dir = bank_root / "train" / condition
        dataset_path = source_dir / "dataset.json"
        manifest_path = source_dir / "manifest.jsonl"
        if not dataset_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"Not an exact dataset: {source_dir}")
        dataset = json.loads(dataset_path.read_text())
        rows = read_jsonl(manifest_path)
        if dataset.get("slot_count") != len(rows):
            raise RuntimeError(
                f"{condition}: slot_count {dataset.get('slot_count')} != "
                f"{len(rows)} manifest rows"
            )
        if [row.get("slot_index") for row in rows] != list(range(1, len(rows) + 1)):
            raise RuntimeError(f"{condition}: manifest slots are not 1..N")
        if "trench_finite_enrichment" not in dataset:
            raise RuntimeError(
                f"{condition} is not finite-enriched; refusing to pool it."
            )
        sources.append(
            {
                "condition": condition,
                "dir": source_dir,
                "dataset": dataset,
                "rows": rows,
            }
        )

    reference = sources[0]["dataset"]
    for source in sources[1:]:
        divergent = [
            field
            for field in UNIFORM_FIELDS
            if source["dataset"].get(field) != reference.get(field)
        ]
        if divergent:
            raise RuntimeError(
                f"{source['condition']} diverges from "
                f"{sources[0]['condition']} on {divergent}"
            )
    if reference.get("source_registry_sha256") != registry_sha:
        raise RuntimeError(
            "Condition dataset.json source_registry_sha256 does not match "
            f"{registry_path}"
        )
    minimum_ratios = {
        source["dataset"].get("minimum_dump_capacity_ratio") for source in sources
    }
    if len(minimum_ratios) != 1:
        raise RuntimeError(
            f"Conditions disagree on minimum_dump_capacity_ratio: {minimum_ratios}"
        )
    minimum_ratio = minimum_ratios.pop()

    total_slots = sum(len(source["rows"]) for source in sources)

    # ---- materialize ---------------------------------------------------------
    if pooled_dir.exists():
        if not args.force:
            raise RuntimeError(
                f"{pooled_dir} already exists; pass --force to rebuild it."
            )
        shutil.rmtree(pooled_dir)
    for folder in (*ARRAY_FOLDERS, METADATA_FOLDER):
        (pooled_dir / folder).mkdir(parents=True)

    pooled_rows: list[dict] = []
    condition_slots: list[dict] = []
    metadata_sha: dict[str, str] = {}
    slot = 0
    for source in sources:
        first_slot = slot + 1
        for row in source["rows"]:
            slot += 1
            source_slot = int(row["slot_index"])
            for folder in ARRAY_FOLDERS:
                origin = source["dir"] / folder / f"img_{source_slot}.npy"
                target = origin.resolve(strict=True)
                if not target.is_file():
                    raise RuntimeError(f"Array sidecar is not a file: {origin}")
                (pooled_dir / folder / f"img_{slot}.npy").symlink_to(target)
            metadata_origin = (
                source["dir"] / METADATA_FOLDER / f"trench_{source_slot}.json"
            )
            if metadata_origin.is_symlink() or not metadata_origin.is_file():
                raise RuntimeError(
                    "Expected an enriched regular-file trench metadata sidecar: "
                    f"{metadata_origin}"
                )
            metadata_target = pooled_dir / METADATA_FOLDER / f"trench_{slot}.json"
            shutil.copyfile(metadata_origin, metadata_target)
            metadata_payload = json.loads(metadata_target.read_text())
            finite = metadata_payload.get("trench_finite_metadata")
            if (
                finite is None
                or metadata_payload.get("trench_segments_yx") is None
                or metadata_payload.get("trench_half_width_tiles") is None
            ):
                raise RuntimeError(
                    "Pooled metadata is missing finite trench sections: "
                    f"{metadata_origin}"
                )
            metadata_sha[f"trench_{slot}.json"] = sha256_file(metadata_target)

            pooled_row = dict(row)
            pooled_row["slot_index"] = slot
            pooled_row["pooled_from_dataset"] = f"train/{source['condition']}"
            pooled_row["pooled_from_slot_index"] = source_slot
            missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in pooled_row]
            if missing:
                raise RuntimeError(
                    f"Pooled slot {slot} is missing manifest fields {missing}"
                )
            pooled_rows.append(pooled_row)
        condition_slots.append(
            {
                "condition_id": source["dataset"]
                .get("trench_finite_enrichment", {})
                .get("source_dataset", f"train/{source['condition']}"),
                "source_dataset": f"train/{source['condition']}",
                "primary_cell": source["rows"][0]["primary_cell"],
                "map_count": len(source["rows"]),
                "first_slot": first_slot,
                "last_slot": slot,
            }
        )

    if slot != total_slots:
        raise RuntimeError(f"Pooled {slot} slots, expected {total_slots}")

    multiplicities: dict[str, int] = {}
    for row in pooled_rows:
        multiplicities[row["map_id"]] = multiplicities.get(row["map_id"], 0) + 1
    for row in pooled_rows:
        row["identity_slot_multiplicity"] = multiplicities[row["map_id"]]

    manifest_payload = (
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in pooled_rows
        )
    ).encode()
    (pooled_dir / "manifest.jsonl").write_bytes(manifest_payload)

    pooled_dataset = {
        field: reference[field] for field in UNIFORM_FIELDS if field in reference
    }
    pooled_dataset.update(
        {
            "slot_count": total_slots,
            "unique_identity_count": len(multiplicities),
            "source_registry": os.path.relpath(registry_path, pooled_dir),
            "source_registry_sha256": registry_sha,
            "trench_finite_enrichment": {
                "schema": "terra_trench_finite_sections_v1",
                "enriched_slots": total_slots,
                "dropped_slots": 0,
                "unenriched_slots_kept": 0,
                "source_bank": reference["trench_finite_enrichment"]["source_bank"],
                "source_dataset": [f"train/{c}" for c in conditions],
                "tool_sha256": reference["trench_finite_enrichment"]["tool_sha256"],
            },
            "trench_pilot_pooling": {
                "schema": POOLING_SCHEMA,
                "tool": str(tool_path),
                "tool_sha256": tool_sha,
                "bank_root": str(bank_root),
                "condition_count": len(conditions),
                "conditions": [f"train/{c}" for c in conditions],
                "order": "sorted condition directory name, then numeric source slot",
            },
        }
    )
    if minimum_ratio is not None:
        pooled_dataset["minimum_dump_capacity_ratio"] = minimum_ratio
    dataset_payload = canonical_json(pooled_dataset)
    (pooled_dir / "dataset.json").write_bytes(dataset_payload)

    pooling_manifest = {
        "schema": POOLING_SCHEMA,
        "generated_unix": int(started),
        "tool": {"path": str(tool_path), "sha256": tool_sha},
        "bank_root": str(bank_root),
        "pooled_dataset": os.path.relpath(pooled_dir, bank_root),
        "slot_count": total_slots,
        "unique_identity_count": len(multiplicities),
        "source_registry": {
            "path": os.path.relpath(registry_path, bank_root),
            "sha256": registry_sha,
        },
        "link_mode": {
            "arrays": "symlink_to_resolved_target",
            "metadata": "copy",
        },
        "conditions": condition_slots,
        "dataset_json_sha256": sha256_bytes(dataset_payload),
        "manifest_jsonl_sha256": sha256_bytes(manifest_payload),
        "metadata_sha256": metadata_sha,
        "wall_seconds": round(time.time() - started, 2),
    }
    manifest_out.write_bytes(canonical_json(pooling_manifest))

    print(f"pooled_dataset={pooled_dir}")
    print(f"slot_count={total_slots}")
    print(f"conditions={len(conditions)}")
    print(f"dataset_json_sha256={pooling_manifest['dataset_json_sha256']}")
    print(f"manifest_jsonl_sha256={pooling_manifest['manifest_jsonl_sha256']}")
    print(f"pooling_manifest={manifest_out}")
    print(f"pooling_manifest_sha256={sha256_file(manifest_out)}")
    print(f"tool_sha256={tool_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
