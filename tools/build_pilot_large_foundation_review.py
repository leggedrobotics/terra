#!/usr/bin/env python3
"""Build one narrow, train-only large-foundation visual review cell.

The cell deliberately changes no site factor beyond the foundation raster:
every map has all-around dumping, no obstacles, no action restrictions, and
the frozen Terra protocol.  Its procedural targets occupy only 328--340 cells
(8.01--8.30% of a 64 x 64 site).  This is a useful large-tail visual candidate,
not evidence of broad 8--12% support and not an admitted benchmark condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
from typing import Any, Iterable

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from terra.benchmark_protocol import BENCHMARK_RELEASE_ID
from terra.benchmark_protocol import frozen_benchmark_protocol
from terra.benchmark_protocol import frozen_environment_protocol
from terra.benchmark_state import agent_to_record
from terra.benchmark_state import sample_benchmark_initial_agent
from terra.benchmark_state import validate_benchmark_initial_agent
from terra.maps_buffer import contained_dump_capacity_sanity_check
from terra.maps_buffer import load_maps_from_disk
from terra.maps_buffer import validate_exact_dataset_contract
from tools import audit_pilot_foundation_source_support as support
from tools import build_b0_feasibility_panels as b0
from tools import build_pilot_apron_capacity_review as capacity_review

SCHEMA = "terra_pilot_large_foundation_review_v1"
SPLIT = "public_train"
MAP_SIZE = 64
REVIEW_COUNT = 16
REFERENCE_COUNT = 8
RANK_NAMESPACE = f"{SCHEMA}\0large-tail\0"
REFERENCE_RANK_NAMESPACE = f"{SCHEMA}\0anchor-reference\0"

VOLUME_LOWER_INCLUSIVE = 328
VOLUME_UPPER_INCLUSIVE = 340
ANCHOR_VOLUME_LOWER_INCLUSIVE = 140
ANCHOR_VOLUME_UPPER_INCLUSIVE = 189
COMPACTNESS_LOWER_INCLUSIVE = 0.30
COMPACTNESS_UPPER_INCLUSIVE = 0.65
TARGET_AREA_FRACTION_LOWER = VOLUME_LOWER_INCLUSIVE / MAP_SIZE**2
TARGET_AREA_FRACTION_UPPER = VOLUME_UPPER_INCLUSIVE / MAP_SIZE**2

PRIMARY_CELL = "f_procedural_all_large_tail_v328_340_review"
VOLUME_TOKEN = "large_tail_v328_340"
CAPACITY_TOKEN = "all_around"
DATASET_STRATUM = "S2_large_foundation_visual_candidate"
SOURCE_REGISTRY_RELATIVE_PATH = "../source_registry.jsonl"
GRAPHIC_HASH_PORTABILITY = "local_runtime_only_not_benchmark_identity"
STATIC_STATUS = "pending_exact_static_and_witness"
EXACT_LOADER_STATUS = "exact_loader_format_valid_static_pending"
MINIMUM_ALL_AROUND_CAPACITY_RATIO = 10.0
SUMMARY_NON_ADMISSION_LABELS = {
    "exact_loader_format_valid": True,
    "canonical_benchmark_format_admitted": False,
    "static_status": STATIC_STATUS,
    "witness_status": "not_run",
    "s2_foundation_volume_gate_complete": False,
    "work_volume_condition_admitted": False,
    "source_provenance_matched_to_current_osm_anchors": False,
    "causal_size_comparison_claimed": False,
    "all_maps_all_around_dump": True,
    "all_maps_obstacle_free": True,
    "held_out_or_sealed_selection_performed": False,
    "policy_outcomes_consulted": False,
    "constructor_outcomes_consulted": False,
}
RECORD_NON_ADMISSION_LABELS = dict(SUMMARY_NON_ADMISSION_LABELS)

EXPECTED_B0A_FILES_MANIFEST_SHA256 = (
    "89a5b5325e4e6872f7899b087ac5d0a8cd444dac30315feee4f342f8e532a347"
)
EXPECTED_CAPACITY_FILES_MANIFEST_SHA256 = (
    "03b5af27c3fc2acea05b6ecb001e4dcb7cdcbfe3f2677efa64ad2a6e3e7df80f"
)
EXPECTED_PROCEDURAL_ROW_COUNT = 20_000

MAP_COLORS = ListedColormap(["#f3e6c3", "#ef8b23", "#4daa6b", "#111111"])
TARGET_COLORS = ListedColormap(["#f3e6c3", "#ef8b23"])
CODE_DEPENDENCIES = (
    Path(__file__).resolve().parent / "audit_pilot_foundation_source_support.py",
    Path(__file__).resolve().parent / "build_b0_feasibility_panels.py",
    Path(__file__).resolve().parent / "build_pilot_apron_capacity_review.py",
)
GENERATOR_METADATA_FIELDS = (
    "generator_draw_count",
    "generator_rejections_before_proposal",
    "main_center_y_cells",
    "main_center_x_cells",
    "main_angle_degrees",
    "main_length_cells",
    "main_width_cells",
    "wing_count",
)


def sha256_file(path: Path) -> str:
    return capacity_review.sha256_file(path)


def sha256_text_lines(lines: Iterable[str]) -> str:
    return capacity_review.sha256_text_lines(lines)


def write_json(path: Path, payload: Any) -> None:
    capacity_review.write_json(path, payload)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    capacity_review.write_jsonl(path, rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return capacity_review.load_jsonl(path)


def write_file_manifest(root: Path) -> None:
    capacity_review.write_file_manifest(root)


def verify_file_manifest(root: Path) -> None:
    capacity_review.verify_file_manifest(root)


def _rank(namespace: str, canonical_hash: str) -> str:
    return hashlib.sha256(f"{namespace}{canonical_hash}".encode()).hexdigest()


def _audit_row_sha256(row: dict[str, Any]) -> str:
    payload = (json.dumps(row, sort_keys=True) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _expected_trench_metadata(map_id: str) -> dict[str, Any]:
    return {
        "map_id": map_id,
        "family": "foundation",
        "primary_cell": PRIMARY_CELL,
        "geometry": "foundation_procedural",
        "topology": None,
        "axes_ABC": [],
        "foundation_border_axes_ABC": [],
    }


def _expected_manifest_row(
    slot_index: int,
    *,
    map_id: str,
    source_id: str,
) -> dict[str, Any]:
    return {
        "slot_index": slot_index,
        "map_id": map_id,
        "source_id": source_id,
        "split": SPLIT,
        "family": "foundation",
        "stratum": DATASET_STRATUM,
        "primary_cell": PRIMARY_CELL,
        "slot_weight": 1.0,
        "identity_slot_multiplicity": 1,
    }


def _expected_dataset_document(
    record_count: int,
    *,
    source_registry_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "terra_exact_map_dataset_v1",
        "slot_count": record_count,
        "unique_identity_count": record_count,
        "shape": [MAP_SIZE, MAP_SIZE],
        "distance_metric": "8_connected_cardinal_1_diagonal_sqrt2",
        "distance_normalization": "per_map_max_to_1",
        "accepted_dump_contract": "exact_visible_dump_v1",
        "scenario_identity_contract": "terra_legacy_map_id_v0",
        "minimum_dump_capacity_ratio": MINIMUM_ALL_AROUND_CAPACITY_RATIO,
        "source_registry": SOURCE_REGISTRY_RELATIVE_PATH,
        "source_registry_sha256": source_registry_sha256,
    }


def _expected_registry_row(
    *,
    map_id: str,
    audit_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "map_id": map_id,
        "source_id": audit_row["source_id"],
        "source_group_id": audit_row["source_group_id"],
        "split": SPLIT,
        "canonical_dig_sha256": audit_row["canonical_dig_sha256"],
        "procedural_sample_index": audit_row["sample_index"],
        "generator_revision": audit_row["generator_revision"],
        "audit_row_sha256": _audit_row_sha256(audit_row),
    }


def _expected_identity_record(
    *,
    map_id: str,
    audit_row: dict[str, Any],
    dig: np.ndarray,
    target: np.ndarray,
    initial_agent_state_sha256: str,
    capacity: dict[str, Any],
    separation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "map_id": map_id,
        "source_id": audit_row["source_id"],
        "source_group_id": audit_row["source_group_id"],
        "split": SPLIT,
        "family": "foundation",
        "geometry": "foundation_procedural",
        "primary_cell": PRIMARY_CELL,
        "volume_token": VOLUME_TOKEN,
        "capacity_token": CAPACITY_TOKEN,
        "required_volume": int(dig.sum()),
        "required_volume_unit": "unit_depth_cell_volume",
        "target_area_fraction": float(dig.mean()),
        "perimeter_4_edges": int(audit_row["perimeter_4_edges"]),
        "compactness_4pi_area_over_perimeter_squared": float(
            audit_row["compactness_4pi_area_over_perimeter_squared"]
        ),
        "procedural_sample_index": int(audit_row["sample_index"]),
        "generator_revision": audit_row["generator_revision"],
        "generator_metadata": {
            field: audit_row[field] for field in GENERATOR_METADATA_FIELDS
        },
        "canonical_dig_sha256": audit_row["canonical_dig_sha256"],
        "dig_identity_sha256": b0.sha256_array(dig.astype(np.uint8)),
        "target_identity_sha256": b0.sha256_array(target),
        "initial_agent_state_sha256": initial_agent_state_sha256,
        "capacity": capacity,
        "separation": separation,
        "dump_layout": "all_around",
        "obstacle_layout": "none",
        **RECORD_NON_ADMISSION_LABELS,
    }


def _live_dump_distance_statistics(
    dig: np.ndarray,
    dump: np.ndarray,
    *,
    env_receipt: dict[str, Any],
    environment_protocol: dict[str, Any],
) -> dict[str, float]:
    map_receipt = environment_protocol.get("map")
    expected_map_receipt = {
        key: env_receipt[key]
        for key in (
            "edge_length_px",
            "edge_length_m",
            "tile_size_m_derived_float64",
            "tile_size_m_runtime_float32",
        )
    }
    if map_receipt != expected_map_receipt:
        raise RuntimeError(
            "Frozen benchmark and environment protocol map geometry receipts "
            "disagree."
        )
    tile_size_m = float(env_receipt["edge_length_m"]) / int(
        env_receipt["edge_length_px"]
    )
    if tile_size_m != float(env_receipt["tile_size_m_derived_float64"]):
        raise RuntimeError("Frozen benchmark tile-size derivation disagrees.")

    legacy = b0.dump_distance_statistics(dig, dump)
    tiles = {
        statistic: float(legacy[f"{statistic}_tiles"])
        for statistic in ("p50", "p95", "max")
    }
    return {
        "p50_tiles": tiles["p50"],
        "p95_tiles": tiles["p95"],
        "max_tiles": tiles["max"],
        "p50_metres": tiles["p50"] * tile_size_m,
        "p95_metres": tiles["p95"] * tile_size_m,
        "max_metres": tiles["max"] * tile_size_m,
    }


def _builder_receipt() -> dict[str, str]:
    builder = Path(__file__).resolve()
    repository = builder.parents[1]
    return {
        "path": builder.relative_to(repository).as_posix(),
        "sha256": sha256_file(builder),
    }


def _runtime_receipt() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _in_compactness_support(row: dict[str, Any]) -> bool:
    compactness = float(row["compactness_4pi_area_over_perimeter_squared"])
    return COMPACTNESS_LOWER_INCLUSIVE <= compactness <= COMPACTNESS_UPPER_INCLUSIVE


def _in_large_tail_support(row: dict[str, Any]) -> bool:
    volume = int(row["required_volume"])
    return (
        VOLUME_LOWER_INCLUSIVE <= volume <= VOLUME_UPPER_INCLUSIVE
        and _in_compactness_support(row)
    )


def _in_anchor_support(row: dict[str, Any]) -> bool:
    volume = int(row["required_volume"])
    return (
        ANCHOR_VOLUME_LOWER_INCLUSIVE <= volume <= ANCHOR_VOLUME_UPPER_INCLUSIVE
        and _in_compactness_support(row)
    )


def _foundation_hashes_from_b0a(b0a_bank: Path) -> tuple[set[str], dict[str, Any]]:
    """Return canonical target hashes used by every B0a foundation row."""

    manifest_path = b0a_bank / "files.sha256"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != EXPECTED_B0A_FILES_MANIFEST_SHA256:
        raise RuntimeError(
            "B0a root manifest changed: "
            f"{manifest_sha256} != {EXPECTED_B0A_FILES_MANIFEST_SHA256}."
        )
    verify_file_manifest(b0a_bank)
    identities = load_jsonl(b0a_bank / "identities.jsonl")
    provenance = json.loads((b0a_bank / "provenance.json").read_text())
    identities_sha256 = sha256_file(b0a_bank / "identities.jsonl")
    if provenance.get("identity_manifest_sha256") != identities_sha256:
        raise RuntimeError("B0a identities no longer match provenance.")

    hashes: set[str] = set()
    identity_count = 0
    manifests: dict[Path, dict[str, dict[str, Any]]] = {}
    for row in identities:
        if row.get("family") != "foundation":
            continue
        split = row.get("split")
        cell = row.get("primary_cell")
        map_id = row.get("map_id")
        if not all(isinstance(value, str) and value for value in (split, cell, map_id)):
            raise RuntimeError(f"Malformed B0a foundation identity: {row}")
        dataset = b0a_bank / "cells" / split / cell
        if dataset not in manifests:
            manifests[dataset] = {
                item["map_id"]: item for item in load_jsonl(dataset / "manifest.jsonl")
            }
        manifest_row = manifests[dataset].get(map_id)
        if manifest_row is None:
            raise RuntimeError(f"B0a dataset has no manifest row for {map_id}.")
        target_path = dataset / "images" / f"img_{int(manifest_row['slot_index'])}.npy"
        target = np.load(target_path, allow_pickle=False)
        if b0.sha256_array(target) != row.get("target_identity_sha256"):
            raise RuntimeError(f"B0a target identity changed for {map_id}.")
        hashes.add(support.canonical_dig_sha256(target < 0))
        identity_count += 1
    if identity_count == 0:
        raise RuntimeError("B0a contains no foundation identities.")
    return hashes, {
        "files_manifest_sha256": manifest_sha256,
        "identities_sha256": identities_sha256,
        "foundation_identity_count": identity_count,
        "foundation_canonical_dig_count": len(hashes),
        "foundation_canonical_dig_set_sha256": sha256_text_lines(sorted(hashes)),
    }


def _capacity_review_hashes(
    capacity_artifact: Path,
) -> tuple[set[str], dict[str, Any]]:
    manifest_sha256 = sha256_file(capacity_artifact / "files.sha256")
    if manifest_sha256 != EXPECTED_CAPACITY_FILES_MANIFEST_SHA256:
        raise RuntimeError(
            "Capacity-review root manifest changed: "
            f"{manifest_sha256} != {EXPECTED_CAPACITY_FILES_MANIFEST_SHA256}."
        )
    verification = capacity_review.verify_artifact(capacity_artifact)
    rows = load_jsonl(capacity_artifact / "source_registry.jsonl")
    hashes = {str(row["canonical_dig_sha256"]) for row in rows}
    return hashes, {
        "files_manifest_sha256": manifest_sha256,
        "verification": verification,
        "canonical_dig_count": len(hashes),
        "canonical_dig_set_sha256": sha256_text_lines(sorted(hashes)),
    }


def _support_receipt(
    source_support: Path,
    generator_root: Path,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    receipt = capacity_review._source_support_receipt(source_support)
    summary = json.loads((source_support / "support_summary.json").read_text())
    observed_generator_hashes = support.generator_hashes(generator_root)
    if observed_generator_hashes != summary.get("inputs", {}).get("generator_files"):
        raise RuntimeError("Tracked generator files changed after the source audit.")
    rows = load_jsonl(source_support / "procedural_samples.jsonl")
    if len(rows) != EXPECTED_PROCEDURAL_ROW_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PROCEDURAL_ROW_COUNT} procedural audit rows, "
            f"found {len(rows)}."
        )
    matched = load_jsonl(source_support / "matched_train_pairs.jsonl")
    matched_hashes = {
        str(row["procedural_source_group_id"]).removeprefix("procedural:")
        for row in matched
    }
    if any(len(value) != 64 for value in matched_hashes):
        raise RuntimeError("Malformed matched procedural source identity.")
    receipt.update(
        {
            "procedural_samples_sha256": sha256_file(
                source_support / "procedural_samples.jsonl"
            ),
            "procedural_sample_count": len(rows),
            "matched_train_pairs_sha256": sha256_file(
                source_support / "matched_train_pairs.jsonl"
            ),
            "matched_procedural_source_count": len(matched_hashes),
            "generator_files": observed_generator_hashes,
        }
    )
    return rows, matched_hashes, receipt


def select_large_tail_sources(
    rows: list[dict[str, Any]],
    excluded_hashes: set[str],
    *,
    review_count: int = REVIEW_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select only from the fixed train-only procedural proposal stream."""

    if review_count < 1:
        raise ValueError("review_count must be positive.")
    if len(rows) != len({int(row["sample_index"]) for row in rows}):
        raise RuntimeError("Procedural support rows contain duplicate sample indices.")
    for row in rows:
        if (
            row.get("audit_partition") != "audit_train_candidate"
            or row.get("source_family") != "procedural"
            or row.get("generator_revision") != support.PROCEDURAL_GENERATOR_REVISION
        ):
            raise RuntimeError("Procedural support row left its train-only contract.")

    unique_candidates: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not _in_large_tail_support(row):
            continue
        if int(row["component_count_4"]) != 1 or int(row["hole_cells"]) != 0:
            raise RuntimeError("Large-tail support includes an invalid foundation.")
        canonical_hash = str(row["canonical_dig_sha256"])
        current = unique_candidates.get(canonical_hash)
        if current is None or int(row["sample_index"]) < int(current["sample_index"]):
            unique_candidates[canonical_hash] = row

    fresh = [
        row
        for canonical_hash, row in unique_candidates.items()
        if canonical_hash not in excluded_hashes
    ]
    fresh.sort(
        key=lambda row: (
            _rank(RANK_NAMESPACE, str(row["canonical_dig_sha256"])),
            int(row["sample_index"]),
        )
    )
    if len(fresh) < review_count:
        raise RuntimeError(
            f"Need {review_count} fresh large-tail sources, found {len(fresh)}."
        )
    selected = fresh[:review_count]

    anchor_unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not _in_anchor_support(row):
            continue
        canonical_hash = str(row["canonical_dig_sha256"])
        current = anchor_unique.get(canonical_hash)
        if current is None or int(row["sample_index"]) < int(current["sample_index"]):
            anchor_unique[canonical_hash] = row
    references = sorted(
        (
            row
            for canonical_hash, row in anchor_unique.items()
            if canonical_hash not in excluded_hashes
        ),
        key=lambda row: (
            _rank(REFERENCE_RANK_NAMESPACE, str(row["canonical_dig_sha256"])),
            int(row["sample_index"]),
        ),
    )[:REFERENCE_COUNT]
    if len(references) != REFERENCE_COUNT:
        raise RuntimeError("Not enough train-only anchor references for the gallery.")

    selected_volumes = [int(row["required_volume"]) for row in selected]
    return (
        selected,
        references,
        {
            "rank_namespace": RANK_NAMESPACE,
            "rank_definition": "sha256(rank_namespace + canonical_dig_sha256)",
            "selection_split": SPLIT,
            "procedural_audit_row_count": len(rows),
            "large_tail_unique_support_count_before_exclusion": len(unique_candidates),
            "large_tail_excluded_overlap_count": len(unique_candidates) - len(fresh),
            "large_tail_fresh_support_count": len(fresh),
            "selected_count": len(selected),
            "selected_required_volume": {
                "min": min(selected_volumes),
                "median": float(np.median(selected_volumes)),
                "max": max(selected_volumes),
            },
            "selected_source_set_sha256": sha256_text_lines(
                str(row["source_group_id"]) for row in selected
            ),
            "reference_count": len(references),
            "reference_rank_namespace": REFERENCE_RANK_NAMESPACE,
            "reference_source_set_sha256": sha256_text_lines(
                str(row["source_group_id"]) for row in references
            ),
            "held_out_or_sealed_selection_performed": False,
            "policy_outcomes_consulted": False,
            "constructor_outcomes_consulted": False,
        },
    )


def _resolve_frozen_review_inputs(
    *,
    source_support: Path,
    generator_root: Path,
    b0a_bank: Path,
    capacity_artifact: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    Any,
]:
    """Recompute the complete train-only selection from frozen source inputs."""

    source_support = source_support.resolve()
    generator_root = generator_root.resolve()
    b0a_bank = b0a_bank.resolve()
    capacity_artifact = capacity_artifact.resolve()
    rows, matched_hashes, source_receipt = _support_receipt(
        source_support,
        generator_root,
    )
    b0a_hashes, b0a_receipt = _foundation_hashes_from_b0a(b0a_bank)
    capacity_hashes, capacity_receipt = _capacity_review_hashes(capacity_artifact)
    excluded = matched_hashes | b0a_hashes | capacity_hashes
    selected, references, selection_receipt = select_large_tail_sources(
        rows,
        excluded,
    )
    input_receipts = {
        "source_support": source_receipt,
        "b0a_bank": b0a_receipt,
        "capacity_review": capacity_receipt,
        "excluded_canonical_dig_count": len(excluded),
        "excluded_canonical_dig_set_sha256": sha256_text_lines(sorted(excluded)),
    }
    return (
        selected,
        references,
        selection_receipt,
        input_receipts,
        support.load_base_generator(generator_root),
    )


def _regenerate_mask(base_generator: Any, row: dict[str, Any]) -> np.ndarray:
    sample_index = int(row["sample_index"])
    dig, metadata = support.sample_procedural_foundation(
        base_generator,
        support.proposal_rng(sample_index),
    )
    canonical_hash = support.canonical_dig_sha256(dig)
    if canonical_hash != row["canonical_dig_sha256"]:
        raise RuntimeError(f"Procedural source changed at sample {sample_index}.")
    metrics = support.foundation_metrics(dig)
    for field in (
        "required_volume",
        "perimeter_4_edges",
        "compactness_4pi_area_over_perimeter_squared",
        "component_count_4",
        "hole_cells",
    ):
        if metrics[field] != row[field]:
            raise RuntimeError(
                f"Procedural source metric changed at sample {sample_index}: {field}."
            )
    stored_metadata = row.get("generator_metadata", row)
    for field, value in metadata.items():
        if stored_metadata.get(field) != value:
            raise RuntimeError(
                f"Procedural generator receipt changed at sample {sample_index}: "
                f"{field}."
            )
    return np.asarray(dig, dtype=np.bool_)


def _map_code(target: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    code = np.zeros(target.shape, dtype=np.uint8)
    code[target < 0] = 1
    code[target > 0] = 2
    code[occupancy] = 3
    return code


def _write_dataset(
    directory: Path,
    records: list[dict[str, Any]],
    samples: dict[str, dict[str, np.ndarray]],
    source_registry: Path,
) -> None:
    for name in (
        "images",
        "occupancy",
        "dumpability",
        "actions",
        "distance",
        "metadata",
    ):
        (directory / name).mkdir(parents=True, exist_ok=False)
    manifest = []
    for slot, record in enumerate(records, start=1):
        sample = samples[record["map_id"]]
        np.save(directory / "images" / f"img_{slot}.npy", sample["target"])
        np.save(directory / "occupancy" / f"img_{slot}.npy", sample["occupancy"])
        np.save(
            directory / "dumpability" / f"img_{slot}.npy",
            sample["dumpability"],
        )
        np.save(directory / "actions" / f"img_{slot}.npy", sample["action"])
        np.save(directory / "distance" / f"img_{slot}.npy", sample["distance"])
        write_json(
            directory / "metadata" / f"trench_{slot}.json",
            _expected_trench_metadata(record["map_id"]),
        )
        manifest.append(
            _expected_manifest_row(
                slot,
                map_id=record["map_id"],
                source_id=record["source_id"],
            )
        )
    write_jsonl(directory / "manifest.jsonl", manifest)
    write_json(
        directory / "dataset.json",
        _expected_dataset_document(
            len(records),
            source_registry_sha256=sha256_file(source_registry),
        ),
    )


def _render_example(
    path: Path,
    record: dict[str, Any],
    sample: dict[str, np.ndarray],
) -> None:
    dig = sample["target"] < 0
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(
        dig.astype(np.uint8),
        cmap=TARGET_COLORS,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[0].set_title("Dig target only")
    axes[1].imshow(
        _map_code(sample["target"], sample["occupancy"]),
        cmap=MAP_COLORS,
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    axes[1].set_title(
        "All-around dump\n"
        f"{record['capacity']['single_layer_capacity_ratio']:.2f}x capacity"
    )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"{record['map_id']} — {record['required_volume']} cells "
        f"({100.0 * record['target_area_fraction']:.2f}% of site)",
        fontsize=10,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _render_target_gallery(
    path: Path,
    records: list[dict[str, Any]],
    samples: dict[str, dict[str, np.ndarray]],
) -> None:
    columns = 4
    rows = math.ceil(len(records) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, record in zip(axes.ravel(), records):
        dig = samples[record["map_id"]]["target"] < 0
        axis.imshow(
            dig.astype(np.uint8),
            cmap=TARGET_COLORS,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.set_title(
            f"{record['map_id'].rsplit('-', 1)[-1]}\n"
            f"{record['required_volume']} cells, "
            f"{100.0 * record['target_area_fraction']:.2f}%",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(records) :]:
        axis.axis("off")
    figure.suptitle(
        "Narrow procedural large-tail targets — v328_340 (not broad coverage)",
        fontsize=13,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_full_map_gallery(
    path: Path,
    records: list[dict[str, Any]],
    samples: dict[str, dict[str, np.ndarray]],
) -> None:
    columns = 4
    rows = math.ceil(len(records) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, record in zip(axes.ravel(), records):
        sample = samples[record["map_id"]]
        axis.imshow(
            _map_code(sample["target"], sample["occupancy"]),
            cmap=MAP_COLORS,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        axis.set_title(
            f"{record['required_volume']} cells\n"
            f"all-around, {record['capacity']['single_layer_capacity_ratio']:.2f}x",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(records) :]:
        axis.axis("off")
    figure.suptitle(
        "Large-tail review maps — orange dig, green allowed dump, no obstacles",
        fontsize=13,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_audit_comparison(
    path: Path,
    references: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    reference_masks: list[np.ndarray],
    selected_masks: list[np.ndarray],
) -> None:
    figure = plt.figure(figsize=(12, 12), constrained_layout=True)
    subfigures = figure.subfigures(2, 1)
    groups = (
        (
            subfigures[0],
            "Anchor reference — v140_189 (3.42-4.61% of site)",
            references,
            reference_masks,
        ),
        (
            subfigures[1],
            "Large procedural tail — v328_340 (8.01-8.30% of site)",
            selected[:REFERENCE_COUNT],
            selected_masks[:REFERENCE_COUNT],
        ),
    )
    for subfigure, title, rows, masks in groups:
        axes = subfigure.subplots(2, 4, squeeze=False)
        for axis, row, dig in zip(axes.ravel(), rows, masks):
            axis.imshow(
                dig.astype(np.uint8),
                cmap=TARGET_COLORS,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_title(
                f"{int(row['required_volume'])} cells — "
                f"{100.0 * int(row['required_volume']) / MAP_SIZE**2:.2f}%",
                fontsize=9,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        for axis in axes.ravel()[len(rows) :]:
            axis.axis("off")
        subfigure.suptitle(title, fontsize=12, fontweight="bold")
    figure.suptitle(
        "Same 64 x 64 site and procedural generator; footprint-only visual context",
        fontsize=14,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _code_dependency_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in CODE_DEPENDENCIES
    }


def _readme_text(record_count: int) -> str:
    return (
        "# Narrow large-foundation visual review\n\n"
        f"This train-only artifact contains {record_count} fresh procedural "
        "source groups in `v328_340`, or 8.01-8.30% of the 64 x 64 site. "
        "Every target uses all-around dumping, no obstacles, and no action "
        "restrictions so the review focuses on footprint and required volume.\n\n"
        "This is a narrow generator-tail candidate, not broad 8-12% coverage. "
        "It is not source-matched to the current OSM anchors, so it is not a "
        "causal size comparison. A true >=10% candidate remains pending.\n\n"
        "Status: **exact-loader format valid; exact Static validation and a "
        "450-step witness are pending.** The cell is not admitted to S2, is "
        "not a benchmark release, and does not authorize PPO.\n\n"
        "- `examples/`: target-only and full-map pair for every map.\n"
        "- `galleries/target_masks.png`: target footprint contact sheet.\n"
        "- `galleries/all_around_maps.png`: complete map contact sheet.\n"
        "- `galleries/anchor_vs_large_tail.png`: train-only visual context.\n"
        "- `dataset/`: exact-loader-compatible 16-map dataset.\n"
        "\nPNG hashes are reproducibility checks for the pinned local rendering "
        "runtime only; they are not benchmark map identities and are not "
        "claimed portable across Matplotlib or platform versions.\n"
    )


def _materialize_from_selected(
    *,
    selected: list[dict[str, Any]],
    references: list[dict[str, Any]],
    base_generator: Any,
    output: Path,
    input_receipts: dict[str, Any],
    selection_receipt: dict[str, Any],
    source_state_receipt: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if not selected or not references:
        raise ValueError("Selected and reference rows must be nonempty.")
    source_ids = [str(row["source_group_id"]) for row in selected]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Selected source groups must be unique.")
    output.mkdir(parents=True)

    terra_revision = source_state_receipt.get("terra_revision")
    if not isinstance(terra_revision, str) or not terra_revision:
        raise ValueError("source_state_receipt must contain a Terra revision.")
    env_config, env_receipt = frozen_benchmark_protocol()
    environment_protocol = frozen_environment_protocol(terra_revision)

    selected_masks = [_regenerate_mask(base_generator, row) for row in selected]
    reference_masks = [_regenerate_mask(base_generator, row) for row in references]
    records: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    samples: dict[str, dict[str, np.ndarray]] = {}

    for index, (source_row, dig) in enumerate(zip(selected, selected_masks)):
        if not _in_large_tail_support(source_row):
            raise RuntimeError("Selected source left the frozen large-tail band.")
        source_group_id = str(source_row["source_group_id"])
        map_id = f"large-foundation-{SPLIT}-{index:03d}"
        target = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
        target[dig] = -1
        target[~dig] = 1
        occupancy = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
        dumpability = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        action = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
        distance = capacity_review._geodesic_distance(
            target,
            occupancy.astype(np.bool_),
        )
        capacity = contained_dump_capacity_sanity_check(
            target,
            occupancy,
            dumpability,
            action,
            minimum_single_layer_ratio=MINIMUM_ALL_AROUND_CAPACITY_RATIO,
        )
        separation = _live_dump_distance_statistics(
            dig,
            ~dig,
            env_receipt=env_receipt,
            environment_protocol=environment_protocol,
        )

        agent, seed_receipt = sample_benchmark_initial_agent(
            release_id=BENCHMARK_RELEASE_ID,
            split=SPLIT,
            source_group_id=source_group_id,
            state_index=0,
            env_cfg=env_config,
            padding_mask=occupancy,
            action_map=action,
            dumpability_mask=dumpability,
        )
        validate_benchmark_initial_agent(
            agent,
            env_cfg=env_config,
            padding_mask=occupancy,
            action_map=action,
            dumpability_mask=dumpability,
        )
        states.append(
            {
                **seed_receipt,
                "map_id": map_id,
                "initial_agent_state": agent_to_record(agent),
            }
        )
        samples[map_id] = {
            "target": target,
            "occupancy": occupancy,
            "dumpability": dumpability,
            "action": action,
            "distance": distance,
        }
        records.append(
            _expected_identity_record(
                map_id=map_id,
                audit_row=source_row,
                dig=dig,
                target=target,
                initial_agent_state_sha256=seed_receipt["initial_agent_state_sha256"],
                capacity=capacity,
                separation=separation,
            )
        )

    records.sort(key=lambda row: row["map_id"])
    source_registry = output / "source_registry.jsonl"
    source_rows = {str(row["source_group_id"]): row for row in selected}
    write_jsonl(
        source_registry,
        [
            _expected_registry_row(
                map_id=record["map_id"],
                audit_row=source_rows[record["source_group_id"]],
            )
            for record in records
        ],
    )
    write_jsonl(output / "identities.jsonl", records)
    write_jsonl(output / "initial_states.jsonl", states)
    _write_dataset(output / "dataset", records, samples, source_registry)

    examples = output / "examples"
    galleries = output / "galleries"
    examples.mkdir()
    galleries.mkdir()
    for record in records:
        _render_example(
            examples / f"{record['map_id'].rsplit('-', 1)[-1]}.png",
            record,
            samples[record["map_id"]],
        )
    _render_target_gallery(galleries / "target_masks.png", records, samples)
    _render_full_map_gallery(galleries / "all_around_maps.png", records, samples)
    _render_audit_comparison(
        galleries / "anchor_vs_large_tail.png",
        references,
        selected,
        reference_masks,
        selected_masks,
    )

    volumes = [int(row["required_volume"]) for row in records]
    fractions = [float(row["target_area_fraction"]) for row in records]
    capacities = [
        float(row["capacity"]["single_layer_capacity_ratio"]) for row in records
    ]
    summary = {
        "schema": SCHEMA,
        "status": EXACT_LOADER_STATUS,
        **SUMMARY_NON_ADMISSION_LABELS,
        "map_count": len(records),
        "source_group_count": len(records),
        "split": SPLIT,
        "primary_cell": PRIMARY_CELL,
        "volume_band_inclusive": [
            VOLUME_LOWER_INCLUSIVE,
            VOLUME_UPPER_INCLUSIVE,
        ],
        "anchor_volume_band_inclusive": [
            ANCHOR_VOLUME_LOWER_INCLUSIVE,
            ANCHOR_VOLUME_UPPER_INCLUSIVE,
        ],
        "compactness_band_inclusive": [
            COMPACTNESS_LOWER_INCLUSIVE,
            COMPACTNESS_UPPER_INCLUSIVE,
        ],
        "required_volume": {
            "min": min(volumes),
            "median": float(np.median(volumes)),
            "max": max(volumes),
        },
        "target_area_fraction": {
            "min": min(fractions),
            "median": float(np.median(fractions)),
            "max": max(fractions),
        },
        "single_layer_capacity_ratio": {
            "min": min(capacities),
            "median": float(np.median(capacities)),
            "max": max(capacities),
        },
        "coverage_claim": "narrow_procedural_8.0_to_8.3_percent_tail_only",
        "broad_8_to_12_percent_coverage_claimed": False,
        "true_10_percent_or_larger_slice_pending": True,
        "graphic_hash_portability": GRAPHIC_HASH_PORTABILITY,
        "selection": selection_receipt,
        "environment_protocol_sha256": environment_protocol[
            "environment_protocol_sha256"
        ],
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "provenance.json",
        {
            "schema": SCHEMA,
            "builder": _builder_receipt(),
            "source_state": source_state_receipt,
            "code_dependencies": _code_dependency_hashes(),
            "environment_protocol": environment_protocol,
            "env_config_receipt": env_receipt,
            "inputs": input_receipts,
            "selection": selection_receipt,
            "source_registry_sha256": sha256_file(source_registry),
            "identities_sha256": sha256_file(output / "identities.jsonl"),
            "initial_states_sha256": sha256_file(output / "initial_states.jsonl"),
            "runtime": _runtime_receipt(),
            "graphic_hash_portability": GRAPHIC_HASH_PORTABILITY,
        },
    )
    (output / "README.md").write_text(_readme_text(len(records)))
    write_file_manifest(output)
    return summary


def build_artifact(
    *,
    source_support: Path,
    generator_root: Path,
    b0a_bank: Path,
    capacity_artifact: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()

    source_state_receipt = capacity_review._clean_source_receipt()
    (
        selected,
        references,
        selection_receipt,
        input_receipts,
        base_generator,
    ) = _resolve_frozen_review_inputs(
        source_support=source_support,
        generator_root=generator_root,
        b0a_bank=b0a_bank,
        capacity_artifact=capacity_artifact,
    )
    summary = _materialize_from_selected(
        selected=selected,
        references=references,
        base_generator=base_generator,
        output=output,
        input_receipts=input_receipts,
        selection_receipt=selection_receipt,
        source_state_receipt=source_state_receipt,
    )
    _verify_artifact_from_fixtures(
        output,
        base_generator=base_generator,
        selected=selected,
        references=references,
        input_receipts=input_receipts,
        selection_receipt=selection_receipt,
        expected_source_state_receipt=source_state_receipt,
        expected_count=len(selected),
    )
    return summary


def _file_hash_tree(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (root / "files.sha256").read_text().splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        entries[relative] = digest
    return entries


def _smoke_exact_loader(dataset: Path, expected_count: int) -> None:
    manifest, shape, minimum_ratio = validate_exact_dataset_contract(
        dataset,
        expected_count,
    )
    if (
        len(manifest) != expected_count
        or shape != (MAP_SIZE, MAP_SIZE)
        or minimum_ratio != MINIMUM_ALL_AROUND_CAPACITY_RATIO
    ):
        raise RuntimeError("Exact-loader contract changed.")
    with capacity_review._dataset_size(expected_count):
        loaded = load_maps_from_disk(
            str(dataset),
            require_trench_metadata=False,
            require_exact_contract=True,
        )
    if np.asarray(jax.device_get(loaded[0])).shape != (
        expected_count,
        MAP_SIZE,
        MAP_SIZE,
    ):
        raise RuntimeError("Exact loader returned an unexpected target shape.")


def _verify_artifact_from_fixtures(
    output: Path,
    *,
    base_generator: Any,
    selected: list[dict[str, Any]],
    references: list[dict[str, Any]],
    input_receipts: dict[str, Any],
    selection_receipt: dict[str, Any],
    expected_source_state_receipt: dict[str, Any],
    expected_count: int = REVIEW_COUNT,
) -> dict[str, Any]:
    output = output.resolve()
    verify_file_manifest(output)
    if expected_count < 1:
        raise ValueError("expected_count must be positive.")
    if len(selected) != expected_count or len(references) != REFERENCE_COUNT:
        raise RuntimeError("Recomputed selected/reference counts changed.")
    expected_selected_hash = sha256_text_lines(
        str(row["source_group_id"]) for row in selected
    )
    expected_reference_hash = sha256_text_lines(
        str(row["source_group_id"]) for row in references
    )
    if (
        selection_receipt.get("selected_count") != expected_count
        or selection_receipt.get("selected_source_set_sha256") != expected_selected_hash
        or selection_receipt.get("reference_count") != len(references)
        or selection_receipt.get("reference_source_set_sha256")
        != expected_reference_hash
        or selection_receipt.get("held_out_or_sealed_selection_performed") is not False
        or selection_receipt.get("policy_outcomes_consulted") is not False
        or selection_receipt.get("constructor_outcomes_consulted") is not False
    ):
        raise RuntimeError("Recomputed deterministic selection receipt changed.")
    empty_status_sha256 = hashlib.sha256(b"").hexdigest()
    if (
        set(expected_source_state_receipt)
        != {
            "terra_revision",
            "terra_worktree_clean",
            "git_status_porcelain_sha256",
        }
        or not isinstance(
            expected_source_state_receipt.get("terra_revision"),
            str,
        )
        or not expected_source_state_receipt["terra_revision"]
        or expected_source_state_receipt.get("terra_worktree_clean") is not True
        or expected_source_state_receipt.get("git_status_porcelain_sha256")
        != empty_status_sha256
    ):
        raise RuntimeError("Expected source-state receipt is not clean and complete.")

    with tempfile.TemporaryDirectory(prefix="terra-large-review-verify-") as temp:
        expected_output = Path(temp) / "expected"
        _materialize_from_selected(
            selected=selected,
            references=references,
            base_generator=base_generator,
            output=expected_output,
            input_receipts=input_receipts,
            selection_receipt=selection_receipt,
            source_state_receipt=expected_source_state_receipt,
        )
        expected_tree = _file_hash_tree(expected_output)
        observed_tree = _file_hash_tree(output)
        if observed_tree != expected_tree:
            changed = sorted(
                relative
                for relative in set(observed_tree) | set(expected_tree)
                if observed_tree.get(relative) != expected_tree.get(relative)
            )
            raise RuntimeError(
                "Artifact differs from deterministic rebuild: "
                + ", ".join(changed[:12])
            )
        if (output / "files.sha256").read_text() != (
            expected_output / "files.sha256"
        ).read_text():
            raise RuntimeError("Artifact manifest differs from deterministic rebuild.")
        _smoke_exact_loader(expected_output / "dataset", expected_count)
        _smoke_exact_loader(output / "dataset", expected_count)
    return {
        "status": "passed",
        "exact_loader_format_valid": True,
        "canonical_benchmark_format_admitted": False,
        "static_status": STATIC_STATUS,
        "map_count": expected_count,
        "file_manifest_sha256": sha256_file(output / "files.sha256"),
    }


def verify_artifact(
    output: Path,
    *,
    source_support: Path,
    generator_root: Path,
    b0a_bank: Path,
    capacity_artifact: Path,
) -> dict[str, Any]:
    """Verify against current frozen sources, never artifact-declared selection."""

    expected_source_state_receipt = capacity_review._clean_source_receipt()
    (
        selected,
        references,
        selection_receipt,
        input_receipts,
        base_generator,
    ) = _resolve_frozen_review_inputs(
        source_support=source_support,
        generator_root=generator_root,
        b0a_bank=b0a_bank,
        capacity_artifact=capacity_artifact,
    )
    return _verify_artifact_from_fixtures(
        output,
        base_generator=base_generator,
        selected=selected,
        references=references,
        input_receipts=input_receipts,
        selection_receipt=selection_receipt,
        expected_source_state_receipt=expected_source_state_receipt,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-support", type=Path, required=True)
    build.add_argument("--generator-root", type=Path, required=True)
    build.add_argument("--b0a-bank", type=Path, required=True)
    build.add_argument("--capacity-review", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--source-support", type=Path, required=True)
    verify.add_argument("--generator-root", type=Path, required=True)
    verify.add_argument("--b0a-bank", type=Path, required=True)
    verify.add_argument("--capacity-review", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "build":
        result = build_artifact(
            source_support=args.source_support,
            generator_root=args.generator_root,
            b0a_bank=args.b0a_bank,
            capacity_artifact=args.capacity_review,
            output=args.output,
        )
    else:
        result = verify_artifact(
            args.output,
            source_support=args.source_support,
            generator_root=args.generator_root,
            b0a_bank=args.b0a_bank,
            capacity_artifact=args.capacity_review,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
