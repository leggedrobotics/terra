#!/usr/bin/env python3
"""Audit train-only OSM/procedural support for the S1 foundation pilot.

The audit freezes one narrow source comparison. It partitions canonical OSM
source groups before measuring their distribution, inspects only the fixed
audit-train pool, samples one retuned procedural generator, and constructs
exact volume/perimeter pairs. It does not select held-out sources, build maps,
or modify the hash-pinned B0 builder.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Iterable

import numpy as np
import scipy
from scipy import ndimage as ndi

SCHEMA = "terra_pilot_foundation_source_support_v3"
SEED_NAMESPACE = "terra_pilot_foundation_source_support_v1"
RESERVE_PARTITION = "reserve_identity_and_eligibility_only"
TRAIN_SEED = 2_026_072_702
PROCEDURAL_STREAM_ID = 110
PROPOSAL_COUNT = 20_000
AUDIT_TRAIN_GROUP_COUNT = 256
MATCHED_PAIR_COUNT = 32
MINIMUM_EXACT_CANDIDATES = 2

VOLUME_LOWER_INCLUSIVE = 140
VOLUME_UPPER_INCLUSIVE = 189
COMPACTNESS_LOWER_INCLUSIVE = 0.30
COMPACTNESS_UPPER_INCLUSIVE = 0.65

PROCEDURAL_MAIN_LENGTH_RANGE = (13.0, 22.0)
PROCEDURAL_MAIN_WIDTH_RANGE = (7.0, 13.0)
PROCEDURAL_GENERATOR_REVISION = "foundation_source_match_v1"

EXPECTED_SOURCE_IMAGE_COUNT = 600
EXPECTED_FACTORY_ELIGIBLE_MASK_COUNT = 592
EXPECTED_CANONICAL_OSM_GROUP_COUNT = 591
EXPECTED_SOURCE_IMAGE_MANIFEST_SHA256 = (
    "08ba236244f1571ea1ac1bff1c2c7729f5a2b9395fe7cc8dccace9f6e9cd3c98"
)
EXPECTED_BASE_GENERATOR_SHA256 = (
    "94c92eec14b27885c2f9743602178de1cff307e2a41be13a6a88f791c75a6284"
)
EXPECTED_B0_BUILDER_SHA256 = (
    "3a1bb66798f6a4bfc1dc5b3515c5a4485eb9a28d6ffe7c9e8413a548492b79a9"
)

TOOLS_DIR = Path(__file__).resolve().parent
B0_BUILDER = TOOLS_DIR / "build_b0_feasibility_panels.py"
FOUR_CONNECTED = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=np.uint8,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def numeric_image_key(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"unexpected source image name: {path.name}")
    return int(parts[1])


def crop_mask(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("cannot crop an empty mask")
    lower = points.min(axis=0)
    upper = points.max(axis=0) + 1
    return mask[lower[0] : upper[0], lower[1] : upper[1]]


def canonical_dig_sha256(mask: np.ndarray) -> str:
    """Hash a translated mask modulo rotations and reflections."""

    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("canonical dig hashing requires a non-empty 2D mask")
    payloads = []
    for rotation in range(4):
        rotated = np.rot90(mask, rotation)
        for candidate in (rotated, np.fliplr(rotated)):
            cropped = np.ascontiguousarray(crop_mask(candidate), dtype=np.uint8)
            payloads.append(
                np.asarray(cropped.shape, dtype="<i8").tobytes() + cropped.tobytes()
            )
    return hashlib.sha256(min(payloads)).hexdigest()


def perimeter_4_edges(mask: np.ndarray) -> int:
    """Count exposed four-neighbour cell edges, including the outer border."""

    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("perimeter requires a non-empty 2D mask")
    padded = np.pad(mask, 1, constant_values=False)
    horizontal = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    vertical = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    return int(horizontal + vertical)


def foundation_metrics(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("foundation metrics require a non-empty 2D mask")
    points = np.argwhere(mask)
    required_volume = int(mask.sum())
    perimeter = perimeter_4_edges(mask)
    labels, component_count = ndi.label(mask, structure=FOUR_CONNECTED)
    del labels
    holes = int(ndi.binary_fill_holes(mask).sum() - required_volume)
    height, width = (points.max(axis=0) - points.min(axis=0) + 1).tolist()
    covariance = np.cov(points.astype(np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_index = int(np.argmax(eigenvalues))
    minor = float(max(eigenvalues.min(), np.finfo(np.float64).eps))
    major = float(eigenvalues[major_index])
    major_vector = eigenvectors[:, major_index]
    return {
        "required_volume": required_volume,
        "required_volume_unit": "unit_depth_cell_volume",
        "perimeter_4_edges": perimeter,
        "compactness_4pi_area_over_perimeter_squared": float(
            4.0 * math.pi * required_volume / perimeter**2
        ),
        "component_count_4": int(component_count),
        "hole_cells": holes,
        "bbox_height_cells": int(height),
        "bbox_width_cells": int(width),
        "bbox_aspect_ratio": float(max(height, width) / min(height, width)),
        "moment_aspect_ratio": float(math.sqrt(major / minor)),
        "moment_orientation_degrees": float(
            math.degrees(math.atan2(major_vector[0], major_vector[1])) % 180.0
        ),
    }


def in_frozen_support(metrics: dict[str, Any]) -> bool:
    return bool(
        VOLUME_LOWER_INCLUSIVE
        <= int(metrics["required_volume"])
        <= VOLUME_UPPER_INCLUSIVE
        and COMPACTNESS_LOWER_INCLUSIVE
        <= float(metrics["compactness_4pi_area_over_perimeter_squared"])
        <= COMPACTNESS_UPPER_INCLUSIVE
    )


def factory_source_eligible(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if not len(points):
        return False
    height, width = points.max(axis=0) - points.min(axis=0) + 1
    return bool(90 <= int(mask.sum()) <= 340 and max(height, width) <= 34)


def source_image_manifest(
    source_foundations: Path,
) -> tuple[list[Path], list[str], str]:
    image_root = source_foundations / "images"
    paths = sorted(image_root.glob("img_*.npy"), key=numeric_image_key)
    if len(paths) != EXPECTED_SOURCE_IMAGE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_SOURCE_IMAGE_COUNT} source images, found {len(paths)}"
        )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(source_foundations).as_posix()}"
        for path in paths
    ]
    payload = ("\n".join(lines) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SOURCE_IMAGE_MANIFEST_SHA256:
        raise RuntimeError(
            "source image manifest mismatch: "
            f"{digest} != {EXPECTED_SOURCE_IMAGE_MANIFEST_SHA256}"
        )
    return paths, lines, digest


def load_osm_source_groups(
    source_foundations: Path,
) -> tuple[list[dict[str, Any]], list[str], str, int]:
    paths, manifest_lines, manifest_sha256 = source_image_manifest(source_foundations)
    groups: dict[str, dict[str, Any]] = {}
    eligible_mask_count = 0
    for path in paths:
        target = np.load(path, allow_pickle=False)
        if target.shape != (64, 64):
            raise RuntimeError(f"{path}: expected shape (64, 64), got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise RuntimeError(f"{path}: target contains non-finite values")
        if not np.all(np.isin(target, (-1, 0, 1))):
            raise RuntimeError(f"{path}: target contains values outside {{-1, 0, 1}}")
        dig = np.asarray(target < 0, dtype=np.bool_)
        if not factory_source_eligible(dig):
            continue
        eligible_mask_count += 1
        canonical_hash = canonical_dig_sha256(dig)
        member = {
            "relative_path": path.relative_to(source_foundations).as_posix(),
            "file_sha256": sha256_file(path),
        }
        if canonical_hash not in groups:
            groups[canonical_hash] = {
                "source_family": "osm",
                "source_id": f"osm:{canonical_hash}",
                "source_group_id": f"osm:{canonical_hash}",
                "canonical_dig_sha256": canonical_hash,
                "representative_relative_path": member["relative_path"],
                "member_files": [member],
                "_dig": dig,
            }
        else:
            groups[canonical_hash]["member_files"].append(member)

    if eligible_mask_count != EXPECTED_FACTORY_ELIGIBLE_MASK_COUNT:
        raise RuntimeError(
            "factory-eligible source count mismatch: "
            f"{eligible_mask_count} != {EXPECTED_FACTORY_ELIGIBLE_MASK_COUNT}"
        )
    if len(groups) != EXPECTED_CANONICAL_OSM_GROUP_COUNT:
        raise RuntimeError(
            "canonical OSM source count mismatch: "
            f"{len(groups)} != {EXPECTED_CANONICAL_OSM_GROUP_COUNT}"
        )
    ordered = sorted(groups.values(), key=lambda row: row["source_group_id"])
    return ordered, manifest_lines, manifest_sha256, eligible_mask_count


def audit_rank(canonical_hash: str) -> str:
    return hashlib.sha256(f"{SEED_NAMESPACE}|{canonical_hash}".encode()).hexdigest()


def materialization_rank(canonical_hash: str) -> str:
    return hashlib.sha256(
        f"{SEED_NAMESPACE}|materialize|{canonical_hash}".encode()
    ).hexdigest()


def partition_source_groups(
    groups: list[dict[str, Any]],
    audit_train_count: int = AUDIT_TRAIN_GROUP_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if audit_train_count < 1 or len(groups) <= audit_train_count:
        raise ValueError("source pool must be larger than the audit-train partition")
    ordered = sorted(
        groups,
        key=lambda row: (
            audit_rank(str(row["canonical_dig_sha256"])),
            str(row["source_group_id"]),
        ),
    )
    return ordered[:audit_train_count], ordered[audit_train_count:]


def audit_osm_train_groups(
    groups: list[dict[str, Any]],
    metrics_fn: Callable[[np.ndarray], dict[str, Any]] = foundation_metrics,
) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        metrics = metrics_fn(group["_dig"])
        if metrics["component_count_4"] != 1:
            raise RuntimeError(
                f"{group['source_id']}: source foundation is not 4-connected"
            )
        if metrics["hole_cells"] != 0:
            raise RuntimeError(
                f"{group['source_id']}: source foundation contains holes"
            )
        rows.append(
            {
                "source_family": "osm",
                "source_id": group["source_id"],
                "source_group_id": group["source_group_id"],
                "canonical_dig_sha256": group["canonical_dig_sha256"],
                "representative_relative_path": group["representative_relative_path"],
                "audit_partition": "audit_train",
                **metrics,
                "in_frozen_support": in_frozen_support(metrics),
            }
        )
    return rows


def proposal_rng(sample_index: int) -> np.random.Generator:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    sequence = np.random.SeedSequence([TRAIN_SEED, PROCEDURAL_STREAM_ID, sample_index])
    return np.random.default_rng(sequence)


def sample_procedural_foundation(
    base_generator,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    rejections = Counter(
        {
            "required_volume_outside_generator_bounds": 0,
            "margin_violation": 0,
            "not_4_connected": 0,
            "contains_holes": 0,
        }
    )
    for attempt in range(80):
        center = (
            float(rng.uniform(27.0, 37.0)),
            float(rng.uniform(27.0, 37.0)),
        )
        angle = float(rng.choice(np.deg2rad(np.arange(0, 180, 15))))
        length = float(rng.uniform(*PROCEDURAL_MAIN_LENGTH_RANGE))
        width = float(rng.uniform(*PROCEDURAL_MAIN_WIDTH_RANGE))
        dig = base_generator.rotated_rectangle(center, length, width, angle)

        wing_count = int(rng.integers(1, 4))
        for _ in range(wing_count):
            along = float(rng.uniform(-0.35, 0.35) * length)
            across = float(rng.choice([-1, 1]) * rng.uniform(0.25, 0.55) * width)
            dx = along * math.cos(angle) - across * math.sin(angle)
            dy = along * math.sin(angle) + across * math.cos(angle)
            wing_center = (center[0] + dy, center[1] + dx)
            wing = base_generator.rotated_rectangle(
                wing_center,
                float(rng.uniform(6.0, 13.0)),
                float(rng.uniform(5.0, 10.0)),
                angle + float(rng.choice([0.0, math.pi / 2.0])),
            )
            dig |= wing

        dig = ndi.binary_closing(
            dig,
            structure=base_generator.binary_disk(1),
        )
        dig = base_generator.largest_component(dig)
        points = np.argwhere(dig)
        required_volume = int(dig.sum())
        if not 110 <= required_volume <= 340:
            rejections["required_volume_outside_generator_bounds"] += 1
            continue
        if (
            points[:, 0].min() < 10
            or points[:, 1].min() < 10
            or points[:, 0].max() > 53
            or points[:, 1].max() > 53
        ):
            rejections["margin_violation"] += 1
            continue
        _, component_count = ndi.label(dig, structure=FOUR_CONNECTED)
        if component_count != 1:
            rejections["not_4_connected"] += 1
            continue
        if int(ndi.binary_fill_holes(dig).sum() - required_volume) != 0:
            rejections["contains_holes"] += 1
            continue
        return dig, {
            "generator_draw_count": attempt + 1,
            "generator_rejections_before_proposal": dict(rejections),
            "main_center_y_cells": center[0],
            "main_center_x_cells": center[1],
            "main_angle_degrees": float(math.degrees(angle)),
            "main_length_cells": length,
            "main_width_cells": width,
            "wing_count": wing_count,
        }
    raise RuntimeError("could not generate a valid procedural foundation in 80 draws")


def sample_procedural_rows(
    base_generator,
    proposal_count: int = PROPOSAL_COUNT,
) -> list[dict[str, Any]]:
    if proposal_count < 1:
        raise ValueError("proposal_count must be positive")
    rows = []
    for sample_index in range(proposal_count):
        dig, metadata = sample_procedural_foundation(
            base_generator,
            proposal_rng(sample_index),
        )
        canonical_hash = canonical_dig_sha256(dig)
        metrics = foundation_metrics(dig)
        rows.append(
            {
                "source_family": "procedural",
                "source_id": f"procedural:{canonical_hash}",
                "source_group_id": f"procedural:{canonical_hash}",
                "canonical_dig_sha256": canonical_hash,
                "audit_partition": "audit_train_candidate",
                "sample_index": sample_index,
                "seed": TRAIN_SEED,
                "seed_stream_id": PROCEDURAL_STREAM_ID,
                "generator_revision": PROCEDURAL_GENERATOR_REVISION,
                **metadata,
                **metrics,
                "in_frozen_support": in_frozen_support(metrics),
            }
        )
    return rows


def exact_match_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["required_volume"]), int(row["perimeter_4_edges"])


def exact_candidate_index(
    procedural_rows: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    unique: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in procedural_rows:
        if not row["in_frozen_support"]:
            continue
        key = exact_match_key(row)
        current = unique[key].get(row["source_id"])
        if current is None or int(row["sample_index"]) < int(current["sample_index"]):
            unique[key][row["source_id"]] = row
    return {
        key: sorted(
            rows.values(),
            key=lambda row: (str(row["source_id"]), int(row["sample_index"])),
        )
        for key, rows in unique.items()
    }


def match_exact_pairs(
    osm_rows: list[dict[str, Any]],
    procedural_rows: list[dict[str, Any]],
    pair_count: int = MATCHED_PAIR_COUNT,
    minimum_candidates: int = MINIMUM_EXACT_CANDIDATES,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if pair_count < 1 or minimum_candidates < 1:
        raise ValueError("pair_count and minimum_candidates must be positive")
    candidates_by_key = exact_candidate_index(procedural_rows)
    candidate_counts = {
        str(row["source_id"]): len(candidates_by_key.get(exact_match_key(row), []))
        for row in osm_rows
        if row["in_frozen_support"]
    }
    eligible = [
        row
        for row in osm_rows
        if row["in_frozen_support"]
        and candidate_counts[str(row["source_id"])] >= minimum_candidates
    ]
    eligible.sort(
        key=lambda row: (
            materialization_rank(str(row["canonical_dig_sha256"])),
            str(row["source_id"]),
        )
    )

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    consumed_per_key: Counter[tuple[int, int]] = Counter()
    for osm in eligible:
        key = exact_match_key(osm)
        candidates = candidates_by_key[key]
        candidate_index = consumed_per_key[key]
        if candidate_index >= len(candidates):
            continue
        procedural = candidates[candidate_index]
        consumed_per_key[key] += 1
        selected.append((osm, procedural))
        if len(selected) == pair_count:
            break
    if len(selected) != pair_count:
        raise RuntimeError(
            f"only {len(selected)} unique exact pairs available, expected {pair_count}"
        )

    pairs = []
    for pair_index, (osm, procedural) in enumerate(selected):
        if osm["source_id"] == procedural["source_id"]:
            raise RuntimeError("OSM and procedural source IDs must remain distinct")
        if osm["canonical_dig_sha256"] == procedural["canonical_dig_sha256"]:
            raise RuntimeError(
                "matched OSM/procedural sources share a canonical dig raster"
            )
        if exact_match_key(osm) != exact_match_key(procedural):
            raise RuntimeError(
                "matched source pair does not share exact volume/perimeter"
            )
        osm_compactness = float(osm["compactness_4pi_area_over_perimeter_squared"])
        procedural_compactness = float(
            procedural["compactness_4pi_area_over_perimeter_squared"]
        )
        if osm_compactness != procedural_compactness:
            raise RuntimeError("exact volume/perimeter pair has unequal compactness")
        match_group_id = f"foundation-source-match:train:{pair_index:03d}"
        if match_group_id in {osm["source_id"], procedural["source_id"]}:
            raise RuntimeError("match_group_id must be separate from source IDs")
        pairs.append(
            {
                "match_group_id": match_group_id,
                "osm_source_id": osm["source_id"],
                "osm_source_group_id": osm["source_group_id"],
                "osm_representative_relative_path": osm["representative_relative_path"],
                "procedural_source_id": procedural["source_id"],
                "procedural_source_group_id": procedural["source_group_id"],
                "procedural_sample_index": int(procedural["sample_index"]),
                "required_volume": int(osm["required_volume"]),
                "required_volume_unit": osm["required_volume_unit"],
                "perimeter_4_edges": int(osm["perimeter_4_edges"]),
                "compactness_4pi_area_over_perimeter_squared": osm_compactness,
                "osm_exact_candidate_count": candidate_counts[str(osm["source_id"])],
            }
        )
    if len({row["osm_source_id"] for row in pairs}) != pair_count:
        raise RuntimeError("matched OSM sources are not unique")
    if len({row["procedural_source_id"] for row in pairs}) != pair_count:
        raise RuntimeError("matched procedural sources are not unique")
    return pairs, candidate_counts


def source_registry_rows(
    audit_train: list[dict[str, Any]],
    reserve: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for partition, groups in (
        ("audit_train", audit_train),
        (RESERVE_PARTITION, reserve),
    ):
        for group in groups:
            rows.append(
                {
                    "record_type": "osm_input_source",
                    "source_family": "osm",
                    "source_id": group["source_id"],
                    "source_group_id": group["source_group_id"],
                    "canonical_dig_sha256": group["canonical_dig_sha256"],
                    "partition": partition,
                    "representative_relative_path": group[
                        "representative_relative_path"
                    ],
                    "member_files": group["member_files"],
                }
            )
    for pair in pairs:
        rows.append(
            {
                "record_type": "selected_procedural_source",
                "source_family": "procedural",
                "source_id": pair["procedural_source_id"],
                "source_group_id": pair["procedural_source_group_id"],
                "partition": "audit_train_selected",
                "sample_index": pair["procedural_sample_index"],
                "match_group_id": pair["match_group_id"],
            }
        )
    return rows


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("quantiles require at least one value")
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


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support = [row for row in rows if row["in_frozen_support"]]
    return {
        "row_count": len(rows),
        "support_count": len(support),
        "support_rate": len(support) / len(rows),
        "unique_source_count": len({row["source_id"] for row in rows}),
        "support_unique_source_count": len({row["source_id"] for row in support}),
        "required_volume": quantiles([float(row["required_volume"]) for row in rows]),
        "required_volume_in_support": quantiles(
            [float(row["required_volume"]) for row in support]
        ),
        "perimeter_4_edges": quantiles(
            [float(row["perimeter_4_edges"]) for row in rows]
        ),
        "compactness": quantiles(
            [float(row["compactness_4pi_area_over_perimeter_squared"]) for row in rows]
        ),
        "bbox_aspect_ratio": quantiles(
            [float(row["bbox_aspect_ratio"]) for row in rows]
        ),
        "moment_aspect_ratio": quantiles(
            [float(row["moment_aspect_ratio"]) for row in rows]
        ),
        "moment_orientation_degrees": quantiles(
            [float(row["moment_orientation_degrees"]) for row in rows]
        ),
    }


def generator_hashes(generator_root: Path) -> dict[str, str]:
    files = sorted(generator_root.glob("generate_prototypes*.py"))
    if not files:
        raise FileNotFoundError(
            f"no generate_prototypes*.py files under {generator_root}"
        )
    hashes = {path.name: sha256_file(path) for path in files}
    observed_base = hashes.get("generate_prototypes.py")
    if observed_base != EXPECTED_BASE_GENERATOR_SHA256:
        raise RuntimeError(
            "base generator hash mismatch: "
            f"{observed_base} != {EXPECTED_BASE_GENERATOR_SHA256}"
        )
    return hashes


def load_base_generator(generator_root: Path):
    path = generator_root / "generate_prototypes.py"
    if sha256_file(path) != EXPECTED_BASE_GENERATOR_SHA256:
        raise RuntimeError("refusing to load an unpinned base generator")
    name = "_terra_pilot_foundation_base_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    source_foundations: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    b0_hash_before = sha256_file(B0_BUILDER)
    if b0_hash_before != EXPECTED_B0_BUILDER_SHA256:
        raise RuntimeError(
            "B0 builder hash mismatch: "
            f"{b0_hash_before} != {EXPECTED_B0_BUILDER_SHA256}"
        )
    generator_file_hashes = generator_hashes(generator_root)
    base_generator = load_base_generator(generator_root)

    groups, source_manifest_lines, source_manifest_sha256, eligible_mask_count = (
        load_osm_source_groups(source_foundations)
    )
    audit_train, reserve = partition_source_groups(groups)
    osm_rows = audit_osm_train_groups(audit_train)
    procedural_rows = sample_procedural_rows(base_generator)
    pairs, candidate_counts = match_exact_pairs(osm_rows, procedural_rows)
    for row in osm_rows:
        row["exact_procedural_candidate_count"] = candidate_counts.get(
            str(row["source_id"]),
            0,
        )
        row["selected_for_matched_train_pair"] = row["source_id"] in {
            pair["osm_source_id"] for pair in pairs
        }

    b0_hash_after = sha256_file(B0_BUILDER)
    if b0_hash_after != b0_hash_before:
        raise RuntimeError("B0 builder changed while the audit was running")

    output.mkdir(parents=True)
    (output / "source_images.sha256").write_text(
        "\n".join(source_manifest_lines) + "\n"
    )
    write_jsonl(
        output / "source_registry.jsonl",
        source_registry_rows(audit_train, reserve, pairs),
    )
    write_jsonl(output / "osm_train_support.jsonl", osm_rows)
    write_jsonl(output / "procedural_samples.jsonl", procedural_rows)
    write_jsonl(output / "matched_train_pairs.jsonl", pairs)

    support_procedural = [row for row in procedural_rows if row["in_frozen_support"]]
    rejection_keys = sorted(
        {
            key
            for row in procedural_rows
            for key in row["generator_rejections_before_proposal"]
        }
    )
    exact_supported_osm = sum(
        row["in_frozen_support"] and row["exact_procedural_candidate_count"] >= 1
        for row in osm_rows
    )
    exact_matchable_osm = sum(
        row["in_frozen_support"]
        and row["exact_procedural_candidate_count"] >= MINIMUM_EXACT_CANDIDATES
        for row in osm_rows
    )
    duplicate_groups = [group for group in groups if len(group["member_files"]) > 1]
    summary = {
        "schema": SCHEMA,
        "status": "passed",
        "seed_namespace": "train_only",
        "selection_seed_namespace": SEED_NAMESPACE,
        "reserve_distribution_metrics_emitted": False,
        "held_out_selection_performed": False,
        "reserve_processing_contract": {
            "partition": RESERVE_PARTITION,
            "target_masks_loaded": True,
            "canonical_source_identity_computed": True,
            "factory_eligibility_computed": True,
            "distribution_metrics_computed": False,
            "support_selection_performed": False,
            "matching_selection_performed": False,
        },
        "frozen_contract": {
            "required_volume_lower_inclusive": VOLUME_LOWER_INCLUSIVE,
            "required_volume_upper_inclusive": VOLUME_UPPER_INCLUSIVE,
            "required_volume_unit": "unit_depth_cell_volume",
            "perimeter_definition": (
                "exposed four-neighbour cell edges on false-padded raster"
            ),
            "compactness_definition": ("4*pi*required_volume/perimeter_4_edges**2"),
            "compactness_lower_inclusive": COMPACTNESS_LOWER_INCLUSIVE,
            "compactness_upper_inclusive": COMPACTNESS_UPPER_INCLUSIVE,
            "pair_required_volume_delta": 0,
            "pair_perimeter_4_edges_delta": 0,
            "pair_compactness_delta": 0.0,
            "minimum_exact_candidates_per_selected_osm": (MINIMUM_EXACT_CANDIDATES),
            "matched_pair_count": MATCHED_PAIR_COUNT,
            "source_id_prefixes": ["osm:", "procedural:"],
            "match_group_id_is_not_source_id": True,
            "cross_family_canonical_dig_equality_rejected": True,
        },
        "procedural_generator": {
            "revision": PROCEDURAL_GENERATOR_REVISION,
            "seed": TRAIN_SEED,
            "seed_stream_id": PROCEDURAL_STREAM_ID,
            "proposal_count": PROPOSAL_COUNT,
            "main_center_range_cells": [27.0, 37.0],
            "main_heading_degrees": list(range(0, 180, 15)),
            "main_length_range_cells": list(PROCEDURAL_MAIN_LENGTH_RANGE),
            "main_width_range_cells": list(PROCEDURAL_MAIN_WIDTH_RANGE),
            "wing_count_inclusive": [1, 3],
            "wing_along_fraction_range": [-0.35, 0.35],
            "wing_across_fraction_magnitude_range": [0.25, 0.55],
            "wing_length_range_cells": [6.0, 13.0],
            "wing_width_range_cells": [5.0, 10.0],
            "wing_relative_heading_degrees": [0.0, 90.0],
            "generator_required_volume_bounds_inclusive": [110, 340],
            "minimum_border_cells": 10,
        },
        "osm_sources": {
            "input_image_count": EXPECTED_SOURCE_IMAGE_COUNT,
            "factory_eligible_mask_count": eligible_mask_count,
            "canonical_source_group_count": len(groups),
            "canonical_duplicate_group_count": len(duplicate_groups),
            "canonical_duplicate_groups": [
                {
                    "source_group_id": group["source_group_id"],
                    "member_paths": [
                        member["relative_path"] for member in group["member_files"]
                    ],
                }
                for group in duplicate_groups
            ],
            "audit_train_group_count": len(audit_train),
            "reserve_identity_and_eligibility_only_group_count": len(reserve),
            "support_exact_candidate_at_least_1_count": exact_supported_osm,
            "support_exact_candidate_at_least_2_count": exact_matchable_osm,
            "metrics": summarize_metrics(osm_rows),
        },
        "procedural_sources": {
            "metrics": summarize_metrics(procedural_rows),
            "support_duplicate_proposal_count": len(support_procedural)
            - len({row["source_id"] for row in support_procedural}),
            "generator_draw_count": sum(
                int(row["generator_draw_count"]) for row in procedural_rows
            ),
            "generator_rejections_before_proposal": {
                key: sum(
                    int(
                        row["generator_rejections_before_proposal"].get(
                            key,
                            0,
                        )
                    )
                    for row in procedural_rows
                )
                for key in rejection_keys
            },
        },
        "matched_pairs": {
            "count": len(pairs),
            "unique_osm_source_count": len({row["osm_source_id"] for row in pairs}),
            "unique_procedural_source_count": len(
                {row["procedural_source_id"] for row in pairs}
            ),
            "candidate_count": quantiles(
                [float(row["osm_exact_candidate_count"]) for row in pairs]
            ),
        },
        "inputs": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "source_image_manifest_sha256": source_manifest_sha256,
            "b0_builder_sha256_before": b0_hash_before,
            "b0_builder_sha256_after": b0_hash_after,
            "generator_files": generator_file_hashes,
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
    }
    write_json(output / "support_summary.json", summary)
    file_manifest(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--source-foundations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_audit(
        args.generator_root.resolve(),
        args.source_foundations.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "matched_pair_count": summary["matched_pairs"]["count"],
                "support_rate": summary["procedural_sources"]["metrics"][
                    "support_rate"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
