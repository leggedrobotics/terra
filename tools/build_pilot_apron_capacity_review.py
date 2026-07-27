#!/usr/bin/env python3
"""Build the fresh train-only S1 apron-capacity visual review artifact.

This is deliberately not a benchmark-bank builder. It creates one controlled
review slice: 32 fresh OSM dig masks, each paired with a constrained and a
moderate-capacity apron at the same nominal separation. Exact direct-service
validation remains cost-gated, so this artifact proves only exact-loader
compatibility, never canonical benchmark Format admission.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from heapq import heappop, heappush
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Iterator

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from terra.benchmark_protocol import BENCHMARK_RELEASE_ID
from terra.benchmark_protocol import frozen_benchmark_protocol
from terra.benchmark_protocol import frozen_environment_protocol
from terra.benchmark_state import agent_from_record
from terra.benchmark_state import agent_state_sha256
from terra.benchmark_state import agent_to_record
from terra.benchmark_state import derive_initial_state_seed
from terra.benchmark_state import sample_benchmark_initial_agent
from terra.benchmark_state import validate_benchmark_initial_agent
from terra.maps_buffer import contained_dump_capacity_sanity_check
from terra.maps_buffer import load_maps_from_disk
from terra.maps_buffer import validate_exact_dataset_contract
from tools import audit_pilot_foundation_source_support as support
from tools import build_b0_feasibility_panels as b0
from tools.pilot_map_generation import APRON_CAPACITY_BANDS
from tools.pilot_map_generation import APRON_SEPARATION_BAND_TILES
from tools.pilot_map_generation import build_osm_apron_capacity_pair

SCHEMA = "terra_pilot_apron_capacity_review_v1"
SPLIT = "public_train"
PAIR_COUNT = 32
MAP_SIZE = 64
RANK_NAMESPACE = f"{SCHEMA}\0"
STATIC_STATUS = "pending_direct_service_cost_gate"
EXACT_LOADER_STATUS = "exact_loader_format_valid_static_pending"
EXPECTED_B0A_FILES_MANIFEST_SHA256 = (
    "89a5b5325e4e6872f7899b087ac5d0a8cd444dac30315feee4f342f8e532a347"
)
EXPECTED_SOURCE_SUPPORT_FILES_MANIFEST_SHA256 = (
    "7a656c06a381486b2f69d2eb3150f3227832c0ed8202fdaa0693ce78a5551f15"
)
EXPECTED_B0A_SCHEMA = "terra_b0a_paired_feasibility_panels_v1"
EXPECTED_B0A_IDENTITY_COUNT = 256
EXPECTED_B0A_OSM_IDENTITY_COUNT = 80
EXPECTED_B0A_OSM_CANONICAL_DIG_COUNT = 32
SOURCE_SUPPORT_FILES = (
    "osm_train_support.jsonl",
    "support_summary.json",
    "files.sha256",
)
CAPACITY_TOKENS = ("slcap03_04", "slcap07_10")
CELL_NAMES = {
    "slcap03_04": "f_osm_apron_sep02_slcap03_04_v140_189",
    "slcap07_10": "f_osm_apron_sep02_slcap07_10_v140_189",
}
MAP_COLORS = ListedColormap(["#f3e6c3", "#ef8b23", "#4daa6b", "#111111"])
CODE_DEPENDENCIES = (
    Path(__file__).resolve().parent / "audit_pilot_foundation_source_support.py",
    Path(__file__).resolve().parent / "build_b0_feasibility_panels.py",
    Path(__file__).resolve().parent / "pilot_map_generation.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lines(lines: Iterable[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid JSON in {path} at line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{line_number} must contain an object.")
        rows.append(row)
    return rows


def verify_file_manifest(root: Path, manifest_name: str = "files.sha256") -> None:
    manifest = root / manifest_name
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    declared: set[str] = set()
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        try:
            expected, relative = line.split("  ", maxsplit=1)
        except ValueError as error:
            raise RuntimeError(
                f"Malformed {manifest}:{line_number}: {line!r}"
            ) from error
        path = root / relative
        if relative in declared:
            raise RuntimeError(f"Duplicate path in {manifest}: {relative}")
        declared.add(relative)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Hash mismatch for {path}: expected {expected}, got {actual}."
            )
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest_name
    }
    if observed != declared:
        raise RuntimeError(
            f"{manifest} coverage differs: missing={sorted(observed - declared)}, "
            f"extra={sorted(declared - observed)}."
        )


def write_file_manifest(root: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "files.sha256"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths
    ]
    (root / "files.sha256").write_text("\n".join(lines) + "\n")


def _source_support_receipt(source_support: Path) -> dict[str, Any]:
    for name in SOURCE_SUPPORT_FILES:
        if not (source_support / name).is_file():
            raise FileNotFoundError(source_support / name)
    files_manifest_sha256 = sha256_file(source_support / "files.sha256")
    if files_manifest_sha256 != EXPECTED_SOURCE_SUPPORT_FILES_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen source-support file manifest changed: "
            f"{files_manifest_sha256} != "
            f"{EXPECTED_SOURCE_SUPPORT_FILES_MANIFEST_SHA256}."
        )
    verify_file_manifest(source_support)
    summary = json.loads((source_support / "support_summary.json").read_text())
    contract = summary.get("frozen_contract", {})
    expected = {
        "required_volume_lower_inclusive": support.VOLUME_LOWER_INCLUSIVE,
        "required_volume_upper_inclusive": support.VOLUME_UPPER_INCLUSIVE,
        "compactness_lower_inclusive": support.COMPACTNESS_LOWER_INCLUSIVE,
        "compactness_upper_inclusive": support.COMPACTNESS_UPPER_INCLUSIVE,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"Source-support contract changed for {key}: "
                f"{contract.get(key)!r} != {value!r}."
            )
    return {
        "schema": summary.get("schema"),
        "support_summary_sha256": sha256_file(source_support / "support_summary.json"),
        "osm_train_support_sha256": sha256_file(
            source_support / "osm_train_support.jsonl"
        ),
        "source_image_manifest_sha256": summary["inputs"][
            "source_image_manifest_sha256"
        ],
        "files_manifest_sha256": files_manifest_sha256,
    }


def _b0a_osm_canonical_hashes(b0a_bank: Path) -> tuple[set[str], dict[str, Any]]:
    files_manifest = b0a_bank / "files.sha256"
    files_manifest_sha256 = sha256_file(files_manifest)
    if files_manifest_sha256 != EXPECTED_B0A_FILES_MANIFEST_SHA256:
        raise RuntimeError(
            "B0a root file manifest changed: "
            f"{files_manifest_sha256} != {EXPECTED_B0A_FILES_MANIFEST_SHA256}."
        )
    verify_file_manifest(b0a_bank)
    identities_path = b0a_bank / "identities.jsonl"
    provenance_path = b0a_bank / "provenance.json"
    identities = load_jsonl(identities_path)
    provenance = json.loads(provenance_path.read_text())
    if (
        provenance.get("schema") != EXPECTED_B0A_SCHEMA
        or len(identities) != EXPECTED_B0A_IDENTITY_COUNT
    ):
        raise RuntimeError(
            "B0a schema/cardinality changed: "
            f"schema={provenance.get('schema')!r}, identities={len(identities)}."
        )
    identity_hash = sha256_file(identities_path)
    if provenance.get("identity_manifest_sha256") != identity_hash:
        raise RuntimeError("B0a identity manifest no longer matches provenance.")

    hashes: set[str] = set()
    identity_count = 0
    manifests: dict[Path, dict[str, dict[str, Any]]] = {}
    for row in identities:
        if row.get("geometry") != "foundation_osm":
            continue
        split = row.get("split")
        cell = row.get("primary_cell")
        map_id = row.get("map_id")
        if not all(isinstance(value, str) and value for value in (split, cell, map_id)):
            raise RuntimeError(f"Malformed B0a OSM identity: {row}")
        dataset = b0a_bank / "cells" / split / cell
        if dataset not in manifests:
            manifest_rows = load_jsonl(dataset / "manifest.jsonl")
            manifests[dataset] = {item["map_id"]: item for item in manifest_rows}
        manifest_row = manifests[dataset].get(map_id)
        if manifest_row is None:
            raise RuntimeError(f"B0a dataset has no manifest row for {map_id}.")
        target_path = dataset / "images" / f"img_{int(manifest_row['slot_index'])}.npy"
        target = np.load(target_path, allow_pickle=False)
        if b0.sha256_array(target) != row.get("target_identity_sha256"):
            raise RuntimeError(f"B0a target identity changed for {map_id}.")
        hashes.add(support.canonical_dig_sha256(target < 0))
        identity_count += 1

    if (
        identity_count != EXPECTED_B0A_OSM_IDENTITY_COUNT
        or len(hashes) != EXPECTED_B0A_OSM_CANONICAL_DIG_COUNT
    ):
        raise RuntimeError(
            "B0a OSM burn-set cardinality changed: "
            f"identities={identity_count}, canonical_digs={len(hashes)}."
        )
    return hashes, {
        "schema": provenance.get("schema"),
        "files_manifest_sha256": files_manifest_sha256,
        "identities_sha256": identity_hash,
        "osm_identity_count": identity_count,
        "osm_canonical_dig_count": len(hashes),
        "osm_canonical_dig_set_sha256": sha256_text_lines(sorted(hashes)),
    }


def _capacity_rank(canonical_hash: str) -> str:
    return hashlib.sha256(f"{RANK_NAMESPACE}{canonical_hash}".encode()).hexdigest()


def select_fresh_sources(
    rows: list[dict[str, Any]],
    burned_canonical_hashes: set[str],
    pair_count: int = PAIR_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select before consulting apron construction, static checks, or policy data."""
    support_rows = []
    for row in rows:
        if row.get("audit_partition") != "audit_train":
            continue
        if row.get("source_family") != "osm":
            raise RuntimeError(
                "Audit-train source support contains a non-OSM row: "
                f"{row.get('source_group_id')!r}."
            )
        volume = int(row.get("required_volume", -1))
        compactness = float(
            row.get("compactness_4pi_area_over_perimeter_squared", math.nan)
        )
        frozen = (
            support.VOLUME_LOWER_INCLUSIVE <= volume <= support.VOLUME_UPPER_INCLUSIVE
            and support.COMPACTNESS_LOWER_INCLUSIVE
            <= compactness
            <= support.COMPACTNESS_UPPER_INCLUSIVE
        )
        if row.get("in_frozen_support") is not frozen:
            raise RuntimeError(
                f"Frozen-support declaration changed for {row.get('source_group_id')}."
            )
        if frozen:
            support_rows.append(row)

    source_ids = [row.get("source_group_id") for row in support_rows]
    canonical_hashes = [row.get("canonical_dig_sha256") for row in support_rows]
    if (
        any(not isinstance(value, str) or not value for value in source_ids)
        or len(set(source_ids)) != len(source_ids)
        or len(set(canonical_hashes)) != len(canonical_hashes)
    ):
        raise RuntimeError("Support rows must have unique source and canonical IDs.")

    fresh = [
        row
        for row in support_rows
        if row["canonical_dig_sha256"] not in burned_canonical_hashes
    ]
    ordered = sorted(
        fresh,
        key=lambda row: (
            _capacity_rank(row["canonical_dig_sha256"]),
            row["source_group_id"],
        ),
    )
    if len(ordered) < pair_count:
        raise RuntimeError(
            f"Need {pair_count} fresh capacity sources, found {len(ordered)}."
        )
    selected = ordered[:pair_count]
    return selected, {
        "rank_namespace": RANK_NAMESPACE,
        "rank_definition": "sha256(rank_namespace + canonical_dig_sha256)",
        "audit_train_row_count": sum(
            row.get("audit_partition") == "audit_train" for row in rows
        ),
        "frozen_support_count": len(support_rows),
        "b0a_burned_overlap_count": len(support_rows) - len(fresh),
        "post_exclusion_eligible_count": len(fresh),
        "selected_count": len(selected),
        "selection_consulted_constructor_or_policy_outcomes": False,
        "selected_source_set_sha256": sha256_text_lines(
            row["source_group_id"] for row in selected
        ),
    }


def _load_selected_dig(source_foundations: Path, row: dict[str, Any]) -> np.ndarray:
    target_path = source_foundations / row["representative_relative_path"]
    target = np.load(target_path, allow_pickle=False)
    if target.shape != (MAP_SIZE, MAP_SIZE) or not np.all(np.isin(target, (-1, 0, 1))):
        raise RuntimeError(f"Invalid source target: {target_path}")
    dig = np.asarray(target < 0, dtype=np.bool_)
    canonical_hash = support.canonical_dig_sha256(dig)
    if canonical_hash != row["canonical_dig_sha256"]:
        raise RuntimeError(f"Canonical source identity changed: {target_path}")
    metrics = support.foundation_metrics(dig)
    for field in (
        "required_volume",
        "perimeter_4_edges",
        "compactness_4pi_area_over_perimeter_squared",
    ):
        if metrics[field] != row[field]:
            raise RuntimeError(f"Source metric changed for {target_path}: {field}")
    return dig


def _geodesic_distance(target: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Reproduce the frozen 8-connected normalized dense-reward distance."""
    # Accumulate in float64: an exact heap-value comparison would reject every
    # diagonal node after storing sqrt(2) in float32.
    distance = np.full(target.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for y, x in np.argwhere((target > 0) & ~occupancy):
        distance[y, x] = 0.0
        heappush(heap, (0.0, int(y), int(x)))
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    while heap:
        current, y, x = heappop(heap)
        if current != float(distance[y, x]):
            continue
        for dy, dx, step in moves:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < MAP_SIZE and 0 <= nx < MAP_SIZE):
                continue
            if occupancy[ny, nx]:
                continue
            proposed = current + step
            if proposed < float(distance[ny, nx]):
                distance[ny, nx] = proposed
                heappush(heap, (proposed, ny, nx))
    finite = np.isfinite(distance)
    if not np.any(finite):
        raise RuntimeError("No dump cell can seed the dense-reward distance.")
    maximum = float(distance[finite].max())
    if maximum > 0.0:
        distance[finite] /= maximum
    distance[~finite] = 1.0
    return distance.astype(np.float32)


def _clean_source_receipt() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(
            "Refusing to build the frozen review artifact from a dirty Terra "
            "worktree; commit the exact builder and dependencies first."
        )
    return {
        "terra_revision": revision,
        "terra_worktree_clean": True,
        "git_status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _code_dependency_hashes() -> dict[str, str]:
    return {
        path.relative_to(Path(__file__).resolve().parents[1]).as_posix(): (
            sha256_file(path)
        )
        for path in CODE_DEPENDENCIES
    }


def _map_code(target: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    code = np.zeros(target.shape, dtype=np.uint8)
    code[target < 0] = 1
    code[target > 0] = 2
    code[occupancy] = 3
    return code


def _render_pair(
    path: Path,
    pair: dict[str, Any],
    samples: dict[str, dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, token in zip(axes, CAPACITY_TOKENS):
        sample = samples[pair["map_ids"][token]]
        axis.imshow(
            _map_code(sample["target"], sample["occupancy"]),
            cmap=MAP_COLORS,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        metrics = pair["variants"][token]
        axis.set_title(
            f"{token}: {metrics['capacity_ratio']:.2f}x capacity\n"
            f"sep p50/p95 {metrics['p50_tiles']:.2f}/"
            f"{metrics['p95_tiles']:.2f} tiles",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"{pair['pair_id']} — dig {pair['required_volume']} cells "
        f"({100.0 * pair['required_volume'] / MAP_SIZE**2:.2f}% of map)",
        fontsize=10,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _render_gallery(
    path: Path,
    title: str,
    map_ids: list[str],
    records_by_id: dict[str, dict[str, Any]],
    samples: dict[str, dict[str, np.ndarray]],
) -> None:
    columns = 4
    rows = math.ceil(len(map_ids) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, map_id in zip(axes.ravel(), map_ids):
        record = records_by_id[map_id]
        sample = samples[map_id]
        axis.imshow(
            _map_code(sample["target"], sample["occupancy"]),
            cmap=MAP_COLORS,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        axis.set_title(
            f"{record['pair_id'].rsplit(':', 1)[-1]}\n"
            f"work {record['required_volume']}, "
            f"cap {record['capacity']['single_layer_capacity_ratio']:.2f}x",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(map_ids) :]:
        axis.axis("off")
    figure.suptitle(title, fontsize=13)
    figure.savefig(path, dpi=150)
    plt.close(figure)


@contextmanager
def _dataset_size(count: int) -> Iterator[None]:
    previous = os.environ.get("DATASET_SIZE")
    os.environ["DATASET_SIZE"] = str(count)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DATASET_SIZE", None)
        else:
            os.environ["DATASET_SIZE"] = previous


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
        np.save(
            directory / "occupancy" / f"img_{slot}.npy",
            sample["occupancy"],
        )
        np.save(
            directory / "dumpability" / f"img_{slot}.npy",
            sample["dumpability"],
        )
        np.save(directory / "actions" / f"img_{slot}.npy", sample["action"])
        np.save(directory / "distance" / f"img_{slot}.npy", sample["distance"])
        write_json(
            directory / "metadata" / f"trench_{slot}.json",
            {
                "map_id": record["map_id"],
                "family": "foundation",
                "primary_cell": record["primary_cell"],
                "geometry": "foundation_osm",
                "topology": None,
                "axes_ABC": [],
                "foundation_border_axes_ABC": [],
            },
        )
        manifest.append(
            {
                "slot_index": slot,
                "map_id": record["map_id"],
                "source_id": record["source_id"],
                "split": SPLIT,
                "family": "foundation",
                "stratum": "S1_capacity_visual_review",
                "primary_cell": record["primary_cell"],
                "slot_weight": 1.0,
                "identity_slot_multiplicity": 1,
            }
        )
    write_jsonl(directory / "manifest.jsonl", manifest)
    write_json(
        directory / "dataset.json",
        {
            "schema": "terra_exact_map_dataset_v1",
            "slot_count": len(records),
            "unique_identity_count": len(records),
            "shape": [MAP_SIZE, MAP_SIZE],
            "distance_metric": "8_connected_cardinal_1_diagonal_sqrt2",
            "distance_normalization": "per_map_max_to_1",
            "accepted_dump_contract": "exact_visible_dump_v1",
            "minimum_dump_capacity_ratio": min(
                band[0] for band in APRON_CAPACITY_BANDS.values()
            ),
            "source_registry": os.path.relpath(source_registry, directory),
            "source_registry_sha256": sha256_file(source_registry),
        },
    )


def _build_from_selected(
    *,
    selected: list[dict[str, Any]],
    source_foundations: Path,
    output: Path,
    input_receipts: dict[str, Any],
    selection_receipt: dict[str, Any],
    source_state_receipt: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if not selected:
        raise ValueError("At least one fresh OSM source is required.")
    selected_ids = [row.get("source_group_id") for row in selected]
    if any(not isinstance(value, str) or not value for value in selected_ids) or len(
        set(selected_ids)
    ) != len(selected_ids):
        raise RuntimeError("Selected OSM sources must be nonempty and unique.")
    output.mkdir(parents=True)

    terra_revision = source_state_receipt.get("terra_revision")
    if not isinstance(terra_revision, str) or not terra_revision:
        raise ValueError("source_state_receipt must contain a Terra revision.")
    env_config, env_receipt = frozen_benchmark_protocol()
    environment_protocol = frozen_environment_protocol(terra_revision)
    records: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    samples: dict[str, dict[str, np.ndarray]] = {}

    for pair_index, source_row in enumerate(selected):
        dig = _load_selected_dig(source_foundations, source_row)
        source_group_id = source_row["source_group_id"]
        pair_id = f"apron-capacity:{SPLIT}:{pair_index:03d}"
        generated = build_osm_apron_capacity_pair(dig, source_group_id)
        occupancy = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
        dumpability = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        action = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)

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
        state_record = {
            **seed_receipt,
            "pair_id": pair_id,
            "initial_agent_state": agent_to_record(agent),
        }
        states.append(state_record)

        pair_variants = {}
        map_ids = {}
        for token in CAPACITY_TOKENS:
            variant = generated["variants"][token]
            target = np.asarray(variant["target"], dtype=np.int8)
            capacity = contained_dump_capacity_sanity_check(
                target,
                occupancy,
                dumpability,
                action,
                minimum_single_layer_ratio=APRON_CAPACITY_BANDS[token][0],
            )
            ratio = float(capacity["single_layer_capacity_ratio"])
            if ratio != variant["metadata"]["achieved_single_layer_area_ratio"]:
                raise RuntimeError(f"Capacity recomputation disagrees for {pair_id}.")
            map_id = f"capacity-{SPLIT}-{pair_index:03d}-{token}"
            map_ids[token] = map_id
            sample = {
                "target": target,
                "occupancy": occupancy.copy(),
                "dumpability": dumpability.copy(),
                "action": action.copy(),
                "distance": _geodesic_distance(target, occupancy.astype(bool)),
            }
            samples[map_id] = sample
            metadata = variant["metadata"]
            pair_variants[token] = {
                "capacity_ratio": ratio,
                "p50_tiles": float(metadata["separation_p50_tiles"]),
                "p95_tiles": float(metadata["separation_p95_tiles"]),
                "max_tiles": float(metadata["separation_max_tiles"]),
            }
            records.append(
                {
                    "map_id": map_id,
                    "pair_id": pair_id,
                    "source_id": source_row["source_id"],
                    "source_group_id": source_group_id,
                    "split": SPLIT,
                    "family": "foundation",
                    "geometry": "foundation_osm",
                    "primary_cell": CELL_NAMES[token],
                    "capacity_token": token,
                    "separation_token": "sep02",
                    "volume_token": "v140_189",
                    "required_volume": int(dig.sum()),
                    "required_volume_unit": "unit_depth_cell_volume",
                    "dig_identity_sha256": generated["dig_identity_sha256"],
                    "canonical_dig_sha256": source_row["canonical_dig_sha256"],
                    "target_identity_sha256": b0.sha256_array(target),
                    "initial_agent_state_sha256": seed_receipt[
                        "initial_agent_state_sha256"
                    ],
                    "capacity": capacity,
                    "separation": {
                        "p50_tiles": pair_variants[token]["p50_tiles"],
                        "p95_tiles": pair_variants[token]["p95_tiles"],
                        "max_tiles": pair_variants[token]["max_tiles"],
                        "p50_band_inclusive": list(APRON_SEPARATION_BAND_TILES),
                    },
                    "exact_loader_format_valid": True,
                    "static_status": STATIC_STATUS,
                    "direct_service_status": "not_run_cost_gate",
                }
            )

        if (
            len(
                {
                    record["dig_identity_sha256"]
                    for record in records
                    if record["pair_id"] == pair_id
                }
            )
            != 1
        ):
            raise RuntimeError(f"Paired dig identity changed for {pair_id}.")
        pairs.append(
            {
                "pair_id": pair_id,
                "source_id": source_row["source_id"],
                "source_group_id": source_group_id,
                "canonical_dig_sha256": source_row["canonical_dig_sha256"],
                "required_volume": int(dig.sum()),
                "map_ids": map_ids,
                "initial_agent_state_sha256": seed_receipt[
                    "initial_agent_state_sha256"
                ],
                "variants": pair_variants,
                "separation_delta_high_minus_low": {
                    "p50_tiles": (
                        pair_variants["slcap07_10"]["p50_tiles"]
                        - pair_variants["slcap03_04"]["p50_tiles"]
                    ),
                    "p95_tiles": (
                        pair_variants["slcap07_10"]["p95_tiles"]
                        - pair_variants["slcap03_04"]["p95_tiles"]
                    ),
                },
                "exact_shared_dig_and_volume": True,
                "exact_loader_format_valid": True,
                "static_status": STATIC_STATUS,
            }
        )

    records.sort(key=lambda row: row["map_id"])
    source_registry = output / "source_registry.jsonl"
    selected_by_id = {row["source_group_id"]: row for row in selected}
    write_jsonl(
        source_registry,
        [
            {
                "map_id": record["map_id"],
                "source_id": record["source_id"],
                "source_group_id": record["source_group_id"],
                "split": SPLIT,
                "pair_id": record["pair_id"],
                "canonical_dig_sha256": record["canonical_dig_sha256"],
                "source_representative_relative_path": selected_by_id[
                    record["source_group_id"]
                ]["representative_relative_path"],
                "source_representative_file_sha256": sha256_file(
                    source_foundations
                    / selected_by_id[record["source_group_id"]][
                        "representative_relative_path"
                    ]
                ),
            }
            for record in records
        ],
    )
    write_jsonl(output / "identities.jsonl", records)
    write_jsonl(output / "pairs.jsonl", pairs)
    write_jsonl(output / "initial_states.jsonl", states)
    _write_dataset(output / "dataset", records, samples, source_registry)

    pairs_directory = output / "pairs"
    galleries_directory = output / "galleries"
    pairs_directory.mkdir()
    galleries_directory.mkdir()
    for pair in pairs:
        _render_pair(
            pairs_directory / f"{pair['pair_id'].rsplit(':', 1)[-1]}.png",
            pair,
            samples,
        )
    records_by_id = {record["map_id"]: record for record in records}
    for token in CAPACITY_TOKENS:
        map_ids = [pair["map_ids"][token] for pair in pairs]
        _render_gallery(
            galleries_directory / f"{token}.png",
            f"S1 capacity review — {token}",
            map_ids,
            records_by_id,
            samples,
        )

    p50_deltas = [
        pair["separation_delta_high_minus_low"]["p50_tiles"] for pair in pairs
    ]
    p95_deltas = [
        pair["separation_delta_high_minus_low"]["p95_tiles"] for pair in pairs
    ]
    summary = {
        "schema": SCHEMA,
        "status": EXACT_LOADER_STATUS,
        "exact_loader_format_valid": True,
        "canonical_benchmark_format_admitted": False,
        "static_status": STATIC_STATUS,
        "s1_capacity_gate_complete": False,
        "pair_count": len(pairs),
        "map_count": len(records),
        "source_group_count": len({row["source_group_id"] for row in records}),
        "cells": list(CELL_NAMES.values()),
        "capacity_bands_inclusive": {
            token: list(APRON_CAPACITY_BANDS[token]) for token in CAPACITY_TOKENS
        },
        "separation_p50_band_tiles_inclusive": list(APRON_SEPARATION_BAND_TILES),
        "pairwise_separation_delta_high_minus_low": {
            "p50_tiles": {
                "min": min(p50_deltas),
                "median": float(np.median(p50_deltas)),
                "max": max(p50_deltas),
            },
            "p95_tiles": {
                "min": min(p95_deltas),
                "median": float(np.median(p95_deltas)),
                "max": max(p95_deltas),
            },
        },
        "all_pairs_share_exact_dig_and_volume": all(
            pair["exact_shared_dig_and_volume"] for pair in pairs
        ),
        "all_pairs_share_one_initial_state": len(states) == len(pairs),
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
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "source_state": source_state_receipt,
            "code_dependencies": _code_dependency_hashes(),
            "environment_protocol": environment_protocol,
            "env_config_receipt": env_receipt,
            "inputs": input_receipts,
            "selection": selection_receipt,
            "source_registry_sha256": sha256_file(source_registry),
            "identities_sha256": sha256_file(output / "identities.jsonl"),
            "pairs_sha256": sha256_file(output / "pairs.jsonl"),
            "initial_states_sha256": sha256_file(output / "initial_states.jsonl"),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "jax": jax.__version__,
                "matplotlib": matplotlib.__version__,
            },
        },
    )
    (output / "README.md").write_text(
        "# S1 OSM apron-capacity visual review\n\n"
        f"This train-only artifact contains {len(pairs)} fresh OSM source "
        "groups. Each "
        "source has the exact same dig mask, volume, reset state, and nominal "
        "sep02 treatment under two dump-capacity bands: `slcap03_04` and "
        "`slcap07_10`.\n\n"
        "Status: **exact-loader format valid; Static pending direct-service "
        "cost gate.** This is only a local visual-review receipt. It has not "
        "passed canonical benchmark Format admission, does not complete S1 "
        "Capacity, and is not a benchmark or PPO training bank.\n\n"
        "- `pairs/`: one side-by-side PNG per causal pair.\n"
        "- `galleries/`: one overview for each capacity band.\n"
        "- `dataset/`: 64-map exact-loader dataset.\n"
        "- `identities.jsonl`: per-map geometry and validation receipts.\n"
        "- `pairs.jsonl`: exact pairing and p50/p95 separation deltas.\n"
        "- `initial_states.jsonl`: one shared serialized reset per pair.\n"
        "- `summary.json`: aggregate format and selection receipt.\n"
        "- `provenance.json`: pinned inputs, protocol, code, and hashes.\n"
    )
    write_file_manifest(output)
    verify_artifact(output, expected_pair_count=len(selected))
    return summary


def build_artifact(
    *,
    source_support: Path,
    source_foundations: Path,
    b0a_bank: Path,
    output: Path,
) -> dict[str, Any]:
    source_support = source_support.resolve()
    source_foundations = source_foundations.resolve()
    b0a_bank = b0a_bank.resolve()
    output = output.resolve()
    source_state_receipt = _clean_source_receipt()
    source_support_receipt = _source_support_receipt(source_support)
    _, source_lines, source_manifest_sha256 = support.source_image_manifest(
        source_foundations
    )
    if source_manifest_sha256 != source_support_receipt["source_image_manifest_sha256"]:
        raise RuntimeError("Source foundation bank changed after the support audit.")
    burned, b0a_receipt = _b0a_osm_canonical_hashes(b0a_bank)
    rows = load_jsonl(source_support / "osm_train_support.jsonl")
    selected, selection_receipt = select_fresh_sources(
        rows,
        burned,
        pair_count=PAIR_COUNT,
    )
    return _build_from_selected(
        selected=selected,
        source_foundations=source_foundations,
        output=output,
        input_receipts={
            "source_support": source_support_receipt,
            "source_foundations": {
                "source_image_manifest_sha256": source_manifest_sha256,
                "source_image_manifest_line_count": len(source_lines),
            },
            "b0a_bank": b0a_receipt,
        },
        selection_receipt=selection_receipt,
        source_state_receipt=source_state_receipt,
    )


def verify_artifact(
    output: Path,
    *,
    expected_pair_count: int = PAIR_COUNT,
) -> dict[str, Any]:
    output = output.resolve()
    verify_file_manifest(output)
    summary = json.loads((output / "summary.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    records = load_jsonl(output / "identities.jsonl")
    pairs = load_jsonl(output / "pairs.jsonl")
    states = load_jsonl(output / "initial_states.jsonl")
    registry = load_jsonl(output / "source_registry.jsonl")
    count = len(records)
    if expected_pair_count <= 0:
        raise ValueError("expected_pair_count must be positive.")
    if (
        summary.get("status") != EXACT_LOADER_STATUS
        or summary.get("exact_loader_format_valid") is not True
        or summary.get("canonical_benchmark_format_admitted") is not False
    ):
        raise RuntimeError("Artifact must remain explicitly Static-pending.")
    if (
        len(pairs) != expected_pair_count
        or count != 2 * expected_pair_count
        or len(states) != expected_pair_count
        or summary.get("selection", {}).get("selected_count") != expected_pair_count
    ):
        raise RuntimeError("Map/pair/initial-state counts disagree.")
    if (
        summary.get("map_count") != count
        or summary.get("pair_count") != len(pairs)
        or summary.get("source_group_count") != len(pairs)
        or summary.get("s1_capacity_gate_complete") is not False
    ):
        raise RuntimeError("Summary counts or admission status changed.")
    if provenance.get("identities_sha256") != sha256_file(output / "identities.jsonl"):
        raise RuntimeError("Identity receipt changed.")
    receipt_paths = {
        "pairs_sha256": output / "pairs.jsonl",
        "initial_states_sha256": output / "initial_states.jsonl",
        "source_registry_sha256": output / "source_registry.jsonl",
    }
    for field, path in receipt_paths.items():
        if provenance.get(field) != sha256_file(path):
            raise RuntimeError(f"{field} receipt changed.")
    if provenance.get("builder", {}).get("sha256") != sha256_file(
        Path(__file__).resolve()
    ):
        raise RuntimeError("Builder hash changed.")
    if provenance.get("code_dependencies") != _code_dependency_hashes():
        raise RuntimeError("Imported generator/support helper hashes changed.")
    source_state = provenance.get("source_state", {})
    if source_state.get("terra_worktree_clean") is not True or provenance.get(
        "environment_protocol", {}
    ).get("terra_revision") != source_state.get("terra_revision"):
        raise RuntimeError("Clean committed Terra source receipt changed.")

    manifest, shape, minimum_ratio = validate_exact_dataset_contract(
        output / "dataset",
        count,
    )
    if shape != (MAP_SIZE, MAP_SIZE) or minimum_ratio != 3.0:
        raise RuntimeError("Exact dataset metadata changed.")
    with _dataset_size(count):
        loaded = load_maps_from_disk(
            str(output / "dataset"),
            require_trench_metadata=False,
            require_exact_contract=True,
        )
    loaded_arrays = [np.asarray(jax.device_get(value)) for value in loaded]
    targets, occupancies = loaded_arrays[0], loaded_arrays[1]
    dumpabilities, actions = loaded_arrays[6], loaded_arrays[7]
    distances = loaded_arrays[8]
    records_by_id = {row["map_id"]: row for row in records}
    if len(records_by_id) != count:
        raise RuntimeError("Identity rows contain duplicate map IDs.")
    slots = {row["map_id"]: int(row["slot_index"]) - 1 for row in manifest}
    env_config, _ = frozen_benchmark_protocol()
    states_by_pair = {row["pair_id"]: row for row in states}
    if len(states_by_pair) != len(states):
        raise RuntimeError("Initial-state rows contain duplicate pair IDs.")
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    if len(pairs_by_id) != len(pairs):
        raise RuntimeError("Pair rows contain duplicate pair IDs.")
    registry_by_id = {row["map_id"]: row for row in registry}
    if len(registry_by_id) != count or set(registry_by_id) != set(records_by_id):
        raise RuntimeError("Source registry does not cover every map exactly once.")

    expected_cell_counts = {cell: len(pairs) for cell in CELL_NAMES.values()}
    observed_cell_counts = {cell: 0 for cell in CELL_NAMES.values()}
    for record in records:
        token = record.get("capacity_token")
        if (
            record.get("split") != SPLIT
            or record.get("family") != "foundation"
            or token not in CAPACITY_TOKENS
            or record.get("primary_cell") != CELL_NAMES[token]
        ):
            raise RuntimeError(f"Identity factors changed: {record.get('map_id')}")
        observed_cell_counts[record["primary_cell"]] += 1
        registry_row = registry_by_id[record["map_id"]]
        if (
            registry_row.get("source_id") != record["source_id"]
            or registry_row.get("source_group_id") != record["source_group_id"]
            or registry_row.get("split") != SPLIT
            or registry_row.get("pair_id") != record["pair_id"]
        ):
            raise RuntimeError(f"Source registry changed for {record['map_id']}.")
    if observed_cell_counts != expected_cell_counts:
        raise RuntimeError(f"Capacity-cell counts changed: {observed_cell_counts}.")
    all_pair_map_ids = {map_id for pair in pairs for map_id in pair["map_ids"].values()}
    if all_pair_map_ids != set(records_by_id):
        raise RuntimeError("Pairs do not cover every map identity exactly once.")
    if len({pair["source_group_id"] for pair in pairs}) != len(pairs):
        raise RuntimeError("Capacity pairs must use unique source groups.")

    observed_p50_deltas = []
    observed_p95_deltas = []
    for pair in pairs:
        if set(pair["map_ids"]) != set(CAPACITY_TOKENS):
            raise RuntimeError(f"Pair variants changed: {pair['pair_id']}")
        pair_records = [records_by_id[map_id] for map_id in pair["map_ids"].values()]
        if len({row["dig_identity_sha256"] for row in pair_records}) != 1:
            raise RuntimeError(f"Pair changed dig identity: {pair['pair_id']}")
        if len({row["required_volume"] for row in pair_records}) != 1:
            raise RuntimeError(f"Pair changed work volume: {pair['pair_id']}")
        if any(
            row["source_group_id"] != pair["source_group_id"] for row in pair_records
        ):
            raise RuntimeError(f"Pair changed source group: {pair['pair_id']}")
        if any(
            row["canonical_dig_sha256"] != pair["canonical_dig_sha256"]
            for row in pair_records
        ):
            raise RuntimeError(f"Pair changed canonical dig: {pair['pair_id']}")
        state_row = states_by_pair[pair["pair_id"]]
        expected_seed, expected_digest = derive_initial_state_seed(
            BENCHMARK_RELEASE_ID,
            SPLIT,
            pair["source_group_id"],
            0,
        )
        if (
            state_row.get("release_id") != BENCHMARK_RELEASE_ID
            or state_row.get("split") != SPLIT
            or state_row.get("source_group_id") != pair["source_group_id"]
            or state_row.get("state_index") != 0
            or state_row.get("seed_uint32") != expected_seed
            or state_row.get("seed_digest_sha256") != expected_digest
        ):
            raise RuntimeError(f"State seed receipt changed: {pair['pair_id']}")
        agent = agent_from_record(state_row["initial_agent_state"])
        if agent_state_sha256(agent) != pair["initial_agent_state_sha256"]:
            raise RuntimeError(f"Pair state hash changed: {pair['pair_id']}")
        if (
            state_row["initial_agent_state_sha256"]
            != pair["initial_agent_state_sha256"]
        ):
            raise RuntimeError(f"State receipt changed: {pair['pair_id']}")
        for record in pair_records:
            if (
                record["initial_agent_state_sha256"]
                != pair["initial_agent_state_sha256"]
            ):
                raise RuntimeError(
                    f"Variant changed shared pair state: {record['map_id']}"
                )
            slot = slots[record["map_id"]]
            raw_target = np.load(
                output / "dataset" / "images" / f"img_{slot + 1}.npy",
                allow_pickle=False,
            )
            if b0.sha256_array(raw_target) != record[
                "target_identity_sha256"
            ] or not np.array_equal(targets[slot], raw_target):
                raise RuntimeError(f"Target identity changed: {record['map_id']}")
            if (
                b0.sha256_array((targets[slot] < 0).astype(np.uint8))
                != record["dig_identity_sha256"]
            ):
                raise RuntimeError(f"Dig identity changed: {record['map_id']}")
            if (
                np.any(occupancies[slot] != 0)
                or np.any(actions[slot] != 0)
                or not np.all(dumpabilities[slot])
            ):
                raise RuntimeError(
                    f"Capacity slice map layers changed: {record['map_id']}"
                )
            required_volume = int((targets[slot] < 0).sum())
            if (
                required_volume != record["required_volume"]
                or required_volume != pair["required_volume"]
                or support.canonical_dig_sha256(targets[slot] < 0)
                != record["canonical_dig_sha256"]
            ):
                raise RuntimeError(f"Work receipt changed: {record['map_id']}")
            capacity = contained_dump_capacity_sanity_check(
                targets[slot],
                occupancies[slot],
                dumpabilities[slot],
                actions[slot],
                minimum_single_layer_ratio=APRON_CAPACITY_BANDS[
                    record["capacity_token"]
                ][0],
            )
            if capacity != record["capacity"]:
                raise RuntimeError(f"Capacity receipt changed: {record['map_id']}")
            token = record["capacity_token"]
            lower, upper = APRON_CAPACITY_BANDS[token]
            ratio = float(capacity["single_layer_capacity_ratio"])
            if not lower <= ratio <= upper:
                raise RuntimeError(f"Capacity left its closed band: {record['map_id']}")
            pair_variant = pair["variants"][token]
            separation = b0.dump_distance_statistics(
                targets[slot] < 0,
                targets[slot] > 0,
            )
            if (
                pair_variant["capacity_ratio"] != ratio
                or pair_variant["p50_tiles"] != separation["p50_tiles"]
                or pair_variant["p95_tiles"] != separation["p95_tiles"]
                or pair_variant["max_tiles"] != separation["max_tiles"]
                or record["separation"]["p50_tiles"] != separation["p50_tiles"]
                or record["separation"]["p95_tiles"] != separation["p95_tiles"]
                or record["separation"]["max_tiles"] != separation["max_tiles"]
            ):
                raise RuntimeError(f"Pair metrics changed: {record['map_id']}")
            recomputed_distance = _geodesic_distance(
                raw_target,
                occupancies[slot].astype(np.bool_),
            )
            if not np.array_equal(distances[slot], recomputed_distance):
                raise RuntimeError(f"Dense-reward distance changed: {record['map_id']}")
            p50_lower, p50_upper = APRON_SEPARATION_BAND_TILES
            if not p50_lower <= pair_variant["p50_tiles"] <= p50_upper:
                raise RuntimeError(f"Separation left sep02: {record['map_id']}")
            validate_benchmark_initial_agent(
                agent,
                env_cfg=env_config,
                padding_mask=occupancies[slot],
                action_map=actions[slot],
                dumpability_mask=dumpabilities[slot],
            )
        low = pair["variants"]["slcap03_04"]
        high = pair["variants"]["slcap07_10"]
        expected_deltas = {
            "p50_tiles": high["p50_tiles"] - low["p50_tiles"],
            "p95_tiles": high["p95_tiles"] - low["p95_tiles"],
        }
        if pair["separation_delta_high_minus_low"] != expected_deltas:
            raise RuntimeError(f"Pair separation deltas changed: {pair['pair_id']}")
        observed_p50_deltas.append(expected_deltas["p50_tiles"])
        observed_p95_deltas.append(expected_deltas["p95_tiles"])
        pair_png = output / "pairs" / f"{pair['pair_id'].rsplit(':', 1)[-1]}.png"
        if not pair_png.is_file():
            raise RuntimeError(f"Missing pair review image: {pair_png}")

    expected_summary_deltas = {
        "p50_tiles": {
            "min": min(observed_p50_deltas),
            "median": float(np.median(observed_p50_deltas)),
            "max": max(observed_p50_deltas),
        },
        "p95_tiles": {
            "min": min(observed_p95_deltas),
            "median": float(np.median(observed_p95_deltas)),
            "max": max(observed_p95_deltas),
        },
    }
    if (
        summary.get("pairwise_separation_delta_high_minus_low")
        != expected_summary_deltas
    ):
        raise RuntimeError("Summary pairwise separation deltas changed.")

    return {
        "status": "passed",
        "exact_loader_format_valid": True,
        "canonical_benchmark_format_admitted": False,
        "static_status": STATIC_STATUS,
        "pair_count": len(pairs),
        "map_count": count,
        "file_manifest_sha256": sha256_file(output / "files.sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-support", type=Path, required=True)
    build.add_argument("--source-foundations", type=Path, required=True)
    build.add_argument("--b0a-bank", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "build":
        result = build_artifact(
            source_support=args.source_support,
            source_foundations=args.source_foundations,
            b0a_bank=args.b0a_bank,
            output=args.output,
        )
    else:
        result = verify_artifact(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
