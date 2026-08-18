#!/usr/bin/env python3
"""Deterministically enrich a frozen Terra bank with finite trench sections.

The runtime loader (``terra/maps_buffer.py:_trench_records_from_metadata`` with
``require_finite_segments=True``) needs one finite generated segment per
``axes_ABC`` entry plus the generated ``trench_half_width_tiles``.  Existing
frozen banks only kept the infinite ``axes_ABC`` lines, so this tool rebuilds
the lost fields from the generator provenance that produced them.

Sources, in order, per map:

* finite endpoints  -> ``trench_arms`` of the generator record whose id equals
  the bank ``map_id`` (or the sidecar ``parent_map_id`` for derived control
  conditions).  Arms are already ``[y, x]`` and already in ``axes_ABC`` order.
* generated width   -> the bank sidecar's own ``target_trench_width_tiles / 2``.
  The bank re-widened its parents, so the parent's width must not be reused.

Nothing is re-simulated and no section is fitted to a raster.  Every map is
checked against the paired bank axis, against the real loader, and against the
runtime membership fringe bound before it is written.  Anything unresolved is a
named failure.

The output bank is a derived tree: array payloads are linked (default) or
copied, and only the trench metadata sidecars are new files.  Frozen inputs are
never written to.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra.map import TRENCH_MEMBERSHIP_FRINGE_TOLERANCE_TILES
from terra.maps_buffer import _trench_records_from_metadata


MAX_TRENCH_TYPE = 4
ENRICHMENT_SCHEMA = "terra_trench_finite_sections_v1"
MANIFEST_SCHEMA = "terra_trench_finite_enrichment_manifest_v1"
RESET_ARRAY_FOLDERS = ("images", "occupancy", "dumpability", "actions", "distance")
METADATA_FOLDER = "metadata"
# The loader's own endpoint-on-axis tolerance; enrichment must never emit a
# segment the loader would then reject.
AXIS_RESIDUAL_TOLERANCE_TILES = 0.05
CANDIDATE_MAP_ID = re.compile(r"^(?P<stem>.*?)-(?P<index>\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tree_digest(root: Path, relative_paths: list[str]) -> str:
    """Order-independent digest over an explicit file list."""
    lines = sorted(
        f"{relative}  {sha256_file(root / relative)}" for relative in relative_paths
    )
    return sha256_text("\n".join(lines) + "\n")


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {path} line {number}: {exc}") from exc
    return rows


def write_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class ProvenanceIndex:
    """Generator review records keyed by the id they were published under."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._by_index: dict[str, Path] = {}
        for path in root.glob("img_*.json"):
            self._by_index[path.stem[len("img_") :]] = path
        if not self._by_index:
            raise RuntimeError(f"No img_*.json generator records under {root}.")
        self._cache: dict[str, tuple[Path, dict[str, Any], str]] = {}
        self.used: set[str] = set()

    def lookup(self, map_id: str) -> tuple[Path, dict[str, Any], str] | None:
        match = CANDIDATE_MAP_ID.match(map_id)
        if match is None:
            return None
        index = match.group("index")
        path = self._by_index.get(index)
        if path is None:
            return None
        if index not in self._cache:
            self._cache[index] = (path, json.loads(path.read_text()), sha256_file(path))
        record_path, record, record_sha256 = self._cache[index]
        if str(record.get("map_id", map_id)) != map_id:
            return None
        self.used.add(index)
        return record_path, record, record_sha256


def candidate_ids(map_id: str, sidecar: dict[str, Any]) -> list[str]:
    ids = [map_id]
    parent = sidecar.get("parent_map_id")
    if isinstance(parent, str) and parent:
        ids.append(parent)
    return ids


def half_width_tiles(sidecar: dict[str, Any], map_id: str) -> tuple[float, str]:
    width = sidecar.get("target_trench_width_tiles")
    if width is None:
        raise RuntimeError(
            f"{map_id}: sidecar has no target_trench_width_tiles; the generated "
            "half width cannot be recovered without guessing."
        )
    value = float(width) / 2.0
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{map_id}: non-positive generated half width {value}.")
    return value, "target_trench_width_tiles/2"


def segment_distance(cells: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    length_squared = float(vector @ vector)
    projection = np.clip(((cells - start) @ vector) / max(length_squared, 1e-9), 0.0, 1.0)
    closest = start + projection[:, None] * vector
    return np.linalg.norm(cells - closest, axis=1)


def enrich_sidecar(
    *,
    map_id: str,
    sidecar: dict[str, Any],
    record_path: Path,
    record: dict[str, Any],
    record_sha256: str,
    target_map: np.ndarray,
    tool_sha256: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    axes = list(sidecar.get("axes_ABC") or [])
    arms = record.get("trench_arms")
    if arms is None:
        raise RuntimeError(f"{map_id}: {record_path} has no trench_arms.")
    arms = [np.asarray(arm, dtype=np.float64) for arm in arms]
    if len(arms) != len(axes):
        raise RuntimeError(
            f"{map_id}: {len(axes)} bank axes but {len(arms)} generated arms in "
            f"{record_path}; refusing to pair them by position."
        )
    if len(axes) > MAX_TRENCH_TYPE:
        raise RuntimeError(
            f"{map_id}: {len(axes)} axes exceed the Terra maximum {MAX_TRENCH_TYPE}."
        )
    declared = sidecar.get("trench_axes_count")
    if declared is not None and int(declared) != len(axes):
        raise RuntimeError(
            f"{map_id}: trench_axes_count {declared} disagrees with "
            f"{len(axes)} axes_ABC entries."
        )

    half_width, half_width_source = half_width_tiles(sidecar, map_id)

    segments: list[list[list[float]]] = []
    max_residual = 0.0
    for index, (axis, arm) in enumerate(zip(axes, arms)):
        if arm.ndim != 2 or arm.shape[0] < 2 or arm.shape[1] != 2:
            raise RuntimeError(
                f"{map_id}: generated arm {index} has shape {arm.shape}; expected "
                "at least two [y, x] points."
            )
        if arm.shape[0] > 2:
            raise RuntimeError(
                f"{map_id}: generated arm {index} is a {arm.shape[0]}-point "
                "polyline; Terra stores one straight section per axis and the "
                "interior vertices would be discarded."
            )
        if not np.all(np.isfinite(arm)):
            raise RuntimeError(f"{map_id}: generated arm {index} is not finite.")
        start, end = arm[0], arm[-1]
        if float(np.linalg.norm(end - start)) <= 1e-6:
            raise RuntimeError(f"{map_id}: generated arm {index} has zero length.")
        denominator = float(np.hypot(float(axis["A"]), float(axis["B"])))
        if denominator <= 1e-6:
            raise RuntimeError(f"{map_id}: bank axis {index} has a zero normal.")
        for point in (start, end):
            residual = abs(
                float(axis["A"]) * float(point[1])
                + float(axis["B"]) * float(point[0])
                + float(axis["C"])
            ) / denominator
            max_residual = max(max_residual, residual)
        if max_residual > AXIS_RESIDUAL_TOLERANCE_TILES:
            raise RuntimeError(
                f"{map_id}: generated arm {index} is {max_residual:.4f} tiles off "
                "its paired bank axis; the provenance record does not describe "
                "this map."
            )
        segments.append([[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]])

    enriched = dict(sidecar)
    enriched["trench_segments_yx"] = segments
    enriched["trench_half_width_tiles"] = half_width

    # The real loader contract, on the real record we are about to publish.
    records, trench_type = _trench_records_from_metadata(
        enriched, MAX_TRENCH_TYPE, require_finite_segments=True
    )
    if trench_type != len(axes):
        raise RuntimeError(
            f"{map_id}: loader returned trench_type {trench_type} for {len(axes)} axes."
        )
    del records

    # Runtime membership never assigns a target cell further than
    # half_width + fringe from its nearest finite section.
    cells = np.argwhere(np.squeeze(np.asarray(target_map)) < 0).astype(np.float64)
    max_cell_distance = 0.0
    if cells.size:
        best = np.full(len(cells), np.inf)
        for segment in segments:
            best = np.minimum(
                best,
                segment_distance(
                    cells,
                    np.asarray(segment[0], dtype=np.float64),
                    np.asarray(segment[1], dtype=np.float64),
                ),
            )
        max_cell_distance = float(best.max())
    bound = half_width + TRENCH_MEMBERSHIP_FRINGE_TOLERANCE_TILES
    if max_cell_distance > bound:
        raise RuntimeError(
            f"{map_id}: a target cell is {max_cell_distance:.4f} tiles from every "
            f"generated section but runtime membership only reaches {bound:.4f}."
        )

    enriched["trench_finite_metadata"] = {
        "schema": ENRICHMENT_SCHEMA,
        "provenance_path": str(record_path),
        "provenance_sha256": record_sha256,
        "provenance_map_id": str(record.get("map_id", "")),
        "half_width_source": half_width_source,
        "max_endpoint_axis_residual_tiles": max_residual,
        "max_target_cell_section_distance_tiles": max_cell_distance,
        "membership_fringe_tolerance_tiles": float(
            TRENCH_MEMBERSHIP_FRINGE_TOLERANCE_TILES
        ),
        "tool_sha256": tool_sha256,
    }
    return enriched, {
        "max_endpoint_axis_residual_tiles": max_residual,
        "max_target_cell_section_distance_tiles": max_cell_distance,
        "half_width_tiles": half_width,
    }


def place(source: Path, destination: Path, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link_mode == "symlink":
        destination.symlink_to(source.resolve())
    elif link_mode == "hardlink":
        os.link(source, destination)
    elif link_mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported link mode {link_mode!r}.")


def enrich_dataset(
    *,
    input_root: Path,
    output_root: Path,
    relative_source: str,
    relative_target: str,
    provenance: ProvenanceIndex,
    unenrichable: str,
    link_mode: str,
    tool_sha256: str,
) -> dict[str, Any]:
    source = input_root / relative_source
    target = output_root / relative_target
    dataset_json = json.loads((source / "dataset.json").read_text())
    rows = read_json_lines(source / "manifest.jsonl")
    rows = sorted(rows, key=lambda row: int(row["slot_index"]))

    kept_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    enriched_count = 0
    unenriched_kept = 0
    per_condition: Counter[str] = Counter()
    per_family: Counter[str] = Counter()
    enriched_per_condition: Counter[str] = Counter()
    stats = {
        "max_endpoint_axis_residual_tiles": 0.0,
        "max_target_cell_section_distance_tiles": 0.0,
    }
    half_widths: set[float] = set()
    plan: list[tuple[int, dict[str, Any], dict[str, Any] | None]] = []

    for row in rows:
        slot = int(row["slot_index"])
        sidecar_path = source / METADATA_FOLDER / f"trench_{slot}.json"
        sidecar = json.loads(sidecar_path.read_text())
        axes = list(sidecar.get("axes_ABC") or [])
        map_id = str(row["map_id"])
        if not axes:
            plan.append((slot, row, None))
            continue

        found = None
        for candidate in candidate_ids(map_id, sidecar):
            found = provenance.lookup(candidate)
            if found is not None:
                break
        if found is None:
            entry = {
                "map_id": map_id,
                "slot_index": slot,
                "primary_cell": str(row.get("primary_cell", "")),
                "metadata_schema": str(sidecar.get("schema", "")),
                "source_scenario_id": str(sidecar.get("source_scenario_id", "")),
                "reason": "no generator record carries finite trench_arms for this id",
            }
            unresolved.append(entry)
            if unenrichable == "fail":
                continue
            if unenrichable == "drop":
                dropped.append(entry)
                continue
            plan.append((slot, row, None))
            unenriched_kept += 1
            continue

        record_path, record, record_sha256 = found
        enriched, map_stats = enrich_sidecar(
            map_id=map_id,
            sidecar=sidecar,
            record_path=record_path,
            record=record,
            record_sha256=record_sha256,
            target_map=np.load(source / "images" / f"img_{slot}.npy", allow_pickle=False),
            tool_sha256=tool_sha256,
        )
        for key in stats:
            stats[key] = max(stats[key], map_stats[key])
        half_widths.add(round(map_stats["half_width_tiles"], 12))
        enriched_count += 1
        enriched_per_condition[str(row.get("primary_cell", ""))] += 1
        plan.append((slot, row, enriched))

    if unenrichable == "fail" and unresolved:
        listed = ", ".join(entry["map_id"] for entry in unresolved[:8])
        raise RuntimeError(
            f"{relative_source}: {len(unresolved)} trench maps have no finite "
            f"generator provenance (first: {listed}). Re-run with "
            "--unenrichable keep or --unenrichable drop to record the exclusion "
            "explicitly; do not fit sections to the raster."
        )

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for folder in RESET_ARRAY_FOLDERS:
        (target / folder).mkdir()
    (target / METADATA_FOLDER).mkdir()

    for new_slot, (old_slot, row, enriched) in enumerate(plan, start=1):
        for folder in RESET_ARRAY_FOLDERS:
            place(
                source / folder / f"img_{old_slot}.npy",
                target / folder / f"img_{new_slot}.npy",
                link_mode,
            )
        sidecar_target = target / METADATA_FOLDER / f"trench_{new_slot}.json"
        if enriched is None:
            place(
                source / METADATA_FOLDER / f"trench_{old_slot}.json",
                sidecar_target,
                link_mode,
            )
        else:
            sidecar_target.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n")
        new_row = dict(row)
        new_row["slot_index"] = new_slot
        kept_rows.append(new_row)
        per_condition[str(row.get("primary_cell", ""))] += 1
        per_family[str(row.get("family", ""))] += 1

    multiplicity = Counter(row["map_id"] for row in kept_rows)
    for row in kept_rows:
        row["identity_slot_multiplicity"] = multiplicity[row["map_id"]]
    write_json_lines(target / "manifest.jsonl", kept_rows)

    new_dataset_json = dict(dataset_json)
    new_dataset_json["slot_count"] = len(kept_rows)
    new_dataset_json["unique_identity_count"] = len(multiplicity)
    new_dataset_json["trench_finite_enrichment"] = {
        "schema": ENRICHMENT_SCHEMA,
        "source_dataset": relative_source,
        "source_bank": str(input_root),
        "enriched_slots": enriched_count,
        "unenriched_slots_kept": unenriched_kept,
        "dropped_slots": len(dropped),
        "tool_sha256": tool_sha256,
    }
    (target / "dataset.json").write_text(
        json.dumps(new_dataset_json, indent=2, sort_keys=True) + "\n"
    )

    metadata_relatives = [
        f"{METADATA_FOLDER}/trench_{index}.json" for index in range(1, len(kept_rows) + 1)
    ]
    return {
        "source_dataset": relative_source,
        "output_dataset": relative_target,
        "input_slots": len(rows),
        "output_slots": len(kept_rows),
        "enriched_slots": enriched_count,
        "unenriched_slots_kept": unenriched_kept,
        "dropped_slots": dropped,
        "unresolved_slots": unresolved,
        "per_family_counts": dict(sorted(per_family.items())),
        "per_condition_counts": dict(sorted(per_condition.items())),
        "enriched_per_condition_counts": dict(sorted(enriched_per_condition.items())),
        "generated_half_width_tiles": sorted(half_widths),
        "max_endpoint_axis_residual_tiles": stats["max_endpoint_axis_residual_tiles"],
        "max_target_cell_section_distance_tiles": stats[
            "max_target_cell_section_distance_tiles"
        ],
        "input_metadata_digest": tree_digest(
            source,
            [f"{METADATA_FOLDER}/trench_{int(row['slot_index'])}.json" for row in rows],
        ),
        "output_metadata_digest": tree_digest(target, metadata_relatives),
        "input_array_digest": tree_digest(
            source,
            [
                f"{folder}/img_{int(row['slot_index'])}.npy"
                for row in rows
                for folder in RESET_ARRAY_FOLDERS
            ],
        ),
        "output_array_digest": tree_digest(
            target,
            [
                f"{folder}/img_{index}.npy"
                for index in range(1, len(kept_rows) + 1)
                for folder in RESET_ARRAY_FOLDERS
            ],
        ),
        "output_manifest_sha256": sha256_file(target / "manifest.jsonl"),
        "output_dataset_json_sha256": sha256_file(target / "dataset.json"),
    }


def discover_datasets(root: Path) -> list[str]:
    found = []
    for path in sorted(root.rglob("dataset.json")):
        directory = path.parent
        if directory == root:
            continue
        if (directory / "manifest.jsonl").is_file():
            found.append(str(directory.relative_to(root)))
    return found


def copy_root_files(input_root: Path, output_root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}

    def publish(relative: Path) -> None:
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        shutil.copy2(input_root / relative, destination)
        destination.chmod(0o644)
        digests[str(relative)] = sha256_file(destination)

    for path in sorted(input_root.iterdir()):
        if path.is_dir():
            continue
        publish(path.relative_to(input_root))
    # Panel groups may carry their own registry next to the panels.
    for path in sorted(input_root.rglob("source_registry.jsonl")):
        relative = path.relative_to(input_root)
        if len(relative.parts) == 1:
            continue
        publish(relative)
    # Review galleries are provenance, not payload; link them instead of copying.
    review = input_root / "review"
    if review.is_dir():
        link = output_root / "review"
        if link.exists() or link.is_symlink():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(review.resolve(), target_is_directory=True)
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="generator review_metadata directory holding img_<index>.json records",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="dataset relative path, or SRC=DST to rename it; repeatable. "
        "Defaults to every dataset in the input bank.",
    )
    parser.add_argument(
        "--unenrichable",
        choices=("fail", "keep", "drop"),
        default="fail",
        help="what to do with a trench map that has no finite provenance",
    )
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--skip-root-files",
        action="store_true",
        help="do not re-copy the bank root receipts (for a second pass into an "
        "existing output bank)",
    )
    args = parser.parse_args()

    tool_path = Path(__file__).resolve()
    tool_sha256 = sha256_file(tool_path)
    input_root = args.input_bank.resolve()
    output_root = args.output_bank.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    provenance = ProvenanceIndex(args.provenance.resolve())

    requested = args.dataset or discover_datasets(input_root)
    pairs = []
    for item in requested:
        source, _, destination = item.partition("=")
        pairs.append((source, destination or source))

    root_digests = {} if args.skip_root_files else copy_root_files(input_root, output_root)

    started = time.time()
    datasets = []
    for source, destination in pairs:
        result = enrich_dataset(
            input_root=input_root,
            output_root=output_root,
            relative_source=source,
            relative_target=destination,
            provenance=provenance,
            unenrichable=args.unenrichable,
            link_mode=args.link_mode,
            tool_sha256=tool_sha256,
        )
        datasets.append(result)
        print(
            f"{source} -> {destination}: {result['output_slots']} slots, "
            f"{result['enriched_slots']} enriched, "
            f"{result['unenriched_slots_kept']} unenriched kept, "
            f"{len(result['dropped_slots'])} dropped",
            flush=True,
        )

    totals = {
        "datasets": len(datasets),
        "input_slots": sum(item["input_slots"] for item in datasets),
        "output_slots": sum(item["output_slots"] for item in datasets),
        "enriched_slots": sum(item["enriched_slots"] for item in datasets),
        "unenriched_slots_kept": sum(item["unenriched_slots_kept"] for item in datasets),
        "dropped_slots": sum(len(item["dropped_slots"]) for item in datasets),
        "max_endpoint_axis_residual_tiles": max(
            (item["max_endpoint_axis_residual_tiles"] for item in datasets), default=0.0
        ),
        "max_target_cell_section_distance_tiles": max(
            (item["max_target_cell_section_distance_tiles"] for item in datasets),
            default=0.0,
        ),
    }
    per_condition: Counter[str] = Counter()
    enriched_per_condition: Counter[str] = Counter()
    for item in datasets:
        per_condition.update(item["per_condition_counts"])
        enriched_per_condition.update(item["enriched_per_condition_counts"])

    report = {
        "schema": MANIFEST_SCHEMA,
        "generated_unix": int(started),
        "tool": {"path": str(tool_path), "sha256": tool_sha256},
        "terra_contract": {
            "loader": "terra/maps_buffer.py:_trench_records_from_metadata",
            "require_finite_segments": True,
            "max_trench_type": MAX_TRENCH_TYPE,
            "axis_residual_tolerance_tiles": AXIS_RESIDUAL_TOLERANCE_TILES,
            "membership_fringe_tolerance_tiles": float(
                TRENCH_MEMBERSHIP_FRINGE_TOLERANCE_TILES
            ),
        },
        "input_bank": {"root": str(input_root), "root_file_sha256": root_digests},
        "output_bank": {"root": str(output_root), "link_mode": args.link_mode},
        "provenance": {
            "root": str(provenance.root),
            "records_available": len(provenance._by_index),
            "records_used": len(provenance.used),
        },
        "unenrichable_policy": args.unenrichable,
        "totals": totals,
        "per_condition_counts": dict(sorted(per_condition.items())),
        "enriched_per_condition_counts": dict(sorted(enriched_per_condition.items())),
        "datasets": datasets,
        "wall_seconds": round(time.time() - started, 2),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(totals, indent=2))
    print(f"manifest {args.manifest} sha256 {sha256_file(args.manifest)}")


if __name__ == "__main__":
    main()
