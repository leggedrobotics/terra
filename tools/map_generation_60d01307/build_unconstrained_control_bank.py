#!/usr/bin/env python3
"""Build two explicit no-dump-constraint Terra capability controls.

The controls reuse source-disjoint dig geometry and reset state from one frozen
accepted bank.  Only the dump target and its dense distance map change:

* ``fnd-slab-allfree`` reuses ``fnd-slab-ring3x`` geometry;
* ``trn-straight-allfree`` reuses ``trn-straight-side2`` geometry; and
* every legal, non-dig cell is an accepted dump cell.

This is a diagnostic bank.  Its scores are reported next to, but never pooled
into, the constrained 32-condition benchmark macro.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from terra.maps_buffer import (  # noqa: E402
    EXACT_DATASET_SCHEMA,
    RESET_ARRAY_FOLDERS,
    RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
    reset_array_scenario_sha256,
    validate_exact_dataset_contract,
)
from tools.map_generation.generate_prototypes import (  # noqa: E402
    compute_geodesic_distance,
)
from tools.map_generation.materialize_loader_bank import (  # noqa: E402
    ACCEPTED_DUMP_CONTRACT,
    DISTANCE_METRIC,
    DISTANCE_NORMALIZATION,
    LOADER_BANK_SCHEMA,
    _exact_reset_seeds,
    episode_id,
)

CONTROL_SCHEMA = "terra_unconstrained_control_bank_v1"
CONTROL_STRATUM = "capability_floor_v1"
SPLITS = ("train", "promotion", "development", "sealed")
EVALUATION_SPLITS = SPLITS[1:]
CONTROL_DEFINITIONS = {
    "fnd-slab-allfree": {
        "family": "foundation",
        "parent_condition": "fnd-slab-ring3x",
        "description": "Clean slab foundation; every legal non-dig cell accepts soil.",
    },
    "trn-straight-allfree": {
        "family": "trench",
        "parent_condition": "trn-straight-side2",
        "description": "Clean straight trench; every legal non-dig cell accepts soil.",
    },
}
COLORS = np.asarray(
    [
        (240, 227, 194),  # neutral
        (230, 138, 46),  # dig
        (81, 168, 104),  # accepted dump
        (158, 163, 168),  # non-dumpable
        (32, 33, 36),  # occupied
    ],
    dtype=np.uint8,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain nonempty JSON objects")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    if not token:
        raise ValueError(f"cannot make a filesystem token from {value!r}")
    return token


def make_allfree_arrays(
    target: np.ndarray,
    occupancy: np.ndarray,
    dumpability: np.ndarray,
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return reset arrays with every legal non-dig cell accepted for dumping."""
    arrays = [np.asarray(value) for value in (target, occupancy, dumpability, action)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("control arrays must be two dimensional")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("control arrays must have identical shapes")
    target_array, occupancy_array, dumpability_array, action_array = arrays
    if not np.all(np.isin(target_array, (-1, 0, 1))):
        raise ValueError("target contains values outside {-1, 0, 1}")

    occupied = occupancy_array.astype(np.bool_)
    dumpable = dumpability_array.astype(np.bool_)
    dig = target_array < 0
    if np.any(dig & occupied):
        raise ValueError("parent dig target overlaps occupancy")
    if np.any(action_array):
        raise ValueError("capability controls require untouched full resets")

    accepted = ~dig & ~occupied & dumpable
    transformed = np.zeros(target_array.shape, dtype=np.int8)
    transformed[dig] = -1
    transformed[accepted] = 1
    distance = compute_geodesic_distance(transformed, occupied).astype(np.float32)
    return {
        "images": transformed,
        "occupancy": np.asarray(occupancy_array).copy(),
        "dumpability": np.asarray(dumpability_array).copy(),
        "actions": np.asarray(action_array).copy(),
        "distance": distance,
    }


def _load_parent_arrays(dataset: Path, slot: int) -> dict[str, np.ndarray]:
    return {
        folder: np.load(dataset / folder / f"img_{slot}.npy", allow_pickle=False)
        for folder in RESET_ARRAY_FOLDERS
    }


def _parent_training_sources(
    parent_bank: Path, parent_summary: dict[str, Any]
) -> dict[str, Path]:
    entries = parent_summary.get("train")
    if not isinstance(entries, list):
        raise ValueError("parent bank has no train-level registry")
    by_condition = {
        str(entry["condition_id"]): parent_bank / str(entry["maps_path"])
        for entry in entries
    }
    missing = {
        definition["parent_condition"]
        for definition in CONTROL_DEFINITIONS.values()
        if definition["parent_condition"] not in by_condition
    }
    if missing:
        raise ValueError(f"parent bank is missing control parents: {sorted(missing)}")
    return by_condition


def _source_rows(
    parent_bank: Path,
    parent_summary: dict[str, Any],
    training_sources: dict[str, Path],
    split: str,
    control_id: str,
) -> tuple[Path, list[dict[str, Any]]]:
    parent_condition = CONTROL_DEFINITIONS[control_id]["parent_condition"]
    if split == "train":
        dataset = training_sources[parent_condition]
        rows = _read_jsonl(dataset / "manifest.jsonl")
    else:
        panel = parent_summary.get("evaluation_panels", {}).get(split)
        if not isinstance(panel, dict) or not isinstance(panel.get("maps_path"), str):
            raise ValueError(f"parent bank is missing evaluation panel {split!r}")
        dataset = parent_bank / panel["maps_path"]
        rows = [
            row
            for row in _read_jsonl(dataset / "manifest.jsonl")
            if row.get("primary_cell") == parent_condition
        ]
    rows = sorted(rows, key=lambda row: (str(row["map_id"]), int(row["slot_index"])))
    if not rows:
        raise ValueError(f"{split}/{parent_condition}: no parent maps")
    if any(
        row.get("family") != CONTROL_DEFINITIONS[control_id]["family"] for row in rows
    ):
        raise ValueError(f"{split}/{parent_condition}: family mismatch")
    return dataset, rows


def _make_record(
    *,
    source_dataset: Path,
    parent_row: dict[str, Any],
    output_dataset: Path,
    output_slot: int,
    control_id: str,
    split: str,
) -> dict[str, Any]:
    parent_slot = int(parent_row["slot_index"])
    parent_arrays = _load_parent_arrays(source_dataset, parent_slot)
    transformed = make_allfree_arrays(
        parent_arrays["images"],
        parent_arrays["occupancy"],
        parent_arrays["dumpability"],
        parent_arrays["actions"],
    )
    # Both selected parents are explicitly clean.  Fail closed if a future bank
    # silently adds a site restriction to either capability floor.
    if np.any(transformed["occupancy"]):
        raise ValueError(f"{parent_row['map_id']}: allfree parent is not obstacle-free")
    if not np.all(transformed["dumpability"]):
        raise ValueError(
            f"{parent_row['map_id']}: allfree parent is not fully dumpable"
        )

    for folder, array in transformed.items():
        np.save(output_dataset / folder / f"img_{output_slot}.npy", array)

    parent_metadata_path = source_dataset / "metadata" / f"trench_{parent_slot}.json"
    metadata = _read_json(parent_metadata_path)
    metadata.update(
        {
            "control_condition_id": control_id,
            "control_schema": CONTROL_SCHEMA,
            "dump_layout": "allfree",
            "parent_condition_id": parent_row["primary_cell"],
            "parent_map_id": parent_row["map_id"],
        }
    )
    _write_json(output_dataset / "metadata" / f"trench_{output_slot}.json", metadata)

    scenario_id = reset_array_scenario_sha256(transformed)
    dig_volume = int((transformed["images"] < 0).sum())
    dump_cells = int((transformed["images"] > 0).sum())
    map_id = f"{control_id}:{split}:{scenario_id[:16]}"
    return {
        "slot_index": output_slot,
        "map_id": map_id,
        "scenario_id": scenario_id,
        "source_id": parent_row["source_id"],
        "split": split,
        "family": CONTROL_DEFINITIONS[control_id]["family"],
        "stratum": CONTROL_STRATUM,
        "primary_cell": control_id,
        "slot_weight": 1.0,
        "identity_slot_multiplicity": 1,
        "pair_slot_id": parent_row.get("pair_slot_id", parent_row["source_id"]),
        "parent_condition_id": parent_row["primary_cell"],
        "parent_map_id": parent_row["map_id"],
        "parent_scenario_id": parent_row.get("scenario_id"),
        "accepted_dump_cells": dump_cells,
        "required_dig_volume": dig_volume,
        "single_layer_capacity_ratio": dump_cells / dig_volume,
        "control_definition": "all_legal_non_dig_cells",
    }


def _dataset_metadata(
    *,
    count: int,
    shape: tuple[int, int],
    source_registry: Path,
    output_dataset: Path,
) -> dict[str, Any]:
    return {
        "schema": EXACT_DATASET_SCHEMA,
        "slot_count": count,
        "unique_identity_count": count,
        "shape": list(shape),
        "distance_metric": DISTANCE_METRIC,
        "distance_normalization": DISTANCE_NORMALIZATION,
        "accepted_dump_contract": ACCEPTED_DUMP_CONTRACT,
        "scenario_identity_contract": RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
        "source_registry": os.path.relpath(source_registry, output_dataset),
        "source_registry_sha256": _sha256_file(source_registry),
        "control_schema": CONTROL_SCHEMA,
        "included_in_constrained_macro": False,
    }


def _prepare_dataset(path: Path) -> None:
    for folder in (*RESET_ARRAY_FOLDERS, "metadata"):
        (path / folder).mkdir(parents=True, exist_ok=True)


def _render_code(
    target: np.ndarray, occupancy: np.ndarray, dumpability: np.ndarray
) -> Image.Image:
    occupied = occupancy.astype(np.bool_)
    dumpable = dumpability.astype(np.bool_)
    code = np.zeros(target.shape, dtype=np.uint8)
    code[target < 0] = 1
    code[target > 0] = 2
    code[~dumpable & ~occupied] = 3
    code[occupied] = 4
    return Image.fromarray(COLORS[code], mode="RGB").resize(
        (256, 256), Image.Resampling.NEAREST
    )


def _render_gallery(
    parent_bank: Path,
    parent_summary: dict[str, Any],
    training_sources: dict[str, Path],
    output_root: Path,
) -> None:
    review = output_root / "review"
    review.mkdir()
    gallery_paths = []
    for control_id in sorted(CONTROL_DEFINITIONS):
        parent_dataset, parent_rows = _source_rows(
            parent_bank,
            parent_summary,
            training_sources,
            "development",
            control_id,
        )
        control_rows = [
            row
            for row in _read_jsonl(output_root / "development" / "manifest.jsonl")
            if row["primary_cell"] == control_id
        ]
        count = min(8, len(parent_rows))
        positions = (
            [0]
            if count == 1
            else [
                index * (len(parent_rows) - 1) // (count - 1) for index in range(count)
            ]
        )
        row_height = 286
        canvas = Image.new("RGB", (512, 38 + count * row_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 7),
            f"{control_id}: constrained parent (left) vs all-free control (right)",
            fill="black",
        )
        for gallery_row, position in enumerate(positions):
            parent = parent_rows[position]
            control = control_rows[position]
            parent_arrays = _load_parent_arrays(
                parent_dataset, int(parent["slot_index"])
            )
            control_arrays = _load_parent_arrays(
                output_root / "development", int(control["slot_index"])
            )
            y = 38 + gallery_row * row_height
            canvas.paste(
                _render_code(
                    parent_arrays["images"],
                    parent_arrays["occupancy"],
                    parent_arrays["dumpability"],
                ),
                (0, y),
            )
            canvas.paste(
                _render_code(
                    control_arrays["images"],
                    control_arrays["occupancy"],
                    control_arrays["dumpability"],
                ),
                (256, y),
            )
            draw.text(
                (8, y + 260),
                f"source={parent['source_id'][-12:]}  parent={parent['primary_cell']}",
                fill="black",
            )
        path = review / f"{control_id}.png"
        canvas.save(path)
        gallery_paths.append(path.name)

    (review / "README.md").write_text(
        "# Unconstrained capability-floor review\n\n"
        "Orange is excavation; green is accepted dumping. Each image compares "
        "the constrained P5 parent on the left with the exact same dig geometry "
        "and an all-free dump mask on the right.\n\n"
        + "\n".join(f"- [{name}]({name})" for name in gallery_paths)
        + "\n"
    )


def build_control_bank(parent_bank: Path, output: Path) -> dict[str, Any]:
    parent_bank = parent_bank.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    parent_summary = _read_json(parent_bank / "dataset.json")
    if parent_summary.get("schema") != LOADER_BANK_SCHEMA:
        raise ValueError(f"{parent_bank} is not a Terra loader bank")
    training_sources = _parent_training_sources(parent_bank, parent_summary)

    protocol_path = parent_bank / str(parent_summary["environment_protocol"])
    protocol = _read_json(protocol_path)
    protocol_sha = protocol.get("environment_protocol_sha256")
    if protocol_sha != parent_summary.get("environment_protocol_sha256"):
        raise ValueError("parent environment protocol hash mismatch")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.tmp-"
    ) as temporary:
        root = Path(temporary)
        shutil.copy2(protocol_path, root / "environment_protocol.json")
        all_records: list[dict[str, Any]] = []
        datasets: list[tuple[Path, list[dict[str, Any]]]] = []
        training_levels = []
        common_shape: tuple[int, int] | None = None

        for level_index, control_id in enumerate(sorted(CONTROL_DEFINITIONS)):
            source_dataset, parent_rows = _source_rows(
                parent_bank,
                parent_summary,
                training_sources,
                "train",
                control_id,
            )
            relative = Path("train") / f"{level_index:03d}__{_safe_token(control_id)}"
            destination = root / relative
            _prepare_dataset(destination)
            records = [
                _make_record(
                    source_dataset=source_dataset,
                    parent_row=parent_row,
                    output_dataset=destination,
                    output_slot=slot,
                    control_id=control_id,
                    split="train",
                )
                for slot, parent_row in enumerate(parent_rows, start=1)
            ]
            _write_jsonl(destination / "manifest.jsonl", records)
            datasets.append((destination, records))
            all_records.extend(records)
            shape = np.load(destination / "images" / "img_1.npy").shape
            if common_shape is None:
                common_shape = (int(shape[0]), int(shape[1]))
            elif tuple(shape) != common_shape:
                raise ValueError("control map shapes differ")
            training_levels.append(
                {
                    "level_index": level_index,
                    "condition_id": control_id,
                    "family": CONTROL_DEFINITIONS[control_id]["family"],
                    # Existing loaders accept the three benchmark depth tokens.
                    # The separate control_schema keeps this out of anchor macros.
                    "branch_depth": "Anchor",
                    "maps_path": relative.as_posix(),
                    "map_count": len(records),
                }
            )

        evaluation_panels = {}
        for split in EVALUATION_SPLITS:
            destination = root / split
            _prepare_dataset(destination)
            records = []
            sources = []
            for control_id in sorted(CONTROL_DEFINITIONS):
                source_dataset, parent_rows = _source_rows(
                    parent_bank,
                    parent_summary,
                    training_sources,
                    split,
                    control_id,
                )
                sources.extend((control_id, source_dataset, row) for row in parent_rows)
            for slot, (control_id, source_dataset, parent_row) in enumerate(
                sources, start=1
            ):
                records.append(
                    _make_record(
                        source_dataset=source_dataset,
                        parent_row=parent_row,
                        output_dataset=destination,
                        output_slot=slot,
                        control_id=control_id,
                        split=split,
                    )
                )
            reset_seeds = _exact_reset_seeds(len(records))
            for record, reset_seed in zip(records, reset_seeds):
                record["reset_seed"] = reset_seed
                record["episode_id"] = episode_id(
                    record["scenario_id"], reset_seed, protocol_sha
                )
                record["environment_protocol_sha256"] = protocol_sha
            _write_jsonl(destination / "manifest.jsonl", records)
            datasets.append((destination, records))
            all_records.extend(records)
            evaluation_panels[split] = {
                "maps_path": split,
                "slot_count": len(records),
                "conditions": len(CONTROL_DEFINITIONS),
            }

        assert common_shape is not None
        registry_rows = [
            {
                "map_id": record["map_id"],
                "scenario_id": record["scenario_id"],
                "source_id": record["source_id"],
                "split": record["split"],
                "family": record["family"],
                "primary_cell": record["primary_cell"],
            }
            for record in all_records
        ]
        source_registry = root / "source_registry.jsonl"
        _write_jsonl(source_registry, registry_rows)

        for dataset, records in datasets:
            _write_json(
                dataset / "dataset.json",
                _dataset_metadata(
                    count=len(records),
                    shape=common_shape,
                    source_registry=source_registry,
                    output_dataset=dataset,
                ),
            )
            validate_exact_dataset_contract(dataset, len(records))

        control_contract = {
            "schema": CONTROL_SCHEMA,
            "status": "diagnostic_capability_floor",
            "included_in_constrained_macro": False,
            "definition": "target > 0 on every legal, non-dig cell",
            "reset_distribution": "untouched_full_task",
            "parent_bank": str(parent_bank),
            "parent_bank_dataset_sha256": _sha256_file(parent_bank / "dataset.json"),
            "conditions": CONTROL_DEFINITIONS,
            "splits": {
                "train": 64,
                "promotion": 16,
                "development": 16,
                "sealed": 32,
            },
        }
        _write_json(root / "control_contract.json", control_contract)
        summary = {
            "schema": LOADER_BANK_SCHEMA,
            "control_schema": CONTROL_SCHEMA,
            "control_contract": "control_contract.json",
            "control_contract_sha256": _sha256_file(root / "control_contract.json"),
            "included_in_constrained_macro": False,
            "source_registry": "source_registry.jsonl",
            "source_registry_sha256": _sha256_file(source_registry),
            "environment_protocol": "environment_protocol.json",
            "environment_protocol_sha256": protocol_sha,
            "scenario_identity_contract": RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
            "shape": list(common_shape),
            "train": training_levels,
            "evaluation_panels": evaluation_panels,
        }
        _write_json(root / "dataset.json", summary)
        _render_gallery(parent_bank, parent_summary, training_sources, root)
        root.rename(output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_control_bank(args.parent_bank, args.output)
    print(
        "built unconstrained controls: "
        f"{len(summary['train'])} conditions, "
        f"{summary['evaluation_panels']['development']['slot_count']} dev maps"
    )


if __name__ == "__main__":
    main()
