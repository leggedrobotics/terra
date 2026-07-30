#!/usr/bin/env python3
"""Export a generated curriculum bank as a small, inspectable review gallery.

The generated bank remains read-only. The gallery copies its review images and
records the source scenario hashes so that comments never become map identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MAX_EXAMPLES_PER_CONDITION = 16
FACTOR_COLUMNS = ("geometry", "dump", "capacity", "site", "distance", "scale")

BRANCHES = {
    "anchor_easy": ("00_anchor_easy", "Anchor / easy"),
    "dump_capacity": ("10_dump_capacity", "Dump capacity"),
    "dump_distance": ("20_dump_distance", "Dump distance"),
    "dump_layout": ("30_dump_layout", "Dump layout"),
    "geometry_topology": ("40_geometry_topology", "Geometry / topology"),
    "site_constraints": ("50_site_constraints", "Site constraints"),
    "composed": ("60_composed", "Composed"),
}
FACTOR_TO_BRANCH = {
    "capacity": "dump_capacity",
    "distance": "dump_distance",
    "dump": "dump_layout",
    "geometry": "geometry_topology",
    "scale": "geometry_topology",
    "site": "site_constraints",
}

INDEX_COLUMNS = (
    "branch",
    "condition_id",
    "family",
    "tier",
    "anchor_condition_id",
    *FACTOR_COLUMNS,
    "map_id",
    "map_index",
    "source_group_id",
    "scenario_sha256",
    "dig_cells",
    "dump_cells",
    "capacity_ratio",
    "object_count",
    "preview_path",
    "decision",
    "comment",
)

COLORS = np.asarray(
    [
        (240, 227, 194),  # neutral sand
        (230, 138, 46),  # dig target
        (81, 168, 104),  # dump target
        (158, 163, 168),  # non-dumpable area
        (32, 33, 36),  # obstacle
    ],
    dtype=np.uint8,
)


def read_csv(path: Path, required_columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required_columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def classify_branch(
    condition: dict[str, str], registry: dict[str, dict[str, str]]
) -> str:
    condition_id = condition["condition_id"]
    try:
        tier = int(condition["tier"])
    except ValueError as error:
        raise ValueError(
            f"{condition_id}: tier must be an integer, got {condition['tier']!r}"
        ) from error

    anchor_id = condition["anchor_condition_id"]
    if tier == 0:
        if anchor_id:
            raise ValueError(
                f"{condition_id}: tier-0 condition unexpectedly has an anchor"
            )
        return "anchor_easy"
    if not anchor_id:
        raise ValueError(f"{condition_id}: non-anchor condition has no anchor")
    if anchor_id not in registry:
        raise ValueError(
            f"{condition_id}: anchor is absent from conditions.csv: {anchor_id}"
        )

    anchor = registry[anchor_id]
    changed = [
        factor for factor in FACTOR_COLUMNS if condition[factor] != anchor[factor]
    ]
    if not changed:
        raise ValueError(f"{condition_id}: differs from its anchor on no factors")
    if len(changed) > 1:
        return "composed"
    return FACTOR_TO_BRANCH[changed[0]]


def representative_positions(map_count: int) -> list[int]:
    if map_count <= 0:
        raise ValueError(f"map count must be positive, got {map_count}")
    count = min(map_count, MAX_EXAMPLES_PER_CONDITION)
    if count == 1:
        return [0]
    return [position * (map_count - 1) // (count - 1) for position in range(count)]


def _bank_file(bank: Path, relative_path: str) -> Path:
    path = (bank / relative_path).resolve()
    if not path.is_relative_to(bank.resolve()):
        raise ValueError(f"bank manifest path escapes the bank: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"bank manifest file is missing: {path}")
    return path


def _render_from_arrays(bank: Path, map_record: dict[str, Any], output: Path) -> None:
    arrays = map_record.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError(f"{map_record.get('id')}: arrays must be a mapping")
    required = {"images", "occupancy", "dumpability"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(
            f"{map_record.get('id')}: array paths are missing: {sorted(missing)}"
        )

    target = np.load(_bank_file(bank, arrays["images"]), allow_pickle=False)
    occupancy = np.load(_bank_file(bank, arrays["occupancy"]), allow_pickle=False)
    dumpability = np.load(_bank_file(bank, arrays["dumpability"]), allow_pickle=False)
    if (
        target.ndim != 2
        or occupancy.shape != target.shape
        or dumpability.shape != target.shape
    ):
        raise ValueError(
            f"{map_record.get('id')}: expected matching 2-D target, occupancy, "
            f"and dumpability arrays"
        )

    code = np.zeros(target.shape, dtype=np.uint8)
    code[target < 0] = 1
    code[target > 0] = 2
    code[~dumpability.astype(bool) & ~occupancy.astype(bool)] = 3
    code[occupancy.astype(bool)] = 4
    image = Image.fromarray(COLORS[code], mode="RGB")
    image.resize((512, 512), Image.Resampling.NEAREST).save(output)


def _copy_or_render_preview(
    bank: Path,
    source_condition: Path,
    map_record: dict[str, Any],
    output: Path,
) -> None:
    map_id = map_record["id"]
    matches = sorted((source_condition / "previews").glob(f"*__{map_id}.png"))
    if len(matches) > 1:
        raise ValueError(f"{map_id}: multiple source previews found: {matches}")
    if matches:
        shutil.copy2(matches[0], output)
        return
    _render_from_arrays(bank, map_record, output)


def _write_overview(example_paths: list[Path], output: Path) -> None:
    columns = min(4, len(example_paths))
    rows = (len(example_paths) + columns - 1) // columns
    tile_size = 260
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for index, path in enumerate(example_paths):
        with Image.open(path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((tile_size - 8, tile_size - 8))
            x = (index % columns) * tile_size + (tile_size - tile.width) // 2
            y = (index // columns) * tile_size + (tile_size - tile.height) // 2
            canvas.paste(tile, (x, y))
    canvas.save(output)


def _validate_condition(
    bank: Path,
    manifest_path: Path,
    registry_row: dict[str, str],
    root_rows: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    condition = json.loads(manifest_path.read_text())
    required = {
        "conditionId",
        "family",
        "tier",
        "anchorConditionId",
        "factorLevels",
        "mapCount",
        "maps",
    }
    missing = required - condition.keys()
    if missing:
        raise ValueError(f"{manifest_path} is missing fields: {sorted(missing)}")

    condition_id = condition["conditionId"]
    if manifest_path.parent.name != condition_id:
        raise ValueError(
            f"{manifest_path}: directory does not match conditionId {condition_id}"
        )
    if condition["family"] != registry_row["family"]:
        raise ValueError(f"{condition_id}: family differs between manifests")
    if int(condition["tier"]) != int(registry_row["tier"]):
        raise ValueError(f"{condition_id}: tier differs between manifests")
    if (condition["anchorConditionId"] or "") != registry_row["anchor_condition_id"]:
        raise ValueError(f"{condition_id}: anchor differs between manifests")

    levels = condition["factorLevels"]
    for factor in FACTOR_COLUMNS:
        if factor not in levels:
            raise ValueError(f"{condition_id}: factorLevels is missing {factor}")
        if levels[factor] != registry_row[factor]:
            raise ValueError(f"{condition_id}: {factor} differs between manifests")

    maps = condition["maps"]
    if not isinstance(maps, list) or int(condition["mapCount"]) != len(maps):
        raise ValueError(f"{condition_id}: mapCount does not match maps")
    if not maps:
        raise ValueError(f"{condition_id}: empty generated condition")

    ids = [map_record.get("id") for map_record in maps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{condition_id}: duplicate map ids")
    if set(ids) != set(root_rows):
        raise ValueError(
            f"{condition_id}: condition and root manifests list different maps"
        )

    for map_record in maps:
        map_id = map_record["id"]
        root = root_rows[map_id]
        expected = {
            "map_index": str(map_record["mapIndex"]),
            "source_group_id": map_record["sourceGroupId"],
            "scenario_sha256": map_record["scenarioSha256"],
        }
        for field, value in expected.items():
            if root[field] != value:
                raise ValueError(
                    f"{condition_id}/{map_id}: {field} differs between manifests"
                )
    return condition, sorted(maps, key=lambda record: int(record["mapIndex"]))


def export_review_gallery(bank: Path, output: Path) -> None:
    bank = bank.resolve()
    output = output.resolve()
    if not bank.is_dir():
        raise NotADirectoryError(f"bank does not exist: {bank}")
    if output == bank or output.is_relative_to(bank):
        raise ValueError("review output must be outside the generated bank")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"review output is not empty: {output}")

    condition_rows = read_csv(
        bank / "conditions.csv",
        ("condition_id", "family", "tier", "anchor_condition_id", *FACTOR_COLUMNS),
    )
    registry = {row["condition_id"]: row for row in condition_rows}
    if len(registry) != len(condition_rows):
        raise ValueError("conditions.csv contains duplicate condition ids")

    manifest_rows = read_csv(
        bank / "manifest.csv",
        (
            "condition_id",
            "map_id",
            "map_index",
            "source_group_id",
            "scenario_sha256",
        ),
    )
    root_by_condition: dict[str, dict[str, dict[str, str]]] = {}
    for row in manifest_rows:
        condition_id = row["condition_id"]
        maps = root_by_condition.setdefault(condition_id, {})
        if row["map_id"] in maps:
            raise ValueError(f"manifest.csv contains duplicate map id: {row['map_id']}")
        maps[row["map_id"]] = row

    manifest_paths = sorted(bank.glob("*/manifest.json"))
    if not manifest_paths:
        raise ValueError(f"no condition manifests found under {bank}")
    generated_ids = {path.parent.name for path in manifest_paths}
    if generated_ids != set(root_by_condition):
        raise ValueError(
            "root manifest and condition directories list different conditions"
        )

    output.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    markdown_by_branch: dict[str, list[str]] = {branch: [] for branch in BRANCHES}

    for manifest_path in manifest_paths:
        condition_id = manifest_path.parent.name
        if condition_id not in registry:
            raise ValueError(f"{condition_id}: absent from conditions.csv")
        registry_row = registry[condition_id]
        branch = classify_branch(registry_row, registry)
        branch_directory, _ = BRANCHES[branch]
        destination = output / branch_directory / condition_id
        examples = destination / "examples"
        examples.mkdir(parents=True)

        condition, maps = _validate_condition(
            bank,
            manifest_path,
            registry_row,
            root_by_condition[condition_id],
        )
        chosen = [maps[position] for position in representative_positions(len(maps))]
        example_paths: list[Path] = []
        for map_record in chosen:
            map_id = map_record["id"]
            map_index = int(map_record["mapIndex"])
            filename = f"{map_index:03d}__{map_id}.png"
            preview = examples / filename
            _copy_or_render_preview(bank, manifest_path.parent, map_record, preview)
            example_paths.append(preview)
            index_rows.append(
                {
                    "branch": branch,
                    "condition_id": condition_id,
                    "family": condition["family"],
                    "tier": condition["tier"],
                    "anchor_condition_id": condition["anchorConditionId"] or "",
                    **{
                        factor: condition["factorLevels"][factor]
                        for factor in FACTOR_COLUMNS
                    },
                    "map_id": map_id,
                    "map_index": map_index,
                    "source_group_id": map_record["sourceGroupId"],
                    "scenario_sha256": map_record["scenarioSha256"],
                    "dig_cells": map_record["digCells"],
                    "dump_cells": map_record["dumpCells"],
                    "capacity_ratio": map_record["capacityRatio"],
                    "object_count": map_record["objectCount"],
                    "preview_path": preview.relative_to(output).as_posix(),
                    "decision": "",
                    "comment": "",
                }
            )

        _write_overview(example_paths, destination / "overview.png")

        relative_overview = (
            (destination / "overview.png").relative_to(output).as_posix()
        )
        factors = ", ".join(
            f"{factor}={condition['factorLevels'][factor]}" for factor in FACTOR_COLUMNS
        )
        markdown_by_branch[branch].extend(
            [
                f"### `{condition_id}`",
                "",
                f"{condition['family']}; tier {condition['tier']}; "
                f"{len(maps)} maps; {len(chosen)} shown.",
                "",
                f"Factors: {factors}.",
                "",
                f"![{condition_id}]({relative_overview})",
                "",
            ]
        )

    with (output / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(index_rows)

    markdown = [
        "# Terra curriculum map review",
        "",
        f"Source bank: `{bank}`",
        "",
        "The gallery is grouped by independent curriculum branch. `index.csv` "
        "pins every displayed map to its source scenario hash and provides blank "
        "`decision` and `comment` columns for review.",
        "",
    ]
    for branch, (_, label) in BRANCHES.items():
        if not markdown_by_branch[branch]:
            continue
        markdown.extend([f"## {label}", "", *markdown_by_branch[branch]])
    (output / "index.md").write_text("\n".join(markdown) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a generated Terra curriculum bank for visual review."
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_review_gallery(args.bank, args.output)


if __name__ == "__main__":
    main()
