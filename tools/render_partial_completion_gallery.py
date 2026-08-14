#!/usr/bin/env python3
"""Render partial-reset action maps and relay-corridor diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw

import terra.env_generation.partial_completion as partial_completion
from terra.env_generation.partial_completion import PartialCompletionConfig
from terra.env_generation.partial_completion import _relay_corridor_masks
from terra.env_generation.partial_completion import compute_dynamic_dumpability_numpy
from terra.env_generation.partial_completion import validate_partial_state

CELL = 8
MAP_PIXELS = 64 * CELL
TEXT_PIXELS = 132
COLORS = {
    "background": np.asarray((239, 224, 190), dtype=np.uint8),
    "corridor": np.asarray((190, 220, 229), dtype=np.uint8),
    "remaining": np.asarray((225, 132, 44), dtype=np.uint8),
    "hole": np.asarray((62, 80, 103), dtype=np.uint8),
    "dump": np.asarray((77, 164, 98), dtype=np.uint8),
    "obstacle": np.asarray((31, 32, 35), dtype=np.uint8),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_layer(dataset: Path, folder: str, index: int) -> np.ndarray:
    path = dataset / folder / f"img_{index}.npy"
    value = np.asarray(np.load(path, allow_pickle=False)).squeeze()
    if value.shape != (64, 64):
        raise ValueError(f"{path}: expected 64x64, got {value.shape}")
    return value


def _blend(base: np.ndarray, mask: np.ndarray, color: np.ndarray) -> None:
    base[mask] = ((base[mask].astype(np.uint16) + color) // 2).astype(np.uint8)


def _pile_color(heights: np.ndarray) -> np.ndarray:
    normalized = np.clip(heights.astype(np.float32) / max(1, int(heights.max())), 0, 1)
    low = np.asarray((194, 139, 75), dtype=np.float32)
    high = np.asarray((113, 63, 34), dtype=np.float32)
    return (low + normalized[..., None] * (high - low)).astype(np.uint8)


def _cell_center(coordinate: list[int] | tuple[int, int]) -> tuple[int, int]:
    row, column = int(coordinate[0]), int(coordinate[1])
    return column * CELL + CELL // 2, row * CELL + CELL // 2


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    draw.line((start, end), fill=color, width=3)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(1.0, float(np.hypot(dx, dy)))
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = end
    back = (end[0] - 10 * ux, end[1] - 10 * uy)
    left = (back[0] + 5 * px, back[1] + 5 * py)
    right = (back[0] - 5 * px, back[1] - 5 * py)
    draw.polygon((tip, left, right), fill=color)


def _distance_descent_path(
    start: list[int] | tuple[int, int], distance: np.ndarray
) -> list[list[int]]:
    """Follow one deterministic four-neighbor shortest path to distance zero."""
    row, column = int(start[0]), int(start[1])
    path = [[row, column]]
    while int(distance[row, column]) > 0:
        current = int(distance[row, column])
        candidates = []
        for next_row, next_column in (
            (row - 1, column),
            (row, column - 1),
            (row, column + 1),
            (row + 1, column),
        ):
            if (
                0 <= next_row < distance.shape[0]
                and 0 <= next_column < distance.shape[1]
                and int(distance[next_row, next_column]) == current - 1
            ):
                candidates.append((next_row, next_column))
        if not candidates:
            raise ValueError(f"No distance-descent path from {start}")
        row, column = min(candidates)
        path.append([row, column])
    return path


def _draw_path_arrow(
    draw: ImageDraw.ImageDraw,
    path: list[list[int]],
    color: tuple[int, int, int],
) -> None:
    pixels = [_cell_center(coordinate) for coordinate in path]
    if len(pixels) < 2:
        return
    draw.line(pixels, fill=color, width=3)
    _draw_arrow(draw, pixels[-2], pixels[-1], color)


def _render_entry(
    dataset: Path,
    record: dict,
    config: PartialCompletionConfig,
    output: Path,
) -> dict:
    index = int(record["output_index"])
    target = _load_layer(dataset, "images", index)
    occupancy = _load_layer(dataset, "occupancy", index).astype(bool)
    dumpability = _load_layer(dataset, "dumpability", index).astype(bool)
    action = _load_layer(dataset, "actions", index).astype(np.int16)
    if int(action[action > 0].sum()) != -int(action[action < 0].sum()):
        raise ValueError(f"{dataset}/img_{index}: soil mass is not conserved")
    diagnostics = validate_partial_state(
        target,
        occupancy,
        dumpability,
        action,
        config=config,
        expected_mode=str(record["pile_mode"]),
    )
    for key in (
        "negative_volume",
        "positive_volume",
        "remaining_component_sizes",
        "maximum_pile_height",
        "positive_support_area",
        "positive_component_count",
        "valid_spawn_center_count",
        "minimum_conservative_workspace_pickups_for_staged_volume",
        "relay_direct_service_center_count",
        "relay_no_shared_conservative_proxy_center",
    ):
        if diagnostics.get(key) != record.get(key):
            raise ValueError(
                f"{dataset}/img_{index}: manifest {key} differs from validation"
            )

    dynamic = compute_dynamic_dumpability_numpy(
        dumpability,
        np.where(action < 0, action, 0),
    )
    corridor = np.zeros_like(occupancy)
    corridor_core = np.zeros_like(occupancy)
    corridor_data = None
    if record.get("pile_mode") == "relay_corridor":
        corridor, _, corridor_data = _relay_corridor_masks(
            target,
            occupancy,
            dynamic,
            action < 0,
        )
        corridor_core = corridor_data["route_core"]
        if int(record["pile_count"]) != 1 or len(record.get("piles", [])) != 1:
            raise ValueError(f"{dataset}/img_{index}: relay mode requires one pile")
        pile = record["piles"][0]
        center = tuple(int(value) for value in pile["center"])
        if not corridor_core[center] or action[center] <= 0:
            raise ValueError(
                f"{dataset}/img_{index}: pile center is not on the route core"
            )
        if int(pile["volume"]) != int(action[action > 0].sum()):
            raise ValueError(f"{dataset}/img_{index}: pile manifest loses soil mass")
        if int(pile["route_excess_tiles"]) != int(
            corridor_data["route_excess"][center]
        ):
            raise ValueError(
                f"{dataset}/img_{index}: pile route-excess diagnostic is stale"
            )

    pixels = np.broadcast_to(COLORS["background"], (64, 64, 3)).copy()
    _blend(pixels, corridor & (target == 0), COLORS["corridor"])
    _blend(
        pixels,
        corridor_core & (target == 0),
        np.asarray((107, 181, 206), dtype=np.uint8),
    )
    pixels[(target < 0) & (action >= 0)] = COLORS["remaining"]
    pixels[(target < 0) & (action < 0)] = COLORS["hole"]
    pixels[target > 0] = COLORS["dump"]
    positive = action > 0
    pixels[positive] = _pile_color(np.where(positive, action, 0))[positive]
    pixels[occupancy] = COLORS["obstacle"]

    map_image = Image.fromarray(pixels, mode="RGB").resize(
        (MAP_PIXELS, MAP_PIXELS), Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (MAP_PIXELS, MAP_PIXELS + TEXT_PIXELS), "white")
    canvas.paste(map_image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    source_anchor = record.get("relay_source_anchor")
    piles = sorted(
        record.get("piles", []),
        key=lambda pile: int(pile.get("terminal_distance_tiles", 0)),
        reverse=True,
    )
    pile_centers = [pile["center"] for pile in piles]
    if source_anchor and pile_centers and corridor_data is not None:
        center = pile_centers[0]
        source_path = list(
            reversed(_distance_descent_path(center, corridor_data["source_distance"]))
        )
        terminal_path = _distance_descent_path(
            center, corridor_data["terminal_distance"]
        )
        _draw_path_arrow(draw, source_path, (178, 48, 38))
        _draw_path_arrow(draw, terminal_path, (178, 48, 38))
        current = _cell_center(center)
        draw.ellipse(
            (current[0] - 5, current[1] - 5, current[0] + 5, current[1] + 5),
            outline=(255, 230, 40),
            width=3,
        )
        source_pixel = _cell_center(source_anchor)
        draw.rectangle(
            (
                source_pixel[0] - 6,
                source_pixel[1] - 6,
                source_pixel[0] + 6,
                source_pixel[1] + 6,
            ),
            outline=(235, 235, 235),
            width=3,
        )

    label_y = MAP_PIXELS + 6
    condition = dataset.name
    lines = [
        f"{condition} | source img {record['source_index']} | fraction {record['achieved_completion_fraction']:.2f}",
        f"mass {record['positive_volume']} | piles {record['pile_count']} | conservative workspace pickups >= {record['minimum_conservative_workspace_pickups_for_staged_volume']}",
        f"route {record.get('relay_shortest_route_tiles', 'n/a')} tiles | no shared conservative proxy center {record.get('relay_no_shared_conservative_proxy_center', 'n/a')}",
        f"center route excess {max(int(pile.get('route_excess_tiles', 0)) for pile in piles)} / {record.get('relay_center_max_route_excess_tiles', 'n/a')} tiles | static only",
    ]
    for line in lines:
        draw.text((8, label_y), line, fill=(20, 20, 20))
        label_y += 26
    canvas.save(output)
    return {
        "condition": condition,
        "output_index": index,
        "source_index": int(record["source_index"]),
        "completion_fraction": float(record["achieved_completion_fraction"]),
        "image": output.name,
        "image_sha256": _sha256(output),
        "positive_volume": int(record["positive_volume"]),
        "pile_count": int(record["pile_count"]),
        "no_shared_conservative_proxy_center": bool(
            record.get("relay_no_shared_conservative_proxy_center", False)
        ),
        "action_sha256": _sha256(dataset / "actions" / f"img_{index}.npy"),
    }


def render_gallery(inputs: list[Path], output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    image_paths: list[Path] = []
    input_receipts: list[dict] = []
    for dataset in inputs:
        manifest = dataset / "partial_completion_manifest.jsonl"
        config_path = dataset / "partial_completion_config.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config_payload = json.loads(config_path.read_text())
        config_values = dict(config_payload["config"])
        config_values["completion_fractions"] = tuple(
            config_values["completion_fractions"]
        )
        config_values["mode_weights"] = tuple(
            tuple(pair) for pair in config_values["mode_weights"]
        )
        config = PartialCompletionConfig(**config_values)
        input_receipts.append(
            {
                "dataset": str(dataset.resolve()),
                "manifest_sha256": _sha256(manifest),
                "config_sha256": _sha256(config_path),
            }
        )
        for record in map(json.loads, manifest.read_text().splitlines()):
            filename = (
                f"{dataset.name}__f{float(record['achieved_completion_fraction']):.2f}"
                f"__img{int(record['source_index'])}.png"
            )
            path = output / filename
            entries.append(_render_entry(dataset, record, config, path))
            image_paths.append(path)

    columns = min(3, len(image_paths))
    rows = (len(image_paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * MAP_PIXELS, rows * (MAP_PIXELS + TEXT_PIXELS)),
        (242, 242, 242),
    )
    for number, path in enumerate(image_paths):
        with Image.open(path) as image:
            x = (number % columns) * MAP_PIXELS
            y = (number // columns) * (MAP_PIXELS + TEXT_PIXELS)
            sheet.paste(image, (x, y))
    sheet_path = output / "contact_sheet.png"
    sheet.save(sheet_path)

    rows_html = []
    for entry in entries:
        rows_html.append(
            "<figure><img src='{}'><figcaption>{}, f={:.2f}, mass={}, "
            "no shared conservative proxy center={}</figcaption></figure>".format(
                html.escape(entry["image"]),
                html.escape(entry["condition"]),
                entry["completion_fraction"],
                entry["positive_volume"],
                entry["no_shared_conservative_proxy_center"],
            )
        )
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Terra relay resets</title>"
        "<style>body{font-family:sans-serif;background:#eee}main{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px}"
        "figure{margin:0;background:white;padding:8px}img{width:100%;image-rendering:"
        "pixelated}figcaption{padding:6px}</style><h1>Terra relay-corridor resets</h1>"
        "<p>Blue is the admissible corridor pocket; red paths are obstacle-aware "
        "shortest paths from source through the staged pile to the nearest terminal "
        "cell. Geometry checks are static, not an action witness.</p>"
        f"<main>{''.join(rows_html)}</main>",
        encoding="utf-8",
    )
    receipt = {
        "status": "passed",
        "generator_source_sha256": _sha256(Path(partial_completion.__file__)),
        "renderer_source_sha256": _sha256(Path(__file__)),
        "validated_entry_count": len(entries),
        "inputs": input_receipts,
        "entries": entries,
        "contact_sheet": sheet_path.name,
        "contact_sheet_sha256": _sha256(sheet_path),
    }
    (output / "gallery_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(entries)} relay resets to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_gallery([path.resolve() for path in args.inputs], args.output.resolve())


if __name__ == "__main__":
    main()
