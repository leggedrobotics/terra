#!/usr/bin/env python3
"""Render a small v7 geometry-only review bank.

The review deliberately uses all-free dumping so geometry can be judged before
capacity, obstacles, or transport are crossed into the distribution. It is not
a Terra training bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import terra_geom


MAP_SIZE = terra_geom.MAP_SIZE
MAP_CENTRE = (MAP_SIZE - 1) / 2.0
TILE_SIZE_M = terra_geom.TILE_SIZE
TARGET_TRENCH_WIDTH_M = 1.3
TARGET_TRENCH_WIDTH_TILES = TARGET_TRENCH_WIDTH_M / TILE_SIZE_M
# The rasterizer defines full width as 2 * half_width + the centre cell.
TARGET_TRENCH_HALF_WIDTH_TILES = (TARGET_TRENCH_WIDTH_TILES - 1.0) / 2.0
GLOBAL_HEADINGS_DEG = tuple(float(value) for value in range(0, 180, 15))
REALISTIC_JUNCTION_ANGLES_DEG = (60.0, 90.0, 120.0)
JUNCTION_ANGLE_DRAW = (90.0, 90.0, 90.0, 90.0, 60.0, 120.0)
MIN_JUNCTION_SEPARATION_TILES = 8.0
MAP_MARGIN_TILES = 5

TRENCH_GEOMETRIES = (
    "straight",
    "dogleg",
    "tee",
    "cross",
    "double_t",
    "network3",
    "disconnected_pair",
)
FOUNDATION_GEOMETRIES = (
    "slab",
    "irregular",
    "courtyard",
    "bearing_walls",
    "pads",
    "courtyard_pads",
)

COLORS = {
    "dump": (174, 218, 177),
    "dig": (232, 137, 48),
    "line": (70, 72, 75),
    "text": (25, 27, 29),
    "paper": (249, 248, 244),
}


@dataclass(frozen=True)
class ReviewScenario:
    scenario_id: str
    family: str
    geometry: str
    dig: np.ndarray
    metadata: dict[str, Any]


def _direction(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=float)


def _line_coefficients(points: list[np.ndarray]) -> list[dict[str, float]]:
    """Return Terra A*x + B*y + C axes for row/column segment points."""
    axes = []
    for segment in points:
        row_1, column_1 = map(float, segment[0])
        row_2, column_2 = map(float, segment[-1])
        axes.append(
            {
                "A": float(row_2 - row_1),
                "B": float(column_1 - column_2),
                "C": float(column_2 * row_1 - column_1 * row_2),
            }
        )
    return axes


def _rotate(points: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    matrix = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    return np.asarray(points, dtype=float) @ matrix.T


def _place_segments(
    segments: list[np.ndarray], rng: np.random.Generator
) -> tuple[list[np.ndarray], float]:
    heading = float(rng.choice(GLOBAL_HEADINGS_DEG))
    centre = np.asarray(
        [MAP_CENTRE + rng.uniform(-3.0, 3.0), MAP_CENTRE + rng.uniform(-3.0, 3.0)]
    )
    return [_rotate(segment, heading) + centre for segment in segments], heading


def _junction_angle(rng: np.random.Generator) -> float:
    return float(rng.choice(JUNCTION_ANGLE_DRAW))


def _trench_segments(
    geometry: str, rng: np.random.Generator
) -> tuple[list[np.ndarray], dict[str, Any]]:
    main_length = float(rng.uniform(24.0, 36.0))
    main = np.asarray([[-main_length / 2.0, 0.0], [main_length / 2.0, 0.0]])
    segments: list[np.ndarray] = [main]
    angles: list[float] = []
    junctions: list[np.ndarray] = []
    branch_junction_count = 0
    turn_count = 0

    if geometry == "straight":
        pass
    elif geometry == "dogleg":
        turn = _junction_angle(rng)
        sign = float(rng.choice((-1.0, 1.0)))
        first = float(rng.uniform(12.0, 18.0))
        second = float(rng.uniform(12.0, 18.0))
        junction = np.zeros(2)
        segments = [
            np.vstack([-_direction(0.0) * first, junction]),
            np.vstack([junction, _direction(sign * turn) * second]),
        ]
        angles = [turn]
        junctions = [junction]
        turn_count = 1
    elif geometry in {"tee", "cross"}:
        turn = _junction_angle(rng)
        junction = np.asarray([rng.uniform(-0.12, 0.12) * main_length, 0.0])
        branch_direction = _direction(turn)
        branch_a = float(rng.uniform(9.0, 15.0))
        if geometry == "tee":
            side = float(rng.choice((-1.0, 1.0)))
            branch = np.vstack([junction, junction + side * branch_direction * branch_a])
        else:
            branch_b = float(rng.uniform(9.0, 15.0))
            branch = np.vstack(
                [junction - branch_direction * branch_b, junction + branch_direction * branch_a]
            )
        segments.append(branch)
        angles = [turn]
        junctions = [junction]
        branch_junction_count = 1
    elif geometry in {"double_t", "network3"}:
        count = 2 if geometry == "double_t" else 3
        span = float(rng.uniform(0.26, 0.34)) * main_length
        positions = (
            np.asarray([-span, span])
            if count == 2
            else np.asarray([-span, 0.0, span])
        )
        positions += rng.uniform(-0.8, 0.8, size=count)
        if np.min(np.diff(np.sort(positions))) < MIN_JUNCTION_SEPARATION_TILES:
            raise ValueError("sampled trench junctions are too close")
        # Small construction networks are intentionally orthogonal. The
        # 60/120-degree variants remain useful for single dog-legs, tees, and
        # crossings, but combining several oblique branches quickly produces
        # implausible star-like targets.
        if geometry == "double_t":
            branch_sides = (
                (-1.0, -1.0)
                if bool(rng.integers(0, 2))
                else (-1.0, 1.0)
            )
            network_style = "comb" if branch_sides[0] == branch_sides[1] else "opposed"
        else:
            patterns = (
                (-1.0, -1.0, -1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (1.0, -1.0, 1.0),
            )
            branch_sides = patterns[int(rng.integers(0, len(patterns)))]
            network_style = "comb" if len(set(branch_sides)) == 1 else "alternating"
        for index, along in enumerate(positions):
            angle = 90.0
            side = branch_sides[index]
            length = float(rng.uniform(8.0, 13.0))
            junction = np.asarray([along, 0.0])
            branch = np.vstack([junction, junction + side * _direction(angle) * length])
            segments.append(branch)
            angles.append(angle)
            junctions.append(junction)
        branch_junction_count = count
    elif geometry == "disconnected_pair":
        angle = float(rng.choice((0.0, 15.0, -15.0)))
        length_a = float(rng.uniform(16.0, 24.0))
        length_b = float(rng.uniform(12.0, 20.0))
        offset = float(rng.uniform(8.0, 12.0))
        segments = [
            np.asarray([[-length_a / 2.0, -offset / 2.0], [length_a / 2.0, -offset / 2.0]]),
            np.vstack(
                [
                    np.asarray([-length_b / 2.0, offset / 2.0]),
                    np.asarray([-length_b / 2.0, offset / 2.0])
                    + _direction(angle) * length_b,
                ]
            ),
        ]
    else:
        raise ValueError(f"unsupported trench geometry: {geometry}")

    placed, heading = _place_segments(segments, rng)
    # Junction coordinates are not consumed downstream in this review. Publish
    # their separation in the canonical frame, where rigid placement cannot change it.
    separation = None
    if len(junctions) >= 2:
        separation = float(
            min(
                np.linalg.norm(a - b)
                for index, a in enumerate(junctions)
                for b in junctions[index + 1 :]
            )
        )
    metadata = {
        "global_heading_deg": heading,
        "junction_angles_deg": angles,
        "junction_count": len(junctions),
        "branch_junction_count": branch_junction_count,
        "turn_count": turn_count,
        "junction_separation_min_tiles": separation,
        "segment_count": len(segments),
        "target_width_m": TARGET_TRENCH_WIDTH_M,
        "target_width_tiles": TARGET_TRENCH_WIDTH_TILES,
    }
    if geometry in {"double_t", "network3"}:
        metadata["network_style"] = network_style
    return placed, metadata


def _rasterize_trench(segments: list[np.ndarray]) -> np.ndarray:
    dig = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    for segment in segments:
        for start, end in zip(segment[:-1], segment[1:]):
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length < 1e-6:
                continue
            angle = math.atan2(delta[1], delta[0])
            centre = tuple((start + end) / 2.0)
            dig |= _rectangle(
                centre,
                length,
                TARGET_TRENCH_WIDTH_TILES,
                angle,
            )
        # Round each endpoint enough to avoid raster notches at a join. This
        # affects only the local junction cap, not the straight-section width.
        for point in segment:
            rows, columns = np.indices((MAP_SIZE, MAP_SIZE), dtype=float)
            dig |= (
                (rows - point[0]) ** 2 + (columns - point[1]) ** 2
                <= round(TARGET_TRENCH_HALF_WIDTH_TILES) ** 2
            )
    return dig


def _component_regions(mask: np.ndarray) -> list[tuple[int, bool]]:
    """Return (size, touches_border) for four-connected true regions."""
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[tuple[int, bool]] = []
    height, width = mask.shape
    for row, column in np.argwhere(mask):
        row = int(row)
        column = int(column)
        if visited[row, column]:
            continue
        stack = [(row, column)]
        visited[row, column] = True
        size = 0
        touches_border = False
        while stack:
            current_row, current_column = stack.pop()
            size += 1
            touches_border |= current_row in (0, height - 1) or current_column in (
                0,
                width - 1,
            )
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        regions.append((size, touches_border))
    return regions


def _components(mask: np.ndarray) -> int:
    return len(_component_regions(mask))


def _holes(mask: np.ndarray) -> int:
    return sum(not touches_border for _, touches_border in _component_regions(~mask))


def _inside_margin(mask: np.ndarray) -> bool:
    interior = np.zeros_like(mask)
    interior[
        MAP_MARGIN_TILES : MAP_SIZE - MAP_MARGIN_TILES,
        MAP_MARGIN_TILES : MAP_SIZE - MAP_MARGIN_TILES,
    ] = True
    return bool(mask.any() and np.all(~mask | interior))


def make_trench_scenario(
    geometry: str, index: int, rng: np.random.Generator
) -> ReviewScenario:
    expected_components = 2 if geometry == "disconnected_pair" else 1
    for _ in range(300):
        try:
            segments, metadata = _trench_segments(geometry, rng)
        except ValueError:
            continue
        dig = _rasterize_trench(segments)
        components = _components(dig)
        if not _inside_margin(dig) or components != expected_components:
            continue
        if not 45 <= int(dig.sum()) <= 260:
            continue
        angles = metadata["junction_angles_deg"]
        if any(value not in REALISTIC_JUNCTION_ANGLES_DEG for value in angles):
            continue
        scenario_id = f"v7-trench-{geometry}-{index:03d}"
        metadata = {
            **metadata,
            "components": components,
            "dig_cells": int(dig.sum()),
            "allfree_dump_cells": int((~dig).sum()),
            "axes_ABC": _line_coefficients(segments),
        }
        return ReviewScenario(scenario_id, "trench", geometry, dig, metadata)
    raise RuntimeError(f"could not generate {geometry} trench {index}")


def _rectangle(
    centre: tuple[float, float], length: float, width: float, angle: float
) -> np.ndarray:
    rows, columns = np.indices((MAP_SIZE, MAP_SIZE), dtype=float)
    relative_rows = rows - centre[0]
    relative_columns = columns - centre[1]
    direction = _direction(math.degrees(angle))
    normal = np.asarray([-direction[1], direction[0]])
    along = relative_rows * direction[0] + relative_columns * direction[1]
    across = relative_rows * normal[0] + relative_columns * normal[1]
    return (np.abs(along) <= length / 2.0) & (np.abs(across) <= width / 2.0)


def _foundation_mask(geometry: str, rng: np.random.Generator) -> np.ndarray:
    centre = (MAP_CENTRE + rng.uniform(-2.5, 2.5), MAP_CENTRE + rng.uniform(-2.5, 2.5))
    angle = math.radians(float(rng.choice(GLOBAL_HEADINGS_DEG)))
    length = float(rng.uniform(22.0, 34.0))
    width = float(rng.uniform(18.0, 28.0))
    outer = _rectangle(centre, length, width, angle)

    if geometry == "slab":
        return outer
    if geometry == "irregular":
        direction = _direction(math.degrees(angle))
        normal = np.asarray([-direction[1], direction[0]])
        if bool(rng.integers(0, 2)):
            # L-shaped building footprint: remove a corner notch that remains
            # open to two outer edges, so it cannot become an accidental hole.
            along_sign = float(rng.choice((-1.0, 1.0)))
            across_sign = float(rng.choice((-1.0, 1.0)))
            notch_centre = (
                np.asarray(centre)
                + direction * along_sign * length * 0.38
                + normal * across_sign * width * 0.38
            )
            notch = _rectangle(
                tuple(notch_centre),
                length * rng.uniform(0.35, 0.48),
                width * rng.uniform(0.38, 0.52),
                angle,
            )
            return outer & ~notch

        # T-shaped footprint: a stem and a perpendicular bar overlap by a
        # small but explicit amount. Both dimensions derive from the same
        # medium work envelope as the slab control.
        stem = _rectangle(centre, length * 0.82, width * 0.48, angle)
        direction_sign = float(rng.choice((-1.0, 1.0)))
        bar_centre = np.asarray(centre) + direction * direction_sign * length * 0.27
        bar = _rectangle(
            tuple(bar_centre),
            width * 0.9,
            length * 0.28,
            angle + math.pi / 2.0,
        )
        return stem | bar

    inner_length = max(7.0, length - rng.uniform(6.0, 9.0))
    inner_width = max(6.0, width - rng.uniform(6.0, 9.0))
    inner = _rectangle(centre, inner_length, inner_width, angle)
    perimeter = outer & ~inner

    if geometry == "courtyard":
        return perimeter
    if geometry == "bearing_walls":
        direction = _direction(math.degrees(angle))
        normal = np.asarray([-direction[1], direction[0]])
        wall_a = np.vstack(
            [
                np.asarray(centre) - direction * inner_length / 2.0,
                np.asarray(centre) + direction * inner_length / 2.0,
            ]
        )
        wall_b = np.vstack(
            [
                np.asarray(centre) - normal * inner_width / 2.0,
                np.asarray(centre) + normal * inner_width / 2.0,
            ]
        )
        return perimeter | _rasterize_trench([wall_a, wall_b])
    if geometry == "pads":
        result = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
        direction = _direction(math.degrees(angle))
        normal = np.asarray([-direction[1], direction[0]])
        for along in (-6.0, 6.0):
            for across in (-5.0, 5.0):
                pad_centre = np.asarray(centre) + direction * along + normal * across
                result |= _rectangle(
                    tuple(pad_centre),
                    rng.uniform(4.5, 6.5),
                    rng.uniform(4.5, 6.5),
                    angle,
                )
        return result
    if geometry == "courtyard_pads":
        result = perimeter.copy()
        direction = _direction(math.degrees(angle))
        normal = np.asarray([-direction[1], direction[0]])
        along_offset = min(2.5, inner_length / 4.0)
        across_offset = min(2.0, inner_width / 4.0)
        for along, across in (
            (-along_offset, -across_offset),
            (along_offset, across_offset),
        ):
            pad_centre = np.asarray(centre) + direction * along + normal * across
            result |= _rectangle(tuple(pad_centre), 2.0, 2.0, angle)
        return result
    raise ValueError(f"unsupported foundation geometry: {geometry}")


def make_foundation_scenario(
    geometry: str, index: int, rng: np.random.Generator
) -> ReviewScenario:
    for _ in range(200):
        dig = _foundation_mask(geometry, rng)
        if not _inside_margin(dig) or not 35 <= int(dig.sum()) <= 800:
            continue
        holes = _holes(dig)
        components = _components(dig)
        if geometry in {"courtyard", "bearing_walls", "courtyard_pads"} and holes < 1:
            continue
        if geometry == "pads" and components < 4:
            continue
        if geometry == "irregular" and (components != 1 or holes != 0):
            continue
        if geometry == "courtyard_pads" and (components != 3 or holes != 1):
            continue
        scenario_id = f"v7-foundation-{geometry}-{index:03d}"
        return ReviewScenario(
            scenario_id,
            "foundation",
            geometry,
            dig,
            {
                "components": components,
                "holes": holes,
                "dig_cells": int(dig.sum()),
                "allfree_dump_cells": int((~dig).sum()),
            },
        )
    raise RuntimeError(f"could not generate {geometry} foundation {index}")


def generate_scenarios(samples_per_geometry: int, seed: int) -> list[ReviewScenario]:
    if samples_per_geometry < 1:
        raise ValueError("samples_per_geometry must be positive")
    rng = np.random.default_rng(seed)
    scenarios = [
        make_trench_scenario(geometry, index, rng)
        for geometry in TRENCH_GEOMETRIES
        for index in range(samples_per_geometry)
    ]
    scenarios.extend(
        make_foundation_scenario(geometry, index, rng)
        for geometry in FOUNDATION_GEOMETRIES
        for index in range(samples_per_geometry)
    )
    identities = [hashlib.sha256(scenario.dig.tobytes()).hexdigest() for scenario in scenarios]
    if len(identities) != len(set(identities)):
        raise RuntimeError("geometry review contains an exact duplicate dig raster")
    return scenarios


def _scenario_image(scenario: ReviewScenario, scale: int = 4) -> Image.Image:
    pixels = np.empty((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    pixels[:] = COLORS["dump"]
    pixels[scenario.dig] = COLORS["dig"]
    map_image = Image.fromarray(pixels, mode="RGB").resize(
        (MAP_SIZE * scale, MAP_SIZE * scale), Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (MAP_SIZE * scale, MAP_SIZE * scale + 44), COLORS["paper"])
    canvas.paste(map_image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, MAP_SIZE * scale + 5), scenario.scenario_id, fill=COLORS["text"])
    detail = f"cells={scenario.metadata['dig_cells']} comp={scenario.metadata['components']}"
    if scenario.family == "foundation":
        detail += f" holes={scenario.metadata['holes']}"
    else:
        detail += f" junctions={scenario.metadata['junction_count']}"
    draw.text((6, MAP_SIZE * scale + 22), detail, fill=COLORS["text"])
    return canvas


def _render_gallery(scenarios: list[ReviewScenario], output: Path) -> None:
    columns = 4
    tile_width = MAP_SIZE * 3
    tile_height = tile_width + 44
    rows = math.ceil(len(scenarios) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), COLORS["paper"])
    for index, scenario in enumerate(scenarios):
        image = _scenario_image(scenario, scale=3)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(image, (x, y))
    canvas.save(output)


def write_review(output: Path, samples_per_geometry: int, seed: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    scenarios = generate_scenarios(samples_per_geometry, seed)

    manifest_rows = []
    for scenario in scenarios:
        directory = output / scenario.family / scenario.geometry
        directory.mkdir(parents=True, exist_ok=True)
        _scenario_image(scenario).save(directory / f"{scenario.scenario_id}.png")
        manifest_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "geometry": scenario.geometry,
                "dump_layout": "allfree",
                "dig_sha256": hashlib.sha256(scenario.dig.tobytes()).hexdigest(),
                **scenario.metadata,
            }
        )

    for family, geometries in (
        ("trench", TRENCH_GEOMETRIES),
        ("foundation", FOUNDATION_GEOMETRIES),
    ):
        for geometry in geometries:
            selected = [
                scenario
                for scenario in scenarios
                if scenario.family == family and scenario.geometry == geometry
            ]
            _render_gallery(selected, output / f"{family}__{geometry}.png")

    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows)
    )
    geometry_counts = {
        f"{family}/{geometry}": sum(
            row["family"] == family and row["geometry"] == geometry
            for row in manifest_rows
        )
        for family, geometries in (
            ("trench", TRENCH_GEOMETRIES),
            ("foundation", FOUNDATION_GEOMETRIES),
        )
        for geometry in geometries
    }
    junction_angle_counts = {
        str(int(angle)): sum(
            angle in row.get("junction_angles_deg", []) for row in manifest_rows
        )
        for angle in REALISTIC_JUNCTION_ANGLES_DEG
    }
    summary = {
        "schema": "terra_v7_geometry_review_v1",
        "seed": seed,
        "samples_per_geometry": samples_per_geometry,
        "scenario_count": len(scenarios),
        "exact_duplicate_count": 0,
        "tile_size_m": TILE_SIZE_M,
        "target_trench_width_m": TARGET_TRENCH_WIDTH_M,
        "target_trench_width_tiles": TARGET_TRENCH_WIDTH_TILES,
        "junction_angles_deg": list(REALISTIC_JUNCTION_ANGLES_DEG),
        "junction_angle_scenario_counts": junction_angle_counts,
        "geometry_counts": geometry_counts,
        "trench_scenarios_with_junctions": sum(
            row.get("junction_count", 0) > 0 for row in manifest_rows
        ),
        "trench_scenarios_with_branch_junctions": sum(
            row.get("branch_junction_count", 0) > 0 for row in manifest_rows
        ),
        "minimum_multi_junction_separation_tiles": min(
            row["junction_separation_min_tiles"]
            for row in manifest_rows
            if row.get("junction_separation_min_tiles") is not None
        ),
        "foundation_scenarios_with_holes": sum(
            row["family"] == "foundation" and row.get("holes", 0) > 0
            for row in manifest_rows
        ),
        "dump_layout": "allfree",
        "training_bank": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    trench_galleries = [
        f"- [{geometry}](trench__{geometry}.png)" for geometry in TRENCH_GEOMETRIES
    ]
    foundation_galleries = [
        f"- [{geometry}](foundation__{geometry}.png)"
        for geometry in FOUNDATION_GEOMETRIES
    ]
    (output / "README.md").write_text(
        "# Terra v7 geometry review\n\n"
        "Orange is required excavation and green is unconstrained legal dumping. "
        "This review isolates geometry; it is not a training bank.\n\n"
        f"Target trench width: {TARGET_TRENCH_WIDTH_M:.1f} m "
        f"({TARGET_TRENCH_WIDTH_TILES:.3f} tiles at the live Terra scale).\n\n"
        "Review the target shape and diversity only. Dump distance, capacity, "
        "obstacles, and transport are deliberately absent from this pass.\n\n"
        "Suggested order: straight/slab controls, single-junction trenches, "
        "multi-junction trenches, then foundations with holes/walls/pads.\n\n"
        "## Trenches\n\n"
        + "\n".join(trench_galleries)
        + "\n\n## Foundations\n\n"
        + "\n".join(foundation_galleries)
        + "\n\n## What to comment on\n\n"
        "For each stratum: accept/reject, realistic or not, too repetitive, "
        "and too small/large. Refer to the scenario ID printed below a map "
        "when commenting on one example.\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-geometry", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    summary = write_review(args.output, args.samples_per_geometry, args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
