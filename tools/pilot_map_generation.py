"""Small, versioned generators used by the Terra benchmark pilot."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tools import build_b0_feasibility_panels as b0

APRON_CAPACITY_BANDS = {
    "slcap03_04": (3.0, 4.0),
    "slcap07_10": (7.0, 10.0),
}
APRON_NOMINAL_CAPACITY_RATIOS = {
    "slcap03_04": 3.25,
    "slcap07_10": 8.5,
}
APRON_SEPARATION_CENTER_TILES = 2
APRON_SEPARATION_BAND_TILES = (1.25, 2.75)


def _line_coefficients(
    point_a_yx: np.ndarray,
    point_b_yx: np.ndarray,
) -> dict[str, float]:
    y1, x1 = (float(value) for value in point_a_yx)
    y2, x2 = (float(value) for value in point_b_yx)
    return {
        "A": y2 - y1,
        "B": x1 - x2,
        "C": x2 * y1 - x1 * y2,
    }


def sample_segmented_trench(
    v5,
    rng: np.random.Generator,
    segment_count: int,
    length_range: tuple[float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample one fixed-width no-junction trench for the S1 pilot."""
    length_min, length_max = length_range
    if segment_count not in (2, 3):
        raise ValueError(f"unsupported segment count: {segment_count}")
    if not 0.0 < length_min < length_max:
        raise ValueError(f"invalid segment length range: {length_range}")

    rejections = {
        "segmented_out_of_bounds": 0,
        "segmented_volume_outside_prefilter": 0,
    }
    for _ in range(300):
        heading = float(rng.choice(np.deg2rad(np.arange(0, 180, 15))))
        headings = [heading]
        headings.append(
            heading + float(rng.choice(np.deg2rad(np.asarray([-45, -30, 30, 45]))))
        )
        if segment_count == 3:
            headings.append(
                headings[-1]
                + float(rng.choice(np.deg2rad(np.asarray([-30, -15, 15, 30]))))
            )
        lengths = rng.uniform(length_min, length_max, size=segment_count)
        points = [np.zeros(2, dtype=np.float64)]
        for local_heading, length in zip(headings, lengths):
            points.append(
                points[-1]
                + length
                * np.asarray([math.sin(local_heading), math.cos(local_heading)])
            )
        points_array = np.asarray(points)
        box_center = (points_array.min(axis=0) + points_array.max(axis=0)) / 2.0
        points_array += rng.uniform(29.0, 35.0, size=2) - box_center
        if not v5.v3.points_inside(points_array, margin=10):
            rejections["segmented_out_of_bounds"] += 1
            continue
        dig = v5.base.rasterize_polyline(points_array, radius=1)
        if not 55 <= int(dig.sum()) <= 125:
            rejections["segmented_volume_outside_prefilter"] += 1
            continue
        return dig, {
            "trench_width_radius_tiles": 1,
            "segment_lengths_tiles": [float(value) for value in lengths],
            "turn_angles_deg": [
                float(math.degrees(next_heading - previous_heading))
                for previous_heading, next_heading in zip(headings[:-1], headings[1:])
            ],
            "axes_ABC": [
                _line_coefficients(start, end)
                for start, end in zip(
                    points_array[:-1],
                    points_array[1:],
                )
            ],
            "audit_generator_draw_count": 1 + sum(rejections.values()),
            "audit_generator_rejections": rejections,
        }
    raise RuntimeError(f"could not generate a {segment_count}-segment trench")


def build_osm_apron_capacity_pair(
    dig: np.ndarray,
    source_group_id: str,
) -> dict[str, Any]:
    """Build the constrained/moderate apron pair on one exact OSM dig mask."""
    dig = np.asarray(dig)
    if dig.shape != (b0.MAP_SIZE, b0.MAP_SIZE):
        raise ValueError(f"dig must be 64 x 64, got {dig.shape}")
    if dig.dtype != np.bool_:
        raise ValueError(f"dig must have boolean dtype, got {dig.dtype}")
    if not np.any(dig):
        raise ValueError("dig must contain at least one excavation cell")
    if not source_group_id:
        raise ValueError("source_group_id must be non-empty")

    dig = dig.copy()
    dig_identity_sha256 = b0.sha256_array(dig.astype(np.uint8))
    variants = {}
    for capacity_token, nominal_ratio in APRON_NOMINAL_CAPACITY_RATIOS.items():
        dump, apron_metadata = b0.build_apron_dump(
            dig,
            APRON_SEPARATION_CENTER_TILES,
            side_access="all",
            target_capacity_ratio=nominal_ratio,
        )
        achieved_ratio = float(dump.sum() / dig.sum())
        capacity_lower, capacity_upper = APRON_CAPACITY_BANDS[capacity_token]
        if not capacity_lower <= achieved_ratio <= capacity_upper:
            raise RuntimeError(
                f"{capacity_token} achieved capacity {achieved_ratio:.6f} "
                f"outside [{capacity_lower}, {capacity_upper}]"
            )
        separation_p50 = float(apron_metadata["p50_tiles"])
        separation_lower, separation_upper = APRON_SEPARATION_BAND_TILES
        if not separation_lower <= separation_p50 <= separation_upper:
            raise RuntimeError(
                f"{capacity_token} achieved separation {separation_p50:.6f} "
                f"outside [{separation_lower}, {separation_upper}]"
            )

        target = np.zeros(dig.shape, dtype=np.int8)
        target[dig] = -1
        target[dump] = 1
        variants[capacity_token] = {
            "target": target,
            "dump": dump,
            "metadata": {
                "source_family": "osm",
                "source_group_id": source_group_id,
                "dig_identity_sha256": dig_identity_sha256,
                "capacity_token": capacity_token,
                "nominal_single_layer_area_ratio": nominal_ratio,
                "achieved_single_layer_area_ratio": achieved_ratio,
                "capacity_band_inclusive": [
                    capacity_lower,
                    capacity_upper,
                ],
                "separation_token": "sep02",
                "separation_p50_tiles": separation_p50,
                "separation_p95_tiles": float(apron_metadata["p95_tiles"]),
                "separation_max_tiles": float(apron_metadata["max_tiles"]),
                "separation_band_p50_tiles_inclusive": [
                    separation_lower,
                    separation_upper,
                ],
            },
        }

    return {
        "source_family": "osm",
        "source_group_id": source_group_id,
        "dig": dig,
        "dig_identity_sha256": dig_identity_sha256,
        "variants": variants,
    }
