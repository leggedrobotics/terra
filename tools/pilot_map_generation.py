"""Small, versioned generators used by the Terra benchmark pilot."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


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
