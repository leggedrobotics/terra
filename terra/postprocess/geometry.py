"""Small geometry kernels for the independent offline postprocessing model.

These formulas preserve the ROS converter's October 8, 2026 offline behavior.
They are not the live workspace planner or a source of robot limits. The 1e-6
roundoff/area tolerances and 0.45 m footprint model are explicit legacy model
assumptions; caller profiles still own machine geometry and clearance.
"""

import math

import numpy as np
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

GEOMETRY_ROUNDOFF_M = 1e-6
CONTINUOUS_RESIDUAL_TOLERANCE_M2 = 1e-6
DUMP_FOOTPRINT_RADIUS_M = 0.45


def source_mask_geometry(mask, alignment):
    """Physical rectangles for actual raster hazards or source assignment IDs."""
    x, y = np.where(mask)
    if not len(x):
        return Polygon()
    geometry = shapely.union_all(shapely.box(x, y, x + 1, y + 1))
    tile = float(alignment["meters_per_tile"])
    yaw = float(alignment["yaw_map_from_plan_rad"])
    cy, sy = math.cos(yaw), math.sin(yaw)
    return affine_transform(
        geometry,
        [tile * cy, -tile * sy, tile * sy, tile * cy, *alignment["origin_map_xy_m"]],
    )


def pull_band_polygon(origin, unit, near, far, half_width):
    origin, unit = np.asarray(origin, dtype=float), np.asarray(unit, dtype=float)
    perpendicular = np.array([-unit[1], unit[0]])
    return Polygon(
        [
            origin + radius * unit + width * perpendicular
            for radius, width in (
                (near, -half_width),
                (far, -half_width),
                (far, half_width),
                (near, half_width),
            )
        ]
    )


def continuous_radial_runs(permission, origin, unit, near, far, half_width):
    """Radii where the entire perpendicular blade segment is permitted.

    Project connected forbidden pieces of the full sweep onto its radial axis.
    Their interval complement is the blade-segment erosion of permission. A
    circular negative buffer would incorrectly shorten finite trench endcaps.
    """
    sweep = pull_band_polygon(origin, unit, near, far, half_width)
    outside = sweep.difference(permission.buffer(GEOMETRY_ROUNDOFF_M, join_style=2))
    if outside.is_empty:
        return [(near, far)]
    pieces = [outside] if isinstance(outside, Polygon) else list(outside.geoms)
    blocked = []
    for piece in pieces:
        if not isinstance(piece, Polygon) or piece.is_empty or piece.area == 0.0:
            continue
        radii = (np.asarray(piece.exterior.coords) - origin) @ unit
        blocked.append((max(near, float(radii.min())), min(far, float(radii.max()))))
    runs, cursor = [], near
    for low, high in sorted(blocked):
        if low > cursor + 1e-8:
            runs.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < far - 1e-8:
        runs.append((cursor, far))
    return runs


def dump_footprint_offsets(resolution_m: float) -> list[tuple[int, int]]:
    """Cell (row, column) offsets of one pile footprint on a lattice of this resolution.

    The footprint holds the cells whose centres lie within DUMP_FOOTPRINT_RADIUS_M,
    and at least one cell, of the dump centre. Distributed selection requires all
    of them in the dump mask unless its full-footprint fallback is enabled.
    """

    footprint_radius_m = max(DUMP_FOOTPRINT_RADIUS_M, resolution_m)
    cells = int(math.ceil(footprint_radius_m / resolution_m))
    return [
        (dr, dc)
        for dr in range(-cells, cells + 1)
        for dc in range(-cells, cells + 1)
        if math.hypot(float(dr) * resolution_m, float(dc) * resolution_m)
        <= footprint_radius_m + 1e-6
    ]
