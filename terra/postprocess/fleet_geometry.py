"""Continuous body and full-tool reservations for offline fleet plans.

Geometry is supplied per machine. Empty buckets, WAIT and travel never imply
that an attachment has been stowed. Sweeps enclose the declared interpolation;
they do not model steering, stability, hydraulic motion or physical duration.
"""

import math

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

EPS = 1e-8


def validate_geometry(value):
    geometry = dict(value)
    positive = ("body_length_m", "body_width_m", "work_reach_m", "work_half_angle_rad")
    nonnegative = ("clearance_m",)
    for key in positive + nonnegative:
        number = geometry.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ValueError(f"geometry.{key} must be a finite number")
        if number < 0 or (key in positive and number == 0):
            raise ValueError(
                f"geometry.{key} is outside its positive/nonnegative range"
            )
    if geometry["work_half_angle_rad"] > math.pi:
        raise ValueError("geometry.work_half_angle_rad must be at most pi")
    for key, default in (
        ("cell_padding_m", 0.0),
        ("max_drivable_height_m", 0.0),
        ("max_drivable_cut_m", 0.0),
        ("work_min_radius_m", 0.0),
    ):
        geometry.setdefault(key, default)
        if (
            isinstance(geometry[key], bool)
            or not math.isfinite(geometry[key])
            or geometry[key] < 0
        ):
            raise ValueError(f"geometry.{key} must be finite and nonnegative")
    if geometry["work_min_radius_m"] >= geometry["work_reach_m"]:
        raise ValueError("work_min_radius_m must be smaller than work_reach_m")
    if "tool_width_m" in geometry and (
        isinstance(geometry["tool_width_m"], bool)
        or not math.isfinite(geometry["tool_width_m"])
        or geometry["tool_width_m"] <= 0
    ):
        raise ValueError("geometry.tool_width_m must be finite and positive")
    geometry.setdefault("work_offset_xy_m", [0.0, 0.0])
    offset = np.asarray(geometry["work_offset_xy_m"], dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("geometry.work_offset_xy_m must contain two finite numbers")
    return geometry


def angle_delta(start, end):
    return (end - start + math.pi) % (2 * math.pi) - math.pi


def _buffer(geometry, distance):
    if geometry.is_empty or distance <= 0:
        return geometry
    # Circumscribe circular buffers; Shapely's polygonal arcs are inscribed.
    return geometry.buffer(distance / math.cos(math.pi / 64), quad_segs=16)


def body_polygon(state, geometry):
    if "body_polygon_xy_m" in state:
        polygon = Polygon(state["body_polygon_xy_m"])
    else:
        length, width = geometry["body_length_m"] / 2, geometry["body_width_m"] / 2
        polygon = Polygon(
            ((-length, -width), (length, -width), (length, width), (-length, width))
        )
        polygon = affinity.rotate(
            polygon, state["base_yaw_rad"], origin=(0, 0), use_radians=True
        )
        polygon = affinity.translate(polygon, *state["position_xy_m"])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("State has an invalid body polygon")
    return polygon


def work_origin(state, geometry):
    if "work_origin_xy_m" in state:
        return np.asarray(state["work_origin_xy_m"], dtype=float)
    offset = np.asarray(geometry["work_offset_xy_m"], dtype=float)
    c, s = math.cos(state["base_yaw_rad"]), math.sin(state["base_yaw_rad"])
    return (
        np.asarray(state["position_xy_m"], dtype=float)
        + np.array([[c, -s], [s, c]]) @ offset
    )


def work_polygon(state, geometry):
    centre = work_origin(state, geometry)
    half = geometry["work_half_angle_rad"]
    radius = geometry["work_reach_m"]
    heading = state["base_yaw_rad"] + state["cabin_yaw_rad"]
    intervals = max(2, int(math.ceil(math.degrees(2 * half) / 2)))
    angles = np.linspace(heading - half, heading + half, intervals + 1)
    radius /= math.cos(half / intervals)
    arc = centre + radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    polygon = (
        Polygon(arc)
        if half >= math.pi - EPS
        else Polygon(np.vstack((centre, arc, centre)))
    )
    return _buffer(polygon, geometry["cell_padding_m"] + EPS)


def _rotation_sweep(geometry, centre, radians):
    if abs(radians) <= EPS:
        return geometry
    intervals = max(1, int(math.ceil(abs(math.degrees(radians)) / 2)))
    copies = [
        affinity.rotate(geometry, angle, origin=tuple(centre), use_radians=True)
        for angle in np.linspace(0, radians, intervals + 1)
    ]
    cover = unary_union(
        [a.union(b).convex_hull for a, b in zip(copies[:-1], copies[1:])]
    )
    minx, miny, maxx, maxy = geometry.bounds
    radius = max(
        np.linalg.norm(np.asarray(point) - centre)
        for point in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))
    )
    return _buffer(cover, radius * (1 - math.cos(abs(radians) / intervals / 2)) + EPS)


def _sweep(first, last, start, end, radians):
    start, end = np.asarray(start), np.asarray(end)
    if abs(radians) <= EPS:
        return first.union(last).convex_hull
    if np.linalg.norm(end - start) <= EPS:
        # Captured native corner rounding can change the endpoint polygon.
        back = affinity.rotate(last, -radians, origin=tuple(start), use_radians=True)
        return _rotation_sweep(first.union(back).convex_hull, start, radians)
    radius = max(
        np.linalg.norm(np.asarray(first.exterior.coords) - start, axis=1).max(),
        np.linalg.norm(np.asarray(last.exterior.coords) - end, axis=1).max(),
    )
    return _buffer(LineString((start, end)), radius + EPS)


def reservation(before, after, geometry):
    first_body, last_body = body_polygon(before, geometry), body_polygon(
        after, geometry
    )
    body_delta = angle_delta(before["base_yaw_rad"], after["base_yaw_rad"])
    body = _sweep(
        first_body,
        last_body,
        before["position_xy_m"],
        after["position_xy_m"],
        body_delta,
    )
    first_work, last_work = work_polygon(before, geometry), work_polygon(
        after, geometry
    )
    tool_delta = body_delta + angle_delta(
        before["cabin_yaw_rad"], after["cabin_yaw_rad"]
    )
    start, end = work_origin(before, geometry), work_origin(after, geometry)
    start_offset = start - np.asarray(before["position_xy_m"])
    end_offset = end - np.asarray(after["position_xy_m"])
    rotating_offset = abs(body_delta) > EPS and (
        ("work_origin_xy_m" not in before and np.linalg.norm(start_offset) > EPS)
        or not np.allclose(start_offset, end_offset, rtol=0, atol=EPS)
    )
    if rotating_offset:
        # A body-fixed offset follows an arc when BASE turns. Connecting just
        # its endpoint origins misses that arc. Enclose the whole compound
        # rotation around BASE, also for simultaneous base translation.
        radius = max(np.linalg.norm(start_offset), np.linalg.norm(end_offset))
        radius += geometry["work_reach_m"] + geometry["cell_padding_m"] + EPS
        base_start, base_end = before["position_xy_m"], after["position_xy_m"]
        centre_path = (
            Point(base_start)
            if np.allclose(base_start, base_end, rtol=0, atol=EPS)
            else LineString((base_start, base_end))
        )
        work = _buffer(centre_path, radius)
    else:
        work = _sweep(first_work, last_work, start, end, tool_delta)
    return {"body": body, "work": work, "occupied": body.union(work)}


def conflict(left, right, clearance_m):
    gaps = {}
    for name, first, second in (
        ("body/body", left["body"], right["body"]),
        ("body/work", left["body"], right["work"]),
        ("work/body", left["work"], right["body"]),
        ("work/work", left["work"], right["work"]),
    ):
        gaps[name] = float(first.distance(second))
    reasons = [name for name, distance in gaps.items() if distance <= clearance_m + EPS]
    return {
        "conflict": bool(reasons),
        "reasons": reasons,
        "minimum_gap_m": min(gaps.values()),
        "clearance_m": clearance_m,
        "gaps_m": gaps,
    }


def polygon_record(geometry):
    """Plain JSON GeoJSON; no simplification is applied to checked geometry."""
    from shapely.geometry import mapping

    return mapping(geometry)
