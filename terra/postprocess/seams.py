"""Seam lips of a converted plan, for the dashboard (Lorenzo, 4 Oct 2026: at least 0.3 m of overlap between bands).

A seam is where an excavation workspace's new completed ground meets ground completed earlier; its lip is how far the
later workspace's ground continues into that earlier ground. Completed ground is the union of a workspace's pulls
(report witnesses, as workspace_lanes._Ledger builds them) inside the design and, with the fresh-cutting rule, inside its
planned BASE completion ring. The lip adds the workspace's overlap pulls inside the hard completion band: those are dug
when the machine stops on its station. Lips are measured as workspace_lanes.seam_overlap_readout measures them (1 cm
steps up to 1 m along the seam normal; earlier ground narrower than the margin needs only its own width), but only at
real seams, not at edges of earlier ground that a blade-wide lip pull crosses elsewhere.
"""

import math

import numpy as np
import shapely
from shapely import wkt
from shapely.geometry import Point, Polygon

MARGIN_M = 0.3
STEPS = np.arange(0.01, 1.0 + 1e-9, 0.01)


def _area_only(geometry):
    if geometry.is_empty or geometry.area == 0:
        return Polygon()
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    return shapely.union_all(
        [_area_only(part) for part in geometry.geoms if part.area > 0]
    )


def _lines(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    return [line for item in getattr(geometry, "geoms", []) for line in _lines(item)]


def _bands(pose, witnesses, pivot, half_width):
    bands = []
    for witness in witnesses:
        angle = pose[2] + witness["theta_rad"]
        unit = np.array([math.cos(angle), math.sin(angle)])
        perpendicular = np.array([-unit[1], unit[0]])
        origin = (
            np.asarray(pose[:2], dtype=float)
            + pivot[0] * unit
            + pivot[1] * perpendicular
        )
        bands.append(
            Polygon(
                [
                    origin + radius * unit + width * perpendicular
                    for radius, width in (
                        (witness["radius_near_m"], -half_width),
                        (witness["radius_far_m"], -half_width),
                        (witness["radius_far_m"], half_width),
                        (witness["radius_near_m"], half_width),
                    )
                ]
            )
        )
    return shapely.union_all(bands) if bands else Polygon()


def _ring(centre, near, far):
    centre = Point(centre)
    return centre.buffer(far, quad_segs=256).difference(
        centre.buffer(near / math.cos(math.pi / 1024), quad_segs=256)
    )


def seam_lips(report):
    """Per accepted workspace, in execution order: None for a collection, else dict(pose, lip, seams).

    lip: the workspace's lip ground beyond its completed ground (a polygon, possibly empty).
    seams: samples of the real seams into this workspace, [x, y, lip_m, needed_m, weight_m], every 2 cm.
    A report without the continuous design and the pulls (synthetic or older conversions) has no seams: [].
    """
    if any(
        key not in report
        for key in (
            "design_wkt",
            "required_geometry_wkt",
            "control_offset_xy_m",
            "envelope",
        )
    ):
        return []
    design = wkt.loads(report["design_wkt"])
    required = wkt.loads(report["required_geometry_wkt"])
    pivot = np.asarray(report["control_offset_xy_m"], dtype=float)
    half_width = report["envelope"]["blade_width_m"] / 2
    rule = report.get("fresh_cutting_rule")
    pairs = [
        pair
        for pair in report["per_pair"]
        if pair["accepted"] and not pair.get("omitted")
    ]
    out, done, ground, index = [], [], [], []
    for k, pair in enumerate(pairs):
        if pair["workspace_type"] != "excavate":
            out.append(None)
            continue
        pose = pair["chosen_pose"]
        dug = _bands(pose, pair["witnesses"], pivot, half_width).intersection(design)
        lip = _bands(
            pose,
            [w for w in pair["witnesses"] if w.get("kind") == "overlap"],
            pivot,
            half_width,
        ).intersection(design)
        if rule:
            dug = dug.intersection(
                _ring(
                    pose[:2],
                    rule["planning_completion_radius_min_m"],
                    rule["planning_completion_radius_max_m"],
                )
            )
            lip = lip.intersection(
                _ring(
                    pose[:2],
                    rule["completion_radius_min_m"],
                    rule["completion_radius_max_m"],
                )
            )
        own = _area_only(dug)
        whole = _area_only(shapely.union_all([own, _area_only(lip)]))
        out.append(
            dict(pose=list(pose), lip=_area_only(whole.difference(own)), seams=[])
        )
        done.append(own)
        ground.append(whole)
        index.append(k)
    inner = required.buffer(-1e-6)
    earlier = Polygon()
    for k, own, whole in zip(index, done, ground):
        if not earlier.is_empty and not own.is_empty:
            fresh = own.intersection(required).difference(earlier)
            seam = earlier.boundary.intersection(fresh.buffer(1e-4)).intersection(inner)
            shapely.prepare(earlier)
            shapely.prepare(whole)
            for line in _lines(seam):
                if line.length < 0.02:
                    continue
                t = np.arange(0.01, line.length, 0.02)
                point = shapely.get_coordinates(shapely.line_interpolate_point(line, t))
                tangent = shapely.get_coordinates(
                    shapely.line_interpolate_point(
                        line, np.minimum(t + 0.005, line.length)
                    )
                ) - shapely.get_coordinates(
                    shapely.line_interpolate_point(line, np.maximum(t - 0.005, 0.0))
                )
                norm = np.hypot(tangent[:, 0], tangent[:, 1])
                keep = norm >= 1e-9
                normal = (
                    np.column_stack((-tangent[:, 1], tangent[:, 0]))
                    / np.where(keep, norm, 1.0)[:, None]
                )
                ahead = shapely.contains_xy(earlier, *(point + 0.005 * normal).T)
                behind = ~ahead & shapely.contains_xy(
                    earlier, *(point - 0.005 * normal).T
                )
                normal[behind] *= -1
                keep &= ahead | behind
                if not keep.any():
                    continue
                ray = (
                    point[keep, None, :] + STEPS[None, :, None] * normal[keep, None, :]
                )
                in_whole = shapely.contains_xy(whole, ray[..., 0], ray[..., 1])
                in_earlier = shapely.contains_xy(earlier, ray[..., 0], ray[..., 1])
                lip_m = np.where(
                    in_whole.all(axis=1), 1.0, STEPS[np.argmin(in_whole, axis=1)] - 0.01
                )
                earlier_m = np.where(
                    in_earlier.all(axis=1),
                    1.0,
                    STEPS[np.argmin(in_earlier, axis=1)] - 0.01,
                )
                weight = line.length / keep.sum()
                out[k]["seams"].extend(
                    [float(x), float(y), float(a), float(min(MARGIN_M, b)), weight]
                    for (x, y), a, b in zip(point[keep], lip_m, earlier_m)
                )
        earlier = shapely.union_all([earlier, own])
    return out


def share(samples):
    """(seam length m, share of it with a lip of at least the margin) of seam samples, or (0, None)."""
    length = sum(s[4] for s in samples)
    if not length:
        return 0.0, None
    return length, sum(s[4] for s in samples if s[2] + 1e-9 >= s[3]) / length
