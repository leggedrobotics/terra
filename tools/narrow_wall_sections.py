#!/usr/bin/env python3
"""Trench sections for the narrow parts of a foundation design (bearing walls, strip footings).

A strip of the design no wider than --max-width-m (default two 1.3 m buckets) becomes a finite trench section along
its centreline. Terra's fresh-trench gate (enforce_trench_dig_alignment) then admits a fresh cell of that strip only
from a chassis parallel to it and standing on its line, as on trench maps; wider parts keep ordinary foundation
digging, because the gate leaves cells outside every section alone.

Strips come from the continuous design polygon (tile units; cell (row, col) spans [row, row+1) x [col, col+1), the
convention of the v7 generator's map_geometry.json): every pair of facing boundary edges (inward normals opposite)
whose distance is at most the maximum width and whose projections overlap gives a centreline. Collinear centrelines
are merged, and each is extended along itself while it stays inside the design, so sections run through corners and
junctions. Output: the trench metadata fields Terra reads (axes_ABC, trench_segments_yx, trench_half_width_tiles,
trench_axes_count), in Terra's cell-index coordinates (A*col + B*row + C = 0).

Usage: narrow_wall_sections.py METADATA_JSON GEOMETRY_JSON OUT_JSON [--tile-m 0.5714] [--max-width-m 2.6]
"""
import argparse
import json
import math

import numpy as np
from shapely.geometry import LineString, Point, shape

MAX_SECTIONS = 8  # terra.maps_buffer.MAX_TRENCH_SECTIONS: one uint8 membership bit per section


def boundary_edges(poly):
    edges = []
    for ring in [poly.exterior, *poly.interiors]:
        coords = np.asarray(ring.coords, dtype=float)
        for p, q in zip(coords[:-1], coords[1:]):
            d = q - p
            length = float(np.hypot(*d))
            if length < 1e-6:
                continue
            t = d / length
            n = np.array([-t[1], t[0]])
            if not poly.contains(Point(*(0.5 * (p + q) + 0.05 * n))):
                n = -n
            edges.append((p, q, t, n, length))
    return edges


def facing_strips(poly, max_width, min_length, angle_tol):
    edges = boundary_edges(poly)
    strips = []
    for i, (p1, q1, t1, n1, l1) in enumerate(edges):
        for p2, q2, t2, n2, l2 in edges[i + 1:]:
            if float(n1 @ n2) > -math.cos(angle_tol):
                continue
            width = float((0.5 * (p2 + q2) - p1) @ n1)
            if width <= 1e-6 or width > max_width:
                continue
            a, b = sorted([float((p2 - p1) @ t1), float((q2 - p1) @ t1)])
            lo, hi = max(0.0, a), min(l1, b)
            if hi - lo < min_length:
                continue
            m0 = p1 + lo * t1 + 0.5 * width * n1
            m1 = p1 + hi * t1 + 0.5 * width * n1
            samples = [m0 + s * (m1 - m0) for s in np.linspace(0.05, 0.95, 7)]
            if not all(poly.contains(Point(*x)) for x in samples):
                continue
            strips.append({"m0": m0, "m1": m1, "t": t1, "width": width})
    return strips


def merge_collinear(strips, angle_tol, offset_tol):
    merged = []
    for s in sorted(strips, key=lambda s: -np.hypot(*(s["m1"] - s["m0"]))):
        for m in merged:
            if abs(float(np.cross(m["t"], s["t"]))) > math.sin(angle_tol):
                continue
            normal = np.array([-m["t"][1], m["t"][0]])
            if max(abs(float((s["m0"] - m["m0"]) @ normal)), abs(float((s["m1"] - m["m0"]) @ normal))) > offset_tol:
                continue
            proj = [float((x - m["m0"]) @ m["t"]) for x in (m["m0"], m["m1"], s["m0"], s["m1"])]
            lo, hi = min(proj), max(proj)
            m["m0"], m["m1"] = m["m0"] + lo * m["t"], m["m0"] + hi * m["t"]
            m["width"] = max(m["width"], s["width"])
            break
        else:
            t = s["t"] if float((s["m1"] - s["m0"]) @ s["t"]) >= 0 else -s["t"]
            merged.append({"m0": s["m0"].copy(), "m1": s["m1"].copy(), "t": t, "width": s["width"]})
    return merged


def extend_inside(poly, section, step=0.05, limit=12.0):
    for end, sign in (("m0", -1.0), ("m1", 1.0)):
        point = section[end]
        travelled = 0.0
        while travelled < limit and poly.contains(Point(*(point + sign * step * section["t"]))):
            point = point + sign * step * section["t"]
            travelled += step
        section[end] = point
    return section


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata")
    ap.add_argument("geometry")
    ap.add_argument("out")
    ap.add_argument("--tile-m", type=float, default=0.5714285714)
    ap.add_argument("--max-width-m", type=float, default=2.6)
    ap.add_argument("--min-length-m", type=float, default=1.0)
    args = ap.parse_args()

    metadata = json.load(open(args.metadata))
    geometry = json.load(open(args.geometry))
    poly = shape(geometry["continuous_design"]["geometry"])
    max_width = args.max_width_m / args.tile_m
    strips = facing_strips(poly, max_width, args.min_length_m / args.tile_m, math.radians(10.0))
    sections = [extend_inside(poly, s) for s in merge_collinear(strips, math.radians(5.0), 0.6)]
    sections = merge_collinear(sections, math.radians(5.0), 0.6)
    sections.sort(key=lambda s: -np.hypot(*(s["m1"] - s["m0"])))
    if len(sections) > MAX_SECTIONS:
        raise SystemExit(f"{len(sections)} narrow sections; Terra holds at most {MAX_SECTIONS}")
    if not sections:
        raise SystemExit("no strip of the design is narrow enough")

    axes, segments = [], []
    for s in sections:
        # continuous (x, y) = (row, col) + 0.5 at the cell centre -> Terra cell-index coordinates
        (r0, c0), (r1, c1) = s["m0"] - 0.5, s["m1"] - 0.5
        a, b = r1 - r0, -(c1 - c0)
        axes.append({"A": float(a), "B": float(b), "C": float(-(a * c0 + b * r0))})
        segments.append([[float(r0), float(c0)], [float(r1), float(c1)]])
    half_width = max(s["width"] for s in sections) / 2.0
    out = dict(metadata)
    out.update({
        "axes_ABC": axes,
        "trench_segments_yx": segments,
        "trench_axes_count": len(axes),
        "trench_half_width_tiles": half_width,
        "trench_section_half_widths_tiles": [s["width"] / 2.0 for s in sections],
        "trench_topology": "narrow_walls",
        "narrow_wall_sections": {
            "tool": "terra tools/narrow_wall_sections.py",
            "max_width_m": args.max_width_m,
            "section_widths_m": [round(s["width"] * args.tile_m, 3) for s in sections],
            "section_lengths_m": [round(float(np.hypot(*(s["m1"] - s["m0"]))) * args.tile_m, 3) for s in sections],
        },
    })
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out["narrow_wall_sections"]), "half_width_tiles", round(half_width, 3), "sections", len(axes))


if __name__ == "__main__":
    main()
