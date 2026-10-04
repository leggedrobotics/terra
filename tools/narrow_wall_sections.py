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

Training maps carry no design polygon: with --raster the walls are found on the map's dig cells (image == -1). The
cells are first resampled on a grid turned to the design's main direction (the angle, 0-89 deg, whose resampled outline
is shortest), so an oblique wall has straight edges instead of stairs; the sections are turned back afterwards.
Strips shorter than twice their width (pads) are not walls and get no section; a map with more than 8 walls gets none
(all or nothing per map).

Usage: narrow_wall_sections.py METADATA_JSON (GEOMETRY_JSON | --raster IMAGE_NPY) OUT_JSON [--max-width-m 2.6]
"""
import argparse
import json
import math

import numpy as np
from shapely import unary_union
from shapely.geometry import Point, box, shape

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


def raster_wall_sections(image, max_width, min_length):
    """Wall sections of a Terra image (tile units, continuous (row, col) coordinates).

    Walls are found where the dig region is narrow: cells removed by an opening with a disk of diameter max_width.
    Each such cell gets the local wall direction from the structure tensor of the smoothed dig mask; cells with a clear
    direction vote for a line (direction, perpendicular offset). Every well-supported line becomes a section along the
    longest run of narrow cells on it, extended through corners and junctions while it stays on dig cells.
    """
    from scipy import ndimage

    dig = np.asarray(image) == -1
    if not dig.any():
        return []
    radius = max_width / 2.0
    k = int(math.ceil(radius))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = (yy ** 2 + xx ** 2) <= radius ** 2
    narrow = dig & ~ndimage.binary_opening(dig, structure=disk)
    if narrow.sum() < 2 * min_length:
        return []
    smooth = ndimage.gaussian_filter(dig.astype(float), 1.0)
    gr, gc = np.gradient(smooth)
    jrr, jcc, jrc = (ndimage.gaussian_filter(v, 2.0) for v in (gr * gr, gc * gc, gr * gc))
    # Normal of the walls: dominant gradient direction, angle in the (col, row) plane.
    normal_angle = 0.5 * np.arctan2(2.0 * jrc, jcc - jrr)
    coherence = np.hypot(jcc - jrr, 2.0 * jrc) / (jcc + jrr + 1e-12)
    rows, cols = np.nonzero(narrow & (coherence > 0.5))
    if len(rows) < 2 * min_length:
        return []
    nc, nr = np.cos(normal_angle[rows, cols]), np.sin(normal_angle[rows, cols])
    # Fold the normal to angles in [0, 180) and vote in (angle, offset) bins of 6 deg and 1 tile.
    angle = np.mod(np.degrees(np.arctan2(nr, nc)), 180.0)
    centres_r, centres_c = rows + 0.5, cols + 0.5
    used = np.zeros(len(rows), dtype=bool)
    sections = []
    while True:
        free = ~used
        if free.sum() < 2 * min_length:
            break
        hist, cells_of = {}, {}
        for i in np.flatnonzero(free):
            abin_i = int(round(angle[i] / 6.0)) % 30
            a = math.radians(abin_i * 6.0)
            offset = math.cos(a) * centres_c[i] + math.sin(a) * centres_r[i]
            key = (abin_i, int(math.floor(offset)))
            hist[key] = hist.get(key, 0) + 1
            cells_of.setdefault(key, []).append(i)
        (abin, obin), votes = max(hist.items(), key=lambda kv: kv[1])
        if votes < min_length:
            break
        used[cells_of[(abin, obin)]] = True  # progress even when the band below comes out empty
        # The wall's band: cells of this direction whose offsets form one contiguous band (0.5-tile steps) around the
        # peak, at most max_width wide. Its middle is the centreline and its extent the width.
        a = math.radians(abin * 6.0)
        offsets = math.cos(a) * centres_c + math.sin(a) * centres_r
        diff = np.abs((angle - abin * 6.0 + 90.0) % 180.0 - 90.0)
        peak = obin + 0.5
        near = free & (diff <= 9.0) & (np.abs(offsets - peak) <= max_width / 2.0 + 0.5)
        steps = np.floor((offsets[near] - peak) / 0.5).astype(int)
        present = set(steps.tolist())
        lo_step = hi_step = 0
        while lo_step - 1 in present or lo_step - 2 in present:
            lo_step -= 1
        while hi_step + 1 in present or hi_step + 2 in present:
            hi_step += 1
        band_lo, band_hi = peak + lo_step * 0.5, peak + (hi_step + 1) * 0.5
        members = near & (offsets >= band_lo - 1e-9) & (offsets <= band_hi + 1e-9)
        used |= members
        if members.sum() < min_length:
            continue
        centre_offset = 0.5 * (offsets[members].min() + offsets[members].max())
        width = min(float(offsets[members].max() - offsets[members].min()) + 1.0, max_width)
        normal = np.array([math.sin(a), math.cos(a)])  # (row, col)
        tangent = np.array([normal[1], -normal[0]])
        # Along-line extent: narrow cells in the band, longest run with gaps of at most 2 tiles (junctions).
        nr_all, nc_all = np.nonzero(narrow)
        allpts = np.stack([nr_all + 0.5, nc_all + 0.5], axis=1)
        across = allpts @ normal - centre_offset
        band = np.abs(across) <= width / 2.0 + 0.25
        along = np.sort(allpts[band] @ tangent)
        if len(along) == 0:
            continue
        breaks = np.flatnonzero(np.diff(along) > 2.0)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [len(along) - 1]])
        j = int(np.argmax(along[ends] - along[starts]))
        lo, hi = along[starts[j]], along[ends[j]]
        if hi - lo + 1.0 < min_length:
            continue
        base = centre_offset * normal
        sections.append({"m0": base + lo * tangent, "m1": base + hi * tangent, "t": tangent, "width": width})
    # Parallel sections whose bands overlap are one wall.
    merged = []
    for s in sorted(sections, key=lambda s: -s["width"]):
        for m in merged:
            if abs(float(np.cross(m["t"], s["t"]))) > math.sin(math.radians(9.0)):
                continue
            normal = np.array([-m["t"][1], m["t"][0]])
            gap = abs(float((0.5 * (s["m0"] + s["m1"]) - 0.5 * (m["m0"] + m["m1"])) @ normal))
            if gap > 0.5 * max(m["width"], s["width"]):
                continue
            proj = [float((x - m["m0"]) @ m["t"]) for x in (m["m0"], m["m1"], s["m0"], s["m1"])]
            lo, hi = min(proj), max(proj)
            m["m0"], m["m1"] = m["m0"] + lo * m["t"], m["m0"] + hi * m["t"]
            break
        else:
            merged.append(dict(s))
    sections = merged
    merged = merge_collinear(sections, math.radians(6.0), 1.0)
    out = []
    for s in merged:
        # Extend along the line while the centreline stays on dig cells (through corners and junctions).
        for end, sign in (("m0", -1.0), ("m1", 1.0)):
            point = s[end]
            for _ in range(80):
                nxt = point + sign * 0.25 * s["t"]
                r, c = int(math.floor(nxt[0])), int(math.floor(nxt[1]))
                if not (0 <= r < dig.shape[0] and 0 <= c < dig.shape[1]) or not dig[r, c]:
                    break
                point = nxt
            s[end] = point
        if np.hypot(*(s["m1"] - s["m0"])) >= 2.0 * s["width"]:
            out.append(s)
    out.sort(key=lambda s: -np.hypot(*(s["m1"] - s["m0"])))
    return out


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
    ap.add_argument("geometry", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--raster", help="64x64 Terra target image (.npy); its dig cells replace the design polygon")
    ap.add_argument("--tile-m", type=float, default=0.5714285714)
    ap.add_argument("--max-width-m", type=float, default=2.6)
    ap.add_argument("--min-length-m", type=float, default=1.0)
    args = ap.parse_args()

    metadata = json.load(open(args.metadata))
    if args.raster:
        sections = raster_wall_sections(np.load(args.raster), args.max_width_m / args.tile_m, args.min_length_m / args.tile_m)
    else:
        poly = shape(json.load(open(args.geometry))["continuous_design"]["geometry"])
        sections = wall_sections(poly, args.max_width_m / args.tile_m, args.min_length_m / args.tile_m)
    if len(sections) > MAX_SECTIONS:
        raise SystemExit(f"{len(sections)} narrow sections; Terra holds at most {MAX_SECTIONS}")
    if not sections:
        raise SystemExit("no strip of the design is narrow enough")
    out = metadata_with_sections(metadata, sections, args.tile_m, args.max_width_m)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out["narrow_wall_sections"]), "sections", len(sections))


def metadata_with_sections(metadata, sections, tile_m, max_width_m):
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
            "max_width_m": max_width_m,
            "section_widths_m": [round(s["width"] * tile_m, 3) for s in sections],
            "section_lengths_m": [round(float(np.hypot(*(s["m1"] - s["m0"]))) * tile_m, 3) for s in sections],
        },
    })
    return out


if __name__ == "__main__":
    main()
