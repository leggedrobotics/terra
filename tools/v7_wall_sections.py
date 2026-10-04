#!/usr/bin/env python3
"""Exact wall sections of the v7 courtyard, bearing-wall and courtyard-pad foundations, from their generator.

These foundations are an outer rectangle minus an inner one (walls 3-4.5 tiles, 1.7-2.6 m, thick); bearing walls add
two cross walls the generator draws as 1.3 m trenches along given centrelines. This replays the generator
(generate_v7_geometry_review.generate_scenarios) with the split's seed and sample count, records each foundation's
rectangles and cross-wall segments, and turns them into finite trench sections (Terra cell-index coordinates, one per
wall, each with its own half width): the 4 perimeter walls run corner to corner, the cross walls to the perimeter
centrelines. The fresh-trench gate then digs these walls like trenches. Maps are matched to scenarios by their dig
raster (sha256 of the generator's boolean mask, which equals image == -1).

Usage: v7_wall_sections.py BANK_DIR OUT_METADATA_DIR --split SEED:COUNT [--split SEED:COUNT ...]
       [--generator .../generate_v7_geometry_review.py]
Writes OUT_METADATA_DIR/trench_<slot>.json for every bank slot (sections added where a wall foundation matched, the
original metadata otherwise) and OUT_METADATA_DIR/../narrow_wall_report.json.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

GENERATOR = Path("/home/lorenzo/moleworks/.worktrees/terra_v8_combined_20260803/tools/map_generation/generate_v7_geometry_review.py")
WALL_GEOMETRIES = ("courtyard", "bearing_walls", "courtyard_pads")


def load_generator(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("v7_generator_replay", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay(v7, seed, count):
    """{sha256(dig mask): (scenario_id, geometry, rectangles, cross segments)} for the wall foundations of one split."""
    current, last, records = {}, {}, {}
    original_rectangle, original_trench = v7._rectangle, v7._rasterize_trench
    original_mask, original_make = v7._foundation_mask, v7.make_foundation_scenario

    def rectangle(centre, length, width, angle):
        if not current.get("in_trench"):
            current["rectangles"].append((np.asarray(centre, dtype=float), float(length), float(width), float(angle)))
        return original_rectangle(centre, length, width, angle)

    def trench(segments):
        current["segments"] = [np.asarray(segment, dtype=float).copy() for segment in segments]
        current["in_trench"] = True
        try:
            return original_trench(segments)
        finally:
            current["in_trench"] = False

    def mask(geometry, rng):
        current.clear()
        current.update(rectangles=[], segments=None, in_trench=False)
        result = original_mask(geometry, rng)
        last.clear()
        last.update(current, mask=result.copy())
        return result

    def make(geometry, index, rng):
        scenario = original_make(geometry, index, rng)
        assert np.array_equal(last["mask"], scenario.dig)
        if geometry in WALL_GEOMETRIES:
            key = hashlib.sha256(np.ascontiguousarray(scenario.dig).tobytes()).hexdigest()
            records[key] = (scenario.scenario_id, geometry, list(last["rectangles"]), last["segments"])
        return scenario

    v7._rectangle, v7._rasterize_trench, v7._foundation_mask, v7.make_foundation_scenario = rectangle, trench, mask, make
    try:
        v7.generate_scenarios(count, seed)
    finally:
        v7._rectangle, v7._rasterize_trench = original_rectangle, original_trench
        v7._foundation_mask, v7.make_foundation_scenario = original_mask, original_make
    return records


def sections_of(v7, geometry, rectangles, segments):
    """[(start (row, col), end (row, col), width tiles)] of the walls."""
    (centre, length, width, angle), (_, inner_length, inner_width, _) = rectangles[0], rectangles[1]
    direction = np.asarray(v7._direction(math.degrees(angle)), dtype=float)
    normal = np.asarray([-direction[1], direction[0]])
    long_offset = (width + inner_width) / 4.0
    short_offset = (length + inner_length) / 4.0
    out = []
    for sign in (-1.0, 1.0):
        mid = centre + sign * long_offset * normal
        out.append((mid - direction * length / 2.0, mid + direction * length / 2.0, (width - inner_width) / 2.0))
    for sign in (-1.0, 1.0):
        mid = centre + sign * short_offset * direction
        out.append((mid - normal * width / 2.0, mid + normal * width / 2.0, (length - inner_length) / 2.0))
    if geometry == "bearing_walls":
        for segment in segments:
            start, end = segment[0], segment[-1]
            unit = (end - start) / np.linalg.norm(end - start)
            reach = short_offset if abs(float(unit @ direction)) > 0.5 else long_offset
            out.append((centre - unit * reach, centre + unit * reach, float(v7.TARGET_TRENCH_WIDTH_TILES)))
    return out


def metadata_with(metadata, walls, tile_m):
    axes, segments = [], []
    for start, end, _ in walls:
        (r0, c0), (r1, c1) = start, end
        a, b = r1 - r0, -(c1 - c0)
        axes.append({"A": float(a), "B": float(b), "C": float(-(a * c0 + b * r0))})
        segments.append([[float(r0), float(c0)], [float(r1), float(c1)]])
    out = dict(metadata)
    out.update({
        "axes_ABC": axes,
        "trench_segments_yx": segments,
        "trench_axes_count": len(axes),
        "trench_half_width_tiles": max(w for _, _, w in walls) / 2.0,
        "trench_section_half_widths_tiles": [w / 2.0 for _, _, w in walls],
        "trench_topology": "narrow_walls",
        "narrow_wall_sections": {
            "tool": "terra tools/v7_wall_sections.py (generator replay)",
            "section_widths_m": [round(w * tile_m, 3) for _, _, w in walls],
            "section_lengths_m": [round(float(np.hypot(*(e - s))) * tile_m, 3) for s, e, _ in walls],
        },
    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bank")
    ap.add_argument("out_metadata")
    ap.add_argument("--split", action="append", required=True, help="SEED:COUNT of a generate_scenarios call")
    ap.add_argument("--generator", type=Path, default=GENERATOR)
    args = ap.parse_args()
    bank, out = Path(args.bank), Path(args.out_metadata)
    out.mkdir(parents=True, exist_ok=False)
    tile_m = float(json.loads((bank / "dataset.json").read_text())["tile_size_m"])
    v7 = load_generator(args.generator)
    records = {}
    for spec in args.split:
        seed, count = (int(x) for x in spec.split(":"))
        found = replay(v7, seed, count)
        records.update(found)
        print(f"split seed {seed} count {count}: {len(found)} wall foundations", flush=True)
    rows = [json.loads(line) for line in (bank / "manifest.jsonl").read_text().splitlines()]
    report = {"bank": str(bank), "splits": args.split, "generator": str(args.generator), "sectioned": [], "unmatched_wall_maps": []}
    for row in rows:
        slot = row["slot_index"]
        metadata = json.loads((bank / "metadata" / f"trench_{slot}.json").read_text())
        image = np.load(bank / "images" / f"img_{slot}.npy")
        key = hashlib.sha256(np.ascontiguousarray(image == -1).tobytes()).hexdigest()
        if key in records:
            scenario_id, geometry, rectangles, segments = records[key]
            walls = sections_of(v7, geometry, rectangles, segments)
            metadata = metadata_with(metadata, walls, tile_m)
            report["sectioned"].append({"slot": slot, "scenario": scenario_id, "geometry": geometry, "walls": len(walls)})
        elif metadata.get("geometry") in WALL_GEOMETRIES:
            report["unmatched_wall_maps"].append({"slot": slot, "geometry": metadata.get("geometry")})
        (out / f"trench_{slot}.json").write_text(json.dumps(metadata))
    (out.parent / "narrow_wall_report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(report['sectioned'])} maps sectioned, {len(report['unmatched_wall_maps'])} wall maps unmatched, {len(rows)} slots")


if __name__ == "__main__":
    main()
