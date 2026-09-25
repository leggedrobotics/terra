#!/usr/bin/env python3
"""Per-map dig coverage when excavator dumps are limited to a shorter reach.

Excavators cannot drive while loaded, so soil dug from a base pose must be
dumped from that pose. A dig cell is serviceable under a dump reach R when
some base position where the 7 x 11 footprint fits (any of 12 headings) has
the cell within the dig annulus [r_min, r_dig] and an accepted, dumpable cell
within [r_min, R] ("direct"), or relayed through neutral dumpable cells that
are themselves serviceable ("coverage", the admission criterion). The cabin's
12 sectors of +-30 deg cover every direction, so heading does not restrict
either set. Connectivity between base positions
and terrain changes during the episode are ignored; the check compares
reaches on identical geometry.

usage: dump_reach_admission.py DATASET_DIR OUTPUT_JSON [--dump-reach 5.5]
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

TILE_M = 0.5714286
R_MIN_M = 3.6428571
R_DIG_M = 6.5
AGENT_WIDTH, AGENT_HEIGHT = 7, 11
HEADINGS = 12


def annulus(r_max_m):
    half = int(np.ceil(r_max_m / TILE_M)) + 1
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
    radius = np.hypot(dx, dy) * TILE_M
    return (radius >= R_MIN_M - 1e-5) & (radius <= r_max_m + 1e-5)


def footprints():
    half = int(np.ceil(np.hypot(AGENT_WIDTH, AGENT_HEIGHT) / 2)) + 1
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
    kernels = []
    for k in range(HEADINGS):
        angle = 2 * np.pi * k / HEADINGS
        along = dx * np.cos(angle) + dy * np.sin(angle)
        across = -dx * np.sin(angle) + dy * np.cos(angle)
        kernels.append((np.abs(along) <= AGENT_HEIGHT / 2) & (np.abs(across) <= AGENT_WIDTH / 2))
    return kernels


FOOTPRINTS = footprints()


def base_positions(occupancy):
    pad = max(k.shape[0] for k in FOOTPRINTS)
    blocked = np.pad(occupancy, pad, constant_values=True)
    fits = np.zeros(blocked.shape, dtype=bool)
    for kernel in FOOTPRINTS:
        fits |= ~binary_dilation(blocked, structure=kernel[::-1, ::-1])
    return fits[pad:-pad, pad:-pad]


def coverage(target, occupancy, dumpability, action, dump_reaches):
    sources = (target < 0) | ((action > 0) & (target <= 0))
    accepted = (target > 0) & dumpability & ~occupancy
    positions = base_positions(occupancy)
    dig_ring = annulus(R_DIG_M)
    # Soil may also be staged on neutral dumpable cells and lifted again.
    staging = (target == 0) & dumpability & ~occupancy
    result = {}
    for reach in dump_reaches:
        dump_ring = annulus(reach)
        can_dump = positions & binary_dilation(accepted, structure=dump_ring)
        direct = sources & binary_dilation(can_dump, structure=dig_ring)
        good = accepted.copy()
        while True:
            can_dump = positions & binary_dilation(good, structure=dump_ring)
            grown = good | ((sources | staging) & binary_dilation(can_dump, structure=dig_ring))
            if (grown == good).all():
                break
            good = grown
        n = max(sources.sum(), 1)
        result[reach] = (float(direct.sum() / n), float((sources & good).sum() / n))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dump-reach", type=float, default=5.5)
    args = parser.parse_args()
    reaches = (R_DIG_M, args.dump_reach)
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines() if line]
    seen = {}
    per_map = []
    for row in rows:
        slot = row["slot_index"]
        load = lambda kind: np.load(args.dataset / kind / f"img_{slot}.npy")
        target = load("images")
        key = (row["map_id"], target.tobytes())
        if key not in seen:
            action_path = args.dataset / "actions" / f"img_{slot}.npy"
            action = np.load(action_path) if action_path.exists() else np.zeros_like(target)
            seen[key] = coverage(target, load("occupancy").astype(bool),
                                 load("dumpability").astype(bool), action, reaches)
        cov = seen[key]
        per_map.append(dict(slot=slot, map_id=row["map_id"], condition=row["primary_cell"],
                            direct_dig_reach=cov[R_DIG_M][0], direct_dump_reach=cov[args.dump_reach][0],
                            coverage_dig_reach=cov[R_DIG_M][1], coverage_dump_reach=cov[args.dump_reach][1]))

    by_condition = collections.defaultdict(lambda: collections.Counter())
    for row in per_map:
        c = by_condition[row["condition"]]
        c["slots"] += 1
        c["full_at_dig_reach"] += row["coverage_dig_reach"] == 1.0
        c["full_at_dump_reach"] += row["coverage_dump_reach"] == 1.0
        c["lost_full_coverage"] += row["coverage_dig_reach"] == 1.0 and row["coverage_dump_reach"] < 1.0
        c["direct_full_at_dig_reach"] += row["direct_dig_reach"] == 1.0
        c["direct_full_at_dump_reach"] += row["direct_dump_reach"] == 1.0
    summary = dict(
        dataset=str(args.dataset), dump_reach_m=args.dump_reach, slots=len(per_map),
        distinct_maps=len(seen),
        lost_full_coverage=sum(c["lost_full_coverage"] for c in by_condition.values()),
        by_condition={k: dict(v) for k, v in sorted(by_condition.items())},
    )
    args.output.write_text(json.dumps(dict(summary=summary, per_map=per_map), indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
