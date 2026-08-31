#!/usr/bin/env python3
"""Is a gate-admissible fresh-dig cover reachable under Terra's runtime pose rules?

``tools/audit_trench_alignment_feasibility.py`` builds its pose graph from the
agent footprint rasterised in ``(row, col)``.  Terra does not: ``_is_valid_move``
tests the footprint polygon returned by ``compute_polygon_mask``, which is a
``(col, row)`` raster, so occupancy is evaluated at the transposed position and
the *mirror image* of the trench acts as the obstacle field.  This tool redoes
the cover question under Terra's actual rule.

For every trench map it computes, with ``blocked = padding | all target<0``
(the order-independent worst case, so a controller using only these stations can
never wall itself off):

  * the gate-admissible (pose, cabin) dig stations;
  * which of them are legal poses under Terra's transposed footprint test, and
    which under a correct ``(row, col)`` footprint;
  * the target cells covered by each of those two station sets;
  * whether the Terra-legal station set is connected and reachable from the
    frozen episode spawn pose.

The gap between the two cover numbers is the cost of the transposition.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import deque
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from terra.config import EnvConfig
from terra.map import compute_trench_axis_membership

from audit_trench_gate_overrestriction import (
    NH, SHAPE, cases_from_panel, dilate, env_config, free_poses, gather, geometry,
    records_from_metadata,
)

_W = {}


def _init(cfg, cones, fp_true, fp_masked, fwd, bwd, tol, so_min, so_max):
    _W.update(cfg=cfg, cones=cones, fp_true=fp_true, fp_masked=fp_masked,
              fwd=fwd, bwd=bwd, tol=tol, so_min=so_min, so_max=so_max)


def components(free, fwd, bwd):
    seen = np.zeros((NH,) + SHAPE, dtype=bool)
    comp = -np.ones((NH,) + SHAPE, dtype=np.int32)
    cid = 0
    for h0, r0, c0 in np.argwhere(free):
        if seen[h0, r0, c0]:
            continue
        q = deque([(int(h0), int(r0), int(c0))])
        seen[h0, r0, c0] = True
        comp[h0, r0, c0] = cid
        while q:
            h, r, c = q.popleft()
            nb = [(r, c, (h - 1) % NH), (r, c, (h + 1) % NH)]
            for dr, dc in (fwd[h], bwd[h]):
                nb.append((r + dr, c + dc, h))
            for rr, cc, hh in nb:
                if 0 <= rr < SHAPE[0] and 0 <= cc < SHAPE[1] and free[hh, rr, cc] \
                        and not seen[hh, rr, cc]:
                    seen[hh, rr, cc] = True
                    comp[hh, rr, cc] = cid
                    q.append((hh, rr, cc))
        cid += 1
    return comp, cid


def analyze(case):
    cones, tile = _W["cones"], _W["cfg"].tile_size
    tol, so_min, so_max = _W["tol"], _W["so_min"], _W["so_max"]
    fwd, bwd = _W["fwd"], _W["bwd"]

    target = np.load(case["images"], allow_pickle=False).astype(np.int32)
    padding = np.load(case["occupancy"], allow_pickle=False).astype(bool)
    metadata = json.loads(Path(case["metadata"]).read_text())
    records, naxes, half = records_from_metadata(metadata)
    membership = np.asarray(compute_trench_axis_membership(
        jnp.asarray(target.astype(np.int8)), jnp.asarray(records), jnp.int32(naxes)
    )).astype(np.uint8)

    dig = target < 0
    n = int(dig.sum())
    index = -np.ones(SHAPE, dtype=np.int32)
    cells = np.argwhere(dig)
    index[cells[:, 0], cells[:, 1]] = np.arange(n)

    axes3 = records[:naxes, :3].astype(np.float64)
    den = np.maximum(np.linalg.norm(axes3[:, :2], axis=1), 1e-6)
    rows, cols = np.meshgrid(np.arange(SHAPE[0], dtype=np.float64),
                             np.arange(SHAPE[1], dtype=np.float64), indexing="ij")
    standoff = np.stack([
        np.abs(axes3[a, 0] * cols + axes3[a, 1] * rows + axes3[a, 2]) / den[a] * tile
        for a in range(naxes)])
    band = (standoff >= so_min) & (standoff <= so_max)
    tg = np.stack([-axes3[:, 0], axes3[:, 1]], axis=1)
    tn = np.maximum(np.linalg.norm(tg, axis=1), 1e-6)
    yaw_ok = np.zeros((NH, naxes), dtype=bool)
    for bh in range(NH):
        th = 2 * np.pi * bh / NH
        f = np.array([-np.sin(th), np.cos(th)])
        yaw_ok[bh] = np.arccos(np.clip(np.abs(tg @ f) / tn, 0, 1)) <= tol
    bits = np.zeros((NH,) + SHAPE, dtype=np.uint8)
    for bh in range(NH):
        for a in range(naxes):
            if yaw_ok[bh, a]:
                bits[bh] |= np.where(band[a], np.uint8(1 << a), np.uint8(0))

    blocked = padding | dig                     # order-independent worst case
    free_terra = free_poses(blocked, _W["fp_masked"], transposed=True)
    free_true = free_poses(blocked, _W["fp_true"], transposed=False)

    cov_any = np.zeros(n, dtype=bool)
    cov_terra = np.zeros(n, dtype=bool)
    cov_true = np.zeros(n, dtype=bool)
    station_terra = np.zeros((NH,) + SHAPE, dtype=bool)
    near = dilate(dig, 12)
    for bh in range(NH):
        pm = near & ~padding
        poses = np.argwhere(pm)
        if poses.shape[0] == 0:
            continue
        pbits = bits[bh][pm]
        okt = free_terra[bh][pm]
        okr = free_true[bh][pm]
        for cb in range(NH):
            counts, (rrc, ccc, ok) = gather(poses, cones[bh][cb], [dig, padding])
            nfresh, npad = counts
            live = (nfresh > 0) & (npad == 0)
            if not live.any():
                continue
            mem = membership[rrc, ccc]
            freshm = dig[rrc, ccc] & ok
            for i in np.flatnonzero(live):
                b = pbits[i]
                if b == 0:
                    continue
                m = mem[i][freshm[i]]
                tr = m != 0
                if tr.any() and not ((m[tr] & b) != 0).all():
                    continue
                ids = index[rrc[i][freshm[i]], ccc[i][freshm[i]]]
                ids = ids[ids >= 0]
                cov_any[ids] = True
                if okt[i]:
                    cov_terra[ids] = True
                    station_terra[bh, poses[i, 0], poses[i, 1]] = True
                if okr[i]:
                    cov_true[ids] = True

    comp, ncomp = components(free_terra, fwd, bwd)
    st_nodes = np.argwhere(station_terra)
    st_comps = {int(comp[h, r, c]) for h, r, c in st_nodes}
    sizes = {}
    for h, r, c in st_nodes:
        k = int(comp[h, r, c])
        sizes[k] = sizes.get(k, 0) + 1
    return {
        "label": case["label"], "condition": case["condition"],
        "dataset": case["dataset"], "map_id": case["map_id"],
        "axes": naxes, "target_cells": n,
        "cells_admissible_any_pose": int(cov_any.sum()),
        "cells_admissible_terra_legal_persistent_station": int(cov_terra.sum()),
        "cells_admissible_true_footprint_persistent_station": int(cov_true.sum()),
        "cells_lost_to_footprint_transposition": int(np.sum(cov_true & ~cov_terra)),
        "cells_gained_by_transposition": int(np.sum(cov_terra & ~cov_true)),
        "terra_legal_stations": int(station_terra.sum()),
        "station_components": len(st_comps),
        "largest_station_component_share":
            float(max(sizes.values()) / max(len(st_nodes), 1)) if st_nodes.size else 0.0,
        "complete_terra": bool(cov_terra.sum() == n),
        "complete_true": bool(cov_true.sum() == n),
        "complete_any": bool(cov_any.sum() == n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-root", type=Path, required=True)
    ap.add_argument("--dataset", default="evaluation/gate_main/development")
    ap.add_argument("--exclude-prefix", nargs="*", default=["trn-net4-"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = env_config()
    cones, fp_true, fp_masked, fwd, bwd = geometry(cfg)
    base = EnvConfig()
    tol = float(base.trench_dig_yaw_tolerance_rad)
    so_min = float(base.trench_dig_standoff_min_m)
    so_max = float(base.trench_dig_standoff_max_m)
    cases = cases_from_panel(args.bank_root, args.dataset, None, args.exclude_prefix)
    print(f"{len(cases)} trench maps", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init,
                  initargs=(cfg, cones, fp_true, fp_masked, fwd, bwd,
                            tol, so_min, so_max)) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(analyze, cases, chunksize=1)):
            results.append(res)
            if i % 20 == 0:
                print(f"{i + 1}/{len(cases)}", flush=True)

    by = {}
    for r in results:
        by.setdefault(r["condition"], []).append(r)
    summary = []
    for cond in sorted(by):
        g = by[cond]
        summary.append({
            "condition": cond, "maps": len(g),
            "target_cells": sum(r["target_cells"] for r in g),
            "complete_any": sum(r["complete_any"] for r in g),
            "complete_true_footprint": sum(r["complete_true"] for r in g),
            "complete_terra_footprint": sum(r["complete_terra"] for r in g),
            "cells_lost_to_footprint_transposition":
                sum(r["cells_lost_to_footprint_transposition"] for r in g),
            "cells_gained_by_transposition":
                sum(r["cells_gained_by_transposition"] for r in g),
            "mean_station_components": float(np.mean([r["station_components"] for r in g])),
            "min_largest_station_component_share":
                float(min(r["largest_station_component_share"] for r in g)),
        })
    payload = {
        "schema": "terra_trench_persistent_station_cover_v1",
        "contract": {"bank": str(args.bank_root), "dataset": args.dataset,
                     "blocked": "padding | all target<0 (order independent)",
                     "maps": len(cases)},
        "summary_by_condition": summary, "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=1))
    for r in summary:
        print(f"{r['condition']:24s} maps={r['maps']:3d} complete: any={r['complete_any']:3d} "
              f"true_fp={r['complete_true_footprint']:3d} terra_fp={r['complete_terra_footprint']:3d} "
              f"cells_lost={r['cells_lost_to_footprint_transposition']:4d} "
              f"gained={r['cells_gained_by_transposition']:4d} "
              f"ncomp={r['mean_station_components']:.2f} minshare={r['min_largest_station_component_share']:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
