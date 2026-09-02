#!/usr/bin/env python3
"""Is a gate-admissible fresh-dig cover reachable under Terra's runtime pose rules?

This tool answers the cover question under Terra's actual footprint rule, and
under the retired one.  ``compute_polygon_mask`` used to return a ``(col, row)``
raster while ``pos_base`` is ``(row, col)``, so ``_is_valid_move`` evaluated
occupancy at the mirror position and the *mirror image* of the trench acted as
the obstacle field.  terra commit 566867db fixed that, so the ``terra`` columns
below are now the correct ``(row, col)`` model (0.9995 agreement with
``State._is_valid_move``) and the ``legacy_mirror`` columns are the retired
behaviour the pre-fix receipts measured.

For every trench map it computes, with ``blocked = padding | all target<0``
(the order-independent worst case, so a controller using only these stations can
never wall itself off):

  * the gate-admissible (pose, cabin) dig stations;
  * which of them are legal poses under Terra's footprint test today, and which
    under the retired mirror model;
  * the target cells covered by each of those two station sets;
  * whether the Terra-legal station set is connected.

The gap between the two cover numbers is what the transposition used to cost.

Gate semantics follow ``EnvConfig``: v2 (the default) is yaw-parallel AND
"on the line" (perpendicular offset <= ``trench_dig_max_offset_m``,
``--max-offset-m``, <= 0 disables the clause); ``--gate-v1`` restores the
retired [3.5, 7.0] m band.  READ THE PERSISTENT COLUMN WITH THE CLAUSE IN MIND:
a *persistent* station must stay legal with the WHOLE trench dug, and a machine
standing on the trench line does not.  Any bound tight enough to force the
machine onto the line therefore empties the persistent station set by
construction -- that is the definition colliding with the clause, not a coverage
loss in the gate.  The model that matches a monotone dig-ahead-and-retreat is
the ``fresh`` (padding-only) blocked space, reported by
``tools/check_trench_axis_sweep_feasibility.py``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import deque
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from terra.map import compute_trench_axis_membership

from audit_trench_gate_overrestriction import (
    NH, SHAPE, cases_from_panel, dilate, env_config, free_poses, gate_contract,
    gather, geometry, records_from_metadata, standoff_bands,
    terra_gate_selfcheck,
)

_W = {}


def _init(cfg, cones, fp_true, fp_masked, fwd, bwd, tol, so_min, so_max,
          enforce_band=True, max_offset=0.0):
    _W.update(cfg=cfg, cones=cones, fp_true=fp_true, fp_masked=fp_masked,
              fwd=fwd, bwd=bwd, tol=tol, so_min=so_min, so_max=so_max,
              enforce_band=bool(enforce_band), max_offset=float(max_offset))


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
    enforce_band = _W["enforce_band"]
    max_offset = _W["max_offset"]
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
    band, _band_diag = standoff_bands(standoff, so_min, so_max, enforce_band,
                                      max_offset)
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
    free_terra = free_poses(blocked, _W["fp_true"], transposed=False)
    free_legacy = free_poses(blocked, _W["fp_masked"], transposed=True)

    cov_any = np.zeros(n, dtype=bool)
    cov_terra = np.zeros(n, dtype=bool)
    cov_legacy = np.zeros(n, dtype=bool)
    station_terra = np.zeros((NH,) + SHAPE, dtype=bool)
    near = dilate(dig, 12)
    for bh in range(NH):
        pm = near & ~padding
        poses = np.argwhere(pm)
        if poses.shape[0] == 0:
            continue
        pbits = bits[bh][pm]
        okt = free_terra[bh][pm]
        okr = free_legacy[bh][pm]
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
                    cov_legacy[ids] = True

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
        "cells_admissible_legacy_mirror_persistent_station": int(cov_legacy.sum()),
        "cells_lost_to_legacy_mirror": int(np.sum(cov_terra & ~cov_legacy)),
        "cells_gained_by_legacy_mirror": int(np.sum(cov_legacy & ~cov_terra)),
        "terra_legal_stations": int(station_terra.sum()),
        "station_components": len(st_comps),
        "largest_station_component_share":
            float(max(sizes.values()) / max(len(st_nodes), 1)) if st_nodes.size else 0.0,
        "complete_terra": bool(cov_terra.sum() == n),
        "complete_legacy_mirror": bool(cov_legacy.sum() == n),
        "complete_any": bool(cov_any.sum() == n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-root", type=Path, required=True)
    ap.add_argument("--dataset", default="evaluation/gate_main/development")
    ap.add_argument("--exclude-prefix", nargs="*", default=["trn-net4-"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="analyze only the first N maps (timing slice)")
    ap.add_argument("--max-offset-m", type=float, default=None,
                    help="override the v2 'on the line' bound "
                         "(EnvConfig.trench_dig_max_offset_m, metres); "
                         "<= 0 disables the clause (yaw-parallel only). "
                         "Inert under --gate-v1.")
    ap.add_argument("--selfcheck-maps", type=int, default=2,
                    help="maps on which the numpy replica is asserted against "
                         "Terra's exported verdict (0 disables)")
    ap.add_argument("--gate-v1", action="store_true",
                    help="force the retired v1 semantics (perpendicular "
                         "standoff band enforced on top of yaw-parallel)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = env_config()
    cones, fp_true, fp_masked, fwd, bwd = geometry(cfg)
    gate = gate_contract(args.gate_v1, args.max_offset_m)
    tol, so_min, so_max = gate["tol"], gate["so_min"], gate["so_max"]
    enforce_band = gate["enforce_band"]
    max_offset = gate["max_offset"]
    print(f"gate semantics {gate['semantics']} "
          f"(standoff band enforced={enforce_band}, forced={gate['forced']}; "
          f"on-line clause: {gate['on_line_clause']})",
          flush=True)
    cases = cases_from_panel(args.bank_root, args.dataset, None, args.exclude_prefix)
    if args.limit is not None:
        cases = cases[: args.limit]
    print(f"{len(cases)} trench maps", flush=True)

    selfcheck = []
    for case in cases[: max(args.selfcheck_maps, 0)]:
        row = terra_gate_selfcheck(cfg, case, tol=tol, so_min=so_min,
                                   so_max=so_max, enforce_band=enforce_band,
                                   max_offset=max_offset)
        selfcheck.append(row)
        print(f"selfcheck {row['map']}: probes={row['probes']} "
              f"applicable={row['applicable_probes']} "
              f"mismatches={row['mismatches']}", flush=True)
    if any(row["mismatches"] for row in selfcheck):
        raise SystemExit("replica disagrees with Terra's exported gate verdict")

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init,
                  initargs=(cfg, cones, fp_true, fp_masked, fwd, bwd,
                            tol, so_min, so_max, enforce_band,
                            max_offset)) as pool:
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
            "complete_terra_footprint": sum(r["complete_terra"] for r in g),
            "complete_legacy_mirror_footprint":
                sum(r["complete_legacy_mirror"] for r in g),
            "cells_lost_to_legacy_mirror":
                sum(r["cells_lost_to_legacy_mirror"] for r in g),
            "cells_gained_by_legacy_mirror":
                sum(r["cells_gained_by_legacy_mirror"] for r in g),
            "mean_station_components": float(np.mean([r["station_components"] for r in g])),
            "min_largest_station_component_share":
                float(min(r["largest_station_component_share"] for r in g)),
        })
    payload = {
        "schema": "terra_trench_persistent_station_cover_v1",
        "contract": {"bank": str(args.bank_root), "dataset": args.dataset,
                     "blocked": "padding | all target<0 (order independent)",
                     "footprint_terra": "compute_polygon_mask offsets in "
                                        "(row, col) -- Terra since 566867db",
                     "footprint_legacy_mirror": "the retired (col, row) raster",
                     "gate_semantics": gate["semantics"],
                     "standoff_band_enforced": enforce_band,
                     "gate_v1_forced": gate["forced"],
                     "config_trench_dig_standoff_enforced":
                         gate["config_default_enforced"],
                     "max_offset_m": max_offset,
                     "max_offset_forced": gate["max_offset_forced"],
                     "config_trench_dig_max_offset_m":
                         gate["config_default_max_offset"],
                     "on_line_clause": gate["on_line_clause"],
                     "terra_replica_selfcheck": selfcheck,
                     "yaw_tolerance_rad": tol,
                     "standoff_band_m": [so_min, so_max],
                     "maps": len(cases)},
        "summary_by_condition": summary, "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=1))
    for r in summary:
        print(f"{r['condition']:24s} maps={r['maps']:3d} complete: any={r['complete_any']:3d} "
              f"terra_fp={r['complete_terra_footprint']:3d} "
              f"legacy_fp={r['complete_legacy_mirror_footprint']:3d} "
              f"cells_lost_to_legacy={r['cells_lost_to_legacy_mirror']:4d} "
              f"gained_by_legacy={r['cells_gained_by_legacy_mirror']:4d} "
              f"ncomp={r['mean_station_components']:.2f} minshare={r['min_largest_station_component_share']:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
