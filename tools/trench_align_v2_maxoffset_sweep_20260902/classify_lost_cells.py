#!/usr/bin/env python3
"""Why does a target cell stop being admissibly diggable under an offset bound?

For every map in a dataset this recomputes the over-restriction audit's
``cells_admissible`` set twice -- once with the bound disabled (yaw-parallel
only) and once with the bound -- and, for each cell the bound loses, says which
mechanism removed it:

  no_on_line_pose   no (pose, cabin) exists at all whose base centre is within
                    the bound of an owning section, yaw-parallel, off padding,
                    with the cell in a padding-free cone.  Geometry: the lane is
                    off the map or under padding.
  junction_veto     such candidates exist, but every one of them drags an
                    exclusive cell of a perpendicular section into the same
                    cone, so the all-or-nothing DO is refused.

Each lost cell also carries its owner count, its distance to the nearest
padding cell and to the map border, and its distance to the nearest cell owned
by a DIFFERENT section (small = junction neighbourhood).

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=<terra> python .../classify_lost_cells.py \
      --bank-root <bank> --dataset evaluation/gate_main/development \
      --max-offset-m 2.29 --workers 24 --output lost_b229.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np

from tools.audit_trench_gate_overrestriction import (
    NH, SHAPE, cases_from_panel, dilate, env_config, gather, gate_contract,
    geometry, records_from_metadata, standoff_bands,
)
from terra.map import compute_trench_axis_membership
import jax.numpy as jnp

_W = {}


def _init(cfg, cones, tol, so_min, so_max, max_offset):
    _W.update(cfg=cfg, cones=cones, tol=tol, so_min=so_min, so_max=so_max,
              max_offset=float(max_offset))


def _admissible(dig, padding, membership, bits, cones, cell_index, n_target):
    """(cov_admissible, per-cell candidate counts, per-cell vetoed counts)."""
    cov = np.zeros(n_target, dtype=bool)
    cand = np.zeros(n_target, dtype=np.int32)
    veto = np.zeros(n_target, dtype=np.int32)
    near = dilate(dig, 12)
    for bh in range(NH):
        pose_any = near & ~padding
        poses = np.argwhere(pose_any)
        if poses.shape[0] == 0:
            continue
        pbits = bits[bh][pose_any]
        for cb in range(NH):
            offs = cones[bh][cb]
            counts, (rrc, ccc, ok) = gather(poses, offs, [dig, padding])
            n_fresh, n_pad = counts
            live = (n_fresh > 0) & (n_pad == 0)
            if not live.any():
                continue
            for i in np.flatnonzero(live):
                b = pbits[i]
                if b == 0:
                    continue
                freshm = dig[rrc[i], ccc[i]] & ok[i]
                cells = cell_index[rrc[i][freshm], ccc[i][freshm]]
                m = membership[rrc[i][freshm], ccc[i][freshm]]
                trench = m != 0
                if not trench.any():
                    continue
                cand[cells] += 1
                if ((m[trench] & b) != 0).all():
                    cov[cells] = True
                else:
                    veto[cells] += 1
    return cov, cand, veto


def analyze(case):
    cfg, cones = _W["cfg"], _W["cones"]
    tol, so_min, so_max = _W["tol"], _W["so_min"], _W["so_max"]
    tile = cfg.tile_size
    target = np.load(case["images"], allow_pickle=False).astype(np.int8)
    padding = np.load(case["occupancy"], allow_pickle=False).astype(bool)
    metadata = json.loads(Path(case["metadata"]).read_text())
    records, naxes, _half = records_from_metadata(metadata)
    membership = np.asarray(compute_trench_axis_membership(
        jnp.asarray(target), jnp.asarray(records), jnp.int32(naxes)
    )).astype(np.uint8)

    dig = target < 0
    n = int(dig.sum())
    cell_index = -np.ones(SHAPE, dtype=np.int32)
    cells = np.argwhere(dig)
    cell_index[cells[:, 0], cells[:, 1]] = np.arange(n)

    axes3 = records[:naxes, :3].astype(np.float64)
    den = np.maximum(np.linalg.norm(axes3[:, :2], axis=1), 1e-6)
    rows, cols = np.meshgrid(np.arange(SHAPE[0], dtype=np.float64),
                             np.arange(SHAPE[1], dtype=np.float64), indexing="ij")
    standoff = np.stack([
        np.abs(axes3[a, 0] * cols + axes3[a, 1] * rows + axes3[a, 2]) / den[a] * tile
        for a in range(naxes)])
    tg = np.stack([-axes3[:, 0], axes3[:, 1]], axis=1)
    tn = np.maximum(np.linalg.norm(tg, axis=1), 1e-6)
    yaw_ok = np.zeros((NH, naxes), dtype=bool)
    for bh in range(NH):
        th = 2 * np.pi * bh / NH
        f = np.array([-np.sin(th), np.cos(th)])
        yaw_ok[bh] = np.arccos(np.clip(np.abs(tg @ f) / tn, 0, 1)) <= tol

    def bits_for(band):
        b = np.zeros((NH,) + SHAPE, dtype=np.uint8)
        for bh in range(NH):
            for a in range(naxes):
                if yaw_ok[bh, a]:
                    b[bh] |= np.where(band[a], np.uint8(1 << a), np.uint8(0))
        return b

    band_off, _ = standoff_bands(standoff, so_min, so_max, False, 0.0)
    band_bnd, _ = standoff_bands(standoff, so_min, so_max, False, _W["max_offset"])
    cov_off, _c0, _v0 = _admissible(dig, padding, membership, bits_for(band_off),
                                    cones, cell_index, n)
    cov_bnd, cand, veto = _admissible(dig, padding, membership,
                                      bits_for(band_bnd), cones, cell_index, n)

    lost_idx = np.flatnonzero(cov_off & ~cov_bnd)
    # geometric context
    pad_dist = _chebyshev_distance_to(padding)
    border = np.minimum.reduce([rows, cols, SHAPE[0] - 1 - rows, SHAPE[1] - 1 - cols])
    owner_count = np.unpackbits(membership[:, :, None], axis=2).sum(axis=2)
    cross_dist = np.full(SHAPE, 99.0)
    if naxes > 1:
        for a in range(naxes):
            own_a = (membership & np.uint8(1 << a)) != 0
            other = ((membership & ~np.uint8(1 << a)) != 0) & (membership != 0)
            if not other.any():
                continue
            d = _chebyshev_distance_to(other)
            cross_dist = np.where(own_a, np.minimum(cross_dist, d), cross_dist)

    out = []
    for i in lost_idx:
        r, c = cells[i]
        out.append({
            "row": int(r), "col": int(c),
            "owners": int(owner_count[r, c]),
            "mechanism": ("junction_veto" if int(cand[i]) > 0
                          else "no_on_line_pose"),
            "candidates_bounded": int(cand[i]),
            "vetoed_bounded": int(veto[i]),
            "padding_dist": float(pad_dist[r, c]),
            "border_dist": float(border[r, c]),
            "cross_section_dist": float(cross_dist[r, c]),
        })
    return {
        "label": case["label"], "condition": case["condition"],
        "map_id": case["map_id"], "target_cells": n, "axes": naxes,
        "cells_admissible_off": int(cov_off.sum()),
        "cells_admissible_bounded": int(cov_bnd.sum()),
        "lost": len(out), "lost_cells": out,
    }


def _chebyshev_distance_to(mask):
    """Chebyshev distance from every cell to the nearest True in ``mask``."""
    if not mask.any():
        return np.full(SHAPE, 99.0)
    d = np.full(SHAPE, 99.0)
    cur = mask.copy()
    d[cur] = 0.0
    for k in range(1, 20):
        nxt = dilate(cur, 1)
        newly = nxt & ~cur
        if not newly.any():
            break
        d[newly] = float(k)
        cur = nxt
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-root", type=Path, required=True)
    ap.add_argument("--dataset", default="evaluation/gate_main/development")
    ap.add_argument("--exclude-prefix", nargs="*", default=[])
    ap.add_argument("--max-offset-m", type=float, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = env_config()
    cones, _fp_true, _fp_masked, _fwd, _bwd = geometry(cfg)
    gate = gate_contract(False, args.max_offset_m)
    cases = cases_from_panel(args.bank_root, args.dataset, None, args.exclude_prefix)
    print(f"{len(cases)} trench maps, bound {gate['max_offset']} m", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init,
                  initargs=(cfg, cones, gate["tol"], gate["so_min"],
                            gate["so_max"], gate["max_offset"])) as pool:
        results = list(pool.imap_unordered(analyze, cases, chunksize=1))
    results.sort(key=lambda r: r["label"])

    mech = Counter()
    by_cond = {}
    for r in results:
        e = by_cond.setdefault(r["condition"], {"maps": 0, "maps_with_loss": 0,
                                                "lost": 0, "target": 0})
        e["maps"] += 1
        e["target"] += r["target_cells"]
        e["lost"] += r["lost"]
        e["maps_with_loss"] += int(r["lost"] > 0)
        for cell in r["lost_cells"]:
            mech[cell["mechanism"]] += 1
    lost_all = [c for r in results for c in r["lost_cells"]]
    payload = {
        "schema": "terra_trench_maxoffset_lost_cells_v1",
        "contract": {"bank": str(args.bank_root), "dataset": args.dataset,
                     "max_offset_m": gate["max_offset"],
                     "maps": len(cases)},
        "mechanism_counts": dict(mech),
        "by_condition": by_cond,
        "distributions": {
            "owners": dict(Counter(c["owners"] for c in lost_all)),
            "padding_dist": dict(Counter(c["padding_dist"] for c in lost_all)),
            "border_dist": dict(Counter(c["border_dist"] for c in lost_all)),
            "cross_section_dist": dict(Counter(c["cross_section_dist"] for c in lost_all)),
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=1))
    print(json.dumps({"mechanisms": dict(mech),
                      "total_lost": len(lost_all),
                      "maps_with_loss": sum(e["maps_with_loss"] for e in by_cond.values())},
                     indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
