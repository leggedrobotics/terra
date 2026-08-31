#!/usr/bin/env python3
"""Can every trench workspace be completed by moving ALONG the axis, with the
dump side of the contract removed?

Two questions, both purely geometric (no controller, no policy):

  A. Can the machine move along the axis at all?  A "lane" is the set of
     yaw-valid, in-band, Terra-legal base poses at one heading for one section.
     Lanes are connected here by FORWARD / BACKWARD only -- no base rotation,
     no lateral motion -- which is exactly "driving along the axis".  Terra's
     5-tile move is ``round(pos + 5 * unit(heading))``, and for the eight
     oblique headings the ideal step has a component of exactly +/-2.5 tiles, so
     the integer step alternates 2/3 with coordinate parity and the machine
     drifts perpendicular to the lane.  The drift per move is reported.

  B. With every dump constraint removed -- no dumpability mask, no accepted
     zone, no same-base dump reach, no spoil, no pile in the cone, and the
     previous-dig mask treated as cleared between digs, i.e. the bucket is
     magically emptied -- does sweeping those lanes with all 12 cabin headings
     produce a gate-admitted dig for every target cell?

Blocked space is the order-independent worst case ``padding | all target<0``,
so the answer cannot depend on the order cells are dug in.  Both Terra's actual
(transposed) footprint test and a correct ``(row, col)`` footprint are reported,
because they differ (see tools/check_trench_persistent_station_cover.py).

Coverage is screened with the translation-invariant cone offset table and then
every claimed cell is re-verified with an exact per-pose Terra cone, so the
"complete" verdict is exact rather than approximate.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import EnvConfig
from terra.map import compute_trench_axis_membership
from terra.state import State

from audit_trench_gate_overrestriction import (
    NH, SHAPE, cases_from_panel, dilate, env_config, free_poses, geometry,
    records_from_metadata,
)

MAX_AXES = 4
MAX_WITNESSES = 12
_W = {}


def _init(cfg, cones, fp_true, fp_masked, fwd, bwd, tol, so_min, so_max):
    _W.update(cfg=cfg, cones=cones, fp_true=fp_true, fp_masked=fp_masked,
              fwd=fwd, bwd=bwd, tol=tol, so_min=so_min, so_max=so_max)
    _W["cone_fn"] = _make_cone_fn(cfg)
    _W["succ"] = move_tables(cfg)


def _make_cone_fn(cfg):
    state = State.new(
        jax.random.PRNGKey(0), cfg,
        np.zeros(SHAPE, np.int8), np.zeros(SHAPE, np.int8),
        -97.0 * np.ones((MAX_AXES, 8), np.float32), np.int32(-1),
        -97.0 * np.ones((64, 3), np.float32), np.int32(-1),
        np.ones(SHAPE, np.bool_), np.zeros(SHAPE, np.int8),
        distance_map_override=np.ones(SHAPE, np.float32),
    )

    def one(r, c, b, k):
        cur = state._get_current_agent_state()._replace(
            pos_base=jnp.stack([r, c]).astype(jnp.int16),
            angle_base=jnp.reshape(b, (1,)).astype(jnp.int8),
            angle_cabin=jnp.reshape(k, (1,)).astype(jnp.int8),
            loaded=jnp.zeros((1,), dtype=jnp.int8),
        )
        return state._set_current_agent_state(cur)._build_dig_dump_cone().reshape(SHAPE)

    return jax.jit(jax.vmap(one))


def move_tables(cfg):
    """Exact per-pose FORWARD/BACKWARD destinations.

    ``State._move_on_orientation`` computes ``round(pos_base + delta_xy)`` in
    float32 with round-half-to-even.  Eight of the twelve headings have a delta
    component of exactly +/-2.5 tiles, so the integer step alternates 2/3 with
    the parity of the coordinate: a single per-heading delta is wrong for half
    the poses and breaks lane connectivity.  Verified against Terra by
    ``scripts/trench_align_scripted_oracle.py`` (1,741 checks).
    """
    angles = jnp.linspace(0, 2 * jnp.pi, NH, endpoint=False)
    angles = (angles + (jnp.pi / 2)) % (2 * jnp.pi)
    xy = int(cfg.agent.move_tiles) * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    grid = jnp.stack(jnp.meshgrid(
        jnp.arange(SHAPE[0], dtype=jnp.int16),
        jnp.arange(SHAPE[1], dtype=jnp.int16), indexing="ij"), axis=-1)
    succ = np.full((NH, 2) + SHAPE + (2,), -1, dtype=np.int32)
    for bh in range(NH):
        for a, src in enumerate((bh, (bh + NH // 2) % NH)):
            cand = np.asarray(jnp.round(grid + xy[src]).astype(jnp.int32))
            inside = ((cand[:, :, 0] >= 0) & (cand[:, :, 0] < SHAPE[0])
                      & (cand[:, :, 1] >= 0) & (cand[:, :, 1] < SHAPE[1]))
            succ[bh, a] = np.where(inside[:, :, None], cand, -1)
    return succ


def lane_components(pose_ok_2d, succ_bh):
    """Union-find over FORWARD/BACKWARD edges only, at one fixed heading.

    ``succ_bh`` is the exact (2, 64, 64, 2) per-pose destination table.
    """
    idx = -np.ones(SHAPE, dtype=np.int32)
    poses = np.argwhere(pose_ok_2d)
    if poses.shape[0] == 0:
        return poses, np.zeros(0, dtype=np.int32), 0
    idx[poses[:, 0], poses[:, 1]] = np.arange(poses.shape[0])
    parent = np.arange(poses.shape[0], dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(2):
        tgt = succ_bh[a][poses[:, 0], poses[:, 1]]
        inside = (tgt[:, 0] >= 0) & (tgt[:, 1] >= 0)
        src = np.flatnonzero(inside)
        dst = idx[tgt[inside, 0], tgt[inside, 1]]
        for sidx, d in zip(src, dst):
            if d >= 0:
                union(int(sidx), int(d))
    labels = np.array([find(i) for i in range(poses.shape[0])], dtype=np.int32)
    _, labels = np.unique(labels, return_inverse=True)
    return poses, labels.astype(np.int32), int(labels.max()) + 1


def sweep_map(case, use_terra_footprint: bool):
    cones = _W["cones"]
    tile = _W["cfg"].tile_size
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

    blocked = padding | dig
    offsets = _W["fp_masked"] if use_terra_footprint else _W["fp_true"]
    free = free_poses(blocked, offsets, transposed=use_terra_footprint)

    covered = np.zeros(n, dtype=bool)
    witness = {}                       # cell -> (row, col, bh, cb)
    lane_stats = []
    for a in range(naxes):
        own = (membership & np.uint8(1 << a)) != 0
        sec_cells = dig & own
        near = dilate(sec_cells, 12)
        for bh in range(NH):
            if not yaw_ok[bh, a]:
                continue
            pose_ok = band[a] & free[bh] & near
            poses, labels, ncomp = lane_components(pose_ok, _W["succ"][bh])
            if poses.shape[0] == 0:
                continue
            for comp in range(ncomp):
                sel = labels == comp
                cpose = poses[sel]
                if cpose.shape[0] == 0:
                    continue
                comp_cov = np.zeros(n, dtype=bool)
                for cb in range(NH):
                    offs = cones[bh][cb]
                    rr = cpose[:, 0:1] + offs[None, :, 0]
                    cc = cpose[:, 1:2] + offs[None, :, 1]
                    ok = (rr >= 0) & (rr < SHAPE[0]) & (cc >= 0) & (cc < SHAPE[1])
                    rrc, ccc = np.clip(rr, 0, SHAPE[0] - 1), np.clip(cc, 0, SHAPE[1] - 1)
                    freshm = dig[rrc, ccc] & ok
                    live = freshm.any(1)
                    nopad = ~np.any(padding[rrc, ccc] & ok, axis=1)
                    mem = membership[rrc, ccc]
                    bad = np.any(freshm & (mem != 0)
                                 & ((mem & np.uint8(1 << a)) == 0), axis=1)
                    good = live & nopad & ~bad
                    for i in np.flatnonzero(good):
                        ids = index[rrc[i][freshm[i]], ccc[i][freshm[i]]]
                        ids = ids[ids >= 0]
                        comp_cov[ids] = True
                        for cid in ids:
                            w = witness.setdefault(int(cid), [])
                            if len(w) < MAX_WITNESSES:
                                w.append((int(cpose[i, 0]), int(cpose[i, 1]), bh, cb))
                covered |= comp_cov
                along = cpose @ np.array([-axes3[a, 0], axes3[a, 1]]) / den[a]
                lane_stats.append({
                    "axis": a, "heading": bh, "poses": int(cpose.shape[0]),
                    "along_axis_extent_tiles": float(along.max() - along.min()),
                    "section_cells": int(sec_cells.sum()),
                    "section_cells_covered": int(np.sum(comp_cov & own[cells[:, 0], cells[:, 1]])),
                })

    # Exact re-verification: the offset-table screen is only ~81% translation
    # exact, so several candidate witnesses per cell are checked with a real
    # per-pose Terra cone and the cell counts as covered if any one of them
    # holds.  A single witness per cell produced ~1.5% false negatives.
    exact_covered = np.zeros(n, dtype=bool)
    flat = [(cid, w) for cid in sorted(witness) for w in witness[cid]]
    if flat:
        w = np.array([x[1] for x in flat], dtype=np.int32)
        masks = np.asarray(_W["cone_fn"](
            jnp.asarray(w[:, 0]), jnp.asarray(w[:, 1]),
            jnp.asarray(w[:, 2]), jnp.asarray(w[:, 3])))
        for j, (cid, _w) in enumerate(flat):
            if exact_covered[cid]:
                continue
            cone = masks[j].astype(bool)
            r, c = cells[cid]
            if not cone[r, c]:
                continue
            if np.any(padding[cone]):
                continue
            bits = 0
            for a in range(naxes):
                if yaw_ok[int(w[j, 2]), a] and band[a, int(w[j, 0]), int(w[j, 1])]:
                    bits |= 1 << a
            fresh = cone & dig
            mem = membership[fresh]
            tr = mem != 0
            if tr.any() and not ((mem[tr] & np.uint8(bits)) != 0).all():
                continue
            exact_covered[cid] = True

    best_lane = {}
    for ls in lane_stats:
        k = ls["axis"]
        cur = best_lane.get(k)
        if cur is None or ls["section_cells_covered"] > cur["section_cells_covered"]:
            best_lane[k] = ls
    return {
        "label": case["label"], "condition": case["condition"],
        "map_id": case["map_id"], "axes": naxes, "target_cells": n,
        "footprint": "terra_transposed" if use_terra_footprint else "corrected",
        "cells_covered_screen": int(covered.sum()),
        "cells_covered_exact": int(exact_covered.sum()),
        "complete_screen": bool(covered.sum() == n),
        "complete_exact": bool(exact_covered.sum() == n),
        "lane_count": len(lane_stats),
        "max_along_axis_extent_tiles":
            max((ls["along_axis_extent_tiles"] for ls in lane_stats), default=0.0),
        "per_axis_best_single_lane_share": {
            str(a): (ls["section_cells_covered"] / max(ls["section_cells"], 1))
            for a, ls in best_lane.items()
        },
        "min_axis_best_single_lane_share":
            min((ls["section_cells_covered"] / max(ls["section_cells"], 1)
                 for ls in best_lane.values()), default=0.0),
    }


def analyze(case):
    return [sweep_map(case, True), sweep_map(case, False)]


def move_drift_table(cfg, succ):
    """Perpendicular drift of Terra's move, measured over a two-move cycle."""
    out = []
    for bh in range(NH):
        th = 2 * np.pi * bh / NH
        ideal = int(cfg.agent.move_tiles) * np.array(
            [np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)])
        unit = ideal / np.linalg.norm(ideal)
        perp = np.array([-unit[1], unit[0]])
        steps, perps = [], []
        for r in range(20, 44):
            for c in range(20, 44):
                d1 = succ[bh, 0, r, c]
                if d1[0] < 0:
                    continue
                d2 = succ[bh, 0, d1[0], d1[1]]
                if d2[0] < 0:
                    continue
                v1 = d1 - np.array([r, c])
                v2 = d2 - d1
                steps.append(tuple(int(x) for x in v1))
                perps.append(float((v1 + v2) @ perp))
        uniq = sorted(set(steps))
        out.append({
            "heading": bh,
            "ideal_step": [round(float(v), 4) for v in ideal],
            "terra_steps_observed": [list(u) for u in uniq],
            "two_move_perp_drift_tiles": {
                "mean": round(float(np.mean(perps)), 4),
                "max_abs": round(float(np.max(np.abs(perps))), 4),
            },
        })
    return out


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
    succ = move_tables(cfg)
    drift = move_drift_table(cfg, succ)
    print("along-axis motion: exact Terra steps and two-move perpendicular drift",
          flush=True)
    for d in drift:
        print(f"  heading {d['heading']:2d} ideal {d['ideal_step']} "
              f"terra_steps {d['terra_steps_observed']} "
              f"two_move_perp_drift mean {d['two_move_perp_drift_tiles']['mean']:+.4f} "
              f"max|.| {d['two_move_perp_drift_tiles']['max_abs']:.4f}", flush=True)

    cases = cases_from_panel(args.bank_root, args.dataset, None, args.exclude_prefix)
    print(f"{len(cases)} trench maps", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init,
                  initargs=(cfg, cones, fp_true, fp_masked, fwd, bwd,
                            tol, so_min, so_max)) as pool:
        results = []
        for i, pair in enumerate(pool.imap_unordered(analyze, cases, chunksize=1)):
            results.extend(pair)
            if i % 20 == 0:
                print(f"{i + 1}/{len(cases)}", flush=True)

    summary = []
    for fp in ("terra_transposed", "corrected"):
        rs = [r for r in results if r["footprint"] == fp]
        by = {}
        for r in rs:
            by.setdefault(r["condition"], []).append(r)
        for cond in sorted(by):
            g = by[cond]
            summary.append({
                "footprint": fp, "condition": cond, "maps": len(g),
                "target_cells": sum(r["target_cells"] for r in g),
                "complete_exact": sum(r["complete_exact"] for r in g),
                "cells_covered_exact": sum(r["cells_covered_exact"] for r in g),
                "min_axis_best_single_lane_share":
                    float(min(r["min_axis_best_single_lane_share"] for r in g)),
                "mean_axis_best_single_lane_share":
                    float(np.mean([r["min_axis_best_single_lane_share"] for r in g])),
                "max_along_axis_extent_tiles":
                    float(max(r["max_along_axis_extent_tiles"] for r in g)),
            })
    payload = {
        "schema": "terra_trench_axis_sweep_feasibility_v1",
        "contract": {
            "bank": str(args.bank_root), "dataset": args.dataset,
            "blocked": "padding | all target<0 (order independent)",
            "dump_constraints": "removed (no dumpability, accepted zone, dump reach, "
                                "spoil, pile-in-cone; previous-dig mask cleared between digs)",
            "motion": "FORWARD/BACKWARD at a fixed heading only, plus free cabin rotation",
            "yaw_tolerance_rad": tol, "standoff_band_m": [so_min, so_max],
            "maps": len(cases),
        },
        "move_drift": drift,
        "summary_by_condition": summary,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=1))
    for fp in ("terra_transposed", "corrected"):
        rows = [r for r in summary if r["footprint"] == fp]
        tot_m = sum(r["maps"] for r in rows)
        tot_c = sum(r["complete_exact"] for r in rows)
        tot_cells = sum(r["target_cells"] for r in rows)
        tot_cov = sum(r["cells_covered_exact"] for r in rows)
        print(f"\n=== footprint: {fp} ===")
        for r in rows:
            print(f"  {r['condition']:26s} complete {r['complete_exact']:3d}/{r['maps']:3d} "
                  f"cells {r['cells_covered_exact']:6d}/{r['target_cells']:6d} "
                  f"min_single_lane_share {r['min_axis_best_single_lane_share']:.3f} "
                  f"lane_extent {r['max_along_axis_extent_tiles']:.1f}")
        print(f"  TOTAL complete {tot_c}/{tot_m}  cells {tot_cov}/{tot_cells}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
