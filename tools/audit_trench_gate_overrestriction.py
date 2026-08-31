#!/usr/bin/env python3
"""Over-restriction audit of the fresh-trench dig-alignment gate.

The research note's contract says the gate should refuse a fresh trench dig
only when the base pose is physically incompatible with the owning section.
This tool asks the converse question on real bank maps: does the *implemented*
gate ever refuse a dig the contract admits, and how much of the reachable
target set does each of its clauses cost?

Measured per map (initial full-start map, so every target cell is fresh):

(a) standoff band vs workspace annulus
      distinct in-band standoff lanes available at integer base cells; the
      sub-band from which both trench width edges are abeam-reachable; and
      target cells reachable from *some* pose but from no in-band pose.

(b) finite-section membership
      target cells with an empty owner bitmask; cells assigned through the
      +1.5-cell nearest-section fallback rather than the generated
      half-width + 0.5; the largest distance from a target cell to its
      nearest generated section.

(c) junction all-or-nothing veto
      how often a candidate (pose, cabin) that is yaw/standoff-valid for some
      section is nonetheless refused because the same cone holds a cell owned
      exclusively by a perpendicular section, and whether any target cell is
      reachable *only* through such vetoed candidates.

(d) applicability vs resolvability
      target cells that are applicable somewhere but admissible nowhere.

(e) footprint transposition (Terra runtime fact, not the gate)
      ``State._is_valid_move`` rasterises the footprint polygon as ``(col,row)``
      while ``pos_base`` is ``(row,col)``, so occupancy is tested at the mirror
      position.  Reported as the number of admissible dig stations that gain or
      lose legality when the trench is fully dug.

Nothing is written except the output JSON.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig, EnvConfig, MapsDimsConfig
from terra.env import TerraEnvBatch
from terra.map import compute_trench_axis_membership
from terra.state import State
from terra.utils import compute_polygon_mask

SHAPE = (64, 64)
NH = 12
MAX_AXES = 4


def env_config() -> EnvConfig:
    batch_env = object.__new__(TerraEnvBatch)
    batch_env.batch_cfg = BatchConfig()._replace(
        maps_dims=MapsDimsConfig(maps_edge_length=SHAPE[0])
    )
    base = EnvConfig()
    batched = base._replace(
        agent=base.agent._replace(dig_depth=jnp.ones((1,), dtype=jnp.int32))
    )
    updated = batch_env.update_env_cfgs(batched)
    return base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(edge_length_px=SHAPE[0]),
        agent_types=(0,),
        action_types=(0,),
    )


def geometry(cfg):
    state = State.new(
        jax.random.PRNGKey(0), cfg,
        np.zeros(SHAPE, np.int8), np.zeros(SHAPE, np.int8),
        -97.0 * np.ones((MAX_AXES, 8), np.float32), np.int32(-1),
        -97.0 * np.ones((64, 3), np.float32), np.int32(-1),
        np.ones(SHAPE, np.bool_), np.zeros(SHAPE, np.int8),
        distance_map_override=np.ones(SHAPE, np.float32),
    )
    center = np.array([32, 32])

    def pose(bh, cb):
        cur = state._get_current_agent_state()._replace(
            pos_base=jnp.array([32, 32], dtype=jnp.int16),
            angle_base=jnp.array([bh], dtype=jnp.int8),
            angle_cabin=jnp.array([cb], dtype=jnp.int8),
            loaded=jnp.zeros((1,), dtype=jnp.int8),
        )
        return state._set_current_agent_state(cur)

    cones, fp_true, fp_masked, fwd, bwd = [], [], [], [], []
    for bh in range(NH):
        cones.append([
            (np.argwhere(np.asarray(pose(bh, cb)._build_dig_dump_cone()).reshape(SHAPE))
             - center).astype(np.int32)
            for cb in range(NH)
        ])
        p = pose(bh, 0)
        cur = p._get_current_agent_state()
        corners = np.asarray(p._get_agent_corners(
            cur.pos_base, base_orientation=cur.angle_base,
            agent_width=cfg.agent.width, agent_height=cfg.agent.height))
        fpm = np.asarray(compute_polygon_mask(jnp.asarray(corners), 64, 64))
        offs = (np.argwhere(fpm) - center).astype(np.int32)
        fp_masked.append(offs)                       # Terra's (col,row)-indexed raster
        fp_true.append(offs[:, ::-1].copy())         # the footprint a correct raster would give
        f = np.asarray(p._handle_move_forward()._get_current_agent_state().pos_base).reshape(-1)
        b = np.asarray(p._handle_move_backward()._get_current_agent_state().pos_base).reshape(-1)
        fwd.append(tuple(int(v) for v in (f - center)))
        bwd.append(tuple(int(v) for v in (b - center)))
    return cones, fp_true, fp_masked, fwd, bwd


def free_poses(blocked, offsets_per_heading, transposed):
    out = np.zeros((NH,) + SHAPE, dtype=bool)
    pad = 40
    big = np.ones((SHAPE[0] + 2 * pad, SHAPE[1] + 2 * pad), dtype=bool)
    big[pad:pad + SHAPE[0], pad:pad + SHAPE[1]] = blocked
    for bh in range(NH):
        acc = np.zeros(SHAPE, dtype=bool)
        for da, db in offsets_per_heading[bh]:
            acc |= big[pad + da:pad + da + SHAPE[0], pad + db:pad + db + SHAPE[1]]
        out[bh] = (~acc).T if transposed else (~acc)
    return out


def gather(poses, offs, maps):
    rr = poses[:, 0:1] + offs[None, :, 0]
    cc = poses[:, 1:2] + offs[None, :, 1]
    ok = (rr >= 0) & (rr < SHAPE[0]) & (cc >= 0) & (cc < SHAPE[1])
    rrc = np.clip(rr, 0, SHAPE[0] - 1)
    ccc = np.clip(cc, 0, SHAPE[1] - 1)
    return [np.sum(m[rrc, ccc] & ok, axis=1) for m in maps], (rrc, ccc, ok)


def dilate(mask, radius):
    out = mask.copy()
    for _ in range(radius):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def records_from_metadata(metadata):
    axes = metadata["axes_ABC"]
    arms = metadata.get("trench_segments_yx") or metadata["trench_arms"]
    half = float(metadata["trench_half_width_tiles"])
    rows = []
    for axis, arm in zip(axes, arms):
        rows.append([float(axis["A"]), float(axis["B"]), float(axis["C"]),
                     float(arm[0][0]), float(arm[0][1]),
                     float(arm[-1][0]), float(arm[-1][1]), half])
    while len(rows) < MAX_AXES:
        rows.append([-97.0] * 8)
    return np.asarray(rows, dtype=np.float16).astype(np.float32), len(axes), half


def segment_distances(records, naxes):
    rows, cols = np.meshgrid(np.arange(SHAPE[0], dtype=np.float64),
                             np.arange(SHAPE[1], dtype=np.float64), indexing="ij")
    pts = np.stack([rows, cols], axis=-1)
    out = np.full((naxes,) + SHAPE, np.inf)
    for a in range(naxes):
        start = records[a, 3:5].astype(np.float64)
        end = records[a, 5:7].astype(np.float64)
        seg = end - start
        nsq = max(float(seg @ seg), 1e-6)
        t = np.clip(((pts - start) @ seg) / nsq, 0.0, 1.0)
        closest = start + t[..., None] * seg
        out[a] = np.linalg.norm(pts - closest, axis=-1)
    return out


_W = {}


def _init(cfg, cones, fp_true, fp_masked, fwd, bwd, tol, so_min, so_max):
    _W.update(cfg=cfg, cones=cones, fp_true=fp_true, fp_masked=fp_masked,
              fwd=fwd, bwd=bwd, tol=tol, so_min=so_min, so_max=so_max)


def analyze(case):
    cones = _W["cones"]
    tile = _W["cfg"].tile_size
    tol = _W["tol"]
    so_min, so_max = _W["so_min"], _W["so_max"]

    target = np.load(case["images"], allow_pickle=False).astype(np.int32)
    padding = np.load(case["occupancy"], allow_pickle=False).astype(bool)
    metadata = json.loads(Path(case["metadata"]).read_text())
    records, naxes, half = records_from_metadata(metadata)
    membership = np.asarray(compute_trench_axis_membership(
        jnp.asarray(target.astype(np.int8)), jnp.asarray(records), jnp.int32(naxes)
    )).astype(np.uint8)

    dig = target < 0
    n_target = int(dig.sum())
    cell_index = -np.ones(SHAPE, dtype=np.int32)
    tcells = np.argwhere(dig)
    cell_index[tcells[:, 0], tcells[:, 1]] = np.arange(n_target)

    # ---- (b) membership -------------------------------------------------
    segd = segment_distances(records, naxes)
    nearest = segd.min(axis=0)
    generated = segd <= (half + 0.5)
    any_generated = generated.any(axis=0)
    result = {
        "label": case["label"], "condition": case["condition"],
        "dataset": case["dataset"], "map_id": case["map_id"],
        "axes": naxes, "target_cells": n_target, "half_width_tiles": half,
        "membership_empty_cells": int(np.sum(dig & (membership == 0))),
        "fringe_fallback_cells": int(np.sum(dig & ~any_generated)),
        "max_nearest_segment_distance": float(nearest[dig].max()) if n_target else 0.0,
        "cells_owned_by_multiple_sections": int(np.sum(
            dig & (np.unpackbits(membership[:, :, None], axis=2).sum(axis=2) > 1))),
    }

    # ---- geometry of the gate-valid pose set ----------------------------
    axes3 = records[:naxes, :3].astype(np.float64)
    denom = np.maximum(np.linalg.norm(axes3[:, :2], axis=1), 1e-6)
    rows, cols = np.meshgrid(np.arange(SHAPE[0], dtype=np.float64),
                             np.arange(SHAPE[1], dtype=np.float64), indexing="ij")
    standoff = np.stack([
        np.abs(axes3[a, 0] * cols + axes3[a, 1] * rows + axes3[a, 2]) / denom[a] * tile
        for a in range(naxes)
    ])
    band = (standoff >= so_min) & (standoff <= so_max)
    tangents = np.stack([-axes3[:, 0], axes3[:, 1]], axis=1)
    tnorm = np.maximum(np.linalg.norm(tangents, axis=1), 1e-6)
    yaw_ok = np.zeros((NH, naxes), dtype=bool)
    for bh in range(NH):
        theta = 2.0 * np.pi * bh / NH
        forward = np.array([-np.sin(theta), np.cos(theta)])
        yaw_ok[bh] = np.arccos(
            np.clip(np.abs(tangents @ forward) / tnorm, 0.0, 1.0)) <= tol
    bits = np.zeros((NH,) + SHAPE, dtype=np.uint8)
    for bh in range(NH):
        for a in range(naxes):
            if yaw_ok[bh, a]:
                bits[bh] |= np.where(band[a], np.uint8(1 << a), np.uint8(0))

    # in-band lanes actually available on the integer grid
    lanes = []
    for a in range(naxes):
        vals = np.unique(np.round(standoff[a][band[a]], 4))
        lanes.append(sorted(float(v) for v in vals))
    result["inband_standoff_lanes_per_axis"] = lanes
    result["inband_pose_count"] = int(np.sum(bits != 0))

    # ---- candidate enumeration ------------------------------------------
    near = dilate(dig, 12)
    cov_any = np.zeros(n_target, dtype=bool)         # reachable from any pose
    cov_inband = np.zeros(n_target, dtype=bool)      # reachable from a bits!=0 pose
    cov_admissible = np.zeros(n_target, dtype=bool)  # reachable by an admitted DO
    n_appl = n_appl_inband = n_veto = n_veto_mixed = 0
    admissible_stations = np.zeros((NH,) + SHAPE, dtype=bool)

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
            idx = np.flatnonzero(live)
            mem = membership[rrc, ccc]
            freshm = dig[rrc, ccc] & ok
            for i in idx:
                cells = cell_index[rrc[i][freshm[i]], ccc[i][freshm[i]]]
                cov_any[cells] = True
                b = pbits[i]
                if b == 0:
                    continue
                m = mem[i][freshm[i]]
                trench_cells = m != 0
                if not trench_cells.any():
                    continue
                n_appl_inband += 1
                cov_inband[cells] = True
                ok_bits = (m[trench_cells] & b) != 0
                if ok_bits.all():
                    cov_admissible[cells] = True
                    admissible_stations[bh, poses[i, 0], poses[i, 1]] = True
                else:
                    n_veto += 1
                    if ok_bits.any():
                        n_veto_mixed += 1
    # applicability count without the band filter
    result.update({
        "cells_reachable_any_pose": int(cov_any.sum()),
        "cells_reachable_inband_pose": int(cov_inband.sum()),
        "cells_admissible": int(cov_admissible.sum()),
        "cells_reachable_but_never_inband": int(np.sum(cov_any & ~cov_inband)),
        "cells_inband_but_never_admissible": int(np.sum(cov_inband & ~cov_admissible)),
        "cells_reachable_but_never_admissible": int(np.sum(cov_any & ~cov_admissible)),
        "inband_applicable_candidates": int(n_appl_inband),
        "veto_candidates": int(n_veto),
        "veto_candidates_mixed": int(n_veto_mixed),
        "veto_rate": float(n_veto / max(n_appl_inband, 1)),
    })

    # ---- (e) footprint transposition on the fully-dug map ---------------
    blocked_final = padding | dig
    free_true = free_poses(blocked_final, _W["fp_true"], transposed=False)
    free_terra = free_poses(blocked_final, _W["fp_masked"], transposed=True)
    st = admissible_stations
    result.update({
        "admissible_stations": int(st.sum()),
        "admissible_stations_legal_true_footprint": int(np.sum(st & free_true)),
        "admissible_stations_legal_terra_footprint": int(np.sum(st & free_terra)),
        "stations_terra_only": int(np.sum(st & free_terra & ~free_true)),
        "stations_true_only": int(np.sum(st & free_true & ~free_terra)),
    })

    # connectivity of the Terra-legal admissible station set on the dug map
    def components(free):
        seen = np.zeros((NH,) + SHAPE, dtype=bool)
        comp_of = {}
        cid = 0
        nodes = np.argwhere(free)
        for h0, r0, c0 in nodes:
            if seen[h0, r0, c0]:
                continue
            q = deque([(int(h0), int(r0), int(c0))])
            seen[h0, r0, c0] = True
            comp_of[(int(h0), int(r0), int(c0))] = cid
            while q:
                h, r, c = q.popleft()
                nb = [(r, c, (h - 1) % NH), (r, c, (h + 1) % NH)]
                for dr, dc in (_W["fwd"][h], _W["bwd"][h]):
                    nb.append((r + dr, c + dc, h))
                for rr, cc, hh in nb:
                    if 0 <= rr < SHAPE[0] and 0 <= cc < SHAPE[1] and free[hh, rr, cc] \
                            and not seen[hh, rr, cc]:
                        seen[hh, rr, cc] = True
                        comp_of[(hh, rr, cc)] = cid
                        q.append((hh, rr, cc))
            cid += 1
        return comp_of

    comp_of = components(free_terra)
    station_nodes = [tuple(int(v) for v in n) for n in np.argwhere(st & free_terra)]
    comps = {comp_of[(h, r, c)] for h, r, c in station_nodes}
    sizes = {}
    for h, r, c in station_nodes:
        k = comp_of[(h, r, c)]
        sizes[k] = sizes.get(k, 0) + 1
    result["station_components"] = len(comps)
    result["largest_station_component_fraction"] = (
        float(max(sizes.values()) / len(station_nodes)) if station_nodes else 0.0
    )
    return result


def cases_from_panel(bank_root: Path, relative: str, include, exclude):
    directory = bank_root / relative
    out = []
    for line in (directory / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("family") != "trench":
            continue
        cell = str(row.get("primary_cell", ""))
        if include and cell not in include:
            continue
        if exclude and any(cell.startswith(p) for p in exclude):
            continue
        slot = int(row["slot_index"])
        out.append({
            "label": f"{cell}:{relative}:{slot}",
            "condition": cell, "dataset": relative,
            "map_id": str(row.get("map_id", "")),
            "images": str(directory / "images" / f"img_{slot}.npy"),
            "occupancy": str(directory / "occupancy" / f"img_{slot}.npy"),
            "metadata": str(directory / "metadata" / f"trench_{slot}.json"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-root", type=Path, required=True)
    ap.add_argument("--dataset", default="evaluation/gate_main/development")
    ap.add_argument("--exclude-prefix", nargs="*", default=["trn-net4-"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = env_config()
    cones, fp_true, fp_masked, fwd, bwd = geometry(cfg)
    tol = float(EnvConfig().trench_dig_yaw_tolerance_rad)
    so_min = float(EnvConfig().trench_dig_standoff_min_m)
    so_max = float(EnvConfig().trench_dig_standoff_max_m)

    cases = cases_from_panel(args.bank_root, args.dataset, None, args.exclude_prefix)
    print(f"{len(cases)} trench maps in {args.dataset}", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init,
                  initargs=(cfg, cones, fp_true, fp_masked, fwd, bwd, tol, so_min, so_max)) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(analyze, cases, chunksize=1)):
            results.append(res)
            if i % 10 == 0:
                print(f"{i + 1}/{len(cases)} {res['label']}", flush=True)

    by_cond = {}
    for r in results:
        e = by_cond.setdefault(r["condition"], [])
        e.append(r)
    summary = []
    for cond in sorted(by_cond):
        g = by_cond[cond]
        summary.append({
            "condition": cond, "maps": len(g),
            "target_cells": sum(r["target_cells"] for r in g),
            "membership_empty_cells": sum(r["membership_empty_cells"] for r in g),
            "fringe_fallback_cells": sum(r["fringe_fallback_cells"] for r in g),
            "max_nearest_segment_distance": max(r["max_nearest_segment_distance"] for r in g),
            "cells_reachable_but_never_inband": sum(r["cells_reachable_but_never_inband"] for r in g),
            "cells_inband_but_never_admissible": sum(r["cells_inband_but_never_admissible"] for r in g),
            "cells_reachable_but_never_admissible": sum(r["cells_reachable_but_never_admissible"] for r in g),
            "cells_admissible": sum(r["cells_admissible"] for r in g),
            "veto_rate": float(np.mean([r["veto_rate"] for r in g])),
            "veto_candidates": sum(r["veto_candidates"] for r in g),
            "veto_candidates_mixed": sum(r["veto_candidates_mixed"] for r in g),
            "inband_applicable_candidates": sum(r["inband_applicable_candidates"] for r in g),
            "stations_terra_only": sum(r["stations_terra_only"] for r in g),
            "stations_true_only": sum(r["stations_true_only"] for r in g),
            "admissible_stations": sum(r["admissible_stations"] for r in g),
            "admissible_stations_legal_true_footprint":
                sum(r["admissible_stations_legal_true_footprint"] for r in g),
            "admissible_stations_legal_terra_footprint":
                sum(r["admissible_stations_legal_terra_footprint"] for r in g),
            "mean_station_components": float(np.mean([r["station_components"] for r in g])),
            "min_largest_station_component_fraction":
                float(min(r["largest_station_component_fraction"] for r in g)),
        })

    payload = {
        "schema": "terra_trench_gate_overrestriction_audit_v1",
        "contract": {
            "bank": str(args.bank_root), "dataset": args.dataset,
            "yaw_tolerance_rad": tol, "standoff_band_m": [so_min, so_max],
            "tile_size_m": cfg.tile_size,
            "agent_cells": [int(cfg.agent.width), int(cfg.agent.height)],
            "annulus_m": [
                0.5 + cfg.tile_size * max(cfg.agent.width / 2, cfg.agent.height / 2),
                0.5 + cfg.tile_size * max(cfg.agent.width / 2, cfg.agent.height / 2)
                + cfg.agent.dig_radius_tiles * cfg.tile_size,
            ],
            "maps": len(cases),
        },
        "summary_by_condition": summary,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
