#!/usr/bin/env python3
"""How much of the 3.5-7.0 m standoff band is actually usable, given the cone?

For a sample of panel maps, enumerate every (base pose, cabin) whose cone holds
fresh trench soil and no obstacle, and record the standoff to the owning axis
for the candidates the gate ADMITS.  Also record, per target cell, the min and
max admitting standoff, so the effective sub-band can be reported.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818")
sys.path.insert(0, "/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818/tools")
from audit_trench_gate_overrestriction import (  # noqa: E402
    env_config, geometry, records_from_metadata, cases_from_panel, dilate,
)
from terra.map import compute_trench_axis_membership  # noqa: E402

SHAPE = (64, 64)
NH = 12
BANK = Path("/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819")
TOL, SMIN, SMAX = 0.2619, 3.5, 7.0

cfg = env_config()
cones, fp_true, fp_masked, fwd, bwd = geometry(cfg)
tile = cfg.tile_size
cases = cases_from_panel(BANK, "evaluation/gate_main/development", None, ["trn-net4-"])
by_cond = defaultdict(list)
for c in cases:
    by_cond[c["condition"]].append(c)
sample = [c for cond in sorted(by_cond) for c in by_cond[cond][:3]]
print(f"sampling {len(sample)} maps across {len(by_cond)} conditions")

adm_standoffs = []
cell_min, cell_max = [], []
for case in sample:
    target = np.load(case["images"]).astype(np.int32)
    padding = np.load(case["occupancy"]).astype(bool)
    md = json.loads(Path(case["metadata"]).read_text())
    records, naxes, half = records_from_metadata(md)
    membership = np.asarray(compute_trench_axis_membership(
        jnp.asarray(target.astype(np.int8)), jnp.asarray(records), jnp.int32(naxes)
    )).astype(np.uint8)
    dig = target < 0
    axes3 = records[:naxes, :3].astype(np.float64)
    den = np.maximum(np.linalg.norm(axes3[:, :2], axis=1), 1e-6)
    rows, cols = np.meshgrid(np.arange(64.0), np.arange(64.0), indexing="ij")
    standoff = np.stack([
        np.abs(axes3[a, 0] * cols + axes3[a, 1] * rows + axes3[a, 2]) / den[a] * tile
        for a in range(naxes)])
    band = (standoff >= SMIN) & (standoff <= SMAX)
    tg = np.stack([-axes3[:, 0], axes3[:, 1]], axis=1)
    tn = np.maximum(np.linalg.norm(tg, axis=1), 1e-6)
    yaw_ok = np.zeros((NH, naxes), bool)
    for bh in range(NH):
        th = 2 * np.pi * bh / NH
        f = np.array([-np.sin(th), np.cos(th)])
        yaw_ok[bh] = np.arccos(np.clip(np.abs(tg @ f) / tn, 0, 1)) <= TOL
    near = dilate(dig, 12)
    cmin = np.full(SHAPE, np.inf)
    cmax = np.full(SHAPE, -np.inf)
    for a in range(naxes):
        own = (membership & np.uint8(1 << a)) != 0
        for bh in range(NH):
            if not yaw_ok[bh, a]:
                continue
            pm = band[a] & near & ~padding
            poses = np.argwhere(pm)
            if poses.size == 0:
                continue
            so = standoff[a][pm]
            for cb in range(NH):
                offs = cones[bh][cb]
                rr = poses[:, 0:1] + offs[None, :, 0]
                cc = poses[:, 1:2] + offs[None, :, 1]
                ok = (rr >= 0) & (rr < 64) & (cc >= 0) & (cc < 64)
                rrc, ccc = np.clip(rr, 0, 63), np.clip(cc, 0, 63)
                freshm = dig[rrc, ccc] & ok
                nfresh = freshm.sum(1)
                npad = (padding[rrc, ccc] & ok).sum(1)
                mem = membership[rrc, ccc]
                bad = (freshm & (mem != 0) & ((mem & np.uint8(1 << a)) == 0)).sum(1)
                good = (nfresh > 0) & (npad == 0) & (bad == 0)
                for i in np.flatnonzero(good):
                    adm_standoffs.append(so[i])
                    sel = freshm[i]
                    r2, c2 = rrc[i][sel], ccc[i][sel]
                    m = own[r2, c2]
                    if m.any():
                        cmin[r2[m], c2[m]] = np.minimum(cmin[r2[m], c2[m]], so[i])
                        cmax[r2[m], c2[m]] = np.maximum(cmax[r2[m], c2[m]], so[i])
    got = np.isfinite(cmin) & dig
    cell_min.append(cmin[got])
    cell_max.append(cmax[got])
    if not got.sum() == dig.sum():
        print("  WARNING", case["label"], "cells with no admitting standoff:",
              int(dig.sum() - got.sum()))

a = np.asarray(adm_standoffs)
print(f"\nadmitted candidates: n={a.size}")
for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
    print(f"  standoff p{q:3d} = {np.percentile(a, q):.3f} m")
cm = np.concatenate(cell_min)
cM = np.concatenate(cell_max)
print(f"\nper target cell (n={cm.size}): the standoff window that admits it")
print(f"  min admitting standoff: p0={cm.min():.3f} median={np.median(cm):.3f} "
      f"p99={np.percentile(cm,99):.3f} max={cm.max():.3f}")
print(f"  max admitting standoff: p0={cM.min():.3f} median={np.median(cM):.3f} "
      f"p99={np.percentile(cM,99):.3f} max={cM.max():.3f}")
print(f"  cells whose ONLY admitting standoffs are below 4.0 m: {int((cM < 4.0).sum())}")
print(f"  cells whose ONLY admitting standoffs are above 6.0 m: {int((cm > 6.0).sum())}")
print(f"  width of the admitting window, median: {np.median(cM - cm):.3f} m, "
      f"min {np.min(cM - cm):.3f} m")

# analytic reference
maxdim = max(cfg.agent.width / 2, cfg.agent.height / 2)
rmin = 0.5 + tile * maxdim
rmax = rmin + cfg.agent.dig_radius_tiles * tile
pmax = 1.5 * tile  # max measured target-cell offset from its section, in metres
print(f"\nannulus [{rmin:.4f}, {rmax:.4f}] m, wedge +/-30 deg, "
      f"max cell offset from section {pmax:.3f} m")
print(f"  abeam full-width band: [{rmin + pmax:.3f}, {rmax - pmax:.3f}] m "
      f"= {rmax - pmax - rmin - pmax:.3f} m wide, inside a nominal "
      f"{SMAX - SMIN:.1f} m band")
print(f"  any-cell-abeam band:   [{rmin - pmax:.3f}, {rmax + pmax:.3f}] m")
