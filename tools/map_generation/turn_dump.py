"""Turn-only dumpability, as a generation-time gate (spec §8.6).

This is the v4 review panel's measurement layer
(``terra_map_distribution_review_v4/review_bank/turn_dump_scripts/turn_dump_panel.py``)
lifted out of the offline script and made importable, so the v5 generator gates
on exactly the number the panel reported instead of on a proxy. The geometry
comes from ``terra_geom`` (live: 6.375-11.375 tiles, 7x11 footprint, +-30 deg
cone, 12 cabin indices), which is constant-for-constant identical to the panel's
``geom_live`` at the live tile size.

Env rules encoded, each traced in the panel:

* a dig fires only when the cone is obstacle-free (``state.py:2290-2293``) and
  holds >= 2 dig-eligible tiles (``state.py:2143-2152``);
* a dump lands whenever ONE accepted cell (``target > 0``, not obstacle, dumpable)
  is anywhere in the cone (auto-aim, ``state.py:2437-2451``) — and there is NO
  obstacle veto on dump, so the env-faithful test is "accepted cell in the
  annulus", the cabin union being the whole annulus;
* the base centre needs the 7x11 footprint clear at some base angle, inside the
  spawn-connected free component (``utils.py:152-225`` + the border >= 8 rule).

Three station sets, same definitions as the panel:

``any``      any legal base centre (foundations: this IS the answer)
``lane``     the reserved trench corridor, at the axis-aligned base index
``axis``     on the trench line and its prolongation, tracks off the excavation
``armband``  beside ANY arm, at [6.5, 10.0] tiles perpendicular — the working
             offset §8.6 Table E establishes; new in v5, see below
``strict``   lane OR axis OR armband — the v5 headline trench number
``strict4``  lane OR axis — the v4 panel's definition, kept for comparability

**Why `armband` exists.** The v4 panel modelled the natural trench pose as "the
reserved lane, or on the trench line". On a multi-arm trench that is wrong: the
generator reserves ONE lane, alongside the spine, and a branch 10 tiles off the
spine is then only workable from its own prolongation. In reality the machine
works each arm from beside that arm, at the same boom-compliant offset the panel
measured for the spine, and §8.1's pinned semantics make non-designated ground
free to stand on. `armband` is exactly that set — a corridor per arm, at
[6.5, 10.0] tiles perpendicular, tracks off the excavation, footprint clear,
inside the spawn-connected component. It is a strict subset of `any`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon

import terra_geom as geom

MAP_SIZE = geom.MAP_SIZE
NH = geom.ANGLES_CABIN
CONES = geom.CONES
FOOTPRINTS = geom.FOOTPRINTS
ANNULUS = geom.ANNULUS
R_MIN_TILES = geom.R_MIN_TILES
R_MAX_TILES = geom.R_MAX_TILES
SPAWN_BORDER_TILES = 8

# §8.6: dig->dump separation beyond 2 * r_max cannot be closed from one station.
SINGLE_STATION_BUDGET_TILES = 2.0 * R_MAX_TILES  # 22.75
# §8.6: the lane band — turn-dump coverage >= 0.92 over [8, 11) from the spine,
# peak 0.948 at [9, 10), and >= 6.5 keeps the machine out of its own dead ring.
ARM_BAND_TILES = (6.5, 10.0)


def _corr(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    return ndi.correlate(a.astype(np.int32), k.astype(np.int32), mode="constant", cval=0)


def _dil(s: np.ndarray, k: np.ndarray) -> np.ndarray:
    return ndi.convolve(s.astype(np.int32), k.astype(np.int32), mode="constant", cval=0) > 0


def rotated_rectangle(center_yx, length, width, angle) -> np.ndarray:
    """Verbatim from the lane builder, so the modelled lane is the built lane."""
    cy, cx = center_yx
    local = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float64,
    )
    rot = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    xy = local @ rot.T
    rr, cc = polygon(xy[:, 1] + cy, xy[:, 0] + cx, shape=(MAP_SIZE, MAP_SIZE))
    out = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    out[rr, cc] = True
    return out


def heading_index_for_axis(axis_deg: float) -> int:
    return int(round(-axis_deg / 30.0)) % geom.ANGLES_BASE


def _reachable_component(free_centres: np.ndarray, spawn: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(free_centres, structure=np.ones((3, 3), np.uint8))
    if n == 0 or not spawn.any():
        return np.zeros_like(free_centres, dtype=bool)
    counts = np.bincount(labels[spawn], minlength=n + 1)
    counts[0] = 0
    if counts.max() == 0:
        return np.zeros_like(free_centres, dtype=bool)
    return labels == int(counts.argmax())


class MapGeom:
    """Everything the coverage metrics need, computed once per map."""

    def __init__(self, target: np.ndarray, occ: np.ndarray, dmp: np.ndarray) -> None:
        self.dig = target < 0
        self.declared_dump = target > 0
        # maps_buffer.py:261-263 — the accepted zone is the declared zone minus
        # obstacles; the physical dump mask also needs dumpability.
        self.dump = self.declared_dump & ~occ & dmp
        self.occ = occ

        self.fits = [(_corr(occ, f) == 0) for f in FOOTPRINTS]
        free = np.zeros_like(occ, dtype=bool)
        for f in self.fits:
            free |= f
        spawn_free = np.zeros_like(occ, dtype=bool)
        blocked_spawn = occ | ~dmp
        for f in FOOTPRINTS:
            spawn_free |= _corr(blocked_spawn, f) == 0
        yy, xx = np.indices(occ.shape)
        border = np.minimum.reduce([yy, xx, MAP_SIZE - 1 - yy, MAP_SIZE - 1 - xx])
        self.base_any = _reachable_component(free, spawn_free & (border >= SPAWN_BORDER_TILES))

        self.cone_clear = [(_corr(occ, c) == 0) for c in CONES]
        self.dig_count = [_corr(self.dig, c) for c in CONES]
        self.dump_in_annulus = _corr(self.dump, ANNULUS) > 0
        self.footprint_off_dig = [(_corr(self.dig, f) == 0) for f in FOOTPRINTS]

    # -- station sets ------------------------------------------------------
    def trench_bases(
        self,
        heading_deg: float,
        arms: list,
        half_width: float,
        lane: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        lane_ok = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
        if lane is not None and lane.any():
            k = heading_index_for_axis(heading_deg)
            lane_ok = lane & self.fits[k] & self.footprint_off_dig[k]

        axis = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
        axis_ondig = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
        armband = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
        tol = max(1.5, float(half_width) + 0.5)
        yy, xx = np.indices((MAP_SIZE, MAP_SIZE))
        for arm in arms:
            p0 = np.asarray(arm[0], dtype=float)
            p1 = np.asarray(arm[-1], dtype=float)
            seg = p1 - p0
            length = float(np.hypot(*seg))
            if length < 1e-6:
                continue
            u = seg / length
            t = (yy - p0[0]) * u[0] + (xx - p0[1]) * u[1]
            lat = np.abs((yy - p0[0]) * u[1] - (xx - p0[1]) * u[0])
            along = (t >= -R_MAX_TILES) & (t <= length + R_MAX_TILES)
            corridor = along & (lat <= tol)
            k = heading_index_for_axis(math.degrees(math.atan2(u[1], u[0])) % 360.0)
            axis |= corridor & self.fits[k] & self.footprint_off_dig[k]
            axis_ondig |= corridor & self.fits[k]
            band = along & (lat >= ARM_BAND_TILES[0]) & (lat <= ARM_BAND_TILES[1])
            armband |= band & self.fits[k] & self.footprint_off_dig[k]
        return {
            "lane": lane_ok & self.base_any,
            "axis": axis & self.base_any,
            "axis_ondig": axis_ondig & self.base_any,
            "armband": armband & self.base_any,
        }

    # -- metrics -----------------------------------------------------------
    def coverage(self, bases: np.ndarray) -> dict[str, Any]:
        n_dig = int(self.dig.sum())
        if n_dig == 0 or not bases.any():
            return {
                "cov": 0.0,
                "dig_cov": 0.0,
                "cond": float("nan"),
                "dig_bases": np.zeros_like(bases),
                "station_dump_frac": float("nan"),
            }
        can_dump = bases & self.dump_in_annulus
        served = np.zeros_like(self.dig)
        reached = np.zeros_like(self.dig)
        dig_bases = np.zeros_like(bases)
        for h in range(NH):
            can_dig = bases & self.cone_clear[h] & (self.dig_count[h] >= 2)
            dig_bases |= can_dig
            reached |= _dil(can_dig, CONES[h])
            served |= _dil(can_dig & can_dump, CONES[h])
        reached &= self.dig
        served &= self.dig
        n_reached = int(reached.sum())
        n_bases = int(dig_bases.sum())
        return {
            "cov": float(served.sum() / n_dig),
            "dig_cov": float(n_reached / n_dig),
            "cond": float(served.sum() / n_reached) if n_reached else float("nan"),
            "dig_bases": dig_bases,
            "station_dump_frac": (
                float((dig_bases & self.dump_in_annulus).sum() / n_bases)
                if n_bases
                else float("nan")
            ),
        }

    def dump_diagnosis(self, dig_bases: np.ndarray) -> dict[str, float]:
        out = {"dump_served": float("nan"), "dump_too_close": float("nan"),
               "dump_too_far": float("nan"), "near_dump_below_rmin_frac": float("nan")}
        if not self.dump.any() or not dig_bases.any():
            return out
        served = _corr(dig_bases, ANNULUS) > 0
        d_near = ndi.distance_transform_edt(~dig_bases)
        n = int(self.dump.sum())
        unreached = self.dump & ~served
        out["dump_served"] = float((served & self.dump).sum() / n)
        out["dump_too_close"] = float((unreached & (d_near < R_MIN_TILES)).sum() / n)
        out["dump_too_far"] = float((unreached & (d_near > R_MAX_TILES)).sum() / n)
        # structural "hugging bank" statistic: stations whose NEAREST designated
        # dump cell is inside the dead ring, i.e. that can only dump down-line.
        d_dump = ndi.distance_transform_edt(~self.dump)
        nb = int(dig_bases.sum())
        out["near_dump_below_rmin_frac"] = float(
            (dig_bases & (d_dump < R_MIN_TILES)).sum() / nb
        )
        return out


def measure(
    target: np.ndarray,
    occupancy: np.ndarray,
    dumpability: np.ndarray,
    trench: dict[str, Any] | None = None,
    full: bool = True,
) -> dict[str, Any]:
    """The §8.6 panel columns for one map, in the generator's metadata shape.

    ``full=False`` computes only what the gates read (the permissive set and the
    strict set). The per-subset breakdown — lane / axis / armband / relaxed and
    the v4-definition strict number — costs one extra 12-cone pass each and is
    only needed on maps that are actually kept, so the generator screens with
    ``full=False`` and re-measures once on acceptance.
    """
    g = MapGeom(target, occupancy, dumpability)
    perm = g.coverage(g.base_any)
    out: dict[str, Any] = {
        "turn_dump_cov_any": round(perm["cov"], 5),
        "dig_cov_any": round(perm["dig_cov"], 5),
        "turn_dump_station_dump_frac_any": round(perm["station_dump_frac"], 5),
        "single_station_budget_tiles": round(SINGLE_STATION_BUDGET_TILES, 4),
    }
    if trench is None:
        out["turn_dump_cov_strict"] = out["turn_dump_cov_any"]
        out["dig_cov_strict"] = out["dig_cov_any"]
        out["turn_dump_station_dump_frac"] = out["turn_dump_station_dump_frac_any"]
        if full:
            out["turn_dump_cov_strict_v4def"] = out["turn_dump_cov_any"]
        diag_bases = perm["dig_bases"]
    else:
        nat = g.trench_bases(
            trench["heading_deg"], trench["arms"], trench["half_width"], trench.get("lane")
        )
        strict_set = nat["lane"] | nat["axis"] | nat["armband"]
        strict = g.coverage(strict_set)
        lane_bases = nat["lane"]
        out.update(
            turn_dump_cov_strict=round(strict["cov"], 5),
            dig_cov_strict=round(strict["dig_cov"], 5),
            turn_dump_station_dump_frac=round(strict["station_dump_frac"], 5),
            lane_usable_frac=(
                round(float((lane_bases & perm["dig_bases"]).sum()
                            / max(1, int(lane_bases.sum()))), 5)
                if lane_bases.any()
                else 0.0
            ),
            lane_base_cells=int(lane_bases.sum()),
            axis_base_cells=int(nat["axis"].sum()),
            armband_base_cells=int(nat["armband"].sum()),
        )
        if full:
            strict4 = g.coverage(nat["lane"] | nat["axis"])
            relaxed = g.coverage(nat["lane"] | nat["axis_ondig"] | nat["armband"])
            # `dig_cov_lane` is the audit's `covA`: what fraction of the
            # excavation the RESERVED LANE can reach on its own, dig-only (no
            # dump condition). `turn_dump_cov_lane` pools dig with dumpability, so
            # a lane that cannot reach the far row of the trench is invisible in
            # it. Free: the lane coverage pass is already being made here.
            lane_cov = g.coverage(nat["lane"])
            out.update(
                turn_dump_cov_strict_v4def=round(strict4["cov"], 5),
                turn_dump_cov_relaxed=round(relaxed["cov"], 5),
                turn_dump_cov_lane=(
                    round(lane_cov["cov"], 5) if lane_bases.any() else 0.0
                ),
                dig_cov_lane=(
                    round(lane_cov["dig_cov"], 5) if lane_bases.any() else 0.0
                ),
                turn_dump_cov_axis=round(g.coverage(nat["axis"])["cov"], 5),
                turn_dump_cov_armband=round(g.coverage(nat["armband"])["cov"], 5),
            )
        diag_bases = strict["dig_bases"]
    for key, value in g.dump_diagnosis(diag_bases).items():
        out[key] = round(value, 5)
    return out


# --------------------------------------------------------------------------
# §8.5 U10 — start-side sensitivity of a scripted plan


def plan_sensitivity(
    target: np.ndarray,
    occupancy: np.ndarray,
    dumpability: np.ndarray,
    n_steps: int = 4,
) -> dict[str, Any]:
    """Does WHERE you start actually change what the plan can do?

    Two scripted plans on the same map: ``near`` digs the cells closest to the
    designated dump first, ``far`` starts at the other end. Both are simulated
    with the live envelope, and both pay the two costs a real plan pays:

    * the tracks may not sit on the excavation, so cells dug early remove the
      stations that served the cells dug late;
    * designated dump is consumed as it is filled, nearest-first, so spoil put
      down early is not available to the spoil that comes later.

    The reported cost is the fraction of dig cells that, AT THEIR TURN in the
    sequence, have no station that can both dig them and turn-dump. The delta
    ``near - far`` is what U10 asks the generator to verify: a positive delta
    means the greedy near-side start self-blocks and the correct plan starts far.
    """
    dig = target < 0
    accepted = (target > 0) & ~occupancy & dumpability
    n_dig = int(dig.sum())
    if n_dig == 0 or not accepted.any():
        return {"plan_cost_near": 1.0, "plan_cost_far": 1.0, "plan_delta": 0.0}

    cone_clear = [(_corr(occupancy, c) == 0) for c in CONES]
    yy, xx = np.indices(target.shape)
    border = np.minimum.reduce([yy, xx, MAP_SIZE - 1 - yy, MAP_SIZE - 1 - xx])
    spawn_free = np.zeros_like(occupancy, dtype=bool)
    for f in FOOTPRINTS:
        spawn_free |= _corr(occupancy | ~dumpability, f) == 0
    spawn = spawn_free & (border >= SPAWN_BORDER_TILES)

    cells = np.argwhere(dig)
    d_dump = ndi.distance_transform_edt(~accepted)
    order = np.argsort(d_dump[dig], kind="stable")
    dump_cells = np.argwhere(accepted)
    swell = max(1, int(round(int(accepted.sum()) / max(1, n_dig))))

    costs: dict[str, float] = {}
    for name, sequence in (("near", order), ("far", order[::-1])):
        excavated = np.zeros_like(dig)
        filled = np.zeros_like(dig)
        unserved = 0
        for chunk in np.array_split(sequence, n_steps):
            if len(chunk) == 0:
                continue
            chunk_mask = np.zeros_like(dig)
            chunk_mask[cells[chunk, 0], cells[chunk, 1]] = True
            remaining = dig & ~excavated
            live_dump = accepted & ~filled
            fits = np.zeros_like(dig)
            base = np.zeros_like(dig)
            for f in FOOTPRINTS:
                fits |= _corr(occupancy | excavated, f) == 0
            base = _reachable_component(fits, spawn & fits)
            dump_ok = base & (_corr(live_dump, ANNULUS) > 0)
            served = np.zeros_like(dig)
            for h in range(NH):
                can_dig = base & cone_clear[h] & (_corr(remaining, CONES[h]) >= 2)
                served |= _dil(can_dig & dump_ok, CONES[h])
            unserved += int((chunk_mask & ~served).sum())
            excavated |= chunk_mask
            # consume the nearest still-free designated dump for this chunk
            need = min(int(live_dump.sum()), swell * len(chunk))
            if need > 0:
                free_idx = np.flatnonzero(live_dump[dump_cells[:, 0], dump_cells[:, 1]])
                if len(free_idx):
                    centre = cells[chunk].mean(axis=0)
                    pts = dump_cells[free_idx]
                    dist = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
                    take = free_idx[np.argsort(dist, kind="stable")[:need]]
                    filled[dump_cells[take, 0], dump_cells[take, 1]] = True
        costs[name] = unserved / n_dig
    return {
        "plan_cost_near": round(costs["near"], 5),
        "plan_cost_far": round(costs["far"], 5),
        "plan_delta": round(costs["near"] - costs["far"], 5),
        "plan_swell_cells_per_dig": swell,
        "plan_steps": n_steps,
    }


# --------------------------------------------------------------------------
# §8.5 U8 — transport: multi-leg (dig -> stage -> re-dig) feasibility


def staging_feasibility(
    target: np.ndarray,
    occupancy: np.ndarray,
    dumpability: np.ndarray,
    max_hops: int = 6,
) -> dict[str, Any]:
    """Can every dig cell reach the designated dump through staging hops?

    One hop = one station that sees both the pick-up cell and the drop cell in
    its annulus, so soil moves at most ``2 * r_max`` per hop. The relation is
    symmetric, so the BFS runs backwards from the dump and every dig cell gets
    its hop count. Staging on non-designated ground is legal (spec §8.1: the
    dump zone is where soil must END up; other ground may be used in between),
    so the intermediate set is any dumpable, obstacle-free, non-designated cell.
    """
    dig = target < 0
    accepted = (target > 0) & ~occupancy & dumpability
    if not dig.any() or not accepted.any():
        return {"staging_max_hops": -1, "staging_infeasible_frac": 1.0}

    fits = np.zeros_like(occupancy, dtype=bool)
    spawn_free = np.zeros_like(occupancy, dtype=bool)
    for f in FOOTPRINTS:
        fits |= _corr(occupancy, f) == 0
        spawn_free |= _corr(occupancy | ~dumpability, f) == 0
    yy, xx = np.indices(target.shape)
    border = np.minimum.reduce([yy, xx, MAP_SIZE - 1 - yy, MAP_SIZE - 1 - xx])
    stations = _reachable_component(fits, spawn_free & (border >= SPAWN_BORDER_TILES))

    stageable = (~occupancy) & dumpability & ~accepted
    hops = np.full(target.shape, -1, dtype=np.int16)
    visited = accepted.copy()
    hops[accepted] = 0
    frontier = accepted.copy()
    for hop in range(1, max_hops + 1):
        st = (_corr(frontier, ANNULUS) > 0) & stations
        if not st.any():
            break
        nxt = (_dil(st, ANNULUS)) & (stageable | dig) & ~visited
        if not nxt.any():
            break
        hops[nxt] = hop
        visited |= nxt
        frontier = nxt
        if (hops[dig] >= 0).all():
            break
    dig_hops = hops[dig]
    reached = dig_hops >= 0
    return {
        "staging_max_hops": int(dig_hops[reached].max()) if reached.any() else -1,
        "staging_median_hops": float(np.median(dig_hops[reached])) if reached.any() else -1.0,
        "staging_infeasible_frac": round(float((~reached).sum() / dig_hops.size), 5),
    }
