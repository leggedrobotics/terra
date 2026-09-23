"""Live Terra excavator geometry, ported to numpy for the P0 CPU panel.

Every constant is derived the same way the live runtime derives it, so the
envelope is 6.375-11.375 tiles rather than the stale 5.227-10.227 that the
frozen B0a/v3.1 static gate used (MAP_BENCHMARK_SPEC.md:1101-1108).

Sources (worktree terra_map_distribution_review_20260728):
  terra/config.py:10-11    ExcavatorDims WIDTH=6.08 HEIGHT=3.5 (metres)
  terra/config.py:25       edge_length_m = 36.5714285714 over 64 tiles
  terra/config.py:43-44    angles_base = angles_cabin = 12
  terra/config.py:58-61    move_tiles=5, dig_radius_tiles=5, dig_depth=1
  terra/env.py:590-607     tile_size and the odd-rounded 7 x 11 footprint
  terra/state.py:1377-1417 _get_dig_dump_mask_cyl  (the annulus + 60 deg cone)
  terra/state.py:1419-1464 _get_dig_dump_mask      (cartesian body exclusion)
  terra/state.py:1466-1493 _clean_excavator_workspace_inner_teeth
  terra/state.py:1734-1761 _get_map_local_and_cyl_coords (frame conventions)
  terra/utils.py:31-80     apply_rot_transl / apply_local_cartesian_to_cyl
  terra/utils.py:152-203   get_agent_corners (biased-rounding footprint)
"""

from __future__ import annotations

import math

import numpy as np

MAP_SIZE = 64
EDGE_LENGTH_M = 36.5714285714
TILE_SIZE = EDGE_LENGTH_M / MAP_SIZE  # 0.5714285714 m/tile

ANGLES_BASE = 12
ANGLES_CABIN = 12
MOVE_TILES = 5
DIG_RADIUS_TILES = 5
DIG_DEPTH = 1
DUMP_SPREAD_RADIUS = 2.0  # state.py:1551-1554, cells within 2.0 of cone centroid
DUMPABILITY_KILL_KERNEL = 5  # map.py:25-47, dig kills dumpability in a 5x5


def _agent_footprint_tiles() -> tuple[int, int]:
    """env.py:596-607 -- note the deliberate width/height swap and odd rounding."""
    w_m, h_m = 6.08, 3.5
    height = round(w_m / TILE_SIZE)
    height = height if height % 2 != 0 else height + 1
    width = round(h_m / TILE_SIZE)
    width = width if width % 2 != 0 else width + 1
    return width, height


AGENT_WIDTH, AGENT_HEIGHT = _agent_footprint_tiles()  # 7, 11 tiles
assert (AGENT_WIDTH, AGENT_HEIGHT) == (7, 11), (AGENT_WIDTH, AGENT_HEIGHT)

MAX_AGENT_DIM = max(AGENT_WIDTH / 2, AGENT_HEIGHT / 2)  # 5.5 tiles
R_MIN_M = 0.5 + TILE_SIZE * MAX_AGENT_DIM
R_MAX_M = R_MIN_M + DIG_RADIUS_TILES * TILE_SIZE
R_MIN_TILES = R_MIN_M / TILE_SIZE
R_MAX_TILES = R_MAX_M / TILE_SIZE
assert abs(R_MIN_TILES - 6.375) < 1e-9, R_MIN_TILES
assert abs(R_MAX_TILES - 11.375) < 1e-9, R_MAX_TILES

THETA_HALF = 2 * math.pi / ANGLES_CABIN  # +/- 30 deg  (state.py:1407-1408)

KERNEL_RADIUS = int(math.ceil(R_MAX_TILES)) + 1  # 12


def cone_kernels() -> list[np.ndarray]:
    """The 12 arm-heading dig/dump cones as local (2K+1, 2K+1) boolean masks.

    Index [di + K, dj + K] is True when the cell at offset (di, dj) from the
    base centre is inside the workspace for that arm heading.

    The cartesian body exclusion (state.py:1442-1457) is evaluated but is
    provably inert for the excavator: the excluded box has half-extents
    floor((7*ts+ts/2)/2)=2.0 m and floor((11*ts+ts/2)/2)=3.0 m, whose corner
    sits at 3.606 m < r_min = 3.643 m.  It therefore never removes an annulus
    cell, which is why the cone depends only on the arm angle (base+cabin) and
    not separately on the base angle.
    """
    k = KERNEL_RADIUS
    di, dj = np.meshgrid(
        np.arange(-k, k + 1, dtype=np.float64),
        np.arange(-k, k + 1, dtype=np.float64),
        indexing="ij",
    )
    # global metre offsets: cell centres are (idx + 0.5) * tile_size (state.py:1191-1196)
    gx = di * TILE_SIZE
    gy = dj * TILE_SIZE

    body_x_lim = math.floor((AGENT_WIDTH * TILE_SIZE + TILE_SIZE / 2) / 2)
    body_y_lim = math.floor((AGENT_HEIGHT * TILE_SIZE + TILE_SIZE / 2) / 2)

    kernels = []
    for arm in range(ANGLES_CABIN):
        theta = 2 * math.pi * arm / ANGLES_CABIN
        c, s = math.cos(theta), math.sin(theta)
        # apply_rot_transl: local = R^T (global - t)   (utils.py:44-53)
        lx = c * gx + s * gy
        ly = -s * gx + c * gy
        r = np.hypot(lx, ly)
        th = np.arctan2(-lx, ly)  # utils.py:79
        mask = (r >= R_MIN_M) & (r <= R_MAX_M) & (th >= -THETA_HALF) & (th <= THETA_HALF)

        # cartesian body exclusion uses the BASE frame; inert here (see docstring),
        # so evaluate it in the arm frame at zero cabin offset and assert no effect.
        outside = (np.abs(gx) >= body_x_lim) | (np.abs(gy) >= body_y_lim)
        assert np.array_equal(mask, mask & outside), "body exclusion is not inert"

        # inner-tooth cleanup (state.py:1466-1493)
        inner = r < R_MIN_M + TILE_SIZE
        pad = np.pad(mask.astype(np.int32), 1)
        nb4 = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        mask = mask & ~(mask & inner & (nb4 <= 1))
        kernels.append(mask)
    return kernels


CONES = cone_kernels()
CONE_SIZES = [int(c.sum()) for c in CONES]


def dump_landing_kernels() -> list[np.ndarray]:
    """Where a full-load dump actually lands, per arm heading (clean ground).

    _apply_dump_mask (state.py:1521-1601) keeps only cells of the *filtered*
    dump cone within DUMP_SPREAD_RADIUS of that cone's centroid.  On a clean
    map (dumpability all True, no obstacles, nothing dug yet) the filtered cone
    equals the full cone, so the landing pattern is a fixed local mask.
    """
    k = KERNEL_RADIUS
    di, dj = np.meshgrid(
        np.arange(-k, k + 1, dtype=np.float64),
        np.arange(-k, k + 1, dtype=np.float64),
        indexing="ij",
    )
    out = []
    for cone in CONES:
        ci = di[cone].mean()
        cj = dj[cone].mean()
        d = np.hypot(di - ci, dj - cj)
        land = cone & (d <= DUMP_SPREAD_RADIUS)
        assert land.any()
        out.append(land)
    return out


LANDINGS = dump_landing_kernels()
LANDING_SIZES = [int(m.sum()) for m in LANDINGS]


def annulus_kernel() -> np.ndarray:
    """Union of the 12 cones = everything a fixed base pose can reach by slewing."""
    out = np.zeros_like(CONES[0])
    for c in CONES:
        out |= c
    return out


ANNULUS = annulus_kernel()


def footprint_kernels() -> list[np.ndarray]:
    """The 7 x 11 tracked footprint at the 12 base orientations.

    Follows utils.py:152-203 (biased-rounded corners) + utils.py:205-225
    (cell-centre point-in-polygon).
    """
    hw_l, hw_r = math.floor(AGENT_WIDTH / 2), math.ceil(AGENT_WIDTH / 2)
    hh_b, hh_t = math.floor(AGENT_HEIGHT / 2), math.ceil(AGENT_HEIGHT / 2)
    local = np.array(
        [[-hw_l, -hh_b], [hw_r, -hh_b], [hw_r, hh_t], [-hw_l, hh_t]], dtype=np.float64
    )
    k = KERNEL_RADIUS
    gi, gj = np.meshgrid(
        np.arange(-k, k + 1, dtype=np.float64),
        np.arange(-k, k + 1, dtype=np.float64),
        indexing="ij",
    )
    pts = np.stack([gi.ravel() + 0.5, gj.ravel() + 0.5], axis=-1)

    kernels = []
    for b in range(ANGLES_BASE):
        a = 2 * math.pi * b / ANGLES_BASE
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        corners = (rot @ local.T).T
        corners = np.where(corners < 0.0, np.floor(corners), np.ceil(corners))
        edges = np.roll(corners, -1, axis=0) - corners
        diff = pts[None, :, :] - corners[:, None, :]
        cross = edges[:, None, 0] * diff[..., 1] - edges[:, None, 1] * diff[..., 0]
        inside = np.all(cross > 0, axis=0) | np.all(cross < 0, axis=0)
        kernels.append(inside.reshape(gi.shape))
    return kernels


FOOTPRINTS = footprint_kernels()


if __name__ == "__main__":
    print(f"tile_size          {TILE_SIZE:.10f} m")
    print(f"footprint          {AGENT_WIDTH} x {AGENT_HEIGHT} tiles")
    print(f"envelope           {R_MIN_TILES:.3f} - {R_MAX_TILES:.3f} tiles")
    print(f"cone cells         {CONE_SIZES}")
    print(f"annulus cells      {int(ANNULUS.sum())}")
    print(f"landing cells      {LANDING_SIZES}")
    print(f"footprint cells    {[int(f.sum()) for f in FOOTPRINTS]}")
