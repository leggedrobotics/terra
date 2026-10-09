"""Independent offline loose-soil forecast for converted plans.

This preserves the October 8, 2026 converter model. It does not import live
robot parameters or define field-planner behavior. Repose and bucket defaults
are explicit forecast assumptions; heights are not measurements.

The Terra-to-machine ground rules make loose soil lower than the drivable height (0.5 m) drivable, and let a
later cut remove loose soil before native ground. Without this model the converter treats every permitted dump
centre's full analytical support as an obstacle for all later stations. With it, each workspace's soil goes out
as one deposit over its dump region, and only soil at least the drivable height stays an obstacle.

The plan works at workspace level, never at single scoops (Lorenzo, 2 October 2026): a workspace's dump is one
deposit of its whole soil released over its dump region (the plan's dump mask; the part ROS prefers, release_region).
The soil fills the region to one level and runs out at the repose angle around it, stacked on the soil already there
(plateau_deposit); the runtime workspace planner picks the scoops and their centres inside the region. Ground under the soil is flat; soil does not
relax after a cut. Heights are a planning forecast, not a measurement.
"""

import math

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from .geometry import dump_footprint_offsets

REPOSE_DEG = (
    27.0  # Newton workspace-planner soil model, SOIL_MOTION_V2_REPOSE_ANGLE_RAD
)
BUCKET_M3 = (
    0.255  # stock 1.3 m shovel, description/mole_description/config/bucket_specs.yaml
)


def toe_radius(volume_m3, repose_deg=REPOSE_DEG):
    """Base radius of one cone of ``volume_m3`` on flat ground at the repose angle."""
    if volume_m3 <= 0.0:
        return 0.0
    return (3.0 * volume_m3 / (math.pi * math.tan(math.radians(repose_deg)))) ** (
        1.0 / 3.0
    )


def tall_core_radius(volume_m3, height_m, repose_deg=REPOSE_DEG):
    """Radius inside which one cone of ``volume_m3`` stands at least ``height_m`` high (0 when it never does)."""
    if volume_m3 <= 0.0:
        return 0.0
    return max(
        0.0,
        toe_radius(volume_m3, repose_deg)
        - height_m / math.tan(math.radians(repose_deg)),
    )


def release_region(mask, res):
    """The cells of a dump mask the runtime releases soil over: its centres whose whole 0.45 m pile footprint lies in
    the mask, else every centre.

    ROS distributed_local_fill ranks the centres with a full footprint first and uses the others only when none has
    one (its full-footprint fallback; workspace_planner/dump_targets.py). On a wide mask it so drops its soil on the
    mask's interior, a 0.9 m patch on a 0.45 m disk (offline runtime replay, 2 October 2026: loads 0.3-0.4 m apart).
    """
    mask = np.asarray(mask, dtype=bool)
    offsets = dump_footprint_offsets(res)
    pad = max(max(abs(dr), abs(dc)) for dr, dc in offsets)
    padded = np.pad(mask, pad)
    inside = mask.copy()
    rows, cols = mask.shape
    for dr, dc in offsets:
        inside &= padded[pad + dr : pad + dr + rows, pad + dc : pad + dc + cols]
    return inside if inside.any() else mask


def plateau_deposit(surface, region, volume_m3, res, tan_repose, known=None):
    """One workspace's dump as one deposit: ``volume_m3`` released over ``region`` (the dump region's cells) on
    ``surface`` (a height grid, cell size ``res``).

    The soil fills the region to one level and runs out from it at slope ``tan_repose`` (the distance is to the
    nearest region cell centre), only over cells connected to the region through cells that get soil: it does not
    reach across a pit beyond its toe. A single-cell region gives the cone at the repose angle. Returns (window
    slices, soil added in the window, whether soil would leave the grid or ``known`` ground). ``volume_m3`` <= 0 or an
    empty region adds nothing.
    """
    region = np.asarray(region, dtype=bool)
    height, width = surface.shape
    if volume_m3 <= 1e-12 or not region.any():
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0)), False
    rows, cols = np.nonzero(region)
    cone_height = (3.0 * volume_m3 * tan_repose**2 / math.pi) ** (1.0 / 3.0)
    margin = int(math.ceil((cone_height / tan_repose) / res)) + 3
    for _ in range(8):
        r0, r1 = rows.min() - margin, rows.max() + margin + 1
        c0, c1 = cols.min() - margin, cols.max() + margin + 1
        clipped = r0 < 0 or c0 < 0 or r1 > height or c1 > width
        r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, height), min(c1, width)
        window = (slice(r0, r1), slice(c0, c1))
        s = np.asarray(surface[window], dtype=float)
        inside = region[window]
        slope = tan_repose * ndimage.distance_transform_edt(~inside) * res
        top = float(s.max()) + cone_height + 0.1

        def solve(keep):
            lo, hi = float(s[inside].min()), top
            for _ in range(60):
                level = 0.5 * (lo + hi)
                if (
                    keep(np.maximum(level - slope - s, 0.0)).sum() * res * res
                    > volume_m3
                ):
                    hi = level
                else:
                    lo = level
            return keep(np.maximum(hi - slope - s, 0.0))

        def connected(added):
            labels, count = ndimage.label(added > 0.0, structure=np.ones((3, 3)))
            if count <= 1:
                return added
            touching = np.unique(labels[inside & (labels > 0)])
            return np.where(np.isin(labels, touching), added, 0.0)

        added = solve(lambda a: a)
        if ndimage.label(added > 0.0, structure=np.ones((3, 3)))[1] > 1:
            added = solve(connected)
        rim = max(
            added[0].max(), added[-1].max(), added[:, 0].max(), added[:, -1].max()
        )
        if rim <= 0.0 or clipped:
            break
        margin = int(math.ceil(margin * 1.6))
    total = added.sum() * res * res
    if total > 0.0:
        added *= volume_m3 / total
    off_map = bool(clipped and rim > 0.0) or (
        known is not None and bool(np.any((added > 0.0) & ~known[window]))
    )
    return window, added, off_map


class SpoilHeights:
    """Loose-soil thickness on an axis-aligned lattice (``X[r, c] = x0 + c * res``, ``Y[r, c] = y0 + r * res``)."""

    def __init__(self, X, Y, resolution, repose_deg=REPOSE_DEG, bucket_m3=BUCKET_M3):
        self.x0, self.y0, self.res = float(X[0, 0]), float(Y[0, 0]), float(resolution)
        self.height = np.zeros(X.shape)
        self.tan = math.tan(math.radians(repose_deg))
        self.bucket_m3 = bucket_m3
        self._X, self._Y = X, Y

    def copy(self):
        other = SpoilHeights.__new__(SpoilHeights)
        other.__dict__.update(self.__dict__)
        other.height = self.height.copy()
        return other

    def remove(self, mask):
        """Lift all loose soil in ``mask``; returns its volume [m3]."""
        volume = float(self.height[mask].sum()) * self.res**2
        self.height[mask] = 0.0
        return volume

    def deposit(self, centres, volume_m3):
        """Dump a workspace's ``volume_m3`` as one deposit over its dump region ``centres`` (a mask; the runtime
        releases it over release_region), stacked on the soil already there (plateau_deposit).
        """
        if volume_m3 <= 1e-9 or not np.any(centres):
            return
        region = release_region(centres, self.res)
        window, added, _ = plateau_deposit(
            self.height, region, volume_m3, self.res, self.tan
        )
        self.height[window] += added

    def tall(self, height_m):
        """Cells whose loose soil is at least ``height_m`` thick."""
        return self.height >= height_m

    def pile_radius(self, volume_m3, margin_m=0.5):
        """Per-cell toe radius of the pile a centre would carry: ``volume_m3`` on the soil already there.

        Soil within the new cone's toe plus ``margin_m`` joins the pile (a load on a flank runs down to the old
        toe), so each centre gets the toe of one cone holding both. Conservative for loads spread over centres.
        """
        own = (3.0 * max(volume_m3, 0.0) / (math.pi * self.tan)) ** (1.0 / 3.0)
        if own <= 0.0:
            return np.zeros(self.height.shape)
        if not self.height.any():
            return np.full(self.height.shape, own)
        cells = int(math.ceil((own + margin_m) / self.res))
        offsets = np.arange(-cells, cells + 1) * self.res
        disk = (np.hypot(offsets[None, :], offsets[:, None]) <= own + margin_m).astype(
            float
        )
        local = (
            np.maximum(fftconvolve(self.height, disk, mode="same"), 0.0) * self.res**2
        )
        return np.cbrt(3.0 * (volume_m3 + local) / (math.pi * self.tan))
