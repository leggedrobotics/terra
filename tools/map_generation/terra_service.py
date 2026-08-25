"""Live-geometry service checks shared by the map generator and validator.

Two things live here, both ported from code that already exists elsewhere so the
numbers are comparable:

* ``direct_service_coverage`` — spec §8.3 U2. Identical in construction to the
  P0 reference panel's ``direct_service_coverage_initial``
  (``review_bank/p0_scripts/p0_static_panel.py``): a dig cell is *served* when
  ONE fixed base pose can legally dig it and, by cabin rotation only, complete a
  dump inside the accepted mask. The envelope is the live one
  (6.375–11.375 tiles, 7×11 footprint, ±30° cone) from ``terra_geom``, NOT the
  stale 5.227–10.227 that ``generate_prototypes.static_gate`` still uses.

* ``backward_drive_track`` — spec §8.3 U7(d). A numpy port of
  ``State._move_on_orientation`` (``terra/state.py:625-653``): the base position
  is an integer grid point, the step is ``move_tiles * [cos φ, sin φ]`` with
  ``φ = heading_index * 30° + 90°``, and the SUM is rounded (half-to-even, the
  jnp/np default). For a 15° half-bin trench, the witness may alternate the two
  equally aligned headings to stay near the trench axis.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi

import terra_geom as geom

MAP_SIZE = geom.MAP_SIZE
NH = geom.ANGLES_CABIN
CONES = geom.CONES
FOOTPRINTS = geom.FOOTPRINTS
SPAWN_BORDER_TILES = 8
R_MIN_TILES = geom.R_MIN_TILES
R_MAX_TILES = geom.R_MAX_TILES
MOVE_TILES = geom.MOVE_TILES
AGENT_WIDTH = geom.AGENT_WIDTH   # 7 tiles across
AGENT_HEIGHT = geom.AGENT_HEIGHT  # 11 tiles along


def _corr(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    return ndi.correlate(a.astype(np.int32), k.astype(np.int32), mode="constant", cval=0)


def _dilate(s: np.ndarray, k: np.ndarray) -> np.ndarray:
    return ndi.convolve(s.astype(np.int32), k.astype(np.int32), mode="constant", cval=0) > 0


def valid_base_centres(blocked: np.ndarray) -> np.ndarray:
    """Base centres where the live 7x11 footprint fits at some base angle."""
    out = np.zeros(blocked.shape, dtype=bool)
    for footprint in FOOTPRINTS:
        out |= _corr(blocked, footprint) == 0
    return out


def _reachable_component(free_centres: np.ndarray, spawn: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(free_centres, structure=np.ones((3, 3), np.uint8))
    if n == 0 or not spawn.any():
        return np.zeros_like(free_centres, dtype=bool)
    counts = np.bincount(labels[spawn], minlength=n + 1)
    counts[0] = 0
    return labels == int(counts.argmax())


def direct_service_coverage(
    target: np.ndarray, occupancy: np.ndarray, dumpability: np.ndarray
) -> float:
    """Fraction of dig cells a single station can dig AND dump from.

    Same construction as P0's ``direct_service_coverage_initial``, including the
    auto-aim rule (P0 D6): one accepted cell anywhere in the cone already gives
    a fully on-mask dump, so the dump test is ``cone ∩ accepted ≠ ∅``.
    """
    dig = target < 0
    accepted = target > 0
    if not dig.any() or not accepted.any():
        return 0.0

    free = valid_base_centres(occupancy)
    spawn = valid_base_centres(occupancy | ~dumpability)
    yy, xx = np.indices(target.shape)
    border = np.minimum.reduce([yy, xx, MAP_SIZE - 1 - yy, MAP_SIZE - 1 - xx])
    spawn = spawn & (border >= SPAWN_BORDER_TILES)
    base = _reachable_component(free, spawn)
    if not base.any():
        return 0.0

    cone_clear = [_corr(occupancy, CONES[h]) == 0 for h in range(NH)]
    dig_count = [_corr(dig, CONES[h]) for h in range(NH)]
    accepted_in_cone = [_corr(accepted, CONES[h]) > 0 for h in range(NH)]

    can_dig = [base & cone_clear[h] & (dig_count[h] > 0) for h in range(NH)]
    can_dump = [base & cone_clear[h] & accepted_in_cone[h] for h in range(NH)]

    pose_dump = np.zeros_like(base)
    for h in range(NH):
        pose_dump |= can_dump[h]

    served = np.zeros_like(base)
    for h in range(NH):
        served |= _dilate(can_dig[h] & pose_dump, CONES[h])
    return float((served & dig).sum() / max(1, int(dig.sum())))


# --------------------------------------------------------------------------
# U7 — scripted backward drive with the env's own move kinematics


def heading_indices_for_axis(axis_deg: float) -> tuple[int, ...]:
    """Base headings aligned to an axis within the exact 15 degree tolerance.

    Array index 0 is the env's `x` and index 1 its `y` — that is the convention
    `terra_geom` already uses for the cone and the footprint (`gx = di * tile`).
    The env's travel direction for base index k is
    ``[cos(k*30° + 90°), sin(k*30° + 90°)]`` in (x, y) (`state.py:626-630`), and
    the generator's trench axis h points along ``[sin(h), cos(h)]`` in the same
    index order, so ``cos(phi) = sin(h)``, ``phi = 90° - h`` and ``k = -h/30``.
    A 15° half-bin axis has two valid headings; a 30° lattice axis has one.

    Getting this backwards silently rotates the machine 90 deg against its
    travel: the footprint is 7 tiles across travel and 11 along it, and the
    transposed version has those swapped.
    """
    continuous = (-axis_deg / 30.0) % geom.ANGLES_BASE
    indices = np.arange(geom.ANGLES_BASE)
    distance = np.abs(
        (indices - continuous + geom.ANGLES_BASE / 2) % geom.ANGLES_BASE
        - geom.ANGLES_BASE / 2
    )
    aligned = tuple(int(index) for index in indices[distance <= 0.5 + 1e-9])
    if not aligned:
        raise RuntimeError(f"No base heading aligns to trench axis {axis_deg}°.")
    return aligned


def step_delta(heading_index: int) -> np.ndarray:
    """The (index0, index1) = (x, y) displacement of one move, unrounded."""
    angle = 2 * math.pi * heading_index / geom.ANGLES_BASE + math.pi / 2
    return MOVE_TILES * np.array([math.cos(angle), math.sin(angle)])


def backward_drive_track(
    start_yx: tuple[float, float],
    heading_indices: tuple[int, ...],
    steps: int,
    normal: np.ndarray,
    travel_direction: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Greedy discrete-heading BACKWARD witness using the env's rounding.

    BACKWARD is the same primitive with the heading index rolled by half a turn
    (``state.py:417-426``), and ``_move_on_orientation`` rounds ``pos + delta``
    to the integer grid every step (``state.py:634``).

    At each step the policy may use any pose-aligned heading. The witness picks
    the candidate with least centre-line error, then greatest progress. Exact
    half-cell rounding ties remain conservative: the candidate farther from
    the centre line is used before comparing headings.
    """
    position = np.array([round(start_yx[0]), round(start_yx[1])], dtype=float)
    origin = position.copy()
    track = [position.copy()]
    used_headings: list[int] = []
    for _ in range(steps):
        candidates = []
        for heading_index in heading_indices:
            backward_index = (
                heading_index + geom.ANGLES_BASE // 2
            ) % geom.ANGLES_BASE
            raw = position + step_delta(backward_index)
            candidate = np.round(raw)
            for axis in range(2):
                if abs(raw[axis] - math.floor(raw[axis]) - 0.5) < 1e-6:
                    options = [math.floor(raw[axis]), math.ceil(raw[axis])]
                    rounded_options = []
                    for option in options:
                        probe = candidate.copy()
                        probe[axis] = option
                        rounded_options.append(
                            (abs(float((probe - origin) @ normal)), option)
                        )
                    candidate[axis] = max(rounded_options)[1]
            lateral = abs(float((candidate - origin) @ normal))
            progress = float((candidate - origin) @ travel_direction)
            candidates.append((lateral, -progress, heading_index, candidate))
        _, _, heading_index, position = min(
            candidates,
            key=lambda item: item[:3],
        )
        used_headings.append(heading_index)
        track.append(position.copy())
    return np.asarray(track), tuple(used_headings)


def footprint_cells(centre_yx: np.ndarray, base_index: int) -> np.ndarray:
    """Cells covered by the 7x11 footprint at this centre and base angle."""
    kernel = FOOTPRINTS[base_index % geom.ANGLES_BASE]
    k = geom.KERNEL_RADIUS
    offsets = np.argwhere(kernel) - k
    cells = offsets + np.rint(centre_yx).astype(int)
    inside = (
        (cells[:, 0] >= 0)
        & (cells[:, 1] >= 0)
        & (cells[:, 0] < MAP_SIZE)
        & (cells[:, 1] < MAP_SIZE)
    )
    return cells[inside]


def backward_drive_check(
    dig: np.ndarray,
    blocked: np.ndarray,
    axis_deg: float,
    lane_centre_yx: tuple[float, float],
    length_tiles: float,
    lane: np.ndarray | None = None,
    lane_steps: int = 3,
) -> dict[str, Any]:
    """Drive the base backward along the trench axis and measure the drift.

    Three numbers come out, and only two of them are map-specific:

    * ``backward_drive_footprint_clear`` — did the 7x11 footprint stay off the
      excavation and the obstacles for the whole working length? Map-specific,
      and the gate that matters.
    * ``backward_drive_lane_steps`` — how many consecutive 5-tile steps the
      footprint stays inside the reserved lane. Map-specific.
    * ``backward_drive_drift_tiles`` / ``_per_tile`` — the accumulated lateral
      drift of the best pose-aligned discrete-heading witness. It is reported
      and gated only on the rate.
    """
    heading_indices = heading_indices_for_axis(axis_deg)
    steps = max(1, int(math.ceil(length_tiles / MOVE_TILES)))
    heading = math.radians(axis_deg)
    direction = np.array([math.sin(heading), math.cos(heading)])
    normal = np.array([direction[1], -direction[0]])

    # Start half the working length ahead of the centre, so the drive covers it.
    start = np.array(lane_centre_yx) + direction * (length_tiles / 2.0)
    track, used_headings = backward_drive_track(
        tuple(start),
        heading_indices,
        steps,
        normal,
        -direction,
    )

    offsets = (track - track[0]) @ normal
    clear = True
    inside = 0
    for step, heading_index in enumerate(used_headings):
        for position in track[step : step + 2]:
            cells = footprint_cells(position, heading_index)
            if len(cells) == 0 or blocked[cells[:, 0], cells[:, 1]].any():
                clear = False
                break
        if not clear:
            break
    for position, heading_index in zip(track[1:], used_headings):
        cells = footprint_cells(position, heading_index)
        if lane is None or len(cells) == 0 or not lane[cells[:, 0], cells[:, 1]].all():
            break
        inside += 1
    travelled = max(1.0, float(steps * MOVE_TILES))
    drift = float(np.abs(offsets).max())
    return {
        "backward_drive_steps": steps,
        "backward_drive_drift_tiles": round(drift, 4),
        "backward_drive_drift_per_tile": round(drift / travelled, 5),
        "backward_drive_lane_steps": inside,
        "backward_drive_lane_steps_required": lane_steps,
        "backward_drive_footprint_clear": bool(clear),
    }
