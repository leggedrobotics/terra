"""agent.centre_chassis_on_base: the chassis raster rotated about the base cell centre."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from terra.benchmark_protocol import frozen_benchmark_protocol
from terra.tests.test_foundation_behavior import _map
from terra.tests.test_machine_rules import SHAPE, _heading, _state
from terra.utils import compute_polygon_mask, get_agent_corners

POSITION = (32, 32)
WIDTH, HEIGHT = 7, 11  # agent.width (short side) and agent.height (long side), tiles
# The converter's machine footprint (runtime profile base_footprint_xy_json;
# metres, x forward, y left). The converter puts its origin BASE at the base
# cell's centre and turns it by yaw = heading * 30 deg + 90 deg (extract_map:
# heading 0 points along +column). Plan x is Terra's row, plan y its column.
MACHINE_FOOTPRINT_M = np.array([
    [-2.853618, -1.013765], [-2.851897, -1.638759], [1.499506, -2.035989],
    [2.729393, -2.034767], [2.781379, 1.289414], [2.781018, 1.914410],
    [1.551131, 1.913700], [-2.826729, 1.700201],
])

_footprint = jax.jit(lambda s: s._current_base_footprint_mask())
_anticlock = jax.jit(lambda s: s._handle_anticlock())


def _raster(heading, centred, position=POSITION):
    state = _state(position=position, base=heading, centre_chassis_on_base=centred)
    return np.asarray(_footprint(state))


def _tile():
    return float(_state().env_cfg.tile_size)


def _converter_frame(heading):
    """BASE (tiles) and the machine's forward and left unit vectors in the plan."""
    yaw = heading * 2 * math.pi / 12 + math.pi / 2
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    return np.asarray(POSITION, dtype=float) + 0.5, forward, left


def _point_segment_distance(point, a, b):
    t = np.clip((point - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * (b - a))))


def _chassis_gap(cell, heading):
    """Gap (tiles) between a cell square and the converter-placed 7 x 11 chassis."""
    base, forward, left = _converter_frame(heading)
    square = np.asarray(cell, dtype=float) + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    rectangle = base + np.array([
        s * HEIGHT / 2 * forward + t * WIDTH / 2 * left
        for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ])
    for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), forward, left):
        a, b = square @ axis, rectangle @ axis
        if a.max() < b.min() or b.max() < a.min():
            break  # separated: the convex polygons are apart
    else:
        return 0.0
    return min(
        _point_segment_distance(p, q, r)
        for first, second in ((square, rectangle), (rectangle, square))
        for p in first
        for q, r in zip(second, np.roll(second, -1, axis=0))
    )


def _machine_reach_outside(mask, heading):
    """Largest distance (m) from the placed machine footprint to the raster's squares."""
    tile = _tile()
    base, forward, left = _converter_frame(heading)
    polygon = base * tile + MACHINE_FOOTPRINT_M[:, :1] * forward + MACHINE_FOOTPRINT_M[:, 1:] * left
    samples = np.concatenate([
        a + np.linspace(0, 1, 400, endpoint=False)[:, None] * (b - a)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0))
    ])
    lo = np.argwhere(mask) * tile
    hi = lo + tile
    dx = np.maximum(np.maximum(lo[None, :, 0] - samples[:, None, 0], 0), samples[:, None, 0] - hi[None, :, 0])
    dy = np.maximum(np.maximum(lo[None, :, 1] - samples[:, None, 1], 0), samples[:, None, 1] - hi[None, :, 1])
    return float(np.min(np.hypot(dx, dy), axis=1).max())


def test_default_keeps_the_release_raster():
    frozen_benchmark_protocol()  # the new field is inert at False
    release = jax.jit(lambda pos, heading: compute_polygon_mask(
        get_agent_corners(pos, heading, WIDTH, HEIGHT, 12), 64, 64
    ))
    offsets = {}
    for position in (POSITION, (6, 7), (57, 40)):
        for heading in range(12):
            mask = _raster(heading, False, position)
            np.testing.assert_array_equal(mask, np.asarray(release(
                jnp.asarray(position, dtype=jnp.int16), jnp.asarray([heading], dtype=jnp.int8)
            )))
            if position == POSITION:
                offsets[heading] = tuple(np.argwhere(mask).mean(axis=0) - POSITION)
    # The release raster turns about the base cell's corner: one cell off BASE
    # at headings 3, 6 and 9.
    assert [offsets[h] for h in (0, 3, 6, 9)] == [(0, 0), (-1, 0), (-1, -1), (0, -1)]


def test_centred_raster_matches_the_converter_placement_at_every_heading():
    for heading in range(12):
        mask = _raster(heading, True)
        release = _raster(heading, False)
        cells = {tuple(c) for c in np.argwhere(mask)}
        # Exactly the cells whose centres lie inside the converter-placed 7 x 11
        # chassis (BASE at the base cell's centre, yaw heading * 30 deg + 90 deg).
        base, forward, left = _converter_frame(heading)
        inside = set()
        for i, j in np.ndindex(*SHAPE):
            offset = np.array([i + 0.5, j + 0.5]) - base
            if abs(offset @ forward) < HEIGHT / 2 and abs(offset @ left) < WIDTH / 2:
                inside.add((i, j))
        assert cells == inside, heading
        assert len(cells) == (77 if heading % 3 == 0 else 79)
        # Point-symmetric about BASE.
        assert cells == {(2 * POSITION[0] - i, 2 * POSITION[1] - j) for i, j in cells}, heading
        # The machine footprint stays within 0.21 m of the raster's squares
        # (0.036 m at the axis-aligned headings: its right side is 2.035 m from
        # BASE, the raster's 2.0 m). Released, it reaches up to 0.69 m outside,
        # and raster cells lie up to 1.15 tiles off the converter's chassis.
        assert _machine_reach_outside(mask, heading) < (0.04 if heading % 3 == 0 else 0.21), heading
        if heading in (3, 6, 9):
            assert _machine_reach_outside(release, heading) > 0.25, heading
        if heading in (4, 5, 7):
            assert max(_chassis_gap(tuple(c), heading) for c in np.argwhere(release)) > 0.9


def test_centring_is_a_traced_per_environment_value():
    state = _state(position=POSITION, base=6)
    masks = jax.jit(jax.vmap(lambda centred: state._replace(env_cfg=state.env_cfg._replace(
        agent=state.env_cfg.agent._replace(centre_chassis_on_base=centred)
    ))._current_base_footprint_mask()))(jnp.asarray([False, True]))
    np.testing.assert_array_equal(np.asarray(masks[0]), _raster(6, False))
    np.testing.assert_array_equal(np.asarray(masks[1]), _raster(6, True))
    assert not np.array_equal(np.asarray(masks[0]), np.asarray(masks[1]))


def test_turns_check_the_centred_raster():
    # Heading 6: the release raster covers rows 28-34, the centred one 29-35.
    for soil_row, release_turns, centred_turns in ((28, False, True), (35, True, False)):
        action = np.zeros(SHAPE, dtype=np.int8)
        action[soil_row, POSITION[1]] = 2  # a pile the chassis may not cover
        for centred, turns in ((False, release_turns), (True, centred_turns)):
            state = _map(_state(position=POSITION, base=5, centre_chassis_on_base=centred),
                         "action_map", action)
            assert _heading(_anticlock(state)) == (6 if turns else 5), (soil_row, centred)
