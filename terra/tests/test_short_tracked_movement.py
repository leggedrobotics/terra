"""Straight swept paths and endpoint contracts for tracked maneuvers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.tests.test_foundation_behavior import foundation, _map, _pose


@jax.jit
def _forward(state):
    return state._handle_move_forward()


@jax.jit
def _backward(state):
    return state._handle_move_backward()


def _position(state):
    return np.asarray(state._get_current_agent_state().pos_base)


def _nominal_position(state, *, forward=True):
    """The old open-ground endpoint expression, including float32 rounding."""
    agent = state._get_current_agent_state()
    index = int(agent.angle_base[0])
    if not forward:
        index = (index + state.env_cfg.agent.angles_base // 2) % state.env_cfg.agent.angles_base
    angles = jnp.linspace(0, 2 * jnp.pi, 12, endpoint=False)
    angles = (angles + jnp.pi / 2) % (2 * jnp.pi)
    deltas = state.env_cfg.agent.move_tiles * jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
    return np.asarray(jnp.round(agent.pos_base + deltas[index]).astype(jnp.int16))


def _new_front_cell(state, distance):
    """A cell entered at this distance but absent from every earlier footprint."""
    covered = np.asarray(state._current_base_footprint_mask()).astype(bool)
    pos = _position(state)
    for step in range(1, distance + 1):
        moved = _pose(state, position=pos + (0, step))
        footprint = np.asarray(moved._current_base_footprint_mask()).astype(bool)
        entered = footprint & ~covered
        covered |= footprint
    assert entered.any()
    return tuple(np.argwhere(entered)[len(np.argwhere(entered)) // 2])


def test_open_ground_preserves_nominal_endpoints_and_loaded_rejection(foundation):
    for heading in range(12):
        before = _pose(foundation, base=heading)
        for handler, forward in ((_forward, True), (_backward, False)):
            after = handler(before)
            np.testing.assert_array_equal(_position(after), _nominal_position(before, forward=forward))
            np.testing.assert_array_equal(after.world.action_map.map, before.world.action_map.map)
            loaded = _pose(before, loaded=1)
            np.testing.assert_array_equal(_position(handler(loaded)), _position(loaded))


@pytest.mark.parametrize("blocker", ["soil", "hole", "static"])
def test_blocked_nominal_move_keeps_only_the_valid_prefix(foundation, blocker):
    before = _pose(foundation, base=0)
    cells = np.zeros((64, 64), dtype=np.int8)
    cells[_new_front_cell(before, 3)] = -1 if blocker == "hole" else 1
    field = "static_traversability_base" if blocker == "static" else "action_map"
    before = _map(before, field, cells)
    after = _forward(before)
    np.testing.assert_array_equal(_position(after), _position(before) + (0, 2))
    assert not np.any(np.asarray(after._current_base_footprint_mask()) & (cells != 0))
    np.testing.assert_array_equal(after.world.action_map.map, before.world.action_map.map)
    assert bool(before._movement_feasibility_tracked()[0])

    # A blocker at the first candidate leaves no legal prefix.
    cells[_new_front_cell(foundation, 1)] = 1
    blocked = _map(before, field, cells)
    np.testing.assert_array_equal(_position(_forward(blocked)), _position(blocked))
    assert not bool(blocked._movement_feasibility_tracked()[0])


def test_clear_endpoint_cannot_jump_a_blocked_intermediate_footprint(foundation):
    # A longer configured nominal distance makes the gap larger than the real
    # 7x11 footprint. Keep the footprint; do not shrink it to manufacture a gap.
    before = foundation._replace(env_cfg=foundation.env_cfg._replace(
        agent=foundation.env_cfg.agent._replace(move_tiles=15),
    ))
    cells = np.zeros((64, 64), dtype=np.int8)
    cells[_new_front_cell(before, 3)] = -1
    before = _map(before, "action_map", cells)
    endpoint = _pose(before, position=_nominal_position(before))
    current = endpoint._get_current_agent_state()
    corners = endpoint._get_agent_corners(
        current.pos_base, current.angle_base,
        endpoint.env_cfg.agent.width, endpoint.env_cfg.agent.height,
    )
    assert bool(endpoint._is_valid_move(corners))
    after = _forward(before)
    np.testing.assert_array_equal(_position(after), _position(before) + (0, 2))


def test_shortening_does_not_change_straight_wheeled_action_model(foundation):
    before = foundation
    cells = np.zeros((64, 64), dtype=np.int8)
    cells[_new_front_cell(before, 3)] = 1
    before = _map(before, "action_map", cells)
    agent = before._get_current_agent_state()._replace(action_type=jnp.array([1], dtype=jnp.int8))
    before = before._set_current_agent_state(agent)
    after = jax.jit(lambda state: state._handle_move_forward_wheeled())(before)
    np.testing.assert_array_equal(_position(after), _position(before))


@pytest.mark.parametrize("position,heading,forward,cell,height", [
    ((38, 42), 11, True, (32, 40), 3),
    ((34, 30), 11, False, (26, 24), -1),
    ((38, 33), 1, False, (36, 27), -1),
    ((42, 32), 11, True, (36, 30), 1),
    ((30, 36), 11, True, (24, 34), 2),
    ((30, 14), 11, False, (36, 16), 1),
    ((42, 40), 11, True, (36, 38), 2),
])
def test_straight_path_does_not_visit_rounded_intermediate_poses(
    foundation, position, heading, forward, cell, height,
):
    # Minimal reproductions of the seven first movement divergences in the
    # frozen foundation-policy replay. These cells are outside the straight
    # sweep but inside an independently rounded intermediate footprint.
    before = _pose(foundation, position=position, base=heading)
    cells = np.zeros((64, 64), dtype=np.int8)
    cells[cell] = height
    before = _map(before, "action_map", cells)
    after = (_forward if forward else _backward)(before)
    np.testing.assert_array_equal(_position(after), _nominal_position(before, forward=forward))
    assert not np.any(np.asarray(after._current_base_footprint_mask()) & (cells != 0))
    np.testing.assert_array_equal(after.world.action_map.map, before.world.action_map.map)


def test_swept_cells_match_independent_convex_hull(foundation):
    from scipy.spatial import ConvexHull
    from terra.utils import compute_polygon_mask, compute_swept_polygon_mask

    sweep = jax.jit(lambda corners, delta: compute_swept_polygon_mask(corners, delta, 64, 64))
    for heading in range(12):
        state = _pose(foundation, base=heading)
        corners = np.asarray(state._get_agent_corners(
            _position(state), state._get_current_agent_state().angle_base, 7, 11,
        ))
        for delta in ((0, 0), (2, 4), (-2, -4), (0, 15)):
            points = np.concatenate((corners, corners + delta))
            hull = points[ConvexHull(points).vertices]
            expected = np.asarray(compute_polygon_mask(jnp.asarray(hull), 64, 64))
            for polygon in (corners, corners[::-1].copy()):
                np.testing.assert_array_equal(sweep(jnp.asarray(polygon), jnp.asarray(delta)), expected)
