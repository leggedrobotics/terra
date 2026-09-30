"""Opt-in machine working rules: reach annulus radii and clearance from dug ground."""

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.benchmark_protocol import frozen_benchmark_protocol
from terra.env import TerraEnv
from terra.state import State
from terra.tests.test_foundation_behavior import _map, _pose
from terra.tests.test_reward_v2_contract import _env_config

SHAPE = (64, 64)
MACHINE = dict(dig_min_radius_m=4.0, dump_max_radius_m=6.0, dug_clearance_m=0.6)


def _rules(state, **rules):
    agent = state.env_cfg.agent._replace(**rules)
    return state._replace(env_cfg=state.env_cfg._replace(agent=agent))


def _state(target=None, action=None, *, position=(32, 32), base=0, cabin=0, loaded=0, **rules):
    if target is None:
        target = np.zeros(SHAPE, dtype=np.int8)
    if action is None:
        action = np.zeros(SHAPE, dtype=np.int8)
    state = State.new(
        jax.random.PRNGKey(7), _env_config(), target, np.zeros_like(target),
        -97.0 * np.ones((4, 8), dtype=np.float32), np.int32(-1),
        -97.0 * np.ones((64, 3), dtype=np.float32), np.int32(-1),
        np.ones_like(target, dtype=np.bool_), action,
        distance_map_override=np.ones_like(target, dtype=np.float32),
    )
    state = _pose(state, base=base, cabin=cabin, position=position, loaded=loaded)
    return _rules(state, **rules)


def _radius(state):
    cyl, _ = state._get_map_local_and_cyl_coords()
    return np.asarray(cyl[0]).reshape(SHAPE)


_dig_cone = jax.jit(lambda s: s._build_dig_dump_cone().reshape(SHAPE).astype(jnp.bool_))
_dump_cone = jax.jit(lambda s: s._build_dump_cone().reshape(SHAPE).astype(jnp.bool_))
_clearance = jax.jit(lambda s: s._dug_clearance_mask())
_forward = jax.jit(lambda s: s._handle_move_forward())
_backward = jax.jit(lambda s: s._handle_move_backward())
_clock = jax.jit(lambda s: s._handle_clock())
_anticlock = jax.jit(lambda s: s._handle_anticlock())
_do = jax.jit(lambda s: s._handle_do())


def _y(state):
    return int(state._get_current_agent_state().pos_base[1])


def _heading(state):
    return int(state._get_current_agent_state().angle_base[0])


def _loaded(state):
    return int(state._get_current_agent_state().loaded[0])


def test_default_rules_keep_the_release_geometry():
    frozen_benchmark_protocol()  # the new fields are inert at 0
    state = _state(base=1)  # a diagonal heading whose cone reaches inside 4 m
    r_min, r_max = state._dig_cone_radius_bounds()
    np.testing.assert_allclose([float(r_min), float(r_max)], [3.642857, 6.5], atol=1e-5)
    cone = np.asarray(_dig_cone(state))
    assert _radius(state)[cone].min() < 4.0 - 1e-3
    np.testing.assert_array_equal(np.asarray(_dump_cone(state)), cone)
    dug = np.zeros(SHAPE, dtype=np.int8)
    dug[20:25, 20:25] = -1
    assert not np.asarray(_clearance(_map(state, "action_map", dug))).any()


def test_dig_min_radius_raises_the_inner_radius_of_dig_and_dump():
    for base in range(3):
        for cabin in range(12):
            release = _state(base=base, cabin=cabin)
            machine = _rules(release, dig_min_radius_m=4.0, dump_max_radius_m=6.0)
            radius = _radius(release)
            dig_release = np.asarray(_dig_cone(release))
            dig = np.asarray(_dig_cone(machine))
            dump = np.asarray(_dump_cone(machine))
            assert dig.any() and dump.any()
            assert not (dig & ~dig_release).any()
            assert radius[dig].min() >= 4.0 - 1e-4
            assert radius[dig].max() == radius[dig_release].max()
            assert not (dump & ~dig).any()
            assert radius[dump].min() >= 4.0 - 1e-4
            assert radius[dump].max() <= 6.0 + 1e-4


def test_do_leaves_required_cells_inside_the_dig_min_radius():
    probe = _state(base=1)
    near = np.asarray(_dig_cone(probe)) & (_radius(probe) < 4.0 - 1e-4)
    assert near.any()
    target = np.where(near, -1, 0).astype(np.int8)
    release = _state(target, base=1)
    assert _loaded(_do(release)) > 0
    machine = _rules(release, dig_min_radius_m=4.0)
    refused = _do(machine)
    assert _loaded(refused) == 0
    np.testing.assert_array_equal(
        np.asarray(refused.world.action_map.map), np.asarray(machine.world.action_map.map)
    )


def test_dump_min_radius_limits_only_the_dump_cone():
    release = _state()
    rules = _rules(release, dump_min_radius_m=4.5)
    radius = _radius(release)
    dig = np.asarray(_dig_cone(release))
    np.testing.assert_array_equal(np.asarray(_dig_cone(rules)), dig)
    np.testing.assert_array_equal(np.asarray(_dump_cone(rules)), dig & (radius >= 4.5 - 1e-5))


def _square_gap_blocked(dug, clearance, tile):
    blocked = np.zeros_like(dug)
    x, y = np.indices(dug.shape)
    for i, j in np.argwhere(dug):
        gap = tile * np.hypot(np.maximum(abs(x - i) - 1, 0), np.maximum(abs(y - j) - 1, 0))
        blocked |= gap < clearance - 1e-6
    return blocked


def test_dug_clearance_blocks_cells_by_the_gap_between_cell_squares():
    base = _state()
    tile = float(base.env_cfg.tile_size)
    rng = np.random.default_rng(20260930)
    for _ in range(3):
        dug = rng.random(SHAPE) < 0.01
        state = _map(base, "action_map", np.where(dug, -1, 0).astype(np.int8))
        for clearance in (0.3, 0.57, tile, 0.6, 0.81, 1.2, 2.5):
            np.testing.assert_array_equal(
                np.asarray(_clearance(_rules(state, dug_clearance_m=clearance))),
                _square_gap_blocked(dug, clearance, tile),
            )
    one = np.zeros(SHAPE, dtype=np.int8)
    one[32, 32] = -1
    state = _map(base, "action_map", one)
    # 0.6 m exceeds one free cell (0.571 m) but not a free diagonal (0.808 m).
    assert int(np.asarray(_clearance(_rules(state, dug_clearance_m=0.6))).sum()) == 21
    assert int(np.asarray(_clearance(_rules(state, dug_clearance_m=0.57))).sum()) == 9
    pile = np.zeros(SHAPE, dtype=np.int8)
    pile[32, 32] = 3
    state = _rules(_map(base, "action_map", pile), dug_clearance_m=0.6)
    assert not np.asarray(_clearance(state)).any()


def test_clearance_shortens_a_translation_to_the_last_clear_pose():
    # A full forward move ends with one free cell (0.571 m) before the strip.
    action = np.zeros(SHAPE, dtype=np.int8)
    action[26:39, 44] = -1
    for clearance, expected in ((0.0, 37), (0.57, 37), (0.6, 36)):
        assert _y(_forward(_state(action=action, dug_clearance_m=clearance))) == expected
    # Per-environment traced values under vmap, as in the batched environment.
    state = _state(action=action)
    moved = jax.jit(jax.vmap(
        lambda c: _rules(state, dug_clearance_m=c)._handle_move_forward()
        ._get_current_agent_state().pos_base
    ))(jnp.asarray([0.0, 0.57, 0.6]))
    np.testing.assert_array_equal(np.asarray(moved)[:, 1], [37, 37, 36])


def test_clearance_refuses_turns_next_to_dug_ground_but_keeps_translations():
    # A dug strip along the chassis side, three free cells away (1.71 m).
    action = np.zeros(SHAPE, dtype=np.int8)
    action[39, 28:37] = -1
    release = _state(action=action)
    machine = _rules(release, dug_clearance_m=0.6)
    for handler in (_clock, _anticlock):
        assert _heading(handler(release)) != 0
        assert _heading(handler(machine)) == 0
    assert _y(_forward(machine)) == 37
    assert _y(_backward(machine)) == 27


def test_clearance_never_blocks_do_from_a_pose_already_too_close():
    target = np.zeros(SHAPE, dtype=np.int8)
    target[26:39, 38:46] = -1
    action = np.zeros(SHAPE, dtype=np.int8)
    action[26:39, 38] = -1  # dug ground touching the chassis front
    release = _state(target, action)
    machine = _rules(release, **MACHINE)
    footprint = np.asarray(machine._current_base_footprint_mask())
    assert (footprint & np.asarray(_clearance(machine))).any()
    dug_release, dug_machine = _do(release), _do(machine)
    assert _loaded(dug_machine) > 0
    assert _loaded(dug_machine) == _loaded(dug_release)
    np.testing.assert_array_equal(
        np.asarray(dug_machine.world.action_map.map),
        np.asarray(dug_release.world.action_map.map),
    )


def test_trench_retreat_completes_under_the_machine_rules():
    """Dig ahead, dump to the side, back away from the fresh cut: no deadlock."""
    target = np.zeros(SHAPE, dtype=np.int8)
    target[32:34, 12:57] = -1
    env = TerraEnv.new(maps_size_px=64)
    turn_and_dump = (
        [TrackedAction.cabin_clock()] * 3 + [TrackedAction.do()]
        + [TrackedAction.cabin_anticlock()] * 3
    )
    for rules in ({}, MACHINE):
        state = TerraEnv.wrap_state(_state(target, position=(32, 45), **rules))
        step = lambda s, a: env.step_no_reset(s, a, s.env_cfg).state
        for station in range(9):
            y = _y(state)
            state = step(state, TrackedAction.do())
            assert _loaded(state) > 0, (rules, station)
            for action in turn_and_dump:
                state = step(state, action)
            assert _loaded(state) == 0, (rules, station)
            if station == 0:
                # Toward the fresh cut: release closes up by one cell, the
                # clearance keeps the pose.
                probe = step(state, TrackedAction.forward())
                assert _y(probe) == (y if rules else y + 1)
            if station < 8:
                state = step(state, TrackedAction.backward())
                assert _y(state) == y - 5, (rules, station)
        dug = np.asarray(state.world.action_map.map) < 0
        np.testing.assert_array_equal(dug & (target < 0), target < 0)
