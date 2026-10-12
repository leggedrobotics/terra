"""Precision transition invariants and conservative dead-end detection."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from terra.state import compute_swept_polygon_mask, static_rules
from terra.tests.test_machine_rules import _state, MACHINE, SHAPE


def machine(*, target=None, action=None, base=0, safety=True):
    state = _state(target, action, base=base, centre_chassis_on_base=True, **MACHINE)
    return state._replace(env_cfg=state.env_cfg._replace(
        precision_safety=safety, tracked_move_keeps_turn=True))


@partial(jax.jit, static_argnums=(1,))
def dig_probe(state, cfg):
    state = state._replace(env_cfg=cfg)
    mask, volume, relift, _ = state._dig_eligibility(state._build_dig_dump_cone())
    after = state._handle_dig()
    return (mask.reshape(SHAPE), volume, relift, after.world.action_map.map,
            after._get_current_agent_state().loaded,
            state._executable_fresh_dig_counts()[0])


def test_fresh_clearance_matches_movement_but_does_not_block_relifts():
    state = machine(target=-np.ones(SHAPE, np.int8), base=1)
    off = state.env_cfg._replace(precision_safety=False)
    old = dig_probe(state._replace(env_cfg=None), off)
    new = dig_probe(state._replace(env_cfg=None), state.env_cfg)
    protected = np.asarray(state._cell_clearance_mask(state._current_base_footprint_mask()))
    assert np.any(np.asarray(old[0]) & protected), "fixture must expose the clearance asymmetry"
    assert not np.any(np.asarray(new[0]) & protected)
    assert int(new[1]) == int(new[5]) == int(new[4][0])
    assert int(np.asarray(new[3], np.int32).sum()) + int(new[4][0]) == 0
    piles = np.asarray(old[0], np.int8)
    relift = machine(target=-np.ones(SHAPE, np.int8), action=piles, base=1)
    on_lift = dig_probe(relift._replace(env_cfg=None), relift.env_cfg)
    off_lift = dig_probe(relift._replace(env_cfg=None), off)
    assert bool(on_lift[2])
    np.testing.assert_array_equal(on_lift[0], off_lift[0])
    assert int(on_lift[5]) == 0


@partial(jax.jit, static_argnums=(1,))
def soil_probe(state, cfg):
    state = state._replace(env_cfg=cfg)
    mask = jnp.zeros(SHAPE, bool).at[32, 39].set(True)
    deposited = state._apply_dump_mask(
        state.world.action_map.map.reshape(-1), mask.reshape(-1),
        jnp.int32(20), state._accepted_dump_mask(),
    ).reshape(SHAPE)
    relifted = state._handle_dig()
    return deposited, relifted.world.action_map.map, relifted._get_current_agent_state().loaded


def test_dump_and_relift_preserve_inner_soil_and_conserve_mass():
    target = np.ones(SHAPE, np.int8)
    action = np.zeros(SHAPE, np.int8)
    action[32:35, 39] = 100
    action[32, 38] = 1  # historical inner soil must remain, never disappear
    state = machine(target=target, action=action)
    off = soil_probe(state._replace(env_cfg=None), state.env_cfg._replace(precision_safety=False))
    on = soil_probe(state._replace(env_cfg=None), state.env_cfg)
    inner = np.asarray(state._base_cell_radii()) < 4.0 - 1e-5
    assert np.any(np.asarray(off[0])[inner] != action[inner])
    assert np.any(np.asarray(off[1])[inner] != action[inner])
    for new_map in on[:2]:
        np.testing.assert_array_equal(np.asarray(new_map)[inner], action[inner])
    assert np.asarray(on[0], np.int32).sum() == action.astype(np.int32).sum() + 20
    assert np.asarray(on[1], np.int32).sum() + int(on[2][0]) == action.astype(np.int32).sum()
    assert int(on[2][0]) > 0


@partial(jax.jit, static_argnums=(1,))
def dead_end(state, cfg):
    return state._replace(env_cfg=cfg)._unloaded_dead_end()


def test_dead_end_keeps_recoverable_target_and_ignores_history_exclusion():
    empty = machine()
    footprint = np.asarray(empty._current_base_footprint_mask())
    radius = np.asarray(empty._base_cell_radii())
    holes = np.where(~footprint & (radius < 4.0), -1, 0).astype(np.int8)
    trapped = machine(action=holes)
    assert bool(dead_end(trapped._replace(env_cfg=None), trapped.env_cfg))
    target = np.zeros(SHAPE, np.int8)
    target[32, 40] = -1
    holes[32, 40] = 1  # a relift can uncover this fresh target
    recoverable = machine(target=target, action=holes)
    recoverable = recoverable._replace(world=recoverable.world._replace(
        last_dig_mask=recoverable.world.last_dig_mask._replace(map=jnp.ones(SHAPE, bool))))
    assert not bool(dead_end(recoverable._replace(env_cfg=None), recoverable.env_cfg))
    # With no fresh targets left, misplaced soil can still be legally relocated.
    repair = machine(action=holes)
    assert not bool(dead_end(repair._replace(env_cfg=None), repair.env_cfg))
    assert not bool(dead_end(empty._replace(env_cfg=None), empty.env_cfg))
    cur = trapped._get_current_agent_state()
    loaded = trapped._set_current_agent_state(cur._replace(loaded=jnp.ones_like(cur.loaded)))
    assert not bool(dead_end(loaded._replace(env_cfg=None), loaded.env_cfg))


@partial(jax.jit, static_argnums=(1,))
def move_probe(state, cfg):
    state = state._replace(env_cfg=cfg)
    cur = state._get_current_agent_state()
    orientation = state._base_orientation_to_one_hot_forward(cur.angle_base)
    return tuple(state._move_on_orientation(orientation, minimum_move_tiles=m)
                 ._get_current_agent_state().pos_base for m in (1, 2))


def test_oblique_minimum_blocks_short_only_exit_and_keeps_long_exit():
    state = machine(base=1)
    cur = state._get_current_agent_state()
    original = state._get_agent_corners(cur.pos_base, cur.angle_base,
                                       state.env_cfg.agent.width, state.env_cfg.agent.height)
    one_cfg = state.env_cfg._replace(agent=state.env_cfg.agent._replace(move_tiles=1))
    one = move_probe(state._replace(env_cfg=None), one_cfg)[0]
    long = move_probe(state._replace(env_cfg=None), state.env_cfg)[0]
    for endpoint, can_escape in ((one, False), (long, True)):
        allowed = compute_swept_polygon_mask(original, endpoint - cur.pos_base, *SHAPE)
        corridor = state._replace(world=state.world._replace(
            static_traversability_base=state.world.static_traversability_base._replace(
                map=(~allowed).astype(jnp.int8))))
        legacy, corrected = move_probe(corridor._replace(env_cfg=None), corridor.env_cfg)
        if can_escape:
            assert np.sum((np.asarray(corrected) - np.asarray(cur.pos_base)) ** 2) >= 2.25
        else:
            np.testing.assert_array_equal(legacy, one)
            np.testing.assert_array_equal(corrected, cur.pos_base)
        # Production uses static rule switches. The certificate must still
        # disable turn preference to check every permitted endpoint.
        with static_rules(tracked_move_keeps_turn=True):
            assert bool(dead_end(corridor._replace(env_cfg=None), corridor.env_cfg)) != can_escape
