"""Native requested-action loading lifecycle and strict pair-boundary contracts."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.actions import TrackedAction
from terra.env import TerraEnv
from terra.tests.test_relocation_reward_contract import SHAPE, _env_config, _state
from terra.workspace_interactions import (
    candidate_component_exceptions, configured_loading_pairs, loading_pair_mask,
    stationary_component_exceptions, validate_loading_pair_mask, verified_full_transfer,
)


@pytest.fixture(scope="module")
def team():
    target = np.zeros(SHAPE, np.int8)
    target[4:8, 4:8] = -1
    target[:, 54:] = 1
    cfg = _env_config((0, 1, 2))._replace(workspace_guard_enabled=False,
        workspace_loading_pairs=loading_pair_mask((0, 1, 2), [(0, 1)]))
    with jax.disable_jit():
        state = _state(target, env_cfg=cfg)
        for slot, (xy, base, load, credit) in enumerate((
            ([24, 24], 0, 8, 3.25), ([19, 36], 3, 2, 1.), ([50, 45], 0, 0, 0.),
        )):
            a = state.agent.agent_states[slot]
            state = state._set_agent_state_at(slot, a._replace(
                pos_base=jnp.asarray(xy, a.pos_base.dtype),
                angle_base=jnp.full_like(a.angle_base, base),
                angle_cabin=jnp.zeros_like(a.angle_cabin),
                loaded=jnp.full_like(a.loaded, load),
                carry_relocation_credit=jnp.float32(credit),
            ))
        return state._replace(env_cfg=cfg._replace(workspace_guard_enabled=True))._with_traversability_mask()


def step(state, actions, order):
    requested = TrackedAction.new(jnp.asarray(actions, jnp.int32))
    following, terms = state._step_joint(requested, jnp.asarray(order, jnp.int32))
    np.testing.assert_array_equal(requested.action, actions)
    info = TerraEnv._transition_diagnostics(state, following, terms)
    assert int(following.env_steps) == int(state.env_steps) + 1
    return following, info


def same_physics(before, after):
    for a, b in zip(jax.tree_util.tree_leaves(before.agent.agent_states),
                    jax.tree_util.tree_leaves(after.agent.agent_states)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(before.world.action_map.map, after.world.action_map.map)
    for name in ("carry_transport_origin", "retained_work_pose", "retained_work_events", "machine_work_s"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))


def test_explicit_registration_and_directional_component_boundaries(team):
    assert loading_pair_mask((0, 1), [(0, 1)]) == 2
    assert validate_loading_pair_mask((0, 1), 2) == 2
    assert loading_pair_mask((0, 1, 0, 1), [(0, 1), (2, 3)]) == 2 + (1 << 11)
    for types, pairs in (((0, 0), [(0, 1)]), ((0, 2), [(0, 1)]),
                         ((0, 1), [(1, 0)]), ((0, 1), [(0, 0)]),
                         ((0, 1), [(0, 2)]), ((0, 1, 1), [(0, 1), (0, 2)])):
        with pytest.raises(ValueError):
            loading_pair_mask(types, pairs)
    for value in (-1, 65536, 1 << 2, 1.5, True):
        with pytest.raises(ValueError):
            validate_loading_pair_mask((0, 1), value)
    matrix = np.asarray(stationary_component_exceptions(team))
    np.testing.assert_array_equal(matrix[0, 1], [[False, False], [True, True]])
    np.testing.assert_array_equal(matrix[1, 0], matrix[0, 1].T)
    assert not matrix[2].any() and not matrix[:, 2].any()
    assert not matrix[:, :, 0, 0].any()
    none = team._replace(env_cfg=team.env_cfg._replace(workspace_loading_pairs=0))
    assert not np.any(stationary_component_exceptions(none))
    for invalid_mask in (2 | (1 << 2), 2.5, 2.0, True):
        malformed = team._replace(env_cfg=team.env_cfg._replace(workspace_loading_pairs=invalid_mask))
        assert not np.any(configured_loading_pairs(malformed))


def test_native_approach_hold_transfer_and_loaded_departure_in_both_orders(team):
    with jax.disable_jit():
        for order in ([0, 1, 2], [1, 0, 2]):
            arrived, arrival = step(team, [7, 1, 7], order)
            np.testing.assert_array_equal(arrived.agent.agent_states[1].pos_base, [24, 36])
            assert not np.any(arrival["workspace_blocked"])
            assert int(arrival["workspace_conflicts"]) == 0
            held, _ = step(arrived, [7, 7, 7], order)
            same_physics(arrived, held)
            swung, swing_info = step(held, [4, 7, 7], order)
            assert not np.any(swing_info["workspace_blocked"])
            assert int(swung.agent.agent_states[0].angle_cabin[0]) == 11
            restored, _ = step(swung, [5, 7, 7], order)
            same_physics(held, restored)
            transferred, loading = step(held, [6, 7, 7], order)
            assert bool(verified_full_transfer(held, transferred, 0, 1))
            assert [int(a.loaded[0]) for a in transferred.agent.agent_states[:2]] == [0, 10]
            assert not np.any(loading["workspace_blocked"])
            assert int(loading["workspace_conflicts"]) == 0
            departed, leaving = step(transferred, [7, 0, 7], order)
            np.testing.assert_array_equal(departed.agent.agent_states[1].pos_base, [19, 36])
            assert int(departed.agent.agent_states[1].loaded[0]) == 10
            assert not np.any(leaving["workspace_blocked"])
            assert int(leaving["workspace_conflicts"]) == 0
            assert int(departed.env_steps) == int(team.env_steps) + 4
            wheeled = transferred._replace(env_cfg=transferred.env_cfg._replace(action_types=(0, 1, 0)))
            truck = wheeled.agent.agent_states[1]
            wheeled = wheeled._set_agent_state_at(1, truck._replace(
                action_type=jnp.ones_like(truck.action_type), wheel_angle=jnp.zeros_like(truck.wheel_angle)))
            wheeled_departed, wheel_info = step(wheeled, [7, 0, 7], order)
            assert not np.array_equal(wheeled_departed.agent.agent_states[1].pos_base, truck.pos_base)
            assert int(wheeled_departed.agent.agent_states[1].loaded[0]) == 10
            assert not np.any(wheel_info["workspace_blocked"])
            assert int(wheel_info["workspace_conflicts"]) == 0


def test_incompatible_requests_and_failed_loading_never_open_a_motion_window(team):
    with jax.disable_jit():
        arrived, _ = step(team, [7, 1, 7], [0, 1, 2])
        for actions in ([6, 0, 7], [4, 0, 7]):
            for order in ([0, 1, 2], [1, 0, 2]):
                rejected, info = step(arrived, actions, order)
                same_physics(arrived, rejected)
                np.testing.assert_array_equal(info["workspace_blocked"], [True, True, False])
                np.testing.assert_array_equal(info["effective_actions"], [7, 7, 7])
                assert int(info["workspace_conflicts"]) == 0
        full = arrived._set_agent_state_at(1, arrived.agent.agent_states[1]._replace(
            loaded=jnp.full_like(arrived.agent.agent_states[1].loaded, arrived.env_cfg.truck_capacity)))
        rejected, info = step(full, [6, 7, 7], [0, 1, 2])
        same_physics(full, rejected)
        assert bool(info["workspace_blocked"][0])
        assert not bool(verified_full_transfer(full, rejected, 0, 1))
        # Even a manually supplied wrong receiver or soil-edit candidate cannot
        # prove the transfer-only exception.
        altered = arrived._replace(world=arrived.world._replace(
            action_map=arrived.world.action_map._replace(map=arrived.world.action_map.map.at[0, 0].set(1))))
        assert not np.any(candidate_component_exceptions(arrived, altered, [6, 7, 7], 0))
        third = team.agent.agent_states[2]
        approaching = team._set_agent_state_at(2, third._replace(
            pos_base=jnp.asarray([24, 54], third.pos_base.dtype),
            angle_base=jnp.full_like(third.angle_base, 6)))._with_traversability_mask()
        stopped, third_info = step(approaching, [7, 7, 0], [0, 1, 2])
        same_physics(approaching, stopped)
        np.testing.assert_array_equal(third_info["workspace_blocked"], [False, False, True])
        assert int(third_info["workspace_conflicts"]) == 0


def test_two_excavators_have_no_loading_exception_and_keep_default_separation(team):
    with jax.disable_jit():
        before = team._replace(env_cfg=team.env_cfg._replace(
            agent_types=(0, 0, 2), workspace_loading_pairs=0))
        for slot, position in enumerate(([32, 47], [32, 24])):
            a = before.agent.agent_states[slot]
            before = before._set_agent_state_at(slot, a._replace(
                pos_base=jnp.asarray(position, a.pos_base.dtype),
                angle_base=jnp.zeros_like(a.angle_base), angle_cabin=jnp.zeros_like(a.angle_cabin),
                agent_type=jnp.zeros_like(a.agent_type), loaded=jnp.zeros_like(a.loaded),
                carry_relocation_credit=jnp.float32(0.)))
        before = before._with_traversability_mask()
        for order in ([0, 1, 2], [1, 0, 2]):
            after, info = step(before, [7, 0, 7], order)
            same_physics(before, after)
            np.testing.assert_array_equal(info["workspace_blocked"], [False, True, False])
            assert int(info["workspace_conflicts"]) == 0


def test_first_compiled_joint_step_can_import_interactions_without_leaking_tracers():
    # This file imports the module during collection. A fresh process is needed
    # to exercise its actual first import inside State's compiled action path.
    code = textwrap.dedent("""
        import sys
        import jax
        import jax.numpy as jnp
        import numpy as np
        from terra.actions import TrackedAction
        from terra.tests.test_relocation_reward_contract import SHAPE, _env_config, _state
        with jax.disable_jit():
            state = _state(np.zeros(SHAPE, np.int8), env_cfg=_env_config((0, 1))._replace(
                workspace_guard_enabled=False, workspace_loading_pairs=2))
        assert 'terra.workspace_interactions' not in sys.modules
        step = jax.jit(lambda s: s._step(TrackedAction.new(jnp.full((2,), 7, jnp.int32))))
        result = jax.block_until_ready(step(state))
        assert int(result.env_steps) == 1
        from terra.workspace_interactions import configured_loading_pairs
        registered = np.asarray(jax.jit(configured_loading_pairs)(result))
        assert registered[0, 1] and registered.sum() == 1
        assert int(jax.block_until_ready(step(result)).env_steps) == 2
    """)
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                               env=env, text=True, capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_public_prepared_reset_rejects_overlap_but_allows_registered_loading_hold(team):
    with jax.disable_jit():
        docked, _ = step(team, [7, 1, 7], [0, 1, 2])
    env = TerraEnv.new(SHAPE[0])
    layers = (
        docked.world.target_map.map, docked.world.padding_mask.map,
        jnp.full((3, 3), -97., jnp.float32), jnp.int32(-1),
        jnp.full((SHAPE[0], 3), -97., jnp.float32), jnp.int32(-1),
        jnp.ones(SHAPE, jnp.bool_), docked.world.action_map.map,
        jnp.ones(SHAPE, jnp.float32),
    )

    def reset(cfg, agent):
        return jax.block_until_ready(env.reset(jax.random.PRNGKey(17), *layers, cfg,
                                              initial_agent=agent))

    held = reset(docked.env_cfg, docked.agent)
    for actual, expected in zip(jax.tree_util.tree_leaves(held.state.agent.agent_states),
                                jax.tree_util.tree_leaves(docked.agent.agent_states), strict=True):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(Exception, match="Prepared fleet reset has unauthorized workspace overlap"):
        reset(docked.env_cfg._replace(workspace_loading_pairs=0), docked.agent)
    third = docked.agent.agent_states[2]._replace(pos_base=docked.agent.agent_states[0].pos_base)
    third_overlap = docked._set_agent_state_at(2, third)
    with pytest.raises(Exception, match="Prepared fleet reset has unauthorized workspace overlap"):
        reset(third_overlap.env_cfg, third_overlap.agent)
    # Explicit legacy opt-out preserves the supplied physical reset unchanged.
    legacy = reset(docked.env_cfg._replace(workspace_guard_enabled=False, workspace_loading_pairs=0),
                   docked.agent)
    np.testing.assert_array_equal(legacy.state.agent.agent_states[1].pos_base,
                                  docked.agent.agent_states[1].pos_base)
