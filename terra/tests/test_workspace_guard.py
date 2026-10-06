"""Native transition contracts for the fleet-wide full-workspace guard.

Run with JAX_PLATFORMS=cpu and the isolated campaign's Terra on PYTHONPATH.
The small deterministic cases use native handlers, not a second simulator.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.actions import TrackedAction
from terra.config import RewardStage
from terra.env import TerraEnv
from terra.tests.test_relocation_reward_contract import SHAPE, _env_config, _state
from terra.workspace_guard import state_has_conflict


@pytest.fixture(scope="module")
def empty_team():
    target = np.zeros(SHAPE, np.int8)
    target[4:8, 4:8] = -1
    target[:, 54:] = 1
    cfg = _env_config((0, 2))._replace(reward_stage=RewardStage.REWARD_V2)
    with jax.disable_jit():
        return _state(target, env_cfg=cfg)


def pose(state, slot, xy, *, cabin=0):
    a = state.agent.agent_states[slot]
    return state._set_agent_state_at(slot, a._replace(
        pos_base=jnp.asarray(xy, a.pos_base.dtype),
        angle_base=jnp.zeros_like(a.angle_base),
        angle_cabin=jnp.full_like(a.angle_cabin, cabin),
        loaded=jnp.zeros_like(a.loaded),
        shovel_lifted=jnp.zeros_like(a.shovel_lifted),
    ))


def same_material_and_agents(a, b):
    np.testing.assert_array_equal(a.world.action_map.map, b.world.action_map.map)
    for left, right in zip(jax.tree_util.tree_leaves(a.agent.agent_states),
                           jax.tree_util.tree_leaves(b.agent.agent_states)):
        np.testing.assert_array_equal(left, right)
    for field in ("carry_transport_origin", "retained_work_pose", "retained_work_events", "machine_work_s"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_rejected_pickup_rolls_back_every_material_effect_and_rewards_wait(empty_team):
    with jax.disable_jit():
        before = pose(pose(empty_team, 0, [32, 47]), 1, [32, 24])
        soil = before.world.action_map.map.at[32, 40].set(3)
        before = before._replace(world=before.world._replace(
            action_map=before.world.action_map._replace(map=soil)))
        before = before._with_traversability_mask()
        assert not bool(state_has_conflict(before))
        requested = TrackedAction.new(jnp.array([7, 0], jnp.int32))
        legacy = before._replace(env_cfg=before.env_cfg._replace(workspace_guard_enabled=False))
        unguarded, _ = legacy._step_joint(requested)
        assert int(unguarded.agent.agent_states[1].loaded[0]) > 0
        guarded, terms = before._step_joint(requested)
        same_material_and_agents(before, guarded)
        assert int(guarded.env_steps) == int(before.env_steps) + 1
        np.testing.assert_array_equal(requested.action, [7, 0])
        diagnostics = TerraEnv._transition_diagnostics(before, guarded, terms)
        np.testing.assert_array_equal(diagnostics["workspace_blocked"], [False, True])
        np.testing.assert_array_equal(diagnostics["effective_actions"], [7, 7])
        assert int(diagnostics["workspace_conflicts"]) == 0
        assert not bool(diagnostics["material_or_load_changed"])
        np.testing.assert_array_equal(diagnostics["transition_pickup_units"], [0, 0, 0, 0])
        wait = TrackedAction.new(jnp.array([7, 7], jnp.int32))
        waited, wait_terms = before._step_joint(wait)
        np.testing.assert_allclose(before._get_reward(guarded, requested, terms)[0],
                                   before._get_reward(waited, wait, wait_terms)[0])


def test_first_accepted_sweep_stays_reserved_until_next_round(empty_team):
    with jax.disable_jit():
        before = pose(pose(empty_team, 0, [20, 36], cabin=10), 1, [32, 20])
        before = before._with_traversability_mask()
        assert not bool(state_has_conflict(before))
        requested = TrackedAction.new(jnp.array([5, 0], jnp.int32))
        for order in (jnp.array([0, 1], jnp.int32), jnp.array([1, 0], jnp.int32)):
            first, terms = before._step_joint(requested, order)
            assert int(first.agent.agent_states[0].angle_cabin[0]) == 11
            np.testing.assert_array_equal(first.agent.agent_states[1].pos_base, [32, 20])
            diagnostics = TerraEnv._transition_diagnostics(before, first, terms)
            np.testing.assert_array_equal(diagnostics["workspace_blocked"], [False, True])
            np.testing.assert_array_equal(diagnostics["effective_actions"], [5, 7])
            assert int(diagnostics["workspace_conflicts"]) == 0
            # At the accepted endpoint the skid path is clear; its earlier
            # rejection in order[0,1] therefore depends on the retained sweep.
            second, terms2 = first._step_joint(
                TrackedAction.new(jnp.array([7, 0], jnp.int32)))
            np.testing.assert_array_equal(second.agent.agent_states[1].pos_base, [32, 25])
            assert int(second.env_steps) == int(before.env_steps) + 2
            assert not bool(state_has_conflict(second))
            assert not np.any(TerraEnv._transition_diagnostics(first, second, terms2)["workspace_blocked"])


def test_native_env_diagnostics_match_reset_shape_and_two_excavators_are_guarded(empty_team):
    with jax.disable_jit():
        before = pose(pose(empty_team, 0, [16, 16]), 1, [46, 40])._with_traversability_mask()
        native = TerraEnv.new(64)
        assert before.env_cfg.workspace_guard_enabled
        result = native.step_no_reset(before, TrackedAction.new(jnp.array([7, 7], jnp.int32)), before.env_cfg)
        zeros = native._zero_transition_diagnostics(2)
        for key in ("workspace_blocked", "workspace_conflicts", "effective_actions"):
            assert result.info[key].shape == zeros[key].shape
            assert result.info[key].dtype == zeros[key].dtype
        assert int(result.info["workspace_conflicts"]) == 0
        assert result.info["workspace_blocked"].shape == (2,)
        assert int(result.state.env_steps) == int(before.env_steps) + 1
        # Same native movement is allowed with body-only checks but refused
        # when the two excavators' full working envelopes would overlap.
        team = pose(pose(empty_team, 0, [32, 47]), 1, [32, 24])
        second = team.agent.agent_states[1]._replace(agent_type=jnp.array([0], jnp.int8))
        team = team._set_agent_state_at(1, second)._replace(
            env_cfg=team.env_cfg._replace(agent_types=(0, 0)))
        team = team._with_traversability_mask()
        assert not bool(state_has_conflict(team))
        action = TrackedAction.new(jnp.array([7, 0], jnp.int32))
        moved, terms = team._step_joint(action)
        np.testing.assert_array_equal(moved.agent.agent_states[1].pos_base, [32, 24])
        assert bool(terms['workspace_blocked'][1])
        unguarded = team._replace(env_cfg=team.env_cfg._replace(workspace_guard_enabled=False))
        moved, _ = unguarded._step_joint(action)
        assert not np.array_equal(moved.agent.agent_states[1].pos_base, [32, 24])
