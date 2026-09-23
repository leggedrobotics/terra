"""Joint team step: every agent acts once per env step, in the given order."""

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.config import RewardStage
from terra.env import AGENT_VIEW_OBS_KEYS
from terra.env import TerraEnv
from terra.wrappers import LocalMapWrapper
from terra.tests.test_relocation_reward_contract import CENTER
from terra.tests.test_relocation_reward_contract import SHAPE
from terra.tests.test_relocation_reward_contract import _env_config
from terra.tests.test_relocation_reward_contract import _set_pose
from terra.tests.test_relocation_reward_contract import _state


def _team(positions, angles=None):
    target = np.zeros(SHAPE, dtype=np.int8)
    target[40:50, 40:50] = -1
    target[5:10, 5:10] = 1
    state = _state(target, env_cfg=_env_config((0,) * len(positions)))
    for slot, position in enumerate(positions):
        state = _set_pose(state, slot, np.asarray(position, dtype=np.int16))
    if angles is not None:
        for slot, angle in enumerate(angles):
            agent = state.agent.agent_states[slot]._replace(
                angle_base=jnp.array([angle], dtype=jnp.int8)
            )
            state = state._set_agent_state_at(slot, agent)
    return state


def _actions(*indices):
    return TrackedAction.new(jnp.asarray(indices, dtype=jnp.int8))


_step = jax.jit(lambda state, action, order: state._step(action, order))


def _positions(state):
    return [tuple(np.asarray(a.pos_base).tolist()) for a in state.agent.agent_states[:2]]


def test_execution_order_decides_who_takes_contested_space():
    """Two excavators drive head-on into the same gap: the first one wins."""
    facing = (0, 6)  # heading 0 drives along +y, heading 6 along -y
    both_forward = _actions(0, 0)
    first, second = jnp.array([0, 1]), jnp.array([1, 0])
    for gap in range(11, 30):
        other = CENTER + np.asarray((0, gap), dtype=np.int16)
        state = _team([CENTER, other], facing)
        if _positions(state)[0] == _positions(state)[1]:
            continue
        start = _positions(state)
        a_first = _positions(_step(state, both_forward, first))
        b_first = _positions(_step(state, both_forward, second))
        moved_a = [p != s for p, s in zip(a_first, start)]
        moved_b = [p != s for p, s in zip(b_first, start)]
        if moved_a == [True, False] and moved_b == [False, True]:
            return
    raise AssertionError("no contested configuration found")


def test_one_env_step_per_round_and_team_costs_paid_once():
    state = _team([CENTER, CENTER + np.array([0, 20], dtype=np.int16)])
    state = state._replace(
        env_cfg=state.env_cfg._replace(reward_stage=RewardStage.DENSE_SKILL)
    )
    idle = _actions(7, 7)
    after, terms = jax.jit(lambda s: s._step_joint(idle))(state)
    assert int(after.env_steps) == int(state.env_steps) + 1
    np.testing.assert_array_equal(np.asarray(terms["slot"]), [0, 1])
    reward, components = jax.jit(
        lambda s, n, t: s._get_reward(n, idle, t)
    )(state, after, terms)
    existence = float(state.env_cfg.rewards.existence) / float(
        state.env_cfg.rewards.normalizer
    )
    np.testing.assert_allclose(float(components["existence"]), existence, rtol=1e-6)
    np.testing.assert_allclose(float(reward), existence, rtol=1e-6)


def test_team_reward_is_the_sum_of_each_agents_action_terms():
    state = _team([CENTER, CENTER + np.array([0, 20], dtype=np.int16)])
    state = state._replace(
        env_cfg=state.env_cfg._replace(reward_stage=RewardStage.DENSE_SKILL)
    )
    moves = _actions(0, 2)  # slot 0 forward, slot 1 rotates
    after, terms = jax.jit(lambda s: s._step_joint(moves))(state)
    reward, components = jax.jit(
        lambda s, n, t: s._get_reward(n, moves, t)
    )(state, after, terms)
    normalizer = float(state.env_cfg.rewards.normalizer)
    per_agent = np.asarray(components["agent_rewards"])[:2]
    np.testing.assert_allclose(per_agent, np.asarray(terms["agent_reward"]) / normalizer, rtol=1e-6)
    assert np.all(per_agent != 0)
    np.testing.assert_allclose(
        float(reward),
        per_agent.sum() + float(components["existence"]) + float(components["terminal"]),
        rtol=1e-6,
    )


def test_team_observation_stacks_agent_centric_views():
    state = _team([CENTER, CENTER + np.array([0, 20], dtype=np.int16)])
    env = TerraEnv.new(maps_size_px=SHAPE[0])
    observation = env._observation(TerraEnv.wrap_state(state))
    for key in AGENT_VIEW_OBS_KEYS:
        if key in observation:
            assert observation[key].shape[0] == 2, key
    for slot in range(2):
        pos = np.asarray(state.agent.agent_states[slot].pos_base)
        np.testing.assert_array_equal(np.asarray(observation["agent_states"][slot, 0, :2]), pos)
    assert observation["action_map"].shape == SHAPE
    # Each view: own chassis -1, the teammate a blocked cell, own workspace.
    for slot, other in ((0, 1), (1, 0)):
        view = state._replace(agent=state.agent._replace(current_agent=jnp.int32(slot)))
        own = np.asarray(view._current_base_footprint_mask())
        mate = np.asarray(
            state._replace(agent=state.agent._replace(current_agent=jnp.int32(other)))
            ._current_base_footprint_mask()
        )
        traversability = np.asarray(observation["traversability_mask"][slot])
        assert np.all(traversability[own] == -1) and np.all(traversability[mate] == 1)
        np.testing.assert_array_equal(
            np.asarray(observation["interaction_mask"][slot]),
            np.asarray(view._build_dig_dump_cone()).reshape(SHAPE),
        )
    for slot in range(2):
        view = state._replace(agent=state.agent._replace(current_agent=jnp.int32(slot)))
        alone = env._state_to_obs_dict(LocalMapWrapper.wrap(TerraEnv.wrap_state(view)))
        np.testing.assert_array_equal(
            np.asarray(observation["local_map_action_neg"][slot]),
            np.asarray(alone["local_map_action_neg"]),
        )
