"""Random fleet starts are separate; impossible placement terminates loudly."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import terra.agent as agent_module
from terra.agent import Agent
from terra.config import EnvConfig
from terra.utils import get_agent_corners
from terra.workspace_guard import pose_reservation, reservations_conflict


def config(types, *, guarded=True):
    cfg = EnvConfig()
    return cfg._replace(tile_size=np.float32(4 / 7),
        agent=cfg.agent._replace(width=7, height=11),
        maps=cfg.maps._replace(edge_length_px=64),
        agent_types=types, action_types=(0,) * len(types),
        workspace_guard_enabled=guarded, workspace_loading_pairs=0xFFFF)


def reset(cfg, key, padding=None):
    if padding is None:
        padding = jnp.zeros((64, 64), dtype=jnp.int8)
    return Agent.new(key, cfg, 64, 64, padding,
        jnp.zeros((64, 64), dtype=jnp.int8), jnp.ones((64, 64), dtype=jnp.bool_),
        agent_types=cfg.agent_types, action_types=cfg.action_types)[0]


def separate(agent, cfg):
    reservations = [pose_reservation(a, get_agent_corners(a.pos_base, a.angle_base,
        cfg.agent.width, cfg.agent.height, cfg.agent.angles_base), cfg)
        for a in agent.agent_states[:len(cfg.agent_types)]]
    return ~jnp.any(jnp.stack([reservations_conflict(first, second, cfg)
        for i, first in enumerate(reservations) for second in reservations[i+1:]]))


@pytest.mark.parametrize('types', [(0, 0), (0, 2), (0, 1), (0, 1, 2), (0, 0, 2, 1)])
def test_jitted_vmapped_fleets_start_with_disjoint_full_workspaces(types):
    cfg = config(types)
    keys = jax.random.split(jax.random.PRNGKey(42), 8)
    agents = jax.jit(jax.vmap(lambda key: reset(cfg, key)))(keys)
    clear = jax.jit(jax.vmap(lambda agent: separate(agent, cfg)))(agents)
    assert np.asarray(clear).all()
    # Even all explicit loading permissions cannot overlap an empty reset.
    assert cfg.workspace_loading_pairs == 0xFFFF


def test_successful_vmap_never_invokes_failure_callback(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_module, '_raise_reset_placement_error',
                        lambda *args: calls.append(args))
    cfg = config((0, 2))
    agents = jax.jit(jax.vmap(lambda key: reset(cfg, key)))(
        jax.random.split(jax.random.PRNGKey(73), 8))
    jax.block_until_ready(agents)
    jax.effects_barrier()
    assert calls == []


@pytest.mark.parametrize('guarded', [True, False])
def test_impossible_reset_fails_after_bounded_search(guarded):
    cfg = config((0,), guarded=guarded)
    blocked = jnp.ones((64, 64), dtype=jnp.int8)
    with pytest.raises(Exception, match='2048 attempts'):
        jax.block_until_ready(jax.jit(lambda key: reset(cfg, key, blocked))(
            jax.random.PRNGKey(0)))


def test_one_impossible_member_of_vmap_fails_loudly():
    cfg = config((0,))
    padding = jnp.stack((jnp.zeros((64, 64), jnp.int8), jnp.ones((64, 64), jnp.int8)))
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    with pytest.raises(Exception, match='2048 attempts'):
        jax.block_until_ready(jax.jit(jax.vmap(lambda key, mask: reset(cfg, key, mask)))(keys, padding))


def test_disabled_guard_preserves_recorded_historical_random_draws():
    cfg = config((0, 2), guarded=False)
    expected_positions = [[[23, 45], [44, 32]], [[26, 21], [41, 20]], [[32, 43], [10, 38]]]
    expected_angles = [[[5], [4]], [[10], [7]], [[6], [10]]]
    # Baseline values from agent.py at d519c3be, same JAX CPU reset keys.
    for seed in range(3):
        agent = jax.jit(lambda key: reset(cfg, key))(jax.random.PRNGKey(seed))
        np.testing.assert_array_equal([a.pos_base for a in agent.agent_states[:2]], expected_positions[seed])
        np.testing.assert_array_equal([a.angle_base for a in agent.agent_states[:2]], expected_angles[seed])


def test_real_batched_autoreset_preserves_terminal_workspace_diagnostics():
    from terra.actions import TrackedAction
    from terra.env import TerraEnv, TerraEnvBatch

    cfg = config((0, 2))
    env = TerraEnv.new(64)
    zeros = jnp.zeros((64, 64), dtype=jnp.int8)
    target = zeros.at[20, 20].set(-1).at[55, 55].set(1)
    layers = (target, zeros, jnp.full((3, 3), -97., jnp.float32),
              jnp.int32(-1), jnp.full((64, 3), -97., jnp.float32),
              jnp.int32(-1), jnp.ones((64, 64), jnp.bool_), zeros,
              jnp.ones((64, 64), jnp.float32))
    single = env.reset(jax.random.PRNGKey(14), *layers, cfg)
    single = single._replace(state=single.state._replace(env_steps=jnp.int32(37)))
    timestep = jax.tree_util.tree_map(lambda x: jnp.stack((x, x)), single)
    terminal_info = dict(timestep.info,
        task_done=jnp.array([True, False]),
        workspace_blocked=jnp.array([[True, False], [False, True]]),
        workspace_conflicts=jnp.array([1, 2], jnp.int32),
        effective_actions=jnp.array([[7, 0], [6, 7]], jnp.int32),
        transition_unload_units=jnp.array([[0, 7, 0, 0], [0, 3, 0, 0]], jnp.int32))
    timestep = timestep._replace(done=jnp.array([True, False]),
        reward=jnp.array([3., 4.], jnp.float32), info=terminal_info)

    # Stub only map selection: reset states, observations, infos, and both
    # vmapped conditional branches execute the real native implementation.
    batch = object.__new__(TerraEnvBatch)
    batch.terra_env = env
    next_target = zeros.at[12, 12].set(-1).at[55, 55].set(1)
    reset_layers = tuple(jnp.stack((x, x)) for x in (next_target, *layers[1:]))
    batch._get_map = lambda keys, configs: (*reset_layers, keys)
    keys = jax.random.split(jax.random.PRNGKey(25), 2)
    actions = TrackedAction.new(jnp.array([[0, 0], [6, 7]], jnp.int32))
    result = jax.jit(lambda ts: batch._reset_done_envs(ts, actions, keys))(timestep)
    jax.block_until_ready(result)

    np.testing.assert_array_equal(result.state.env_steps, [0, 37])
    np.testing.assert_array_equal(result.state.world.target_map.map[0], next_target)
    np.testing.assert_array_equal(result.state.world.target_map.map[1], target)
    np.testing.assert_array_equal(result.done, timestep.done)
    np.testing.assert_array_equal(result.reward, timestep.reward)
    for name in (*env._zero_transition_diagnostics(2), 'task_done',
                 'reward_components', 'ended_reset_tier'):
        for actual, expected in zip(jax.tree_util.tree_leaves(result.info[name]),
                                    jax.tree_util.tree_leaves(timestep.info[name]), strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(timestep)
    for actual, expected in zip(jax.tree_util.tree_leaves(result),
                                jax.tree_util.tree_leaves(timestep), strict=True):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
