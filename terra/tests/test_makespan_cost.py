"""Per-machine executed-plan time and the team makespan cost."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.actions import TrackedAction
from terra.env import TerraEnv
from terra.state import State
from terra.tests.test_foundation_behavior import _do, _pose, foundation  # noqa: F401
from terra.tests.test_reward_v2_contract import _env_config

SETUP_S = 30.0


def _loose(state):
    cone = np.asarray(state._build_dig_dump_cone()).reshape((64, 64))
    cells = np.argwhere(cone)
    piles = np.zeros((64, 64), dtype=np.int8)
    piles[tuple(cells[len(cells) // 2])] = 3
    world = state.world
    return state._replace(world=world._replace(
        action_map=world.action_map._replace(map=jnp.asarray(piles, dtype=world.action_map.map.dtype)),
        last_dig_mask=world.last_dig_mask._replace(map=jnp.zeros((64, 64), dtype=jnp.bool_)),
    ))


def _with_setup(state, cost=0.0):
    return state._replace(env_cfg=state.env_cfg._replace(
        makespan_setup_s=SETUP_S, makespan_cost=cost,
    ))


def test_work_accumulates_loading_travel_and_setups(foundation):  # noqa: F811
    state = _with_setup(foundation)
    unit_s = float(state._makespan_unit_s())
    tile = float(state.env_cfg.tile_size)
    np.testing.assert_allclose(unit_s, tile ** 3 / 0.3 * 30.0, rtol=1e-6)

    first = _do(state)
    load = int(first.agent.agent_states[0].loaded[0])
    assert load > 0
    np.testing.assert_allclose(float(first.machine_work_s[0]), load * unit_s + SETUP_S, rtol=1e-5)

    # Unloading at the same work pose adds neither loading nor a setup.
    dumped = _do(_pose(first, cabin=6))
    assert int(dumped.agent.agent_states[0].loaded[0]) < load
    np.testing.assert_allclose(float(dumped.machine_work_s[0]), float(first.machine_work_s[0]), rtol=1e-6)

    # A relift at a new pose adds travel from the previous work pose, a setup
    # and the lifted units.
    moved = _loose(_pose(dumped, position=(35, 36), base=0, cabin=0, loaded=0))
    relifted = _do(moved)
    lifted = int(relifted.agent.agent_states[0].loaded[0])
    assert lifted > 0
    expected = (float(dumped.machine_work_s[0]) + 5 * tile / 0.5 + SETUP_S + lifted * unit_s)
    np.testing.assert_allclose(float(relifted.machine_work_s[0]), expected, rtol=1e-5)

    # Motion, cabin swings and a failed DO are not work.
    for after in (relifted._handle_move_forward(), relifted._handle_cabin_clock()):
        np.testing.assert_array_equal(after.machine_work_s, relifted.machine_work_s)


@pytest.fixture(scope="module")
def team():
    target = np.zeros((64, 64), dtype=np.int8)
    target[14:50, 14:50] = -1
    target[4:10, 4:10] = 1
    cfg = _env_config()._replace(agent_types=(0, 0), action_types=(0, 0))
    return State.new(
        jax.random.PRNGKey(7), cfg, target,
        np.zeros_like(target), -97.0 * np.ones((4, 8), dtype=np.float32),
        np.int32(-1), -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1), np.ones_like(target, dtype=np.bool_), np.zeros_like(target),
        distance_map_override=np.ones_like(target, dtype=np.float32),
    )


@jax.jit
def _reward_v2(before, after):
    return before._get_reward_v2(after, jnp.bool_(False), jnp.bool_(False))


def _work(state, *values):
    return state._replace(machine_work_s=jnp.asarray(list(values) + [0.0] * (4 - len(values)), jnp.float32))


def test_makespan_cost_pays_only_growth_of_the_busiest_machine(team):
    job_s = float(team._makespan_job_s())
    before = _work(_with_setup(team, cost=2.0), 100.0, 40.0)
    free = _with_setup(team, cost=0.0)
    for after_work, growth in (((150.0, 40.0), 50.0),   # leader works: charged
                               ((100.0, 90.0), 0.0),    # other machine catches up: free
                               ((100.0, 130.0), 30.0)):  # overtakes: charged beyond the old max
        after = _work(before, *after_work)
        reward, terms = _reward_v2(before, after)
        baseline, _ = _reward_v2(_work(free, 100.0, 40.0), _work(free, *after_work))
        np.testing.assert_allclose(float(terms["reward_v2_makespan"]), -2.0 * growth / job_s, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(float(reward) - float(baseline), -2.0 * growth / job_s, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(float(terms["reward_v2_makespan_fraction"]), max(after_work) / job_s, rtol=1e-6)


def test_zero_cost_leaves_reward_unchanged(team):
    before = _work(team, 100.0, 40.0)
    after = _work(team, 150.0, 40.0)
    reward, terms = _reward_v2(before, after)
    same_work, _ = _reward_v2(before, before)
    reference, _ = _reward_v2(_work(team, 0.0, 0.0), _work(team, 0.0, 0.0))
    assert float(terms["reward_v2_makespan"]) == 0.0
    # With the cost off, machine work does not enter the reward at all.
    assert float(reward) == float(same_work) == float(reference)


def test_views_observe_every_machines_normalized_work(team):
    state = _work(team, 60.0, 30.0)
    job_s = float(state._makespan_job_s())
    env = TerraEnv.new(maps_size_px=64)
    observation = env._observation(TerraEnv.wrap_state(state))
    agent_states = np.asarray(observation["agent_states"])
    assert agent_states.shape == (2, 4, 10)
    # Each view lists its own machine first.
    np.testing.assert_allclose(agent_states[0, :2, 9], [60.0 / job_s, 30.0 / job_s], rtol=1e-6)
    np.testing.assert_allclose(agent_states[1, :2, 9], [30.0 / job_s, 60.0 / job_s], rtol=1e-6)
    np.testing.assert_array_equal(agent_states[:, 2:, 9], 0.0)


def _placed(team, cost=0.0):
    placed = team
    for slot, position in ((0, (24, 24)), (1, (40, 40))):
        agent = placed.agent.agent_states[slot]._replace(
            pos_base=jnp.asarray(position, dtype=jnp.int16),
            angle_base=jnp.array([0], dtype=jnp.int8),
            angle_cabin=jnp.array([0], dtype=jnp.int8),
            loaded=jnp.array([0], dtype=jnp.int8),
        )
        placed = placed._set_agent_state_at(slot, agent)
    return _with_setup(placed, cost=cost)


DIG_BOTH = TrackedAction.new(jnp.asarray([6, 6], dtype=jnp.int8))


def test_joint_round_accumulates_each_machines_work(team):
    placed = _placed(team)
    after = jax.jit(lambda s, a: s._step_joint(a, jnp.arange(2))[0])(placed, DIG_BOTH)
    unit_s = float(placed._makespan_unit_s())
    for slot in (0, 1):
        load = int(after.agent.agent_states[slot].loaded[0])
        assert load > 0
        np.testing.assert_allclose(float(after.machine_work_s[slot]), load * unit_s + SETUP_S, rtol=1e-5)


def test_step_reward_is_reconstructed_by_its_logged_components(team):
    # The episode aggregates check reward = agent rewards + terminal +
    # existence (+ trench); the makespan cost is shared among the agents.
    placed = _placed(team, cost=2.0)
    after, terms = jax.jit(lambda s, a: s._step_joint(a, jnp.arange(2)))(placed, DIG_BOTH)
    reward, components = jax.jit(lambda s, n, t: s._get_reward(n, DIG_BOTH, t))(placed, after, terms)
    assert float(components["reward_v2_makespan"]) < 0
    reconstructed = (
        np.asarray(components["agent_rewards"]).sum() + float(components["terminal"])
        + float(components["existence"]) + float(np.nan_to_num(components["trench"]))
    )
    np.testing.assert_allclose(float(reward), reconstructed, rtol=1e-5, atol=1e-6)
