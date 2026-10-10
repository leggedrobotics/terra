"""Elapsed-time clock: action durations, dependency delays and the reward.

Run with JAX_PLATFORMS=cpu and this Terra on PYTHONPATH.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.actions import TrackedAction
from terra.config import (
    RewardStage, TIME_CABIN_TURN_S_PER_RAD, TIME_DIG_S_PER_M3, TIME_NAV_SPEED_MPS,
    TIME_RELIFT_S_PER_M3, TIME_RELIFT_SETUP_S, TIME_RELOCATION_S, TIME_SETUP_S,
)
from terra.env import TerraEnv
from terra.tests.test_foundation_behavior import _pose, foundation  # noqa: F401
from terra.tests.test_relocation_reward_contract import SHAPE, _env_config, _state

DO, WAIT, FORWARD, CABIN_CLOCK = 6, 7, 0, 4
_step = jax.jit(lambda state, action: state._step(action))
_reward = jax.jit(lambda s, n: s._get_reward_v2(n, jnp.bool_(False), jnp.bool_(False)))


def _act(*indices):
    return TrackedAction.new(jnp.asarray(indices, dtype=jnp.int8))


def _unit_s(state):
    tile = float(state.env_cfg.tile_size)
    return tile ** 3 * TIME_DIG_S_PER_M3


def _load(state, slot=0):
    return int(state.agent.agent_states[slot].loaded[0])


def test_single_machine_durations(foundation):  # noqa: F811
    state = foundation
    tile = float(state.env_cfg.tile_size)
    dug = _step(state, _act(DO))
    load = _load(dug)
    assert load > 0
    # First visit without prior motion: setup and loading, no relocation.
    first = load * _unit_s(state) + TIME_SETUP_S
    np.testing.assert_allclose(float(dug.machine_clock_s[0]), first, rtol=1e-5)
    np.testing.assert_allclose(float(dug.machine_busy_s[0]), first, rtol=1e-5)
    assert bool(dug.machine_visit_open[0]) and not bool(dug.machine_moved[0])

    # WAIT takes no time.
    waited = _step(dug, _act(WAIT))
    np.testing.assert_array_equal(waited.machine_clock_s, dug.machine_clock_s)

    # A cabin step is a swing, not a new visit; dumping is inside the dig rate.
    swung = _step(dug, _act(CABIN_CLOCK))
    swing = 2 * np.pi / 12 * TIME_CABIN_TURN_S_PER_RAD
    np.testing.assert_allclose(float(swung.machine_clock_s[0]), first + swing, rtol=1e-5)
    dumped = _step(_pose(dug, cabin=6), _act(DO))
    assert _load(dumped) < load
    np.testing.assert_allclose(float(dumped.machine_clock_s[0]), first, rtol=1e-5)

    # Driving closes the visit; the next dig pays travel, relocation and setup.
    empty = _pose(dumped, cabin=0)
    moved = _step(empty, _act(FORWARD))
    distance = np.linalg.norm(np.asarray(moved.agent.agent_states[0].pos_base, np.float32)
                              - np.asarray(empty.agent.agent_states[0].pos_base, np.float32))
    assert distance > 0
    travel = distance * tile / TIME_NAV_SPEED_MPS
    np.testing.assert_allclose(float(moved.machine_clock_s[0]), first + travel, rtol=1e-5)
    assert not bool(moved.machine_visit_open[0]) and bool(moved.machine_moved[0])
    redug = _step(moved, _act(DO))
    new_load = _load(redug)
    assert new_load > 0
    expected = first + travel + TIME_RELOCATION_S + TIME_SETUP_S + new_load * _unit_s(state)
    np.testing.assert_allclose(float(redug.machine_clock_s[0]), expected, rtol=1e-5)


@pytest.fixture(scope="module")
def far_team():
    target = np.zeros(SHAPE, np.int8)
    target[8:56, 8:56] = -1
    cfg = _env_config((0, 0))._replace(reward_stage=RewardStage.REWARD_V2,
                                       workspace_guard_enabled=True)
    with jax.disable_jit():
        state = _state(target, env_cfg=cfg)
    for slot, xy in enumerate(((16, 16), (48, 48))):
        a = state.agent.agent_states[slot]
        state = state._set_agent_state_at(slot, a._replace(
            pos_base=jnp.asarray(xy, a.pos_base.dtype),
            angle_base=jnp.zeros_like(a.angle_base), angle_cabin=jnp.zeros_like(a.angle_cabin),
            loaded=jnp.zeros_like(a.loaded)))
    return state


def test_separate_workspaces_dig_in_parallel(far_team):
    both = _step(far_team, _act(DO, DO))
    loads = [_load(both, 0), _load(both, 1)]
    assert min(loads) > 0
    unit = _unit_s(far_team)
    expected = [load * unit + TIME_SETUP_S for load in loads]
    np.testing.assert_allclose(np.asarray(both.machine_clock_s[:2]), expected, rtol=1e-5)
    # T is the later machine, not the sum.
    np.testing.assert_allclose(float(both._time_finish_s()), max(expected), rtol=1e-5)
    np.testing.assert_allclose(float(jnp.sum(both.machine_busy_s)), sum(expected), rtol=1e-5)


def test_overlap_with_an_earlier_reservation_waits_for_its_release(far_team):
    release = np.zeros(np.asarray(far_team.machine_release_s).shape, np.float32)
    # The other machine's earlier work covers this machine's cells until 5000 s.
    release[1, 8:24, 8:24] = 5000.0
    delayed = _step(far_team._replace(machine_release_s=jnp.asarray(release)), _act(DO, WAIT))
    duration = _load(delayed) * _unit_s(far_team) + TIME_SETUP_S
    np.testing.assert_allclose(float(delayed.machine_clock_s[0]), 5000.0 + duration, rtol=1e-5)
    # The same release far away does not delay it.
    release[1] = 0.0
    release[1, 56:, 56:] = 5000.0
    free = _step(far_team._replace(machine_release_s=jnp.asarray(release)), _act(DO, WAIT))
    np.testing.assert_allclose(float(free.machine_clock_s[0]), duration, rtol=1e-5)
    # Its own finish is published over its grown envelope only.
    own = np.asarray(free.machine_release_s[0])
    assert own.max() == pytest.approx(duration, rel=1e-5)
    assert own[16, 16] == pytest.approx(duration, rel=1e-5) and own[48, 48] == 0.0


def _with_clocks(state, clock, busy):
    return state._replace(machine_clock_s=jnp.asarray(clock + [0.0] * (4 - len(clock)), jnp.float32),
                          machine_busy_s=jnp.asarray(busy + [0.0] * (4 - len(busy)), jnp.float32))


def test_reward_charges_finish_growth_and_busy_time(far_team):
    charged = far_team._replace(env_cfg=far_team.env_cfg._replace(
        elapsed_time_cost=2.0, busy_time_cost=0.2))
    reference = float(charged._time_reference_s())
    volume = float(charged._reward_v2_volume())
    np.testing.assert_allclose(reference, volume * _unit_s(charged), rtol=1e-6)
    before = _with_clocks(charged, [1000.0, 400.0], [900.0, 400.0])
    # The lagging machine works 300 s inside the leader's time: busy cost only.
    inside = _with_clocks(charged, [1000.0, 700.0], [900.0, 700.0])
    terms = _reward(before, inside)[1]
    np.testing.assert_allclose(float(terms["reward_v2_elapsed_time"]), -0.2 * 300 / reference, rtol=1e-5)
    # Passing the leader by 100 s also costs 2 * 100 s.
    past = _with_clocks(charged, [1000.0, 1100.0], [900.0, 1100.0])
    terms = _reward(before, past)[1]
    np.testing.assert_allclose(float(terms["reward_v2_elapsed_time"]),
                               -(2.0 * 100 + 0.2 * 700) / reference, rtol=1e-5)
    # Disabled costs leave the reward unchanged.
    plain = far_team
    r_plain = _reward(_with_clocks(plain, [1000.0, 400.0], [900.0, 400.0]),
                      _with_clocks(plain, [1000.0, 1100.0], [900.0, 1100.0]))
    r_same = _reward(plain, plain)
    assert float(r_plain[1]["reward_v2_elapsed_time"]) == 0.0
    np.testing.assert_allclose(float(r_plain[0]), float(r_same[0]), rtol=1e-6)


def test_observation_carries_clocks_only_with_the_objective(far_team):
    state = _with_clocks(far_team, [1000.0, 400.0], [900.0, 400.0])
    plain = np.asarray(TerraEnv._state_to_obs_dict(state)["agent_states"])
    job = float(state._makespan_job_s())
    np.testing.assert_allclose(plain[0, 9], float(state.machine_work_s[0]) / job, rtol=1e-6)
    timed = state._replace(env_cfg=state.env_cfg._replace(elapsed_time_cost=2.0))
    obs = np.asarray(TerraEnv._state_to_obs_dict(timed)["agent_states"])
    reference = float(timed._time_reference_s())
    # Acting slot 0 first, then slot 1.
    np.testing.assert_allclose(obs[:2, 9], [1000.0 / reference, 400.0 / reference], rtol=1e-6)
    np.testing.assert_allclose(obs[:2, 10], [1000.0 / reference] * 2, rtol=1e-6)


def test_setup_per_dug_workspace_and_relift_rates(foundation):  # noqa: F811
    state = foundation
    unit = _unit_s(state)
    dug = _step(state, _act(DO))
    first = _load(dug) * unit + TIME_SETUP_S
    np.testing.assert_allclose(float(dug.machine_clock_s[0]), first, rtol=1e-5)
    dumped = _step(_pose(dug, cabin=6), _act(DO))
    assert _load(dumped) == 0
    # A new cabin sector of the same visit is a new dug workspace.
    side = _step(_pose(dumped, cabin=3), _act(DO))
    assert _load(side) > 0
    np.testing.assert_allclose(float(side.machine_clock_s[0]),
                               first + _load(side) * unit + TIME_SETUP_S, rtol=1e-5)
    # Picking the dumped soil back up is a relift: collect setup and rate.
    relift = _step(_pose(dumped, cabin=6), _act(DO))
    assert _load(relift) > 0
    tile = float(state.env_cfg.tile_size)
    np.testing.assert_allclose(
        float(relift.machine_clock_s[0]),
        first + _load(relift) * tile ** 3 * TIME_RELIFT_S_PER_M3 + TIME_RELIFT_SETUP_S,
        rtol=1e-5)
    # A one-sector Terra dig takes the whole cone, so continue a sector with
    # capped relifts: the second pickup from the same sector pays no setup.
    capped = dumped._replace(env_cfg=dumped.env_cfg._replace(excavator_relift_capacity=20))
    part = _step(_pose(capped, cabin=6), _act(DO))
    assert _load(part) == 20
    away = _step(_pose(part, cabin=9), _act(DO))
    assert _load(away) == 0
    rest = _step(_pose(away, cabin=6), _act(DO))
    assert _load(rest) == 20
    np.testing.assert_allclose(
        float(rest.machine_clock_s[0]),
        float(away.machine_clock_s[0]) + 20 * tile ** 3 * TIME_RELIFT_S_PER_M3, rtol=1e-5)


def _set_slot(state, slot, **fields):
    agent = state.agent.agent_states[slot]
    updates = {name: jnp.asarray([value], dtype=getattr(agent, name).dtype)
               for name, value in fields.items()}
    return state._set_agent_state_at(slot, agent._replace(**updates))


def test_dump_closes_its_dig_interval(far_team):
    dug = _step(far_team, _act(DO, WAIT))
    dig_s = float(dug.machine_clock_s[0])
    assert _load(dug, 0) > 0 and dig_s > 0
    swung = _set_slot(dug, 0, angle_cabin=6)
    # Unobstructed: the dump adds no time.
    free = _step(swung, _act(DO, WAIT))
    assert _load(free, 0) < _load(dug, 0)
    np.testing.assert_allclose(float(free.machine_clock_s[0]), dig_s, rtol=1e-6)
    # The partner used this machine's surroundings until 3000 s after the dig
    # had started: the dig, which contains the dump, cannot have started earlier.
    release = np.asarray(swung.machine_release_s).copy()
    release[1, :32, :32] = np.maximum(release[1, :32, :32], 3000.0)
    late = _step(swung._replace(machine_release_s=jnp.asarray(release)), _act(DO, WAIT))
    np.testing.assert_allclose(float(late.machine_clock_s[0]), 3000.0 + dig_s, rtol=1e-6)
    own = np.asarray(late.machine_release_s[0])
    # The dig envelope and the dump envelope are both held until the dump.
    assert own.max() == pytest.approx(3000.0 + dig_s, rel=1e-6)
    assert own[16, 16] == pytest.approx(3000.0 + dig_s, rel=1e-6)


def test_stall_round_is_charged(far_team):
    stalled = far_team._replace(env_cfg=far_team.env_cfg._replace(stall_cost=0.01))
    waited = _step(stalled, _act(WAIT, WAIT))
    assert int(waited.stall_age_steps) > 0
    np.testing.assert_allclose(float(_reward(stalled, waited)[1]["reward_v2_stall"]), -0.01)
    worked = _step(stalled, _act(DO, WAIT))
    assert float(_reward(stalled, worked)[1]["reward_v2_stall"]) == 0.0
    # Disabled by default.
    plain = _step(far_team, _act(WAIT, WAIT))
    assert float(_reward(far_team, plain)[1]["reward_v2_stall"]) == 0.0


def test_lockstep_sums_each_rounds_slowest_action(far_team):
    both = _step(far_team, _act(DO, DO))
    durations = np.asarray(both.machine_busy_s[:2])
    np.testing.assert_allclose(float(both.lockstep_s), durations.max(), rtol=1e-6)
    waited = _step(both, _act(WAIT, CABIN_CLOCK))
    swing = float(waited.machine_busy_s[1] - both.machine_busy_s[1])
    np.testing.assert_allclose(float(waited.lockstep_s), durations.max() + swing, rtol=1e-6)
