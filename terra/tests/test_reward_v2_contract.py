from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.config import RewardStage
from terra.config import REWARD_V2_ALPHA
from terra.config import REWARD_V2_BETA
from terra.config import REWARD_V2_DISTANCE_BOUND
from terra.config import REWARD_V2_HORIZON_FAILURE_PENALTY
from terra.config import REWARD_V2_POTENTIAL_GAMMA
from terra.config import REWARD_V2_SHAPING_WEIGHT
from terra.config import REWARD_V2_STEP_COST_TOTAL
from terra.config import REWARD_V2_SUCCESS_BONUS
from terra.config import REWARD_V2_TIMING_BASELINE
from terra.config import REWARD_V2_TIMING_V21
from terra.config import REWARD_V2_V21_STEP_COST_TOTAL
from terra.env import TerraEnv
from terra.env import TerraEnvBatch
from terra.state import State

SHAPE = (64, 64)


def _env_config(timing_variant: int = REWARD_V2_TIMING_BASELINE) -> EnvConfig:
    batch_env = object.__new__(TerraEnvBatch)
    batch_env.batch_cfg = BatchConfig()._replace(
        maps_dims=MapsDimsConfig(maps_edge_length=SHAPE[0])
    )
    base = EnvConfig()
    updated = batch_env.update_env_cfgs(
        base._replace(
            agent=base.agent._replace(dig_depth=jnp.ones((1,), dtype=jnp.int32))
        )
    )
    return base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(edge_length_px=SHAPE[0]),
        max_steps_in_episode=450,
        agent_types=(0,),
        action_types=(0,),
        reward_stage=RewardStage.REWARD_V2,
        reward_v2_timing_variant=timing_variant,
    )


def _state(timing_variant: int = REWARD_V2_TIMING_BASELINE) -> State:
    target = np.zeros(SHAPE, dtype=np.int8)
    target[20, 20:24] = -1
    target[40, 40:48] = 1
    distance = np.ones(SHAPE, dtype=np.float32)
    distance[target > 0] = 0.0
    state = State.new(
        jax.random.PRNGKey(20260810),
        _env_config(timing_variant),
        target,
        np.zeros(SHAPE, dtype=np.int8),
        -97.0 * np.ones((4, 3), dtype=np.float32),
        np.int32(-1),
        np.zeros(SHAPE, dtype=np.uint8),
        -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1),
        np.ones(SHAPE, dtype=np.bool_),
        np.zeros(SHAPE, dtype=np.int8),
        distance_map_override=distance,
    )
    return state._replace(agent=state.agent._replace(current_agent=jnp.int32(0)))


def _with_material(
    state: State,
    action: np.ndarray,
    *,
    loaded: int,
    carry_work: float,
    env_steps: int,
) -> State:
    agent = state.agent.agent_states[0]._replace(
        loaded=jnp.asarray([loaded], dtype=jnp.int8),
        carry_relocation_credit=jnp.float32(carry_work),
    )
    return state._replace(
        world=state.world._replace(
            action_map=state.world.action_map._replace(
                map=jnp.asarray(action, dtype=jnp.int8)
            )
        ),
        agent=state.agent._replace(agent_states=(agent, *state.agent.agent_states[1:])),
        env_steps=env_steps,
    )


def _reward_v2(old_state: State, new_state: State):
    done, exact = new_state._is_done(
        new_state.world.action_map.map,
        new_state.world.target_map.map,
    )
    return old_state._get_reward_v2(new_state, done, exact)


def test_reward_v2_material_work_and_carry_observation_are_markov():
    state = _state()
    assert float(state.material_q_reset) == 0.0
    np.testing.assert_array_equal(
        np.asarray(state._reward_v2_state_values()[0]),
        np.asarray(jnp.float32(0.0)),
    )
    action = np.zeros(SHAPE, dtype=np.int8)
    action[20, 20] = -1
    lifted = _with_material(
        state,
        action,
        loaded=1,
        carry_work=1.0,
        env_steps=1,
    )

    q, h, p, phi, valid = lifted._reward_v2_state_values()
    np.testing.assert_allclose(h, state.material_h_reset, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(q, 0.25, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(p, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(
        np.asarray(TerraEnv._state_to_obs_dict(state)["reward_v2_reset_context"]),
        np.asarray(
            [
                state.material_q_reset,
                state.material_h_reset / state._required_excavation_volume(),
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        phi,
        REWARD_V2_ALPHA * 0.25 + REWARD_V2_BETA * REWARD_V2_DISTANCE_BOUND,
        rtol=0.0,
        atol=1e-6,
    )
    assert float(valid) == 1.0

    observation = TerraEnv._state_to_obs_dict(lifted)
    assert observation["agent_states"].shape == (4, 9)
    np.testing.assert_allclose(
        observation["agent_states"][0, 8],
        0.25,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(observation["agent_states"])[1:, 8],
        np.zeros((3,), dtype=np.float32),
    )

    flat_next = lifted._replace(env_steps=2)
    reward, components = _reward_v2(lifted, flat_next)
    expected_shaping = (REWARD_V2_POTENTIAL_GAMMA - 1.0) * float(phi)
    np.testing.assert_allclose(
        components["reward_v2_shaping"],
        expected_shaping,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reward,
        expected_shaping - REWARD_V2_STEP_COST_TOTAL / 450.0,
        rtol=0.0,
        atol=1e-6,
    )
    assert float(reward) < 0.0
    integrated_reward, integrated_components = lifted._get_reward(
        flat_next,
        TrackedAction.do_nothing(),
    )
    np.testing.assert_allclose(integrated_reward, reward, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        integrated_components["reward_v2_phi_next"],
        components["reward_v2_phi_next"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        integrated_reward,
        jnp.sum(integrated_components["agent_rewards"])
        + integrated_components["terminal"]
        + integrated_components["existence"],
        rtol=0.0,
        atol=1e-7,
    )


def test_reward_v2_endpoints_cycles_and_dominance_primitives():
    state = _state()
    initial_phi = float(state._reward_v2_state_values()[3])
    progressed_phi = initial_phi + 0.5
    gamma = REWARD_V2_POTENTIAL_GAMMA
    step = -REWARD_V2_STEP_COST_TOTAL / 450.0
    forward = gamma * progressed_phi - initial_phi + step
    backward = gamma * initial_phi - progressed_phi + step
    assert forward + gamma * backward < 0.0

    timeout = state._replace(env_steps=450)
    timeout_reward, timeout_components = _reward_v2(state, timeout)
    assert float(timeout_components["reward_v2_success"]) == 0.0
    assert float(timeout_components["reward_v2_horizon_failure"]) == -1.0
    assert float(timeout_reward) < -1.0

    success_action = np.zeros(SHAPE, dtype=np.int8)
    success_action[20, 20:24] = -1
    success_action[40, 40] = 4
    success = _with_material(
        state,
        success_action,
        loaded=0,
        carry_work=0.0,
        env_steps=450,
    )
    success_reward, success_components = _reward_v2(state, success)
    assert bool(success._is_done_task(success_action, success.world.target_map.map))
    assert float(success_components["reward_v2_success"]) == 6.0
    assert float(success_components["reward_v2_horizon_failure"]) == 0.0
    assert float(success_reward) > float(timeout_reward)

    phi_reset = REWARD_V2_BETA * REWARD_V2_DISTANCE_BOUND
    phi_success_min = REWARD_V2_ALPHA + phi_reset
    phi_failure_max = REWARD_V2_ALPHA + REWARD_V2_BETA * (
        2.0 * REWARD_V2_DISTANCE_BOUND
    )

    def discounted_return(steps: int, terminal: float, phi_terminal: float) -> float:
        step_return = step * (1.0 - gamma**steps) / (1.0 - gamma)
        shaping_return = gamma**steps * phi_terminal - phi_reset
        return step_return + shaping_return + gamma ** (steps - 1) * terminal

    minimum_success = min(
        discounted_return(steps, REWARD_V2_SUCCESS_BONUS, phi_success_min)
        for steps in range(1, 451)
    )
    maximum_horizon_failure = discounted_return(
        450,
        -REWARD_V2_HORIZON_FAILURE_PENALTY,
        phi_failure_max,
    )
    np.testing.assert_allclose(minimum_success, 0.7710143, atol=1e-6)
    np.testing.assert_allclose(maximum_horizon_failure, -0.8154759, atol=1e-6)
    assert minimum_success > maximum_horizon_failure


def _dwell_and_progress(timing_variant: int):
    """Return (dwell, progress) reward/component pairs under one variant."""
    state = _state(timing_variant)
    action = np.zeros(SHAPE, dtype=np.int8)
    action[20, 20] = -1
    lifted = _with_material(state, action, loaded=1, carry_work=1.0, env_steps=1)
    dwell = _reward_v2(lifted, lifted._replace(env_steps=2))

    progressed_action = np.zeros(SHAPE, dtype=np.int8)
    progressed_action[20, 20:22] = -1
    progressed = _with_material(
        state,
        progressed_action,
        loaded=1,
        carry_work=1.0,
        env_steps=2,
    )
    return dwell, _reward_v2(lifted, progressed)


def test_reward_v2_timing_variant_zero_is_the_frozen_reward():
    """Variant 0 must reproduce w * (gamma * Phi_next - Phi) bit for bit."""
    for (_, components) in _dwell_and_progress(REWARD_V2_TIMING_BASELINE):
        frozen = jnp.float32(REWARD_V2_SHAPING_WEIGHT) * (
            jnp.float32(REWARD_V2_POTENTIAL_GAMMA)
            * components["reward_v2_phi_next"]
            - components["reward_v2_phi"]
        )
        np.testing.assert_array_equal(
            np.asarray(components["reward_v2_shaping"]),
            np.asarray(frozen),
        )


def test_reward_v2_timing_variant_zero_keeps_the_frozen_step_cost():
    """The frozen 1.0/450 step cost is untouched by the selector's presence."""
    for (_, components) in _dwell_and_progress(REWARD_V2_TIMING_BASELINE):
        np.testing.assert_array_equal(
            np.asarray(components["reward_v2_step"]),
            np.asarray(-jnp.float32(REWARD_V2_STEP_COST_TOTAL) / jnp.float32(450.0)),
        )


def test_reward_v21_prices_dwelling_at_the_explicit_step_cost_only():
    """v2.1 shapes undiscounted, so standing still pays the pace and no rent."""
    (dwell_reward, dwell), (progress_reward, progress) = _dwell_and_progress(
        REWARD_V2_TIMING_V21
    )
    assert float(dwell["reward_v2_shaping"]) == 0.0
    np.testing.assert_allclose(
        float(dwell["reward_v2_step"]), -0.0080, rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        float(dwell_reward),
        -REWARD_V2_V21_STEP_COST_TOTAL / 450.0,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(progress["reward_v2_shaping"]),
        np.asarray(
            jnp.float32(REWARD_V2_SHAPING_WEIGHT)
            * (progress["reward_v2_phi_next"] - progress["reward_v2_phi"])
        ),
    )
    assert float(progress_reward) > float(dwell_reward)


def test_reward_v21_replaces_implicit_rent_with_explicit_pace():
    """Total per-step time pressure is preserved; the Phi-dependent part is not."""
    (baseline_reward, baseline), _ = _dwell_and_progress(REWARD_V2_TIMING_BASELINE)
    (v21_reward, v21), _ = _dwell_and_progress(REWARD_V2_TIMING_V21)
    # The baseline's dwell charge is implicit rent: w*(1-gamma)*Phi, so it is
    # set by an untuned constant (beta*D_bound = 3.75 at reset) and drifts with
    # progress. That is the whole defect.
    rent = -float(baseline["reward_v2_shaping"])
    np.testing.assert_allclose(
        rent,
        (1.0 - REWARD_V2_POTENTIAL_GAMMA) * float(baseline["reward_v2_phi"]),
        rtol=0.0,
        atol=1e-6,
    )
    reset_rent = (1.0 - REWARD_V2_POTENTIAL_GAMMA) * (
        REWARD_V2_BETA * REWARD_V2_DISTANCE_BOUND
    )
    np.testing.assert_allclose(reset_rent, 0.0060, rtol=0.0, atol=1e-7)
    assert rent > reset_rent  # Phi-dependent, hence untunable as a pace.
    # v2.1 pays the same total pace explicitly and Phi-independently.
    baseline_pressure = reset_rent + REWARD_V2_STEP_COST_TOTAL / 450.0
    v21_pressure = -float(v21["reward_v2_step"])
    np.testing.assert_allclose(v21_pressure, 0.0080, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        v21_pressure, baseline_pressure, rtol=0.03, atol=0.0
    )
    assert float(v21["reward_v2_shaping"]) == 0.0
    # The guard and the logged potentials are identical under both variants.
    for key in ("reward_v2_valid", "reward_v2_phi", "reward_v2_phi_next",
                "reward_v2_success", "reward_v2_horizon_failure"):
        np.testing.assert_array_equal(np.asarray(v21[key]), np.asarray(baseline[key]))
    assert float(v21_reward) > float(baseline_reward)
