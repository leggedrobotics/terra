"""Optional transport shaping on a real pickup/backtrack/delivery trace."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.config import REWARD_V2_POTENTIAL_GAMMA, RewardStage
from terra.actions import TrackedAction
from terra.env import TerraEnv
from terra.tests.test_skid_steer_relocation import (
    SHAPE, _bucket, _mass, _skid, _zone, backward, do, forward,
)


reward = jax.jit(lambda old, new: old._get_reward_v2(
    new, *new._is_done(new.world.action_map.map, new.world.target_map.map)
))
potential = jax.jit(lambda state: state._transport_credit_potential())
full_reward = jax.jit(lambda old, new, action: old._get_reward(new, action))
GAMMA = float(np.float32(REWARD_V2_POTENTIAL_GAMMA))


def _coefficient(state, value):
    return state._replace(env_cfg=state.env_cfg._replace(transport_credit_coef=value))


@pytest.fixture(scope="module")
def trace():
    soil = np.zeros(SHAPE, dtype=np.int8)
    bucket = _bucket(_skid(_zone(), soil))
    soil[tuple(np.argwhere(bucket)[:40].T)] = 1
    state = _skid(_zone(), soil)
    state = state._replace(env_cfg=state.env_cfg._replace(reward_stage=RewardStage.REWARD_V2))
    states = [state]
    # Native handlers: scoop, raise bucket, loaded progress/backtrack, deliver.
    # Poses and loads are never assigned after initial fixture construction.
    for transition in [forward, do, forward, backward] + [forward] * 5 + [do]:
        state = transition(state)._replace(env_steps=jnp.int32(state.env_steps + 1))
        states.append(state)
    return states


def test_loaded_motion_is_dense_and_does_not_change_stored_credit(trace):
    old, toward, back = trace[2:5]
    credit = float(old.agent.agent_states[0].carry_relocation_credit)
    for state in trace:
        assert _mass(state) == 40
    for state in (toward, back):
        assert float(state.agent.agent_states[0].carry_relocation_credit) == credit
    np.testing.assert_array_equal(back.agent.agent_states[0].pos_base,
                                  old.agent.agent_states[0].pos_base)
    _, advance = reward(_coefficient(old, 1.0), _coefficient(toward, 1.0))
    _, reverse = reward(_coefficient(toward, 1.0), _coefficient(back, 1.0))
    assert float(advance["reward_v2_transport_shaping"]) > 0
    assert float(reverse["reward_v2_transport_shaping"]) < 0
    np.testing.assert_allclose(advance["reward_v2_transport_shaping"],
                               GAMMA * float(potential(toward)) - float(potential(old)),
                               atol=1e-7)
    # Compare all three actions at exactly the same loaded state.
    wait = old._do_nothing()._replace(env_steps=old.env_steps + 1)
    away = backward(old)._replace(env_steps=old.env_steps + 1)
    _, waiting = reward(_coefficient(old, 1.0), _coefficient(wait, 1.0))
    _, retreat = reward(_coefficient(old, 1.0), _coefficient(away, 1.0))
    assert float(advance["reward_v2_transport_shaping"]) > float(waiting["reward_v2_transport_shaping"])
    assert float(waiting["reward_v2_transport_shaping"]) > float(retreat["reward_v2_transport_shaping"])

    # A native scoop is one unit of job-normalized handling. Ground delivery
    # costs no extra loading; diagnostics retain its terminal unload.
    for before, after, action, volume in (
        (trace[0], trace[1], TrackedAction.forward(), 40),
        (trace[-2], trace[-1], TrackedAction.do(), 0),
    ):
        cfg = before.env_cfg._replace(material_handling_cost=0.25)
        total, components = full_reward(before._replace(env_cfg=cfg), after._replace(env_cfg=cfg), action)
        baseline, _ = full_reward(before, after, action)
        assert float(components["reward_v2_handled_volume"]) == volume
        np.testing.assert_allclose(total - baseline, -0.25 * volume / 40, atol=1e-6)
    diagnostics = TerraEnv._transition_diagnostics(trace[-2], trace[-1])
    np.testing.assert_array_equal(diagnostics["transition_unload_units"], [40, 0, 0, 0])
    assert bool(trace[-1]._is_done(trace[-1].world.action_map.map, trace[-1].world.target_map.map)[1])


def test_discounted_correction_telescopes_at_success_and_true_timeout(trace):
    terms = [reward(_coefficient(old, 1.0), _coefficient(new, 1.0))[1]
             for old, new in zip(trace, trace[1:])]
    assert all(float(term["reward_v2_valid"]) == 1 for term in terms)
    assert float(terms[-1]["reward_v2_success"]) > 0
    assert float(terms[-1]["reward_v2_transport_phi_next"]) == 0
    added_return = sum(GAMMA ** i * float(term["reward_v2_transport_shaping"])
                       for i, term in enumerate(terms))
    np.testing.assert_allclose(added_return, -float(potential(trace[0])), atol=3e-7)
    # A closed loaded forward/backward loop has the exact discounted boundary
    # term, not an undiscounted zero-sum claim.
    loop_return = (float(terms[2]["reward_v2_transport_shaping"])
                   + GAMMA * float(terms[3]["reward_v2_transport_shaping"]))
    np.testing.assert_allclose(loop_return, (GAMMA ** 2 - 1) * float(potential(trace[2])),
                               atol=1e-7)

    loaded = _coefficient(trace[3], 1.0)
    phi = float(potential(loaded))
    assert abs(phi) > 0.01
    # PPO's 32-step cut is not terminal. The environment's step 450 is.
    _, cut = reward(loaded._replace(env_steps=jnp.int32(31)),
                    loaded._replace(env_steps=jnp.int32(32)))
    _, timeout = reward(loaded._replace(env_steps=jnp.int32(449)),
                        loaded._replace(env_steps=jnp.int32(450)))
    np.testing.assert_allclose(cut["reward_v2_transport_phi_next"], phi)
    assert float(timeout["reward_v2_transport_phi_next"]) == 0
    assert float(timeout["reward_v2_horizon_failure"]) < 0
    np.testing.assert_allclose(timeout["reward_v2_transport_shaping"], -phi)
    # Complete a 450-step failure after the native first three transitions;
    # stationary waits share the same potential, so no repeated simulation.
    timeout_terms = [float(term["reward_v2_transport_shaping"]) for term in terms[:3]]
    timeout_terms += [float(cut["reward_v2_transport_shaping"])] * 446
    timeout_terms += [float(timeout["reward_v2_transport_shaping"])]
    assert len(timeout_terms) == 450
    np.testing.assert_allclose(sum(GAMMA ** i * term for i, term in enumerate(timeout_terms)),
                               0.0, atol=3e-6)


def test_default_off_reward_is_bitwise_unchanged(trace):
    # Captured on the same native fixture before this implementation.
    expected = [3154556653] * 9 + [1088539630]
    for old, new, bits in zip(trace, trace[1:], expected):
        actual, terms = reward(old, new)
        assert int(np.asarray(actual).view(np.uint32)) == bits
        assert float(terms["reward_v2_transport_shaping"]) == 0
        assert float(terms["reward_v2_transport_phi"]) == 0
    # Golden per-agent values from the previous complete reward implementation,
    # covering loaded motion and terminal delivery (all 51 old fields compared
    # in the one-off reference probe, not only the direct R2 helper).
    for index, action, agent_bits in (
        (2, TrackedAction.forward(), 3150224512),
        (9, TrackedAction.do(), 1065842823),
    ):
        actual, terms = full_reward(trace[index], trace[index + 1], action)
        assert int(np.asarray(actual).view(np.uint32)) == expected[index]
        np.testing.assert_array_equal(np.asarray(terms["agent_rewards"]).view(np.uint32),
                                      [agent_bits, 0, 0, 0])
        for coef in (0.0, 1.0):
            total, components = full_reward(_coefficient(trace[index], coef),
                                             _coefficient(trace[index + 1], coef), action)
            debug_sum = (components["terminal"] + components["existence"]
                         + jnp.sum(components["agent_rewards"]))
            np.testing.assert_allclose(total, debug_sum, atol=1e-6)


def test_eligibility_and_invalid_coefficients_are_explicit(trace):
    loaded = trace[3]
    phi = float(potential(loaded))
    # An inactive skid or an active excavator cannot contribute to this term.
    for active, agent_type in ((False, 2), (True, 0)):
        extra = loaded.agent.agent_states[0]._replace(
            carry_relocation_credit=jnp.float32(999),
            agent_type=jnp.array([agent_type], dtype=jnp.int8),
        )
        state = loaded._set_agent_state_at(1, extra)
        state = state._replace(agent=state.agent._replace(
            agent_active=state.agent.agent_active.at[1].set(active)))
        np.testing.assert_allclose(potential(state), phi)
    for coef in (-1.0, float("nan"), float("inf")):
        value, terms = reward(_coefficient(loaded, coef), _coefficient(loaded, coef))
        assert np.isnan(float(value)) and float(terms["reward_v2_valid"]) == 0
    unsupported = _coefficient(loaded, 1.0)
    unsupported = unsupported._replace(env_cfg=unsupported.env_cfg._replace(
        reward_v2_timing_variant=1))
    value, terms = reward(unsupported, unsupported)
    assert np.isnan(float(value)) and float(terms["reward_v2_valid"]) == 0
