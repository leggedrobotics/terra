"""Optional transport shaping on a real pickup/backtrack/delivery trace."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.config import REWARD_V2_POTENTIAL_GAMMA, RewardStage
from terra.actions import TrackedAction
from terra.env import TerraEnv
from terra.state import State
from terra.tests.test_skid_steer_relocation import (
    SHAPE, _bucket, _mass, _skid, _zone,
)


reward = jax.jit(lambda old, new: old._get_reward_v2(
    new, *new._is_done(new.world.action_map.map, new.world.target_map.map)
))
potential = jax.jit(lambda state: state._transport_credit_potential())
full_reward = jax.jit(lambda old, new, action: old._get_reward(new, action))
# Exercise the exact native handlers plus the bookkeeping wrapper selected by
# _apply_action, without compiling unrelated actions for every trace branch.
forward = jax.jit(lambda s: s._update_transport_origin(s._handle_move_forward()))
backward = jax.jit(lambda s: s._update_transport_origin(s._handle_move_backward()))
do = jax.jit(lambda s: s._update_transport_origin(s._handle_do()))
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


def test_pickup_starts_at_zero_transport_potential_and_keeps_handling_cost(trace):
    before, after = trace[:2]
    agent = after.agent.agent_states[0]
    distance = np.asarray(after.world.relocation_distance_map)[tuple(agent.pos_base)]
    # Regression: ground-source distance is smaller than chassis distance,
    # which previously imposed a spurious negative transport term on pickup.
    assert float(agent.carry_relocation_credit) - int(agent.loaded[0]) * distance < 0
    np.testing.assert_allclose(after.carry_transport_origin[0], 40 * distance)
    assert float(potential(before)) == 0 and float(potential(after)) == 0
    cfg = before.env_cfg._replace(transport_credit_coef=1.0, material_handling_cost=0.25)
    total, components = full_reward(before._replace(env_cfg=cfg),
                                    after._replace(env_cfg=cfg), TrackedAction.forward())
    assert float(components["reward_v2_transport_shaping"]) == 0
    assert float(components["reward_v2_material_handling"]) == -0.25
    assert float(total) < 0  # Handling and the ordinary step cost remain.
    # Check that the public action dispatch actually applies the ledger once.
    dispatched = jax.jit(lambda s: s._apply_action(TrackedAction.forward().action))(before)
    dispatched = dispatched._replace(env_steps=after.env_steps)
    for actual, expected in zip(jax.tree_util.tree_leaves(dispatched),
                                jax.tree_util.tree_leaves(after)):
        np.testing.assert_array_equal(actual, expected)


def test_partial_load_top_up_does_not_erase_existing_transport_progress():
    soil = np.zeros(SHAPE, dtype=np.int8)
    soil[31:34, 24:27] = 1
    soil[31:34, 34:37] = 1
    state = _skid(_zone(), soil)
    states = [state]
    # Pick up 9, raise, advance twice, lower after an unsuccessful off-zone
    # dump. The next FORWARD is blocked by pile 2 and tops up in place.
    for transition in (forward, do, forward, forward, do, forward):
        state = transition(state)._replace(env_steps=state.env_steps + 1)
        states.append(state)
    before, topped_up = states[-2:]
    assert all(_mass(s) == 18 for s in states)
    assert int(before.agent.agent_states[0].loaded[0]) == 9
    assert int(topped_up.agent.agent_states[0].loaded[0]) == 18
    np.testing.assert_array_equal(before.agent.agent_states[0].pos_base,
                                  topped_up.agent.agent_states[0].pos_base)
    phi = float(potential(before))
    assert phi > 0
    np.testing.assert_allclose(potential(topped_up), phi, atol=1e-7)
    _, terms = reward(_coefficient(before, 1), _coefficient(topped_up, 1))
    np.testing.assert_allclose(terms["reward_v2_transport_shaping"],
                               (GAMMA - 1) * phi, atol=1e-7)
    assert float(terms["reward_v2_valid"]) == 1
    # Existing baseline material work remains continuous across the top-up.
    np.testing.assert_allclose(before._compute_material_work(),
                               topped_up._compute_material_work(), atol=1e-6)
    toward = forward(topped_up)
    assert float(potential(toward)) > float(potential(topped_up))

    # Native skid dumps currently unload the whole bucket. The central ledger
    # also apportions an eventual partial unload by its retained mass fraction.
    partial = topped_up._set_agent_state_at(
        0, topped_up.agent.agent_states[0]._replace(loaded=jnp.array([9], jnp.int8)))
    partial = topped_up._update_transport_origin(partial)
    np.testing.assert_allclose(partial.carry_transport_origin[0],
                               topped_up.carry_transport_origin[0] / 2)


def test_prepared_loaded_reset_rebases_only_the_transport_origin(trace):
    loaded = trace[3]
    assert float(potential(loaded)) > 0
    reset = State.new(
        loaded.key, loaded.env_cfg, loaded.world.target_map.map,
        jnp.zeros(SHAPE, jnp.int8), -97 * jnp.ones((3, 3), jnp.float32), jnp.int32(-1),
        -97 * jnp.ones((64, 3), jnp.float32), jnp.int32(-1),
        jnp.ones(SHAPE, jnp.bool_), loaded.world.action_map.map,
        distance_map_override=loaded.world.relocation_distance_map,
        initial_agent=loaded.agent,
    )
    np.testing.assert_allclose(potential(reset), 0, atol=1e-7)
    assert float(reset.carry_transport_origin[0]) > 0
    for actual, expected in zip(jax.tree_util.tree_leaves(reset.agent),
                                jax.tree_util.tree_leaves(loaded.agent)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(reset._compute_material_work(), loaded._compute_material_work())


def test_loaded_motion_is_dense_and_does_not_change_stored_credit(trace):
    old, toward, back = trace[2:5]
    credit = float(old.agent.agent_states[0].carry_relocation_credit)
    for state in trace:
        assert _mass(state) == 40
    for state in (toward, back):
        assert float(state.agent.agent_states[0].carry_relocation_credit) == credit
        np.testing.assert_array_equal(state.carry_transport_origin, old.carry_transport_origin)
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
    np.testing.assert_array_equal(trace[-1].carry_transport_origin, np.zeros(4))
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
