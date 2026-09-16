"""Deployment work-pose charges and optional pose-opportunity diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import REWARD_V2_POTENTIAL_GAMMA
from terra.tests.test_foundation_behavior import foundation, _do, _map, _pose
from terra.tests.test_reward_v2_contract import _state, _with_material


def _charged(state):
    return state._replace(env_cfg=state.env_cfg._replace(
        retained_work_setup_cost=0.2,
        retained_work_travel_cost=0.03,
        retained_work_turn_cost=0.04,
    ))


def _loose(state):
    cone = np.asarray(state._build_dig_dump_cone()).reshape((64, 64))
    position = tuple(np.argwhere(cone)[len(np.argwhere(cone)) // 2])
    piles = np.zeros((64, 64), dtype=np.int8)
    piles[position] = 3
    state = _map(state, "action_map", piles)
    return _map(state, "last_dig_mask", np.zeros_like(piles))


@jax.jit
def _retained(before, after):
    return before._reward_v2_retained_work_costs(after)


def test_only_effective_do_records_fresh_dump_and_relift(foundation):
    state = _charged(foundation)
    cone = np.asarray(state._build_dig_dump_cone()).reshape((64, 64))
    target = np.asarray(state.world.target_map.map).copy()
    target[cone] = 1
    dumping = _pose(_map(state, "target_map", target), loaded=8)
    loose = _loose(state)
    for before in (state, dumping, loose):
        after = _do(before)
        assert int(after.retained_work_events[0]) == 1
        parts = _retained(before, after)
        assert float(parts["reward_v2_retained_work_event"]) == 1
        np.testing.assert_allclose(parts["reward_v2_retained_setup"], -0.2)
        assert float(parts["reward_v2_retained_inter_setup_m"]) == 0
        assert float(parts["reward_v2_retained_heading_rad"]) == 0

    obstacle = np.zeros((64, 64), dtype=np.int8)
    obstacle[tuple(np.argwhere(cone)[0])] = 1
    blocked = _map(state, "padding_mask", obstacle)
    for before, after in (
        (blocked, _do(blocked)),
        (state, state._handle_cabin_clock()),
        (state, state._handle_move_forward()),
    ):
        assert int(after.retained_work_events[0]) == 0
        assert all(float(v) == 0 for v in _retained(before, after).values())


def test_grouping_ignores_discarded_navigation_but_retains_work_transfers(foundation):
    first = _do(_charged(foundation))
    # A new effective relift at the same work pose remains one setup, even
    # after navigation/cabin changes. Only retained poses enter the projection.
    away = _pose(first, position=(35, 36), base=11, cabin=2, loaded=0)
    returned = _loose(_pose(away, position=(32, 32), base=0, cabin=3))
    repeated = _do(returned)
    parts = _retained(returned, repeated)
    assert int(repeated.retained_work_events[0]) == 2
    assert float(parts["reward_v2_retained_work_event"]) == 1
    assert float(parts["reward_v2_retained_new_setup"]) == 0
    assert float(parts["reward_v2_retained_setup"]) == 0

    transfer = _loose(_pose(repeated, position=(35, 36), base=11, cabin=0, loaded=0))
    worked = _do(transfer)
    parts = _retained(transfer, worked)
    distance = 5 * foundation.env_cfg.tile_size
    np.testing.assert_allclose(parts["reward_v2_retained_inter_setup_m"], distance, atol=1e-6)
    np.testing.assert_allclose(parts["reward_v2_retained_heading_rad"], np.pi / 6, atol=1e-7)
    np.testing.assert_allclose(parts["reward_v2_retained_travel"], -0.03 * distance, atol=1e-7)
    np.testing.assert_allclose(parts["reward_v2_retained_turn"], -0.04 * np.pi / 6, atol=1e-7)
    assert float(parts["reward_v2_retained_new_setup"]) == 1
    # Agent handoff must not read the next actor's unrelated retained pose.
    handed_off = worked._replace(agent=worked.agent._replace(current_agent=jnp.int32(1)))
    for key, value in _retained(transfer, handed_off).items():
        np.testing.assert_array_equal(value, parts[key])
    reset = worked._reset(
        worked.env_cfg, worked.world.target_map.map, worked.world.padding_mask.map,
        worked.world.trench_axes, worked.world.trench_type,
        worked.world.foundation_border_axes, worked.world.foundation_border_type,
        worked.world.dumpability_mask_init.map, jnp.zeros_like(worked.world.action_map.map),
        distance_map_override=worked.world.relocation_distance_map,
    )
    np.testing.assert_array_equal(reset.retained_work_events, np.zeros(4))


def test_optional_fresh_union_matches_actual_do_without_counting_overlap(foundation):
    observed = np.asarray(jax.jit(type(foundation)._executable_fresh_dig_union)(foundation))
    masks = []
    for heading in range(12):
        candidate = _pose(foundation, cabin=heading)
        after = _do(candidate)
        masks.append(np.asarray(candidate._get_fresh_target_excavation_map(
            candidate.world.action_map.map, after.world.action_map.map,
            candidate.world.target_map.map,
        )) > 0)
    np.testing.assert_array_equal(observed, np.any(masks, axis=0))
    assert 0 < observed.sum() < np.sum(masks)
    loaded = _pose(foundation, loaded=3)
    assert not np.asarray(loaded._executable_fresh_dig_union()).any()


def test_discounted_reward_retains_partial_progress_at_failure_boundary():
    initial = _state()._replace(env_steps=448)
    action = np.zeros((64, 64), dtype=np.int8)
    action[20, 20] = -1
    carrying = _with_material(initial, action, loaded=1, carry_work=1.0, env_steps=449)
    action[40, 40] = 1
    terminal = _with_material(carrying, action, loaded=0, carry_work=0.0, env_steps=450)
    first, first_parts = initial._get_reward_v2(carrying, jnp.bool_(False), jnp.bool_(False))
    last, last_parts = carrying._get_reward_v2(terminal, jnp.bool_(True), jnp.bool_(False))
    gamma = REWARD_V2_POTENTIAL_GAMMA
    actual_return = float(first) + gamma * float(last)
    boundary = -float(first_parts["reward_v2_phi"]) + gamma ** 2 * float(last_parts["reward_v2_phi_next"])
    explicit = sum(
        gamma ** t * float(parts["reward_v2_step"] + parts["reward_v2_horizon_failure"])
        for t, parts in enumerate((first_parts, last_parts))
    )
    np.testing.assert_allclose(actual_return, boundary + explicit, atol=1e-6)
    assert float(last_parts["reward_v2_phi_next"]) > 0
    assert float(last_parts["reward_v2_horizon_failure"]) == -1
