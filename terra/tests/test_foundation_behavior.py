"""Optional R2 chassis costs and executable fresh-work observations."""

import pickle
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.actions import TrackedAction
from terra.config import EnvConfig, RewardStage
from terra.env import TerraEnv
from terra.settings import INTLOWDIM_MAX, IntMap
from terra.state import State
from terra.tests import test_admissible_dig_local_map as admissible_fixtures
from terra.tests.test_reward_v2_contract import _env_config
from terra.wrappers import LocalMapWrapper


@pytest.fixture(scope="module")
def foundation():
    target = np.zeros((64, 64), dtype=np.int8)
    target[14:50, 14:50] = -1
    target[4:10, 4:10] = 1
    state = State.new(
        jax.random.PRNGKey(7), _env_config(), target,
        np.zeros_like(target), -97.0 * np.ones((4, 8), dtype=np.float32),
        np.int32(-1), -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1), np.ones_like(target, dtype=np.bool_), np.zeros_like(target),
        distance_map_override=np.ones_like(target, dtype=np.float32),
    )
    return _pose(state, base=0, cabin=0, position=(32, 32), loaded=0)


def _pose(state, *, base=None, cabin=None, position=None, loaded=None):
    updates = {}
    for name, value in (("angle_base", base), ("angle_cabin", cabin), ("loaded", loaded)):
        if value is not None:
            updates[name] = jnp.asarray([value], dtype=jnp.int8)
    if position is not None:
        updates["pos_base"] = jnp.asarray(position, dtype=jnp.int16)
    return state._set_current_agent_state(state._get_current_agent_state()._replace(**updates))


def _map(state, field, values):
    current = getattr(state.world, field)
    return state._replace(world=state.world._replace(**{
        field: current._replace(map=jnp.asarray(values, dtype=current.map.dtype)),
    }))


def _costs(state):
    return state._replace(env_cfg=state.env_cfg._replace(
        lateral_dig_cost=0.6, base_travel_cost=0.07, base_turn_cost=0.11,
    ))


@jax.jit
def _do(state):
    return state._handle_do()


@jax.jit
def _counts(state):
    return LocalMapWrapper.wrap(
        state, executable_dig_observation=True,
    ).world.local_map_admissible_dig.map


@jax.jit
def _behavior(before, after):
    return before._reward_v2_behavior_costs(after)


def _fresh_volume(before, after):
    required = np.maximum(-np.asarray(before.world.target_map.map, dtype=np.float32), 0)
    old = np.minimum(np.maximum(-np.asarray(before.world.action_map.map, dtype=np.float32), 0), required)
    new = np.minimum(np.maximum(-np.asarray(after.world.action_map.map, dtype=np.float32), 0), required)
    return float(np.maximum(new - old, 0).sum())


def _assert_all_headings_match_do(state):
    observed = np.asarray(_counts(state))
    assert observed.shape == (12,)
    assert observed.dtype == IntMap
    cabin = int(state._get_current_agent_state().angle_cabin[0])
    for offset in range(12):
        candidate = _pose(state, cabin=(cabin + offset) % 12)
        result = _do(candidate)
        assert int(observed[offset]) == _fresh_volume(candidate, result), offset
    return observed


@pytest.mark.parametrize("base,cabin,factor", [(0, 0, 0), (2, 1, .25), (2, 3, 1), (7, 3, 1), (0, 6, 0), (0, 9, 1)])
def test_lateral_cost_uses_actual_fresh_volume_and_relative_cabin(foundation, base, cabin, factor):
    before = _costs(_pose(foundation, base=base, cabin=cabin))
    after = _do(before)
    volume = _fresh_volume(before, after)
    assert volume > 0
    components = _behavior(before, after)
    assert float(components["reward_v2_fresh_dig_volume"]) == volume
    required = float(before._required_excavation_volume())
    np.testing.assert_allclose(components["reward_v2_lateral_dig"], -.6 * volume / required * factor, atol=2e-8)
    assert float(components["reward_v2_base_travel"]) == 0
    assert float(components["reward_v2_base_turn"]) == 0


def test_dump_relift_cabin_swing_and_failed_do_have_no_lateral_cost(foundation):
    side = _costs(_pose(foundation, cabin=3))
    cone = np.asarray(side._build_dig_dump_cone()).reshape((64, 64))
    coordinate = tuple(np.argwhere(cone)[len(np.argwhere(cone)) // 2])

    piles = np.zeros((64, 64), dtype=np.int8)
    piles[coordinate] = 3
    relift = _map(side, "action_map", piles)
    relifted = _do(relift)
    assert int(relifted._get_current_agent_state().loaded[0]) == 3

    target = np.asarray(side.world.target_map.map).copy()
    target[cone] = 1
    dumping = _pose(_map(side, "target_map", target), loaded=8)
    dumped = _do(dumping)
    assert int(dumped._get_current_agent_state().loaded[0]) == 0

    obstacles = np.zeros((64, 64), dtype=np.int8)
    obstacles[coordinate] = 1
    blocked = _map(side, "padding_mask", obstacles)
    failed = _do(blocked)
    assert int(failed._get_current_agent_state().loaded[0]) == 0

    swung = dumping._handle_cabin_clock()
    assert int(swung._get_current_agent_state().angle_cabin[0]) != 3
    for before, after in ((relift, relifted), (dumping, dumped), (blocked, failed), (dumping, swung)):
        components = _behavior(before, after)
        assert float(components["reward_v2_fresh_dig_volume"]) == 0
        for key in ("reward_v2_lateral_dig", "reward_v2_base_travel", "reward_v2_base_turn"):
            assert float(components[key]) == 0


def test_base_costs_use_executed_distance_wrapped_yaw_and_stable_slot(foundation):
    for base in (0, 1):
        before = _costs(_pose(foundation, base=base))
        after = before._handle_move_forward()
        delta = np.asarray(after._get_current_agent_state().pos_base) - np.asarray(before._get_current_agent_state().pos_base)
        metres = float(np.linalg.norm(delta) * before.env_cfg.tile_size)
        assert metres > 0
        components = _behavior(before, after)
        np.testing.assert_allclose(components["reward_v2_base_travel_m"], metres, atol=1e-6)
        np.testing.assert_allclose(components["reward_v2_base_travel"], -.07 * metres, atol=1e-6)
        # Actor handoff must not compare different agents' unrelated positions.
        handed_off = after._replace(agent=after.agent._replace(current_agent=jnp.int32(1)))
        np.testing.assert_array_equal(_behavior(before, handed_off)["reward_v2_base_travel"], components["reward_v2_base_travel"])

    before = _costs(_pose(foundation, base=11, cabin=3))
    turned = before._handle_anticlock()
    assert int(turned._get_current_agent_state().angle_base[0]) == 0
    components = _behavior(before, turned)
    np.testing.assert_allclose(components["reward_v2_base_turn_rad"], np.pi / 6, atol=1e-7)
    np.testing.assert_allclose(components["reward_v2_base_turn"], -.11 * np.pi / 6, atol=1e-7)
    assert float(components["reward_v2_base_travel"]) == 0

    loaded = _pose(before, loaded=3)
    for after in (loaded._handle_move_forward(), loaded._handle_anticlock()):
        components = _behavior(loaded, after)
        assert float(components["reward_v2_base_travel"]) == 0
        assert float(components["reward_v2_base_turn"]) == 0


@pytest.mark.parametrize("timing", [0, 1])
def test_zero_cost_reward_is_bitwise_frozen_and_components_reconstruct(foundation, timing):
    before = foundation._replace(env_cfg=foundation.env_cfg._replace(reward_v2_timing_variant=timing))
    after = _do(_pose(before, cabin=3))
    before = _pose(before, cabin=3)
    reward, components = before._get_reward_v2(after, jnp.bool_(False), jnp.bool_(False))
    frozen = components["reward_v2_success"] + components["reward_v2_horizon_failure"] + components["reward_v2_step"] + components["reward_v2_shaping"]
    np.testing.assert_array_equal(np.asarray(reward).view(np.uint32), np.asarray(frozen).view(np.uint32))
    charged_before = _costs(before)
    charged_after = after._replace(env_cfg=charged_before.env_cfg)
    total, integrated = charged_before._get_reward(charged_after, TrackedAction.do())
    reconstructed = integrated["agent_rewards"].sum() + integrated["terminal"] + integrated["existence"]
    np.testing.assert_allclose(total, reconstructed, atol=1e-7)
    assert float(total) < float(reward)
    zeros = TerraEnv._zero_reward_components(before)
    assert jax.tree_util.tree_structure(zeros) == jax.tree_util.tree_structure(integrated)
    for key in zeros:
        assert np.asarray(zeros[key]).shape == np.asarray(integrated[key]).shape, key
        assert np.asarray(zeros[key]).dtype == np.asarray(integrated[key]).dtype, key
    legacy_before = charged_before._replace(env_cfg=charged_before.env_cfg._replace(reward_stage=RewardStage.DENSE_SKILL))
    legacy_after = charged_after._replace(env_cfg=legacy_before.env_cfg)
    _, legacy = legacy_before._get_reward(legacy_after, TrackedAction.do())
    for key in ("reward_v2_lateral_dig", "reward_v2_base_travel", "reward_v2_base_turn"):
        assert float(legacy[key]) == 0


@pytest.mark.parametrize("case", ["fresh", "obstacle", "pile", "excluded_pile", "last_dig", "loaded", "capacity", "depth"])
def test_executable_affordance_matches_all_do_headings(foundation, case):
    state = foundation
    cone = np.asarray(state._build_dig_dump_cone()).reshape((64, 64))
    coordinate = tuple(np.argwhere(cone)[len(np.argwhere(cone)) // 2])
    cells = np.zeros((64, 64), dtype=np.int8)
    cells[coordinate] = 1
    if case == "obstacle":
        state = _map(state, "padding_mask", cells)
    elif case in ("pile", "excluded_pile"):
        state = _map(state, "action_map", cells * 3)
        if case == "excluded_pile":
            state = _map(state, "last_dig_mask", cells)
    elif case == "last_dig":
        state = _map(state, "last_dig_mask", cells)
    elif case == "loaded":
        state = _pose(state, loaded=3)
    elif case == "capacity":
        state = state._replace(env_cfg=state.env_cfg._replace(agent=state.env_cfg.agent._replace(dig_radius_tiles=15)))
    elif case == "depth":
        state = _map(state, "target_map", np.asarray(state.world.target_map.map) * 2)
        state = state._replace(env_cfg=state.env_cfg._replace(agent=state.env_cfg.agent._replace(dig_depth=2)))
    observed = _assert_all_headings_match_do(state)
    # Excluded loose soil no longer suppresses eligible fresh digging.
    if case in ("obstacle", "pile", "loaded", "capacity"):
        assert observed[0] == 0
    else:
        assert observed[0] > 0
    if case == "loaded":
        old = LocalMapWrapper.wrap(state, executable_dig_observation=False).world.local_map_admissible_dig.map
        assert np.asarray(old).sum() > 0


def test_affordance_excludes_rotated_footprint_and_foundation_border(foundation):
    target = np.zeros((64, 64), dtype=np.int8)
    target[28, 25] = target[29, 26] = -1
    state = _pose(_map(foundation, "target_map", target), base=7, cabin=1, position=(24, 30))
    cone = np.asarray(state._build_dig_dump_cone()).reshape((64, 64))
    assert cone[28, 25] and cone[29, 26]
    assert int(_assert_all_headings_match_do(state).sum()) == 0

    target[:] = 0
    target[24, 20:50] = -1
    state = _pose(_map(foundation, "target_map", target), base=0, cabin=0, position=(26, 32))
    axes = -97.0 * np.ones((64, 3), dtype=np.float32)
    axes[0] = [0, 1, -24]
    state = state._replace(
        env_cfg=state.env_cfg._replace(enforce_foundation_border_alignment=True),
        world=state.world._replace(foundation_border_axes=jnp.asarray(axes), foundation_border_type=jnp.int32(1)),
    )
    admitted = _assert_all_headings_match_do(state)
    ungated = _counts(state._replace(env_cfg=state.env_cfg._replace(enforce_foundation_border_alignment=False)))
    assert admitted.sum() > 0
    assert np.any(admitted < np.asarray(ungated))


def test_executable_junction_keeps_both_owning_approaches():
    fixture = admissible_fixtures.AdmissibleDigLocalMapTest
    fixture.setUpClass()
    target, axes, shared, horizontal, vertical = fixture._junction()
    for base, position, allowed, excluded in ((0, (24, 32), horizontal, vertical), (3, (34, 40), vertical, horizontal)):
        state = fixture._state(target, axes, base_angle=base, cabin_angle=0, position=position)
        counts = _assert_all_headings_match_do(state)
        assert counts[0] == 2
        dug = np.asarray(_do(state).world.action_map.map) < 0
        assert dug[shared] and dug[allowed] and not dug[excluded]


def test_eligibility_retains_legacy_cone_scoped_selection(foundation):
    fixture = admissible_fixtures.AdmissibleDigLocalMapTest
    fixture.setUpClass()
    target, axes, _, _, _ = fixture._junction()
    junction = fixture._state(target, axes, base_angle=0, cabin_angle=0, position=(24, 32))
    for base in (foundation, junction):
        for angle in range(12):
            state = _pose(base, cabin=angle)
            cone = state._build_dig_dump_cone()
            old_mask = state._mask_out_wrong_dig_tiles(cone)
            if state.env_cfg.enforce_trench_dig_alignment:
                old_mask = state._get_fresh_trench_dig_alignment_details(old_mask)[3]
            selected = int((np.asarray(state.world.action_map.map).reshape(-1).astype(np.int32) * np.asarray(old_mask)).sum())
            old_volume = min(selected, INTLOWDIM_MAX) if selected > 0 else int(np.asarray(old_mask).sum())
            old_admitted = not bool(state._workspace_intersects_obstacle()) and 0 < old_volume <= INTLOWDIM_MAX
            mask, volume, relift, admitted = state._dig_eligibility(cone)
            np.testing.assert_array_equal(mask, old_mask)
            assert int(volume) == old_volume
            assert bool(relift) == (selected > 0)
            assert bool(admitted) == old_admitted


def test_static_false_batch_never_traces_executable_branch(foundation):
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([jnp.asarray(x)] * 2), foundation)
    with mock.patch.object(State, "_executable_fresh_dig_counts", side_effect=AssertionError("disabled branch traced")):
        jax.make_jaxpr(jax.vmap(lambda state: TerraEnv.wrap_state(
            state, executable_dig_observation=False,
        ).world.local_map_admissible_dig.map))(batched)


def test_previous_config_pickle_arguments_keep_new_defaults():
    class PreviousConfig:
        def __reduce__(self):
            return EnvConfig, tuple(EnvConfig())[:-4]

    restored = pickle.loads(pickle.dumps(PreviousConfig()))
    assert isinstance(restored, EnvConfig)
    assert tuple(restored)[:-4] == tuple(EnvConfig())[:-4]
    assert (restored.lateral_dig_cost, restored.base_travel_cost, restored.base_turn_cost) == (0, 0, 0)
    assert restored.executable_dig_observation is False
