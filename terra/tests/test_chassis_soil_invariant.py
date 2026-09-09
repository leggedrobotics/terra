"""Occupied chassis cells are never material sources or destinations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.agent import Agent, AgentState
from terra.benchmark_state import validate_benchmark_initial_agent
from terra.config import EnvConfig
from terra.state import State


SHAPE = (64, 64)


def _state(*, action=None, target=None, position=(32, 32), base=0, cabin=0, loaded=0,
           neighbor=None, inactive_neighbor=None):
    cfg = EnvConfig()
    cfg = cfg._replace(
        tile_size=np.float32(4 / 7),
        agent=cfg.agent._replace(width=7, height=11, dig_depth=1),
        maps=cfg.maps._replace(edge_length_px=64),
        agent_types=(0,) if neighbor is None else (0, 1),
        action_types=(0,) if neighbor is None else (0, 0),
        foundation_dump_min_free_fraction=0.0,
    )
    zero = AgentState(
        pos_base=jnp.zeros(2, dtype=jnp.int16),
        angle_base=jnp.zeros(1, dtype=jnp.int8), angle_cabin=jnp.zeros(1, dtype=jnp.int8),
        wheel_angle=jnp.zeros(1, dtype=jnp.int8), loaded=jnp.zeros(1, dtype=jnp.int8),
        agent_type=jnp.zeros(1, dtype=jnp.int8), action_type=jnp.zeros(1, dtype=jnp.int8),
        shovel_lifted=jnp.zeros(1, dtype=jnp.int8), carry_relocation_credit=jnp.float32(0),
    )
    actor = zero._replace(pos_base=jnp.asarray(position, dtype=jnp.int16),
                          angle_base=jnp.asarray([base], dtype=jnp.int8),
                          angle_cabin=jnp.asarray([cabin], dtype=jnp.int8),
                          loaded=jnp.asarray([loaded], dtype=jnp.int8))
    other = zero if neighbor is None else zero._replace(
        pos_base=jnp.asarray(neighbor, dtype=jnp.int16), agent_type=jnp.asarray([1], dtype=jnp.int8))
    inactive = zero if inactive_neighbor is None else zero._replace(
        pos_base=jnp.asarray(inactive_neighbor, dtype=jnp.int16))
    agent = Agent(width=jnp.int32(7), height=jnp.int32(11),
                  agent_states=(actor, other, inactive, zero),
                  agent_active=jnp.asarray([1, neighbor is not None, 0, 0], dtype=jnp.int8),
                  num_agents=jnp.int32(1 + (neighbor is not None)), current_agent=jnp.int32(0))
    if action is None:
        action = np.zeros(SHAPE, dtype=np.int8)
    if target is None:
        target = np.zeros(SHAPE, dtype=np.int8)
    return State.new(
        jax.random.PRNGKey(7), cfg, jnp.asarray(target), jnp.zeros(SHAPE, dtype=jnp.int8),
        -97 * jnp.ones((3, 3), dtype=jnp.float32), jnp.int32(-1),
        -97 * jnp.ones((64, 3), dtype=jnp.float32), jnp.int32(-1),
        jnp.ones(SHAPE, dtype=jnp.bool_), jnp.asarray(action),
        distance_map_override=jnp.ones(SHAPE, dtype=jnp.float32), initial_agent=agent,
    )


def _mass(state):
    return int(np.asarray(state.world.action_map.map, dtype=np.int32).sum()) + sum(
        int(a.loaded[0]) for a in state.agent.agent_states)


def test_sparse_soil_blocks_move_and_turn_footprints_without_removing_soil():
    # Exact first ingress in recorded slot379: action53 backed from[48,56]
    # to[44,54] over the already present height-one cell[41,48].
    action = np.zeros(SHAPE, dtype=np.int8)
    action[41, 48] = 1
    state = _state(action=action, position=(48, 56), base=10)
    assert not bool(state._current_base_footprint_mask()[41, 48])
    candidate = state._get_agent_corners(jnp.asarray([44, 54]), jnp.asarray([10]), 7, 11)
    assert not bool(state._is_valid_move(candidate))
    after = jax.jit(lambda s: s._handle_move_backward())(state)
    assert not np.any(np.asarray(after._active_base_footprint_mask()) & (np.asarray(after.world.action_map.map) > 0))
    np.testing.assert_array_equal(after.world.action_map.map, action)

    clear = _state()
    rotated = clear._set_current_agent_state(clear._get_current_agent_state()._replace(
        angle_base=jnp.asarray([11], dtype=jnp.int8)))
    newly_covered = np.asarray(rotated._current_base_footprint_mask()) & ~np.asarray(clear._current_base_footprint_mask())
    cell = tuple(np.argwhere(newly_covered)[0])
    action = np.zeros(SHAPE, dtype=np.int8)
    action[cell] = 1
    state = _state(action=action)
    turned = jax.jit(lambda s: s._handle_clock())(state)
    assert int(turned._get_current_agent_state().angle_base[0]) == 0
    np.testing.assert_array_equal(turned.world.action_map.map, action)
    # Existing loaded-motion restriction also remains in force on empty ground.
    loaded = _state(loaded=3)
    moved = loaded._handle_move_forward()
    np.testing.assert_array_equal(moved._get_current_agent_state().pos_base, [32, 32])


def test_eligible_material_selection_protects_all_active_chassis_and_history():
    state = _state(neighbor=(32, 45), inactive_neighbor=(32, 39))
    cone = np.asarray(state._build_dig_dump_cone()).reshape(SHAPE)
    occupied = np.asarray(state._active_base_footprint_mask())
    neighbor_mask = np.asarray(state._agent_base_footprint_mask(state.agent.agent_states[1]))
    blocked = tuple(np.argwhere(cone & neighbor_mask)[0])
    free = np.argwhere(cone & ~occupied)
    fresh_cell, loose_cell = tuple(free[0]), tuple(free[-1])
    assert fresh_cell != loose_cell
    target = np.zeros(SHAPE, dtype=np.int8)
    target[fresh_cell] = target[blocked] = -1
    clear = _state(target=target, neighbor=(32, 45), inactive_neighbor=fresh_cell)
    assert bool(clear._agent_base_footprint_mask(clear.agent.agent_states[2])[fresh_cell])
    dug_clear = clear._handle_dig()
    assert int(dug_clear.world.action_map.map[fresh_cell]) == -1
    assert int(dug_clear.world.action_map.map[blocked]) == 0
    action = np.zeros(SHAPE, dtype=np.int8)
    action[blocked] = 1  # Deliberately invalid historical soil; never sanitize it.
    state = _state(action=action, target=target, neighbor=(32, 45), inactive_neighbor=fresh_cell)
    after = jax.jit(lambda s: s._handle_dig())(state)
    assert int(after.world.action_map.map[fresh_cell]) == -1
    assert int(after.world.action_map.map[blocked]) == 1
    assert int(after._get_current_agent_state().loaded[0]) == 1
    assert _mass(after) == _mass(state)
    counts = state._executable_fresh_dig_counts()
    assert int(counts[0]) == 1

    # An eligible loose pile still takes priority over fresh work.
    action = action.copy()
    action[loose_cell] = 2
    state = _state(action=action, target=target, neighbor=(32, 45))
    picked = state._handle_dig()
    assert int(picked._get_current_agent_state().loaded[0]) == 2
    assert int(picked.world.action_map.map[fresh_cell]) == 0
    assert int(picked.world.action_map.map[blocked]) == 1
    assert _mass(picked) == _mass(state)

    # Last-work-excluded loose material must not select loose mode either.
    last = np.zeros(SHAPE, dtype=bool)
    last[loose_cell] = True
    excluded = state._replace(world=state.world._replace(
        last_dig_mask=state.world.last_dig_mask._replace(map=jnp.asarray(last))))
    dug = excluded._handle_dig()
    assert int(dug._get_current_agent_state().loaded[0]) == 1
    assert int(dug.world.action_map.map[loose_cell]) == 2
    assert _mass(dug) == _mass(excluded)


def test_dump_and_relaxation_exclude_active_chassis_without_losing_material():
    state = _state(target=np.ones(SHAPE, dtype=np.int8), loaded=20, neighbor=(32, 45))
    occupied = np.asarray(state._active_base_footprint_mask())
    after = jax.jit(lambda s: s._handle_dump())(state)
    assert int(after._get_current_agent_state().loaded[0]) == 0
    assert not np.any(occupied & (np.asarray(after.world.action_map.map) > 0))
    assert _mass(after) == _mass(state)

    # A high pile adjacent to a chassis must not relax underneath it.
    expanded = np.asarray(state._dilate_mask(jnp.asarray(occupied)))
    adjacent = tuple(np.argwhere(expanded & ~occupied)[0])
    soil = np.zeros(SHAPE, dtype=np.int8)
    soil[adjacent] = 12
    affected = np.zeros(SHAPE, dtype=bool)
    affected[adjacent] = True
    relaxed = jax.jit(lambda s, m, a: s._apply_local_soil_mechanics(m, a))(
        state, jnp.asarray(soil), jnp.asarray(affected))
    assert int(relaxed.astype(jnp.int32).sum()) == 12
    assert not np.any(occupied & (np.asarray(relaxed) > 0))

    # With only occupied dump cells available, retain the complete load.
    blocked = state._replace(world=state.world._replace(
        dumpability_mask=state.world.dumpability_mask._replace(map=jnp.asarray(occupied))))
    rejected = blocked._handle_dump()
    assert int(rejected._get_current_agent_state().loaded[0]) == 20
    np.testing.assert_array_equal(rejected.world.action_map.map, state.world.action_map.map)


def test_random_and_validated_benchmark_resets_exclude_existing_soil():
    action = np.zeros(SHAPE, dtype=np.int8)
    action[28:36, 28:36] = 1
    action[12:20, 12:20] = -1
    template = _state(action=action, neighbor=(32, 45))
    cfg = template.env_cfg
    def reset(key):
        return Agent.new(key, cfg, 64, 64, jnp.zeros(SHAPE, dtype=jnp.int8),
                         jnp.asarray(action), jnp.ones(SHAPE, dtype=jnp.bool_),
                         agent_types=(0, 1), action_types=(0, 0))[0]
    agents = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(3), 8))
    for i in range(8):
        agent = jax.tree_util.tree_map(lambda x: x[i], agents)
        state = template._replace(agent=agent)
        assert not np.any(np.asarray(state._active_base_footprint_mask()) & (action != 0))

    valid = _state()
    kwargs = dict(env_cfg=valid.env_cfg, padding_mask=np.zeros(SHAPE, dtype=np.int8),
                  action_map=np.zeros(SHAPE, dtype=np.int8), dumpability_mask=np.ones(SHAPE, dtype=bool))
    validate_benchmark_initial_agent(valid.agent, **kwargs)
    kwargs["action_map"][32, 32] = 1
    with pytest.raises(ValueError, match="existing soil work"):
        validate_benchmark_initial_agent(valid.agent, **kwargs)
