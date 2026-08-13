from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.env import TerraEnv
from terra.env import TerraEnvBatch
from terra.state import State


SHAPE = (64, 64)


def _state() -> State:
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
    cfg = base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(edge_length_px=SHAPE[0]),
        agent_types=(0,),
        action_types=(0,),
    )
    return State.new(
        jax.random.PRNGKey(20260813),
        cfg,
        np.zeros(SHAPE, dtype=np.int8),
        np.zeros(SHAPE, dtype=np.int8),
        -97.0 * np.ones((4, 3), dtype=np.float32),
        np.int32(-1),
        -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1),
        np.ones(SHAPE, dtype=np.bool_),
        np.zeros(SHAPE, dtype=np.int8),
        distance_map_override=np.ones(SHAPE, dtype=np.float32),
    )


def test_stall_age_counts_no_material_change_and_is_normalized():
    state = _state()
    observation = TerraEnv._state_to_obs_dict(state)
    assert observation["stall_age"].shape == ()
    assert observation["stall_age"].dtype == jnp.float32
    assert float(observation["stall_age"]) == 0.0

    for _ in range(40):
        state = state._step(TrackedAction.do_nothing())

    assert int(state.stall_age_steps) == 32
    assert float(TerraEnv._state_to_obs_dict(state)["stall_age"]) == 1.0


def test_only_active_material_changes_reset_stall_age():
    state = _state()._replace(stall_age_steps=jnp.int32(7))

    moved_actor = state.agent.agent_states[0]._replace(
        pos_base=state.agent.agent_states[0].pos_base
        + jnp.array([1, 0], dtype=state.agent.agent_states[0].pos_base.dtype)
    )
    moved = state._set_agent_state_at(0, moved_actor)
    assert int(state._next_stall_age_steps(moved)) == 8

    changed_map = state.world.action_map.map.at[0, 0].set(1)
    changed_soil = state._replace(
        world=state.world._replace(
            action_map=state.world.action_map._replace(map=changed_map)
        )
    )
    assert int(state._next_stall_age_steps(changed_soil)) == 0

    loaded_actor = state.agent.agent_states[0]._replace(
        loaded=jnp.array([1], dtype=jnp.int8)
    )
    loaded = state._set_agent_state_at(0, loaded_actor)
    assert int(state._next_stall_age_steps(loaded)) == 0

    credited_actor = state.agent.agent_states[0]._replace(
        carry_relocation_credit=jnp.float32(0.25)
    )
    credited = state._set_agent_state_at(0, credited_actor)
    assert int(state._next_stall_age_steps(credited)) == 0

    inactive = state.agent.agent_states[1]._replace(
        loaded=jnp.array([1], dtype=jnp.int8),
        carry_relocation_credit=jnp.float32(0.25),
    )
    inactive_changed = state._set_agent_state_at(1, inactive)
    assert int(state._next_stall_age_steps(inactive_changed)) == 8
