"""Unit receipts for the agent-neutral relocation reward contract."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.env import TerraEnvBatch
from terra.state import State

SEED = 20260730
SHAPE = (64, 64)


def _env_config(agent_types: tuple[int, ...] = (0,)) -> EnvConfig:
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
        maps=base.maps._replace(
            edge_length_px=int(np.asarray(updated.maps.edge_length_px)[0])
        ),
        agent_types=agent_types,
        action_types=tuple(0 for _ in agent_types),
        max_steps_in_episode=450,
    )


def _state(
    target: np.ndarray,
    *,
    action: np.ndarray | None = None,
    agent_types: tuple[int, ...] = (0,),
) -> State:
    if action is None:
        action = np.zeros(SHAPE, dtype=np.int8)
    distance = np.ones(SHAPE, dtype=np.float32)
    distance[target > 0] = 0.0
    return State.new(
        jax.random.PRNGKey(SEED),
        _env_config(agent_types),
        target,
        np.zeros(SHAPE, dtype=np.int8),
        -97.0 * np.ones((3, 3), dtype=np.float32),
        np.int32(-1),
        np.zeros(SHAPE, dtype=np.uint8),
        -97.0 * np.ones((SHAPE[0], 3), dtype=np.float32),
        np.int32(-1),
        np.ones(SHAPE, dtype=np.bool_),
        action,
        distance_map_override=distance,
    )


def test_fresh_progress_ignores_a_positive_pile_on_a_target_cell():
    target = np.zeros((3, 3), dtype=np.int8)
    target[1, 1] = -1

    pile = np.zeros_like(target)
    pile[1, 1] = 2
    cleared = pile.copy()
    cleared[1, 1] = 0
    assert float(State._get_action_map_dig_progress(pile, cleared, target)) == 0.0

    excavated = cleared.copy()
    excavated[1, 1] = -1
    assert float(State._get_action_map_dig_progress(cleared, excavated, target)) == 1.0

    overdug = excavated.copy()
    overdug[1, 1] = -2
    assert float(State._get_action_map_dig_progress(excavated, overdug, target)) == 0.0


def test_skid_mask_selects_positive_soil_and_never_a_negative_hole():
    target = np.zeros(SHAPE, dtype=np.int8)
    action = np.zeros(SHAPE, dtype=np.int8)
    action[20, 20] = 2
    action[20, 21] = -1
    state = _state(target, action=action, agent_types=(2,))
    candidate = np.zeros(SHAPE, dtype=np.bool_)
    candidate[20, 20:22] = True

    selected = np.asarray(
        state._mask_out_wrong_dig_tiles_skidsteer(candidate.reshape(-1))
    ).reshape(SHAPE)
    assert selected[20, 20]
    assert not selected[20, 21]


def test_relocation_progress_is_signed_and_not_capped():
    target = np.zeros(SHAPE, dtype=np.int8)
    target[0, 0] = 1
    state = _state(target)
    carrier = state._get_current_agent_state()._replace(
        loaded=jnp.array([1], dtype=jnp.int8),
        carry_relocation_credit=jnp.float32(-500.0),
    )
    state = state._set_current_agent_state(carrier)

    assert float(state._get_relocation_progress(state)) == pytest.approx(-500.0)

    deposited = np.asarray(state.world.action_map.map).copy()
    deposited[10, 10] = 1
    emptied = carrier._replace(
        loaded=jnp.zeros((1,), dtype=jnp.int8),
        carry_relocation_credit=jnp.float32(0.0),
    )
    after = state._replace(
        world=state.world._replace(
            action_map=state.world.action_map._replace(
                map=jnp.asarray(deposited),
            )
        )
    )._set_current_agent_state(emptied)
    progress = float(state._get_relocation_progress(after))
    reward = float(state._handle_rewards_dump(after, None))
    dig_tiles = float(np.sum(target < 0))
    scale = np.clip(170.0 / max(1.0, dig_tiles), 2.0, 5.0) / 2.0
    expected = (
        progress
        * state.env_cfg.relocation_progress_mult
        * scale
        * state.env_cfg.rewards.dump_correct
    )
    assert progress == pytest.approx(-501.0)
    assert reward == pytest.approx(expected)
    assert reward < -200.0


def test_env_config_exposes_only_one_relocation_multiplier():
    fields = set(EnvConfig._fields)
    assert "relocation_progress_mult" in fields
    assert "dump_bonus_mult" not in fields
    assert "excavator_relocate_dumped_mult" not in fields
    assert "excavator_relocate_dug_dirt_mult" not in fields
    assert "transport_relocate_mult" not in fields
