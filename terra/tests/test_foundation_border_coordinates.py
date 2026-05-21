import sys

import jax
import jax.numpy as jnp
import numpy as np

from terra.agent import Agent, AgentState
from terra.config import AgentConfig, EnvConfig
from terra.map import GridWorld
from terra.settings import IntLowDim, IntMap
from terra.state import State


def _agent_state(pos_row_col: tuple[int, int], cabin_angle_idx: int = 0) -> AgentState:
    return AgentState(
        pos_base=jnp.array(pos_row_col, dtype=IntMap),
        angle_base=jnp.array([0], dtype=IntLowDim),
        angle_cabin=jnp.array([cabin_angle_idx], dtype=IntLowDim),
        wheel_angle=jnp.array([0], dtype=IntLowDim),
        loaded=jnp.array([0], dtype=IntLowDim),
        agent_type=jnp.array([0], dtype=IntLowDim),
        action_type=jnp.array([0], dtype=IntLowDim),
        shovel_lifted=jnp.array([0], dtype=IntLowDim),
    )


def _state_at(pos_row_col: tuple[int, int], cabin_angle_idx: int = 0) -> State:
    map_size = 32
    agent_cfg = AgentConfig(width=3, height=3, move_tiles=3)
    env_cfg = EnvConfig()._replace(
        agent=agent_cfg,
        maps=EnvConfig().maps._replace(edge_length_px=map_size),
        tile_size=1.0,
        max_steps_in_episode=16,
        foundation_border_width_tiles=1,
        foundation_border_proximity_tiles=0.75,
        foundation_border_hv_tolerance_rad=0.01,
        foundation_border_diag_tolerance_rad=0.01,
    )
    target_map = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    target_map = target_map.at[10:15, 4:14].set(jnp.int8(-1))
    foundation_border_axes = jnp.full((64, 3), -97.0, dtype=jnp.float32)
    # Metadata line y(row)=10, represented as A*x(col) + B*y(row) + C = 0.
    foundation_border_axes = foundation_border_axes.at[0].set(
        jnp.array([0.0, 1.0, -10.0], dtype=jnp.float32)
    )
    world = GridWorld.new(
        target_map=target_map,
        padding_mask=jnp.zeros((map_size, map_size), dtype=jnp.int8),
        trench_axes=jnp.zeros((3, 3), dtype=jnp.float32),
        trench_type=jnp.int32(0),
        foundation_border_axes=foundation_border_axes,
        foundation_border_type=jnp.int32(1),
        dumpability_mask_init=jnp.ones((map_size, map_size), dtype=jnp.bool_),
        action_map=jnp.zeros((map_size, map_size), dtype=jnp.int8),
        relocation_distance_map_override=jnp.zeros(
            (map_size, map_size), dtype=jnp.float32
        ),
    )
    dummy = _agent_state((2, 2))
    return State(
        key=jax.random.PRNGKey(0),
        env_cfg=env_cfg,
        world=world,
        agent=Agent(
            width=agent_cfg.width,
            height=agent_cfg.height,
            moving_dumped_dirt=False,
            agent_states=(
                _agent_state(pos_row_col, cabin_angle_idx),
                dummy,
                dummy,
                dummy,
            ),
            agent_active=jnp.array([1, 0, 0, 0], dtype=jnp.int8),
            num_agents=1,
            current_agent=jnp.int32(0),
        ),
        env_steps=0,
    )


def _border_tile_allowed(
    pos_row_col: tuple[int, int], cabin_angle_idx: int = 0
) -> bool:
    workspace = jnp.zeros((32, 32), dtype=jnp.bool_)
    workspace = workspace.at[10, 5].set(True).reshape(-1)
    state = _state_at(pos_row_col, cabin_angle_idx)
    mask = state._get_foundation_border_alignment_mask(workspace)
    return bool(np.asarray(mask).reshape(32, 32)[10, 5])


def test_foundation_border_metadata_uses_agent_row_col() -> None:
    assert _border_tile_allowed((10, 5))
    assert not _border_tile_allowed((10, 5), cabin_angle_idx=3)
    assert not _border_tile_allowed((5, 10))


if __name__ == "__main__":
    try:
        jax.config.update("jax_disable_jit", True)
        test_foundation_border_metadata_uses_agent_row_col()
    except AssertionError as exc:
        print("FAILED:", exc)
        sys.exit(1)
    print("OK: foundation border metadata uses pos_base [row, col].")
