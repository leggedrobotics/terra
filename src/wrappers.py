import jax.numpy as jnp

from src.state import State
from src.utils import IntLowDim


class TraversabilityMaskWrapper:
    @staticmethod
    def wrap(state: State) -> State:
        """
        Encodes the traversability mask in GridWorld.

        The traversability mask has the same size as the action map, and encodes:
        - 0: no obstacle
        - 1: obstacle (digged or dumped tile)
        - (-1): agent occupying the tile
        """
        # encode map obstacles
        traversability_mask = (state.world.action_map.map != 0).astype(IntLowDim)

        # encode agent pos and size in the map
        agent_corners = state._get_agent_corners(
            state.agent.agent_state.pos_base,
            state.agent.agent_state.angle_base,
            state.env_cfg.agent.width,
            state.env_cfg.agent.height,
        )
        x, y = state._get_agent_corners_xy(agent_corners)

        map_width = state.world.width
        map_height = state.world.height
        traversability_mask = jnp.where(
            jnp.logical_or(
                jnp.logical_or(
                    (jnp.arange(map_width) > x[1])[:, None].repeat(map_height, axis=1),
                    (jnp.arange(map_width) < x[0])[:, None].repeat(map_height, axis=1),
                ),
                jnp.logical_or(
                    (jnp.arange(map_height) > y[1])[None].repeat(map_width, axis=0),
                    (jnp.arange(map_height) < y[0])[None].repeat(map_width, axis=0),
                ),
            ),
            traversability_mask,
            -1,
        )

        return state._replace(
            world=state.world._replace(
                traversability_mask=state.world.traversability_mask._replace(
                    map=traversability_mask.astype(IntLowDim)
                )
            )
        )
