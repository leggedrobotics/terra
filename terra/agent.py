from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.config import EnvConfig
from terra.settings import IntLowDim
from terra.settings import IntMap
from terra.utils import compute_polygon_mask
from terra.utils import get_agent_corners


class AgentState(NamedTuple):
    """
    Clarifications on the agent state representation.

    angle_base:
    Orientations of the agent are an integer between 0 and 3 (included),
    where 0 means that it is aligned with the x axis, and for every positive
    increment, 90 degrees are added in the direction of the arrow going from
    the x axis to the y axes (anti-clockwise).
    """

    pos_base: IntMap
    angle_base: IntLowDim
    angle_cabin: IntLowDim
    loaded: IntLowDim
    loaded_dumped: bool


class Agent(NamedTuple):
    """
    Defines the state and type of the agent.
    """

    agent_state_1: AgentState
    agent_state_2: AgentState 
    width: int
    height: int

    @staticmethod
    def new(
        key: jax.random.PRNGKey,
        env_cfg: EnvConfig,
        max_traversable_x: int,
        max_traversable_y: int,
        padding_mask: Array,
    ) -> tuple["Agent", jax.random.PRNGKey]:
        # Split the key for the two agent initializations
        key, key_agent1, key_agent2 = jax.random.split(key, 3)
        
        # Initialize first agent
        pos_base_1, angle_base_1, _ = jax.lax.cond(
            env_cfg.agent.random_init_state,
            lambda k: _get_random_init_state(
                k,
                env_cfg,
                max_traversable_x,
                max_traversable_y,
                padding_mask,
                env_cfg.agent.width,
                env_cfg.agent.height,
            ),
            lambda k: _get_top_left_init_state(k, env_cfg),
            key_agent1,
        )

        # Initialize second agent with similar logic but different random key
        pos_base_2, angle_base_2, _ = jax.lax.cond(
            env_cfg.agent.random_init_state,
            lambda k: _get_random_init_state(
                k,
                env_cfg,
                max_traversable_x,
                max_traversable_y,
                padding_mask,
                env_cfg.agent.width,
                env_cfg.agent.height,
            ),
            # For second agent, use bottom right if not random
            lambda k: _get_bottom_right_init_state(k, env_cfg),
            key_agent2,
        )

        agent_state_1 = AgentState(
            pos_base=pos_base_1,
            angle_base=angle_base_1,
            angle_cabin=jnp.full((1,), 0, dtype=IntLowDim),
            loaded=jnp.full((1,), 0, dtype=IntLowDim),
            loaded_dumped=False,  # Initialize loaded_dumped to False
        )
        
        agent_state_2 = AgentState(
            pos_base=pos_base_2,
            angle_base=angle_base_2,
            angle_cabin=jnp.full((1,), 0, dtype=IntLowDim),
            loaded=jnp.full((1,), 0, dtype=IntLowDim),
            loaded_dumped=False,  # Initialize loaded_dumped to False
        )

        width = env_cfg.agent.width
        height = env_cfg.agent.height

        return Agent(agent_state_1=agent_state_1, agent_state_2=agent_state_2, width=width, height=height), key


def _get_top_left_init_state(key: jax.random.PRNGKey, env_cfg: EnvConfig):
    max_center_coord = jnp.ceil(
        jnp.max(
            jnp.array([env_cfg.agent.width // 2 - 1, env_cfg.agent.height // 2 - 1])
        )
    ).astype(IntMap)
    pos_base = IntMap(jnp.array([max_center_coord, max_center_coord]))
    theta = jnp.full((1,), fill_value=0, dtype=IntMap)
    return pos_base, theta, key


def _get_bottom_right_init_state(key: jax.random.PRNGKey, env_cfg: EnvConfig):
    # Place the agent in the bottom-right corner
    max_center_coord = jnp.ceil(
        jnp.max(
            jnp.array([env_cfg.agent.width // 2 - 1, env_cfg.agent.height // 2 - 1])
        )
    ).astype(IntMap)
    
    # Calculate position at bottom right
    edge_length = env_cfg.maps.edge_length_px
    pos_base = IntMap(jnp.array([edge_length - max_center_coord - 1, 
                                edge_length - max_center_coord - 1]))
    
    # Start with a different orientation (180° from first agent)
    theta = jnp.full((1,), fill_value=2, dtype=IntMap)  # 2 = 180 degrees if angles are 0-3
    
    return pos_base, theta, key


def _get_random_init_state(
    key: jax.random.PRNGKey,
    env_cfg: EnvConfig,
    max_traversable_x: int,
    max_traversable_y: int,
    padding_mask: Array,
    agent_width: int,
    agent_height: int,
):
    def _get_random_agent_state(carry):
        key, padding_mask, pos_base, angle_base = carry
        max_center_coord = jnp.ceil(
            jnp.max(
                jnp.array([env_cfg.agent.width / 2 - 1, env_cfg.agent.height / 2 - 1])
            )
        ).astype(IntMap)
        key, subkey_x, subkey_y, subkey_angle = jax.random.split(key, 4)

        max_w = jnp.minimum(max_traversable_x, env_cfg.maps.edge_length_px)
        max_h = jnp.minimum(max_traversable_y, env_cfg.maps.edge_length_px)

        x = jax.random.randint(
            subkey_x,
            (1,),
            minval=max_center_coord,
            maxval=max_w - max_center_coord,
        )
        y = jax.random.randint(
            subkey_y,
            (1,),
            minval=max_center_coord,
            maxval=max_h - max_center_coord,
        )
        pos_base = IntMap(jnp.concatenate((x, y)))
        angle_base = jax.random.randint(
            subkey_angle, (1,), 0, env_cfg.agent.angles_base, dtype=IntMap
        )
        return key, padding_mask, pos_base, angle_base

    def _check_agent_obstacles_intersection(carry):
        key, padding_mask, pos_base, angle_base = carry
        map_width = padding_mask.shape[0]
        map_height = padding_mask.shape[1]

        def _check_intersection():
            """
            Checks that the agent does not spawn where an obstacle is (or else it will get stuck forever).
            The check takes the four agent corners and checks that in the tiles included
            within the corners there is no obstacle-encoded tile.
            The padding mask is the map encoding obstacles (1 for obstacle and 0 for no obstacle).
            """
            agent_corners_xy = get_agent_corners(
                pos_base, angle_base, agent_width, agent_height, env_cfg.agent.angles_base
            )
            polygon_mask = compute_polygon_mask(
                agent_corners_xy, map_width, map_height
            )

            obstacle_inside = jnp.any(jnp.logical_and(polygon_mask, padding_mask == 1))
            return obstacle_inside

        keep_searching = jax.lax.cond(
            jnp.any(pos_base < 0) | jnp.any(angle_base < 0),
            lambda: True,
            _check_intersection,
        )
        return keep_searching

    key, padding_mask, pos_base, angle_base = jax.lax.while_loop(
        _check_agent_obstacles_intersection,
        _get_random_agent_state,
        (
            key,
            padding_mask,
            jnp.array([-1, -1], dtype=IntMap),
            jnp.full((1,), -1, dtype=IntMap),
        ),
    )

    return pos_base, angle_base, key
