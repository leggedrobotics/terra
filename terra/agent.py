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
    wheel_angle: IntLowDim
    loaded: IntLowDim


class Agent(NamedTuple):
    """
    Defines the state and type of the agent.
    """

    agent_state: AgentState
    agent_state_2: AgentState

    width: int
    height: int

    moving_dumped_dirt: bool

    @staticmethod
    def new(
        key: jax.random.PRNGKey,
        env_cfg: EnvConfig,
        max_traversable_x: int,
        max_traversable_y: int,
        padding_mask: Array,
        action_map: Array,
    ) -> tuple["Agent", jax.random.PRNGKey]:
        # Split the key for the two agent initializations
        key, key_agent1, key_agent2 = jax.random.split(key, 3)

        # Initialize first agent
        pos_base_1, angle_base_1, key_agent1 = _get_random_init_state(
            key_agent1,
            env_cfg,
            max_traversable_x,
            max_traversable_y,
            padding_mask,
            action_map,
            env_cfg.agent.width,
            env_cfg.agent.height,
        )

        # Create a temporary agent mask to prevent agent 2 spawning on agent 1
        map_width, map_height = padding_mask.shape
        agent1_corners = get_agent_corners(
            pos_base_1, angle_base_1, env_cfg.agent.width, env_cfg.agent.height, 
            env_cfg.agent.angles_base
        )
        agent1_mask = compute_polygon_mask(agent1_corners, map_width, map_height)
        
        # Combined mask including both padding and first agent location
        combined_mask = jnp.logical_or(padding_mask == 1, agent1_mask)
        
        # Initialize second agent, avoiding both obstacles and first agent
        pos_base_2, angle_base_2, key_agent2 = _get_random_init_state(
            key_agent2,
            env_cfg,
            max_traversable_x,
            max_traversable_y,
            combined_mask,  # Use the combined mask to avoid agent 1
            action_map,
            env_cfg.agent.width,
            env_cfg.agent.height,
        )

        agent_state_1 = AgentState(
            pos_base=pos_base_1,
            angle_base=angle_base_1,
            angle_cabin=jnp.full((1,), 0, dtype=IntLowDim),
            wheel_angle=jnp.full((1,), 0, dtype=IntLowDim),
            loaded=jnp.full((1,), 0, dtype=IntLowDim),
        )
        
        agent_state_2 = AgentState(
            pos_base=pos_base_2,
            angle_base=angle_base_2,
            angle_cabin=jnp.full((1,), 0, dtype=IntLowDim),
            wheel_angle=jnp.full((1,), 0, dtype=IntLowDim),
            loaded=jnp.full((1,), 0, dtype=IntLowDim),
        )

        width = env_cfg.agent.width
        height = env_cfg.agent.height
        moving_dumped_dirt = False

        return Agent(
            agent_state=agent_state_1, 
            agent_state_2=agent_state_2, 
            width=width, 
            height=height, 
            moving_dumped_dirt=moving_dumped_dirt
        ), key


def _get_top_left_init_state(key: jax.random.PRNGKey, env_cfg: EnvConfig):
    max_center_coord = jnp.ceil(
        jnp.max(
            jnp.array([env_cfg.agent.width // 2 - 1, env_cfg.agent.height // 2 - 1])
        )
    ).astype(IntMap)
    pos_base = IntMap(jnp.array([max_center_coord, max_center_coord]))
    theta = jnp.full((1,), fill_value=0, dtype=IntMap)
    return pos_base, theta, key


def _get_random_init_state(
    key: jax.random.PRNGKey,
    env_cfg: EnvConfig,
    max_traversable_x: int,
    max_traversable_y: int,
    padding_mask: Array,
    action_map: Array,
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
            action_illegal = jnp.any(jnp.logical_and(polygon_mask, action_map != 0))
            return obstacle_inside | action_illegal

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
