from typing import NamedTuple, Optional, Tuple

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


class Agent(NamedTuple):
    """
    Defines the state and type of the agent.
    """

    agent_state: AgentState

    width: int
    height: int

    @staticmethod
    def new(
        key: jax.random.PRNGKey,
        env_cfg: EnvConfig,
        max_traversable_x: int,
        max_traversable_y: int,
        padding_mask: Array,
        custom_pos: Optional[Tuple[int, int]] = None,
        custom_angle: Optional[int] = None,
    ) -> tuple["Agent", jax.random.PRNGKey]:
        """
        Create a new agent with specified parameters.
        
        Args:
            key: JAX random key
            env_cfg: Environment configuration
            max_traversable_x: Maximum traversable x coordinate
            max_traversable_y: Maximum traversable y coordinate
            padding_mask: Mask indicating obstacles
            custom_pos: Optional custom position (x, y) to place the agent
            custom_angle: Optional custom angle for the agent
            
        Returns:
            New agent instance and updated random key
        """
        # Handle custom position or default based on config
        has_custom_args = (custom_pos is not None) or (custom_angle is not None)
        
        def use_custom_position(k):
            # Create position based on custom args or defaults
            temp_pos = IntMap(jnp.array(custom_pos)) if custom_pos is not None else IntMap(jnp.array([-1, -1]))
            temp_angle = jnp.full((1,), custom_angle, dtype=IntMap) if custom_angle is not None else jnp.full((1,), -1, dtype=IntMap)
            
            # Get default position for missing components
            def_pos, def_angle, _ = _get_top_left_init_state(k, env_cfg)
            
            # Combine custom and default values
            pos = jnp.where(jnp.any(temp_pos < 0), def_pos, temp_pos)
            angle = jnp.where(jnp.any(temp_angle < 0), def_angle, temp_angle)
            
            # Check validity and return result using jax.lax.cond
            valid = _validate_agent_position(
                pos, angle, env_cfg, padding_mask, 
                env_cfg.agent.width, env_cfg.agent.height
            )
            
            # Define the true and false branches for jax.lax.cond
            def true_fn(_):
                return (pos, angle, k)
                
            def false_fn(_):
                return jax.lax.cond(
                    env_cfg.agent.random_init_state,
                    lambda k_inner: _get_random_init_state(
                        k_inner, env_cfg, max_traversable_x, max_traversable_y, 
                        padding_mask, env_cfg.agent.width, env_cfg.agent.height,
                    ),
                    lambda k_inner: _get_top_left_init_state(k_inner, env_cfg),
                    k
                )
            
            # Use jax.lax.cond to handle the validity check
            return jax.lax.cond(valid, true_fn, false_fn, None)
        
        def use_default_position(k):
            # Use existing logic for random or top-left position
            return jax.lax.cond(
                env_cfg.agent.random_init_state,
                lambda k_inner: _get_random_init_state(
                    k_inner, env_cfg, max_traversable_x, max_traversable_y, 
                    padding_mask, env_cfg.agent.width, env_cfg.agent.height,
                ),
                lambda k_inner: _get_top_left_init_state(k_inner, env_cfg),
                k
            )
        
        # Use jax.lax.cond for JAX-compatible control flow
        pos_base, angle_base, key = jax.lax.cond(
            has_custom_args,
            use_custom_position,
            use_default_position,
            key
        )

        agent_state = AgentState(
            pos_base=pos_base,
            angle_base=angle_base,
            angle_cabin=jnp.full((1,), 0, dtype=IntLowDim),
            loaded=jnp.full((1,), 0, dtype=IntLowDim),
        )

        width = env_cfg.agent.width
        height = env_cfg.agent.height

        return Agent(agent_state=agent_state, width=width, height=height), key


def _validate_agent_position(
    pos_base: Array,
    angle_base: Array,
    env_cfg: EnvConfig,
    padding_mask: Array,
    agent_width: int,
    agent_height: int,
) -> Array:
    """
    Validate if an agent position is valid (within bounds and not intersecting obstacles).
    
    Returns:
        JAX array with boolean value indicating if the position is valid
    """
    map_width = padding_mask.shape[0]
    map_height = padding_mask.shape[1]
    
    # Check if position is within bounds
    max_center_coord = jnp.ceil(
        jnp.max(jnp.array([agent_width / 2 - 1, agent_height / 2 - 1]))
    ).astype(IntMap)
    
    max_w = jnp.minimum(env_cfg.maps.edge_length_px, map_width)
    max_h = jnp.minimum(env_cfg.maps.edge_length_px, map_height)
    
    within_bounds = jnp.logical_and(
        jnp.logical_and(pos_base[0] >= max_center_coord, pos_base[0] < max_w - max_center_coord),
        jnp.logical_and(pos_base[1] >= max_center_coord, pos_base[1] < max_h - max_center_coord)
    )
    
    # Check if position intersects with obstacles
    def check_obstacle_intersection(_):
        agent_corners_xy = get_agent_corners(
            pos_base, angle_base, agent_width, agent_height, env_cfg.agent.angles_base
        )
        polygon_mask = compute_polygon_mask(agent_corners_xy, map_width, map_height)
        has_obstacle = jnp.any(jnp.logical_and(polygon_mask, padding_mask == 1))
        return jnp.logical_not(has_obstacle)
    
    def return_false(_):
        return jnp.array(False)
    
    # Only check obstacles if we're within bounds (to avoid unnecessary computations)
    valid = jax.lax.cond(
        within_bounds,
        check_obstacle_intersection,
        return_false,
        None
    )
    
    return valid


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