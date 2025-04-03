from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.config import EnvConfig
from terra.settings import IntLowDim
from terra.settings import IntMap
from terra.utils import get_agent_corners
from terra.utils import get_agent_corners_xy


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
    ) -> tuple["Agent", jax.random.PRNGKey]:
        pos_base, angle_base, key = jax.lax.cond(
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
            key,
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
                pos_base, angle_base, agent_width, agent_height
            )
            x_minmax_agent, y_minmax_agent = get_agent_corners_xy(agent_corners_xy)

            padding_mask_reduced = jnp.where(
                (jnp.arange(map_width) < x_minmax_agent[0])[:, None].repeat(
                    map_height, axis=1
                ),
                0,
                padding_mask,
            )
            padding_mask_reduced = jnp.where(
                (jnp.arange(map_width) > x_minmax_agent[1])[:, None].repeat(
                    map_height, axis=1
                ),
                0,
                padding_mask_reduced,
            )
            padding_mask_reduced = jnp.where(
                (jnp.arange(map_height) < y_minmax_agent[0])[None].repeat(
                    map_width, axis=0
                ),
                0,
                padding_mask_reduced,
            )
            padding_mask_reduced = jnp.where(
                (jnp.arange(map_height) > y_minmax_agent[1])[None].repeat(
                    map_width, axis=0
                ),
                0,
                padding_mask_reduced,
            )
            valid_move = jnp.all(padding_mask_reduced == 0)
            return ~valid_move

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
