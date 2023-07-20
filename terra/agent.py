from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from terra.config import EnvConfig
from terra.utils import IntLowDim
from terra.utils import IntMap


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
    arm_extension: IntLowDim
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
    ) -> "Agent":
        pos_base, key = jax.lax.cond(
            env_cfg.agent.random_init_pos,
            partial(
                _get_random_init_pos,
                max_traversable_x=max_traversable_x,
                max_traversable_y=max_traversable_y,
            ),
            _get_top_left_init_pos,
            key,
            env_cfg,
        )

        angle_base, key = jax.lax.cond(
            env_cfg.agent.random_init_base_angle,
            _get_random_init_base_angle,
            lambda x, y: (jnp.full((1,), fill_value=0, dtype=IntLowDim), key),
            key,
            env_cfg,
        )

        agent_state = AgentState(
            pos_base=pos_base,
            angle_base=angle_base,
            angle_cabin=jnp.full((1,), fill_value=0, dtype=IntLowDim),
            arm_extension=jnp.full((1,), fill_value=0, dtype=IntLowDim),
            loaded=jnp.full((1,), fill_value=0, dtype=IntLowDim),
        )

        width = env_cfg.agent.width
        height = env_cfg.agent.height

        return Agent(agent_state=agent_state, width=width, height=height), key


def _get_top_left_init_pos(key: jax.random.PRNGKey, env_cfg: EnvConfig):
    max_center_coord = jnp.ceil(
        jnp.max(jnp.array([env_cfg.agent.width / 2 - 1, env_cfg.agent.height / 2 - 1]))
    ).astype(IntMap)
    pos_base = IntMap(jnp.array([max_center_coord, max_center_coord]))
    return pos_base, key


def _get_random_init_pos(
    key: jax.random.PRNGKey,
    env_cfg: EnvConfig,
    max_traversable_x: int,
    max_traversable_y: int,
):
    max_center_coord = jnp.ceil(
        jnp.max(jnp.array([env_cfg.agent.width / 2 - 1, env_cfg.agent.height / 2 - 1]))
    ).astype(IntMap)
    key, subkey = jax.random.split(key)

    max_w = jnp.minimum(max_traversable_x, env_cfg.maps.max_width)
    max_h = jnp.minimum(max_traversable_y, env_cfg.maps.max_height)
    x = jax.random.randint(
        subkey,
        (1,),
        minval=max_center_coord,
        maxval=max_w - max_center_coord,
    )
    y = jax.random.randint(
        subkey,
        (1,),
        minval=max_center_coord,
        maxval=max_h - max_center_coord,
    )
    pos_base = IntMap(jnp.concatenate((x, y)))
    return pos_base, key


def _get_random_init_base_angle(key: jax.random.PRNGKey, env_cfg: EnvConfig):
    key, subkey = jax.random.split(key)
    theta = jax.random.randint(
        subkey, (1,), minval=0, maxval=env_cfg.agent.angles_base + 1
    )
    return IntLowDim(theta), key
