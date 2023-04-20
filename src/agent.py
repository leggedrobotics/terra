import numpy as np
import jax.numpy as jnp
from typing import NamedTuple
from src.actions import TrackedAction
from src.config import EnvConfig
from src.utils import IntLowDim, INTLOWDIM_MAX, IntMap, INTMAP_MAX


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
    loaded: jnp.bool_


class Agent(NamedTuple):
    """
    Defines the state and type of the agent.
    """
    # agent_type  # TODO implement later on
    action: TrackedAction  # TODO replace with something more generic
    agent_state: AgentState

    @staticmethod
    def new(env_cfg: EnvConfig) -> "Agent":
        max_center_coord = np.ceil(np.max([env_cfg.agent.width / 2 - 1, env_cfg.agent.height / 2 - 1])).astype(IntMap)
        agent_state = AgentState(
            pos_base=IntMap(jnp.array([max_center_coord, max_center_coord])),
            angle_base=jnp.full((1, ), fill_value=0, dtype=IntLowDim),
            angle_cabin=jnp.full((1, ), fill_value=0, dtype=IntLowDim),
            arm_extension=jnp.full((1, ), fill_value=0, dtype=IntLowDim),
            loaded=jnp.full((1, ), fill_value=False, dtype=jnp.bool_)
        )
        
        return Agent(
            action=TrackedAction.do_nothing(),
            agent_state=agent_state
        )
