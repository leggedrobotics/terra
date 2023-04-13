import numpy as np
import jax.numpy as jnp
from typing import NamedTuple
from src.actions import TrackedAction
from src.config import EnvConfig
from src.utils import IntLowDim, INTLOWDIM_MAX, IntMap, INTMAP_MAX


class AgentState(NamedTuple):
    pos_base: IntMap  # = jnp.full((2, ), fill_value=INTMAP_MAX, dtype=IntMap)
    angle_base: IntLowDim  # = jnp.full((1, ), fill_value=INTLOWDIM_MAX, dtype=IntLowDim)
    angle_cabin: IntLowDim  # = jnp.full((1, ), fill_value=INTLOWDIM_MAX, dtype=IntLowDim)
    arm_extension: IntLowDim  # = jnp.full((1, ), fill_value=INTLOWDIM_MAX, dtype=IntLowDim)
    loaded: jnp.bool_  # = jnp.full((1, ), fill_value=False, dtype=jnp.bool_)


class Agent(NamedTuple):
    """
    Defines the state and type of the agent.
    """
    # agent_type  # TODO implement later on
    action: TrackedAction  # TODO replace with something more generic
    agent_state: AgentState

    @staticmethod
    def new(env_cfg: EnvConfig) -> "Agent":
        agent_state = AgentState(
            pos_base=IntMap(np.ceil([env_cfg.agent.width / 2, env_cfg.agent.height / 2])),
            angle_base=jnp.full((1, ), fill_value=0, dtype=IntLowDim),
            angle_cabin=jnp.full((1, ), fill_value=0, dtype=IntLowDim),
            arm_extension=jnp.full((1, ), fill_value=0, dtype=IntLowDim),
            loaded=jnp.full((1, ), fill_value=False, dtype=jnp.bool_)
        )
        
        return Agent(
            action=TrackedAction.do_nothing(),
            agent_state=agent_state
        )
