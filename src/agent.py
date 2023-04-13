import jax.numpy as jnp
from typing import NamedTuple
from src.actions import ActionBatch
from src.config import EnvConfig
from src.utils import IntLowDim, INTLOWDIM_MAX, IntMap, INTMAP_MAX


class AgentState(NamedTuple):
    pos_base: IntMap = jnp.full((2, ), fill_value=INTMAP_MAX, dtype=IntMap)
    angle_base: IntLowDim = jnp.full((1, ), fill_value=INTLOWDIM_MAX, dtype=IntLowDim)
    angle_cabin: IntLowDim = jnp.full((1, ), fill_value=INTLOWDIM_MAX, dtype=IntLowDim)
    arm_extension: IntLowDim = jnp.full((1, ), fill_value=INTLOWDIM_MAX, dtype=IntLowDim)
    loaded: jnp.bool_ = jnp.full((1, ), fill_value=False, dtype=jnp.bool_)


class Agent(NamedTuple):
    """
    Defines the state and type of the agent.
    """
    # agent_type  # TODO implement later on
    action_batch: ActionBatch  # in JUX an ActionQueue is used
    agent_state: AgentState

    @staticmethod
    def new(env_cfg: EnvConfig) -> "Agent":
        return Agent(
            action_batch=ActionBatch.empty(env_cfg.action_batch_capacity),
            agent_state=AgentState()
        )
