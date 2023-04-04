import torch
from typing import NamedTuple
from src.config import EnvConfig, BufferConfig
from src.map import GridWorld
from src.agent import Agent
from src.frontend import StateFrontend

class State(NamedTuple):
    """
    Stores the current state of the environment.
    Given perfect information, the observation corresponds to the state.
    """
    env_cfg: EnvConfig

    world: GridWorld
    agent: Agent

    @classmethod
    def new(cls, env_cfg: EnvConfig, buf_cfg: BufferConfig) -> "State":
        world = GridWorld.new(env_cfg, buf_cfg)
        agent = Agent.new(env_cfg)

        # TODO: in JUX they do this (their Unit is out Agent):
        """
        empty_unit = Unit.empty(env_cfg)
        empty_unit = jax.tree_map(lambda x: x if isinstance(x, Array) else np.array(x), empty_unit)
        units = jax.tree_map(lambda x: x[None].repeat(buf_cfg.MAX_N_UNITS, axis=0), empty_unit)
        units = jax.tree_map(lambda x: x[None].repeat(2, axis=0), units)
        """
        return State(
            env_cfg=env_cfg,
            world=world,
            agent=agent
        )

    @classmethod
    def from_frontend(cls, state_frontend: StateFrontend, buf_cfg: BufferConfig = BufferConfig()) -> "State":
        pass

    def to_frontend(self) -> StateFrontend:
        pass
