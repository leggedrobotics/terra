import torch
from typing import NamedTuple
from config import EnvConfig, BufferConfig
from map import GridWorld
from agent import Agent
from frontend import StateFrontend

class State(NamedTuple):
    """
    Stores the current state of the environment.
    Given perfect information, the observation corresponds to the state.
    """
    env_cfg: EnvConfig
    seed: torch.int32

    world: GridWorld
    agent: Agent

    @classmethod
    def new(cls, seed: int, env_cfg: EnvConfig, buf_cfg: BufferConfig) -> "State":
        pass

    @classmethod
    def from_frontend(cls, state_frontend: StateFrontend, buf_cfg: BufferConfig = BufferConfig()) -> "State":
        pass

    def to_frontend(self) -> StateFrontend:
        pass
