import torch
from typing import NamedTuple
from config import EnvConfig, BufferConfig
from map import GridWorld
from agent import Agent

class State(NamedTuple):
    """
    Stores the current state of the environment.
    Given perfect information, the observation corresponds to the state.
    """
    env_cfg: EnvConfig
    seed: torch.int32

    target_map: GridWorld
    action_map: GridWorld
    traversability_mask_map: GridWorld
    agent: Agent

    @classmethod
    def new(cls, seed: int, env_cfg: EnvConfig, buf_cfg: BufferConfig) -> "State":
        pass
