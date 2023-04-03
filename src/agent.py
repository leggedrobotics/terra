import torch
from typing import NamedTuple
from enum import IntEnum

class AgentType(IntEnum):
    WHEELED = 0
    TRUCKED = 1

class AgentState(NamedTuple):
    pos_base: torch.Tensor
    angle_base: torch.Tensor
    angle_cabin: torch.Tensor
    arm_extension: torch.Tensor
    loaded: torch.Tensor

class Agent(NamedTuple):
    """
    Defines the state and type of the agent.
    """
    agent_type: AgentType
    agent_state: AgentState = AgentState()
