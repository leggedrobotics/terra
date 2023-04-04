import torch
from typing import NamedTuple
from enum import IntEnum
from src.actions import ActionQueue
from src.config import EnvConfig

class AgentType(IntEnum):
    WHEELED = 0
    TRACKED = 1

class AgentState(NamedTuple):
    pos_base: torch.Tensor = torch.zeros((2,), dtype=torch.int)
    angle_base: torch.Tensor = torch.zeros((1,), dtype=torch.int)
    angle_cabin: torch.Tensor = torch.zeros((1,), dtype=torch.int)
    arm_extension: torch.Tensor = torch.zeros((1,), dtype=torch.int)
    loaded: torch.Tensor = torch.zeros((1,), dtype=torch.int)

class Agent(NamedTuple):
    """
    Defines the state and type of the agent.
    """
    agent_type: torch.Tensor
    action_queue: ActionQueue
    agent_state: AgentState

    @staticmethod
    def new(env_cfg: EnvConfig) -> "Agent":
        return Agent(
            agent_type=torch.tensor([AgentType[env_cfg.agent_type]], dtype=torch.int),
            action_queue=ActionQueue.empty(env_cfg.action_queue_capacity, env_cfg.agent_type),
            agent_state=AgentState()
        )
