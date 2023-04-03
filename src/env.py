import torch
from typing import Tuple, Dict
from config import EnvConfig, BufferConfig
from state import State

from frontend import TerraEnvFrontend

class TerraEnv:
    def __init__(self, env_cfg: EnvConfig, buf_cfg: BufferConfig) -> None:
        self.env_cfg = env_cfg
        self.buf_cfg = buf_cfg
        self._dummy_env = TerraEnvFrontend()  # for rendering

    def reset(self, seed: int) -> State:
        """
        Resets the environment using values from config files, and a seed.
        """
        return State.new(seed, self.env_cfg, self.buf_cfg)

    def render(self, state: State):
        """
        Renders the environment at a given state.
        """
        # TODO define return
        pass

    # TODO JIT
    def step(self, state: State, action: torch.Tensor) -> Tuple[State, Tuple[Dict, torch.Tensor, torch.Tensor, Dict]]:
        """
        Step the env given an action

        Args:
            state (State)
            action (torch.Tensor): action converted to backend

        Returns:
            state, (observations, rewards, dones, infos)

            state (State): new state.
            observations (Dict): same as the state, as we are assuming perfect observability.
            rewards (torch.Tensor): # TODO define reward dtype (float?), reward for the agent.
            done (torch.Tensor): bool, done indicator. If episode ends, then done = True.
            infos (Dict): #TODO should be empty.
        """
        pass

    def close(self):
        return self._dummy_env.close()
    
    @staticmethod
    def from_frontend(lux_env: TerraEnvFrontend, buf_cfg=BufferConfig()) -> Tuple['TerraEnv', 'State']:
        """
        Create a TerraEnv from a TerraEnvFrontend env.
        """
        pass

class TerraEnvBatch:
    """
    Takes care of the parallelization of the environment.
    """

    def __init__(self, env_cfg: EnvConfig = EnvConfig(), buf_cfg: BufferConfig = BufferConfig()) -> None:
        self.terra_env = TerraEnv(env_cfg, buf_cfg)

    # TODO JIT
    def reset(self, seeds: torch.Tensor) -> Tuple[State, Tuple[Dict, int, bool, Dict]]:
        pass

    # TODO JIT
    def step(self, state: State, action: torch.Tensor) -> Tuple[State, Tuple[Dict, torch.Tensor, torch.Tensor, Dict]]:
        pass
