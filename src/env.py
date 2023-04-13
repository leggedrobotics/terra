from jax import Array
from src.config import EnvConfig, BufferConfig
from src.state import State
from typing import Tuple, Dict


class TerraEnv:
    def __init__(self, env_cfg: EnvConfig, buf_cfg: BufferConfig) -> None:
        self.env_cfg = env_cfg
        self.buf_cfg = buf_cfg

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
    def step(self, state: State, action: int) -> Tuple[State, Tuple[Dict, Array, Array, Dict]]:
        """
        Step the env given an action

        Args:
            state (State)
            action (int): the integer corresponding to one of the actions

        Returns:
            state, (observations, rewards, dones, infos)

            state (State): new state.
            observations (Dict): same as the state, as we are assuming perfect observability.
            rewards (jnp.float32): reward for the agent.
            done (jnp.bool_): done indicator. If episode ends, then done = True.
            infos (Dict): additional information (currently empty)
        """
        state = state._step(action)

        reward = state._get_reward()

        infos = {}

        dones = state._is_done()

        return state, (state, reward, dones, infos)

    # def close(self):
    #     return self._dummy_env.close()


class TerraEnvBatch:
    """
    Takes care of the parallelization of the environment.
    """

    def __init__(self, env_cfg: EnvConfig = EnvConfig(), buf_cfg: BufferConfig = BufferConfig()) -> None:
        self.terra_env = TerraEnv(env_cfg, buf_cfg)

    # TODO JIT
    def reset(self, seeds: Array) -> State:
        pass

    # TODO JIT
    def step(self, state: State, actions: Array) -> Tuple[State, Tuple[Dict, Array, Array, Dict]]:
        pass
