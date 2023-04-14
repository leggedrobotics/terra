from jax import Array
from src.config import EnvConfig
from src.state import State
from typing import Tuple, Dict, Optional, Callable
from viz.window import Window
from viz.rendering import RenderingEngine


class TerraEnv:
    def __init__(self, env_cfg: EnvConfig) -> None:
        self.env_cfg = env_cfg

        # TODO remove following
        self.window = Window("Terra")
        self.rendering_engine = RenderingEngine(
            x_dim=env_cfg.target_map.width,
            y_dim=env_cfg.target_map.height
        )

    def reset(self, seed: int) -> State:
        """
        Resets the environment using values from config files, and a seed.
        """
        return State.new(seed, self.env_cfg)

    def render(self, state: State,
               key_handler: Optional[Callable],
               mode: str = "human",
               block: bool = False,
               tile_size: int = 32) -> Array:
        """
        Renders the environment at a given state.
        """
        img = self.rendering_engine.render_grid(tile_size=tile_size, height_grid=state.world.action_map.map,)

        print(f"{img.shape=}")

        if key_handler:
            if mode == "human":
                self.window.show_img(img)
                self.window.reg_key_handler(key_handler)
                self.window.show(block)
        
        return img


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

    def __init__(self, env_cfg: EnvConfig = EnvConfig()) -> None:
        self.terra_env = TerraEnv(env_cfg)

    # TODO JIT
    def reset(self, seeds: Array) -> State:
        pass

    # TODO JIT
    def step(self, state: State, actions: Array) -> Tuple[State, Tuple[Dict, Array, Array, Dict]]:
        pass
