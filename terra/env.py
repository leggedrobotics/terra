from collections.abc import Callable
from functools import partial

import jax
from jax import Array

from terra.actions import Action
from terra.config import EnvConfig
from terra.state import State
from terra.wrappers import LocalMapWrapper
from terra.wrappers import TraversabilityMaskWrapper


class TerraEnv:
    def __init__(self, env_cfg: EnvConfig, rendering: bool = False) -> None:
        self.env_cfg = env_cfg

        if rendering:
            from viz.window import Window
            from viz.rendering import RenderingEngine

            self.window = Window("Terra")
            self.rendering_engine = RenderingEngine(
                x_dim=env_cfg.target_map.width, y_dim=env_cfg.target_map.height
            )

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TerraEnv) and self.env_cfg == __o.env_cfg

    def __hash__(self) -> int:
        return hash((TerraEnv, self.env_cfg))

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, seed: int) -> State:
        """
        Resets the environment using values from config files, and a seed.
        """
        return State.new(seed, self.env_cfg)

    def render(
        self,
        state: State,
        key_handler: Callable | None,
        mode: str = "human",
        block: bool = False,
        tile_size: int = 32,
    ) -> Array:
        """
        Renders the environment at a given state.

        # TODO write a cleaner rendering engine
        """
        img_global = self.rendering_engine.render_global(
            tile_size=tile_size,
            active_grid=state.world.action_map.map,
            target_grid=state.world.target_map.map,
            agent_pos=state.agent.agent_state.pos_base,
            base_dir=state.agent.agent_state.angle_base,
            cabin_dir=state.agent.agent_state.angle_cabin,
            agent_width=self.env_cfg.agent.width,
            agent_height=self.env_cfg.agent.height,
        )

        img_local = state.world.local_map.map

        if key_handler:
            if mode == "human":
                self.window.set_title(
                    title=f"Arm extension = {state.agent.agent_state.arm_extension.item()}"
                )
                self.window.show_img(img_global, img_local)
                self.window.reg_key_handler(key_handler)
                self.window.show(block)

        return img_global, img_local

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: State, action: Action
    ) -> tuple[State, tuple[dict, Array, Array, dict]]:
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
        new_state = state._step(action)

        reward = state._get_reward(new_state, action)

        dones = State._is_done(
            new_state.world.action_map.map,
            new_state.world.target_map.map,
            new_state.agent.agent_state.loaded,
        )

        infos = {}

        observations = new_state
        observations = TraversabilityMaskWrapper.wrap(observations)
        observations = LocalMapWrapper.wrap(observations)

        # jax.debug.print("Reward = {x}", x=reward)
        # jax.debug.print("Dones = {x}", x=dones)
        # jax.debug.print("local map = \n{x}", x=observations.world.local_map.map.T)

        return new_state, (observations, reward, dones, infos)


class TerraEnvBatch:
    """
    Takes care of the parallelization of the environment.
    """

    def __init__(self, env_cfg: EnvConfig = EnvConfig()) -> None:
        self.terra_env = TerraEnv(env_cfg)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, seeds: Array) -> State:
        return jax.vmap(self.terra_env.reset)(seeds)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, states: State, actions: Action
    ) -> tuple[State, tuple[dict, Array, Array, dict]]:
        _, (states, rewards, dones, infos) = jax.vmap(self.terra_env.step)(
            states, actions
        )
        return states, (states, rewards, dones, infos)
