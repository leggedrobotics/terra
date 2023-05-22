from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from terra.actions import Action
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.state import State
from terra.wrappers import LocalMapWrapper
from terra.wrappers import TraversabilityMaskWrapper


class TerraEnv:
    def __init__(
        self,
        env_cfg: EnvConfig = EnvConfig(),
        rendering: bool = False,
        n_imgs_row: int = 1,
    ) -> None:
        self.env_cfg = env_cfg

        if rendering:
            from viz.window import Window
            from viz.rendering import RenderingEngine

            self.window = Window("Terra", n_imgs_row)
            self.rendering_engine = RenderingEngine(
                x_dim=env_cfg.target_map.width, y_dim=env_cfg.target_map.height
            )

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TerraEnv) and self.env_cfg == __o.env_cfg

    def __hash__(self) -> int:
        return hash((TerraEnv, self.env_cfg))

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, seed: int) -> tuple[State, dict[str, Array]]:
        """
        Resets the environment using values from config files, and a seed.
        """
        key = jax.random.PRNGKey(seed)
        state = State.new(key, self.env_cfg)
        # TODO remove wrappers from state
        state = TraversabilityMaskWrapper.wrap(state)
        state = LocalMapWrapper.wrap(state)
        observations = self._state_to_obs_dict(state)
        return state, observations

    @partial(jax.jit, static_argnums=(0,))
    def _reset_existent(self, state: State) -> tuple[State, dict[str, Array]]:
        """
        Resets the env, assuming that it already exists.
        """
        state = state._reset(self.env_cfg)
        # TODO remove wrappers from state
        state = TraversabilityMaskWrapper.wrap(state)
        state = LocalMapWrapper.wrap(state)
        observations = self._state_to_obs_dict(state)
        return state, observations

    def render(
        self,
        state: State,
        key_handler: Callable | None = None,
        mode: str = "human",
        block: bool = False,
        tile_size: int = 32,
    ) -> Array:
        """
        Renders the environment at a given state.

        # TODO write a cleaner rendering engine
        """
        imgs_global = self.rendering_engine.render_global(
            tile_size=tile_size,
            active_grid=state.world.action_map.map,
            target_grid=state.world.target_map.map,
            agent_pos=state.agent.agent_state.pos_base,
            base_dir=state.agent.agent_state.angle_base,
            cabin_dir=state.agent.agent_state.angle_cabin,
            agent_width=self.env_cfg.agent.width,
            agent_height=self.env_cfg.agent.height,
        )

        imgs_local = state.world.local_map.map

        if key_handler:
            if mode == "human":
                self.window.set_title(
                    title=f"Arm extension = {state.agent.agent_state.arm_extension.item()}",
                    idx=0,
                )
                self.window.show_img(imgs_global, [imgs_local], mode)
                self.window.reg_key_handler(key_handler)
                self.window.show(block)
        if mode == "gif":
            self.window.set_title(
                title=f"Arm extension = {state.agent.agent_state.arm_extension.item()}",
                idx=0,
            )
            self.window.show_img(imgs_global, [imgs_local], mode)
            # self.window.show(block)

        return imgs_global, imgs_local

    def render_obs(
        self,
        obs: dict[str, Array],
        key_handler: Callable | None = None,
        mode: str = "human",
        block: bool = False,
        tile_size: int = 32,
    ) -> Array:
        """
        Renders the environment at a given observation.

        # TODO write a cleaner rendering engine
        """
        imgs_global = self.rendering_engine.render_global(
            tile_size=tile_size,
            active_grid=obs["action_map"],
            target_grid=obs["target_map"],
            agent_pos=obs["agent_state"][..., [0, 1]],
            base_dir=obs["agent_state"][..., [2]],
            cabin_dir=obs["agent_state"][..., [3]],
            agent_width=self.env_cfg.agent.width,
            agent_height=self.env_cfg.agent.height,
        )

        imgs_local = obs["local_map"]

        if key_handler:
            if mode == "human":
                # self.window.set_title(
                #     title=f"Arm extension = {obs['agent_state'][..., 4].item()}"
                # )
                self.window.show_img(imgs_global, imgs_local, mode)
                self.window.reg_key_handler(key_handler)
                self.window.show(block)
        if mode == "gif":
            # self.window.set_title(
            #     title=f"Arm extension = {obs['agent_state'][..., 4].item()}"
            # )
            self.window.show_img(imgs_global, imgs_local, mode)
            # self.window.show(block)

        return imgs_global, imgs_local

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

        new_state = TraversabilityMaskWrapper.wrap(new_state)
        new_state = LocalMapWrapper.wrap(new_state)

        observations = self._state_to_obs_dict(new_state)

        done = state._is_done(
            new_state.world.action_map.map,
            new_state.world.target_map.map,
            new_state.agent.agent_state.loaded,
        )

        new_state, observations = jax.lax.cond(
            done,
            self._reset_existent,
            lambda x: (new_state, observations),
            new_state,
        )

        infos = new_state._get_infos(action)

        return new_state, (observations, reward, done, infos)

    @staticmethod
    def _state_to_obs_dict(state: State) -> dict[str, Array]:
        """
        Transforms a State object to an observation dictionary.
        """
        agent_state = jnp.hstack(
            [
                state.agent.agent_state.pos_base,  # pos_base is encoded in traversability_mask
                state.agent.agent_state.angle_base,
                state.agent.agent_state.angle_cabin,
                state.agent.agent_state.arm_extension,
                state.agent.agent_state.loaded,
            ]
        )
        return {
            "agent_state": agent_state,
            "local_map": state.world.local_map.map,
            "traversability_mask": state.world.traversability_mask.map,
            "action_map": state.world.action_map.map,
            "target_map": state.world.target_map.map,
        }


class TerraEnvBatch:
    """
    Takes care of the parallelization of the environment.
    """

    def __init__(
        self,
        env_cfg: EnvConfig = EnvConfig(),
        batch_cfg: BatchConfig = BatchConfig(),
        rendering: bool = False,
        n_imgs_row: int = 1,
    ) -> None:
        self.terra_env = TerraEnv(env_cfg, rendering=rendering, n_imgs_row=n_imgs_row)
        self.batch_cfg = batch_cfg
        self.env_cfg = env_cfg

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, seeds: Array) -> State:
        return jax.vmap(self.terra_env.reset)(seeds)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, states: State, actions: Action
    ) -> tuple[State, tuple[dict, Array, Array, dict]]:
        states, (obs, rewards, dones, infos) = jax.vmap(self.terra_env.step)(
            states, actions
        )
        return states, (obs, rewards, dones, infos)

    @property
    def actions_size(self) -> int:
        """
        Number of actions played at every env step.
        """
        return ()

    @property
    def num_actions(self) -> int:
        """
        Total number of actions
        """
        return self.batch_cfg.action_type.get_num_actions()

    # @property
    # def observation_space(self) -> Dict[str, Dict[str, Array]]:
    #     """
    #     box around:
    #         angle_base: 0, 3 -- discrete
    #         angle_cabin: 0, 7 -- discrete
    #         arm_extension: 0, 1 -- discrete
    #         loaded: 0, N -- discrete

    #         local_map: 8 x 2 x [-M, M] -- discrete
    #         action_map: L1 x L2 x [-M, M] -- discrete
    #         target_map: L1 x L2 x [-M, 0] -- discrete
    #         traversability mask: L1 x L2 x [-1, 1] -- discrete

    #     Note: both low and high values are inclusive.
    #     """
    #     observation_space = {"low": {},
    #                          "high": {}}

    #     low_agent_state = jnp.array(
    #         [
    #             0,  # angle_base
    #             0,  # angle_cabin
    #             0,  # arm_extension
    #             0,  # loaded
    #         ],
    #         dtype=IntMap
    #     )
    #     high_agent_state = jnp.array(
    #         [
    #             self.env_cfg.agent.angles_base - 1,  # angle_base
    #             self.env_cfg.agent.angles_cabin - 1,  # angle_cabin
    #             self.env_cfg.agent.max_arm_extension,  # arm_extension
    #             self.env_cfg.agent.max_loaded,  # loaded
    #         ],
    #         dtype=IntMap
    #     )
    #     observation_space["low"]["agent_states"] = low_agent_state
    #     observation_space["high"]["agent_states"] = high_agent_state

    #     arm_extensions = self.env_cfg.agent.max_arm_extension + 1
    #     angles_cabin = self.env_cfg.agent.angles_cabin
    #     low_local_map = jnp.array(
    #         [
    #             self.env_cfg.action_map.min_height
    #         ],
    #         dtype=IntMap
    #     )[None].repeat(arm_extensions, 0)[None].repeat(angles_cabin, 0)
    #     high_local_map = jnp.array(
    #         [
    #             self.env_cfg.action_map.max_height
    #         ],
    #         dtype=IntMap
    #     )[None].repeat(arm_extensions, 0)[None].repeat(angles_cabin, 0)
    #     observation_space["low"]["local_map"] = low_local_map
    #     observation_space["high"]["local_map"] = high_local_map

    #     n_grid_maps = 2
    #     grid_map_width = self.env_cfg.action_map.width
    #     grid_map_height = self.env_cfg.action_map.height
    #     low_grid_maps = jnp.array(
    #         [
    #             self.env_cfg.action_map.min_height
    #         ],
    #         dtype=IntMap
    #     )[None].repeat(grid_map_height, 0)[None].repeat(grid_map_width, 0)
    #     high_grid_maps = jnp.array(
    #         [
    #             self.env_cfg.action_map.max_height
    #         ],
    #         dtype=IntMap
    #     )[None].repeat(grid_map_height, 0)[None].repeat(grid_map_width, 0)
    #     observation_space["low"]["active_map"] = low_local_map
    #     observation_space["high"]["active_map"] = high_local_map

    #     # TODO target map and traversability mask

    #     return observation_space

    @property
    def observation_shapes(self) -> dict[str, tuple]:
        """
        Returns a dict that has:
            - key: name of the input feature (e.g. "local_map")
            - value: the tuple representing the shape of the input feature
        """
        return {
            "agent_states": (6,),
            "local_map": (
                self.env_cfg.agent.angles_cabin,
                self.env_cfg.agent.max_arm_extension + 1,
            ),
            "action_map": (
                self.env_cfg.action_map.width,
                self.env_cfg.action_map.height,
            ),
            "target_map": (
                self.env_cfg.action_map.width,
                self.env_cfg.action_map.height,
            ),
            "traversability_mask": (
                self.env_cfg.action_map.width,
                self.env_cfg.action_map.height,
            ),
        }