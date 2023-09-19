from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.actions import Action
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.maps_buffer import init_maps_buffer
from terra.state import State
from terra.wrappers import LocalMapWrapper
from terra.wrappers import TraversabilityMaskWrapper
from viz.rendering import RenderingEngine
from viz.window import Window
import pygame as pg
from viz_pygame.game.game import Game


class TerraEnv(NamedTuple):
    rendering_engine: Game | RenderingEngine | None = None
    window: Window | None = None  # Note: not used if pygame rendering engine is used

    @classmethod
    def new(cls, rendering: bool = False, n_envs_x: int = 1, n_envs_y: int = 1, display: bool = False, rendering_engine: str = "numpy") -> "TerraEnv":
        rendering_engine = None
        window = None
        if rendering:
            print(f"Using {rendering_engine} rendering_engine")
            if rendering_engine == "numpy":
                window = Window("Terra", n_envs_x)
                rendering_engine = RenderingEngine()
            elif rendering_engine == "pygame":
                pg.init()
                pg.mixer.init()
                if not display:
                    print("TerraEnv: disabling display...")
                    screen = pg.display.set_mode((1920 // 2, 1080 // 2), pg.FULLSCREEN | pg.HIDDEN)
                else:
                    screen = pg.display.set_mode((1920 // 2, 1080 // 2))

                clock = pg.time.Clock()
                rendering_engine = Game(screen, clock, n_envs_x=n_envs_x, n_envs_y=n_envs_y, display=display)
            else:
                raise(ValueError(f"{rendering_engine=}"))
        return TerraEnv(rendering_engine=rendering_engine, window=window)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        seed: int,
        target_map: Array,
        padding_mask: Array,
        trench_axes: Array,
        trench_type: Array,
        dumpability_mask_init: Array,
        env_cfg: EnvConfig,
    ) -> tuple[State, dict[str, Array]]:
        """
        Resets the environment using values from config files, and a seed.
        """
        key = jax.random.PRNGKey(seed)
        state = State.new(
            key, env_cfg, target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init
        )
        # TODO remove wrappers from state
        state = TraversabilityMaskWrapper.wrap(state)
        state = LocalMapWrapper.wrap_target_map(state)
        state = LocalMapWrapper.wrap_action_map(state)
        observations = self._state_to_obs_dict(state)

        # TODO make it nicer
        observations["do_preview"] = state._handle_do().world.action_map.map

        return state, observations

    @partial(jax.jit, static_argnums=(0,))
    def _reset_existent(
        self,
        state: State,
        target_map: Array,
        padding_mask: Array,
        trench_axes: Array,
        trench_type: Array,
        dumpability_mask_init: Array,
        env_cfg: EnvConfig,
    ) -> tuple[State, dict[str, Array]]:
        """
        Resets the env, assuming that it already exists.
        """
        state = state._reset(
            env_cfg, target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init
        )
        # TODO remove wrappers from state
        state = TraversabilityMaskWrapper.wrap(state)
        state = LocalMapWrapper.wrap_target_map(state)
        state = LocalMapWrapper.wrap_action_map(state)
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
            padding_mask=state.world.padding_mask.map,
            dumpability_mask=state.world.dumpability_mask.map,
            agent_pos=state.agent.agent_state.pos_base,
            base_dir=state.agent.agent_state.angle_base,
            cabin_dir=state.agent.agent_state.angle_cabin,
            agent_width=state.agent.width,
            agent_height=state.agent.height,
        )

        imgs_local = state.world.local_map_action_neg.map

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

    def render_obs_pygame(
        self,
        obs: dict[str, Array],
        info=None,
        generate_gif : bool = False,
    ) -> Array:
        """
        Renders the environment at a given observation.

        # TODO write a cleaner rendering engine
        """
        if info is not None:
            target_tiles = info["target_tiles"]
            do_preview = info["do_preview"]
        else:
            target_tiles = None
            do_preview = None

        self.rendering_engine.run(
            active_grid=obs["action_map"],
            target_grid=obs["target_map"],
            agent_pos=obs["agent_state"][..., [0, 1]],
            base_dir=obs["agent_state"][..., [2]],
            cabin_dir=obs["agent_state"][..., [3]],
            generate_gif=generate_gif,
            # agent_width=obs["agent_width"],
            # agent_height=obs["agent_height"],
        )

    def render_obs(
        self,
        obs: dict[str, Array],
        key_handler: Callable | None = None,
        mode: str = "human",
        block: bool = False,
        tile_size: int = 32,
        info=None,
    ) -> Array:
        """
        Renders the environment at a given observation.

        # TODO write a cleaner rendering engine
        """
        if info is not None:
            target_tiles = info["target_tiles"]
            do_preview = info["do_preview"]
        else:
            target_tiles = None
            do_preview = None

        imgs_global = self.rendering_engine.render_global(
            tile_size=tile_size,
            active_grid=obs["action_map"],
            target_grid=obs["target_map"],
            padding_mask=obs["padding_mask"],
            dumpability_mask=obs["dumpability_mask"],
            agent_pos=obs["agent_state"][..., [0, 1]],
            base_dir=obs["agent_state"][..., [2]],
            cabin_dir=obs["agent_state"][..., [3]],
            agent_width=obs["agent_width"],
            agent_height=obs["agent_height"],
            target_tiles=target_tiles,
            do_preview=do_preview,
        )

        imgs_local = obs["local_map_action_neg"]

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
        self,
        state: State,
        action: Action,
        target_map: Array,
        padding_mask: Array,
        trench_axes: Array,
        trench_type: Array,
        dumpability_mask_init: Array,
        env_cfg: EnvConfig,
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
        new_state = LocalMapWrapper.wrap_target_map(new_state)
        new_state = LocalMapWrapper.wrap_action_map(new_state)

        observations = self._state_to_obs_dict(new_state)

        done, task_done = state._is_done(
            new_state.world.action_map.map,
            new_state.world.target_map.map,
            new_state.agent.agent_state.loaded,
        )

        new_state, observations = jax.lax.cond(
            done,
            self._reset_existent,
            lambda x, y, z, k, w, j, l: (new_state, observations),
            new_state,
            target_map,
            padding_mask,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            env_cfg,
        )

        infos = new_state._get_infos(action, task_done)

        observations = self._update_obs_with_info(observations, infos)

        return new_state, (observations, reward, done, infos)

    @staticmethod
    def _update_obs_with_info(obs, info):
        obs["do_preview"] = info["do_preview"]
        return obs

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
            "local_map_action_neg": state.world.local_map_action_neg.map,
            "local_map_action_pos": state.world.local_map_action_pos.map,
            "local_map_target_neg": state.world.local_map_target_neg.map,
            "local_map_target_pos": state.world.local_map_target_pos.map,
            "traversability_mask": state.world.traversability_mask.map,
            "action_map": state.world.action_map.map,
            "target_map": state.world.target_map.map,
            "agent_width": state.agent.width,
            "agent_height": state.agent.height,
            "padding_mask": state.world.padding_mask.map,
            "dig_map": state.world.dig_map.map,
            "dumpability_mask": state.world.dumpability_mask.map,
        }


class TerraEnvBatch:
    """
    Takes care of the parallelization of the environment.
    """

    def __init__(
        self,
        batch_cfg: BatchConfig = BatchConfig(),
        rendering: bool = False,
        n_envs_x_rendering: int = 1,
        n_envs_y_rendering: int = 1,
        display: bool = False,
        rendering_engine: str = "numpy",
    ) -> None:
        self.terra_env = TerraEnv.new(
            rendering=rendering,
            n_envs_x=n_envs_x_rendering,
            n_envs_y=n_envs_y_rendering,
            display=display,
            rendering_engine=rendering_engine,
        )
        self.batch_cfg = batch_cfg
        self.maps_buffer = init_maps_buffer(batch_cfg)

    def reset(self, seeds: Array, env_cfgs: EnvConfig) -> State:
        (
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            maps_buffer_keys,
        ) = jax.vmap(self.maps_buffer.get_map_init)(seeds, env_cfgs)
        return (
            *self._reset(
                seeds, target_maps, padding_masks, trench_axes, trench_type, dumpability_mask_init, env_cfgs
            ),
            maps_buffer_keys,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _reset(
        self,
        seeds: Array,
        target_maps: Array,
        padding_masks: Array,
        trench_axes: Array,
        trench_type: Array,
        dumpability_mask_init: Array,
        env_cfgs: EnvConfig,
    ) -> State:
        return jax.vmap(self.terra_env.reset)(
            seeds, target_maps, padding_masks, trench_axes, trench_type, dumpability_mask_init, env_cfgs
        )

    def step(
        self,
        states: State,
        actions: Action,
        env_cfgs: EnvConfig,
        maps_buffer_keys: jax.random.KeyArray,
    ) -> tuple[State, tuple[dict, Array, Array, dict]]:
        (
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            maps_buffer_keys,
        ) = jax.vmap(self.maps_buffer.get_map)(maps_buffer_keys, env_cfgs)
        return (
            *self._step(
                states,
                actions,
                target_maps,
                padding_masks,
                trench_axes,
                trench_type,
                dumpability_mask_init,
                env_cfgs,
            ),
            maps_buffer_keys,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step(
        self,
        states: State,
        actions: Action,
        target_maps: Array,
        padding_masks: Array,
        trench_axes: Array,
        trench_type: Array,
        dumpability_mask_init: Array,
        env_cfgs: EnvConfig,
    ) -> tuple[State, tuple[dict, Array, Array, dict]]:
        states, (obs, rewards, dones, infos) = jax.vmap(self.terra_env.step)(
            states,
            actions,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            env_cfgs,
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

    @property
    def observation_shapes(self) -> dict[str, tuple]:
        """
        Returns a dict that has:
            - key: name of the input feature (e.g. "local_map")
            - value: the tuple representing the shape of the input feature
        """
        return {
            "agent_states": (6,),
            "local_map_action_neg": (
                self.batch_cfg.agent.angles_cabin,
                self.batch_cfg.agent.max_arm_extension + 1,
            ),
            "local_map_action_pos": (
                self.batch_cfg.agent.angles_cabin,
                self.batch_cfg.agent.max_arm_extension + 1,
            ),
            "local_map_target_neg": (
                self.batch_cfg.agent.angles_cabin,
                self.batch_cfg.agent.max_arm_extension + 1,
            ),
            "local_map_target_pos": (
                self.batch_cfg.agent.angles_cabin,
                self.batch_cfg.agent.max_arm_extension + 1,
            ),
            "action_map": (
                self.batch_cfg.maps.max_width,
                self.batch_cfg.maps.max_height,
            ),
            "target_map": (
                self.batch_cfg.maps.max_width,
                self.batch_cfg.maps.max_height,
            ),
            "traversability_mask": (
                self.batch_cfg.maps.max_width,
                self.batch_cfg.maps.max_height,
            ),
            "do_preview": (
                self.batch_cfg.maps.max_width,
                self.batch_cfg.maps.max_height,
            ),
            "dig_map": (
                self.batch_cfg.maps.max_width,
                self.batch_cfg.maps.max_height,
            ),
            "dumpability_mask": (
                self.batch_cfg.maps.max_width,
                self.batch_cfg.maps.max_height,
            ),
        }
