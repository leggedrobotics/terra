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
from terra.actions import TrackedAction, WheeledAction
from terra.curriculum import CurriculumManager
import pygame as pg
from terra.viz_legacy.rendering import RenderingEngine
from terra.viz_legacy.window import Window
from terra.viz.game.game import Game
from terra.viz.game.settings import TILE_SIZE
from terra.viz.game.settings import MAP_EDGE

class TimeStep(NamedTuple):
    state: State
    observation: dict[str, jax.Array]
    reward: jax.Array
    done: jax.Array
    info: dict
    env_cfg: EnvConfig

class TerraEnv(NamedTuple):
    rendering_engine: Game | RenderingEngine | None = None
    window: Window | None = None  # Note: not used if pygame rendering engine is used

    @classmethod
    def new(cls, rendering: bool = False, n_envs_x: int = 1, n_envs_y: int = 1, display: bool = False, progressive_gif: bool = False, rendering_engine: str = "pygame") -> "TerraEnv":
        re = None
        window = None
        if rendering:
            print(f"Using {rendering_engine} rendering_engine")
            if rendering_engine == "numpy":
                window = Window("Terra", n_envs_x)
                re = RenderingEngine()
            elif rendering_engine == "pygame":
                pg.init()
                pg.mixer.init()
                display_dims = (n_envs_y * (MAP_EDGE + 4) * TILE_SIZE + 4*TILE_SIZE, n_envs_x * (MAP_EDGE + 4) * TILE_SIZE + 4*TILE_SIZE)
                if not display:
                    print("TerraEnv: disabling display...")
                    screen = pg.display.set_mode(display_dims, pg.FULLSCREEN | pg.HIDDEN)
                else:
                    screen = pg.display.set_mode(display_dims)
                surface = pg.Surface(display_dims, pg.SRCALPHA)
                clock = pg.time.Clock()
                re = Game(screen, surface, clock, n_envs_x=n_envs_x, n_envs_y=n_envs_y, display=display, progressive_gif=progressive_gif)
            else:
                raise(ValueError(f"{rendering_engine=}"))
        return TerraEnv(rendering_engine=re, window=window)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: jax.random.PRNGKey,
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
        state = State.new(
            key, env_cfg, target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init
        )
        state = self.wrap_state(state)
        
        observations = self._state_to_obs_dict(state)

        observations["do_preview"] = state._handle_do().world.action_map.map

        dummy_action = BatchConfig().action_type.do_nothing()

        return TimeStep(
            state=state,
            observation=observations,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
            info=state._get_infos(dummy_action, False),
            env_cfg=env_cfg,
        )
    
    @staticmethod
    def wrap_state(state: State) -> State:
        state = TraversabilityMaskWrapper.wrap(state)
        state = LocalMapWrapper.wrap(state)
        return state

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
        state = self.wrap_state(state)
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
            padding_mask=obs["padding_mask"],
            dumpability_mask=obs["dumpability_mask"],
            agent_pos=obs["agent_state"][..., [0, 1]],
            base_dir=obs["agent_state"][..., [2]],
            cabin_dir=obs["agent_state"][..., [3]],
            generate_gif=generate_gif,
            # agent_width=obs["agent_width"],
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
    ) -> TimeStep:
        new_state = state._step(action)
        reward = state._get_reward(new_state, action)
        new_state = self.wrap_state(new_state)
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

        return TimeStep(
            state=new_state,
            observation=observations,
            reward=reward,
            done=done,
            info=infos,
            env_cfg=env_cfg,
        )

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
            "local_map_dumpability": state.world.local_map_dumpability.map,
            "local_map_obstacles": state.world.local_map_obstacles.map,
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
        progressive_gif: bool = False,
        rendering_engine: str = "pygame",
        shuffle_maps: bool = False,
    ) -> None:
        self.terra_env = TerraEnv.new(
            rendering=rendering,
            n_envs_x=n_envs_x_rendering,
            n_envs_y=n_envs_y_rendering,
            display=display,
            rendering_engine=rendering_engine,
            progressive_gif=progressive_gif,
        )
        self.maps_buffer, self.batch_cfg = init_maps_buffer(batch_cfg, shuffle_maps)
        max_curriculum_level = len(batch_cfg.curriculum_global.levels) - 1
        max_steps_in_episode_per_level = jnp.array(
            [level["max_steps_in_episode"] for level in batch_cfg.curriculum_global.levels], dtype=jnp.int32
        )
        apply_trench_rewards_per_level = jnp.array(
            [level["apply_trench_rewards"] for level in batch_cfg.curriculum_global.levels], dtype=jnp.bool_
        )
        reward_type_per_level = jnp.array(
            [level["rewards_type"] for level in batch_cfg.curriculum_global.levels], dtype=jnp.int32
        )
        self.curriculum_manager = CurriculumManager(
            max_level=max_curriculum_level,
            increase_level_threshold=batch_cfg.curriculum_global.increase_level_threshold,
            decrease_level_threshold=batch_cfg.curriculum_global.decrease_level_threshold,
            max_steps_in_episode_per_level=max_steps_in_episode_per_level,
            apply_trench_rewards_per_level=apply_trench_rewards_per_level,
            reward_type_per_level=reward_type_per_level,
            last_level_type=batch_cfg.curriculum_global.last_level_type,
        )

    def update_env_cfgs(self, env_cfgs: EnvConfig) -> EnvConfig:
        tile_size = self.batch_cfg.maps.edge_length_m / self.batch_cfg.maps_dims.maps_edge_length
        print(f"tile_size: {tile_size}")
        agent_w = self.batch_cfg.agent.dimensions.WIDTH
        agent_h = self.batch_cfg.agent.dimensions.HEIGHT
        agent_height = (
            round(agent_w / tile_size)
            if (round(agent_w / tile_size)) % 2 != 0
            else round(agent_w / tile_size) + 1
        )
        agent_width = (
            round(agent_h / tile_size)
            if (round(agent_h / tile_size)) % 2 != 0
            else round(agent_h / tile_size) + 1
        )
        print(f"agent_width: {agent_width}, agent_height: {agent_height}")

        # Repeat to match the number of environments
        n_envs = env_cfgs.agent.dig_depth.shape[0]  # leading dimension of any field in the config is the number of envs
        tile_size = jnp.repeat(jnp.array([tile_size], dtype=jnp.float32), n_envs)
        agent_width = jnp.repeat(jnp.array([agent_width], dtype=jnp.int32), n_envs)
        agent_height = jnp.repeat(jnp.array([agent_height], dtype=jnp.int32), n_envs)
        edge_length_px = jnp.repeat(jnp.array([self.batch_cfg.maps_dims.maps_edge_length], dtype=jnp.int32), n_envs)
        env_cfgs = env_cfgs._replace(
            tile_size=tile_size,
            agent=env_cfgs.agent._replace(
                width=agent_width, height=agent_height
            ),
            maps=env_cfgs.maps._replace(
                edge_length_px=edge_length_px
            ),
        )
        return env_cfgs

    def _get_map_init(self, key: jax.random.PRNGKey, env_cfgs: EnvConfig):
        return jax.vmap(self.maps_buffer.get_map_init)(key, env_cfgs)

    def _get_map(self, maps_buffer_keys: jax.random.PRNGKey, env_cfgs: EnvConfig):
        return jax.vmap(self.maps_buffer.get_map)(maps_buffer_keys, env_cfgs)
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, env_cfgs: EnvConfig, rng_key: jax.random.PRNGKey) -> State:
        target_maps, padding_masks, trench_axes, trench_type, dumpability_mask_init, new_rng_key = self._get_map_init(rng_key, env_cfgs)
        env_cfgs = self.curriculum_manager.reset_cfgs(env_cfgs)
        env_cfgs = self.update_env_cfgs(env_cfgs)
        timestep = jax.vmap(self.terra_env.reset)(
            rng_key, target_maps, padding_masks, trench_axes, trench_type, dumpability_mask_init, env_cfgs
        )
        return timestep

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        timestep: TimeStep,
        actions: Action,
        maps_buffer_keys: jax.random.PRNGKey,
    ) -> tuple[State, tuple[dict, Array, Array, dict]]:
        # Update env_cfgs based on the curriculum, and get the new maps
        timestep = self.curriculum_manager.update_cfgs(timestep, maps_buffer_keys)
        (
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            maps_buffer_keys,
        ) = self._get_map(maps_buffer_keys, timestep.env_cfg)
        # Step the environment
        timestep = jax.vmap(self.terra_env.step)(
            timestep.state,
            actions,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            timestep.env_cfg,
        )
        return timestep

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
            "local_map_dumpability": (
                self.batch_cfg.agent.angles_cabin,
                self.batch_cfg.agent.max_arm_extension + 1,
            ),
            "local_map_obstacles": (
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
