from collections.abc import Callable
from functools import partial
from typing import NamedTuple
from typing import Any, Optional, Tuple


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
from terra.curriculum import CurriculumManager
import pygame as pg
from terra.viz.game.game import Game
from terra.viz.game.settings import MAP_TILES


class TimeStep(NamedTuple):
    state: State
    observation: dict[str, jax.Array]
    reward: jax.Array
    done: jax.Array
    info: dict
    env_cfg: EnvConfig


class TerraEnv(NamedTuple):
    rendering_engine: Game | None = None

    @classmethod
    def new(
        cls,
        maps_size_px: int,
        rendering: bool = False,
        n_envs_x: int = 1,
        n_envs_y: int = 1,
        display: bool = False,
    ) -> "TerraEnv":
        re = None
        tile_size_rendering = MAP_TILES // maps_size_px
        if rendering:
            pg.init()
            pg.mixer.init()
            display_dims = (
                n_envs_y * (maps_size_px + 4) * tile_size_rendering
                + 4 * tile_size_rendering,
                n_envs_x * (maps_size_px + 4) * tile_size_rendering
                + 4 * tile_size_rendering,
            )
            if not display:
                print("TerraEnv: disabling display...")
                screen = pg.display.set_mode(
                    display_dims, pg.FULLSCREEN | pg.HIDDEN
                )
            else:
                screen = pg.display.set_mode(display_dims)
            surface = pg.Surface(display_dims, pg.SRCALPHA)
            clock = pg.time.Clock()
            re = Game(
                screen,
                surface,
                clock,
                maps_size_px=maps_size_px,
                n_envs_x=n_envs_x,
                n_envs_y=n_envs_y,
                display=display,
            )
        return TerraEnv(rendering_engine=re)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: jax.random.PRNGKey,
        target_map: Array,
        padding_mask: Array,
        trench_axes: Array,
        trench_type: Array,
        dumpability_mask_init: Array,
        action_map: Array,
        env_cfg: EnvConfig,
        custom_pos: Optional[Tuple[int, int]] = None,
        custom_angle: Optional[int] = None,
    ) -> tuple[State, dict[str, Array]]:
        """
        Resets the environment using values from config files, and a seed.
        """
        state = State.new(
            key,
            env_cfg,
            target_map,
            padding_mask,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            action_map,
            custom_pos,
            custom_angle,
        )
        state = self.wrap_state(state)

        observations = self._state_to_obs_dict(state)
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
        action_map: Array,
        env_cfg: EnvConfig,
    ) -> tuple[State, dict[str, Array]]:
        """
        Resets the env, assuming that it already exists.
        """
        state = state._reset(
            env_cfg,
            target_map,
            padding_mask,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            action_map
        )
        state = self.wrap_state(state)
        observations = self._state_to_obs_dict(state)
        return state, observations


    def render_obs_pygame(
        self,
        obs: dict[str, Array],
        info=None,
        generate_gif: bool = False,
    ) -> Array:
        """
        Renders the environment at a given observation.
        """
        if info is not None:
            target_tiles = info.get("target_tiles", None)
        else:
            target_tiles = None

        self.rendering_engine.run(
            active_grid=obs["action_map"],
            target_grid=obs["target_map"],
            padding_mask=obs["padding_mask"],
            dumpability_mask=obs["dumpability_mask"],
            agent_pos=obs["agent_state"][..., [0, 1]],
            base_dir=obs["agent_state"][..., [2]],
            cabin_dir=obs["agent_state"][..., [3]],
            loaded=obs["agent_state"][..., [5]],
            target_tiles=target_tiles,
            generate_gif=generate_gif,
            info=info,  # Pass the entire info object
        )

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
        action_map: Array,
        env_cfg: EnvConfig,
    ) -> TimeStep:
        new_state = state._step(action)
        reward = state._get_reward(new_state, action)
        new_state = self.wrap_state(new_state)
        obs = self._state_to_obs_dict(new_state)

        done, task_done = state._is_done(
            new_state.world.action_map.map,
            new_state.world.target_map.map,
            new_state.agent.agent_state.loaded,
        )

        def _reset_branch(s, o, cfg):
            s_reset, o_reset = self._reset_existent(
                s,
                target_map,
                padding_mask,
                trench_axes,
                trench_type,
                dumpability_mask_init,
                action_map,
                cfg,
            )
            return s_reset, o_reset, cfg

        def _nominal_branch(s, o, cfg):
            return s, o, cfg

        new_state, obs, env_cfg = jax.lax.cond(
            done,
            _reset_branch,
            _nominal_branch,
            new_state,
            obs,
            env_cfg,
        )

        infos = new_state._get_infos(action, task_done)
        return TimeStep(
            state=new_state,
            observation=obs,
            reward=reward,
            done=done,
            info=infos,
            env_cfg=env_cfg,   # now the right, possibly flipped `apply_trench_rewards`
        )

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
                state.agent.agent_state.wheel_angle,
                state.agent.agent_state.loaded,
            ]
        )
        # Note: not all of those fields are used by the network for training!
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
        shuffle_maps: bool = False,
        single_map_path: str = None,
    ) -> None:
        self.maps_buffer, self.batch_cfg = init_maps_buffer(batch_cfg, shuffle_maps, single_map_path)
        self.terra_env = TerraEnv.new(
            maps_size_px=self.batch_cfg.maps_dims.maps_edge_length,
            rendering=rendering,
            n_envs_x=n_envs_x_rendering,
            n_envs_y=n_envs_y_rendering,
            display=display,
        )
        max_curriculum_level = len(batch_cfg.curriculum_global.levels) - 1
        max_steps_in_episode_per_level = jnp.array(
            [
                level["max_steps_in_episode"]
                for level in batch_cfg.curriculum_global.levels
            ],
            dtype=jnp.int32,
        )
        apply_trench_rewards_per_level = jnp.array(
            [
                level["apply_trench_rewards"]
                for level in batch_cfg.curriculum_global.levels
            ],
            dtype=jnp.bool_,
        )
        reward_type_per_level = jnp.array(
            [level["rewards_type"] for level in batch_cfg.curriculum_global.levels],
            dtype=jnp.int32,
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
        tile_size = (
            self.batch_cfg.maps.edge_length_m
            / self.batch_cfg.maps_dims.maps_edge_length
        )
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
        n_envs = env_cfgs.agent.dig_depth.shape[
            0
        ]  # leading dimension of any field in the config is the number of envs
        tile_size = jnp.repeat(jnp.array([tile_size], dtype=jnp.float32), n_envs)
        agent_width = jnp.repeat(jnp.array([agent_width], dtype=jnp.int32), n_envs)
        agent_height = jnp.repeat(jnp.array([agent_height], dtype=jnp.int32), n_envs)
        edge_length_px = jnp.repeat(
            jnp.array([self.batch_cfg.maps_dims.maps_edge_length], dtype=jnp.int32),
            n_envs,
        )
        env_cfgs = env_cfgs._replace(
            tile_size=tile_size,
            agent=env_cfgs.agent._replace(width=agent_width, height=agent_height),
            maps=env_cfgs.maps._replace(edge_length_px=edge_length_px),
        )
        return env_cfgs

    def _get_map_init(self, key: jax.random.PRNGKey, env_cfgs: EnvConfig):
        return jax.vmap(self.maps_buffer.get_map_init)(key, env_cfgs)

    def _get_map(self, maps_buffer_keys: jax.random.PRNGKey, env_cfgs: EnvConfig):
        return jax.vmap(self.maps_buffer.get_map)(maps_buffer_keys, env_cfgs)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, env_cfgs: EnvConfig, rng_key: jax.random.PRNGKey, 
          custom_pos: Optional[Tuple[int, int]] = None, 
          custom_angle: Optional[int] = None) -> State:
        
        env_cfgs = self.curriculum_manager.reset_cfgs(env_cfgs)

        env_cfgs = self.update_env_cfgs(env_cfgs)
        (
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            action_maps,
            new_rng_key,
        ) = self._get_map_init(rng_key, env_cfgs)

        timestep = jax.vmap(
            self.terra_env.reset,
            in_axes=(0,0,0,0,0,0,0,0,None,None)
            )(
                rng_key,
                target_maps,
                padding_masks,
                trench_axes,
                trench_type,
                dumpability_mask_init,
                action_maps,
                env_cfgs,
                custom_pos,
                custom_angle,
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
            action_maps,
            maps_buffer_keys,
        ) = self._get_map(maps_buffer_keys, timestep.env_cfg)
        #print(f"Actions: {actions}, type: {type(actions)}")

        # Step the environment
        timestep = jax.vmap(self.terra_env.step)(
            timestep.state,
            actions,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            action_maps,
            timestep.env_cfg,
        )
        return timestep

    @property
    def actions_size(self) -> int:
        """
        Number of actions played at every env step.
        """
        return self.num_actions

    @property
    def num_actions(self) -> int:
        """
        Total number of actions
        """
        return self.batch_cfg.action_type.get_num_actions()


