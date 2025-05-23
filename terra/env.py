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
        progressive_gif: bool = False,
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
                progressive_gif=progressive_gif,
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
        env_cfg: EnvConfig,
        custom_pos: Optional[Tuple[int, int]] = None,
        custom_angle: Optional[int] = None,
    ) -> tuple[State, dict[str, Array]]:
        """
        Resets the environment using values from config files, and a seed.
        """
        # print("terra_env.reset")
        # print(target_map.shape)
        # print(padding_mask.shape)
        # print(trench_axes.shape)
        # print(trench_type.shape)
        # print(dumpability_mask_init.shape)
        # print("finished printing shapes")
        state = State.new(
            key,
            env_cfg,
            target_map,
            padding_mask,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            custom_pos=custom_pos,
            custom_angle=custom_angle
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
            target_tiles = info["target_tiles"]
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
            loaded=obs["agent_state"][..., [4]],
            target_tiles=target_tiles,
            generate_gif=generate_gif,
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
    # @partial(jax.jit, static_argnums=(0,))
    # def step(
    #     self,
    #     state: State,
    #     action: Action,
    #     target_map: Array,
    #     padding_mask: Array,
    #     trench_axes: Array,
    #     trench_type: Array,
    #     dumpability_mask_init: Array,
    #     env_cfg: EnvConfig,
    # ) -> TimeStep:
    #     # Ensure all maps are padded to 128x128 at the beginning
    #     state = self._ensure_padded_state(state, target_size=(128, 128))
    
    #     new_state = state._step(action)
    #     reward = state._get_reward(new_state, action)
    #     new_state = self.wrap_state(new_state)
    #     obs = self._state_to_obs_dict(new_state)
    
    #     done, task_done = state._is_done(
    #         new_state.world.action_map.map,
    #         new_state.world.target_map.map,
    #         new_state.agent.agent_state.loaded,
    #     )
    
    #     # Now both branches will work with 128x128 maps
    #     def _reset_branch(s, o, cfg):
    #         # Ensure target_map and other inputs are also 128x128
    #         padded_target_map = self._ensure_padded_map(target_map, (128, 128))
    #         padded_padding_mask = self._ensure_padded_map(padding_mask, (128, 128))
    #         padded_dumpability_mask_init = self._ensure_padded_map(dumpability_mask_init, (128, 128))
        
    #         s_reset, o_reset = self._reset_existent(
    #             s,
    #             padded_target_map,
    #             padded_padding_mask,
    #             trench_axes,
    #             trench_type,
    #             padded_dumpability_mask_init,
    #             cfg,
    #         )
    #         return s_reset, o_reset, cfg
    
    #     def _nominal_branch(s, o, cfg):
    #         return s, o, cfg
    
    #     new_state, obs, env_cfg = jax.lax.cond(
    #         done,
    #         _reset_branch,
    #         _nominal_branch,
    #         new_state,
    #         obs,
    #         env_cfg,
    #     )
    
    #     infos = new_state._get_infos(action, task_done)
    #     return TimeStep(
    #         state=new_state,
    #         observation=obs,
    #         reward=reward,
    #         done=done,
    #         info=infos,
    #         env_cfg=env_cfg,
    #     )

    # def _ensure_padded_state(self, state, target_size):
    #     """Ensure that all maps in the state are padded to target_size."""
    #     # Create a function to pad each map
    #     def pad_map(map_array, target_size):
    #         current_shape = map_array.shape
    #         if current_shape == target_size:
    #             return map_array
        
    #         # Calculate padding needed
    #         padding_height = max(0, target_size[0] - current_shape[0])
    #         padding_width = max(0, target_size[1] - current_shape[1])
        
    #         # Pad the map (centered)
    #         padding = [
    #             (padding_height//2, padding_height - padding_height//2),
    #             (padding_width//2, padding_width - padding_width//2)
    #         ]
        
    #         # Use jax.numpy.pad instead of tf.pad
    #         return jnp.pad(map_array, padding, mode='constant', constant_values=0)
    
    #     # Apply padding to all maps in the world
    #     padded_world = state.world._replace(
    #         target_map=state.world.target_map._replace(
    #             map=pad_map(state.world.target_map.map, target_size)
    #         ),
    #         action_map=state.world.action_map._replace(
    #             map=pad_map(state.world.action_map.map, target_size)
    #         ),
    #         padding_mask=state.world.padding_mask._replace(
    #             map=pad_map(state.world.padding_mask.map, target_size)
    #         ),
    #         dig_map=state.world.dig_map._replace(
    #             map=pad_map(state.world.dig_map.map, target_size)
    #         ),
    #         dumpability_mask=state.world.dumpability_mask._replace(
    #             map=pad_map(state.world.dumpability_mask.map, target_size)
    #         ),
    #         dumpability_mask_init=state.world.dumpability_mask_init._replace(
    #             map=pad_map(state.world.dumpability_mask_init.map, target_size)
    #         ),
    #         traversability_mask=state.world.traversability_mask._replace(
    #             map=pad_map(state.world.traversability_mask.map, target_size)
    #         ),
    #     )
    
    #     return state._replace(world=padded_world)

    # def _ensure_padded_map(self, map_array, target_size):
    #     """Ensure that a map is padded to target_size."""
    #     current_shape = map_array.shape
    #     if current_shape == target_size:
    #         return map_array
    
    #     padding_height = max(0, target_size[0] - current_shape[0])
    #     padding_width = max(0, target_size[1] - current_shape[1])
    
    #     padding = [
    #         (padding_height//2, padding_height - padding_height//2),
    #         (padding_width//2, padding_width - padding_width//2)
    #     ]
    
    #     return jnp.pad(map_array, padding, mode='constant', constant_values=0)
    
    # @partial(jax.jit, static_argnums=(0,))
    # def step(
    #     self,
    #     state: State,
    #     action: Action,
    #     target_map: Array,
    #     padding_mask: Array,
    #     trench_axes: Array,
    #     trench_type: Array,
    #     dumpability_mask_init: Array,
    #     env_cfg: EnvConfig,
    # ) -> TimeStep:
    #     # Original step logic - no changes to how the environment works internally
    #     new_state = state._step(action)
    #     reward = state._get_reward(new_state, action)
    #     new_state = self.wrap_state(new_state)
    #     obs = self._state_to_obs_dict(new_state)
    
    #     done, task_done = state._is_done(
    #         new_state.world.action_map.map,
    #         new_state.world.target_map.map,
    #         new_state.agent.agent_state.loaded,
    #     )
    
    #     # Original conditional logic
    #     def _reset_branch(s, o, cfg):
    #         s_reset, o_reset = self._reset_existent(
    #             s,
    #             target_map,
    #             padding_mask,
    #             trench_axes,
    #             trench_type,
    #             dumpability_mask_init,
    #             cfg,
    #         )
    #         return s_reset, o_reset, cfg
    
    #     def _nominal_branch(s, o, cfg):
    #         return s, o, cfg
    
    #     # Use a try-except to handle the shape mismatch
    #     try:
    #         new_state, obs, env_cfg = jax.lax.cond(
    #             done,
    #             _reset_branch,
    #             _nominal_branch,
    #             new_state,
    #             obs,
    #             env_cfg,
    #         )
    #     except TypeError as e:
    #         # If we get a shape error, handle reset manually
    #         if done:
    #             new_state, obs = self._reset_existent(
    #                 new_state,
    #                 target_map,
    #                 padding_mask,
    #                 trench_axes,
    #                 trench_type,
    #                 dumpability_mask_init,
    #                 env_cfg,
    #             )
    
    #     infos = new_state._get_infos(action, task_done)
    
    #     # CRITICAL: Before returning observations to the agent, ensure they're consistently sized
    #     # This is the key fix - resize all map observations to 64x64 before sending to the model
    #     consistent_obs = self._resize_observations_to_64x64(obs)
    
    #     return TimeStep(
    #         state=new_state,
    #         observation=consistent_obs,  # Use the consistently sized observations
    #         reward=reward,
    #         done=done,
    #         info=infos,
    #         env_cfg=env_cfg,
    #     )

    def _resize_observations_to_64x64(self, obs):
        """Resize all map observations to 64x64 for consistent model input."""
        resized_obs = dict(obs)  # Create a copy of the observations
    
        # Resize each map observation to 64x64
        map_keys = ['target_map', 'action_map', 'padding_mask', 'dig_map', 
                'dumpability_mask', 'traversability_mask']
    
        for key in map_keys:
            if key in resized_obs:
                # Get the current map
                current_map = resized_obs[key]
            
                # Check shape and resize if needed
                if current_map.shape[-2:] != (64, 64):  # Check the H,W dimensions
                    # Resize to 64x64 using JAX-compatible operations
                    # If it's a batch of maps (e.g., shape [1, 128, 128])
                    if len(current_map.shape) == 3:
                        batch_size = current_map.shape[0]
                        # Use JAX's image resize function 
                        # Note: You may need to adapt this based on your available JAX functions
                        resized = jax.image.resize(
                            current_map, 
                            shape=(batch_size, 64, 64), 
                            method='nearest'
                        )
                    # If it's a single map (e.g., shape [128, 128])
                    else:
                        resized = jax.image.resize(
                            current_map, 
                            shape=(64, 64), 
                            method='nearest'
                        )
                    resized_obs[key] = resized
    
        return resized_obs

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
        shuffle_maps: bool = False,
    ) -> None:
        self.maps_buffer, self.batch_cfg = init_maps_buffer(batch_cfg, shuffle_maps)
        self.terra_env = TerraEnv.new(
            maps_size_px=self.batch_cfg.maps_dims.maps_edge_length,
            rendering=rendering,
            n_envs_x=n_envs_x_rendering,
            n_envs_y=n_envs_y_rendering,
            display=display,
            progressive_gif=progressive_gif,
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

    @partial(jax.jit, static_argnums=(0,3,4))
    def reset(self, env_cfgs: EnvConfig, rng_key: jax.random.PRNGKey, 
          custom_pos: Optional[Tuple[int, int]] = None, 
          custom_angle: Optional[int] = None) -> State:
        
        env_cfgs = self.curriculum_manager.reset_cfgs(env_cfgs)
        #print("terra_env batch.reset")
        #print(env_cfgs)
        env_cfgs = self.update_env_cfgs(env_cfgs)
        (
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            new_rng_key,
        ) = self._get_map_init(rng_key, env_cfgs)
        #jax.debug.print("custom_pos: {}, custom_angle: {}", custom_pos, custom_angle)
        #print(target_maps.shape)
        #print(padding_masks.shape)
        #print(trench_axes.shape)
        timestep = jax.vmap(
            self.terra_env.reset,
            in_axes=(0,0,0,0,0,0,0,None,None)
            )(
                rng_key,
                target_maps,
                padding_masks,
                trench_axes,
                trench_type,
                dumpability_mask_init,
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


