from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from terra.actions import Action
from terra.actions import TrackedActionType
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.maps_buffer import init_maps_buffer
from terra.state import METRIC_COMPONENT_KEYS
from terra.state import REWARD_COMPONENT_KEYS
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
        baseline_map_size = 64
        tile_size_rendering = MAP_TILES // baseline_map_size  # 192//64 = 3
        if rendering:
            pg.init()
            pg.mixer.init()
            # Map pixel size per env = maps_size_px * tile_size_rendering (constant tile size)
            total_display_size = maps_size_px * tile_size_rendering
            display_dims = (
                n_envs_y * (total_display_size + 4 * tile_size_rendering)
                + 4 * tile_size_rendering,
                n_envs_x * (total_display_size + 4 * tile_size_rendering)
                + 4 * tile_size_rendering,
            )
            if not display:
                # One-time notice suppressed to avoid recurring prints
                screen = pg.display.set_mode(display_dims, pg.HIDDEN)

                # screen = pg.display.set_mode(
                #     display_dims, pg.FULLSCREEN | pg.HIDDEN
                # )# Set the display to the screen
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
        foundation_border_axes: Array,
        foundation_border_type: Array,
        dumpability_mask_init: Array,
        action_map: Array,
        distance_map: Array,
        env_cfg: EnvConfig,
    ) -> TimeStep:
        """
        Resets the environment using values from config files, and a seed.
        """
        # One-time debug can be done at trainer; avoid recurring prints here
        state = State.new(
            key,
            env_cfg,
            target_map,
            padding_mask,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_map,
            distance_map_override=distance_map,
        )
        state = self.wrap_state(state)

        observations = self._state_to_obs_dict(state)
        dummy_info = {
            "action_mask": observations["action_mask"],
            "edge_features": observations["edge_features"],
            "episode_progress": observations["episode_progress"],
            "final_observation": observations,
            "target_tiles": jnp.zeros(
                (state.world.width * state.world.height,), dtype=jnp.bool_
            ),
            "task_done": jnp.zeros((), dtype=jnp.bool_),
            "timeout_done": jnp.zeros((), dtype=jnp.bool_),
            "reward_components": self._zero_reward_components(state),
        }

        return TimeStep(
            state=state,
            observation=observations,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
            info=dummy_info,
            env_cfg=env_cfg,
        )

    @staticmethod
    def _zero_reward_components(state: State) -> dict[str, Array]:
        components = {
            key: jnp.float32(0.0)
            for key in REWARD_COMPONENT_KEYS + METRIC_COMPONENT_KEYS
        }
        components.update({
            "agent_rewards": jnp.zeros((4,), dtype=jnp.float32),
            "agent_active": state.agent.agent_active.astype(jnp.int32),
            "num_agents": jnp.asarray(state.agent.num_agents, dtype=jnp.int32),
        })
        return components

    @staticmethod
    def wrap_state(state: State, update_reachability: jnp.bool_ = jnp.bool_(True)) -> State:
        state = TraversabilityMaskWrapper.wrap(
            state,
            update_reachability=update_reachability,
        )
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
        foundation_border_axes: Array,
        foundation_border_type: Array,
        dumpability_mask_init: Array,
        action_map: Array,
        distance_map: Array,
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
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_map,
            distance_map_override=distance_map,
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

        # Pass all agent states and masks to the renderer
        self.rendering_engine.run(
            active_grid=obs["action_map"],
            target_grid=obs["target_map"],
            padding_mask=obs["padding_mask"],
            dumpability_mask=obs["dumpability_mask"],
            interaction_mask=obs["interaction_mask"],  # [H, W] - dig/dump cones for all active agents
            agent_states=obs["agent_states"],  # [MAX_AGENTS, 8] with active agent at index 0
            agent_active=obs["agent_active"],  # [MAX_AGENTS] mask
            num_agents=obs["num_agents"],      # scalar
            generate_gif=generate_gif,
            target_tiles=target_tiles,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_no_reset(
        self,
        state: State,
        action: Action,
        env_cfg: EnvConfig,
    ) -> TimeStep:
        """Step one environment and leave reset handling to the caller."""
        new_state = state._step(action)
        reward, reward_components = state._get_reward(new_state, action)
        # Recompute reachability only for effective DO actions that changed terrain.
        # For all other actions (or no-op DO), keep previous reachability to reduce overhead.
        is_do = action.action[0] == TrackedActionType.DO
        terrain_changed = jnp.any(
            new_state.world.action_map.map != state.world.action_map.map
        )
        update_reachability = jnp.logical_and(is_do, terrain_changed)
        new_state = self.wrap_state(
            new_state,
            update_reachability=update_reachability,
        )
        obs = self._state_to_obs_dict(new_state)
        done, task_done = new_state._is_done(
            new_state.world.action_map.map,
            new_state.world.target_map.map,
        )
        timeout_done = jnp.logical_and(done, jnp.logical_not(task_done))
        infos = {
            "action_mask": obs["action_mask"],
            "edge_features": obs["edge_features"],
            "episode_progress": obs["episode_progress"],
            "final_observation": obs,
            "target_tiles": new_state.world.interaction_mask.map.reshape(-1),
            "task_done": task_done,
            "timeout_done": timeout_done,
            "reward_components": reward_components,
        }
        return TimeStep(
            state=new_state,
            observation=obs,
            reward=reward,
            done=done,
            info=infos,
            env_cfg=env_cfg,
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
        foundation_border_axes: Array,
        foundation_border_type: Array,
        dumpability_mask_init: Array,
        action_map: Array,
        distance_map: Array,
        env_cfg: EnvConfig,
    ) -> TimeStep:
        timestep = self.step_no_reset(state, action, env_cfg)

        def _reset_branch(ts):
            s_reset, o_reset = self._reset_existent(
                ts.state,
                target_map,
                padding_mask,
                trench_axes,
                trench_type,
                foundation_border_axes,
                foundation_border_type,
                dumpability_mask_init,
                action_map,
                distance_map,
                ts.env_cfg,
            )
            infos = {
                **ts.info,
                "action_mask": o_reset["action_mask"],
                "edge_features": o_reset["edge_features"],
                "episode_progress": o_reset["episode_progress"],
                "target_tiles": s_reset.world.interaction_mask.map.reshape(-1),
            }
            return ts._replace(state=s_reset, observation=o_reset, info=infos)

        return jax.lax.cond(
            timestep.done,
            _reset_branch,
            lambda ts: ts,
            timestep,
        )

    @staticmethod
    def _state_to_obs_dict(state: State) -> dict[str, Array]:
        """
        Transforms a State object to an observation dictionary.
        """
        # Build per-agent features for fixed-size agent array and reorder so current agent is first
        # Feature order mirrors legacy single-agent vector
        def _feat(a):
            return jnp.hstack([
                a.pos_base,
                a.angle_base,
                a.angle_cabin,
                a.wheel_angle,
                a.loaded,
                a.agent_type,
                a.shovel_lifted,
            ])

        # Fixed MAX_AGENTS consistent with Agent.new
        MAX_AGENTS = 4
        # Assemble [MAX_AGENTS, feat_dim]
        agents_feat = jnp.stack([_feat(s) for s in state.agent.agent_states[:MAX_AGENTS]], axis=0)

        # Reorder so that the current (acting) agent is first, then pack actives contiguously
        agents_feat_rolled = jnp.roll(agents_feat, -state.agent.current_agent, axis=0)
        agent_active_rolled = jnp.roll(state.agent.agent_active, -state.agent.current_agent, axis=0)
        # Permutation that ensures the acting agent (index 0 after roll) stays first,
        # then other active agents, then inactive agents. Use a static-shape-friendly key.
        idx = jnp.arange(MAX_AGENTS)
        sort_key = (1 - agent_active_rolled) * 2 + (idx != 0)
        perm = jnp.argsort(sort_key)
        agents_feat_ordered = agents_feat_rolled[perm]
        agent_active_ordered = agent_active_rolled[perm]
        num_agents = state.agent.num_agents

        # Note: not all of those fields are used by the network for training!
        action_mask, edge_features = state._get_action_mask_and_edge_features(
            BatchConfig().action_type.do_nothing()
        )

        episode_progress = jnp.clip(
            jnp.asarray(state.env_steps, dtype=jnp.float32)
            / jnp.maximum(
                jnp.asarray(state.env_cfg.max_steps_in_episode, dtype=jnp.float32),
                1.0,
            ),
            0.0,
            1.0,
        )

        return {
            # New multi-agent observation tensors
            "agent_states": agents_feat_ordered,            # [MAX_AGENTS, feat]
            "agent_active": agent_active_ordered,           # [MAX_AGENTS]
            "num_agents": jnp.array(num_agents),            # scalar
            "local_map_action_neg": state.world.local_map_action_neg.map,
            "local_map_action_pos": state.world.local_map_action_pos.map,
            "local_map_target_neg": state.world.local_map_target_neg.map,
            "local_map_target_pos": state.world.local_map_target_pos.map,
            "local_map_dumpability": state.world.local_map_dumpability.map,
            "local_map_obstacles": state.world.local_map_obstacles.map,
            "local_map_border_workspace": state.world.local_map_border_workspace.map,
            "local_map_edge_alignment_error": state.world.local_map_edge_alignment_error.map,
            "local_map_border_diggable": state.world.local_map_border_diggable.map,
            # "local_map_action_neg_2": state.world.local_map_action_neg_2.map,
            # "local_map_action_pos_2": state.world.local_map_action_pos_2.map,
            # "local_map_target_neg_2": state.world.local_map_target_neg_2.map,
            # "local_map_target_pos_2": state.world.local_map_target_pos_2.map,
            # "local_map_dumpability_2": state.world.local_map_dumpability_2.map,
            # "local_map_obstacles_2": state.world.local_map_obstacles_2.map,
            "traversability_mask": state.world.traversability_mask.map,
            "reachability_mask": state.world.reachability_mask.map,
            "action_map": state.world.action_map.map,
            "target_map": state.world.target_map.map,
            "agent_width": state.agent.width,
            "agent_height": state.agent.height,
            "padding_mask": state.world.padding_mask.map,
            "dumpability_mask": state.world.dumpability_mask.map,
            "interaction_mask": state.world.interaction_mask.map,
            "action_mask": action_mask,
            "edge_features": edge_features,
            "episode_progress": episode_progress.reshape(()),
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
        # Recurring prints removed for performance
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
        # Recurring prints removed for performance

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

    def _validate_foundation_border_metadata_requirements(self, env_cfgs: EnvConfig) -> None:
        """
        Hard fail when border-alignment enforcement is enabled but foundation border metadata
        is missing in loaded maps. If enforcement is disabled, allow training to proceed.
        """
        # During jit/pmap tracing, env_cfgs fields can be tracers and cannot be converted
        # to NumPy. Skip Python-side validation in traced contexts.
        if isinstance(env_cfgs.enforce_foundation_border_alignment, jax.core.Tracer):
            return
        enforce = np.asarray(env_cfgs.enforce_foundation_border_alignment)
        if not np.any(enforce):
            return
        border_types = np.asarray(self.maps_buffer.foundation_border_types)
        missing = np.argwhere(border_types <= 0)
        if missing.size > 0:
            raise RuntimeError(
                "Missing `foundation_border_axes_ABC` metadata for curriculum/map indices "
                f"{missing.tolist()} while "
                "`enforce_foundation_border_alignment=True`. "
                "Either regenerate/add border metadata or set "
                "`enforce_foundation_border_alignment=False`."
            )

    def _get_map_init(self, key: jax.random.PRNGKey, env_cfgs: EnvConfig):
        return jax.vmap(self.maps_buffer.get_map_init)(key, env_cfgs)

    def _get_map(self, maps_buffer_keys: jax.random.PRNGKey, env_cfgs: EnvConfig):
        return jax.vmap(self.maps_buffer.get_map)(maps_buffer_keys, env_cfgs)

    def _sample_reset_maps(self, key: jax.random.PRNGKey, env_cfgs: EnvConfig):
        return jax.vmap(self.maps_buffer.sample_map_init)(key, env_cfgs)

    def _prepare_reset_device(self, env_cfgs: EnvConfig, rng_key: jax.random.PRNGKey):
        env_cfgs = self.curriculum_manager.reset_cfgs(env_cfgs)
        env_cfgs = self.update_env_cfgs(env_cfgs)
        (
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_maps,
            distance_maps,
            _,
        ) = self._sample_reset_maps(rng_key, env_cfgs)
        return (
            env_cfgs,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_maps,
            distance_maps,
        )

    def prepare_reset(self, env_cfgs: EnvConfig, rng_key: jax.random.PRNGKey):
        self._validate_foundation_border_metadata_requirements(env_cfgs)
        return jax.vmap(self._prepare_reset_device)(env_cfgs, rng_key)

    @partial(jax.jit, static_argnums=(0,))
    def reset_prepared(
        self,
        env_cfgs: EnvConfig,
        rng_key: jax.random.PRNGKey,
        target_maps: Array,
        padding_masks: Array,
        trench_axes: Array,
        trench_type: Array,
        foundation_border_axes: Array,
        foundation_border_type: Array,
        dumpability_mask_init: Array,
        action_maps: Array,
        distance_maps: Array,
    ) -> TimeStep:
        timestep = jax.vmap(self.terra_env.reset)(
            rng_key,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_maps,
            distance_maps,
            env_cfgs,
        )
        return timestep

    def reset(self, env_cfgs: EnvConfig, rng_key: jax.random.PRNGKey) -> TimeStep:
        self._validate_foundation_border_metadata_requirements(env_cfgs)
        (
            env_cfgs,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_maps,
            distance_maps,
        ) = self._prepare_reset_device(env_cfgs, rng_key)
        return self.reset_prepared(
            env_cfgs,
            rng_key,
            target_maps,
            padding_masks,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_maps,
            distance_maps,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        timestep: TimeStep,
        actions: Action,
        maps_buffer_keys: jax.random.PRNGKey,
    ) -> TimeStep:
        # Update curriculum state from the previous timestep. Reset maps are only
        # needed when the current step ends at least one environment.
        timestep = self.curriculum_manager.update_cfgs(timestep, maps_buffer_keys)
        timestep_no_reset = jax.vmap(self.terra_env.step_no_reset)(
            timestep.state,
            actions,
            timestep.env_cfg,
        )

        def _reset_done_envs(ts: TimeStep) -> TimeStep:
            (
                target_maps,
                padding_masks,
                trench_axes,
                trench_type,
                foundation_border_axes,
                foundation_border_type,
                dumpability_mask_init,
                action_maps,
                distance_maps,
                _,
            ) = self._get_map(maps_buffer_keys, ts.env_cfg)

            def _reset_one(
                ts_one,
                target_map,
                padding_mask,
                trench_axis,
                trench_kind,
                foundation_border_axis,
                foundation_border_kind,
                dumpability_mask,
                action_map,
                distance_map,
            ):
                def _reset_branch(item):
                    state_reset, obs_reset = self.terra_env._reset_existent(
                        item.state,
                        target_map,
                        padding_mask,
                        trench_axis,
                        trench_kind,
                        foundation_border_axis,
                        foundation_border_kind,
                        dumpability_mask,
                        action_map,
                        distance_map,
                        item.env_cfg,
                    )
                    infos = {
                        **item.info,
                        "action_mask": obs_reset["action_mask"],
                        "edge_features": obs_reset["edge_features"],
                        "episode_progress": obs_reset["episode_progress"],
                        "target_tiles": state_reset.world.interaction_mask.map.reshape(-1),
                    }
                    return item._replace(
                        state=state_reset,
                        observation=obs_reset,
                        info=infos,
                    )

                return jax.lax.cond(
                    ts_one.done,
                    _reset_branch,
                    lambda item: item,
                    ts_one,
                )

            return jax.vmap(_reset_one)(
                ts,
                target_maps,
                padding_masks,
                trench_axes,
                trench_type,
                foundation_border_axes,
                foundation_border_type,
                dumpability_mask_init,
                action_maps,
                distance_maps,
            )

        return jax.lax.cond(
            jnp.any(timestep_no_reset.done),
            _reset_done_envs,
            lambda ts: ts,
            timestep_no_reset,
        )
