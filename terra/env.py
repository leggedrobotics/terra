from collections.abc import Callable
from functools import partial

import jax
from jax import Array

from terra.actions import Action
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.state import State
from terra.wrappers import LocalMapWrapper
from terra.wrappers import TraversabilityMaskWrapper


class TerraEnv:
    def __init__(
        self, env_cfg: EnvConfig = EnvConfig(), rendering: bool = False
    ) -> None:
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
        observations = State.new(seed, self.env_cfg)
        # TODO remove wrap from here -- the wrapped objects will stay in the state this way
        observations = TraversabilityMaskWrapper.wrap(observations)
        observations = LocalMapWrapper.wrap(observations)
        return observations

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
                self.window.show_img(img_global, img_local, mode)
                self.window.reg_key_handler(key_handler)
                self.window.show(block)
        if mode == "gif":
            self.window.set_title(
                title=f"Arm extension = {state.agent.agent_state.arm_extension.item()}"
            )
            self.window.show_img(img_global, img_local, mode)
            # self.window.show(block)

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

        # jax.debug.print("tm = {x}", x=observations.world.traversability_mask.map.shape)
        # jax.debug.print("lm = {x}", x=observations.world.local_map.map.shape)

        # jax.debug.print("Reward = {x}", x=reward)
        # jax.debug.print("Dones = {x}", x=dones)
        # jax.debug.print("local map = \n{x}", x=observations.world.local_map.map.T)

        return new_state, (observations, reward, dones, infos)


class TerraEnvBatch:
    """
    Takes care of the parallelization of the environment.
    """

    def __init__(
        self,
        env_cfg: EnvConfig = EnvConfig(),
        batch_cfg: BatchConfig = BatchConfig(),
        rendering: bool = False,
    ) -> None:
        self.terra_env = TerraEnv(env_cfg, rendering=rendering)
        self.batch_cfg = batch_cfg
        self.env_cfg = env_cfg

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
            "agent_states": (4,),
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
