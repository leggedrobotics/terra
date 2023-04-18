import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from typing import NamedTuple
from src.config import EnvConfig
from src.map import GridWorld
from src.agent import Agent
from src.actions import Action, TrackedActionType
from src.utils import increase_angle_circular, decrease_angle_circular, Float, IntLowDim

class State(NamedTuple):
    """
    Stores the current state of the environment.
    Given perfect information, the observation corresponds to the state.
    """
    seed: jnp.uint32

    env_cfg: EnvConfig

    world: GridWorld
    agent: Agent

    env_steps: int

    @classmethod
    def new(cls, seed: int, env_cfg: EnvConfig) -> "State":
        key = jax.random.PRNGKey(seed)
        world = GridWorld.new(jnp.uint32(seed), env_cfg)

        key, subkey = jax.random.split(key)

        agent = Agent.new(env_cfg)
        agent = jax.tree_map(lambda x: x if isinstance(x, Array) else jnp.array(x), agent)
        # TODO: implement here multiple agents if required (see JUX)

        return State(
            seed=jnp.uint32(seed),
            env_cfg=env_cfg,
            world=world,
            agent=agent,
            env_steps=0
        )

    def _step(self, action: Action) -> "State":
        # TrackedAction type only

        # The DO_NOTHING action should not be played
        # valid_action_mask = (
        #     (action > TrackedActionType.DO_NOTHING) &
        #     (action < TrackedActionType.DO)
        # )

        state = jax.lax.cond(
            action == TrackedActionType.FORWARD,
            self._handle_move_forward,
            lambda: jax.lax.cond(
                action == TrackedActionType.BACKWARD,
                self._handle_move_backward,
                lambda: jax.lax.cond(
                    action == TrackedActionType.CLOCK,
                    self._handle_clock,
                    lambda: jax.lax.cond(
                        action == TrackedActionType.ANTICLOCK,
                        self._handle_anticlock,
                        self._do_nothing
                    )
                )
            )
        )

        return state
    
    def _do_nothing(self):
        return self

    def _handle_move_forward_naive(self):
        """
        Non-vectorized version
        """
        base_orientation = self.agent.agent_state.angle_base
        assert base_orientation.item() in (0, 1, 2, 3)

        move_tiles = self.env_cfg.agent.move_tiles
        agent_width = self.env_cfg.agent.width
        agent_height = self.env_cfg.agent.height

        if base_orientation.item() in (0, 2):
            agent_x_dim = agent_width
            agent_y_dim = agent_height
        elif base_orientation.item() in (1, 3):
            agent_x_dim = agent_height
            agent_y_dim = agent_width
        
        agent_occupancy_x = int(move_tiles + np.ceil(agent_x_dim / 2).item())
        agent_occupancy_y = int(move_tiles + np.ceil(agent_y_dim / 2).item())

        map_width = self.world.width
        map_height = self.world.height
        new_pos_base = self.agent.agent_state.pos_base

        if base_orientation.item() == 0:
            # positive y
            if new_pos_base[1] + agent_occupancy_y < map_height:
                new_pos_base = new_pos_base.at[1].add(move_tiles)
        elif base_orientation.item() == 2:
            # negative y
            if new_pos_base[1] - agent_occupancy_y >= 0:
                new_pos_base = new_pos_base.at[1].add(-move_tiles)
        elif base_orientation.item() == 3:
            # positive x
            if new_pos_base[0] + agent_occupancy_x < map_width:
                new_pos_base = new_pos_base.at[0].add(move_tiles)
        elif base_orientation.item() == 1:
            # negative x
            if new_pos_base[0] - agent_occupancy_x >= 0:
                new_pos_base = new_pos_base.at[0].add(-move_tiles)
        
        assert 0 <= new_pos_base[0] < map_width
        assert 0 <= new_pos_base[1] < map_height
        
        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    pos_base=new_pos_base
                )
            )
        )
    
    @staticmethod
    def _base_orientation_to_one_hot_forward(base_orientation: IntLowDim):
        return jax.nn.one_hot(base_orientation, 4, dtype=IntLowDim)
    
    def _base_orientation_to_one_hot_backwards(self, base_orientation: IntLowDim):
        fwd_to_bkwd_transformation = jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=IntLowDim)
        orientation_one_hot = self._base_orientation_to_one_hot_forward(base_orientation)
        return orientation_one_hot @ fwd_to_bkwd_transformation
    
    @staticmethod
    def _get_agent_corners(pos_base: Array,
                           base_orientation:IntLowDim,
                           agent_width: IntLowDim,
                           agent_height: IntLowDim):
        """
        Gets the coordinates of the 4 corners of the agent.
        """
        orientation_vector_xy = jax.nn.one_hot(base_orientation % 2, 2, dtype=IntLowDim)
        agent_xy_matrix = jnp.array([[agent_width, agent_height],
                                     [agent_height, agent_width]], dtype=IntLowDim)
        agent_xy_dimensions = orientation_vector_xy @ agent_xy_matrix

        x_base = pos_base[0]
        y_base = pos_base[1]
        x_half_dim = jnp.floor(agent_xy_dimensions[0, 0] / 2)
        y_half_dim = jnp.floor(agent_xy_dimensions[0, 1] / 2)

        agent_corners = jnp.array([
            [x_base + x_half_dim, y_base + y_half_dim],
            [x_base - x_half_dim, y_base + y_half_dim],
            [x_base + x_half_dim, y_base - y_half_dim],
            [x_base - x_half_dim, y_base - y_half_dim]
        ])
        return agent_corners
    
    def _move_on_orientation(self, orientation_vector: Array) -> "State":
        
        move_tiles = self.env_cfg.agent.move_tiles
        map_width = self.world.width
        map_height = self.world.height
        new_pos_base = self.agent.agent_state.pos_base

        # assert (
        #     (base_orientation[0] == 0) |
        #     (base_orientation[0] == 1) |
        #     (base_orientation[0] == 2) |
        #     (base_orientation[0] == 3)
        # )

        # Propagate action
        possible_deltas_xy = jnp.array([
            [0, move_tiles],
            [-move_tiles, 0],
            [0, -move_tiles],
            [move_tiles, 0]
        ],
        dtype=IntLowDim)
        delta_xy = orientation_vector @ possible_deltas_xy

        new_pos_base = (new_pos_base + delta_xy)[0]
        
        # Get occupancy of the agent based on its position and orientation
        agent_corners_xy = self._get_agent_corners(new_pos_base,
                                                   base_orientation = self.agent.agent_state.angle_base,
                                                   agent_width = self.env_cfg.agent.width,
                                                   agent_height = self.env_cfg.agent.height,
                                                   )

        # Compute mask (if to apply the action based on the agent occupancy vs map position)
        # valid_move_mask = [1, 0] means that the move is not valid
        # valid_move_mask = [0, 1] means that the move is valid

        # conditions = jnp.array([
        #     [new_pos_base[1] + agent_occupancy_xy[0, 1] < map_height],
        #     [new_pos_base[0] - agent_occupancy_xy[0, 0] >= 0],
        #     [new_pos_base[1] - agent_occupancy_xy[0, 1] >= 0],
        #     [new_pos_base[0] + agent_occupancy_xy[0, 0] < map_width]
        # ])
        # valid_move = orientation_vector @ conditions

        print(f"{agent_corners_xy=}")

        # valid_matrix =  < jnp.array([map_width, map_height])
        valid_matrix_bottom = jnp.array([0, 0]) <= agent_corners_xy
        valid_matrix_up = agent_corners_xy < jnp.array([map_width, map_height])

        valid_move = jnp.all(jnp.concatenate((valid_matrix_bottom[None], valid_matrix_up[None]), axis=0))

        print(f"{valid_move=}")

        valid_move_mask = jax.nn.one_hot(valid_move.astype(IntLowDim), 2, dtype=IntLowDim)

        print(f"{valid_move_mask=}")

        # Apply mask
        old_new_pos_base = jnp.array([
            self.agent.agent_state.pos_base,
            new_pos_base
        ])
        new_pos_base = (valid_move_mask @ old_new_pos_base)

        print(f"{new_pos_base=}")
        
        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    pos_base=new_pos_base
                )
            )
        )

    def _handle_move_forward(self) -> "State":
        base_orientation = self.agent.agent_state.angle_base
        orientation_vector = self._base_orientation_to_one_hot_forward(base_orientation)
        return self._move_on_orientation(orientation_vector)
    
    def _handle_move_backward(self) -> "State":
        base_orientation = self.agent.agent_state.angle_base
        orientation_vector = self._base_orientation_to_one_hot_backwards(base_orientation)
        return self._move_on_orientation(orientation_vector)
    
    def _handle_clock(self) -> "State":
        # Rotate
        old_angle_base = self.agent.agent_state.angle_base
        new_angle_base = decrease_angle_circular(old_angle_base, self.env_cfg.agent.angles_base)

        # TODO in case the agent can reach the limit of the map (currently not possible)
        # 1. Check occupancy
        # 2. Apply or mask action

        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    angle_base=new_angle_base
                )
            )
        )
    
    def _handle_anticlock(self) -> "State":
        old_angle_base = self.agent.agent_state.angle_base
        new_angle_base = increase_angle_circular(old_angle_base, self.env_cfg.agent.angles_base)
        
        # TODO in case the agent can reach the limit of the map (currently not possible)
        # 1. Check occupancy
        # 2. Apply or mask action
        
        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    angle_base=new_angle_base
                )
            )
        )

    def _get_reward(self) -> Float:
        pass

    def _is_done(self) -> jnp.bool_:
        pass
