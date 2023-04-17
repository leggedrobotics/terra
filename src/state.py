import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from typing import NamedTuple
from src.config import EnvConfig
from src.map import GridWorld
from src.agent import Agent
from src.actions import Action, TrackedActionType
from src.utils import Float, IntLowDim

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
        valid_action_mask = (
            (action > TrackedActionType.DO_NOTHING) &
            (action < TrackedActionType.DO)
        )

        return self._handle_move_forward()

        # if action == TrackedActionType.FORWARD:
        #     return self._handle_move_forward()
    

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


    def _handle_move_forward(self):
        base_orientation = self.agent.agent_state.angle_base

        # assert (
        #     (base_orientation[0] == 0) |
        #     (base_orientation[0] == 1) |
        #     (base_orientation[0] == 2) |
        #     (base_orientation[0] == 3)
        # )

        move_tiles = self.env_cfg.agent.move_tiles
        agent_width = self.env_cfg.agent.width
        agent_height = self.env_cfg.agent.height
        map_width = self.world.width
        map_height = self.world.height
        new_pos_base = self.agent.agent_state.pos_base

        # One-hot encode orientation
        orientation_vector = jax.nn.one_hot(base_orientation, 4, dtype=IntLowDim)

        # Get occupancy of the agent based on its position and orientation
        orientation_vector_xy = jax.nn.one_hot(base_orientation % 2, 2, dtype=IntLowDim)
        agent_xy_matrix = jnp.array([[agent_width, agent_height],
                                     [agent_height, agent_width]], dtype=IntLowDim)
        agent_xy = orientation_vector_xy @ agent_xy_matrix
        agent_occupancy_xy = IntLowDim(move_tiles + jnp.ceil(agent_xy / 2))

        # Compute mask (if to apply the action based on the agent occupancy vs map position)
        # valid_move_mask = [1, 0] means that the move is not valid
        # valid_move_mask = [0, 1] means that the move is valid
        conditions = jnp.array([
            [new_pos_base[1] + agent_occupancy_xy[0, 1] < map_height],
            [new_pos_base[0] - agent_occupancy_xy[0, 0] >= 0],
            [new_pos_base[1] - agent_occupancy_xy[0, 1] >= 0],
            [new_pos_base[0] + agent_occupancy_xy[0, 0] < map_width]
        ])
        valid_move = orientation_vector @ conditions
        valid_move_mask = jax.nn.one_hot(valid_move[0], 2, dtype=IntLowDim)

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
        
        # assert 0 <= new_pos_base[0] < map_width
        # assert 0 <= new_pos_base[1] < map_height

        # Apply mask
        old_new_pos_base = jnp.array([
            self.agent.agent_state.pos_base,
            new_pos_base
        ])
        new_pos_base = (valid_move_mask @ old_new_pos_base)[0]
        
        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    pos_base=new_pos_base
                )
            )
        )


    def _get_reward(self) -> Float:
        pass

    def _is_done(self) -> jnp.bool_:
        pass
