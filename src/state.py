import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from typing import NamedTuple, Tuple
from src.config import EnvConfig
from src.map import GridWorld
from src.agent import Agent
from src.actions import Action, TrackedActionType
from src.utils import (
    apply_rot_transl,
    increase_angle_circular,
    decrease_angle_circular,
    apply_local_cartesian_to_cyl,
    angle_idx_to_rad,
    wrap_angle_rad,
    Float,
    IntLowDim,
    IntMap
)

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

        return State(
            seed=jnp.uint32(seed),
            env_cfg=env_cfg,
            world=world,
            agent=agent,
            env_steps=0
        )

    def _step(self, action: Action) -> "State":
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
                        lambda: jax.lax.cond(
                            action == TrackedActionType.CABIN_CLOCK,
                            self._handle_cabin_clock,
                            lambda: jax.lax.cond(
                                action == TrackedActionType.CABIN_ANTICLOCK,
                                self._handle_cabin_anticlock,
                                lambda: jax.lax.cond(
                                    action == TrackedActionType.EXTEND_ARM,
                                    self._handle_extend_arm,
                                    lambda: jax.lax.cond(
                                        action == TrackedActionType.RETRACT_ARM,
                                        self._handle_retract_arm,
                                        lambda: jax.lax.cond(
                                            action == TrackedActionType.DO,
                                            self._handle_do,
                                            self._do_nothing
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        return state
    
    def _do_nothing(self):
        return self
    
    @staticmethod
    def _base_orientation_to_one_hot_forward(base_orientation: IntLowDim):
        """
        Converts the base orientation (int 0 to N) to a one-hot encoded vector.
        Use for the forward action.
        """
        return jax.nn.one_hot(base_orientation, 4, dtype=IntLowDim)
    
    def _base_orientation_to_one_hot_backwards(self, base_orientation: IntLowDim):
        """
        Converts the base orientation (int 0 to N) to a one-hot encoded vector.
        Use for the backwards action.
        """
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
    
    @staticmethod
    def _get_agent_corners_xy(agent_corners: Array) -> Tuple[Array, Array]:
        """
        Args:
            - agent_corners: (4, 2) Array with agent corners [x, y] column order
        Returns:
            - x: (2, ) Array of min and max x values as [min, max]
            - y: (2, ) Array of min and max y values as [min, max]
        """

        x = jnp.array([
            jnp.min(agent_corners[:, 0]),
            jnp.max(agent_corners[:, 0])
        ])
        y = jnp.array([
            jnp.min(agent_corners[:, 1]),
            jnp.max(agent_corners[:, 1])
        ])
        return x, y
    
    @staticmethod
    def _build_traversability_mask(map: Array) -> Array:
        """
        Args:
            - map: (N, M) Array of ints
        Returns:
            - traversability_mask: (N, M) Array of ints
                1 for non traversable, 0 for traversable
        """
        return (~(map == 0)).astype(IntLowDim)
    
    def _build_one_hot_move_mask(self,
                                 base_position: Array,
                                 base_orientation: Array,
                                 agent_width: IntLowDim,
                                 agent_height: IntLowDim 
                                 ) -> Array:
        """
        Builds a one-hot encoded 2D vector, encoding the masking or not masking of the
        proposed move action.

        - valid_move_mask [1, 0] means that the move is not valid
        - valid_move_mask [0, 1] means that the move is valid

        Args:
            - base_position: (2, ) Array with [x, y] proposed base position
            - base_orientation: (1, ) Array with int-based orientation encoding of the agent (e.g. 3)
            - agent_width: width of the agent
            - agent_height: height of the agent
                Note: the width and height parameters can be exploited to mask out also the tiles occupied
                    during a rotation (e.g. width = height = max(width, height)) 
        Returns:
            - (2, 1) Array, move mask
        """
        map_width = self.world.width
        map_height = self.world.height

        # Get occupancy of the agent based on its position and orientation
        agent_corners_xy = self._get_agent_corners(base_position,
                                                   base_orientation = base_orientation,
                                                   agent_width = agent_width,
                                                   agent_height = agent_height,
                                                   )

        # Map size constraints
        valid_matrix_bottom = jnp.array([0, 0]) <= agent_corners_xy
        valid_matrix_up = agent_corners_xy < jnp.array([map_width, map_height])

        valid_move_map_size = jnp.all(jnp.concatenate((valid_matrix_bottom[None], valid_matrix_up[None]), axis=0))

        # Traversability constraints
        traversability_mask = self._build_traversability_mask(self.world.action_map.map)
        x_minmax_agent, y_minmax_agent = self._get_agent_corners_xy(agent_corners_xy)

        traversability_mask_reduced = jnp.where(
            (jnp.arange(map_width) < x_minmax_agent[0])[:, None].repeat(map_height, axis=1),
            0,
            traversability_mask
        )
        traversability_mask_reduced = jnp.where(
            (jnp.arange(map_width) > x_minmax_agent[1])[:, None].repeat(map_height, axis=1),
            0,
            traversability_mask_reduced
        )
        traversability_mask_reduced = jnp.where(
            (jnp.arange(map_height) < y_minmax_agent[0])[None].repeat(map_width, axis=0),
            0,
            traversability_mask_reduced
        )
        traversability_mask_reduced = jnp.where(
            (jnp.arange(map_height) > y_minmax_agent[1])[None].repeat(map_width, axis=0),
            0,
            traversability_mask_reduced
        )
        valid_move_traversability = jnp.all(traversability_mask_reduced == 0)
        valid_move = jnp.logical_and(valid_move_map_size, valid_move_traversability)
        return jax.nn.one_hot(valid_move.astype(IntLowDim), 2, dtype=IntLowDim)
    
    def _move_on_orientation(self, orientation_vector: Array) -> "State":
        
        move_tiles = self.env_cfg.agent.move_tiles
        new_pos_base = self.agent.agent_state.pos_base

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
        
        valid_move_mask = self._build_one_hot_move_mask(new_pos_base,
                                                        self.agent.agent_state.angle_base,
                                                        self.env_cfg.agent.width,
                                                        self.env_cfg.agent.height)

        old_new_pos_base = jnp.array([
            self.agent.agent_state.pos_base,
            new_pos_base
        ])
        new_pos_base = (valid_move_mask @ old_new_pos_base)
        
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
    
    def _apply_base_rotation_mask(self,new_angle_base: Array) -> Array:
        max_agent_dim = jnp.max(jnp.array([self.env_cfg.agent.width,self.env_cfg.agent.height], dtype=IntLowDim))
        valid_move_mask = self._build_one_hot_move_mask(self.agent.agent_state.pos_base,
                                                        new_angle_base,
                                                        max_agent_dim,
                                                        max_agent_dim)
        old_new_angle_base = jnp.array([
            self.agent.agent_state.angle_base,
            new_angle_base
        ])
        return valid_move_mask @ old_new_angle_base
    
    def _handle_clock(self) -> "State":
        # Rotate
        old_angle_base = self.agent.agent_state.angle_base
        new_angle_base = decrease_angle_circular(old_angle_base, self.env_cfg.agent.angles_base)
        new_angle_base = self._apply_base_rotation_mask(new_angle_base)

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
        new_angle_base = self._apply_base_rotation_mask(new_angle_base)
        
        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    angle_base=new_angle_base
                )
            )
        )
    
    def _handle_cabin_clock(self) -> "State":
        old_angle_cabin = self.agent.agent_state.angle_cabin
        new_angle_cabin = decrease_angle_circular(old_angle_cabin, self.env_cfg.agent.angles_cabin)

        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    angle_cabin=new_angle_cabin
                )
            )
        )
    
    def _handle_cabin_anticlock(self) -> "State":
        old_angle_cabin = self.agent.agent_state.angle_cabin
        new_angle_cabin = increase_angle_circular(old_angle_cabin, self.env_cfg.agent.angles_cabin)
        
        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    angle_cabin=new_angle_cabin
                )
            )
        )
    
    def _handle_extend_arm(self) -> "State":
        new_arm_extension = jnp.min(
            jnp.array(
                    [self.agent.agent_state.arm_extension + 1,
                    jnp.full((1, ), fill_value=self.env_cfg.agent.max_arm_extension, dtype=IntLowDim)]
                    ),
                    axis=0
        )
        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    arm_extension=new_arm_extension
                )
            )
        )
    
    def _handle_retract_arm(self) -> "State":
        new_arm_extension = jnp.max(
            jnp.array(
                    [self.agent.agent_state.arm_extension - 1,
                    jnp.full((1, ), fill_value=0, dtype=IntLowDim)]
                    ),
                    axis=0
        )
        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    arm_extension=new_arm_extension
                )
            )
        )
    
    @staticmethod
    def _get_current_pos_vector_idx(pos_base: Array, map_height: IntMap) -> IntMap:
        """
        If the map is flattened to an array of shape (2, N), this
        function returns the idx of the current position on the axis=1.

        Return:
            - index of the current position in the flattened map
        """
        return pos_base @ jnp.array([[map_height], [1]], dtype=IntMap)
    
    @staticmethod
    def _map_to_flattened_global_coords(map_width: IntMap, map_height: IntMap, tile_size: Float) -> Array:
        """
        Args:
            - map_width: width dim of the map
            - map_height: height dim of the map
            - tile_size: the dimension of each tile of the map
        Return:
            - a (2, width * height) Array, where the first row are the x coords,
                and the second row are the y coords.
        """
        tile_offset = tile_size / 2
        x_row = jnp.tile(jnp.vstack(jnp.arange(map_width)), map_height).reshape(-1)
        y_row = jnp.tile(jnp.arange(map_height), map_width)
        flat_map = jnp.vstack([x_row, y_row])
        flat_map = flat_map * tile_size
        flat_map = flat_map + tile_offset
        return flat_map
    
    @staticmethod
    def _get_current_pos_from_flattened_map(flattened_map: Array, idx: IntMap) -> Array:
        """
        Given the flattened map and the index of the current position in the vector,
        it returns the current position as [x, y].
        """
        num_tiles = flattened_map.shape[1]
        idx_one_hot = jax.nn.one_hot(idx, num_tiles, dtype=Float)
        current_pos = flattened_map @ idx_one_hot[0]
        return current_pos

    def _get_cabin_angle_rad(self) -> Float:
        return angle_idx_to_rad(self.agent.agent_state.angle_cabin, self.env_cfg.agent.angles_cabin)
    
    def _get_base_angle_rad(self) -> Float:
        return angle_idx_to_rad(self.agent.agent_state.angle_base, self.env_cfg.agent.angles_base)
    
    def _get_arm_angle_rad(self) -> Float:
        base_angle = self._get_base_angle_rad()
        cabin_angle = self._get_cabin_angle_rad()
        return wrap_angle_rad(base_angle + cabin_angle)
    
    def _get_dig_dump_mask_cyl(self, map_cyl_coords: Array) -> Array:
        """
        Note: the map is assumed to be local -> the area to dig is in front of us.
        
        Args:
            - map_cyl_coords: (2, N) Array with [r, theta] rows
        Returns:
            - dig_mask: (N, ) Array of bools, where True means dig here
        """
        arm_extension = self.agent.agent_state.arm_extension
        dig_portion_radius = self.env_cfg.agent.move_tiles
        tile_size = self.env_cfg.tile_size

        # TODO: the following is rough.. make it better (compute ellipse around machine and get min distance based on arm angle)
        max_agent_dim = jnp.max(jnp.array([self.env_cfg.agent.width / 2, self.env_cfg.agent.height / 2]))
        min_distance_from_agent = tile_size * max_agent_dim

        r_max = (arm_extension + 1) * dig_portion_radius * tile_size + min_distance_from_agent
        r_min = arm_extension * dig_portion_radius * tile_size + min_distance_from_agent

        theta_max = np.pi / self.env_cfg.agent.angles_cabin
        theta_min = -theta_max

        dig_mask_r = jnp.logical_and(
            map_cyl_coords[0] >= r_min,
            map_cyl_coords[0] <= r_max
        )

        dig_mask_theta = jnp.logical_and(
            map_cyl_coords[1] >= theta_min,
            map_cyl_coords[1] <= theta_max
        )
        
        return jnp.logical_and(dig_mask_r, dig_mask_theta)

    def _get_dig_dump_mask(self, map_cyl_coords: Array, map_local_coords: Array) -> Array:
        """
        Gets the dig dump mask usign the cylindrical coordinates local map,
        and applies a further masking to avoid digging/dumping where the agent stands.

        Args:
            - map_cyl_coords: (2, N) Array with [r, theta] rows
            - map_local_coords: (2, N) Array with [x, y] rows
        Returns:
            - dig_mask: (N, ) Array of bools, where True means dig here
        """
        dig_dump_mask_cyl = self._get_dig_dump_mask_cyl(map_cyl_coords)

        agent_width = self.env_cfg.agent.width * self.env_cfg.tile_size
        agent_height = self.env_cfg.agent.height * self.env_cfg.tile_size

        dig_dump_mask_cart_x = map_local_coords[0].copy()  # TODO is copy necessary?
        dig_dump_mask_cart_y = map_local_coords[1].copy()  # TODO is copy necessary?
        
        dig_dump_mask_cart_x = jnp.where(
            jnp.logical_or(dig_dump_mask_cart_x >= jnp.floor(agent_width / 2), dig_dump_mask_cart_x <= -jnp.floor(agent_width / 2)),
            1,
            0
        )
        dig_dump_mask_cart_y = jnp.where(
            jnp.logical_or(dig_dump_mask_cart_y >= jnp.floor(agent_height / 2), dig_dump_mask_cart_y <= -jnp.floor(agent_height / 2)),
            1,
            0
        )
        dig_dump_mask_cart = (dig_dump_mask_cart_x + dig_dump_mask_cart_y).astype(jnp.bool_)

        # jax.debug.print("agent_width= {x}", x=agent_width)
        # jax.debug.print("agent_height= {x}", x=agent_height)
        # jax.debug.print("map_local_coords[0]= {x}", x=map_local_coords[0])
        # jax.debug.print("map_local_coords[0]= {x}", x=map_local_coords[1])
        # jax.debug.print("dig_dump_mask_cart_x= {x}", x=dig_dump_mask_cart_x.reshape(self.world.action_map.map.shape))
        # jax.debug.print("dig_dump_mask_cart_y= {x}", x=dig_dump_mask_cart_y.reshape(self.world.action_map.map.shape))
        # jax.debug.print("dig_dump_mask_cart= {x}", x=dig_dump_mask_cart)

        dig_dump_mask = dig_dump_mask_cyl * dig_dump_mask_cart
        # jax.debug.print("x = {x}", x=dig_dump_mask_cart_x.sum())
        # jax.debug.print("y = {x}", x=dig_dump_mask_cart_y.sum())
        jax.debug.print("cyl = {x}", x=dig_dump_mask_cyl.sum())
        jax.debug.print("both = {x}", x=dig_dump_mask.sum())
        return dig_dump_mask
    
    def _apply_dig_mask(self, flattened_map: Array, dig_mask: Array) -> Array:
        """
        Args:
            - flattened_map: (N, ) Array flattened height map
            - dig_mask: (N, ) Array of where to dig bools
        Returns:
            - new_flattened_map: (N, ) Array flattened new height map
        """
        delta_dig = self.env_cfg.agent.dig_depth * dig_mask.astype(IntMap)
        return flattened_map - delta_dig
    
    def _apply_dump_mask(self, flattened_map: Array, dump_mask: Array) -> Array:
        """
        Args:
            - flattened_map: (N, ) Array flattened height map
            - dump_mask: (N, ) Array of where to dump bools
        Returns:
            - new_flattened_map: (N, ) Array flattened new height map
        """
        delta_dig = self.env_cfg.agent.dig_depth * dump_mask.astype(IntMap)
        return flattened_map + delta_dig
    
    def _build_dig_dump_mask(self) -> Array:
        current_pos_idx = self._get_current_pos_vector_idx(
            pos_base=self.agent.agent_state.pos_base,
            map_height=self.env_cfg.action_map.height
        )
        map_global_coords = self._map_to_flattened_global_coords(self.world.width,
                                                                 self.world.height,
                                                                 self.env_cfg.tile_size)
        current_pos = self._get_current_pos_from_flattened_map(map_global_coords, current_pos_idx)
        current_arm_angle = self._get_arm_angle_rad()

        # Local coordinates including the cabin rotation
        current_state_arm = jnp.hstack((current_pos, current_arm_angle))
        map_local_coords_arm = apply_rot_transl(current_state_arm, map_global_coords)
        map_cyl_coords = apply_local_cartesian_to_cyl(map_local_coords_arm)

        # Local coordinates excluding the cabin rotation
        current_state_base = jnp.hstack((current_pos, self._get_base_angle_rad()))
        map_local_coords_base = apply_rot_transl(current_state_base, map_global_coords)

        return self._get_dig_dump_mask(map_cyl_coords, map_local_coords_base)

    def _exclude_dump_tiles_from_dig_mask(self, dig_mask: Array) -> Array:
        """
        Takes the dig mask and turns into False the elements that correspond to
        a dumped tile.
        """
        dumped_mask_action_map = self.world.action_map.map > 0
        jax.debug.print("dumped_mask_action_map= {x}", x=dumped_mask_action_map)
        return dig_mask * (~dumped_mask_action_map).reshape(-1)
    
    def _exclude_dig_tiles_from_dump_mask(self, dump_mask: Array) -> Array:
        """
        Takes the dump mask and turns into False the elements that correspond to
        a digged tile.
        """
        digged_mask_action_map = self.world.action_map.map < 0
        jax.debug.print("digged_mask_action_map= {x}", x=digged_mask_action_map)
        return dump_mask * (~digged_mask_action_map).reshape(-1)

    def _handle_dig(self) -> "State":
        dig_mask = self._build_dig_dump_mask()
        dig_mask = self._exclude_dump_tiles_from_dig_mask(dig_mask)
        flattened_action_map = self.world.action_map.map.reshape(-1)
        new_map_global_coords = self._apply_dig_mask(flattened_action_map, dig_mask)
        new_map_global_coords = new_map_global_coords.reshape(self.world.target_map.map.shape)

        return self._replace(
            world=self.world._replace(
                action_map=self.world.action_map._replace(
                    map=new_map_global_coords
                )
            ),
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    loaded=jnp.full((1, ), fill_value=True, dtype=jnp.bool_)
                )
            )
        )
    
    def _handle_dump(self) -> "State":
        dump_mask = self._build_dig_dump_mask()
        dump_mask = self._exclude_dig_tiles_from_dump_mask(dump_mask)
        flattened_action_map = self.world.action_map.map.reshape(-1)
        new_map_global_coords = self._apply_dump_mask(flattened_action_map, dump_mask)
        new_map_global_coords = new_map_global_coords.reshape(self.world.target_map.map.shape)

        return self._replace(
            world=self.world._replace(
                action_map=self.world.action_map._replace(
                    map=new_map_global_coords
                )
            ),
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    loaded=jnp.full((1, ), fill_value=False, dtype=jnp.bool_)
                )
            )
        )
    
    def _handle_do(self) -> "State":
        state = jax.lax.cond(
            jnp.all(self.agent.agent_state.loaded),
            self._handle_dump,
            self._handle_dig
        )
        return state

    def _get_reward(self) -> Float:
        pass

    def _is_done(self) -> jnp.bool_:
        pass
