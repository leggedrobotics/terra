from typing import Any
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from terra.actions import Action
from terra.actions import ActionType
from terra.actions import TrackedActionType
from terra.actions import WheeledActionType
from terra.agent import Agent
from terra.config import EnvConfig
from terra.map import GridWorld
from terra.utils import angle_idx_to_rad
from terra.utils import apply_local_cartesian_to_cyl
from terra.utils import apply_rot_transl
from terra.utils import decrease_angle_circular
from terra.settings import Float
from terra.utils import get_min_distance_point_to_lines
from terra.utils import increase_angle_circular
from terra.settings import IntLowDim
from terra.settings import IntMap
from terra.utils import wrap_angle_rad


class State(NamedTuple):
    """
    Stores the current state of the environment.
    Given perfect information, the observation corresponds to the state.

    Different agent embodiments allow for different transitions.
    - Tracked Agent
        - Move Forward
        - Move Backward
        - Rotate Clockwise
        - Rotate Anticlockwise
        - Do
    - Wheeled Agent
        - Move Forward
        - Move Backward
        - Move Clockwise Forward
        - Move Clockwise Backward
        - Move Anticlockwise Forward
        - Move Anticlockwise Backward
        - Rotate Cabin Clockwise
        - Rotate Cabin Anticlockwise
        - Do
    """

    key: jax.random.PRNGKey

    env_cfg: EnvConfig

    world: GridWorld
    agent: Agent

    env_steps: int

    @classmethod
    def new(
        cls,
        key: jax.random.PRNGKey,
        env_cfg: EnvConfig,
        target_map: Array,
        padding_mask: Array,
        trench_axes: Array,
        trench_type: Array,
        dumpability_mask_init: Array,
    ) -> "State":
        world = GridWorld.new(
            target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init
        )

        agent, key = Agent.new(
            key, env_cfg, world.max_traversable_x, world.max_traversable_y, padding_mask
        )
        agent = jax.tree_map(
            lambda x: x if isinstance(x, Array) else jnp.array(x), agent
        )

        return State(
            key=key,
            env_cfg=env_cfg,
            world=world,
            agent=agent,
            env_steps=0,
        )

    def _reset(
        self,
        env_cfg: EnvConfig,
        target_map: Array,
        padding_mask: Array,
        trench_axes: Array,
        trench_type: Array,
        dumpability_mask_init: Array,
    ) -> "State":
        """
        Resets the already-existing State
        """
        key, _ = jax.random.split(self.key)
        return self.new(
            key=key,
            env_cfg=env_cfg,
            target_map=target_map,
            padding_mask=padding_mask,
            trench_axes=trench_axes,
            trench_type=trench_type,
            dumpability_mask_init=dumpability_mask_init,
        )

    def _step(self, action: Action) -> "State":
        """
        TrackedAction type --> 0
        WheeledAction type --> 1
        """
        handlers_list = [
            # Tracked
            self._handle_move_forward,
            self._handle_move_backward,
            self._handle_clock,
            self._handle_anticlock,
            self._handle_cabin_clock,
            self._handle_cabin_anticlock,
            self._handle_do,
            # Wheeled
            self._handle_move_forward,
            self._handle_move_backward,
            self._handle_move_clock_forward,
            self._handle_move_clock_backward,
            self._handle_move_anticlock_forward,
            self._handle_move_anticlock_backward,
            self._handle_cabin_clock,
            self._handle_cabin_anticlock,
            self._handle_do,
        ]
        cumulative_len = jnp.array([0, 7], dtype=IntLowDim)
        offset_idx = (cumulative_len @ jax.nn.one_hot(action.type[0], 2)).astype(
            IntLowDim
        )

        state = jax.lax.cond(
            action.action[0] == -1,
            self._do_nothing,
            lambda: jax.lax.switch(offset_idx + action.action[0], handlers_list),
        )

        return state._replace(env_steps=state.env_steps + 1)

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
        fwd_to_bkwd_transformation = jnp.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=IntLowDim,
        )
        orientation_one_hot = self._base_orientation_to_one_hot_forward(
            base_orientation
        )
        return orientation_one_hot @ fwd_to_bkwd_transformation

    @staticmethod
    def _get_agent_corners(
        pos_base: Array,
        base_orientation: IntLowDim,
        agent_width: IntLowDim,
        agent_height: IntLowDim,
    ):
        """
        Gets the coordinates of the 4 corners of the agent.
        """
        orientation_vector_xy = jax.nn.one_hot(base_orientation % 2, 2, dtype=IntLowDim)
        agent_xy_matrix = jnp.array(
            [[agent_width, agent_height], [agent_height, agent_width]], dtype=IntLowDim
        )
        agent_xy_dimensions = orientation_vector_xy @ agent_xy_matrix

        x_base = pos_base[0]
        y_base = pos_base[1]
        x_half_dim = jnp.floor(agent_xy_dimensions[0, 0] / 2)
        y_half_dim = jnp.floor(agent_xy_dimensions[0, 1] / 2)

        agent_corners = jnp.array(
            [
                [x_base + x_half_dim, y_base + y_half_dim],
                [x_base - x_half_dim, y_base + y_half_dim],
                [x_base + x_half_dim, y_base - y_half_dim],
                [x_base - x_half_dim, y_base - y_half_dim],
            ]
        )
        return agent_corners

    @staticmethod
    def _get_agent_corners_xy(agent_corners: Array) -> tuple[Array, Array]:
        """
        Args:
            - agent_corners: (4, 2) Array with agent corners [x, y] column order
        Returns:
            - x: (2, ) Array of min and max x values as [min, max]
            - y: (2, ) Array of min and max y values as [min, max]
        """

        x = jnp.array([jnp.min(agent_corners[:, 0]), jnp.max(agent_corners[:, 0])])
        y = jnp.array([jnp.min(agent_corners[:, 1]), jnp.max(agent_corners[:, 1])])
        return x, y

    @staticmethod
    def _build_traversability_mask(map: Array, padding_mask: Array) -> Array:
        """
        Args:
            - map: (N, M) Array of ints
            - padding_mask: (N, M) Array of ints, 1 if not traversable, 0 if traversable
        Returns:
            - traversability_mask: (N, M) Array of ints
                1 for non traversable, 0 for traversable
        """
        return (~((map == 0) * ~padding_mask)).astype(IntLowDim)

    def _is_valid_move(self, agent_corners_xy: Array) -> Array:
        """
        Returns true if the move action proposed is valid, false otherwise.

        Args:
            - base_position: (2, ) Array with [x, y] proposed base position
            - base_orientation: (1, ) Array with int-based orientation encoding of the agent (e.g. 3)
            - agent_width: width of the agent
            - agent_height: height of the agent
                Note: the width and height parameters can be exploited to mask out also the tiles occupied
                    during a rotation (e.g. width = height = max(width, height))
        Returns:
            - bool, true if proposed action is valid
        """
        map_width = self.world.width
        map_height = self.world.height

        # Map size constraints
        valid_matrix_bottom = jnp.array([0, 0]) <= agent_corners_xy
        valid_matrix_up = agent_corners_xy < jnp.array([map_width, map_height])

        valid_move_map_size = jnp.all(
            jnp.concatenate((valid_matrix_bottom[None], valid_matrix_up[None]), axis=0)
        )

        # Traversability constraints
        traversability_mask = self._build_traversability_mask(
            self.world.action_map.map, self.world.padding_mask.map
        )
        x_minmax_agent, y_minmax_agent = self._get_agent_corners_xy(agent_corners_xy)

        traversability_mask_reduced = jnp.where(
            (jnp.arange(map_width) < x_minmax_agent[0])[:, None].repeat(
                map_height, axis=1
            ),
            0,
            traversability_mask,
        )
        traversability_mask_reduced = jnp.where(
            (jnp.arange(map_width) > x_minmax_agent[1])[:, None].repeat(
                map_height, axis=1
            ),
            0,
            traversability_mask_reduced,
        )
        traversability_mask_reduced = jnp.where(
            (jnp.arange(map_height) < y_minmax_agent[0])[None].repeat(
                map_width, axis=0
            ),
            0,
            traversability_mask_reduced,
        )
        traversability_mask_reduced = jnp.where(
            (jnp.arange(map_height) > y_minmax_agent[1])[None].repeat(
                map_width, axis=0
            ),
            0,
            traversability_mask_reduced,
        )
        valid_move_traversability = jnp.all(traversability_mask_reduced == 0)
        valid_move = jnp.logical_and(valid_move_map_size, valid_move_traversability)
        return valid_move

    @staticmethod
    def _valid_move_to_valid_mask(valid_move: jnp.bool_) -> Array:
        """
        Builds a one-hot encoded 2D vector, encoding a bool value.

        - [1, 0] encodes a False
        - [0, 1] encodes a True
        """
        return jax.nn.one_hot(valid_move.astype(IntLowDim), 2, dtype=IntLowDim)

    def _move_on_orientation(self, orientation_vector: Array) -> "State":
        move_tiles = self.env_cfg.agent.move_tiles
        new_pos_base = self.agent.agent_state.pos_base

        # Propagate action
        possible_deltas_xy = jnp.array(
            [[0, move_tiles], [-move_tiles, 0], [0, -move_tiles], [move_tiles, 0]],
            dtype=IntLowDim,
        )
        delta_xy = orientation_vector @ possible_deltas_xy

        new_pos_base = (new_pos_base + delta_xy)[0]

        agent_corners_xy = self._get_agent_corners(
            new_pos_base,
            base_orientation=self.agent.agent_state.angle_base,
            agent_width=self.env_cfg.agent.width,
            agent_height=self.env_cfg.agent.height,
        )
        valid_move = self._is_valid_move(agent_corners_xy)
        valid_move_mask = self._valid_move_to_valid_mask(valid_move)

        old_new_pos_base = jnp.array([self.agent.agent_state.pos_base, new_pos_base])
        new_pos_base = valid_move_mask @ old_new_pos_base

        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(pos_base=new_pos_base)
            )
        )

    def _handle_move_forward(self) -> "State":
        """
        Moves the base forward - if not loaded
        """

        def _move_forward():
            base_orientation = self.agent.agent_state.angle_base
            orientation_vector = self._base_orientation_to_one_hot_forward(
                base_orientation
            )
            return self._move_on_orientation(orientation_vector)

        return jax.lax.cond(
            self.agent.agent_state.loaded[0] > 0, self._do_nothing, _move_forward
        )

    def _handle_move_backward(self) -> "State":
        """
        Moves the base backward - if not loaded
        """

        def _move_backward():
            base_orientation = self.agent.agent_state.angle_base
            orientation_vector = self._base_orientation_to_one_hot_backwards(
                base_orientation
            )
            return self._move_on_orientation(orientation_vector)

        return jax.lax.cond(
            self.agent.agent_state.loaded[0] > 0, self._do_nothing, _move_backward
        )

    def _apply_base_rotation_mask(
        self, old_angle_base: Array, new_angle_base: Array, clockwise: jnp.bool_
    ) -> Array:
        """
        This function creates a move mask and applies it to the new angle of the base.

        The approach is a simplified one: the agent is split in two parts (front and rear), and on each part
        a square is built taking the max of the two dimensions - and positioned it accordingly to the rotation direction.
        In the end the mask is going to be the union of these two squares plus the end position square.
        This approach allows to diversify between clock and anticlock rotation - even if in a rough way.
        """
        agent_width = self.env_cfg.agent.width
        agent_height = self.env_cfg.agent.height
        pos_base = self.agent.agent_state.pos_base
        x_base = pos_base[0]
        y_base = pos_base[1]

        orientation_vector_xy = jax.nn.one_hot(old_angle_base % 2, 2, dtype=IntLowDim)
        agent_xy_matrix = jnp.array(
            [[agent_width, agent_height], [agent_height, agent_width]], dtype=IntLowDim
        )
        agent_xy_dimensions = orientation_vector_xy @ agent_xy_matrix

        x_half_dim_f = jnp.floor(agent_xy_dimensions[0, 0] / 2)
        x_half_dim_c = jnp.ceil(agent_xy_dimensions[0, 0] / 2)
        y_half_dim_f = jnp.floor(agent_xy_dimensions[0, 1] / 2)
        y_half_dim_c = jnp.ceil(agent_xy_dimensions[0, 1] / 2)

        # TODO: the following 4 functions can become 2 if parametrized on the clockwise bool

        # Clock
        def _get_corners_start_horizontal_clock():
            front_corners = jnp.array(
                [
                    [x_base, y_base],
                    [x_base + x_half_dim_c, y_base],
                    [x_base + x_half_dim_c, y_base + y_half_dim_f],
                    [x_base, y_base + y_half_dim_f],
                ]
            )
            back_corners = jnp.array(
                [
                    [x_base, y_base],
                    [x_base - x_half_dim_c, y_base],
                    [x_base - x_half_dim_c, y_base - y_half_dim_f],
                    [x_base, y_base - y_half_dim_f],
                ]
            )
            return front_corners, back_corners

        def _get_corners_start_vertical_clock():
            front_corners = jnp.array(
                [
                    [x_base, y_base],
                    [x_base - x_half_dim_f, y_base],
                    [x_base - x_half_dim_f, y_base + y_half_dim_c],
                    [x_base, y_base + y_half_dim_c],
                ]
            )
            back_corners = jnp.array(
                [
                    [x_base, y_base],
                    [x_base + x_half_dim_f, y_base],
                    [x_base + x_half_dim_f, y_base - y_half_dim_c],
                    [x_base, y_base - y_half_dim_c],
                ]
            )
            return front_corners, back_corners

        # Anticlock
        def _get_corners_start_vertical_anticlock():
            front_corners = jnp.array(
                [
                    [x_base, y_base],
                    [x_base + y_half_dim_c, y_base],
                    [x_base + y_half_dim_c, y_base + x_half_dim_f],
                    [x_base, y_base + x_half_dim_f],
                ]
            )
            back_corners = jnp.array(
                [
                    [x_base, y_base],
                    [x_base - y_half_dim_c, y_base],
                    [x_base - y_half_dim_c, y_base - x_half_dim_f],
                    [x_base, y_base - x_half_dim_f],
                ]
            )
            return front_corners, back_corners

        def _get_corners_start_horizontal_anticlock():
            front_corners = jnp.array(
                [
                    [x_base, y_base],
                    [x_base - y_half_dim_f, y_base],
                    [x_base - y_half_dim_f, y_base + x_half_dim_c],
                    [x_base, y_base + x_half_dim_c],
                ]
            )
            back_corners = jnp.array(
                [
                    [x_base, y_base],
                    [x_base + y_half_dim_f, y_base],
                    [x_base + y_half_dim_f, y_base - x_half_dim_c],
                    [x_base, y_base - x_half_dim_c],
                ]
            )
            return front_corners, back_corners

        front_corners, back_corners = jax.lax.cond(
            clockwise,
            lambda: jax.lax.cond(
                orientation_vector_xy[0, 0],
                _get_corners_start_horizontal_clock,
                _get_corners_start_vertical_clock,
            ),
            lambda: jax.lax.cond(
                orientation_vector_xy[0, 0],
                _get_corners_start_horizontal_anticlock,
                _get_corners_start_vertical_anticlock,
            ),
        )

        # x and y here are inverted because we use the [x, y] before the 90deg rotation
        # to compute the corners after the rotation.
        new_agent_corners = jnp.array(
            [
                [x_base + y_half_dim_f, y_base + x_half_dim_f],
                [x_base - y_half_dim_f, y_base + x_half_dim_f],
                [x_base + y_half_dim_f, y_base - x_half_dim_f],
                [x_base - y_half_dim_f, y_base - x_half_dim_f],
            ]
        )

        valid_move_front = self._is_valid_move(front_corners)
        valid_move_back = self._is_valid_move(back_corners)
        valid_move_corners = self._is_valid_move(new_agent_corners)
        valid_move = valid_move_front * valid_move_back * valid_move_corners
        valid_move_mask = self._valid_move_to_valid_mask(valid_move)

        # Concatenate new and old angles to then matmul and pick only one of them
        #   this operation is equivalent to an if else statement
        old_new_angle_base = jnp.array(
            [self.agent.agent_state.angle_base, new_angle_base]
        )
        return valid_move_mask @ old_new_angle_base

    def _handle_clock(self) -> "State":
        def _rotate_clock():
            old_angle_base = self.agent.agent_state.angle_base
            new_angle_base = decrease_angle_circular(
                old_angle_base, self.env_cfg.agent.angles_base
            )
            new_angle_base = self._apply_base_rotation_mask(
                old_angle_base, new_angle_base, clockwise=True
            )

            return self._replace(
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        angle_base=new_angle_base
                    )
                )
            )

        return jax.lax.cond(
            self.agent.agent_state.loaded[0] > 0, self._do_nothing, _rotate_clock
        )

    def _handle_anticlock(self) -> "State":
        def _rotate_anticlock():
            old_angle_base = self.agent.agent_state.angle_base
            new_angle_base = increase_angle_circular(
                old_angle_base, self.env_cfg.agent.angles_base
            )
            new_angle_base = self._apply_base_rotation_mask(
                old_angle_base, new_angle_base, clockwise=False
            )

            return self._replace(
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        angle_base=new_angle_base
                    )
                )
            )

        return jax.lax.cond(
            self.agent.agent_state.loaded[0] > 0, self._do_nothing, _rotate_anticlock
        )

    def _handle_cabin_clock(self) -> "State":
        old_angle_cabin = self.agent.agent_state.angle_cabin
        new_angle_cabin = decrease_angle_circular(
            old_angle_cabin, self.env_cfg.agent.angles_cabin
        )

        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(angle_cabin=new_angle_cabin)
            )
        )

    def _handle_cabin_anticlock(self) -> "State":
        old_angle_cabin = self.agent.agent_state.angle_cabin
        new_angle_cabin = increase_angle_circular(
            old_angle_cabin, self.env_cfg.agent.angles_cabin
        )

        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(angle_cabin=new_angle_cabin)
            )
        )

    def _handle_move_clock_forward(self) -> "State":
        valid_chain = True
        s0 = self._handle_move_forward()
        valid_chain *= self._check_agent_moved_on_move_action(self, s0)
        s1 = s0._handle_clock()
        valid_chain *= self._check_agent_turn_on_turn_action(s0, s1)
        s2 = s1._handle_move_forward()
        valid_chain *= self._check_agent_moved_on_move_action(s1, s2)
        return jax.lax.cond(
            valid_chain,
            lambda: s2,
            lambda: self,
        )

    def _handle_move_clock_backward(self) -> "State":
        valid_chain = True
        s0 = self._handle_move_backward()
        valid_chain *= self._check_agent_moved_on_move_action(self, s0)
        s1 = s0._handle_clock()
        valid_chain *= self._check_agent_turn_on_turn_action(s0, s1)
        s2 = s1._handle_move_backward()
        valid_chain *= self._check_agent_moved_on_move_action(s1, s2)
        return jax.lax.cond(
            valid_chain,
            lambda: s2,
            lambda: self,
        )

    def _handle_move_anticlock_forward(self) -> "State":
        valid_chain = True
        s0 = self._handle_move_forward()
        valid_chain *= self._check_agent_moved_on_move_action(self, s0)
        s1 = s0._handle_anticlock()
        valid_chain *= self._check_agent_turn_on_turn_action(s0, s1)
        s2 = s1._handle_move_forward()
        valid_chain *= self._check_agent_moved_on_move_action(s1, s2)
        return jax.lax.cond(
            valid_chain,
            lambda: s2,
            lambda: self,
        )

    def _handle_move_anticlock_backward(self) -> "State":
        valid_chain = True
        s0 = self._handle_move_backward()
        valid_chain *= self._check_agent_moved_on_move_action(self, s0)
        s1 = s0._handle_anticlock()
        valid_chain *= self._check_agent_turn_on_turn_action(s0, s1)
        s2 = s1._handle_move_backward()
        valid_chain *= self._check_agent_moved_on_move_action(s1, s2)
        return jax.lax.cond(
            valid_chain,
            lambda: s2,
            lambda: self,
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
    def _map_to_flattened_global_coords(
        map_width: IntMap, map_height: IntMap, tile_size: Float
    ) -> Array:
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
        return angle_idx_to_rad(
            self.agent.agent_state.angle_cabin, self.env_cfg.agent.angles_cabin
        )

    def _get_base_angle_rad(self) -> Float:
        return angle_idx_to_rad(
            self.agent.agent_state.angle_base, self.env_cfg.agent.angles_base
        )

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
        dig_portion_radius = self.env_cfg.agent.move_tiles
        tile_size = self.env_cfg.tile_size

        max_agent_dim = jnp.max(
            jnp.array([self.env_cfg.agent.width / 2, self.env_cfg.agent.height / 2])
        )
        min_distance_from_agent = tile_size * max_agent_dim

        # Fixed middle-point arm extension (halfway between 0 and 1)
        fixed_extension = 0.5
        r_min = fixed_extension * dig_portion_radius * tile_size + min_distance_from_agent
        r_max = (fixed_extension + 1) * dig_portion_radius * tile_size + min_distance_from_agent
        theta_max = np.pi / self.env_cfg.agent.angles_cabin
        theta_min = -theta_max

        dig_mask_r = jnp.logical_and(
            map_cyl_coords[0] >= r_min, map_cyl_coords[0] <= r_max
        )
        dig_mask_theta = jnp.logical_and(
            map_cyl_coords[1] >= theta_min, map_cyl_coords[1] <= theta_max
        )

        return jnp.logical_and(dig_mask_r, dig_mask_theta)

    def _get_dig_dump_mask(
        self, map_cyl_coords: Array, map_local_coords: Array
    ) -> Array:
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

        dig_dump_mask_cart_x = map_local_coords[0].copy()
        dig_dump_mask_cart_y = map_local_coords[1].copy()

        eps = self.env_cfg.tile_size / 2  # add margin to avoid rounding errors

        dig_dump_mask_cart_x = jnp.where(
            jnp.logical_or(
                dig_dump_mask_cart_x >= jnp.floor((agent_width + eps) / 2),
                dig_dump_mask_cart_x <= -jnp.floor((agent_width + eps) / 2),
            ),
            1,
            0,
        )
        dig_dump_mask_cart_y = jnp.where(
            jnp.logical_or(
                dig_dump_mask_cart_y >= jnp.floor((agent_height + eps) / 2),
                dig_dump_mask_cart_y <= -jnp.floor((agent_height + eps) / 2),
            ),
            1,
            0,
        )
        dig_dump_mask_cart = (dig_dump_mask_cart_x + dig_dump_mask_cart_y).astype(
            jnp.bool_
        )
        dig_dump_mask = dig_dump_mask_cyl * dig_dump_mask_cart
        return dig_dump_mask

    def _apply_dig_mask(
        self, flattened_map: Array, dig_mask: Array, moving_dumped_dirt: bool
    ) -> Array:
        """
        this function does the following:
            if we are moving dumped dirt, we move all of it regardless of the amount
            if we are instead digging dirt, then we dig as much as self.env_cfg.agent.dig_depth

        Args:
            - flattened_map: (N, ) Array flattened height map
            - dig_mask: (N, ) Array of where to dig bools
        Returns:
            - new_flattened_map: (N, ) Array flattened new height map
        """
        delta_dig = self.env_cfg.agent.dig_depth * dig_mask.astype(IntMap)
        m = jax.lax.cond(
            moving_dumped_dirt,
            lambda: jnp.where(dig_mask, 0, flattened_map).astype(IntMap),
            #  (flattened_map * (~dig_mask)).astype(flattened_map.dtype),
            lambda: (flattened_map - delta_dig).astype(IntMap),
        )
        return m

    def _apply_dump_mask(
        self,
        flattened_map: Array,
        dump_mask: Array,
        even_volume_per_tile: IntLowDim,
        remaining_volume: IntLowDim,
        target_map: Array,
    ) -> Array:
        """
        TODO: delta_dig_remaining now is added with a naive approach - should be added
            either to the closest tiles or randomly

        Args:
            - flattened_map: (N, ) Array flattened height map
            - dump_mask: (N, ) Array of where to dump bools
            - even_volume_per_tile: IntLowDim, volume to add to each of the tiles in the mask (per tile)
            - remaining_volume: IntLowDim, remaining volume to add to some of the tiles in the mask (total)
        Returns:
            - new_flattened_map: (N, ) Array flattened new height map
        """
        # Check if there is any target dump tile within the mask
        target_map_dump_mask = jnp.clip(target_map.reshape(-1), a_min=0) * dump_mask
        target_dump_volume = target_map_dump_mask.sum()
        dump_mask, dump_volume = jax.lax.cond(
            target_dump_volume > 0,
            lambda: (IntMap(target_map_dump_mask), target_dump_volume),
            lambda: (IntMap(dump_mask), dump_mask.sum()),
        )

        loaded_volume = self.agent.agent_state.loaded
        remaining_volume = loaded_volume % dump_volume
        even_volume_per_tile = (loaded_volume - remaining_volume) / dump_volume

        delta_dig = self.env_cfg.agent.dig_depth * dump_mask * even_volume_per_tile
        delta_dig_remaining = jnp.zeros_like(delta_dig, dtype=IntMap)

        delta_dig_remaining = jnp.where(
            jnp.logical_and(jnp.cumsum(dump_mask) <= remaining_volume, dump_mask),
            1,
            delta_dig_remaining,
        )

        return (flattened_map + delta_dig + delta_dig_remaining).astype(IntMap)

    def _get_map_local_and_cyl_coords(self):
        """
        Returns:
            - map_cyl_coords: (2, width*height) map with [r, theta] rows
            - map_local_coords_base: (2, width*height) map with [x, y] rows
        """
        current_pos_idx = self._get_current_pos_vector_idx(
            pos_base=self.agent.agent_state.pos_base,
            map_height=self.env_cfg.maps.edge_length_px,
        )
        map_global_coords = self._map_to_flattened_global_coords(
            self.world.width, self.world.height, self.env_cfg.tile_size
        )
        current_pos = self._get_current_pos_from_flattened_map(
            map_global_coords, current_pos_idx
        )
        current_arm_angle = self._get_arm_angle_rad()

        # Local coordinates including the cabin rotation
        current_state_arm = jnp.hstack((current_pos, current_arm_angle))
        map_local_coords_arm = apply_rot_transl(current_state_arm, map_global_coords)
        map_cyl_coords = apply_local_cartesian_to_cyl(map_local_coords_arm)

        # Local coordinates excluding the cabin rotation
        current_state_base = jnp.hstack((current_pos, self._get_base_angle_rad()))
        map_local_coords_base = apply_rot_transl(current_state_base, map_global_coords)
        return map_cyl_coords, map_local_coords_base

    def _build_dig_dump_cone(self) -> Array:
        """
        Returns the masked workspace cone in cartesian coords. Every tile in the cone is included as +1.
        """
        map_cyl_coords, map_local_coords_base = self._get_map_local_and_cyl_coords()
        return self._get_dig_dump_mask(map_cyl_coords, map_local_coords_base)

    def _exclude_dig_tiles_from_dump_mask(self, dump_mask: Array) -> Array:
        """
        Takes the dump mask and turns into False the elements that correspond to
        a dug tile.
        """
        digged_mask_action_map = self.world.dig_map.map < 0
        return dump_mask * (~digged_mask_action_map).reshape(-1)

    def _exclude_dumpability_mask_tiles_from_dump_mask(self, dump_mask: Array) -> Array:
        """Applies dumpability mask to the dump mask"""
        return dump_mask * self.world.dumpability_mask.map.reshape(-1)

    def _exclude_traversability_mask_tiles_from_dump_mask(
        self, dump_mask: Array
    ) -> Array:
        """Applies traversability mask to the dump mask"""
        return dump_mask * (self.world.traversability_mask.map == 0).reshape(-1)

    def _exclude_just_moved_tiles_from_dump_mask(self, dump_mask: Array) -> Array:
        """
        Removes the possibility of moving some dump tiles in the same spot.

        Also, removes the possibility of moving the tiles within the same workspace,
        even if not all tiles are occupied.
        """
        cone_mask = self._build_dig_dump_cone()
        dig_map_mask = jax.lax.cond(
            (
                (self.world.dig_map.map != self.world.action_map.map).reshape(-1)
                * (self.world.action_map.map.reshape(-1) > 0)
                * cone_mask
            ).sum()
            > 0,
            lambda: ~cone_mask,
            lambda: jnp.ones_like(dump_mask),
        )

        return dump_mask * dig_map_mask

    def _mask_out_wrong_dig_tiles(self, dig_mask: Array) -> Array:
        """
        Takes the dig mask and turns into False the elements that do not correspond to
        a tile that has to be digged in the target map or that are dumped tiles in the action map.
        Also masks out the tiles that are digged as much as the target map requires.
        """
        dig_mask_target_map = self.world.target_map.map < 0
        dig_mask_action_map = self.world.action_map.map > 0
        dig_mask_maps = jnp.logical_or(dig_mask_target_map, dig_mask_action_map)

        flat_action_map = self.world.action_map.map.reshape(-1)
        dig_mask_cone = self._build_dig_dump_cone()
        ambiguity_mask_dig_movesoil = jax.lax.cond(
            jnp.any(flat_action_map * dig_mask_cone.reshape(-1) > 0),
            lambda: flat_action_map > 0,
            lambda: flat_action_map == 0,
        )

        # respect max dig limit
        max_dig_limit_mask = (
            self.world.action_map.map > -self.env_cfg.agent.dig_depth
        ).reshape(-1)

        return (
            dig_mask
            * dig_mask_maps.reshape(-1)
            * ambiguity_mask_dig_movesoil
            * max_dig_limit_mask
        ).astype(jnp.bool_)

    def _get_new_dumpability_mask(self, action_map: Array) -> Array:
        new_dumpability_mask = self.world.dumpability_mask_init.map
        action_mask = (action_map < 0).astype(jnp.float16)
        kernel = jnp.ones((3, 3), dtype=jnp.float16)
        action_mask_contoured = jax.scipy.signal.convolve2d(
            action_mask,
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        return new_dumpability_mask * (action_mask_contoured == 0)

    def _handle_dig(self) -> "State":
        dig_mask = self._build_dig_dump_cone()
        # dig_mask = self._exclude_dump_tiles_from_dig_mask(dig_mask)
        dig_mask = self._mask_out_wrong_dig_tiles(dig_mask)
        flattened_action_map = self.world.action_map.map.reshape(-1)
        selected_tiles_sum = flattened_action_map @ dig_mask
        moving_dumped_dirt = selected_tiles_sum > 0
        # if moving dumped dirt, move it all at once
        dig_volume = jax.lax.cond(
            moving_dumped_dirt,
            lambda: selected_tiles_sum.astype(jnp.int32),
            lambda: dig_mask.sum(),
        )

        def _apply_dig(volume, fam):
            new_map_global_coords = self._apply_dig_mask(
                fam, dig_mask, moving_dumped_dirt
            )
            new_map_global_coords = new_map_global_coords.reshape(
                self.world.target_map.map.shape
            )

            return self._replace(
                world=self.world._replace(
                    dig_map=self.world.dig_map._replace(
                        map=IntLowDim(new_map_global_coords)
                    )
                ),
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        loaded=jnp.full((1,), fill_value=volume, dtype=IntLowDim)
                    )
                ),
            )

        s = jax.lax.cond(
            dig_volume > 0,
            lambda v, fam: _apply_dig(v, fam),
            lambda v, fam: self._do_nothing(),
            dig_volume,
            flattened_action_map,
        )
        return s

    def _handle_dump(self) -> "State":
        dump_mask = self._build_dig_dump_cone()
        dump_mask = self._exclude_dig_tiles_from_dump_mask(dump_mask)
        dump_mask = self._exclude_dumpability_mask_tiles_from_dump_mask(dump_mask)
        dump_mask = self._exclude_traversability_mask_tiles_from_dump_mask(dump_mask)
        dump_mask = self._exclude_just_moved_tiles_from_dump_mask(dump_mask)
        dump_volume = dump_mask.sum()

        # dump_volume_per_tile = jnp.rint(
        #     self.agent.agent_state.loaded / (dump_volume + 1e-6)
        # ).astype(IntLowDim)

        remaining_volume = self.agent.agent_state.loaded % dump_volume
        even_volume_per_tile = (
            self.agent.agent_state.loaded - remaining_volume
        ) / dump_volume

        def _apply_dump():
            flattened_dig_map = self.world.dig_map.map.reshape(-1)
            new_map_global_coords = self._apply_dump_mask(
                flattened_dig_map,
                dump_mask,
                even_volume_per_tile,
                remaining_volume,
                self.world.target_map.map,
            )
            new_map_global_coords = new_map_global_coords.reshape(
                self.world.target_map.map.shape
            )

            new_dumpability_mask = self._get_new_dumpability_mask(
                new_map_global_coords,
            )

            return self._replace(
                world=self.world._replace(
                    action_map=self.world.action_map._replace(
                        map=IntLowDim(new_map_global_coords)
                    ),
                    dig_map=self.world.dig_map._replace(
                        map=IntLowDim(new_map_global_coords)
                    ),
                    dumpability_mask=self.world.dumpability_mask._replace(
                        map=jnp.bool_(new_dumpability_mask),
                    ),
                ),
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        loaded=jnp.full((1,), fill_value=0, dtype=IntLowDim)
                    )
                ),
            )

        return jax.lax.cond(dump_volume > 0, _apply_dump, self._do_nothing)

    def _handle_do(self) -> "State":
        state = jax.lax.cond(
            jnp.all(self.agent.agent_state.loaded.astype(jnp.bool_)),
            self._handle_dump,
            self._handle_dig,
        )
        return state

    @staticmethod
    def _check_agent_moved_on_move_action(
        old_state: "State", new_state: "State"
    ) -> bool:
        """True if agent moved"""
        return ~jnp.allclose(
            old_state.agent.agent_state.pos_base, new_state.agent.agent_state.pos_base
        )

    @staticmethod
    def _check_agent_turn_on_turn_action(
        old_state: "State", new_state: "State"
    ) -> bool:
        """True if agent turned"""
        return ~jnp.allclose(
            old_state.agent.agent_state.angle_base,
            new_state.agent.agent_state.angle_base,
        )

    def _handle_rewards_move(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        reward = 0.0
        # Collision
        reward += jax.lax.cond(
            ~self._check_agent_moved_on_move_action(self, new_state),
            lambda: self.env_cfg.rewards.collision_move,
            lambda: 0.0,
        )

        # Move while loaded
        reward += jax.lax.cond(
            jnp.all(self.agent.agent_state.loaded > 0),
            lambda: self.env_cfg.rewards.move_while_loaded,
            lambda: 0.0,
        )

        # Move
        reward += self.env_cfg.rewards.move

        return reward

    def _handle_rewards_base_turn(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        reward = 0.0

        # Collision turn
        reward += jax.lax.cond(
            ~self._check_agent_turn_on_turn_action(self, new_state),
            lambda: self.env_cfg.rewards.collision_turn,
            lambda: 0.0,
        )

        # Base turn
        reward += self.env_cfg.rewards.base_turn
        return reward

    def _handle_rewards_cabin_turn(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        # Cabin turn
        return self.env_cfg.rewards.cabin_turn

    @staticmethod
    def _get_action_map_positive_progress(
        action_map_old: Array, action_map_new: Array, target_map: Array
    ) -> IntMap:
        """
        Returns
        > 0 if there was progress on the dump tiles
        < 0 if there was -progress on the dump tiles
        = 0 if there was no progress on the dump tiles
        """
        action_map_clip_old = jnp.clip(action_map_old, a_min=0)
        action_map_clip_new = jnp.clip(action_map_new, a_min=0)

        target_map_dump_mask = target_map > 0

        action_map_progress = (
            (action_map_clip_new - action_map_clip_old) * target_map_dump_mask
        ).sum()

        return action_map_progress

    @staticmethod
    def _get_action_map_spread_out_rate(
        action_map_old: Array, action_map_new: Array, target_map: Array, loaded: int
    ) -> IntMap:
        """
        Returns the spread-out rate of the terrain that has just been dumped.
        The rate is defined as (#tiles-dumped / #tiles-loaded)
        """
        action_map_mask_old = (action_map_old > 0).astype(IntLowDim)
        action_map_mask_new = (action_map_new > 0).astype(IntLowDim)

        target_map_mask = target_map >= 0  # include also neutral tiles

        action_map_progress = (
            (action_map_mask_new - action_map_mask_old) * target_map_mask
        ).sum()
        return action_map_progress.astype(jnp.float32) / loaded[0].astype(jnp.float32)

    def _handle_rewards_dump(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        """
        Handles reward assignment at dump time.
        This includes both the dump part and the realization
        of the previously digged terrain.
        """

        # Dig
        action_map_negative_progress = self._get_action_map_negative_progress(
            self.world.action_map.map,
            new_state.world.action_map.map,
            self.world.target_map.map,
        )
        dig_reward = jax.lax.cond(
            action_map_negative_progress > 0,
            lambda: self.env_cfg.rewards.dig_correct,
            lambda: 0.0,
        )

        # Dump
        action_map_positive_progress = self._get_action_map_positive_progress(
            self.world.dig_map.map,  # note dig_map here
            new_state.world.action_map.map,
            self.world.target_map.map,
        )
        spread_out_rate = self._get_action_map_spread_out_rate(
            self.world.dig_map.map,  # note dig_map here
            new_state.world.action_map.map,
            self.world.target_map.map,
            self.agent.agent_state.loaded,
        )

        dump_reward_condition = jnp.allclose(
            self.agent.agent_state.loaded, new_state.agent.agent_state.loaded
        )

        def dump_reward_fn() -> Float:
            return jax.lax.cond(
                action_map_positive_progress < 0,
                lambda: self.env_cfg.rewards.dump_no_dump_area,
                lambda: jax.lax.cond(
                    action_map_negative_progress == 0,
                    lambda: 0.0,
                    lambda: jax.lax.cond(
                        action_map_positive_progress > 0,
                        lambda: spread_out_rate * self.env_cfg.rewards.dump_correct,
                        lambda: 0.0,
                    ),
                ),
            )

        dump_reward = jax.lax.cond(
            dump_reward_condition,
            lambda: self.env_cfg.rewards.dump_wrong,
            dump_reward_fn,
        )

        r_trenches = self._get_trench_specific_rewards()
        return dig_reward + dump_reward + r_trenches

    @staticmethod
    def _get_action_map_negative_progress(
        action_map_old: Array, action_map_new: Array, target_map: Array
    ) -> IntMap:
        """
        Returns
        > 0 if there was progress on the dig tiles
        < 0 if there was -progress on the dig tiles (shouldn't be allowed)
        = 0 if there was no progress on the dig tiles
        """
        action_map_clip_old = jnp.clip(action_map_old, a_min=None, a_max=0)
        action_map_clip_new = jnp.clip(action_map_new, a_min=None, a_max=0)

        target_map_mask = target_map < 0
        action_map_progress = (
            (action_map_clip_old - action_map_clip_new) * target_map_mask
        ).sum()

        return action_map_progress

    def _handle_rewards_dig(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        # Dig
        return jax.lax.cond(
            jnp.allclose(
                self.agent.agent_state.loaded, new_state.agent.agent_state.loaded
            ),
            lambda: self.env_cfg.rewards.dig_wrong,
            lambda: 0.0,
        )

    def _handle_rewards_do(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        return jax.lax.cond(
            jnp.all(self.agent.agent_state.loaded > 0),
            self._handle_rewards_dump,
            self._handle_rewards_dig,
            new_state,
            action,
        )

    def _get_rewards_tracked(self, new_state: "State", action: ActionType) -> Float:
        reward = 0.0
        action = action[0]

        reward += jax.lax.cond(
            (action == TrackedActionType.FORWARD)
            | (action == TrackedActionType.BACKWARD),
            self._handle_rewards_move,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )

        reward += jax.lax.cond(
            (action == TrackedActionType.CLOCK)
            | (action == TrackedActionType.ANTICLOCK),
            self._handle_rewards_base_turn,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )

        reward += jax.lax.cond(
            (action == TrackedActionType.CABIN_CLOCK)
            | (action == TrackedActionType.CABIN_ANTICLOCK),
            self._handle_rewards_cabin_turn,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )

        reward += jax.lax.cond(
            action == TrackedActionType.DO,
            self._handle_rewards_do,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )
        return reward

    def _get_rewards_wheeled(self, new_state: "State", action: ActionType) -> Float:
        reward = 0.0
        action = action[0]

        reward += jax.lax.cond(
            (action == WheeledActionType.FORWARD)
            | (action == WheeledActionType.BACKWARD),
            self._handle_rewards_move,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )

        reward += jax.lax.cond(
            (action == WheeledActionType.CLOCK_FORWARD)
            | (action == WheeledActionType.ANTICLOCK_FORWARD)
            | (action == WheeledActionType.CLOCK_BACKWARD)
            | (action == WheeledActionType.ANTICLOCK_BACKWARD),
            lambda new_state, action: self._handle_rewards_base_turn(new_state, action)
            + self._handle_rewards_move(new_state, action),
            lambda new_state, action: 0.0,
            new_state,
            action,
        )

        reward += jax.lax.cond(
            (action == WheeledActionType.CABIN_CLOCK)
            | (action == WheeledActionType.CABIN_ANTICLOCK),
            self._handle_rewards_cabin_turn,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )

        reward += jax.lax.cond(
            action == WheeledActionType.DO,
            self._handle_rewards_do,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )
        return reward

    def _get_trench_specific_rewards(
        self,
    ) -> Float:
        def _get_trench_reward():
            agent_pos = self.agent.agent_state.pos_base
            trench_axes = self.world.trench_axes
            trench_type = self.world.trench_type

            d = get_min_distance_point_to_lines(
                agent_pos, trench_axes, trench_type
            )  # in tiles
            d = jax.lax.cond(d > self.env_cfg.agent.width / 2, lambda: d, lambda: 0.0)
            d *= self.env_cfg.tile_size  # in meters
            return d * self.env_cfg.distance_coefficient

        r = jax.lax.cond(
            self.env_cfg.apply_trench_rewards,
            _get_trench_reward,
            lambda: 0.0,
        )
        return r

    def _get_terminal_completed_tiles_reward(
        self,
    ) -> Float:
        tiles_digged = (self.world.action_map.map == -1).sum()
        total_tiles = (self.world.target_map.map == -1).sum()
        return (
            tiles_digged / total_tiles
        ) * self.env_cfg.rewards.terminal_completed_tiles

    def _get_reward(self, new_state: "State", action_handler: Action) -> Float:
        action = action_handler.action

        reward = 0.0

        # Action-dependent
        reward += jax.lax.cond(
            action_handler.type[0] == 0,
            self._get_rewards_tracked,
            self._get_rewards_wheeled,
            new_state,
            action,
        )

        # Terminal
        reward += jax.lax.cond(
            self._is_done_task(
                new_state.world.action_map.map,
                self.world.target_map.map,
                new_state.agent.agent_state.loaded,
            ),
            lambda: self.env_cfg.rewards.terminal,
            lambda: 0.0,
        )

        # Terminal completed tiles
        reward += jax.lax.cond(
            self._is_done(
                new_state.world.action_map.map,
                self.world.target_map.map,
                new_state.agent.agent_state.loaded,
            )[0],
            self._get_terminal_completed_tiles_reward,
            lambda: 0.0,
        )

        # Existence
        reward += self.env_cfg.rewards.existence

        # Constant scaling factor
        reward /= self.env_cfg.rewards.normalizer

        return reward

    @staticmethod
    def _is_done_task(action_map: Array, target_map: Array, agent_loaded: Array):
        """
        Checks if the target map matches the action map,
        but only on the relevant tiles.

        On top of that, the agent should not be loaded.

        The relevant tiles are defined as the tiles where the target map is not zero.
        """

        def _check_done_dump():
            relevant_action_map_positive_inverse = jnp.where(
                target_map > 0, 0, action_map
            )
            done_dump = jnp.all(relevant_action_map_positive_inverse <= 0)
            return done_dump

        done_dump = jax.lax.cond(
            jnp.all(target_map <= 0),
            lambda: True,
            _check_done_dump,
        )

        relevant_action_map_negative = jnp.where(target_map < 0, action_map, 0)
        target_map_negative = jnp.clip(target_map, a_max=0)
        done_dig = jnp.all(target_map_negative - relevant_action_map_negative >= 0)

        done_unload = agent_loaded[0] == 0

        done_task = done_dump & done_dig & done_unload
        return done_task

    def _is_done(
        self, action_map: Array, target_map: Array, agent_loaded: Array
    ) -> tuple[jnp.bool_, jnp.bool_]:
        done_task = self._is_done_task(action_map, target_map, agent_loaded)
        done_steps = self.env_steps >= self.env_cfg.max_steps_in_episode
        return jnp.logical_or(done_task, done_steps), done_task

    def _get_action_mask_tracked(self):
        # forward
        new_state = self._handle_move_forward()
        bool_forward = ~jnp.all(
            new_state.agent.agent_state.pos_base == self.agent.agent_state.pos_base
        )

        # backward
        new_state = self._handle_move_backward()
        bool_backward = ~jnp.all(
            new_state.agent.agent_state.pos_base == self.agent.agent_state.pos_base
        )

        # clock
        new_state = self._handle_clock()
        bool_clock = ~jnp.all(
            new_state.agent.agent_state.angle_base == self.agent.agent_state.angle_base
        )

        # anticlock
        new_state = self._handle_anticlock()
        bool_anticlock = ~jnp.all(
            new_state.agent.agent_state.angle_base == self.agent.agent_state.angle_base
        )

        # cabin clock
        new_state = self._handle_cabin_clock()
        bool_cabin_clock = ~jnp.all(
            new_state.agent.agent_state.angle_cabin
            == self.agent.agent_state.angle_cabin
        )

        # cabin clock
        new_state = self._handle_cabin_anticlock()
        bool_cabin_anticlock = ~jnp.all(
            new_state.agent.agent_state.angle_cabin
            == self.agent.agent_state.angle_cabin
        )

        # do
        new_state = self._handle_do()
        bool_do = ~jnp.all(
            new_state.agent.agent_state.loaded == self.agent.agent_state.loaded
        )

        action_mask = jnp.array(
            [
                bool_forward,
                bool_backward,
                bool_clock,
                bool_anticlock,
                bool_cabin_clock,
                bool_cabin_anticlock,
                bool_do,
                0,  # dummy
                0,  # dummy
            ],
            dtype=jnp.bool_,
        )
        return action_mask

    def _get_action_mask_wheeled(self):
        # forward
        new_state = self._handle_move_forward()
        bool_forward = ~jnp.all(
            new_state.agent.agent_state.pos_base == self.agent.agent_state.pos_base
        )

        # backward
        new_state = self._handle_move_backward()
        bool_backward = ~jnp.all(
            new_state.agent.agent_state.pos_base == self.agent.agent_state.pos_base
        )

        # move clock forward
        new_state = self._handle_move_clock_forward()
        bool_move_clock_forward = ~jnp.all(
            new_state.agent.agent_state.angle_base == self.agent.agent_state.angle_base
        )

        # move clock backward
        new_state = self._handle_move_clock_backward()
        bool_move_clock_backward = ~jnp.all(
            new_state.agent.agent_state.angle_base == self.agent.agent_state.angle_base
        )

        # move anticlock forward
        new_state = self._handle_move_anticlock_forward()
        bool_move_anticlock_forward = ~jnp.all(
            new_state.agent.agent_state.angle_cabin
            == self.agent.agent_state.angle_cabin
        )

        # move anticlock backward
        new_state = self._handle_move_anticlock_backward()
        bool_move_anticlock_backward = ~jnp.all(
            new_state.agent.agent_state.angle_cabin
            == self.agent.agent_state.angle_cabin
        )

        # cabin clock
        new_state = self._handle_cabin_clock()
        bool_cabin_clock = ~jnp.all(
            new_state.agent.agent_state.angle_cabin
            == self.agent.agent_state.angle_cabin
        )

        # cabin anticlock
        new_state = self._handle_cabin_anticlock()
        bool_cabin_anticlock = ~jnp.all(
            new_state.agent.agent_state.angle_cabin
            == self.agent.agent_state.angle_cabin
        )

        # do
        new_state = self._handle_do()
        bool_do = ~jnp.all(
            new_state.agent.agent_state.loaded == self.agent.agent_state.loaded
        )

        action_mask = jnp.array(
            [
                bool_forward,
                bool_backward,
                bool_move_clock_forward,
                bool_move_clock_backward,
                bool_move_anticlock_forward,
                bool_move_anticlock_backward,
                bool_cabin_clock,
                bool_cabin_anticlock,
                bool_do,
            ],
            dtype=jnp.bool_,
        )
        return action_mask

    def _get_action_mask(self, dummy_action: Action):
        """
        Returns a 1D array of bools, where 1 is allowed action, and 0 is not allowed.
        """
        num_actions = dummy_action.get_num_actions()
        action_mask = jax.lax.cond(
            dummy_action.type[0] == 0,
            self._get_action_mask_tracked,
            self._get_action_mask_wheeled,
        )
        action_mask = action_mask[:num_actions]

        return action_mask

    def _get_infos(self, dummy_action: Action, task_done: bool) -> dict[str, Any]:
        infos = {
            "action_mask": self._get_action_mask(dummy_action),
            "target_tiles": self._build_dig_dump_cone(),
            # Include termination_type directly without done_task
            "task_done": task_done,
        }
        return infos
