from typing import Any
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax import lax

from terra.actions import Action
from terra.actions import ActionType
from terra.actions import TrackedActionType
from terra.actions import WheeledActionType
from terra.agent import Agent
from terra.config import AgentConfig
from terra.config import EnvConfig
from terra.map import GridWorld
from terra.utils import angle_idx_to_rad
from terra.utils import apply_local_cartesian_to_cyl
from terra.utils import apply_rot_transl
from terra.utils import compute_polygon_mask
from terra.utils import decrease_angle_circular
from terra.settings import Float
from terra.utils import get_distance_point_to_line, get_min_distance_point_to_lines
from terra.utils import increase_angle_circular
from terra.settings import IntLowDim
from terra.settings import IntMap
from terra.utils import wrap_angle_rad

# Add training performance flag
ENABLE_SOIL_MECHANICS_IN_TRAINING = True


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
        action_map: Array,
    ) -> "State":
        # TEMP HACK: Set all dirt height 1 to 5 for testing
        action_map = jnp.where(action_map == 1, 5, action_map)
        world = GridWorld.new(
            target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init, action_map
        )

        agent, key = Agent.new(
            key, env_cfg, world.max_traversable_x, world.max_traversable_y, padding_mask, action_map,
            agent_types=(0, 2)  # Agent 1: tracked (0), Agent 2: skid steer (2)
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
        action_map: Array,
    ) -> "State":
        """
        Resets the already-existing State
        """
        key, _ = jax.random.split(self.key)
        # TEMP HACK: Set all dirt height 1 to 5 for testing
        action_map = jnp.where(action_map == 1, 5, action_map)
        return self.new(
            key=key,
            env_cfg=env_cfg,
            target_map=target_map,
            padding_mask=padding_mask,
            trench_axes=trench_axes,
            trench_type=trench_type,
            dumpability_mask_init=dumpability_mask_init,
            action_map=action_map,
        )

    def _step(self, action: Action, turn:bool = True) -> "State":
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
            self._handle_move_forward_wheeled,
            self._handle_move_backward_wheeled,
            self._handle_turn_wheels_left,
            self._handle_turn_wheels_right,
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
        state = jax.lax.cond(
            turn, 
            state._swap,
            lambda: state
        )
        return state._replace(env_steps=state.env_steps + 1)

    def _do_nothing(self):
        return self
    
    def _swap(self):
        """Swaps agent_state_1 and agent_state_2"""
        #jax.debug.print("Swapping agent states")
        return self._replace(
            agent=self.agent._replace(
                # agent_state=self.agent.agent_state_2,
                # agent_state_2=self.agent.agent_state,
                agent_state=self.agent.agent_state_2,
                agent_state_2=self.agent.agent_state
            )
        )
    def _base_orientation_to_one_hot_forward(self, base_orientation: IntLowDim):
        """
        Converts the base orientation (int 0 to N) to a one-hot encoded vector.
        Use for the forward action.
        """
        return jax.nn.one_hot(base_orientation, AgentConfig().angles_base, dtype=IntLowDim)

    def _base_orientation_to_one_hot_backwards(self, base_orientation: IntLowDim):
        """
        Converts the base orientation (int 0 to N) to a one-hot encoded vector
        for the backwards direction.
        """
        # Create a permutation matrix by shifting the identity
        fwd_to_bkwd_transformation = jnp.roll(jnp.eye(AgentConfig().angles_base, dtype=IntLowDim),
                                              shift=AgentConfig().angles_base // 2, axis=0)
        orientation_one_hot = self._base_orientation_to_one_hot_forward(base_orientation)
        return orientation_one_hot @ fwd_to_bkwd_transformation

    def _get_agent_corners(
        self,
        pos_base: Array,
        base_orientation: IntLowDim,
        agent_width: IntLowDim,
        agent_height: IntLowDim,
    ) -> Array:
        """
        Gets the coordinates of the 4 corners of the agent.
        The function uses a biased rounding strategy to avoid rectangle shrinkage.
        """
        # Determine half dimensions using floor/ceil to properly handle odd dimensions.
        half_width_left = jnp.floor(agent_width / 2.0)
        half_width_right = jnp.ceil(agent_width / 2.0)
        half_height_bottom = jnp.floor(agent_height / 2.0)
        half_height_top = jnp.ceil(agent_height / 2.0)

        # Define corners in local coordinates relative to the center.
        local_corners = jnp.array([
            [-half_width_left, -half_height_bottom],
            [ half_width_right, -half_height_bottom],
            [ half_width_right,  half_height_top],
            [-half_width_left,  half_height_top]
        ])

        # Convert degrees to radians using JAX.
        angle_rad = (base_orientation.astype(jnp.float32) / jnp.array(self.env_cfg.agent.angles_base, dtype=jnp.float32)) * (2 * jnp.pi)
        cos_a = jnp.cos(angle_rad)
        sin_a = jnp.sin(angle_rad)
        # Build the rotation matrix.
        R = jnp.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])
        R = R.squeeze()

        # Rotate local corners and translate by the center position.
        global_corners_float = (R @ local_corners.T).T + jnp.array(pos_base, dtype=float)

        # Bias the rounding: use floor if below the center, ceil otherwise.
        center_arr = jnp.array(pos_base, dtype=float)
        biased_corners = jnp.where(
            global_corners_float < center_arr,
            jnp.floor(global_corners_float),
            jnp.ceil(global_corners_float)
        ).astype(IntLowDim)

        return biased_corners

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

    def _is_valid_move(self, agent_corners: Array) -> Array:
        """
        Checks if the move is valid by computing the agent occupancy mask (using a
        polygon mask) and ensuring all affected grid cells are traversable.
        """
        map_width = self.world.width
        map_height = self.world.height

        # Verify that the corners are within map bounds.
        valid_bounds = jnp.all(
            jnp.logical_and(
                agent_corners >= jnp.array([0, 0]),
                agent_corners < jnp.array([map_width, map_height])
            )
        )

        # Determine the occupancy mask for a grid of size map_width x map_height.
        polygon_mask = compute_polygon_mask(agent_corners, map_width, map_height)

        agent_2_corners_xy = self._get_agent_corners(
            self.agent.agent_state_2.pos_base,
            base_orientation=self.agent.agent_state_2.angle_base,
            agent_width=self.env_cfg.agent.width,
            agent_height=self.env_cfg.agent.height,
        )

        polygon_mask_2 = compute_polygon_mask(agent_2_corners_xy, map_width, map_height)

        
        # Build the traversability mask (0 = traversable, 1 = non-traversable).

        traversability_mask = self._build_traversability_mask(
            self.world.action_map.map, self.world.padding_mask.map
            
        )
        
        
        dig_mask = self._build_dig_dump_cone().reshape(map_width, map_height)
        dig_mask_2 = self._build_dig_dump_cone_2().reshape(map_width, map_height)

        #traversability_mask = jnp.where(dig_mask_2, 1, traversability_mask)
        traversability_mask = jnp.where(polygon_mask_2, 1, traversability_mask)
        #traversability_mask = jnp.where( dig_mask,1, traversability_mask)
        # For a valid move, all cells covered by the agent must be traversable (== 0).
        # Mask out the cells where the agent is located.
        # jnp.where(polygon_mask_2, 1 ,traversability_mask)
        #valid_traversability_2 = jnp.all(jnp.where(polygon_mask_2,dig_mask, 0) == 0)    
        valid_traversability = jnp.all(jnp.where(polygon_mask, traversability_mask, 0) == 0)
        #jax.debug.print("Valid bounds: {valid_bounds}, Valid traversability: {valid_traversability}",valid_bounds=valid_bounds, valid_traversability=valid_traversability)
        return jnp.logical_and(jnp.logical_and(valid_bounds, valid_traversability),valid_traversability)

    @staticmethod
    def _valid_move_to_valid_mask(valid_move: jnp.bool_) -> Array:
        """
        Builds a one-hot encoded 2D vector, encoding a bool value.

        - [1, 0] encodes a False
        - [0, 1] encodes a True
        """
        return jax.nn.one_hot(valid_move.astype(IntLowDim), 2, dtype=IntLowDim)

    def _move_on_orientation(self, orientation_vector: Array) -> "State":
        # Compute the xy delta for a forward move along that angle.
        angles = jnp.linspace(0, 2 * jnp.pi, AgentConfig().angles_base, endpoint=False)
        angles = (angles + (jnp.pi / 2)) % (2 * jnp.pi)
        xy_delta = self.env_cfg.agent.move_tiles * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        delta_xy = orientation_vector @ xy_delta

        # Compute candidate new position and immediately round it to discrete grid points.
        candidate_pos = self.agent.agent_state.pos_base + delta_xy
        candidate_pos = jnp.round(candidate_pos).astype(IntMap)  # Fix: use IntMap not IntLowDim
        candidate_pos = jnp.squeeze(candidate_pos, axis=0)

        # Compute the agent's corners based on the candidate (rounded) position.
        agent_corners_xy = self._get_agent_corners(
            candidate_pos,
            base_orientation=self.agent.agent_state.angle_base,
            agent_width=self.env_cfg.agent.width,
            agent_height=self.env_cfg.agent.height,
        )

        # Check if the new position is valid.
        valid_move = self._is_valid_move(agent_corners_xy)
        valid_move_mask = self._valid_move_to_valid_mask(valid_move)

        # Choose between the old position and the new candidate position.
        old_new_pos = jnp.array([self.agent.agent_state.pos_base, candidate_pos])
        new_pos_base = valid_move_mask @ old_new_pos

        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(pos_base=new_pos_base)
            )
        )

    def _move_on_orientation_with_steering(self, orientation_vector: Array, is_forward: jnp.bool_) -> "State":
        return jax.lax.cond(
            self.agent.agent_state.wheel_angle[0] == 0,
            lambda: self._move_on_orientation(orientation_vector),
            lambda: self._execute_curved_movement(angle_idx_to_rad(self.agent.agent_state.angle_base,
                                                                   self.env_cfg.agent.angles_base),
                                                                   is_forward),
        )

    def _execute_curved_movement(self, orientation_angle: float, is_forward: jnp.bool_) -> "State":
        # Shift to different orientation coordinates
        orientation_angle = orientation_angle + (jnp.pi / 2)

        # For backward movement, reverse the wheel angle effect
        wheel_angle = self.agent.agent_state.wheel_angle[0]
        wheel_angle_rad = jnp.deg2rad(wheel_angle * self.env_cfg.agent.wheel_step)
        # Use width as wheelbase for turning radius calculation
        turn_radius = self.env_cfg.agent.width / (jnp.tan(wheel_angle_rad) + 1e-6)

        # Calculate center of rotation (perpendicular to current orientation)
        # Positive wheel angle means turn left, so center is to the left
        center_offset = np.squeeze(jnp.array([
            -jnp.sin(orientation_angle) * turn_radius,
            jnp.cos(orientation_angle) * turn_radius
        ]))
        center_of_rotation = self.agent.agent_state.pos_base + center_offset

        # Compute how far we move along the arc and new orientation
        angle_change = self.env_cfg.agent.move_tiles / turn_radius
        angle_change = jnp.where(is_forward, angle_change, -angle_change)
        new_base_angle_rad = orientation_angle + angle_change

        # Rotate the digger around the center of rotation
        rotation_matrix = jnp.array([
            [jnp.cos(angle_change), -jnp.sin(angle_change)],
            [jnp.sin(angle_change), jnp.cos(angle_change)]
        ])
        relative_pos = self.agent.agent_state.pos_base - center_of_rotation
        new_relative_pos = rotation_matrix @ relative_pos
        candidate_pos = center_of_rotation + new_relative_pos

        # Round to grid position
        candidate_pos = jnp.round(candidate_pos).astype(IntMap)  # Fix: use IntMap not IntLowDim

        # Calculate new orientation angle in our discrete system
        new_angle_base = jnp.round(
            ((new_base_angle_rad - (jnp.pi / 2)) / (2 * jnp.pi)) * self.env_cfg.agent.angles_base
        ).astype(IntLowDim)
        new_angle_base = new_angle_base % self.env_cfg.agent.angles_base

        # Check if the new position and orientation are valid
        agent_corners_xy = self._get_agent_corners(
            candidate_pos,
            base_orientation=new_angle_base,
            agent_width=self.env_cfg.agent.width,
            agent_height=self.env_cfg.agent.height,
        )

        valid_move = self._is_valid_move(agent_corners_xy)
        valid_move_mask = self._valid_move_to_valid_mask(valid_move)

        # Choose between old and new positions
        old_new_pos = jnp.array([self.agent.agent_state.pos_base, candidate_pos])
        new_pos_base = valid_move_mask @ old_new_pos

        # Choose between old and new angles
        old_new_angle = jnp.array([self.agent.agent_state.angle_base, new_angle_base])
        new_angle_base = valid_move_mask @ old_new_angle

        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    pos_base=new_pos_base,
                    angle_base=new_angle_base
                )
            )
        )

    def _skid_steer_auto_load_dirt(self, new_state: "State") -> "State":
        """
        Auto-loading function for skid steer when moving with shovel lowered.
        Applies soil mechanics directly when dirt is loaded.
        """
        # Only applies to skid steer with shovel lowered and not already loaded
        is_skid_steer = new_state.agent.agent_state.agent_type[0] == 2
        shovel_lowered = new_state.agent.agent_state.shovel_lifted[0] == 0
        not_loaded = new_state.agent.agent_state.loaded[0] == 0
        
        should_auto_load = jnp.logical_and(
            jnp.logical_and(is_skid_steer, shovel_lowered), 
            not_loaded
        )
        
        def _apply_auto_load():
            # Use the closer cylindrical workspace for skid steer auto-loading
            map_cyl_coords, map_local_coords = new_state._get_map_local_and_cyl_coords()
            
            # Get the skid steer cylindrical workspace
            auto_load_mask = new_state._get_dig_dump_mask_skidsteer(map_cyl_coords, map_local_coords)
            
            # Apply skid steer dig masking (only allow loading from existing dirt)
            auto_load_mask = new_state._mask_out_wrong_dig_tiles_skidsteer(auto_load_mask)
            
            # Calculate how much dirt to load (from current state)
            current_flattened_action_map = new_state.world.action_map.map.reshape(-1)
            actual_dirt_to_remove = current_flattened_action_map @ auto_load_mask
            
            def _perform_auto_load():
                # Only perform auto-load if there is dirt to remove
                def _do_load():
                    # First remove dirt cleanly (without soil mechanics)
                    new_flattened_action_map = new_state._apply_dig_mask(
                        current_flattened_action_map,
                        auto_load_mask,
                        moving_dumped_dirt=True  # We're moving existing dirt
                    )
                    new_map_2d = new_flattened_action_map.reshape(new_state.world.action_map.map.shape)
                    
                    # Apply soil mechanics directly using the auto-load mask (the actual holes)
                    def _apply_soil_collapse():
                        auto_load_mask_2d = auto_load_mask.reshape(new_state.world.action_map.map.shape)
                        expanded_mask = new_state._expand_mask_for_soil_mechanics(auto_load_mask_2d)
                        collapsed_map = new_state._apply_local_soil_mechanics(
                            new_map_2d, expanded_mask
                        )
                        return collapsed_map
                    
                    def _skip_soil_collapse():
                        return new_map_2d
                    
                    final_map = jax.lax.cond(
                        ENABLE_SOIL_MECHANICS_IN_TRAINING,
                        _apply_soil_collapse,
                        _skip_soil_collapse
                    )
                    final_map = final_map.astype(new_state.world.action_map.map.dtype)
                    return new_state._replace(
                        world=new_state.world._replace(
                            action_map=new_state.world.action_map._replace(map=final_map),
                        ),
                        agent=new_state.agent._replace(
                            agent_state=new_state.agent.agent_state._replace(
                                loaded=jnp.array([actual_dirt_to_remove], dtype=new_state.agent.agent_state.loaded.dtype),
                            )
                        )
                    )
                def _no_load():
                    # No dirt to load, keep loaded value unchanged
                    return new_state
                return jax.lax.cond(actual_dirt_to_remove > 0, _do_load, _no_load)
            return _perform_auto_load()
        return jax.lax.cond(should_auto_load, _apply_auto_load, lambda: new_state)

    def _handle_move_forward(self) -> "State":
        """
        Moves the base forward with realistic restrictions:
        - Excavators/Wheeled: can only move when not loaded
        - Skid steer: can move when not loaded OR when loaded with shovel lifted
        - Skid steer with lowered shovel + loaded: BLOCKED (forces shovel lifting)
        """

        def _move_forward():
            base_orientation = self.agent.agent_state.angle_base
            orientation_vector = self._base_orientation_to_one_hot_forward(
                base_orientation
            )
            new_state = self._move_on_orientation(orientation_vector)
            
            # Apply auto-loading for skid steer if moved successfully
            return self._skid_steer_auto_load_dirt(new_state)

        # Check agent conditions
        is_skid_steer = self.agent.agent_state.agent_type[0] == 2
        is_loaded = self.agent.agent_state.loaded[0] > 0
        shovel_lifted = self.agent.agent_state.shovel_lifted[0] > 0
        
        # Movement rules:
        # - Non-skid steers: only when not loaded
        # - Skid steer: not loaded OR (loaded AND shovel lifted)
        # - Blocked: skid steer with loaded + lowered shovel
        skid_steer_can_move = jnp.logical_or(
            jnp.logical_not(is_loaded),  # Not loaded - can always move
            jnp.logical_and(is_loaded, shovel_lifted)  # Loaded but shovel lifted - optimal transport
        )
        
        can_move = jnp.logical_or(
            jnp.logical_not(is_skid_steer),  # Non-skid steer (use old logic)
            jnp.logical_and(is_skid_steer, skid_steer_can_move)  # Skid steer with restrictions
        )
        
        # Final check: non-skid steers still can't move when loaded
        can_move = jnp.logical_and(
            can_move,
            jnp.logical_or(is_skid_steer, jnp.logical_not(is_loaded))
        )
        
        return jax.lax.cond(can_move, _move_forward, self._do_nothing)

    def _handle_move_backward(self) -> "State":
        """
        Moves the base backward with realistic restrictions:
        - Excavators/Wheeled: can only move when not loaded
        - Skid steer: can move when not loaded OR when loaded with shovel lifted
        - Skid steer with lowered shovel + loaded: BLOCKED (forces shovel lifting)
        """

        def _move_backward():
            base_orientation = self.agent.agent_state.angle_base
            orientation_vector = self._base_orientation_to_one_hot_backwards(
                base_orientation
            )
            new_state = self._move_on_orientation(orientation_vector)
            
            # No auto-loading on backward movement - only forward movement should auto-load
            return new_state

        # Check agent conditions
        is_skid_steer = self.agent.agent_state.agent_type[0] == 2
        is_loaded = self.agent.agent_state.loaded[0] > 0
        shovel_lifted = self.agent.agent_state.shovel_lifted[0] > 0
        
        # Movement rules:
        # - Non-skid steers: only when not loaded
        # - Skid steer: not loaded OR (loaded AND shovel lifted)
        # - Blocked: skid steer with loaded + lowered shovel
        skid_steer_can_move = jnp.logical_or(
            jnp.logical_not(is_loaded),  # Not loaded - can always move
            jnp.logical_and(is_loaded, shovel_lifted)  # Loaded but shovel lifted - optimal transport
        )
        
        can_move = jnp.logical_or(
            jnp.logical_not(is_skid_steer),  # Non-skid steer (use old logic)
            jnp.logical_and(is_skid_steer, skid_steer_can_move)  # Skid steer with restrictions
        )
        
        # Final check: non-skid steers still can't move when loaded
        can_move = jnp.logical_and(
            can_move,
            jnp.logical_or(is_skid_steer, jnp.logical_not(is_loaded))
        )
        
        return jax.lax.cond(can_move, _move_backward, self._do_nothing)

    def _handle_move_forward_wheeled(self) -> "State":
        """
        Moves the wheeled vehicle forward along an arc determined by wheel angle - if not loaded
        """
        def _move_forward_wheeled():
            base_orientation = self.agent.agent_state.angle_base
            orientation_vector = self._base_orientation_to_one_hot_forward(base_orientation)
            return self._move_on_orientation_with_steering(orientation_vector, jnp.bool_(True))

        return jax.lax.cond(
            self.agent.agent_state.loaded[0] > 0, self._do_nothing, _move_forward_wheeled
        )

    def _handle_move_backward_wheeled(self) -> "State":
        """
        Moves the wheeled vehicle backward along an arc determined by wheel angle - if not loaded
        """
        def _move_backward_wheeled():
            base_orientation = self.agent.agent_state.angle_base
            orientation_vector = self._base_orientation_to_one_hot_backwards(base_orientation)
            return self._move_on_orientation_with_steering(orientation_vector, jnp.bool_(False))

        return jax.lax.cond(
            self.agent.agent_state.loaded[0] > 0, self._do_nothing, _move_backward_wheeled
        )

    def _apply_base_rotation_mask(self, old_angle_base: Array, new_angle_base: Array) -> Array:
        """
        Given an old and a candidate new base angle for the agent, compute the
        rotated polygon using _get_agent_corners (which works for arbitrary rotations)
        and then check if that position is valid.
        If it is, return new_angle_base; otherwise, return old_angle_base.
        """
        # Compute the agent's polygon for the candidate new angle.
        candidate_corners = self._get_agent_corners(
            self.agent.agent_state.pos_base,
            base_orientation=new_angle_base,
            agent_width=self.env_cfg.agent.width,
            agent_height=self.env_cfg.agent.height,
        )
        
        # Check if this rotated polygon is valid.
        valid_move = self._is_valid_move(candidate_corners)
        
        # Choose the new angle if valid, else the old angle.
        return jax.lax.cond(valid_move, 
                            lambda: new_angle_base,
                            lambda: old_angle_base)

    def _apply_cabin_rotation_mask(self, old_angle_cabin: Array, new_angle_cabin: Array) -> Array:
        """
        Given an old and a candidate new cabin angle, check if the rotation is valid.
        For cabin rotation, we don't need to check body collision (since the body doesn't move),
        but we should ensure the new cabin angle is within valid bounds.
        Since cabin angles are already constrained by the circular angle system,
        we can just return the new angle.
        """
        # Cabin rotation is always valid since it's just changing the arm direction
        # and the angle system already constrains it to valid values
        return new_angle_cabin

    def _handle_clock(self) -> "State":
        def _rotate_clock():
            old_angle_base = self.agent.agent_state.angle_base
            new_angle_base = decrease_angle_circular(
                old_angle_base, self.env_cfg.agent.angles_base
            )
            new_angle_base = self._apply_base_rotation_mask(
                old_angle_base, new_angle_base
            )

            return self._replace(
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        angle_base=new_angle_base
                    )
                )
            )

        # Check agent conditions
        is_skid_steer = self.agent.agent_state.agent_type[0] == 2
        is_loaded = self.agent.agent_state.loaded[0] > 0
        shovel_lifted = self.agent.agent_state.shovel_lifted[0] > 0
        
        # Rotation rules:
        # - Non-skid steers: only when not loaded (existing logic)
        # - Skid steer: not loaded OR (loaded AND shovel lifted)
        # - Blocked: skid steer with loaded + lowered shovel (realistic constraint)
        skid_steer_can_rotate = jnp.logical_or(
            jnp.logical_not(is_loaded),  # Not loaded - can always rotate
            jnp.logical_and(is_loaded, shovel_lifted)  # Loaded but shovel lifted - safe rotation
        )
        
        can_rotate = jnp.logical_or(
            jnp.logical_not(is_skid_steer),  # Non-skid steer (use old logic)
            jnp.logical_and(is_skid_steer, skid_steer_can_rotate)  # Skid steer with realistic restrictions
        )
        
        # Final check: non-skid steers still can't rotate when loaded
        can_rotate = jnp.logical_and(
            can_rotate,
            jnp.logical_or(is_skid_steer, jnp.logical_not(is_loaded))
        )
        
        return jax.lax.cond(can_rotate, _rotate_clock, self._do_nothing)

    def _handle_anticlock(self) -> "State":
        def _rotate_anticlock():
            old_angle_base = self.agent.agent_state.angle_base
            new_angle_base = increase_angle_circular(
                old_angle_base, self.env_cfg.agent.angles_base
            )
            new_angle_base = self._apply_base_rotation_mask(
                old_angle_base, new_angle_base
            )

            return self._replace(
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        angle_base=new_angle_base
                    )
                )
            )

        # Check agent conditions
        is_skid_steer = self.agent.agent_state.agent_type[0] == 2
        is_loaded = self.agent.agent_state.loaded[0] > 0
        shovel_lifted = self.agent.agent_state.shovel_lifted[0] > 0
        
        # Rotation rules:
        # - Non-skid steers: only when not loaded (existing logic)
        # - Skid steer: not loaded OR (loaded AND shovel lifted)
        # - Blocked: skid steer with loaded + lowered shovel (realistic constraint)
        skid_steer_can_rotate = jnp.logical_or(
            jnp.logical_not(is_loaded),  # Not loaded - can always rotate
            jnp.logical_and(is_loaded, shovel_lifted)  # Loaded but shovel lifted - safe rotation
        )
        
        can_rotate = jnp.logical_or(
            jnp.logical_not(is_skid_steer),  # Non-skid steer (use old logic)
            jnp.logical_and(is_skid_steer, skid_steer_can_rotate)  # Skid steer with realistic restrictions
        )
        
        # Final check: non-skid steers still can't rotate when loaded
        can_rotate = jnp.logical_and(
            can_rotate,
            jnp.logical_or(is_skid_steer, jnp.logical_not(is_loaded))
        )
        
        return jax.lax.cond(can_rotate, _rotate_anticlock, self._do_nothing)

    def _handle_cabin_clock(self) -> "State":
        """Handle cabin clockwise rotation. Does nothing for skid steer."""
        # Skid steer cannot rotate cabin
        is_skid_steer = self.agent.agent_state.agent_type[0] == 2
        
        def _cabin_clock():
            old_angle_cabin = self.agent.agent_state.angle_cabin
            new_angle_cabin = decrease_angle_circular(
                old_angle_cabin, self.env_cfg.agent.angles_cabin
            )

            return self._replace(
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        angle_cabin=new_angle_cabin
                    )
                )
            )
        
        return jax.lax.cond(is_skid_steer, self._do_nothing, _cabin_clock)

    def _handle_cabin_anticlock(self) -> "State":
        """Handle cabin anti-clockwise rotation. Does nothing for skid steer."""
        # Skid steer cannot rotate cabin
        is_skid_steer = self.agent.agent_state.agent_type[0] == 2
        
        def _cabin_anticlock():
            old_angle_cabin = self.agent.agent_state.angle_cabin
            new_angle_cabin = increase_angle_circular(
                old_angle_cabin, self.env_cfg.agent.angles_cabin
            )

            return self._replace(
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        angle_cabin=new_angle_cabin
                    )
                )
            )
        
        return jax.lax.cond(is_skid_steer, self._do_nothing, _cabin_anticlock)

    def _handle_turn_wheels_left(self) -> "State":
        old_wheel_angle = self.agent.agent_state.wheel_angle
        new_wheel_angle = jnp.min(
            jnp.array([
                old_wheel_angle + 1,
                jnp.full((1,), fill_value=self.env_cfg.agent.max_wheel_angle, dtype=IntLowDim),
            ]),
            axis=0,
        )

        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    wheel_angle=new_wheel_angle
                )
            )
        )

    def _handle_turn_wheels_right(self) -> "State":
        old_wheel_angle = self.agent.agent_state.wheel_angle
        new_wheel_angle = jnp.max(
            jnp.array([
                old_wheel_angle - 1,
                jnp.full((1,), fill_value=-self.env_cfg.agent.max_wheel_angle, dtype=IntLowDim),
            ]),
            axis=0,
        )

        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    wheel_angle=new_wheel_angle
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
    
    def _get_cabin_angle_rad_2(self) -> Float:
        return angle_idx_to_rad(
            self.agent.agent_state_2.angle_cabin, self.env_cfg.agent.angles_cabin
        )

    def _get_base_angle_rad_2(self) -> Float:
        return angle_idx_to_rad(
            self.agent.agent_state_2.angle_base, self.env_cfg.agent.angles_base
        )

    def _get_arm_angle_rad(self) -> Float:
        base_angle = self._get_base_angle_rad()
        cabin_angle = self._get_cabin_angle_rad()
        return wrap_angle_rad(base_angle + cabin_angle)


    def _get_arm_angle_rad_2(self) -> Float:
        base_angle = self._get_base_angle_rad_2()
        cabin_angle = self._get_cabin_angle_rad_2()
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
        theta_max = 2 * np.pi / self.env_cfg.agent.angles_cabin
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
            if we are digging dirt, then we dig as much as self.env_cfg.agent.dig_depth

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
        # Removed apply_soil_mechanics param, always skip for dumping
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
        # Check if this is a skid steer - they can dump anywhere in workspace
        is_skid_steer = self.agent.agent_state.agent_type[0] == 2
        
        # For skid steers, use original dump mask without target map filtering
        # For excavators/wheeled, apply target map filtering as before
        def _apply_target_filtering():
            # Check if there is any target dump tile within the mask
            target_map_dump_mask = jnp.clip(target_map.reshape(-1), a_min=0) * dump_mask
            target_dump_volume = target_map_dump_mask.sum()
            return jax.lax.cond(
                target_dump_volume > 0,
                lambda: (IntMap(target_map_dump_mask), target_dump_volume),
                lambda: (IntMap(dump_mask), dump_mask.sum()),
            )
        
        def _use_original_mask():
            # Use original dump mask without target map filtering
            return (IntMap(dump_mask), dump_mask.sum())
        
        dump_mask, dump_volume = jax.lax.cond(
            is_skid_steer,
            _use_original_mask,
            _apply_target_filtering
        )

        # Use the safely calculated values passed in from _apply_dump
        # Don't recalculate here to avoid division by zero
        
        # CONDITIONAL SOIL MECHANICS: Use simplified dumping for training performance
        def _apply_simple_dump():
            """Simple uniform distribution without Gaussian spreading or soil mechanics"""
            # Distribute dirt uniformly across selected tiles
            volume_per_tile = even_volume_per_tile * dump_mask.astype(IntMap)
            
            # Handle remaining volume by adding to first N tiles
            remaining_units = remaining_volume
            bonus_mask = jnp.arange(len(dump_mask)) < remaining_units
            bonus_volume = (dump_mask * bonus_mask).astype(IntMap)
            
            new_flattened_map = (flattened_map + volume_per_tile + bonus_volume).astype(IntMap)
            return new_flattened_map
        
        def _apply_gaussian_dump():
            """Dump with Gaussian distribution only (no soil mechanics)"""
            dump_volume = even_volume_per_tile * jnp.sum(dump_mask) + remaining_volume
            map_2d_shape = self.world.action_map.map.shape
            dump_mask_2d = dump_mask.reshape(map_2d_shape)
            y_coords, x_coords = jnp.meshgrid(jnp.arange(map_2d_shape[0]), jnp.arange(map_2d_shape[1]), indexing='ij')
            centroid_y = jnp.sum(y_coords * dump_mask_2d) / jnp.sum(dump_mask_2d)
            centroid_x = jnp.sum(x_coords * dump_mask_2d) / jnp.sum(dump_mask_2d)
            distances_sq = (y_coords - centroid_y)**2 + (x_coords - centroid_x)**2
            gaussian = jnp.exp(-distances_sq / 8.0)
            masked_gaussian = gaussian * dump_mask_2d
            normalized_weights = masked_gaussian / jnp.sum(masked_gaussian)
            volume_per_tile = normalized_weights.flatten() * dump_volume
            floor_values = jnp.floor(volume_per_tile).astype(IntMap)
            fractional_parts = volume_per_tile - floor_values
            remaining_units = dump_volume - jnp.sum(floor_values)
            sorted_indices = jnp.argsort(-fractional_parts)
            bonus_mask = jnp.arange(len(floor_values)) < remaining_units
            reordered_bonus = jnp.zeros_like(floor_values).at[sorted_indices].set(bonus_mask.astype(IntMap))
            new_flattened_map = (flattened_map + floor_values + reordered_bonus).astype(IntMap)
            return new_flattened_map
        
        # Choose between simple and full dump based on performance flag
        return jax.lax.cond(
            ENABLE_SOIL_MECHANICS_IN_TRAINING,
            _apply_gaussian_dump,
            _apply_simple_dump
        )

    def _apply_dig_mask_with_soil_mechanics(
        self, 
        flattened_map: Array, 
        dig_mask: Array, 
        moving_dumped_dirt: bool,
        apply_soil_mechanics: bool = True,
        collapse_threshold: float = 2.0,
        collapse_alpha: float = 0.5,
        use_iterative_collapse: bool = True
    ) -> Array:
        """
        Enhanced version of _apply_dig_mask that optionally applies simple or iterative soil mechanics.
        When enabled, after lifting dirt, dirt from adjacent border tiles collapses into the cone if the border is much higher.
        If use_iterative_collapse is True, perform up to 3 local relaxation steps starting from the cone.
        JAX/JIT compatible, global mask-based, efficient.
        Parameters:
            collapse_threshold: difference required to trigger collapse
            collapse_alpha: fraction of difference to move
            use_iterative_collapse: if True, use iterative local relaxation (3 steps)
        """
        # Apply the normal dig mask logic first
        delta_dig = self.env_cfg.agent.dig_depth * dig_mask.astype(IntMap)
        new_flattened_map = jax.lax.cond(
            moving_dumped_dirt,
            lambda: jnp.where(dig_mask, 0, flattened_map).astype(IntMap),
            lambda: (flattened_map - delta_dig).astype(IntMap),
        )

        def _apply_simple_collapse():
            # Reshape to 2D
            map_2d = new_flattened_map.reshape(self.world.action_map.map.shape)
            cone_mask = dig_mask.reshape(self.world.action_map.map.shape).astype(jnp.bool_)
            # Border mask: dilate cone, subtract cone
            kernel = jnp.ones((3, 3), dtype=jnp.float32)
            dilated = jax.scipy.signal.convolve2d(cone_mask.astype(jnp.float32), kernel, mode='same', boundary='fill', fillvalue=0.0) > 0
            border_mask = jnp.logical_and(dilated, ~cone_mask)
            # For each cone tile, check max neighbor (border) height
            # Use 2D convolution to get max border height for each cone tile
            border_heights = jnp.where(border_mask, map_2d, -jnp.inf)
            # Replace maximum_filter with lax.reduce_window
            max_border = lax.reduce_window(
                border_heights,
                -jnp.inf,
                lax.max,
                window_dimensions=(3, 3),
                window_strides=(1, 1),
                padding='SAME'
            )
            # Only consider for cone tiles
            max_border = jnp.where(cone_mask, max_border, 0.0)
            # Compute difference
            diff = max_border - map_2d
            flow = jnp.where((cone_mask) & (diff > collapse_threshold), collapse_alpha * diff, 0.0)
            # Subtract from border, add to cone
            # For each cone tile, find which border tile(s) contributed max (could be multiple)
            # For simplicity, just subtract total flow from all border tiles equally (approximate)
            # Distribute flow equally to all border tiles
            total_flow = jnp.sum(flow)
            n_border = jnp.sum(border_mask)
            border_flow = jnp.where(border_mask, -total_flow / jnp.maximum(n_border, 1), 0.0)
            # Update map
            new_map_2d = map_2d + flow + border_flow
            return new_map_2d.reshape(-1).astype(IntMap)

        def _apply_iterative_collapse():
            map_2d = new_flattened_map.reshape(self.world.action_map.map.shape).astype(jnp.float32)
            cone_mask = dig_mask.reshape(self.world.action_map.map.shape).astype(jnp.bool_)
            # Border mask: dilate cone, subtract cone
            kernel = jnp.ones((3, 3), dtype=jnp.float32)
            dilated = jax.scipy.signal.convolve2d(cone_mask.astype(jnp.float32), kernel, mode='same', boundary='fill', fillvalue=0.0) > 0
            border_mask = jnp.logical_and(dilated, ~cone_mask)
            update_mask = jnp.logical_or(cone_mask, border_mask)

            def body_fn(i, map_2d):
                # For each cone tile, get max neighbor in update region
                border_heights = jnp.where(border_mask, map_2d, -jnp.inf)
                max_border = lax.reduce_window(
                    border_heights,
                    -jnp.inf,
                    lax.max,
                    window_dimensions=(3, 3),
                    window_strides=(1, 1),
                    padding='SAME'
                )
                # Only consider for cone tiles
                max_border = jnp.where(cone_mask, max_border, 0.0)
                diff = max_border - map_2d
                flow = jnp.where((cone_mask) & (diff > collapse_threshold), collapse_alpha * diff, 0.0)
                # Subtract from border, add to cone
                total_flow = jnp.sum(flow)
                n_border = jnp.sum(border_mask)
                border_flow = jnp.where(border_mask, -total_flow / jnp.maximum(n_border, 1), 0.0)
                # Only update in update_mask region
                new_map_2d = jnp.where(update_mask, map_2d + flow + border_flow, map_2d)
                return new_map_2d
            # Run 3 iterations
            map_2d_final = lax.fori_loop(0, 3, body_fn, map_2d)
            return map_2d_final.reshape(-1).astype(IntMap)

        def _apply_soil_mech_dispatch():
            return jax.lax.cond(
                use_iterative_collapse,
                _apply_iterative_collapse,
                _apply_simple_collapse
            )
        return jax.lax.cond(
            apply_soil_mechanics,
            _apply_soil_mech_dispatch,
            lambda: new_flattened_map
        )

    def _expand_mask_for_soil_mechanics(self, mask: Array) -> Array:
        """
        Expand the mask to include all valid neighbors (3x3 kernel).
        Only includes neighbors that are valid for dumping (not obstacles, is dumpable).
        """
        H, W = mask.shape
        padding_mask = self.world.padding_mask.map
        dumpability_mask = self.world.dumpability_mask.map
        
        # Create validity mask - tiles where dirt can be placed
        validity_mask = jnp.logical_and(
            padding_mask == 0,  # Not an obstacle
            dumpability_mask == 1  # Is dumpable
        )
        
        # Use convolution to expand the mask to include neighbors
        # Create a 3x3 kernel that includes center and all 8 neighbors
        kernel = jnp.ones((3, 3), dtype=jnp.float32)
        
        # Convert mask to float for convolution
        mask_float = mask.astype(jnp.float32)
        
        # Apply 2D convolution to expand the mask
        # This is JAX-compatible and vectorized
        expanded_float = jax.scipy.signal.correlate2d(
            mask_float, kernel, mode='same', boundary='fill', fillvalue=0.0
        )
        
        # Convert back to boolean - any neighbor (including self) was affected
        expanded = expanded_float > 0
        
        # CRITICAL: Only include valid tiles in the expanded mask
        # This ensures soil mechanics don't affect obstacles or non-dumpable areas
        return jnp.logical_and(expanded, validity_mask)

    def _get_map_local_and_cyl_coords_2(self):
        """
        Returns:
            - map_cyl_coords: (2, width*height) map with [r, theta] rows
            - map_local_coords_base: (2, width*height) map with [x, y] rows
        """
        current_pos_idx = self._get_current_pos_vector_idx(
            pos_base=self.agent.agent_state_2.pos_base,
            map_height=self.env_cfg.maps.edge_length_px,
        )
        map_global_coords = self._map_to_flattened_global_coords(
            self.world.width, self.world.height, self.env_cfg.tile_size
        )
        current_pos = self._get_current_pos_from_flattened_map(
            map_global_coords, current_pos_idx
        )
        current_arm_angle = self._get_arm_angle_rad_2()

        # Local coordinates including the cabin rotation
        current_state_arm = jnp.hstack((current_pos, current_arm_angle))
        map_local_coords_arm = apply_rot_transl(current_state_arm, map_global_coords)
        map_cyl_coords = apply_local_cartesian_to_cyl(map_local_coords_arm)

        # Local coordinates excluding the cabin rotation
        current_state_base = jnp.hstack((current_pos, self._get_base_angle_rad_2()))
        map_local_coords_base = apply_rot_transl(current_state_base, map_global_coords)
        return map_cyl_coords, map_local_coords_base
    
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

    def _build_dig_dump_cone_standalone(self) -> Array:
        """
        Returns the masked workspace cone in cartesian coords without any function dependencies.
        All transformations and calculations are performed inline.
        """
        # Get agent state information
        pos_base = self.agent.agent_state.pos_base
        map_width = self.world.width
        map_height = self.world.height
        tile_size = self.env_cfg.tile_size
        
        # Calculate position index in flattened map
        current_pos_idx = pos_base @ jnp.array([[self.env_cfg.maps.edge_length_px], [1]], dtype=IntMap)
        
        # Create flattened global coordinates
        tile_offset = tile_size / 2
        x_row = jnp.tile(jnp.vstack(jnp.arange(map_width)), map_height).reshape(-1)
        y_row = jnp.tile(jnp.arange(map_height), map_width)
        map_global_coords = jnp.vstack([x_row, y_row])
        map_global_coords = map_global_coords * tile_size
        map_global_coords = map_global_coords + tile_offset
        
        # Get current position from flattened map
        num_tiles = map_global_coords.shape[1]
        idx_one_hot = jax.nn.one_hot(current_pos_idx, num_tiles, dtype=Float)
        current_pos = map_global_coords @ idx_one_hot[0]
        # Ensure current_pos is properly shaped
        current_pos = jnp.reshape(current_pos, (2,))
        
        # Get angles and ensure they're scalars
        base_angle_rad = angle_idx_to_rad(
            self.agent.agent_state.angle_base, self.env_cfg.agent.angles_base
        )
        base_angle_rad = jnp.squeeze(base_angle_rad)  # Ensure scalar
    
        cabin_angle_rad = angle_idx_to_rad(
            self.agent.agent_state.angle_cabin, self.env_cfg.agent.angles_cabin
        )
        cabin_angle_rad = jnp.squeeze(cabin_angle_rad)  # Ensure scalar
    
        # Calculate arm angle and ensure it's a scalar
        arm_angle_rad = wrap_angle_rad(base_angle_rad + cabin_angle_rad)
        arm_angle_rad = jnp.squeeze(arm_angle_rad)  # Ensure scalar
    
        # Calculate local coordinates with arm rotation (apply_rot_transl inlined)
        cos_a_arm = jnp.cos(arm_angle_rad)
        sin_a_arm = jnp.sin(arm_angle_rad)
        # Create 2x2 rotation matrix
        rotation_arm = jnp.array([
            [cos_a_arm, -sin_a_arm],
            [sin_a_arm, cos_a_arm]
        ])
    
        # Apply rotation and translation for arm coordinates
        map_local_coords_arm = rotation_arm @ map_global_coords
        map_local_coords_arm = map_local_coords_arm - rotation_arm @ current_pos.reshape(2, 1)
    
        # Convert to cylindrical coordinates
        r = jnp.sqrt(map_local_coords_arm[0]**2 + map_local_coords_arm[1]**2)
        theta = jnp.arctan2(map_local_coords_arm[1], map_local_coords_arm[0])
        map_cyl_coords = jnp.vstack([r, theta])
    
        # Create base local coordinates
        cos_a_base = jnp.cos(base_angle_rad)
        sin_a_base = jnp.sin(base_angle_rad)
        rotation_base = jnp.array([
            [cos_a_base, -sin_a_base],
            [sin_a_base, cos_a_base]
        ])
        map_local_coords_base = rotation_base @ map_global_coords
        map_local_coords_base = map_local_coords_base - rotation_base @ current_pos.reshape(2, 1)
    
        # Rest of the function remains unchanged
        # Calculate dig mask from cylindrical coordinates
        dig_portion_radius = self.env_cfg.agent.move_tiles
        max_agent_dim = jnp.max(
            jnp.array([self.env_cfg.agent.width / 2, self.env_cfg.agent.height / 2])
        )
        min_distance_from_agent = tile_size * max_agent_dim
        
        # Fixed extension parameters
        fixed_extension = 0.5
        r_min = fixed_extension * dig_portion_radius * tile_size + min_distance_from_agent
        r_max = (fixed_extension + 1) * dig_portion_radius * tile_size + min_distance_from_agent
        theta_max = 2 * jnp.pi / self.env_cfg.agent.angles_cabin
        theta_min = -theta_max
        
        # Apply radius and angle constraints
        dig_mask_r = jnp.logical_and(
            map_cyl_coords[0] >= r_min, map_cyl_coords[0] <= r_max
        )
        dig_mask_theta = jnp.logical_and(
            map_cyl_coords[1] >= theta_min, map_cyl_coords[1] <= theta_max
        )
        dig_dump_mask_cyl = jnp.logical_and(dig_mask_r, dig_mask_theta)
        
        # Apply agent size constraints (from _get_dig_dump_mask)
        agent_width = self.env_cfg.agent.width * tile_size
        agent_height = self.env_cfg.agent.height * tile_size
        eps = tile_size / 2  # margin to avoid rounding errors
        
        # Check if coordinates are outside agent's bounds
        dig_dump_mask_cart_x = jnp.where(
            jnp.logical_or(
                map_local_coords_base[0] >= jnp.floor((agent_width + eps) / 2),
                map_local_coords_base[0] <= -jnp.floor((agent_width + eps) / 2),
            ),
            1,
            0,
        )
        dig_dump_mask_cart_y = jnp.where(
            jnp.logical_or(
                map_local_coords_base[1] >= jnp.floor((agent_height + eps) / 2),
                map_local_coords_base[1] <= -jnp.floor((agent_height + eps) / 2),
            ),
            1,
            0,
        )
        dig_dump_mask_cart = (dig_dump_mask_cart_x + dig_dump_mask_cart_y).astype(jnp.bool_)
        
        # Combine cylindrical and cartesian masks
        dig_dump_mask = dig_dump_mask_cyl * dig_dump_mask_cart
        
        # Reshape to match map dimensions
        return dig_dump_mask.reshape(map_width, map_height)
        
    def _get_dig_dump_mask_rectangular_skidsteer(self, map_local_coords: Array) -> Array:
        """
        Creates a rectangular workspace for skid steer that rotates with the agent.
        Uses local coordinates but with more stable rectangular logic.
        
        Args:
            - map_local_coords: (2, N) Array with [x, y] rows in local coordinates
        Returns:
            - dig_mask: (N, ) Array of bools, where True means can interact here
        """
        tile_size = self.env_cfg.tile_size
        
        # Skid steer specific parameters - large, well-positioned rectangle
        # Width: 8 tiles (wide and clearly visible)
        # Depth: 6 tiles (substantial depth)
        # Start: 5.0 tiles in front of agent (well separated from agent)
        
        workspace_width_tiles = 8.0
        workspace_depth_tiles = 6.0
        workspace_start_tiles = 5.0
        
        workspace_width = workspace_width_tiles * tile_size
        workspace_depth = workspace_depth_tiles * tile_size
        workspace_start = workspace_start_tiles * tile_size
        
        # Local coordinates relative to agent
        local_x = map_local_coords[0]
        local_y = map_local_coords[1]
        
        # Create rectangular workspace in front of agent (positive Y direction)
        # X limits: within workspace width, centered on agent
        x_mask = jnp.logical_and(
            local_x >= -workspace_width / 2,
            local_x <= workspace_width / 2
        )
        
        # Y limits: from workspace_start to workspace_start + workspace_depth
        y_mask = jnp.logical_and(
            local_y >= workspace_start,
            local_y <= workspace_start + workspace_depth
        )
        
        # Combine to create rectangular workspace
        rectangular_mask = jnp.logical_and(x_mask, y_mask)
        
        # Simplified agent exclusion - just exclude a small area around origin
        # This avoids complex coordinate transformations that might cause shape changes
        agent_exclusion_radius = 2.0 * tile_size  # 2 tile radius around agent
        distance_from_agent = jnp.sqrt(local_x**2 + local_y**2)
        agent_exclusion_mask = distance_from_agent > agent_exclusion_radius
        
        return jnp.logical_and(rectangular_mask, agent_exclusion_mask)

    def _build_dig_dump_cone(self) -> Array:
        """
        Returns the masked workspace cone in cartesian coords. Every tile in the cone is included as +1.
        Uses different workspace shapes for different agent types:
        - Excavator/Wheeled: Cylindrical cone (original parameters)
        - Skid Steer: Cylindrical cone (closer, smaller parameters)
        """
        current_agent_type = self.agent.agent_state.agent_type[0]
        
        # Get coordinates for cylindrical approach (used by both agent types)
        map_cyl_coords, map_local_coords_base = self._get_map_local_and_cyl_coords()
        
        def _get_excavator_cone():
            return self._get_dig_dump_mask(map_cyl_coords, map_local_coords_base)
        
        def _get_skidsteer_cone():
            return self._get_dig_dump_mask_skidsteer(map_cyl_coords, map_local_coords_base)
        
        # Use JAX conditional to select workspace type based on agent type
        return jax.lax.cond(
            current_agent_type == 2,  # Skid steer
            _get_skidsteer_cone,
            _get_excavator_cone       # Default for excavator (0) and wheeled (1)
        )
    

    def _build_dig_dump_cone_2(self) -> Array:
        """
        Returns the masked workspace cone in cartesian coords. Every tile in the cone is included as +1.
        Uses different workspace shapes for different agent types:
        - Excavator/Wheeled: Cylindrical cone (original parameters)
        - Skid Steer: Cylindrical cone (closer, smaller parameters)
        """
        current_agent_type = self.agent.agent_state_2.agent_type[0]
        
        # Get coordinates for cylindrical approach (used by both agent types)
        map_cyl_coords, map_local_coords_base = self._get_map_local_and_cyl_coords_2()
        
        def _get_excavator_cone():
            return self._get_dig_dump_mask(map_cyl_coords, map_local_coords_base)
        
        def _get_skidsteer_cone():
            return self._get_dig_dump_mask_skidsteer(map_cyl_coords, map_local_coords_base)
        
        # Use JAX conditional to select workspace type based on agent type
        return jax.lax.cond(
            current_agent_type == 2,  # Skid steer
            _get_skidsteer_cone,
            _get_excavator_cone       # Default for excavator (0) and wheeled (1)
        )

    def _exclude_dig_tiles_from_dump_mask(self, dump_mask: Array) -> Array:
        """
        Takes the dump mask and turns into False the elements that correspond to
        a dug tile.
        """
        digged_mask_action_map = self.world.action_map.map < 0
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
                self.world.last_dig_mask.map.reshape(-1)
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

    def _mask_out_wrong_dig_tiles_skidsteer(self, dig_mask: Array) -> Array:
        """
        For skid steer: Allow lifting from any dirt (action_map != 0)
        NEVER allow digging new holes (target_map < 0)
        """
        # Allow lifting from any dirt (action_map != 0) - both natural and dumped
        dig_mask_action_map = self.world.action_map.map != 0
        
        # Respect max dig limit
        max_dig_limit_mask = (
            self.world.action_map.map > -self.env_cfg.agent.dig_depth
        ).reshape(-1)

        return (
            dig_mask
            * dig_mask_action_map.reshape(-1)  # Any dirt tiles (natural or dumped)
            * max_dig_limit_mask
        ).astype(jnp.bool_)

    def _get_new_dumpability_mask(self, action_map: Array) -> Array:
        new_dumpability_mask = self.world.dumpability_mask_init.map
        action_mask = (action_map < 0).astype(jnp.float16)
        kernel = jnp.ones((5, 5), dtype=jnp.float16)
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
        dig_mask = self._mask_out_wrong_dig_tiles(dig_mask)
        flattened_action_map = self.world.action_map.map.reshape(-1)
        selected_tiles_sum = flattened_action_map @ dig_mask
        moving_dumped_dirt = selected_tiles_sum > 0
        # if moving dumped dirt, move it all at once
        # Ensure both branches return the same dtype (int32)
        dig_volume = jax.lax.cond(
            moving_dumped_dirt,
            lambda: selected_tiles_sum.astype(jnp.int32),
            lambda: dig_mask.sum().astype(jnp.int32),
        )

        def _apply_dig(volume, fam):
            # First remove dirt cleanly (without soil mechanics)
            new_map_global_coords = self._apply_dig_mask(
                fam, dig_mask, moving_dumped_dirt
            )
            new_map_global_coords = new_map_global_coords.reshape(
                self.world.action_map.map.shape
            )
            
            # Now apply soil mechanics using the saved cone mask
            def _apply_soil_collapse():
                # Use the saved cone mask for soil mechanics
                cone_mask_2d = dig_mask.reshape(self.world.action_map.map.shape)
                
                # Apply soil mechanics to collapse dirt into the hole
                expanded_mask = self._expand_mask_for_soil_mechanics(cone_mask_2d)
                collapsed_map = self._apply_local_soil_mechanics(
                    new_map_global_coords, expanded_mask
                )
                return collapsed_map
            
            def _skip_soil_collapse():
                return new_map_global_coords
            
            # Apply soil mechanics only if enabled during training
            final_map = jax.lax.cond(
                ENABLE_SOIL_MECHANICS_IN_TRAINING,
                _apply_soil_collapse,
                _skip_soil_collapse
            )
            
            # CONSERVATION FIX: Calculate the actual amount removed after soil mechanics
            # The soil mechanics might move additional dirt from border into cone
            # So we need to calculate the difference between original and new map
            original_total = jnp.sum(fam)
            new_total = jnp.sum(final_map.reshape(-1))
            actual_volume_loaded = original_total - new_total  # This is the actual amount removed
            
            new_dumpability_mask = self._get_new_dumpability_mask(
                final_map,
            )

            return self._replace(
                world=self.world._replace(
                    action_map=self.world.action_map._replace(
                        map=IntLowDim(final_map)
                    ),
                    dumpability_mask=self.world.dumpability_mask._replace(
                        map=jnp.bool_(new_dumpability_mask),
                    ),
                    last_dig_mask=self.world.last_dig_mask._replace(
                        map=jnp.bool_(dig_mask.reshape(self.world.action_map.map.shape)),
                    )
                ),
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        loaded=jnp.array([actual_volume_loaded], dtype=IntLowDim)
                    ),
                    moving_dumped_dirt=jnp.bool_(moving_dumped_dirt),
                )
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

        def _apply_dump():
            # Calculate volume distribution only when we actually have a valid dump area
            remaining_volume = self.agent.agent_state.loaded % dump_volume
            even_volume_per_tile = (
                self.agent.agent_state.loaded - remaining_volume
            ) / dump_volume
            
            flattened_action_map = self.world.action_map.map.reshape(-1)
            new_map_global_coords = self._apply_dump_mask(
                flattened_action_map,
                dump_mask,
                even_volume_per_tile,
                remaining_volume,
                self.world.target_map.map,
            )
            new_map_global_coords = new_map_global_coords.reshape(
                self.world.target_map.map.shape
            )

            return self._replace(
                world=self.world._replace(
                    action_map=self.world.action_map._replace(
                        map=IntLowDim(new_map_global_coords)
                    ),
                    last_dig_mask=self.world.last_dig_mask._replace(
                        map=jnp.zeros_like(self.world.last_dig_mask.map, dtype=jnp.bool_)
                    ),
                ),
                agent=self.agent._replace(
                    agent_state=self.agent.agent_state._replace(
                        loaded=jnp.full((1,), fill_value=0, dtype=IntLowDim)
                    ),
                    moving_dumped_dirt=jnp.bool_(False),
                ),
            )

        return jax.lax.cond(dump_volume > 0, _apply_dump, self._do_nothing)





    def _handle_lift_dumped_dirt(self) -> "State":
        """
        For skid steer: Lifting the shovel only toggles the shovel state to up.
        Does NOT move dirt or change loaded. All loading is handled in auto-load.
        """
        return self._replace(
            agent=self.agent._replace(
                agent_state=self.agent.agent_state._replace(
                    shovel_lifted=jnp.array([1], dtype=IntLowDim)
                )
            )
        )

    def _handle_do(self) -> "State":
        """
        Handle the DO action based on agent type:
        - Tracked/Wheeled (0,1): dig (not loaded) / dump (loaded)
        - Skid steer (2): simple shovel control:
          * If shovel lifted + loaded: dump dirt and lower shovel
          * If shovel lifted + not loaded: lower shovel  
          * If shovel lowered: lift shovel (regardless of loading)
        """
        is_skid_steer = self.agent.agent_state.agent_type[0] == 2
        
        def _skid_steer_do():
            is_loaded = self.agent.agent_state.loaded[0] > 0
            shovel_lifted = self.agent.agent_state.shovel_lifted[0] > 0
            
            def _dump_and_lower():
                # Always try to dump dirt and then lower shovel regardless of dump success
                dumped_state = self._handle_dump()
                # Always lower the shovel after dump attempt
                return dumped_state._replace(
                    agent=dumped_state.agent._replace(
                        agent_state=dumped_state.agent.agent_state._replace(
                            shovel_lifted=jnp.array([0], dtype=IntLowDim)  # Lower shovel
                        )
                    )
                )
            
            def _lift_shovel():
                # Check if agent is loaded
                agent_is_loaded = self.agent.agent_state.loaded[0] > 0
                
                def _lift_with_soil_mechanics():
                    # Agent has dirt, so lifting will disturb surrounding soil
                    # First lift the dirt with soil mechanics (already applied in _handle_lift_dumped_dirt)
                    lifted_state = self._handle_lift_dumped_dirt()
                    
                    # Just lift the shovel (soil mechanics already applied)
                    return lifted_state._replace(
                        agent=lifted_state.agent._replace(
                            agent_state=lifted_state.agent.agent_state._replace(
                                shovel_lifted=jnp.array([1], dtype=IntLowDim)  # Ensure shovel is lifted
                            )
                        )
                    )
                
                def _lift_without_soil_mechanics():
                    # Agent has no dirt, just lift shovel without terrain disturbance
                    return self._replace(
                        agent=self.agent._replace(
                            agent_state=self.agent.agent_state._replace(
                                shovel_lifted=jnp.array([1], dtype=IntLowDim)  # Lift shovel
                            )
                        )
                    )
                
                # Apply soil mechanics only if agent is loaded (has dirt to lift)
                return jax.lax.cond(
                    agent_is_loaded,
                    _lift_with_soil_mechanics,
                    _lift_without_soil_mechanics
                )
            
            def _lower_shovel():
                # Lower shovel (keeping current loading state)
                return self._replace(
                    agent=self.agent._replace(
                        agent_state=self.agent.agent_state._replace(
                            shovel_lifted=jnp.array([0], dtype=IntLowDim)  # Lower shovel
                        )
                    )
                )
            
            # Simple logic: if shovel is up, either dump (if loaded) or just lower it
            # If shovel is down, always lift it (toggle behavior)
            return jax.lax.cond(
                shovel_lifted,
                lambda: jax.lax.cond(is_loaded, _dump_and_lower, _lower_shovel),  # Shovel up: dump if loaded, else lower
                _lift_shovel  # Shovel down: always lift (toggle)
            )
        
        def _tracked_wheeled_do():
            is_loaded = self.agent.agent_state.loaded[0] > 0
            return jax.lax.cond(is_loaded, self._handle_dump, self._handle_dig)
        
        return jax.lax.cond(is_skid_steer, _skid_steer_do, _tracked_wheeled_do)

    @staticmethod
    def _check_agent_moved_on_move_action(
        old_state: "State", new_state: "State"
    ) -> bool:
        """True if agent moved"""
        return ~jnp.allclose(
            old_state.agent.agent_state.pos_base, new_state.agent.agent_state_2.pos_base
        )

    @staticmethod
    def _check_agent_turn_on_turn_action(
        old_state: "State", new_state: "State"
    ) -> bool:
        """True if agent turned"""
        return ~jnp.allclose(
            old_state.agent.agent_state.angle_base,
            new_state.agent.agent_state_2.angle_base,
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

        # Moving with turned wheels
        reward += jax.lax.cond(
            jnp.any(self.agent.agent_state.wheel_angle != 0),
            lambda: self.env_cfg.rewards.move_with_turned_wheels,
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

    def _handle_rewards_wheel_turn(
        self, new_state: "State", action: WheeledActionType
    ) -> Float:
        reward = 0.0

        # Check if wheels actually turned
        wheel_not_turned = jnp.allclose(
            self.agent.agent_state.wheel_angle,
            new_state.agent.agent_state_2.wheel_angle
        )

        # Apply extra reward if wheels did not turn
        reward += jax.lax.cond(
            wheel_not_turned,
            lambda: self.env_cfg.rewards.wheel_turn,
            lambda: 0.0,
        )

        reward += self.env_cfg.rewards.wheel_turn
        return reward

    def _handle_rewards_cabin_turn(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        return self.env_cfg.rewards.cabin_turn

    @staticmethod
    def _get_action_map_dump_progress(
        action_map_old: Array, action_map_new: Array, target_map: Array
    ) -> IntMap:
        """
        Returns
        > 0 if there was progress on the dump tiles after the action
        = 0 if there was no progress on the dump tiles after the action
        """
        action_map_clip_old = jnp.clip(action_map_old, a_min=0)
        action_map_clip_new = jnp.clip(action_map_new, a_min=0)

        target_map_dump_mask = target_map > 0

        action_map_progress = (
            (action_map_clip_new - action_map_clip_old) * target_map_dump_mask
        ).sum()

        return action_map_progress.astype(jnp.float32)

    @staticmethod
    def _get_action_map_dig_progress(
        action_map_old: Array, action_map_new: Array, target_map: Array
    ) -> IntMap:
        """
        Returns
        > 0 if there was progress on the dig tiles after the action
        = 0 if there was no progress on the dig tiles after the action
        """
        action_map_clip_old = jnp.clip(action_map_old, a_max=0)
        action_map_clip_new = jnp.clip(action_map_new, a_max=0)

        target_map_dump_mask = target_map < 0

        action_map_progress = (
            (action_map_clip_old - action_map_clip_new) * target_map_dump_mask
        ).sum()

        return action_map_progress.astype(jnp.float32)

    @staticmethod
    def _get_action_map_dump_regress(
        action_map_old: Array, action_map_new: Array, target_map: Array
    ) -> IntMap:
        """
        Returns
        > 0 if there was regress on the dig tiles after the action
        = 0 if there was no regress on the dig tiles after the action
        """
        action_map_clip_old = jnp.clip(action_map_old, a_min=0)
        action_map_clip_new = jnp.clip(action_map_new, a_min=0)

        target_map_dump_mask = target_map > 0

        action_map_regress = (
            (action_map_clip_old - action_map_clip_new) * target_map_dump_mask
        ).sum()

        return action_map_regress.astype(jnp.float32)

    def _handle_rewards_dump(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        """
        Handles reward assignment at dump time.
        This includes both the dump part and the realization
        of the previously digged terrain.
        """
        action_map_dump_progress = self._get_action_map_dump_progress(
            self.world.action_map.map,
            new_state.world.action_map.map,
            self.world.target_map.map,
        )

        dump_reward_condition = jnp.allclose(
            self.agent.agent_state.loaded, new_state.agent.agent_state_2.loaded
        )

        dump_reward = jax.lax.cond(
            dump_reward_condition,
            lambda: self.env_cfg.rewards.dump_wrong,
            lambda: action_map_dump_progress * self.env_cfg.rewards.dump_correct,
        )

        return dump_reward

    def _handle_rewards_dig(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        action_map_dig_progress = self._get_action_map_dig_progress(
            self.world.action_map.map,
            new_state.world.action_map.map,
            self.world.target_map.map,
        )

        action_map_dig_regress = self._get_action_map_dump_regress(
            self.world.action_map.map,
            new_state.world.action_map.map,
            self.world.target_map.map,
        )

        dig_reward = jax.lax.cond(
            action_map_dig_progress > 0,
            lambda: action_map_dig_progress * self.env_cfg.rewards.dig_correct,
            lambda: 0.0,
        )

        # Make penalty bigger than dumping reward
        dig_on_dump_penalty = jax.lax.cond(
            action_map_dig_regress > 0,
            lambda: -1.2 * action_map_dig_regress * self.env_cfg.rewards.dump_correct,
            lambda: 0.0,
        )

        dig_wrong_reward = jax.lax.cond(
            jnp.allclose(
                self.agent.agent_state.loaded, new_state.agent.agent_state.loaded
            ),
            lambda: self.env_cfg.rewards.dig_wrong,
            lambda: 0.0,
        )

        return dig_reward + dig_on_dump_penalty + dig_wrong_reward

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

    def _handle_rewards_skid_steer_auto_load(
        self, new_state: "State"
    ) -> Float:
        """Reward for successful auto-loading during movement"""
        # Check if dirt was auto-loaded during movement
        old_loaded = self.agent.agent_state.loaded[0]
        new_loaded = new_state.agent.agent_state.loaded[0]
        dirt_gained = new_loaded - old_loaded
        
        # Give reward proportional to dirt auto-loaded
        return jax.lax.cond(
            dirt_gained > 0,
            lambda: dirt_gained * self.env_cfg.rewards.skid_auto_load,
            lambda: 0.0
        )

    def _handle_rewards_skid_steer_dump(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        """Specialized dump rewards for skid steer"""
        # Check if dump was successful (dirt was unloaded)
        old_loaded = self.agent.agent_state.loaded[0]
        new_loaded = new_state.agent.agent_state.loaded[0]
        dirt_dumped = old_loaded - new_loaded
        
        def _successful_dump():
            # Check if dumped in correct areas (target map > 0)
            action_map_dump_progress = self._get_action_map_dump_progress(
                self.world.action_map.map,
                new_state.world.action_map.map,
                self.world.target_map.map,
            )
            
            # Reward based on correct placement
            return jax.lax.cond(
                action_map_dump_progress > 0,
                lambda: action_map_dump_progress * self.env_cfg.rewards.skid_dump_correct,
                lambda: dirt_dumped * self.env_cfg.rewards.skid_dump_wrong  # Dumped but not in correct area
            )
        
        def _failed_dump():
            # Tried to dump but failed (no dirt unloaded)
            return self.env_cfg.rewards.skid_dump_wrong
        
        return jax.lax.cond(
            dirt_dumped > 0,
            _successful_dump,
            _failed_dump
        )

    def _handle_rewards_skid_steer_lift(
        self, new_state: "State"
    ) -> Float:
        """Reward for successfully lifting dirt (not auto-loading)"""
        # Check if dirt was gained from manual lifting (DO action while shovel down)
        old_loaded = self.agent.agent_state.loaded[0]
        new_loaded = new_state.agent.agent_state.loaded[0]
        dirt_gained = new_loaded - old_loaded
        
        # Only reward if shovel was lifted during this action (indicating manual lift)
        old_shovel = self.agent.agent_state.shovel_lifted[0]
        new_shovel = new_state.agent.agent_state.shovel_lifted[0]
        shovel_was_lifted = new_shovel > old_shovel
        
        return jax.lax.cond(
            jnp.logical_and(dirt_gained > 0, shovel_was_lifted),
            lambda: dirt_gained * self.env_cfg.rewards.skid_lift_correct,
            lambda: 0.0
        )

    def _handle_rewards_skid_steer_shovel_control(
        self, new_state: "State"
    ) -> Float:
        """Small reward for effective shovel control"""
        old_shovel = self.agent.agent_state.shovel_lifted[0]
        new_shovel = new_state.agent.agent_state.shovel_lifted[0]
        shovel_changed = old_shovel != new_shovel
        
        # Small reward for shovel state changes (encourages learning control)
        return jax.lax.cond(
            shovel_changed,
            lambda: self.env_cfg.rewards.skid_shovel_control,
            lambda: 0.0
        )

    def _handle_rewards_skid_steer_do(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        """Specialized DO action rewards for skid steer"""
        reward = 0.0
        
        # Check what happened during DO action
        old_loaded = self.agent.agent_state.loaded[0]
        new_loaded = new_state.agent.agent_state.loaded[0]
        
        # Add dump rewards if dirt was unloaded
        reward += jax.lax.cond(
            old_loaded > new_loaded,
            lambda: self._handle_rewards_skid_steer_dump(new_state, action),
            lambda: 0.0
        )
        
        # Add lift rewards if dirt was gained (manual lifting)
        reward += self._handle_rewards_skid_steer_lift(new_state)
        
        # Add shovel control rewards
        reward += self._handle_rewards_skid_steer_shovel_control(new_state)
        
        return reward

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
            (action == WheeledActionType.WHEELS_LEFT)
            | (action == WheeledActionType.WHEELS_RIGHT),
            self._handle_rewards_wheel_turn,
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

    def _get_trench_specific_rewards(self) -> Float:
        def _get_trench_reward():
            agent_pos = self.agent.agent_state.pos_base
            trench_axes = self.world.trench_axes
            trench_type = self.world.trench_type

            # 1. Distance reward
            d_tiles = get_min_distance_point_to_lines(agent_pos, trench_axes, trench_type)
            d_tiles = jax.lax.cond(d_tiles > self.env_cfg.agent.width / 2, lambda: d_tiles, lambda: 0.0)
            d_meters = d_tiles * self.env_cfg.tile_size
            proximity_reward = d_meters * self.env_cfg.distance_coefficient

            # 2. Alignment reward
            def find_closest_trench_idx(i, state):
                dist, best_idx = state
                curr_dist = get_distance_point_to_line(agent_pos, trench_axes[i])
                new_best_idx = jax.lax.cond(curr_dist < dist, lambda: i, lambda: best_idx)
                new_dist = jnp.minimum(dist, curr_dist)
                return (new_dist, new_best_idx)

            _, closest_idx = jax.lax.fori_loop(
                0, trench_type,
                find_closest_trench_idx,
                (jnp.array(9999.0), jnp.array(0))
            )

            # Get trench line equation [a, b, c] for ax + by = c
            closest_trench = trench_axes[closest_idx]

            # Get trench direction vector [-b, a]
            trench_direction = jnp.array([-closest_trench[1], closest_trench[0]])
            trench_angle = jnp.arctan2(trench_direction[1], trench_direction[0])

            # Get agent angle in radians
            agent_angle = self._get_base_angle_rad()

            # Calculate angular difference (normalized between 0 and pi/2)
            angle_diff = jnp.abs(wrap_angle_rad(trench_angle - agent_angle))
            angle_diff = jnp.minimum(angle_diff, jnp.pi - angle_diff)
            angle_diff = jnp.minimum(angle_diff, jnp.pi / 2)

            # Calculate alignment score (0 = perfectly aligned, 1 = perpendicular)
            alignment_score = 2.0 * angle_diff / jnp.pi

            # Apply alignment reward - lower when aligned with trench
            alignment_reward = alignment_score * self.env_cfg.alignment_coefficient

            # Final calculation should be scalar + scalar
            total_reward = proximity_reward + alignment_reward
            # Explicitly ensure the final result is scalar before returning
            return jnp.squeeze(total_reward)

        r = jax.lax.cond(
            self.env_cfg.apply_trench_rewards,
            _get_trench_reward,
            lambda: 0.0,
        )
        return r

    def _get_reward(self, new_state: "State", action_handler: Action) -> Float:
        action = action_handler.action

        reward = 0.0

        # Action-dependent - route to appropriate reward function based on agent type
        current_agent_type = self.agent.agent_state.agent_type[0]
        
        def get_tracked_rewards():
            return self._get_rewards_tracked(new_state, action)
        
        def get_wheeled_rewards():
            return self._get_rewards_wheeled(new_state, action)
            
        def get_skidsteer_rewards():
            return self._get_rewards_skidsteer(new_state, action)
        
        # Route rewards based on agent type: 0=tracked, 1=wheeled, 2=skidsteer
        reward_functions = [get_tracked_rewards, get_wheeled_rewards, get_skidsteer_rewards]
        clamped_agent_type = jnp.clip(current_agent_type, 0, len(reward_functions) - 1)
        reward += jax.lax.switch(clamped_agent_type, reward_functions)

        # Terminal
        reward += jax.lax.cond(
            self._is_done_task(
                new_state.world.action_map.map,
                self.world.target_map.map,
                new_state.agent.agent_state_2.loaded,
                new_state.agent.agent_state.loaded,
            ),
            lambda: self.env_cfg.rewards.terminal,
            lambda: 0.0,
        )

        # Apply trench rewards
        reward += self._get_trench_specific_rewards()

        # Existence
        reward += self.env_cfg.rewards.existence

        # Constant scaling factor
        reward /= self.env_cfg.rewards.normalizer

        return reward

    @staticmethod
    def _is_done_task(action_map: Array, target_map: Array, agent_loaded: Array , agent_laoded_2: Array):
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
        done_unload2 = agent_laoded_2[0] == 0
        done_task = done_dump & done_dig & done_unload  & done_unload2
        return done_task

    def _is_done(
        self, action_map: Array, target_map: Array, agent_loaded: Array , agent_laoded_2: Array
    ) -> tuple[jnp.bool_, jnp.bool_]:
        done_task = self._is_done_task(action_map, target_map, agent_loaded, agent_laoded_2)
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

        # turn wheels left
        new_state = self._handle_turn_wheels_left()
        bool_turn_wheels_left = ~jnp.all(
            new_state.agent.agent_state.wheel_angle == self.agent.agent_state.wheel_angle
        )

        # turn wheels right
        new_state = self._handle_turn_wheels_right()
        bool_turn_wheels_right = ~jnp.all(
            new_state.agent.agent_state.wheel_angle == self.agent.agent_state.wheel_angle
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
                bool_turn_wheels_left,
                bool_turn_wheels_right,
                bool_cabin_clock,
                bool_cabin_anticlock,
                bool_do,
            ],
            dtype=jnp.bool_,
        )
        return action_mask

    def _get_action_mask_skidsteer(self):
        # Get the tracked action mask as base
        mask = self._get_action_mask_tracked()
        # Disable cabin actions for skid steer
        mask = mask.at[4].set(False)  # cabin_clock
        mask = mask.at[5].set(False)  # cabin_anticlock
        return mask

    def _get_action_mask(self, dummy_action: Action):
        """
        Returns a 1D array of bools, where 1 is allowed action, and 0 is not allowed.
        """
        num_actions = dummy_action.get_num_actions()
        
        # Check agent type from the current agent state
        current_agent_type = self.agent.agent_state.agent_type[0]
        
        # Use JAX switch instead of if statements for JIT compatibility
        def get_tracked_mask():
            return self._get_action_mask_tracked()
        
        def get_wheeled_mask():
            return self._get_action_mask_wheeled()
        
        def get_skidsteer_mask():
            return self._get_action_mask_skidsteer()
        
        # Create a list of functions for jax.lax.switch
        mask_functions = [get_tracked_mask, get_wheeled_mask, get_skidsteer_mask]
        
        # Use jax.lax.switch with clamped index to handle any agent_type value
        clamped_agent_type = jnp.clip(current_agent_type, 0, len(mask_functions) - 1)
        action_mask = jax.lax.switch(clamped_agent_type, mask_functions)
            
        action_mask = action_mask[:num_actions]
        return action_mask

    def _get_infos(self, dummy_action: Action, task_done: bool) -> dict[str, Any]:
        infos = {
            "action_mask": self._get_action_mask(dummy_action),
            "target_tiles": ~(~self._build_dig_dump_cone().reshape(-1)*~self._build_dig_dump_cone_2()),
            # Include termination_type directly without done_task
            "task_done": task_done,
        }
        return infos

    def _get_dig_dump_mask_cyl_skidsteer(self, map_cyl_coords: Array) -> Array:
        """
        Skid steer specific cylindrical workspace - similar size to excavator but slightly closer.
        Uses nearly identical parameters to excavator with minor adjustments.

        Args:
            - map_cyl_coords: (2, N) Array with [r, theta] rows
        Returns:
            - dig_mask: (N, ) Array of bools, where True means dig here
        """
        # Use same radius as excavator
        dig_portion_radius = self.env_cfg.agent.move_tiles  # Same as excavator
        tile_size = self.env_cfg.tile_size

        max_agent_dim = jnp.max(
            jnp.array([self.env_cfg.agent.width / 2, self.env_cfg.agent.height / 2])
        )
        min_distance_from_agent = tile_size * max_agent_dim

        # Slightly closer to agent than excavator but similar range
        fixed_extension = 0.2  # Slightly closer than excavator's 0.5  
        r_min = fixed_extension * dig_portion_radius * tile_size + min_distance_from_agent
        r_max = (fixed_extension + 0.8) * dig_portion_radius * tile_size + min_distance_from_agent  # Slimmer range than excavator's 1.0
        
        # Same angular range as excavator
        theta_max = 2 * np.pi / self.env_cfg.agent.angles_cabin  # Same as excavator
        theta_min = -theta_max

        dig_mask_r = jnp.logical_and(
            map_cyl_coords[0] >= r_min, map_cyl_coords[0] <= r_max
        )
        dig_mask_theta = jnp.logical_and(
            map_cyl_coords[1] >= theta_min, map_cyl_coords[1] <= theta_max
        )

        return jnp.logical_and(dig_mask_r, dig_mask_theta)

    def _get_dig_dump_mask_skidsteer(
        self, map_cyl_coords: Array, map_local_coords: Array
    ) -> Array:
        """
        Skid steer workspace using closer cylindrical coordinates with agent exclusion.

        Args:
            - map_cyl_coords: (2, N) Array with [r, theta] rows
            - map_local_coords: (2, N) Array with [x, y] rows
        Returns:
            - dig_mask: (N, ) Array of bools, where True means dig here
        """
        # Use skid steer specific cylindrical mask
        dig_dump_mask_cyl = self._get_dig_dump_mask_cyl_skidsteer(map_cyl_coords)

        # Apply same agent exclusion logic as excavator
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

    def _apply_local_soil_mechanics_simplified(self, action_map: Array, affected_mask: Array) -> Array:
        if action_map.ndim != 2:
            return action_map
        def collapse_body(map_2d, mask):
            mask = mask.astype(jnp.bool_)
            n_iters = 3  # Number of collapse iterations
            def collapse_step(i, map_2d):
                """One iteration of soil collapse - move dirt between neighbors."""
                # JAX-compatible defensive check: ensure map_2d is always treated as 2D
                # Use jax.lax.cond to handle potential dimension issues during tracing
                def handle_valid_2d(map_2d):
                    # Process the valid 2D map with soil mechanics
                    result = map_2d
                    # Check all 4 directional neighbors
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        # Get neighbor heights by shifting the map
                        shifted = jnp.roll(result, shift=(dy, dx), axis=(0, 1))
                        diff = shifted - result
                        neighbor_mask = jnp.roll(mask, shift=(dy, dx), axis=(0, 1))
                        move = (diff >= 2) & mask & neighbor_mask
                        result = result + move.astype(result.dtype)
                        result = jnp.roll(result, shift=(dy, dx), axis=(0, 1)) - move.astype(result.dtype)
                        result = jnp.roll(result, shift=(-dy, -dx), axis=(0, 1))
                    return result
                
                def handle_invalid_shape(map_2d):
                    # Return a safe default - this should never execute at runtime
                    # but is needed for JAX tracing completeness
                    return jnp.zeros_like(action_map, dtype=map_2d.dtype)
                
                # JAX-compatible shape check using jnp.where for static shape determination
                is_valid_2d = (jnp.ndim(map_2d) == 2) & (jnp.shape(map_2d)[0] > 0) & (jnp.shape(map_2d)[1] > 0)
                return jax.lax.cond(is_valid_2d, handle_valid_2d, handle_invalid_shape, map_2d)
            map_2d = jax.lax.fori_loop(0, n_iters, collapse_step, map_2d)
            return map_2d.astype(action_map.dtype)
        has_affected = jnp.any(affected_mask)
        def do_collapse(_):
            return collapse_body(action_map, affected_mask)
        return jax.lax.cond(has_affected, do_collapse, lambda _: action_map, operand=None)

    def _apply_local_soil_mechanics(self, action_map: Array, affected_mask: Array) -> Array:
        # Defensive: skip soil mechanics if not 2D
        if action_map.ndim != 2:
            return action_map
        return jax.lax.cond(
            ENABLE_SOIL_MECHANICS_IN_TRAINING,
            lambda: self._apply_local_soil_mechanics_simplified(action_map, affected_mask),
            lambda: action_map  # Return unchanged map when soil mechanics disabled
        )

    def _get_rewards_skidsteer(self, new_state: "State", action: ActionType) -> Float:
        """Specialized reward function for skid steer operations"""
        reward = 0.0
        action = action[0]

        # Movement rewards (same as tracked but with auto-loading bonus)
        movement_reward = jax.lax.cond(
            (action == TrackedActionType.FORWARD)
            | (action == TrackedActionType.BACKWARD),
            self._handle_rewards_move,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )
        reward += movement_reward
        
        # Auto-loading bonus for forward movement
        reward += jax.lax.cond(
            action == TrackedActionType.FORWARD,
            lambda: self._handle_rewards_skid_steer_auto_load(new_state),
            lambda: 0.0
        )

        # Base turn rewards (same as tracked)
        reward += jax.lax.cond(
            (action == TrackedActionType.CLOCK)
            | (action == TrackedActionType.ANTICLOCK),
            self._handle_rewards_base_turn,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )

        # Specialized DO action rewards for skid steer
        reward += jax.lax.cond(
            action == TrackedActionType.DO,
            self._handle_rewards_skid_steer_do,
            lambda new_state, action: 0.0,
            new_state,
            action,
        )
        
        return reward

