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
from terra.agent import AgentState
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

# Flag to enable soil mechanics
ENABLE_SOIL_MECHANICS = True
# If TRUE: Call _apply_dump() with use_condensed_dump=True to use a more concentrated dump that works better for soil mechanics


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

    # Note: Removed current_relocation_potential cache - always compute fresh for multi-agent reliability

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
        distance_map_override: Array | None = None,
    ) -> "State":
        # TEMP HACK: Set all dirt height 1 to 5 for testing
        #action_map = jnp.where(action_map == 1, 5, action_map)

        world = GridWorld.new(
            target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init, action_map,
            relocation_distance_map_override=distance_map_override,
        )

        # Get agent types from env_cfg, defaulting to (0, 2) for backwards compatibility
        agent_types = getattr(env_cfg, 'agent_types', (0, 2))
        agent, key = Agent.new(
            key, env_cfg, world.max_traversable_x, world.max_traversable_y, padding_mask, action_map,
            agent_types=agent_types
        )
        agent = jax.tree_map(
            lambda x: x if isinstance(x, Array) else jnp.array(x), agent
        )

        # Compute initial relocation potential using the cached distance map
        def _compute_initial_potential():
            # Guard: if distance map is all zeros, raise an error sentinel
            distance_map_all_zero = jnp.allclose(world.relocation_distance_map, 0.0)
            def _raise_distance_map_error():
                return jnp.array(-999999.0)
            def _compute_potential():
                return jnp.sum(
                    jnp.where(
                        world.target_map.map <= 0,
                        jnp.clip(world.action_map.map, a_min=0),
                        0,
                    ) * world.relocation_distance_map
                )
            return jax.lax.cond(distance_map_all_zero, _raise_distance_map_error, _compute_potential)
        def _zero_potential():
            return jnp.float32(0.0)
        initial_potential = jax.lax.cond(
            jnp.any(world.target_map.map > 0),
            _compute_initial_potential,
            _zero_potential,
        )

        # Initialize per-agent baselines/after-lift to initial_potential
        def _set_baselines(a: AgentState):
            return a._replace(
                carry_baseline_potential=jnp.float32(initial_potential),
                carry_potential_after_lift=jnp.float32(initial_potential),
            )
        agent = agent._replace(
            agent_states=tuple(_set_baselines(a) for a in agent.agent_states)
        )
        
        # Randomize starting agent uniformly among active agents to prevent first-mover advantage
        key, cat_key = jax.random.split(key)
        active_mask = agent.agent_active.astype(jnp.bool_)
        # Uniform logits over active indices; large negative for inactive to mask them out
        logits = jnp.where(active_mask, 0.0, -1e9)
        start_idx = jax.random.categorical(cat_key, logits).astype(jnp.int32)
        agent = agent._replace(current_agent=start_idx)

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
        distance_map_override: Array | None = None,
    ) -> "State":
        """
        Resets the already-existing State
        """
        key, _ = jax.random.split(self.key)
        # TEMP HACK: Set all dirt height 1 to 5 for testing
        #action_map = jnp.where(action_map == 1, 5, action_map)
        return self.new(
            key=key,
            env_cfg=env_cfg,
            target_map=target_map,
            padding_mask=padding_mask,
            trench_axes=trench_axes,
            trench_type=trench_type,
            dumpability_mask_init=dumpability_mask_init,
            action_map=action_map,
            distance_map_override=distance_map_override,
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
            self._do_nothing,
            # Wheeled
            self._handle_move_forward_wheeled,
            self._handle_move_backward_wheeled,
            self._handle_turn_wheels_left,
            self._handle_turn_wheels_right,
            self._handle_cabin_clock,
            self._handle_cabin_anticlock,
            self._handle_do,
            self._do_nothing,
        ]
        cumulative_len = jnp.array([0, 8], dtype=IntLowDim)
        offset_idx = (cumulative_len @ jax.nn.one_hot(action.type[0], 2)).astype(
            IntLowDim
        )

        state = jax.lax.cond(
            jnp.logical_or(action.action[0] == -1, action.action[0] == 7),
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
        """Advance to next active agent (defaults to 2 agents)."""
        def _next_agent_idx(current: int, num_agents: int, active_mask):
            # simple 2-agent fallback without scanning
            return (current + 1) % jnp.maximum(1, num_agents)

        next_idx = _next_agent_idx(self.agent.current_agent, self.agent.num_agents, self.agent.agent_active)
        # DEBUG: track agent turn cycling
        # Swap debug print silenced for cleaner training logs
        return self._replace(
            agent=self.agent._replace(
                current_agent=next_idx,
            )
        )

    # --- Helper accessors to migrate to array-only agent states ---
    def _get_current_agent_state(self):
        # Fixed cases for MAX_AGENTS=4
        return jax.lax.switch(
            self.agent.current_agent,
            [
                lambda: self.agent.agent_states[0],
                lambda: self.agent.agent_states[1],
                lambda: self.agent.agent_states[2],
                lambda: self.agent.agent_states[3],
            ]
        )

    def _get_next_agent_state(self):
        next_idx = (self.agent.current_agent + 1) % jnp.maximum(1, self.agent.num_agents)
        return jax.lax.switch(
            next_idx,
            [
                lambda: self.agent.agent_states[0],
                lambda: self.agent.agent_states[1],
                lambda: self.agent.agent_states[2],
                lambda: self.agent.agent_states[3],
            ]
        )

    def _get_prev_agent_state(self):
        prev_idx = (self.agent.current_agent + self.agent.num_agents - 1) % jnp.maximum(1, self.agent.num_agents)
        return jax.lax.switch(
            prev_idx,
            [
                lambda: self.agent.agent_states[0],
                lambda: self.agent.agent_states[1],
                lambda: self.agent.agent_states[2],
                lambda: self.agent.agent_states[3],
            ]
        )

    def _set_agent_state_at(self, idx: int, new_state):
        # Use jax.lax.switch to set the agent state at the given index
        def set_at_0():
            return self.agent._replace(agent_states=(
                new_state, self.agent.agent_states[1], self.agent.agent_states[2], self.agent.agent_states[3]
            ))
        def set_at_1():
            return self.agent._replace(agent_states=(
                self.agent.agent_states[0], new_state, self.agent.agent_states[2], self.agent.agent_states[3]
            ))
        def set_at_2():
            return self.agent._replace(agent_states=(
                self.agent.agent_states[0], self.agent.agent_states[1], new_state, self.agent.agent_states[3]
            ))
        def set_at_3():
            return self.agent._replace(agent_states=(
                self.agent.agent_states[0], self.agent.agent_states[1], self.agent.agent_states[2], new_state
            ))
        
        updated_agent = jax.lax.switch(
            idx,
            [set_at_0, set_at_1, set_at_2, set_at_3]
        )
        return self._replace(agent=updated_agent)

    def _current_idx(self):
        return self.agent.current_agent

    def _next_idx(self):
        return (self.agent.current_agent + 1) % jnp.maximum(1, self.agent.num_agents)

    def _prev_idx(self):
        return (self.agent.current_agent + self.agent.num_agents - 1) % jnp.maximum(1, self.agent.num_agents)

    def _set_current_agent_state(self, new_state):
        return self._set_agent_state_at(self._current_idx(), new_state)

    def _set_next_agent_state(self, new_state):
        return self._set_agent_state_at(self._next_idx(), new_state)

    def _set_prev_agent_state(self, new_state):
        return self._set_agent_state_at(self._prev_idx(), new_state)

    
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
        Efficient traversability mask with selective dirt collision.
        Small dirt patches are traversable, only very dense dirt formations block movement.
        
        Args:
            - map: (N, M) Array of ints (action_map)
            - padding_mask: (N, M) Array of ints, 1 if not traversable, 0 if traversable
        Returns:
            - traversability_mask: (N, M) Array of ints
                1 for non traversable, 0 for traversable
                
        Behavior:
            - High dirt piles (>1 height): Always blocked
            - Dug holes/trenches (negative values): Always blocked
            - 3x3 dirt patches: Mostly traversable (edges passable)
            - Large solid dirt formations: Blocked (8+ dirt tiles in 3x3 area)
            - Scattered dirt: Always traversable
            - Padding obstacles: Always blocked
        """
        # Fast path: if no dirt, just return padding mask
        has_dirt = jnp.any(map != 0)
        
        def _with_selective_dirt_collision():
            # Efficient direct neighbor counting (no convolution)
            dirt_mask = (map != 0).astype(jnp.int32)
            H, W = dirt_mask.shape
            
            # Pad with zeros to handle boundaries
            padded = jnp.pad(dirt_mask, 1, mode='constant', constant_values=0)
            
            # Count dirt in 3x3 neighborhood (8 neighbors + center = 9 total)
            dirt_count_3x3 = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +    # Top row
                padded[1:-1, :-2] + dirt_mask +        padded[1:-1, 2:] +    # Middle row (include center)
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]         # Bottom row
            )
            
            # Block only tiles in very dense dirt areas (8+ out of 9 tiles are dirt)
            # This allows 3x3 patches to be mostly traversable, blocks larger solid formations
            large_dirt_patches = jnp.logical_and(
                map != 0,  # Is dirt
                dirt_count_3x3 >= 6  # 8+ dirt tiles in 3x3 area (very dense)
            )
            
            # Also block high dirt piles (>1 dirt height) - always non-traversable
            high_dirt_piles = map > 1
            
            # Also block dug holes/trenches (negative values) - always non-traversable
            dug_holes = map < 0
            
            # Combine all conditions: dense areas OR high piles OR dug holes
            dirt_obstacles = jnp.logical_or(
                jnp.logical_or(large_dirt_patches, high_dirt_piles),
                dug_holes
            )
            
            # Combine: block padding obstacles OR dirt obstacles
            return jnp.logical_or(padding_mask == 1, dirt_obstacles).astype(IntLowDim)
        
        def _without_dirt():
            # No dirt present, just return padding mask
            return (padding_mask == 1).astype(IntLowDim)
        
        # Use JAX conditional to avoid unnecessary computation when no dirt is present
        return jax.lax.cond(has_dirt, _with_selective_dirt_collision, _without_dirt)

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

        # Build occupancy mask for all OTHER active agents (exclude current), or zero mask if single agent
        def _mask_for_agent_idx(i):
            st = self.agent.agent_states[i]
            corners_xy = self._get_agent_corners(
                st.pos_base,
                base_orientation=st.angle_base,
                agent_width=self.env_cfg.agent.width,
                agent_height=self.env_cfg.agent.height,
            )
            return compute_polygon_mask(corners_xy, map_width, map_height)

        def _zero_mask():
            return jnp.zeros((map_height, map_width), dtype=jnp.bool_)

        current_idx = self.agent.current_agent

        def _maybe_mask(i):
            include = jnp.logical_and(self.agent.agent_active[i] == 1, i != current_idx)
            return jax.lax.cond(include, lambda: _mask_for_agent_idx(i), _zero_mask)

        mask0 = _maybe_mask(0)
        mask1 = _maybe_mask(1)
        mask2 = _maybe_mask(2)
        mask3 = _maybe_mask(3)
        # Combine masks (int masks 0/1)
        polygon_mask_2 = jnp.maximum(jnp.maximum(mask0, mask1), jnp.maximum(mask2, mask3))

        
        # Build the traversability mask (0 = traversable, 1 = non-traversable).

        traversability_mask = self._build_traversability_mask(
            self.world.action_map.map, self.world.padding_mask.map
            
        )
        # Note: Interaction cones are not used for movement validity; collisions are handled via other agents' polygons
        # DIAG: disable other-agent blocking to test crowding hypothesis
        # traversability_mask = jnp.where(polygon_mask_2, 1, traversability_mask)

        
        #traversability_mask = traversability_mask
        traversability_mask = jnp.where(polygon_mask_2, 1, traversability_mask)
        # For a valid move, all cells covered by the agent must be traversable (== 0).
        # Mask out the cells where the agent is located.
        # jnp.where(polygon_mask_2, 1 ,traversability_mask)
        valid_traversability = jnp.all(jnp.where(polygon_mask, traversability_mask, 0) == 0)
        #jax.debug.print("Valid bounds: {valid_bounds}, Valid traversability: {valid_traversability}",valid_bounds=valid_bounds, valid_traversability=valid_traversability)
        return jnp.logical_and(valid_bounds, valid_traversability)

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
        cur = self._get_current_agent_state()
        candidate_pos = cur.pos_base + delta_xy
        candidate_pos = jnp.round(candidate_pos).astype(IntMap)  # Fix: use IntMap not IntLowDim
        candidate_pos = jnp.squeeze(candidate_pos, axis=0)

        # Compute the agent's corners based on the candidate (rounded) position.
        agent_corners_xy = self._get_agent_corners(
            candidate_pos,
            base_orientation=cur.angle_base,
            agent_width=self.env_cfg.agent.width,
            agent_height=self.env_cfg.agent.height,
        )

        # Check if the new position is valid.
        valid_move = self._is_valid_move(agent_corners_xy)
        valid_move_mask = self._valid_move_to_valid_mask(valid_move)

        # Choose between the old position and the new candidate position.
        old_new_pos = jnp.array([cur.pos_base, candidate_pos])
        new_pos_base = valid_move_mask @ old_new_pos
        return self._set_current_agent_state(cur._replace(pos_base=new_pos_base))

    def _move_on_orientation_with_steering(self, orientation_vector: Array, is_forward: jnp.bool_) -> "State":
        cur = self._get_current_agent_state()
        return jax.lax.cond(
            cur.wheel_angle[0] == 0,
            lambda: self._move_on_orientation(orientation_vector),
            lambda: self._execute_curved_movement(angle_idx_to_rad(cur.angle_base,
                                                                   self.env_cfg.agent.angles_base),
                                                                   is_forward),
        )

    def _execute_curved_movement(self, orientation_angle: float, is_forward: jnp.bool_) -> "State":
        # Shift to different orientation coordinates
        orientation_angle = orientation_angle + (jnp.pi / 2)

        # For backward movement, reverse the wheel angle effect
        cur = self._get_current_agent_state()
        wheel_angle = cur.wheel_angle[0]
        wheel_angle_rad = jnp.deg2rad(wheel_angle * self.env_cfg.agent.wheel_step)
        # Use width as wheelbase for turning radius calculation
        turn_radius = self.env_cfg.agent.width / (jnp.tan(wheel_angle_rad) + 1e-6)

        # Calculate center of rotation (perpendicular to current orientation)
        # Positive wheel angle means turn left, so center is to the left
        center_offset = np.squeeze(jnp.array([
            -jnp.sin(orientation_angle) * turn_radius,
            jnp.cos(orientation_angle) * turn_radius
        ]))
        center_of_rotation = cur.pos_base + center_offset

        # Compute how far we move along the arc and new orientation
        angle_change = self.env_cfg.agent.move_tiles / turn_radius
        angle_change = jnp.where(is_forward, angle_change, -angle_change)
        new_base_angle_rad = orientation_angle + angle_change

        # Rotate the digger around the center of rotation
        rotation_matrix = jnp.array([
            [jnp.cos(angle_change), -jnp.sin(angle_change)],
            [jnp.sin(angle_change), jnp.cos(angle_change)]
        ])
        relative_pos = cur.pos_base - center_of_rotation
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
        old_new_pos = jnp.array([cur.pos_base, candidate_pos])
        new_pos_base = valid_move_mask @ old_new_pos

        # Choose between old and new angles
        old_new_angle = jnp.array([cur.angle_base, new_angle_base])
        new_angle_base = valid_move_mask @ old_new_angle
        return self._set_current_agent_state(cur._replace(pos_base=new_pos_base, angle_base=new_angle_base))



    def _skid_steer_auto_load_dirt(self, new_state: "State") -> "State":
        """
        Auto-loading function for skid steer when moving with shovel lowered.
        Supports partial loading: loads up to capacity, removes exact amount from workspace.
        Applies soil mechanics directly when dirt is loaded.
        """
        # Only applies to skid steer with shovel lowered
        cur = new_state._get_current_agent_state()
        is_skid_steer = cur.agent_type[0] == 2
        shovel_lowered = cur.shovel_lifted[0] == 0
        
        # Fixed workspace capacity and current load
        workspace_capacity = jnp.int32(127)  # Fixed capacity within int8 range
        current_load = jnp.int32(cur.loaded[0])  # Convert to int32 (use current)
        
        should_auto_load = jnp.logical_and(is_skid_steer, shovel_lowered)
        
        def _apply_auto_load():
            
            # Use the closer cylindrical workspace for skid steer auto-loading
            map_cyl_coords, map_local_coords = new_state._get_map_local_and_cyl_coords()
            
            # Get the skid steer cylindrical workspace
            auto_load_mask = new_state._get_dig_dump_mask_skidsteer(map_cyl_coords, map_local_coords)
            
            # Apply skid steer dig masking (only allow loading from existing dirt)
            auto_load_mask = new_state._mask_out_wrong_dig_tiles_skidsteer(auto_load_mask)
            
            # Calculate how much dirt is available to load from workspace
            current_flattened_action_map = new_state.world.action_map.map.reshape(-1)
            # Convert to int32 to prevent overflow in dot product
            available_dirt = jnp.int32(current_flattened_action_map) @ jnp.int32(auto_load_mask)
            
            # Simple all-or-nothing loading: only load if entire workspace fits in capacity
            can_load_all = current_load + available_dirt <= workspace_capacity
            

            
            def _load_all_workspace():
                # Remove ALL dirt from workspace (simple and clean)
                new_flattened_action_map = jnp.where(
                    auto_load_mask,
                    0,  # Clear all dirt from workspace tiles
                    current_flattened_action_map  # Keep other tiles unchanged
                )
                
                new_map_2d = new_flattened_action_map.reshape(new_state.world.action_map.map.shape)
                
                # Apply soil mechanics (conserves dirt - only redistributes)
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
                    ENABLE_SOIL_MECHANICS,
                    _apply_soil_collapse,
                    _skip_soil_collapse
                )
                final_map = final_map.astype(new_state.world.action_map.map.dtype)
                
                # Load agent with the exact amount that was in the workspace
                # (soil mechanics conserves dirt, so this is perfectly conserving)
                new_loaded = current_load + available_dirt
                
                # Cache baseline when starting a carry (0 -> >0) - compute BEFORE removing dirt
                potential_before_load = self._compute_relocation_potential(self.world.action_map.map)
                started_loading = jnp.logical_and(current_load == 0, available_dirt > 0)
                
                # For subsequent loads: adjust baseline for world changes (like in dump rewards)
                def _adjust_baseline_for_world_changes():
                    # Current world potential before this auto-load
                    current_potential = potential_before_load
                    # Previous after-lift potential from last load
                    previous_after_lift = cur.carry_potential_after_lift
                    # Adjust baseline: if world got worse (higher potential), increase baseline
                    # If world got better (lower potential), decrease baseline (make it harder)
                    world_change = current_potential - previous_after_lift
                    return cur.carry_baseline_potential + world_change
                
                new_carry_base = jax.lax.select(
                    started_loading, 
                    potential_before_load,  # First load: use current potential as baseline
                    _adjust_baseline_for_world_changes()  # Subsequent loads: adjust for world changes
                )
                
                # Compute potential immediately after auto-load (post-removal map)
                after_lift_potential = self._compute_relocation_potential(final_map)
                

                
                new_cur = cur._replace(
                    loaded=jnp.array([new_loaded], dtype=cur.loaded.dtype),
                    carry_baseline_potential=jnp.float32(new_carry_base),
                    carry_potential_after_lift=jnp.float32(after_lift_potential),
                )
                return new_state._replace(
                    world=new_state.world._replace(
                        action_map=new_state.world.action_map._replace(map=final_map),
                    )
                )._set_current_agent_state(new_cur)
            
            def _no_load():
                # Can't fit entire workspace, so don't load anything
                return new_state
            
            # Load all workspace dirt if it fits, otherwise load nothing
            return jax.lax.cond(
                jnp.logical_and(available_dirt > 0, can_load_all),
                _load_all_workspace,
                _no_load
            )
        
        return jax.lax.cond(should_auto_load, _apply_auto_load, lambda: new_state)

    def _handle_move_forward(self) -> "State":
        """
        Moves the base forward with realistic restrictions:
        - Excavators/Wheeled: can only move when not loaded
        - Skid steer: can move when not loaded OR when loaded with shovel lifted OR when loaded with shovel down (push mode)
        - Skid steer push mode: can move forward while pushing dirt with lowered shovel
        """

        def _move_forward():
            cur = self._get_current_agent_state()
            base_orientation = cur.angle_base
            orientation_vector = self._base_orientation_to_one_hot_forward(
                base_orientation
            )
            new_state = self._move_on_orientation(orientation_vector)
            
            # Apply auto-loading for skid steer if moved successfully
            return self._skid_steer_auto_load_dirt(new_state)

        # Check agent conditions
        cur = self._get_current_agent_state()
        is_skid_steer = cur.agent_type[0] == 2
        is_truck = cur.agent_type[0] == 3
        is_loaded = cur.loaded[0] > 0
        
        # Movement rules:
        # - Non-skid steers/truck: allow movement when:
        #   * Skid steer or Truck: always can move
        #   * Others: only when not loaded
        can_move = jnp.logical_or(
            jnp.logical_or(is_skid_steer, is_truck),  # Skid steer or Truck - can always move
            jnp.logical_not(is_loaded)  # Others - only when not loaded
        )
        
        return jax.lax.cond(can_move, _move_forward, self._do_nothing)

    def _handle_move_backward(self) -> "State":
        """
        Moves the base backward with realistic restrictions:
        - Excavators/Wheeled: can only move when not loaded
        - Skid steer: can move when not loaded OR when loaded with shovel lifted OR when loaded with shovel down (drops dirt)
        - Skid steer with lowered shovel + loaded: moves backward and drops dirt (realistic behavior)
        - Skid steer with lowered shovel + loaded + no valid dump tiles: blocked from moving backward
        """

        def _move_backward():
            cur = self._get_current_agent_state()
            base_orientation = cur.angle_base
            orientation_vector = self._base_orientation_to_one_hot_backwards(
                base_orientation
            )
            
            # Check if skid steer should drop dirt when attempting to reverse
            cur2 = self._get_current_agent_state()
            is_skid_steer = cur2.agent_type[0] == 2
            is_loaded = cur2.loaded[0] > 0
            shovel_down = cur2.shovel_lifted[0] == 0
            
            # Check if backward movement would be possible (without actually moving yet)
            test_new_state = self._move_on_orientation(orientation_vector)
            movement_possible = ~jnp.allclose(
                cur2.pos_base,
                test_new_state._get_current_agent_state().pos_base,
                atol=1e-6
            )
            
            # Check if there are valid dump tiles under the agent (for skid steer)
            # Use the same logic as the dump function
            dump_mask = self._build_dig_dump_cone()
            # Only restrict dumping to dump zones for skid steer agents (commented out dump zone restriction)
            is_skid_steer = cur2.agent_type[0] == 2
            
            def _apply_dump_zone_restriction():
                # For skid steer: restrict to only dump zones (target_map > 0)
                dump_zone_mask = (self.world.target_map.map > 0).reshape(-1)
                return dump_mask * dump_zone_mask
            
            def _no_dump_zone_restriction():
                # For excavators: allow dumping on any valid tile (including neutral)
                return dump_mask
            
            dump_mask = jax.lax.cond(
                is_skid_steer,
                _apply_dump_zone_restriction,
                _no_dump_zone_restriction
            )
            
            # Apply the same exclude masks that are used in the dump function
            dump_mask = self._exclude_dig_tiles_from_dump_mask(dump_mask)
            dump_mask = self._exclude_dumpability_mask_tiles_from_dump_mask(dump_mask)
            dump_mask = self._exclude_traversability_mask_tiles_from_dump_mask(dump_mask)
            
            has_valid_dump_tiles = jnp.any(dump_mask)
            
            # Check if dump would be regressive (increase potential) - same logic as _handle_dump
            dump_volume = dump_mask.sum()
            
            # Skip potential check if no dump volume - can't increase potential if not dumping
            def _check_potential_increase():
                def _predict_potential():
                    remaining_volume = cur2.loaded % dump_volume
                    even_volume_per_tile = (
                        cur2.loaded - remaining_volume
                    ) / dump_volume
                    flattened_action_map = self.world.action_map.map.reshape(-1)
                    predicted_map_flat = self._apply_dump_mask(
                        flattened_action_map,
                        dump_mask,
                        even_volume_per_tile,
                        remaining_volume,
                        self.world.target_map.map,
                        use_condensed_dump=True,
                    )
                    predicted_map = predicted_map_flat.reshape(self.world.target_map.map.shape)
                    return self._compute_relocation_potential(predicted_map)
                
                predicted_potential = _predict_potential()
                
                # Use same potential gating logic as in _handle_dump
                current_potential = self._compute_relocation_potential(self.world.action_map.map)
                baseline_before = cur2.carry_baseline_potential
                after_lift = cur2.carry_potential_after_lift
                baseline_eff = baseline_before + (current_potential - after_lift)
                return predicted_potential > baseline_eff
            
            would_increase_potential = jax.lax.cond(
                dump_volume > 0,
                _check_potential_increase,
                lambda: jnp.bool_(False)  # No dump volume = can't increase potential
            )
            
            # Block movement if skid steer is loaded, shovel down, but no valid dump tiles OR dump would be regressive
            should_block_movement = jnp.logical_and(
                jnp.logical_and(is_skid_steer, is_loaded),
                jnp.logical_and(shovel_down, jnp.logical_or(jnp.logical_not(has_valid_dump_tiles), would_increase_potential))
            )
            
            # If movement should be blocked, return current state
            def _block_movement():
                return self
            
            def _allow_movement():
                # Drop dirt first if skid steer is loaded, shovel down, AND movement is possible
                # (realistic: blade lifts when starting to reverse, dirt falls off)
                should_drop_dirt = jnp.logical_and(
                    jnp.logical_and(is_skid_steer, movement_possible),  # Skid steer AND can move
                    jnp.logical_and(is_loaded, shovel_down)             # Loaded AND shovel down
                )
                
                # First dump dirt if needed
                state_after_dump = jax.lax.cond(
                    should_drop_dirt,
                    self._handle_dump,
                    lambda: self
                )
                
                # Then move backward
                new_state = state_after_dump._move_on_orientation(orientation_vector)
                
                return new_state
            
            return jax.lax.cond(should_block_movement, _block_movement, _allow_movement)

        # Check agent conditions
        cur = self._get_current_agent_state()
        is_skid_steer = cur.agent_type[0] == 2
        is_truck = cur.agent_type[0] == 3
        is_loaded = cur.loaded[0] > 0
        shovel_lifted = cur.shovel_lifted[0] > 0
        
        # Movement rules:
        # - Non-skid steers/truck: allow movement when:
        #   * Skid steer or Truck: can always move
        #   * Others: only when not loaded
        can_move = jnp.logical_or(
            jnp.logical_or(is_skid_steer, is_truck),
            jnp.logical_not(is_loaded)
        )
        
        return jax.lax.cond(can_move, _move_backward, self._do_nothing)

    def _handle_move_forward_wheeled(self) -> "State":
        """
        Moves the wheeled vehicle forward along an arc determined by wheel angle - if not loaded
        """
        def _move_forward_wheeled():
            cur = self._get_current_agent_state()
            base_orientation = cur.angle_base
            orientation_vector = self._base_orientation_to_one_hot_forward(base_orientation)
            return self._move_on_orientation_with_steering(orientation_vector, jnp.bool_(True))

        return jax.lax.cond(
            self._get_current_agent_state().loaded[0] > 0, self._do_nothing, _move_forward_wheeled
        )

    def _handle_move_backward_wheeled(self) -> "State":
        """
        Moves the wheeled vehicle backward along an arc determined by wheel angle - if not loaded
        """
        def _move_backward_wheeled():
            cur = self._get_current_agent_state()
            base_orientation = cur.angle_base
            orientation_vector = self._base_orientation_to_one_hot_backwards(base_orientation)
            return self._move_on_orientation_with_steering(orientation_vector, jnp.bool_(False))

        return jax.lax.cond(
            self._get_current_agent_state().loaded[0] > 0, self._do_nothing, _move_backward_wheeled
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
            self._get_current_agent_state().pos_base,
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
            cur = self._get_current_agent_state()
            old_angle_base = cur.angle_base
            new_angle_base = decrease_angle_circular(
                old_angle_base, self.env_cfg.agent.angles_base
            )
            new_angle_base = self._apply_base_rotation_mask(
                old_angle_base, new_angle_base
            )

            return self._set_current_agent_state(cur._replace(angle_base=new_angle_base))

        # Check agent conditions
        cur = self._get_current_agent_state()
        is_skid_steer = cur.agent_type[0] == 2
        is_truck = cur.agent_type[0] == 3
        is_loaded = cur.loaded[0] > 0
        
        # Rotation rules:
        # - Non-skid steers/truck:
        #   * Skid steer or Truck: can always rotate
        #   * Others: only when not loaded
        can_rotate = jnp.logical_or(
            jnp.logical_or(is_skid_steer, is_truck),
            jnp.logical_not(is_loaded)
        )
        
        return jax.lax.cond(can_rotate, _rotate_clock, self._do_nothing)

    def _handle_anticlock(self) -> "State":
        def _rotate_anticlock():
            cur = self._get_current_agent_state()
            old_angle_base = cur.angle_base
            new_angle_base = increase_angle_circular(
                old_angle_base, self.env_cfg.agent.angles_base
            )
            new_angle_base = self._apply_base_rotation_mask(
                old_angle_base, new_angle_base
            )

            return self._set_current_agent_state(cur._replace(angle_base=new_angle_base))

        # Check agent conditions
        cur = self._get_current_agent_state()
        is_skid_steer = cur.agent_type[0] == 2
        is_truck = cur.agent_type[0] == 3
        is_loaded = cur.loaded[0] > 0
        
        # Rotation rules:
        # - Non-skid steers/truck:
        #   * Skid steer or Truck: can always rotate
        #   * Others: only when not loaded
        can_rotate = jnp.logical_or(
            jnp.logical_or(is_skid_steer, is_truck),
            jnp.logical_not(is_loaded)
        )
        
        return jax.lax.cond(can_rotate, _rotate_anticlock, self._do_nothing)

    def _handle_cabin_clock(self) -> "State":
        """Handle cabin clockwise rotation. Does nothing for skid steer."""
        # Skid steer and Truck cannot rotate cabin
        cur = self._get_current_agent_state()
        is_skid_steer = cur.agent_type[0] == 2
        is_truck = cur.agent_type[0] == 3
        
        def _cabin_clock():
            cur2 = self._get_current_agent_state()
            old_angle_cabin = cur2.angle_cabin
            new_angle_cabin = decrease_angle_circular(
                old_angle_cabin, self.env_cfg.agent.angles_cabin
            )

            return self._set_current_agent_state(cur2._replace(angle_cabin=new_angle_cabin))
        
        return jax.lax.cond(jnp.logical_or(is_skid_steer, is_truck), self._do_nothing, _cabin_clock)

    def _handle_cabin_anticlock(self) -> "State":
        """Handle cabin anti-clockwise rotation. Does nothing for skid steer."""
        # Skid steer and Truck cannot rotate cabin
        cur = self._get_current_agent_state()
        is_skid_steer = cur.agent_type[0] == 2
        is_truck = cur.agent_type[0] == 3
        
        def _cabin_anticlock():
            cur2 = self._get_current_agent_state()
            old_angle_cabin = cur2.angle_cabin
            new_angle_cabin = increase_angle_circular(
                old_angle_cabin, self.env_cfg.agent.angles_cabin
            )

            return self._set_current_agent_state(cur2._replace(angle_cabin=new_angle_cabin))
        
        return jax.lax.cond(jnp.logical_or(is_skid_steer, is_truck), self._do_nothing, _cabin_anticlock)

    def _handle_turn_wheels_left(self) -> "State":
        cur = self._get_current_agent_state()
        old_wheel_angle = cur.wheel_angle
        new_wheel_angle = jnp.min(
            jnp.array([
                old_wheel_angle + 1,
                jnp.full((1,), fill_value=self.env_cfg.agent.max_wheel_angle, dtype=IntLowDim),
            ]),
            axis=0,
        )
        return self._set_current_agent_state(cur._replace(wheel_angle=new_wheel_angle))

    def _handle_turn_wheels_right(self) -> "State":
        cur = self._get_current_agent_state()
        old_wheel_angle = cur.wheel_angle
        new_wheel_angle = jnp.max(
            jnp.array([
                old_wheel_angle - 1,
                jnp.full((1,), fill_value=-self.env_cfg.agent.max_wheel_angle, dtype=IntLowDim),
            ]),
            axis=0,
        )
        return self._set_current_agent_state(cur._replace(wheel_angle=new_wheel_angle))

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
        cur = self._get_current_agent_state()
        return angle_idx_to_rad(cur.angle_cabin, self.env_cfg.agent.angles_cabin)

    def _get_base_angle_rad(self) -> Float:
        cur = self._get_current_agent_state()
        return angle_idx_to_rad(cur.angle_base, self.env_cfg.agent.angles_base)

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

        #r_min = min_distance_from_agent
        #r_max = min_distance_from_agent + dig_portion_radius * tile_size
        
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
            if we are instead digging dirt, then we dig as much as self.env_cfg.agent.dig_depth

        Args:
            - flattened_map: (N, ) Array flattened height map
            - dig_mask: (N, ) Array of where to dig bools
        Returns:
            - new_flattened_map: (N, ) Array flattened new height map
        """
        delta_dig = self.env_cfg.agent.dig_depth * dig_mask.astype(IntMap)
        new_flattened_map = jax.lax.cond(
            moving_dumped_dirt,
            lambda: jnp.where(dig_mask, 0, flattened_map).astype(IntMap),
            lambda: (flattened_map - delta_dig).astype(IntMap),
        )
        #Optionally apply soil mechanics using the global flag
        def apply_soil_mech():
            map_2d = new_flattened_map.reshape(self.world.action_map.map.shape)
            dig_mask_2d = dig_mask.reshape(self.world.action_map.map.shape)
            return self._apply_local_soil_mechanics_simplified(map_2d, dig_mask_2d).reshape(-1)
        return jax.lax.cond(
            ENABLE_SOIL_MECHANICS,
            apply_soil_mech,
            lambda: new_flattened_map
        )

    def _apply_dump_mask(
        self,
        flattened_map: Array,
        dump_mask: Array,
        even_volume_per_tile: IntLowDim,
        remaining_volume: IntLowDim,
        target_map: Array,
        use_condensed_dump: bool = False
    ) -> Array:
        """
        TODO: delta_dig_remaining now is added with a naive approach - should be added
            either to the closest tiles or randomly

        Args:
            - flattened_map: (N, ) Array flattened height map
            - dump_mask: (N, ) Array of where to dump bools
            - even_volume_per_tile: IntLowDim, volume to add to each of the tiles in the mask (per tile)
            - remaining_volume: IntLowDim, remaining volume to add to some of the tiles in the mask (total)
            - use_condensed_dump: If True, use concentrated dump with soil collapse; else use original logic.
        Returns:
            - new_flattened_map: (N, ) Array flattened new height map
        """
        map_2d_shape = self.world.action_map.map.shape
        dump_mask_2d = dump_mask.reshape(map_2d_shape)

        def _apply_simple_dump():
            # Original logic
            target_map_dump_mask = jnp.clip(target_map.reshape(-1), a_min=0) * dump_mask
            target_dump_volume = target_map_dump_mask.sum()
            dump_mask_final, dump_volume = jax.lax.cond(
                target_dump_volume > 0,
                lambda: (IntMap(target_map_dump_mask), target_dump_volume),
                lambda: (IntMap(dump_mask), dump_mask.sum()),
            )

            cur = self._get_current_agent_state()
            loaded_volume = cur.loaded
            remaining_volume_final = loaded_volume % dump_volume
            even_volume_per_tile_final = (loaded_volume - remaining_volume_final) / dump_volume

            delta_dig = self.env_cfg.agent.dig_depth * dump_mask_final * even_volume_per_tile_final
            delta_dig_remaining = jnp.zeros_like(delta_dig, dtype=IntMap)

            delta_dig_remaining = jnp.where(
                jnp.logical_and(jnp.cumsum(dump_mask_final) <= remaining_volume_final, dump_mask_final),
                1,
                delta_dig_remaining,
            )

            simple_result = (flattened_map + delta_dig + delta_dig_remaining).astype(IntMap)
            # Optionally apply soil mechanics using the global flag
            def apply_soil_mech():
                map_2d = simple_result.reshape(self.world.action_map.map.shape)
                mask_2d = dump_mask_final.reshape(self.world.action_map.map.shape)
                return self._apply_local_soil_mechanics_simplified(map_2d, self._expand_mask_for_soil_mechanics(mask_2d)).reshape(-1)
            return jax.lax.cond(
                ENABLE_SOIL_MECHANICS,
                apply_soil_mech,
                lambda: simple_result
            )

        def _apply_concentrated_dump():
            y_coords, x_coords = jnp.meshgrid(jnp.arange(map_2d_shape[0]), jnp.arange(map_2d_shape[1]), indexing='ij')
            centroid_y = jnp.sum(y_coords * dump_mask_2d) / jnp.maximum(jnp.sum(dump_mask_2d), 1)
            centroid_x = jnp.sum(x_coords * dump_mask_2d) / jnp.maximum(jnp.sum(dump_mask_2d), 1)
            distance_from_center = jnp.sqrt((y_coords - centroid_y)**2 + (x_coords - centroid_x)**2)
            concentrated_mask_2d = jnp.logical_and(
                dump_mask_2d,
                distance_from_center <= 2.0
            )
            # Fallback to closest tile if no tiles are within 2.0 units from the centroid
            def _fallback_to_closest_tile():
                distances_in_mask = jnp.where(dump_mask_2d, distance_from_center, jnp.inf)
                min_dist_idx_flat = jnp.argmin(distances_in_mask)
                y_closest, x_closest = jnp.unravel_index(min_dist_idx_flat, map_2d_shape)
                new_distance_from_center = jnp.sqrt((y_coords - y_closest)**2 + (x_coords - x_closest)**2)
                return jnp.logical_and(
                    dump_mask_2d,
                    new_distance_from_center <= 2.0
                )
            final_concentrated_mask_2d = jax.lax.cond(
                jnp.any(concentrated_mask_2d),
                lambda: concentrated_mask_2d,
                _fallback_to_closest_tile
            )
            total_volume_to_dump = even_volume_per_tile * jnp.sum(dump_mask) + remaining_volume
            concentrated_tiles_count = jnp.maximum(jnp.sum(final_concentrated_mask_2d), 1)
            even_volume_per_concentrated_tile = total_volume_to_dump // concentrated_tiles_count
            remaining_concentrated_volume = total_volume_to_dump % concentrated_tiles_count
            volume_per_tile_2d = (even_volume_per_concentrated_tile * final_concentrated_mask_2d).astype(IntMap)
            concentrated_mask_flat = final_concentrated_mask_2d.flatten()
            bonus_indices = jnp.where(
                concentrated_mask_flat, 
                jnp.cumsum(concentrated_mask_flat.astype(jnp.int32)), 
                concentrated_mask_flat.size + 1
            )
            bonus_mask_flat = bonus_indices <= remaining_concentrated_volume
            bonus_volume_2d = bonus_mask_flat.reshape(map_2d_shape).astype(IntMap)
            new_map_2d = flattened_map.reshape(map_2d_shape).astype(IntMap) + volume_per_tile_2d + bonus_volume_2d
            final_map_2d = jax.lax.cond(
                ENABLE_SOIL_MECHANICS,
                lambda: self._apply_local_soil_mechanics_simplified(new_map_2d, self._expand_mask_for_soil_mechanics(final_concentrated_mask_2d)),
                lambda: new_map_2d
            )
            return final_map_2d.flatten()

        return jax.lax.cond(
            use_condensed_dump,
            _apply_concentrated_dump,
            _apply_simple_dump
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
            ENABLE_SOIL_MECHANICS,
            lambda: self._apply_local_soil_mechanics_simplified(action_map, affected_mask),
            lambda: action_map  # Return unchanged map when soil mechanics disabled
        )

    # Removed unused _get_map_local_and_cyl_coords_2
    
    def _get_map_local_and_cyl_coords(self):
        """
        Returns:
            - map_cyl_coords: (2, width*height) map with [r, theta] rows
            - map_local_coords_base: (2, width*height) map with [x, y] rows
        """
        cur = self._get_current_agent_state()
        current_pos_idx = self._get_current_pos_vector_idx(
            pos_base=cur.pos_base,
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
        cur = self._get_current_agent_state()
        pos_base = cur.pos_base
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
        base_angle_rad = angle_idx_to_rad(cur.angle_base, self.env_cfg.agent.angles_base)
        base_angle_rad = jnp.squeeze(base_angle_rad)  # Ensure scalar
    
        cabin_angle_rad = angle_idx_to_rad(cur.angle_cabin, self.env_cfg.agent.angles_cabin)
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
        current_agent_type = self._get_current_agent_state().agent_type[0]
        
        # Get coordinates for cylindrical approach (used by both agent types)
        map_cyl_coords, map_local_coords_base = self._get_map_local_and_cyl_coords()
        
        def _get_excavator_cone():
            return self._get_dig_dump_mask(map_cyl_coords, map_local_coords_base)
        
        def _get_skidsteer_cone():
            return self._get_dig_dump_mask_skidsteer(map_cyl_coords, map_local_coords_base)
        
        # Use JAX conditional to select workspace type based on agent type
        return jax.lax.cond(
            jnp.logical_or(current_agent_type == 2, current_agent_type == 3),  # Skid steer or Truck
            _get_skidsteer_cone,
            _get_excavator_cone       # Default for excavator (0) and wheeled (1)
        )
    

    # Removed unused _build_dig_dump_cone_2

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
        cone_mask_2d = self._build_dig_dump_cone().reshape(self.world.action_map.map.shape)
        # Use boolean logic and any() to avoid large integer reductions on flattened arrays
        cond_has_overlap = jnp.any(
            self.world.last_dig_mask.map & (self.world.action_map.map > 0) & cone_mask_2d
        )
        dig_map_mask = jax.lax.cond(
            cond_has_overlap,
            lambda: (~cone_mask_2d).reshape(-1),
            lambda: jnp.ones_like(dump_mask, dtype=jnp.bool_),
        )

        return dump_mask * dig_map_mask

    def _mask_out_wrong_dig_tiles(self, dig_mask: Array) -> Array:
        """
        Takes the dig mask and turns into False the elements that do not correspond to
        a tile that has to be digged in the target map or that are dumped tiles in the action map.
        Also masks out the tiles that are digged as much as the target map requires.
        For excavators: also excludes the last dig area to prevent dump-load cycles.
        """
        dig_mask_target_map = self.world.target_map.map < 0
        dig_mask_action_map = self.world.action_map.map > 0
        dig_mask_maps = jnp.logical_or(dig_mask_target_map, dig_mask_action_map)

        flat_action_map = self.world.action_map.map.reshape(-1)
        dig_mask_cone = self._build_dig_dump_cone().reshape(self.world.action_map.map.shape)
        # Prefer boolean any over integer sum for compile-time efficiency
        has_dumped_dirt_in_cone = jnp.any((self.world.action_map.map > 0) & dig_mask_cone)
        ambiguity_mask_dig_movesoil = jax.lax.cond(
            has_dumped_dirt_in_cone,
            lambda: (flat_action_map > 0),
            lambda: (flat_action_map == 0),
        )

        # respect max dig limit
        max_dig_limit_mask = (
            self.world.action_map.map > -self.env_cfg.agent.dig_depth
        ).reshape(-1)

        # For excavators: exclude last dig area to prevent dump-load cycles
        is_excavator = self._get_current_agent_state().agent_type[0] != 2  # Not skidsteer
        
        def _exclude_last_dig_area():
            # Exclude the last dig/dump area from digging for excavators to prevent dump-load cycles
            return ~self.world.last_dig_mask.map.reshape(-1)
        
        def _no_dig_exclusion():
            # No exclusion for skidsteers
            return jnp.ones_like(dig_mask, dtype=jnp.bool_)
        
        dig_exclusion_mask = jax.lax.cond(
            is_excavator,
            _exclude_last_dig_area,
            _no_dig_exclusion
        )

        return (
            dig_mask
            * dig_mask_maps.reshape(-1)
            * ambiguity_mask_dig_movesoil
            * max_dig_limit_mask
            * dig_exclusion_mask
        ).astype(jnp.bool_)

    def _mask_out_wrong_dig_tiles_skidsteer(self, dig_mask: Array) -> Array:
        """
        For skid steer: Allow lifting from any dirt (action_map != 0)
        NEVER allow digging new holes (target_map < 0)
        Now allows loading from dump zones (target_map > 0) but with penalty.
        """
        # Allow lifting from any dirt (action_map != 0) - both natural and dumped
        dig_mask_action_map = self.world.action_map.map != 0
        
        # Respect max dig limit
        max_dig_limit_mask = (
            self.world.action_map.map > -self.env_cfg.agent.dig_depth
        ).reshape(-1)
        
        # Combine all masks (no dump zone exclusion)
        combined_mask = dig_mask & dig_mask_action_map.reshape(-1) & max_dig_limit_mask
        return combined_mask

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
                ENABLE_SOIL_MECHANICS,
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

            # Only update last_dig_mask for excavators, not skidsteers
            is_excavator = self._get_current_agent_state().agent_type[0] != 2  # Not skidsteer
            
            def _update_last_dig_mask():
                return self.world.last_dig_mask._replace(
                    map=jnp.bool_(dig_mask.reshape(self.world.action_map.map.shape))
                )
            
            def _keep_last_dig_mask():
                return self.world.last_dig_mask  # Keep existing mask for skidsteers (don't update it)
            
            new_last_dig_mask = jax.lax.cond(
                is_excavator,
                _update_last_dig_mask,
                _keep_last_dig_mask
            )
            
            			# Cache potentials for telescoping and add artificial source for newly dug dirt
            potential_before_dig = self._compute_relocation_potential(self.world.action_map.map)
            after_lift_potential = self._compute_relocation_potential(final_map)
            cur2 = self._get_current_agent_state()
            started_loading = jnp.logical_and(cur2.loaded[0] == 0, actual_volume_loaded > 0)
            # Artificial source potential for newly dug dirt (only when not moving dumped dirt)
            def _compute_artificial_source_potential():
                old_map_2d = self.world.action_map.map
                new_map_2d = final_map
                delta_removed = jnp.clip(old_map_2d - new_map_2d, a_min=0)
                non_dump_mask = (self.world.target_map.map <= 0)
                return jnp.sum(delta_removed * self.world.relocation_distance_map * non_dump_mask)
            artificial_source_potential = jax.lax.cond(
                jnp.logical_not(moving_dumped_dirt),
                _compute_artificial_source_potential,
                lambda: jnp.float32(0.0)
            )
            # If starting to load:
            # - existing dumped dirt: baseline = potential_before_dig
            # - newly dug dirt: baseline = potential_before_dig + artificial_source_potential
            def _baseline_when_starting():
                return jax.lax.select(
                    moving_dumped_dirt,
                    potential_before_dig,
                    potential_before_dig + artificial_source_potential
                )
            cur3 = self._get_current_agent_state()
            new_carry_base = jax.lax.select(
                started_loading,
                _baseline_when_starting(),
                cur3.carry_baseline_potential
            )
            updated_state = self._replace(
                world=self.world._replace(
                    action_map=self.world.action_map._replace(
                        map=IntLowDim(final_map)
                    ),
                    dumpability_mask=self.world.dumpability_mask._replace(
                        map=jnp.bool_(new_dumpability_mask),
                    ),
                    last_dig_mask=new_last_dig_mask,
                ),
                agent=self.agent._replace(
                    moving_dumped_dirt=jnp.bool_(moving_dumped_dirt),
                )
            )
            return updated_state._set_current_agent_state(
                cur3._replace(
                    loaded=jnp.array([actual_volume_loaded], dtype=IntLowDim),
                    carry_baseline_potential=jnp.float32(new_carry_base),
                    carry_potential_after_lift=jnp.float32(after_lift_potential),
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
    def _try_truck_transfer_on_excavator_dump(self) -> "State":
        """
        Attempt to transfer excavator load into any truck whose base center lies in the dump cone.
        Returns a potentially updated State; caller can compare loaded before/after to detect transfer.
        """
        # Use excavator's dump cone for truck transfer
        dump_mask = self._build_dig_dump_cone()
        curd = self._get_current_agent_state()
        is_excavator = (curd.agent_type[0] == 0)
        is_loaded = (curd.loaded[0] > 0)

        if not (is_excavator and is_loaded):
            return self

        map_shape = self.world.action_map.map.shape
        dump_mask_2d = dump_mask.reshape(map_shape)

        def base_in_cone_for(idx: int):
            st = self.agent.agent_states[idx]
            x = jnp.clip(st.pos_base[0].astype(jnp.int32), 0, self.world.width - 1)
            y = jnp.clip(st.pos_base[1].astype(jnp.int32), 0, self.world.height - 1)
            return dump_mask_2d[x, y]

        current_idx = self.agent.current_agent
        active = self.agent.agent_active.astype(jnp.bool_)
        types = jnp.array([
            self.agent.agent_states[0].agent_type[0],
            self.agent.agent_states[1].agent_type[0],
            self.agent.agent_states[2].agent_type[0],
            self.agent.agent_states[3].agent_type[0],
        ])
        base_in = jnp.array([
            base_in_cone_for(0),
            base_in_cone_for(1),
            base_in_cone_for(2),
            base_in_cone_for(3),
        ])
        not_current = jnp.array([
            0 != current_idx,
            1 != current_idx,
            2 != current_idx,
            3 != current_idx,
        ])
        is_truck_vec = (types == 3)
        candidates = jnp.logical_and(jnp.logical_and(active, is_truck_vec), jnp.logical_and(base_in, not_current))

        any_candidate = jnp.any(candidates)
        if not any_candidate:
            return self

        # Select first candidate index deterministically
        large = jnp.int32(1000000)
        idxs = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        scores = jnp.where(candidates, idxs, idxs + large)
        sel_idx = jnp.argmin(scores)
        capacity = jnp.int32(getattr(self.env_cfg, 'truck_capacity', 127))

        def _get_sel_state(i):
            return jax.lax.switch(i, [
                lambda: self.agent.agent_states[0],
                lambda: self.agent.agent_states[1],
                lambda: self.agent.agent_states[2],
                lambda: self.agent.agent_states[3],
            ])
        
        sel_state = _get_sel_state(sel_idx)
        truck_loaded = sel_state.loaded[0].astype(jnp.int32)
        cur_loaded = curd.loaded[0].astype(jnp.int32)
        remaining_cap = jnp.maximum(capacity - truck_loaded, 0)
        transfer = jnp.minimum(cur_loaded, remaining_cap)

        if transfer <= 0:
            return self

        # Apply transfer
        new_truck = sel_state._replace(
            loaded=jnp.array([truck_loaded + transfer], dtype=IntLowDim),
            carry_baseline_potential=jnp.float32(curd.carry_baseline_potential),
            carry_potential_after_lift=jnp.float32(curd.carry_potential_after_lift),
        )
        updated = self._set_agent_state_at(sel_idx, new_truck)
        cur_after = updated._get_current_agent_state()
        new_cur = cur_after._replace(
            loaded=jnp.array([jnp.maximum(cur_loaded - transfer, 0)], dtype=IntLowDim)
        )
        return updated._set_current_agent_state(new_cur)

    def _handle_dump(self) -> "State":
        dump_mask = self._build_dig_dump_cone()
        dump_mask = self._build_dig_dump_cone()
        # Only restrict dumping to dump zones for skid steer and truck agents
        cur = self._get_current_agent_state()
        is_skid_steer = cur.agent_type[0] == 2
        is_truck = cur.agent_type[0] == 3
        
        def _apply_dump_zone_restriction():
            # For skid steer: restrict to only dump zones (target_map > 0)
            dump_zone_mask = (self.world.target_map.map > 0).reshape(-1)
            return dump_mask * dump_zone_mask
        
        def _no_dump_zone_restriction():
            # For excavators: allow dumping on any valid tile (including neutral)
            return dump_mask
        
        dump_mask = jax.lax.cond(
            jnp.logical_or(is_skid_steer, is_truck),
            _apply_dump_zone_restriction,
            _no_dump_zone_restriction
        )
        dump_mask = self._exclude_dig_tiles_from_dump_mask(dump_mask)
        dump_mask = self._exclude_dumpability_mask_tiles_from_dump_mask(dump_mask)
        dump_mask = self._exclude_traversability_mask_tiles_from_dump_mask(dump_mask)
        dump_mask = self._exclude_just_moved_tiles_from_dump_mask(dump_mask)
        dump_volume = dump_mask.sum()

        def _apply_dump():
            # Calculate volume distribution only when we actually have a valid dump area
            curd = self._get_current_agent_state()

            # Try truck transfer outside via helper (called from DO). Here proceed with world dump only.
            remaining_volume = curd.loaded % dump_volume
            even_volume_per_tile = (
                curd.loaded - remaining_volume
            ) / dump_volume
            
            flattened_action_map = self.world.action_map.map.reshape(-1)
            new_map_global_coords = self._apply_dump_mask(
                flattened_action_map,
                dump_mask,
                even_volume_per_tile,
                remaining_volume,
                self.world.target_map.map,
                use_condensed_dump=True
            )
            new_map_global_coords = new_map_global_coords.reshape(
                self.world.target_map.map.shape
            )

            			# Reset last_dig_mask after a successful dump like single-agent (allow re-lift next time)
            is_excavator = self._get_current_agent_state().agent_type[0] != 2  # Not skidsteer
        
            def _clear_last_dig_mask():
                return self.world.last_dig_mask._replace(
                    map=jnp.zeros_like(self.world.last_dig_mask.map, dtype=jnp.bool_)
                )
            
            def _keep_last_dig_mask():
                return self.world.last_dig_mask  # Skidsteer: leave unchanged
            
            new_last_dig_mask = jax.lax.cond(
                is_excavator,
                _clear_last_dig_mask,
                _keep_last_dig_mask
            )
            current_potential = self._compute_relocation_potential(self.world.action_map.map)            
            # Predict potential post-dump and gate regressive dumps relative to effective baseline
            predicted_potential = self._compute_relocation_potential(new_map_global_coords)
            curb = self._get_current_agent_state()
            baseline_before = curb.carry_baseline_potential
            after_lift = curb.carry_potential_after_lift
            baseline_eff = baseline_before + (current_potential - after_lift)

            would_increase_potential = predicted_potential > baseline_eff

            # jax.debug.print("[DEBUG] Baseline Caching:")
            # jax.debug.print("  potential (before lift): {}", baseline_before)
            # jax.debug.print("  potential (after lift): {}", after_lift)
            # jax.debug.print("  potential (current): {}", current_potential)
            # jax.debug.print("  potential (new): {}", predicted_potential)
            # jax.debug.print("  baseline effective: {}", baseline_eff)
            # jax.debug.print("  would increase potential: {}", would_increase_potential)


            def _prevent_regressive_dump():
                return self
            def _allow_progressive_dump():
                # Update world and current potential
                #new_rel_pot = self._compute_relocation_potential(new_map_global_coords)
                updated_state = self._replace(
                    world=self.world._replace(
                        action_map=self.world.action_map._replace(
                            map=IntLowDim(new_map_global_coords)
                        ),
                        last_dig_mask=new_last_dig_mask,
                    ),
                    agent=self.agent._replace(
                        moving_dumped_dirt=jnp.bool_(False),
                    ),
                )
                return updated_state._set_current_agent_state(
                    updated_state._get_current_agent_state()._replace(
                        loaded=jnp.full((1,), fill_value=0, dtype=IntLowDim)
                    )
                )
            return jax.lax.cond(would_increase_potential, _prevent_regressive_dump, _allow_progressive_dump)

        return jax.lax.cond(dump_volume > 0, _apply_dump, self._do_nothing)





    def _handle_lift_dumped_dirt(self) -> "State":
        """
        For skid steer: Lifting the shovel only toggles the shovel state to up.
        Does NOT move dirt or change loaded. All loading is handled in auto-load.
        """
        cur = self._get_current_agent_state()
        new_cur = cur._replace(shovel_lifted=jnp.array([1], dtype=IntLowDim))
        return self._set_current_agent_state(new_cur)

    def _handle_do(self) -> "State":
        """
        Handle the DO action based on agent type:
        - Tracked/Wheeled (0,1): dig (not loaded) / dump (loaded)
        - Skid steer (2): simple shovel control:
          * If shovel lifted + loaded: dump dirt and lower shovel
          * If shovel lifted + not loaded: lower shovel  
          * If shovel lowered: lift shovel (regardless of loading)
        """
        cur = self._get_current_agent_state()
        is_skid_steer = cur.agent_type[0] == 2
        is_truck = cur.agent_type[0] == 3
        
        def _skid_steer_do():
            cur = self._get_current_agent_state()
            is_loaded = cur.loaded[0] > 0
            shovel_lifted = cur.shovel_lifted[0] > 0
            
            def _dump_and_lower():
                # Always try to dump dirt and then lower shovel regardless of dump success
                dumped_state = self._handle_dump()
                # Always lower the shovel after dump attempt
                next_cur = dumped_state._get_current_agent_state()
                return dumped_state._set_current_agent_state(
                    next_cur._replace(shovel_lifted=jnp.array([0], dtype=IntLowDim))
                )
            
            def _lift_shovel():
                # Check if agent is loaded
                cur3 = self._get_current_agent_state()
                agent_is_loaded = cur3.loaded[0] > 0
                
                def _lift_with_soil_mechanics():
                    # Agent has dirt, so lifting will disturb surrounding soil
                    # First lift the dirt with soil mechanics (already applied in _handle_lift_dumped_dirt)
                    lifted_state = self._handle_lift_dumped_dirt()
                    
                    # Just lift the shovel (soil mechanics already applied)
                    lc = lifted_state._get_current_agent_state()
                    return lifted_state._set_current_agent_state(
                        lc._replace(shovel_lifted=jnp.array([1], dtype=IntLowDim))
                    )
                
                def _lift_without_soil_mechanics():
                    # Agent has no dirt, just lift shovel without terrain disturbance
                    cur4 = self._get_current_agent_state()
                    return self._set_current_agent_state(
                        cur4._replace(shovel_lifted=jnp.array([1], dtype=IntLowDim))
                    )
                
                # Apply soil mechanics only if agent is loaded (has dirt to lift)
                return jax.lax.cond(
                    agent_is_loaded,
                    _lift_with_soil_mechanics,
                    _lift_without_soil_mechanics
                )
            
            def _lower_shovel():
                # Lower shovel (keeping current loading state)
                cur5 = self._get_current_agent_state()
                return self._set_current_agent_state(
                    cur5._replace(shovel_lifted=jnp.array([0], dtype=IntLowDim))
                )
            
            # Simple logic: if shovel is up, either dump (if loaded) or just lower it
            # If shovel is down, always lift it (toggle behavior)
            return jax.lax.cond(
                shovel_lifted,
                lambda: jax.lax.cond(is_loaded, _dump_and_lower, _lower_shovel),  # Shovel up: dump if loaded, else lower
                _lift_shovel  # Shovel down: always lift (toggle)
            )
        
        def _tracked_wheeled_do():
            cur = self._get_current_agent_state()
            is_loaded = cur.loaded[0] > 0
            # For trucks: DO dumps when loaded, no-op when empty (no digging)
            def _truck_do():
                return jax.lax.cond(is_loaded, self._handle_dump, self._do_nothing)
            def _excavator_do():
                # Try truck transfer; if it transfers, return that state; else proceed to dump/dig
                before = self
                after = before._try_truck_transfer_on_excavator_dump()
                cur_before = before._get_current_agent_state()
                cur_after = after._get_current_agent_state()
                did_transfer = cur_after.loaded[0] < cur_before.loaded[0]
                def _return_after_transfer():
                    return after
                def _fallback_dump_or_dig():
                    return jax.lax.cond(is_loaded, self._handle_dump, self._handle_dig)
                return jax.lax.cond(did_transfer, _return_after_transfer, _fallback_dump_or_dig)
            return jax.lax.cond(is_truck, _truck_do, _excavator_do)
        
        return jax.lax.cond(is_skid_steer, _skid_steer_do, _tracked_wheeled_do)

    @staticmethod
    def _check_agent_moved_on_move_action(
        old_state: "State", new_state: "State"
    ) -> bool:
        """True if agent moved"""
        return ~jnp.allclose(
            old_state._get_current_agent_state().pos_base,
            new_state._get_next_agent_state().pos_base,
        )

    @staticmethod
    def _check_agent_turn_on_turn_action(
        old_state: "State", new_state: "State"
    ) -> bool:
        """True if agent turned"""
        return ~jnp.allclose(
            old_state._get_current_agent_state().angle_base,
            new_state._get_next_agent_state().angle_base,
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
        cur = self._get_current_agent_state()
        reward += jax.lax.cond(
            jnp.all(cur.loaded > 0),
            lambda: self.env_cfg.rewards.move_while_loaded,
            lambda: 0.0,
        )

        # Moving with turned wheels
        reward += jax.lax.cond(
            jnp.any(cur.wheel_angle != 0),
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
            self._get_current_agent_state().wheel_angle,
            new_state._get_next_agent_state().wheel_angle,
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
        # Check if agent is excavator (not skid steer)
        cur = self._get_current_agent_state()
        is_excavator = cur.agent_type[0] == 0
        is_skidsteer = cur.agent_type[0] == 2
        
        
        # Telescoping relocation reward: use progress since lift with effective baseline
        baseline_before = cur.carry_baseline_potential
        after_lift = cur.carry_potential_after_lift
        current_potential = self._compute_relocation_potential(self.world.action_map.map)
        new_potential = self._compute_relocation_potential(new_state.world.action_map.map)
        # Use same effective baseline logic as in dump gating
        baseline_eff = baseline_before + (current_potential - after_lift)
        effective_progress = (baseline_eff - new_potential)
        cap = jnp.float32(200.0)
        progress_clamped = jnp.clip(effective_progress, -cap, cap)
        # Dump success/fail
        dump_failed = jnp.allclose(
            cur.loaded, new_state._get_prev_agent_state().loaded
        )

        
        # jax.debug.print("[DEBUG] Potential Rewards:")
        # jax.debug.print("  potential (before lift): {}", baseline_before)
        # jax.debug.print("  potential (after lift): {}", after_lift)
        # jax.debug.print("  potential (new after dump): {}", new_potential)
        # jax.debug.print("  effective progress: {}", effective_progress)
        # jax.debug.print("  progress clamped: {}", progress_clamped) 

        is_moving_dumped_dirt = self.agent.moving_dumped_dirt
        potential_multiplier = jax.lax.cond(
            is_skidsteer,
            lambda: 1.0,  # Skidsteer gets full reward for all dirt
            lambda: jax.lax.cond(
                is_moving_dumped_dirt,
                lambda: 0.1,  # Excavator gets 0.1x reward for relocating dumped dirt
                lambda: 3.0   # Excavator gets full reward for relocating newly dug dirt
            )
        )
        # Per-map normalization: factor=1 when target tiles ~= 173; >1 for smaller maps
        avg_target_tiles = jnp.float32(170.0)
        # Use only dig target tiles (foundations): target_map < 0
        dig_target_tiles = jnp.sum(self.world.target_map.map < 0)
        scale_raw = avg_target_tiles / jnp.maximum(jnp.float32(1.0), dig_target_tiles.astype(jnp.float32))
        scale = jnp.clip(scale_raw, jnp.float32(2.0), jnp.float32(5.0)) / 2


        progress_clamped = progress_clamped * potential_multiplier * scale
        #progress_clamped = progress_clamped * potential_multiplier * 1
        #progress_clamped = progress_clamped * potential_multiplier

        def _success_reward():
            dump_progress = self._get_action_map_dump_progress(
                self.world.action_map.map,
                new_state.world.action_map.map,
                self.world.target_map.map,
            )
            # Dump bonus only for skidsteers (relocation specialists)
            
            dump_bonus = jax.lax.cond(
                is_skidsteer,
                lambda: jnp.maximum(dump_progress, 0.0) * self.env_cfg.rewards.dump_correct * 0.5,   # Strong relocation specialization
                lambda: jnp.maximum(dump_progress, 0.0) * self.env_cfg.rewards.dump_correct * 0.5 * potential_multiplier # Small efficiency bonus for direct dumps
            )
            meaningful_threshold = jnp.float32(0.1)
            
            return jax.lax.cond(
                progress_clamped > meaningful_threshold,
                lambda: (progress_clamped * self.env_cfg.rewards.dump_correct + dump_bonus) - jnp.float32(1.0),
                lambda: -jnp.float32(1.0),
            )
        def _failed_dump():
            return self.env_cfg.rewards.dump_wrong


        return jax.lax.cond(dump_failed, _failed_dump, _success_reward)
        

    def _handle_rewards_dig(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        # Unified dig rewards for all agent types - both excavators and skidsteers use same logic
        cur = self._get_current_agent_state()
        prev_new = new_state._get_prev_agent_state()
        started_loading = jnp.logical_and(
            cur.loaded[0] == 0,
            prev_new.loaded[0] > 0,
        )
        dig_reward = jax.lax.cond(
            started_loading,
            lambda: jnp.float32(1.0),
            lambda: jnp.float32(0.0),
        )
        
        # Penalty for digging on dump zones (regress on positive target)
        action_map_dig_regress = self._get_action_map_dump_regress(
            self.world.action_map.map,
            new_state.world.action_map.map,
            self.world.target_map.map,
        )
        dig_on_dump_penalty = jax.lax.cond(
            action_map_dig_regress > 0,
            lambda: -1.2 * action_map_dig_regress * self.env_cfg.rewards.dump_correct,
            lambda: 0.0,
        )
        
        # Wrong dig when loaded (no dirt loaded despite dig action)
        dig_wrong_reward = jax.lax.cond(
            jnp.allclose(
                cur.loaded, prev_new.loaded
            ),
            lambda: self.env_cfg.rewards.dig_wrong,
            lambda: 0.0,
        )
        
        return dig_reward + dig_on_dump_penalty + dig_wrong_reward

    def _handle_rewards_do(
        self, new_state: "State", action: TrackedActionType
    ) -> tuple[Float, dict]:
        def _dump():
            total = self._handle_rewards_dump(new_state, action)
            return total, {"dump": total, "lift": 0.0}
        def _dig():
            total = self._handle_rewards_dig(new_state, action)
            return total, {"dump": 0.0, "lift": total}
        return jax.lax.cond(
            jnp.all(self._get_current_agent_state().loaded > 0),
            _dump,
            _dig,
        )

    def _handle_rewards_skid_steer_auto_load(
        self, new_state: "State"
    ) -> Float:
        """Reward for successful auto-loading during movement, with penalty if dirt is removed from a dump zone."""
        cur = self._get_current_agent_state()
        prev_new = new_state._get_prev_agent_state()
        old_loaded = cur.loaded[0]
        new_loaded = prev_new.loaded[0]
        dirt_gained = new_loaded - old_loaded

        # Only check for penalty if dirt was gained
        # def _reward_or_penalty():
        #     # Use the existing progress function to check if dump zone dirt decreased
        #     progress = self._get_action_map_dump_progress(
        #         self.world.action_map.map,
        #         new_state.world.action_map.map,
        #         self.world.target_map.map,
        #     )
        #     # If progress is negative, dirt was removed from a dump zone
        #     # return jax.lax.cond(
        #     #     progress < 0,
        #     #     lambda: self.env_cfg.rewards.skid_auto_load_from_dumpzone_penalty,
        #     #     lambda: self.env_cfg.rewards.skid_auto_load,
        #     # )

        #     return jax.lax.cond(
        #         progress < 0,
        #         lambda: self._calculate_dump_zone_reward(progress, cur.loaded[0]), #- self.env_cfg.rewards.skid_auto_load,  # Use unified function
        #         lambda: self.env_cfg.rewards.skid_auto_load,
        #     )


            # Penalty commented out:
            #return self.env_cfg.rewards.skid_auto_load

        # Route successful auto-load to dig rewards (single-agent parity)
        return jax.lax.cond(
            dirt_gained > 0,
            lambda: self._handle_rewards_dig(new_state, TrackedActionType.FORWARD),
            lambda: 0.0
        )

    def _handle_rewards_skid_steer_dump(
        self, new_state: "State", action: TrackedActionType
    ) -> Float:
        """Specialized dump rewards for skid steer with efficiency-based rewards"""
        # Check if dump was successful (dirt was unloaded)
        cur = self._get_current_agent_state()
        prev_new = new_state._get_prev_agent_state()
        old_loaded = cur.loaded[0]
        new_loaded = prev_new.loaded[0]
        dirt_dumped = old_loaded - new_loaded
        
        # def _successful_dump():
        #     # Check if dumped in correct areas (target map > 0)
        #     action_map_dump_progress = self._get_action_map_dump_progress(
        #         self.world.action_map.map,
        #         new_state.world.action_map.map,
        #         self.world.target_map.map,
        #     )
            
        #     # EFFICIENCY FIX: Calculate efficiency as ratio of correct placement
        #     # This eliminates the edge-dumping exploit by rewarding based on percentage
        #     def _calculate_efficiency_reward():
        #         # Give reward based on absolute amount of dirt moved, not percentage
                
                
        #         # min_reward = 15.0
        #         # max_reward = self.env_cfg.rewards.skid_dump_correct
                
        #         # # Scale reward based on absolute dirt amount moved
        #         # # Small dumps (1-5 units) get min_reward, large dumps (20+ units) get max_reward
        #         # min_dirt = 1.0
        #         # max_dirt = 20.0
                
        #         # # Clamp dirt amount to reasonable range
        #         # clamped_dirt = jnp.clip(action_map_dump_progress, min_dirt, max_dirt)
                
        #         # # Linear interpolation between min and max reward based on dirt amount
        #         # reward_ratio = (clamped_dirt - min_dirt) / (max_dirt - min_dirt + 1e-8)
        #         # scaled_reward = min_reward + reward_ratio * (max_reward - min_reward)
                
        #         # return scaled_reward
        #         return self._calculate_dump_zone_reward(action_map_dump_progress, cur.loaded[0])
            
            
        #     def _wrong_dump_penalty():
        #         # If no progress in dump zones, give penalty proportional to amount dumped
        #         return self.env_cfg.rewards.skid_dump_wrong
            
        #     # Reward based on correct placement efficiency
        #     return jax.lax.cond(
        #         action_map_dump_progress > 0,
        #         _calculate_efficiency_reward,
        #         _wrong_dump_penalty
        #     )
        
        def _failed_dump():
            # Tried to dump but failed (no dirt unloaded)
            return self.env_cfg.rewards.skid_dump_wrong
        
        return jax.lax.cond(
            dirt_dumped > 0,
            lambda: self._handle_rewards_dump(new_state, action),
            _failed_dump
        )

    def _handle_rewards_skid_steer_shovel_control(
        self, new_state: "State"
    ) -> Float:
        """Small reward for effective shovel control"""
        cur = self._get_current_agent_state()
        prev_new = new_state._get_prev_agent_state()
        old_shovel = cur.shovel_lifted[0]
        new_shovel = prev_new.shovel_lifted[0]
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
        cur = self._get_current_agent_state()
        prev_new = new_state._get_prev_agent_state()
        old_loaded = cur.loaded[0]
        new_loaded = prev_new.loaded[0]
        
        # Add dump rewards if dirt was unloaded
        reward += jax.lax.cond(
            old_loaded > new_loaded,
            lambda: self._handle_rewards_dump(new_state, action),
            lambda: 0.0
        )
        
       
        
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
            lambda new_state, action: self._handle_rewards_do(new_state, action)[0],  # Just take total
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
            lambda new_state, action: self._handle_rewards_do(new_state, action)[0],  # Just take total
            lambda new_state, action: 0.0,
            new_state,
            action,
        )
        return reward

    def _get_trench_specific_rewards(self) -> Float:
        def _get_trench_reward():
            cur = self._get_current_agent_state()
            agent_pos = cur.pos_base
            trench_axes = self.world.trench_axes
            trench_type = self.world.trench_type

            # Find the closest trench for both distance and alignment calculations
            def find_closest_trench_idx(i, state):
                dist, best_idx = state
                curr_dist = get_distance_point_to_line(agent_pos, trench_axes[i])
                new_best_idx = jax.lax.cond(curr_dist < dist, lambda: i, lambda: best_idx)
                new_dist = jnp.minimum(dist, curr_dist)
                return (new_dist, new_best_idx)

            d_tiles, closest_idx = jax.lax.fori_loop(
                0, trench_type,
                find_closest_trench_idx,
                (jnp.array(9999.0, dtype=jnp.float32), jnp.array(0))
            )

            # 1. Distance reward - only for closest trench
            d_tiles = jax.lax.cond(d_tiles > self.env_cfg.agent.width / 2, lambda: d_tiles, lambda: jnp.array(0.0, dtype=jnp.float32))
            d_meters = d_tiles * self.env_cfg.tile_size
            
            proximity_reward = d_meters * self.env_cfg.distance_coefficient

            # 2. Alignment reward - only for closest trench
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

        # Calculate trench reward for the current agent
        # Agent type check is now handled in the main reward function
        return _get_trench_reward()

    def _get_reward(self, new_state: "State", action_handler: Action):
        action = action_handler.action

        reward = 0.0
        # Components for W&B logging - generalized to MAX_AGENTS per-agent rewards
        MAX_AGENTS = 4
        components = {
            "agent_rewards": jnp.zeros((MAX_AGENTS,), dtype=jnp.float32),
            "terminal": 0.0,
            "trench": 0.0,
            "existence": 0.0,
            "num_agents": self.agent.num_agents.astype(jnp.int32),
            "agent_active": self.agent.agent_active.astype(jnp.int32),
        }

        # Action-dependent - route to appropriate reward function based on agent type
        current_agent_type = self._get_current_agent_state().agent_type[0]
        
        def get_tracked_rewards():
            return self._get_rewards_tracked(new_state, action)
        
        def get_wheeled_rewards():
            return self._get_rewards_wheeled(new_state, action)
            
        def get_skidsteer_rewards():
            return self._get_rewards_skidsteer(new_state, action)
        
        # Route rewards based on agent type: 0=tracked, 1=wheeled, 2=skidsteer, 3=truck
        # For truck, use dedicated reward handler
        def get_truck_rewards():
            return self._get_rewards_truck(new_state, action)
        reward_functions = [get_tracked_rewards, get_wheeled_rewards, get_skidsteer_rewards, get_truck_rewards]
        clamped_agent_type = jnp.clip(current_agent_type, 0, len(reward_functions) - 1)
        agent_reward = jax.lax.switch(clamped_agent_type, reward_functions)
        reward += agent_reward
        
        # Attribute per-step agent reward to the current agent index (active-first ordering in obs only)
        def set_idx(vec):
            return vec.at[self.agent.current_agent].set(agent_reward.astype(jnp.float32))
        components["agent_rewards"] = set_idx(components["agent_rewards"])

        # Terminal reward based on completion percentage
        completion_percentage = self._calculate_completion_percentage(
            new_state.world.action_map.map,
            self.world.target_map.map
        )
        done, done_task = self._is_done(
            new_state.world.action_map.map,
            self.world.target_map.map,
        )
        
        terminal_r = jax.lax.cond(
            done,
            lambda: self._calculate_terminal_reward(completion_percentage),
            lambda: 0.0,
        )
        # Divide terminal reward by number of active agents to share credit fairly
        active_count_f32 = jnp.sum(self.agent.agent_active.astype(jnp.float32))
        denom = jnp.maximum(active_count_f32, jnp.float32(1.0))
        #terminal_r = terminal_r * 2 / denom
        terminal_r = terminal_r * 2 / denom


        # terminal_r = jax.lax.cond(
        #     done_task,
        #     lambda: self.env_cfg.rewards.terminal,
        #     lambda: 0.0,
        # )

        reward += terminal_r
        components["terminal"] = terminal_r

        # Attribute terminal reward to BOTH agents in logging components (shared policy)
        # Count full terminal for both to reflect joint success in logs
        #components["agent1_rewards"] = components["agent1_rewards"] + terminal_r
        #components["agent2_rewards"] = components["agent2_rewards"] + terminal_r

        # Apply trench rewards - only for excavators (type 0)
        # Each excavator gets its own trench reward based on its position
        current_agent_type = self._get_current_agent_state().agent_type[0]
        is_excavator = current_agent_type == 0
        should_apply_trench = jnp.logical_and(self.env_cfg.apply_trench_rewards, is_excavator)
        
        trench_r = jax.lax.cond(
            should_apply_trench,
            self._get_trench_specific_rewards,
            lambda: 0.0,
        )
        reward += trench_r
        
        # Only log trench reward when it's an excavator's turn to avoid noise in plots
        components["trench"] = jax.lax.cond(
            is_excavator,
            lambda: trench_r,
            lambda: jnp.nan,  # Use NaN for skidsteer turns so they don't appear in plots
        )

        # Existence
        existence_r = self.env_cfg.rewards.existence
        reward += existence_r
        components["existence"] = existence_r

        # Constant scaling factor
        normalizer = self.env_cfg.rewards.normalizer
        reward = reward / normalizer
        # Normalize only reward-bearing components; keep masks/counts as integers
        components = {
            "agent_rewards": components["agent_rewards"] / normalizer,
            "terminal": components["terminal"] / normalizer,
            "trench": components["trench"] / normalizer,
            "existence": components["existence"] / normalizer,
            "agent_active": components["agent_active"],
            "num_agents": components["num_agents"],
        }

        return reward, components

    def _is_done_task(self, action_map: Array, target_map: Array):
        """
        Checks if the task is complete based on the type of task:
        1. Traditional tasks: target map requirements must be met
        2. Relocation tasks: all dirt must be in dump zones (no dirt in neutral areas)
        3. Cooperative tasks: excavator digs, skidsteer moves dirt - both must be complete

        On top of that, all agents should not be loaded.
        """

        def _check_done_dump():
            # For dump completion: check if all dirt is in designated dump zones OR 1-pixel buffer around them
            # This allows for more lenient termination while keeping rewards precise
            designated_dump_zones = target_map > 0
            
            # Expand dump zones by 1 pixel using the existing soil mechanics function
            expanded_dump_zones = self._expand_mask_for_soil_mechanics(designated_dump_zones)
            
            # Check if there's any dirt outside expanded dump zones
            dirt_outside_expanded = jnp.logical_and(action_map > 0, jnp.logical_not(expanded_dump_zones))
            
            # Task complete if NO dirt exists outside expanded dump zones
            done_dump = jnp.logical_not(jnp.any(dirt_outside_expanded))
            return done_dump

        def _check_done_dig():
            # For dig completion: action_map must be <= target_map for all target_map < 0
            # (since target_map < 0 means "dig to this depth" and action_map < 0 means "dug to this depth")
            dig_requirements = jnp.where(target_map < 0, target_map, 0)
            actual_digs = jnp.where(target_map < 0, action_map, 0)
            done_dig = jnp.all(actual_digs <= dig_requirements)
            return done_dig

        def _check_relocation_done():
            """
            For relocation tasks: Check if all dirt is in designated dump zones (target_map > 0).
            Optimized version that early exits and avoids full map operations.
            """
            # Early exit: if any active agent is loaded, task cannot be complete
            loaded_per_agent_dyn = jnp.array([
                self.agent.agent_states[0].loaded[0],
                self.agent.agent_states[1].loaded[0],
                self.agent.agent_states[2].loaded[0],
                self.agent.agent_states[3].loaded[0]
            ])
            active_dyn = self.agent.agent_active.astype(jnp.bool_)
            agents_loaded = jnp.any(jnp.logical_and(active_dyn, loaded_per_agent_dyn > 0))
            
            def _do_expensive_check():
                # Only do expensive operations when agents are unloaded
                # Get designated dump zones from target_map (target_map > 0)
                designated_dump_zones = target_map > 0
                # Create 1-tile buffer around dump zones using morphological dilation (3x3 kernel)
                kernel_3x3 = jnp.ones((3, 3), dtype=jnp.float32)
                dump_zones_with_buffer = jax.scipy.signal.correlate2d(
                    designated_dump_zones.astype(jnp.float32),
                    kernel_3x3,
                    mode='same',
                    boundary='fill',
                    fillvalue=0.0
                ) > 0
                
                # Check total dirt in environment
                total_dirt = jnp.sum(action_map > 0)
                
                # If there's no dirt at all, something is wrong - don't terminate
                # (Relocation tasks should always have dirt to move)
                def _check_dirt_distribution():
                    # Find dirt locations outside the buffered dump zones
                    dirt_outside_buffered = jnp.logical_and(action_map > 0, jnp.logical_not(dump_zones_with_buffer))
                    
                    # Task complete if NO dirt exists outside buffered dump zones
                    return jnp.logical_not(jnp.any(dirt_outside_buffered))
                
                def _no_dirt_case():
                    # If no dirt exists, don't terminate (likely environment initialization issue)
                    return False
                
                # Only check dirt distribution if there's actually dirt in the environment
                return jax.lax.cond(
                    total_dirt > 0,
                    _check_dirt_distribution,
                    _no_dirt_case
                )
            
            def _early_exit():
                return False  # Task not complete if agents still loaded
            
            # Only do expensive check if agents are unloaded
            return jax.lax.cond(
                agents_loaded,
                _early_exit,
                _do_expensive_check
            )

        def _check_cooperative_task_done():
            """
            For cooperative tasks (excavator + skidsteer):
            - Excavator digs: check if all dig requirements are met (target_map < 0)
            - Skidsteer moves dirt: check if all dirt is in dump zones (target_map > 0)
            - Both agents must be unloaded
            """
            # Both dig and dump must be complete for cooperative tasks
            done_dig = _check_done_dig()
            done_dump = _check_done_dump()
            
            # For cooperative tasks, we require BOTH conditions to be met
            # This ensures the excavator has finished digging AND the skidsteer has moved all dirt
            cooperative_complete = jnp.logical_and(done_dig, done_dump)
            
            # Additional check: ensure all active agents are unloaded for cooperative completion
            # This prevents premature termination when any agent still has dirt
            loaded_per_agent_dyn = jnp.array([
                self.agent.agent_states[0].loaded[0],
                self.agent.agent_states[1].loaded[0],
                self.agent.agent_states[2].loaded[0],
                self.agent.agent_states[3].loaded[0]
            ])
            active_dyn = self.agent.agent_active.astype(jnp.bool_)
            all_unloaded_dyn = jnp.all(jnp.logical_or(~active_dyn, loaded_per_agent_dyn == 0))
            cooperative_complete = jnp.logical_and(cooperative_complete, all_unloaded_dyn)
            
            return cooperative_complete

        # Check if this is a relocation task:
        # Relocation tasks have dump zones (target_map > 0) but no dig requirements (no target_map < 0)
        # AND at least one skidsteer agent is available (agent_type == 2)
        has_dump_requirements = jnp.any(target_map > 0)
        has_dig_requirements = jnp.any(target_map < 0)
        
        # Check if there are skidsteer agents available (type 2) among active agents
        # Check for any active skidsteer agents without dynamic tuple indexing
        agent_types = jnp.array([
            self.agent.agent_states[0].agent_type[0],
            self.agent.agent_states[1].agent_type[0],
            self.agent.agent_states[2].agent_type[0],
            self.agent.agent_states[3].agent_type[0]
        ])
        has_skidsteer_agent = jnp.any(jnp.logical_and(self.agent.agent_active, agent_types == 2))
        
        # Only use relocation logic if skidsteer agents are available
        is_relocation_task = jnp.logical_and(
            jnp.logical_and(has_dump_requirements, jnp.logical_not(has_dig_requirements)),
            has_skidsteer_agent
        )
        # Only use cooperative logic if skidsteer agents are available
        is_cooperative_task = jnp.logical_and(
            jnp.logical_and(has_dig_requirements, has_dump_requirements),
            has_skidsteer_agent
        )

        # Choose termination logic based on task type
        def _traditional_task_logic():
            # Traditional logic for tasks with specific target map requirements
            # Only check digging requirements - dump requirements are handled by relocation/cooperative logic
            done_dig = jax.lax.cond(
                jnp.all(target_map >= 0),  # No dig requirements
                lambda: True,
                _check_done_dig,
            )
            
            # Traditional tasks are complete when digging requirements are met
            return done_dig

        def _relocation_task_logic():
            # New logic for relocation tasks - check if all dirt is in dump zones
            return _check_relocation_done()

        # Select appropriate task completion logic
        task_requirements_met = jax.lax.cond(
            is_relocation_task,
            _relocation_task_logic,
            lambda: jax.lax.cond(
                is_cooperative_task,
                _check_cooperative_task_done,
                _traditional_task_logic
            )
        )
        
        # Task is complete when requirements are met AND all active agents are unloaded
        loaded_per_agent_dyn = jnp.array([
            self.agent.agent_states[0].loaded[0],
            self.agent.agent_states[1].loaded[0],
            self.agent.agent_states[2].loaded[0],
            self.agent.agent_states[3].loaded[0]
        ])
        active_dyn = self.agent.agent_active.astype(jnp.bool_)
        all_unloaded_dyn = jnp.all(jnp.logical_or(~active_dyn, loaded_per_agent_dyn == 0))
        done_task = jnp.logical_and(task_requirements_met, all_unloaded_dyn)
        return done_task

    def _is_done(
        self, action_map: Array, target_map: Array
    ) -> tuple[jnp.bool_, jnp.bool_]:
        done_task = self._is_done_task(action_map, target_map)
        done_steps = self.env_steps >= self.env_cfg.max_steps_in_episode
        return jnp.logical_or(done_task, done_steps), done_task

    def _get_action_mask_tracked(self):
        # forward
        new_state = self._handle_move_forward()
        bool_forward = ~jnp.all(
            new_state._get_prev_agent_state().pos_base == self._get_current_agent_state().pos_base
        )

        # backward
        new_state = self._handle_move_backward()
        bool_backward = ~jnp.all(
            new_state._get_prev_agent_state().pos_base == self._get_current_agent_state().pos_base
        )

        # clock
        new_state = self._handle_clock()
        bool_clock = ~jnp.all(
            new_state._get_prev_agent_state().angle_base == self._get_current_agent_state().angle_base
        )

        # anticlock
        new_state = self._handle_anticlock()
        bool_anticlock = ~jnp.all(
            new_state._get_prev_agent_state().angle_base == self._get_current_agent_state().angle_base
        )

        # cabin clock
        new_state = self._handle_cabin_clock()
        bool_cabin_clock = ~jnp.all(
            new_state._get_prev_agent_state().angle_cabin
            == self._get_current_agent_state().angle_cabin
        )

        # cabin clock
        new_state = self._handle_cabin_anticlock()
        bool_cabin_anticlock = ~jnp.all(
            new_state._get_prev_agent_state().angle_cabin
            == self._get_current_agent_state().angle_cabin
        )

        # do
        new_state = self._handle_do()
        bool_do = ~jnp.all(
            new_state._get_prev_agent_state().loaded == self._get_current_agent_state().loaded
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
            new_state._get_prev_agent_state().pos_base == self._get_current_agent_state().pos_base
        )

        # backward
        new_state = self._handle_move_backward()
        bool_backward = ~jnp.all(
            new_state._get_prev_agent_state().pos_base == self._get_current_agent_state().pos_base
        )

        # turn wheels left
        new_state = self._handle_turn_wheels_left()
        bool_turn_wheels_left = ~jnp.all(
            new_state._get_prev_agent_state().wheel_angle == self._get_current_agent_state().wheel_angle
        )

        # turn wheels right
        new_state = self._handle_turn_wheels_right()
        bool_turn_wheels_right = ~jnp.all(
            new_state._get_prev_agent_state().wheel_angle == self._get_current_agent_state().wheel_angle
        )

        # cabin clock
        new_state = self._handle_cabin_clock()
        bool_cabin_clock = ~jnp.all(
            new_state._get_prev_agent_state().angle_cabin
            == self._get_current_agent_state().angle_cabin
        )

        # cabin anticlock
        new_state = self._handle_cabin_anticlock()
        bool_cabin_anticlock = ~jnp.all(
            new_state._get_prev_agent_state().angle_cabin
            == self._get_current_agent_state().angle_cabin
        )

        # do
        new_state = self._handle_do()
        bool_do = ~jnp.all(
            new_state._get_prev_agent_state().loaded == self._get_current_agent_state().loaded
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
        current_agent_type = self._get_current_agent_state().agent_type[0]
        
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
        # Recompute target_tiles as union of cones for all active agents
        map_width = self.world.width
        map_height = self.world.height
        def cone_for_agent(agent_idx):
            agent_active = jax.lax.switch(
                agent_idx,
                [
                    lambda: self.agent.agent_active[0],
                    lambda: self.agent.agent_active[1],
                    lambda: self.agent.agent_active[2],
                    lambda: self.agent.agent_active[3],
                ]
            )
            def gen():
                temp_state = self._replace(agent=self.agent._replace(current_agent=agent_idx))
                return temp_state._build_dig_dump_cone().reshape(map_width, map_height)
            def zeros():
                return jnp.zeros((map_width, map_height), dtype=jnp.bool_)
            return jax.lax.cond(agent_active == 1, gen, zeros)
        cones = jax.vmap(cone_for_agent)(jnp.arange(4))
        target_tiles_mask = jnp.any(cones, axis=0).reshape(-1)
        infos = {
            "action_mask": self._get_action_mask(dummy_action),
            "target_tiles": target_tiles_mask,
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
        min_distance_from_agent = tile_size * (max_agent_dim - 2.0)

        # Slightly closer to agent than excavator but similar range
        fixed_extension = 0.1  # Slightly closer than excavator's 0.5  
        r_min = fixed_extension * dig_portion_radius * tile_size + min_distance_from_agent
        r_max = (fixed_extension + 1.4) * dig_portion_radius * tile_size + min_distance_from_agent  # Slimmer range than excavator's 1.0
        
        # Same angular range as excavator
        theta_max = 2 * np.pi / (self.env_cfg.agent.angles_cabin / 1.2)  # Same as excavator
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

    def _get_rewards_skidsteer(self, new_state: "State", action: ActionType) -> Float:
        """Specialized reward function for skid steer operations"""
        reward = 0.0
        action = action[0]

        # Movement rewards (skidsteer-specific, no move_while_loaded penalty)
        movement_reward = jax.lax.cond(
            (action == TrackedActionType.FORWARD)
            | (action == TrackedActionType.BACKWARD),
            lambda new_state, action: self._handle_rewards_move_skidsteer(new_state, action),
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

        # Holding dirt penalty
        reward += jax.lax.cond(
            self._get_current_agent_state().loaded[0] > 0,
            lambda: self.env_cfg.rewards.holding_dirt,
            lambda: 0.0)

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
        
        # Reward for moving or rotating while loaded and shovel is up, only if actually moved or rotated
        reward += jax.lax.cond(
            (
                ((action == TrackedActionType.FORWARD) | (action == TrackedActionType.BACKWARD))
                & (self._get_current_agent_state().loaded[0] > 0)
                & (self._get_current_agent_state().shovel_lifted[0] > 0)
                & self._check_agent_moved_on_move_action(self, new_state)
            )
            |
            (
                ((action == TrackedActionType.CLOCK) | (action == TrackedActionType.ANTICLOCK))
                & (self._get_current_agent_state().loaded[0] > 0)
                & (self._get_current_agent_state().shovel_lifted[0] > 0)
                & self._check_agent_turn_on_turn_action(self, new_state)
            ),
            lambda: self.env_cfg.rewards.skid_move_loaded_shovel_up,
            lambda: 0.0
        )
        
        return reward

    def _get_rewards_truck(self, new_state: "State", action: ActionType) -> Float:
        """Truck-specific rewards: proximity shaping when empty, avoid dig tiles penalty, reuse movement rewards."""
        reward = 0.0
        action = action[0]

        # Reuse tracked movement/base/cabin rewards logic (truck has cabin disabled already)
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
            action == TrackedActionType.DO,
            lambda new_state, action: self._handle_rewards_do(new_state, action)[0],
            lambda new_state, action: 0.0,
            new_state,
            action,
        )

        # Proximity shaping when empty
        cur = self._get_current_agent_state()
        is_truck = cur.agent_type[0] == 3
        is_empty = cur.loaded[0] == 0

        def _proximity_term():
            # Distance to nearest excavator base (squared)
            active = self.agent.agent_active.astype(jnp.bool_)
            types = jnp.array([
                self.agent.agent_states[0].agent_type[0],
                self.agent.agent_states[1].agent_type[0],
                self.agent.agent_states[2].agent_type[0],
                self.agent.agent_states[3].agent_type[0],
            ])
            posxs = jnp.array([
                self.agent.agent_states[0].pos_base[0],
                self.agent.agent_states[1].pos_base[0],
                self.agent.agent_states[2].pos_base[0],
                self.agent.agent_states[3].pos_base[0],
            ]).astype(jnp.float32)
            posys = jnp.array([
                self.agent.agent_states[0].pos_base[1],
                self.agent.agent_states[1].pos_base[1],
                self.agent.agent_states[2].pos_base[1],
                self.agent.agent_states[3].pos_base[1],
            ]).astype(jnp.float32)

            is_excavator = (types == 0)
            mask = jnp.logical_and(active, is_excavator)
            dx = posxs - cur.pos_base[0].astype(jnp.float32)
            dy = posys - cur.pos_base[1].astype(jnp.float32)
            d2 = dx * dx + dy * dy
            d2 = jnp.where(mask, d2, jnp.float32(1e9))
            min_d2 = jnp.min(d2)

            # Binary proximity bonus: +p if within R tiles
            R = jnp.float32(15.0)
            p = jnp.float32(0.1)
            near = min_d2 <= (R * R)
            bonus = jnp.where(near, p, 0.0)

            # Optional small penalty for sitting on dig tiles when empty
            on_dig = (self.world.target_map.map[cur.pos_base[0], cur.pos_base[1]] < 0)
            q = jnp.float32(0.02)
            penalty = jnp.where(on_dig, q, 0.0)

            # Disable bonus when on dig tiles; apply penalty instead
            return jnp.where(on_dig, -penalty, bonus)

        reward += jax.lax.cond(jnp.logical_and(is_truck, is_empty), _proximity_term, lambda: 0.0)
        return reward

    def _handle_rewards_move_skidsteer(
        self, new_state: "State", action: TrackedActionType
    ) -> tuple[Float, dict]:
        reward = 0.0
        # Sophisticated collision detection (matches single-agent)
        old_loaded = self._get_current_agent_state().loaded[0]
        new_loaded = new_state._get_prev_agent_state().loaded[0]  # After swap in multi-agent
        loaded_increased = new_loaded > old_loaded
        no_movement = ~self._check_agent_moved_on_move_action(self, new_state)
        collision_applicable = jnp.logical_and(no_movement, jnp.logical_not(loaded_increased))
        
        reward += jax.lax.cond(
            collision_applicable,
            lambda: self.env_cfg.rewards.collision_move,
            lambda: self.env_cfg.rewards.skid_move,
        )

        # Move while loaded
        reward += jax.lax.cond(
            jnp.all(self._get_current_agent_state().loaded > 0),
            lambda: self.env_cfg.rewards.move_while_loaded,
            lambda: 0.0,
        )

        # Moving with turned wheels (not applicable to skidsteer, but keeping for consistency)
        reward += jax.lax.cond(
            jnp.any(self._get_current_agent_state().wheel_angle != 0),
            lambda: self.env_cfg.rewards.move_with_turned_wheels,
            lambda: 0.0,
        )

        # Check for backwards dumping reward
        # This happens when moving backwards with shovel down and loaded
        old_loaded = self._get_current_agent_state().loaded[0]
        new_loaded = new_state._get_prev_agent_state().loaded[0]
        reward += jax.lax.cond(
            (action == TrackedActionType.BACKWARD) & (old_loaded > new_loaded),
            lambda: self._handle_rewards_skid_steer_dump(new_state, action),
            lambda: 0.0,
        )

        # Directional distance reward: only reward if getting closer to dump zones
        # FIX: Use correct agent positions to avoid state swapping bug
        # old_dist = self._get_min_distance_to_dump_zone_for_agent(self.agent.agent_state.pos_base)
        # new_dist = self._get_min_distance_to_dump_zone_for_agent(new_state.agent.agent_state_2.pos_base)
        # distance_improvement = old_dist - new_dist  # Positive if getting closer
        
        # # Reward/penalize based on distance change to dump zones when loaded
        # reward += jax.lax.cond(
        #     self.agent.agent_state.loaded[0] > 0,
        #     lambda: self.env_cfg.rewards.move_to_dump_zone * distance_improvement,
        #     lambda: 0.0,
        # )

        return reward
    def _get_min_distance_to_dump_zone_for_agent(self, agent_pos: Array) -> Float:
        """
        Returns the minimum Euclidean distance from a given agent position to any DESIGNATED dump zone tile,
        normalized by the map diagonal. Matches single-agent helper.
        """
        dump_zones_mask = self.world.target_map.map > 0
        width = self.world.width
        height = self.world.height
        xs = jnp.arange(width)
        ys = jnp.arange(height)
        grid_x, grid_y = jnp.meshgrid(xs, ys, indexing='ij')
        agent_pos_flat = jnp.atleast_1d(jnp.ravel(agent_pos))
        agent_x = agent_pos_flat[0]
        agent_y = agent_pos_flat[1]
        dists = (grid_x - agent_x) ** 2 + (grid_y - agent_y) ** 2
        masked_dists = jnp.where(dump_zones_mask, dists, jnp.inf)
        min_dist = jnp.sqrt(jnp.min(masked_dists))
        map_diagonal = jnp.sqrt(width**2 + height**2)
        return jnp.where(jnp.isfinite(min_dist), min_dist / map_diagonal, 0.0)

    def _calculate_dump_zone_reward(self, action_map_progress: Float, loaded_capacity: Float) -> Float:
        """
        Calculate reward/penalty for dump zone progress using consistent scaling.
        Positive progress (dumping) gets positive reward, negative progress (lifting) gets negative penalty.
        Uses loaded_capacity as reference for efficiency bonus.
        """
        # Use the same scaling logic as _calculate_efficiency_reward
        min_reward = 15.0
        max_reward = self.env_cfg.rewards.skid_dump_correct
        
        # Scale based on absolute dirt amount
        # Small amounts (1-5 units) get min_reward, large amounts (20+ units) get max_reward
        min_dirt = 1.0
        max_dirt = 20.0
        
        # Use absolute value for scaling
        abs_progress = jnp.abs(action_map_progress)
        clamped_dirt = jnp.clip(abs_progress, min_dirt, max_dirt)
        
        # Linear interpolation between min and max reward/penalty based on dirt amount
        reward_ratio = (clamped_dirt - min_dirt) / (max_dirt - min_dirt + 1e-8)
        scaled_reward = min_reward + reward_ratio * (max_reward - min_reward)
        
        # Apply perfect efficiency bonus based on percentage of loaded capacity used
        # This prevents reward farming: agent can't lift small amounts and dump large amounts for net positive reward
        efficiency_ratio = abs_progress / jnp.maximum(loaded_capacity, 1e-6)
        perfect_bonus = jnp.where(
            efficiency_ratio >= 0.95,  # Using 95%+ of capacity
            1.2,  # 20% bonus for efficient use of capacity
            1.0
        )
        
        # Apply additional 20% penalty increase for negative progress (lifting from dump zones)
        penalty_multiplier = jax.lax.cond(
            action_map_progress < 0,  # Only for penalties
            lambda: 1.05,  # 20% stronger penalty
            lambda: 1.0   # No change for rewards
        )
        
        # Return positive reward for positive progress, negative penalty for negative progress
        return (scaled_reward * perfect_bonus * penalty_multiplier) * jnp.sign(action_map_progress)

    def _calculate_terminal_reward(self, completion_percentage: Float) -> Float:
        """
        Calculate terminal reward based on completion percentage.
        Uses threshold + exponential scaling to discourage low effort and encourage high completion.
        """
        base_reward = self.env_cfg.rewards.terminal
        min_threshold = 0.50 #0.45 # 30% minimum completion required
        
        # No reward for very poor performance
        def _no_reward():
            return 0.0
        
        def _calculate_scaled_reward():
            # Scale from min_threshold to 100% with exponential curve
            # This makes the curve steeper and discourages mediocre completion
            scaled_percentage = (completion_percentage - min_threshold) / (1.0 - min_threshold)
            exponential_percentage = scaled_percentage ** 2  # Square for steep curve
            return base_reward * exponential_percentage
        
        return jax.lax.cond(
            completion_percentage < min_threshold,
            _no_reward,
            _calculate_scaled_reward
        )

    def _calculate_completion_percentage(self, action_map: Array, target_map: Array) -> Float:
        """
        Calculate completion percentage based on how much dirt is in correct dump zones.
        Returns a value between 0.0 and 1.0.
        Now includes undug tiles (target_map == -1) as dirt not relocated.
        """
        # Get designated dump zones (target_map > 0)
        designated_dump_zones = target_map > 0
        
        # Get areas that need to be dug (target_map < 0) but haven't been dug yet (action_map >= 0)
        undug_areas = jnp.logical_and(target_map < 0, action_map >= 0)
        
        # Calculate total dirt volume in the environment:
        # 1. Dirt that has been moved and dumped (action_map > 0)
        # 2. Undug tiles that still need to be relocated (target_map == -1)
        moved_dirt = jnp.sum(jnp.where(action_map > 0, action_map, 0))
        undug_dirt = jnp.sum(jnp.where(undug_areas, 1.0, 0.0))  # Count undug tiles as 1 unit each
        # Sum dirt currently loaded across all active agents.
        # Avoid dynamic tuple indexing by statically stacking per-agent loaded values.
        loaded_per_agent = jnp.array([
            self.agent.agent_states[0].loaded[0],
            self.agent.agent_states[1].loaded[0],
            self.agent.agent_states[2].loaded[0],
            self.agent.agent_states[3].loaded[0]
        ])
        loaded_dirt = jnp.sum(jnp.where(self.agent.agent_active, loaded_per_agent, 0))
        total_dirt = moved_dirt + undug_dirt + loaded_dirt
        
        # Calculate dirt volume in correct dump zones (sum of heights)
        dirt_in_correct_zones = jnp.sum(jnp.where(
            jnp.logical_and(action_map > 0, designated_dump_zones),
            action_map,
            0
        ))
        
        # Calculate completion percentage
        # If no dirt exists, return 0.0 (no completion)
        completion_percentage = jax.lax.cond(
            total_dirt > 0,
            lambda: dirt_in_correct_zones / total_dirt,
            lambda: 0.0
        )
        
        return completion_percentage

    def _compute_relocation_potential(self, action_map: Array) -> Float:
        """
        Relocation potential: sum over non-dump tiles of positive dirt times distance to nearest dump zone.
        Uses cached world.relocation_distance_map (float32, normalized).
        """
        dist_map = self.world.relocation_distance_map
        return jnp.sum(
            jnp.where(
                self.world.target_map.map <= 0,
                jnp.clip(action_map, a_min=0) * dist_map,
                0,
            )
        )


