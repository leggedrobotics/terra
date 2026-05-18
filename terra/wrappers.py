import jax
import jax.numpy as jnp
from jax import Array

from terra.config import EnvConfig
from terra.state import State
from terra.utils import angle_idx_to_rad
from terra.utils import apply_local_cartesian_to_cyl
from terra.utils import apply_rot_transl
from terra.utils import compute_polygon_mask
from terra.utils import get_arm_angle_int
from terra.settings import IntLowDim


class TraversabilityMaskWrapper:
    @staticmethod
    def _downsample_or_factor2(mask: Array) -> Array:
        h, w = mask.shape
        h2 = h // 2
        w2 = w // 2
        trimmed = mask[: 2 * h2, : 2 * w2]
        return trimmed.reshape(h2, 2, w2, 2).any(axis=(1, 3))

    @staticmethod
    def _upsample_nearest_factor2(mask: Array, out_h: int, out_w: int) -> Array:
        up = jnp.repeat(jnp.repeat(mask, 2, axis=0), 2, axis=1)
        return up[:out_h, :out_w]

    @staticmethod
    def _inflate_blocked(blocked: Array, radius: int) -> Array:
        radius = jnp.maximum(jnp.int32(radius), 0)

        def _dilate_once(x):
            p = jnp.pad(x, 1, mode="constant", constant_values=False)
            return (
                p[1:-1, 1:-1]
                | p[:-2, 1:-1]
                | p[2:, 1:-1]
                | p[1:-1, :-2]
                | p[1:-1, 2:]
                | p[:-2, :-2]
                | p[:-2, 2:]
                | p[2:, :-2]
                | p[2:, 2:]
            )

        def _body(_, x):
            return _dilate_once(x)

        return jax.lax.fori_loop(0, radius, _body, blocked)

    @staticmethod
    def _build_reachability_mask(passable: Array, start_mask: Array) -> Array:
        start = jnp.logical_and(passable, start_mask)
        frontier = start
        visited = start
        max_steps = passable.shape[0] * passable.shape[1]

        def _expand(n):
            p = jnp.pad(n, 1, mode="constant", constant_values=False)
            return (
                p[:-2, 1:-1]
                | p[2:, 1:-1]
                | p[1:-1, :-2]
                | p[1:-1, 2:]
            )

        def _step(carry):
            step_i, fr, vis = carry
            nbr = _expand(fr)
            new_fr = jnp.logical_and(jnp.logical_and(nbr, passable), jnp.logical_not(vis))
            vis2 = jnp.logical_or(vis, new_fr)
            return (step_i + 1, new_fr, vis2)

        def _cond(carry):
            step_i, fr, _ = carry
            return jnp.logical_and(step_i < max_steps, jnp.any(fr))

        _, _, visited = jax.lax.while_loop(_cond, _step, (jnp.int32(0), frontier, visited))
        return visited.astype(IntLowDim)

    @staticmethod
    def wrap(state: State, update_reachability: jnp.bool_ = jnp.bool_(True)) -> State:
        """
        Encodes the traversability mask in GridWorld.

        The traversability mask has the same size as the action map, and encodes:
        - 0: no obstacle
        - 1: obstacle (digged or dumped tile)
        - (-1): agent occupying the tile
        """
        # encode map obstacles
        traversability_mask = (state.world.action_map.map != 0).astype(IntLowDim)

        # encode agent pos and size in the map for all active agents
        map_width = state.world.width
        map_height = state.world.height
        
        # Process all active agents using jax.lax.switch for JAX compatibility
        def process_agent_idx(agent_idx):
            # Get agent state using jax.lax.switch
            agent_state = jax.lax.switch(
                agent_idx,
                [
                    lambda: state.agent.agent_states[0],
                    lambda: state.agent.agent_states[1],
                    lambda: state.agent.agent_states[2],
                    lambda: state.agent.agent_states[3],
                ]
            )
            # Get agent active status
            agent_active = jax.lax.switch(
                agent_idx,
                [
                    lambda: state.agent.agent_active[0],
                    lambda: state.agent.agent_active[1],
                    lambda: state.agent.agent_active[2],
                    lambda: state.agent.agent_active[3],
                ]
            )
            
            agent_corners = state._get_agent_corners(
                agent_state.pos_base,
                agent_state.angle_base,
                state.env_cfg.agent.width,
                state.env_cfg.agent.height,
            )
            polygon_mask = compute_polygon_mask(agent_corners, map_width, map_height)
            return jnp.where(agent_active, polygon_mask, jnp.zeros_like(polygon_mask))
        
        is_single_agent = state.agent.num_agents == 1

        def _single_agent_masks():
            agent_state = state.agent.agent_states[0]
            agent_corners = state._get_agent_corners(
                agent_state.pos_base,
                agent_state.angle_base,
                state.env_cfg.agent.width,
                state.env_cfg.agent.height,
            )
            polygon_mask = compute_polygon_mask(agent_corners, map_width, map_height)
            combined_agent_mask = jnp.where(
                state.agent.agent_active[0],
                polygon_mask,
                jnp.zeros_like(polygon_mask),
            )

            temp_state = state._replace(agent=state.agent._replace(current_agent=jnp.int32(0)))
            interaction_mask = temp_state._build_dig_dump_cone().reshape(map_width, map_height)
            return combined_agent_mask, interaction_mask

        def _multi_agent_masks():
            # Process all 4 agents (some may be inactive)
            agent_masks = jax.vmap(process_agent_idx)(jnp.arange(4))
            combined_agent_mask = jnp.any(agent_masks, axis=0)

            # Generate interaction mask for all active agents
            def get_agent_interaction_mask(agent_idx):
                # Get agent state using jax.lax.switch
                agent_state = jax.lax.switch(
                    agent_idx,
                    [
                        lambda: state.agent.agent_states[0],
                        lambda: state.agent.agent_states[1],
                        lambda: state.agent.agent_states[2],
                        lambda: state.agent.agent_states[3],
                    ]
                )
                # Get agent active status
                agent_active = jax.lax.switch(
                    agent_idx,
                    [
                        lambda: state.agent.agent_active[0],
                        lambda: state.agent.agent_active[1],
                        lambda: state.agent.agent_active[2],
                        lambda: state.agent.agent_active[3],
                    ]
                )
                
                # Only generate cone for active agents
                def generate_cone():
                    # Temporarily set current agent to this agent to generate its cone
                    temp_state = state._replace(agent=state.agent._replace(current_agent=agent_idx))
                    return temp_state._build_dig_dump_cone()
                
                def no_cone():
                    return jnp.zeros((map_width * map_height,), dtype=jnp.bool_)
                
                cone = jax.lax.cond(agent_active, generate_cone, no_cone)
                return cone.reshape(map_width, map_height)
            
            agent_interaction_masks = jax.vmap(get_agent_interaction_mask)(jnp.arange(4))
            interaction_mask = jnp.any(agent_interaction_masks, axis=0)
            return combined_agent_mask, interaction_mask

        combined_agent_mask, interaction_mask = jax.lax.cond(
            is_single_agent,
            _single_agent_masks,
            _multi_agent_masks,
        )
        traversability_mask = jnp.where(combined_agent_mask, -1, traversability_mask)

        static_base = state.world.static_traversability_base.map
        tm = jnp.where(static_base == 1, static_base, traversability_mask)

        # Optional global reachability channel (from current agent footprint),
        # with inflated blocked-space to account for excavator clearance.
        def _compute_reachability():
            current_idx = state.agent.current_agent
            cur = jax.lax.switch(
                current_idx,
                [
                    lambda: state.agent.agent_states[0],
                    lambda: state.agent.agent_states[1],
                    lambda: state.agent.agent_states[2],
                    lambda: state.agent.agent_states[3],
                ],
            )
            cur_corners = state._get_agent_corners(
                cur.pos_base,
                cur.angle_base,
                state.env_cfg.agent.width,
                state.env_cfg.agent.height,
            )
            current_agent_mask = compute_polygon_mask(cur_corners, map_width, map_height)
            blocked = tm == 1
            inflated = TraversabilityMaskWrapper._inflate_blocked(
                blocked, getattr(state.env_cfg, "reachability_inflation_tiles", 2)
            )
            passable = jnp.logical_not(inflated)
            return TraversabilityMaskWrapper._build_reachability_mask(
                passable,
                current_agent_mask,
            )

        should_update_reachability = jnp.logical_and(
            jnp.bool_(getattr(state.env_cfg, "enable_reachability_obs", False)),
            update_reachability,
        )
        reachability_mask = jax.lax.cond(
            should_update_reachability,
            _compute_reachability,
            lambda: state.world.reachability_mask.map.astype(IntLowDim),
        )

        return state._replace(
            # increase number of steps as well
            # env_steps=state.env_steps + 1,
            world=state.world._replace(
            traversability_mask=state.world.traversability_mask._replace(
                map=tm.astype(IntLowDim)
            ),
            reachability_mask=state.world.reachability_mask._replace(
                map=reachability_mask.astype(IntLowDim)
            ),
            interaction_mask=state.world.interaction_mask._replace(
                map=interaction_mask.astype(jnp.bool_)
            ),
            )
        )



class LocalMapWrapper:
    @staticmethod
    def _obstacle_boundary_mask(padding_mask: Array) -> Array:
        """
        Convert a dense obstacle mask into a thinner boundary-only mask for observations.

        This keeps full obstacles for collision checks elsewhere, but avoids feeding
        large filled regions into the local obstacle summary map.
        """
        obstacle = padding_mask == 1
        padded = jnp.pad(obstacle, 1, mode="constant", constant_values=False)
        interior = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        boundary = obstacle & (~interior)
        return boundary.astype(padding_mask.dtype)

    @staticmethod
    def _build_local_cartesian_masks(state: State, agent_idx: int = 0) -> tuple[Array, Array]:
        """
        Build the per-angle local workspace masks once for a specific agent.
        """
        # Get agent state using jax.lax.switch for JAX compatibility
        def get_agent_state(idx):
            return jax.lax.switch(
                idx,
                [
                    lambda: state.agent.agent_states[0],
                    lambda: state.agent.agent_states[1],
                    lambda: state.agent.agent_states[2],
                    lambda: state.agent.agent_states[3],
                ]
            )
        
        agent_state = get_agent_state(agent_idx)
        
        current_pos_idx = state._get_current_pos_vector_idx(
            pos_base=agent_state.pos_base,
            map_height=state.env_cfg.maps.edge_length_px,
        )
        map_global_coords = state._map_to_flattened_global_coords(
            state.world.width, state.world.height, state.env_cfg.tile_size
        )
        current_pos = state._get_current_pos_from_flattened_map(
            map_global_coords, current_pos_idx
        )

        # Get the cumsum of the action height map in cyl coords, for every of [r, theta] portion of local space
        angles_cabin = (
            EnvConfig().agent.angles_cabin
        )  # TODO: make state.env_cfg work instead of recreating the object every time
        arm_angles_ints = jnp.arange(angles_cabin)
        arm_angles_rads = jax.vmap(
            lambda angle: angle_idx_to_rad(angle, EnvConfig().agent.angles_cabin)
        )(arm_angles_ints)

        possible_states_arm = jax.vmap(lambda angle: jnp.hstack([current_pos, angle]))(
            arm_angles_rads
        )
        possible_maps_local_coords_arm = jax.vmap(
            lambda arm_state: apply_rot_transl(arm_state, map_global_coords)
        )(possible_states_arm)
        possible_maps_cyl_coords = jax.vmap(apply_local_cartesian_to_cyl)(
            possible_maps_local_coords_arm
        )  # (n_angles x 2 x width*height)

        local_cartesian_masks = jax.vmap(lambda map: state._get_dig_dump_mask_cyl(map))(
            possible_maps_cyl_coords
        )
        current_arm_angle = get_arm_angle_int(
            agent_state.angle_base,
            agent_state.angle_cabin,
            state.env_cfg.agent.angles_base,
            state.env_cfg.agent.angles_cabin,
        )
        return local_cartesian_masks, current_arm_angle

    @staticmethod
    def _wrap_with_masks(state: State, map_to_wrap: Array, local_cartesian_masks: Array, current_arm_angle: Array) -> Array:
        """
        Encodes the local map using precomputed per-angle workspace masks.
        """
        map_to_wrap_reshaped = map_to_wrap.reshape(state.world.height, state.world.width)
        local_cyl_height_map = jax.vmap(
            lambda mask: (map_to_wrap_reshaped * mask.reshape(state.world.height, state.world.width)).sum()
        )(local_cartesian_masks)

        # Roll it to bring it back in agent view
        local_cyl_height_map = jnp.roll(
            local_cyl_height_map, -current_arm_angle, axis=0
        )

        return local_cyl_height_map.astype(IntLowDim)

    @staticmethod
    def _wrap(state: State, map_to_wrap: Array, agent_idx: int = 0) -> Array:
        """
        Encodes the local map in the GridWorld for a specific agent.

        The local map is of dim angles_cabin, and encodes the cumulative
        sum of tiles to dig in the area spanned by the cyilindrical tile.
        """
        local_cartesian_masks, current_arm_angle = LocalMapWrapper._build_local_cartesian_masks(
            state, agent_idx
        )
        return LocalMapWrapper._wrap_with_masks(
            state, map_to_wrap, local_cartesian_masks, current_arm_angle
        )

    
    @staticmethod
    def wrap_target_map(state: State, agent_idx: int = 0) -> State:
        target_map_pos = jnp.clip(state.world.target_map.map, a_min=0)
        target_map_neg = jnp.clip(state.world.target_map.map, a_max=0)
        local_map_target_pos = LocalMapWrapper._wrap(state, target_map_pos, agent_idx)
        local_map_target_neg = LocalMapWrapper._wrap(state, target_map_neg, agent_idx)
        return state._replace(
            world=state.world._replace(
                local_map_target_pos=state.world.local_map_target_pos._replace(
                    map=local_map_target_pos
                ),
                local_map_target_neg=state.world.local_map_target_neg._replace(
                    map=local_map_target_neg
                ),
            )
        )
    

    @staticmethod
    def wrap_action_map(state: State, agent_idx: int = 0) -> State:
        action_map_pos = jnp.clip(state.world.action_map.map, a_min=0)
        action_map_neg = jnp.clip(state.world.action_map.map, a_max=0)
        local_map_action_pos = LocalMapWrapper._wrap(state, action_map_pos, agent_idx)
        local_map_action_neg = LocalMapWrapper._wrap(state, action_map_neg, agent_idx)
        return state._replace(
            world=state.world._replace(
                local_map_action_pos=state.world.local_map_action_pos._replace(
                    map=local_map_action_pos
                ),
                local_map_action_neg=state.world.local_map_action_neg._replace(
                    map=local_map_action_neg
                ),
            )
        )
    

    @staticmethod
    def wrap_dumpability_mask(state: State, agent_idx: int = 0) -> State:
        local_map_dumpability = LocalMapWrapper._wrap(state, state.world.dumpability_mask.map, agent_idx)
        return state._replace(
            world=state.world._replace(
                local_map_dumpability=state.world.local_map_dumpability._replace(
                    map=local_map_dumpability
                )
            )
        )
    

    @staticmethod
    def wrap_obstacle_mask(state: State, agent_idx: int = 0) -> State:
        obstacle_obs_mask = LocalMapWrapper._obstacle_boundary_mask(
            state.world.padding_mask.map
        )
        local_map_obstacles = LocalMapWrapper._wrap(
            state,
            obstacle_obs_mask,
            agent_idx,
        )
        return state._replace(
            world=state.world._replace(
                local_map_obstacles=state.world.local_map_obstacles._replace(
                    map=local_map_obstacles
                )
            )
        )
    
    
    @staticmethod
    def wrap(state: State) -> State:
        """Wrapper that calls all the single-map wrappers for the currently active agent"""
        # Debug prints removed for training performance
        
        # Create local maps for the currently active agent
        current_agent_idx = state.agent.current_agent
        local_cartesian_masks, current_arm_angle = LocalMapWrapper._build_local_cartesian_masks(
            state, current_agent_idx
        )

        target_map_pos = jnp.clip(state.world.target_map.map, a_min=0)
        target_map_neg = jnp.clip(state.world.target_map.map, a_max=0)
        action_map_pos = jnp.clip(state.world.action_map.map, a_min=0)
        action_map_neg = jnp.clip(state.world.action_map.map, a_max=0)
        obstacle_obs_mask = LocalMapWrapper._obstacle_boundary_mask(
            state.world.padding_mask.map
        )

        local_map_target_pos = LocalMapWrapper._wrap_with_masks(
            state, target_map_pos, local_cartesian_masks, current_arm_angle
        )
        local_map_target_neg = LocalMapWrapper._wrap_with_masks(
            state, target_map_neg, local_cartesian_masks, current_arm_angle
        )
        local_map_action_pos = LocalMapWrapper._wrap_with_masks(
            state, action_map_pos, local_cartesian_masks, current_arm_angle
        )
        local_map_action_neg = LocalMapWrapper._wrap_with_masks(
            state, action_map_neg, local_cartesian_masks, current_arm_angle
        )
        local_map_dumpability = LocalMapWrapper._wrap_with_masks(
            state, state.world.dumpability_mask.map, local_cartesian_masks, current_arm_angle
        )
        local_map_obstacles = LocalMapWrapper._wrap_with_masks(
            state, obstacle_obs_mask, local_cartesian_masks, current_arm_angle
        )
        border_mask = state._get_foundation_border_mask().astype(jnp.float32)
        local_map_border_workspace = LocalMapWrapper._wrap_with_masks(
            state, border_mask, local_cartesian_masks, current_arm_angle
        )

        dig_target = (state.world.target_map.map < 0).astype(jnp.float32)
        kernel_dx = jnp.array([[-1.0, 0.0, 1.0]], dtype=jnp.float32)
        kernel_dy = jnp.array([[-1.0], [0.0], [1.0]], dtype=jnp.float32)
        grad_x = jax.scipy.signal.convolve2d(
            dig_target, kernel_dx, mode="same", boundary="fill", fillvalue=0
        )
        grad_y = jax.scipy.signal.convolve2d(
            dig_target, kernel_dy, mode="same", boundary="fill", fillvalue=0
        )
        normal_angle = jnp.arctan2(grad_y, grad_x)
        edge_angle = (normal_angle + (jnp.pi / 2.0) + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        arm_angle = jnp.squeeze(state._get_arm_angle_rad())
        angle_diff = jnp.abs((edge_angle - arm_angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi)
        angle_diff = jnp.minimum(angle_diff, jnp.pi - angle_diff)
        edge_alignment_error_map = angle_diff * border_mask
        local_map_edge_alignment_error = LocalMapWrapper._wrap_with_masks(
            state, edge_alignment_error_map, local_cartesian_masks, current_arm_angle
        )

        workspace_mask = local_cartesian_masks[current_arm_angle].reshape(
            state.world.height, state.world.width
        )
        border_allowed_mask = state._get_foundation_border_alignment_mask(
            workspace_mask.reshape(-1)
        ).reshape(state.world.height, state.world.width)
        border_diggable_map = jnp.logical_and(border_mask > 0, border_allowed_mask).astype(
            jnp.float32
        )
        local_map_border_diggable = LocalMapWrapper._wrap_with_masks(
            state, border_diggable_map, local_cartesian_masks, current_arm_angle
        )

        state = state._replace(
            world=state.world._replace(
                local_map_target_pos=state.world.local_map_target_pos._replace(
                    map=local_map_target_pos
                ),
                local_map_target_neg=state.world.local_map_target_neg._replace(
                    map=local_map_target_neg
                ),
                local_map_action_pos=state.world.local_map_action_pos._replace(
                    map=local_map_action_pos
                ),
                local_map_action_neg=state.world.local_map_action_neg._replace(
                    map=local_map_action_neg
                ),
                local_map_dumpability=state.world.local_map_dumpability._replace(
                    map=local_map_dumpability
                ),
                local_map_obstacles=state.world.local_map_obstacles._replace(
                    map=local_map_obstacles
                ),
                local_map_border_workspace=state.world.local_map_border_workspace._replace(
                    map=local_map_border_workspace
                ),
                local_map_edge_alignment_error=state.world.local_map_edge_alignment_error._replace(
                    map=local_map_edge_alignment_error
                ),
                local_map_border_diggable=state.world.local_map_border_diggable._replace(
                    map=local_map_border_diggable
                ),
            )
        )
        
        # Debug prints removed for training performance
        
        return state
