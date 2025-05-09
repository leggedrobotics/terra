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
    def wrap(state: State) -> State:
        """
        Encodes the traversability mask in GridWorld.

        The traversability mask has the same size as the action map, and encodes:
        - 0: no obstacle
        - 1: obstacle (digged or dumped tile)
        - (-1): agent occupying the tile
        """
        # encode map obstacles
        traversability_mask = (state.world.action_map.map != 0).astype(IntLowDim)

        # encode agent pos and size in the map
        agent_corners_1 = state._get_agent_corners(
            state.agent.agent_state_1.pos_base,
            state.agent.agent_state_1.angle_base,
            state.env_cfg.agent.width,
            state.env_cfg.agent.height,
        )

        agent_corners_2 = state._get_agent_corners(
            state.agent.agent_state_2.pos_base,
            state.agent.agent_state_2.angle_base,
            state.env_cfg.agent.width,
            state.env_cfg.agent.height,
        )

        map_width = state.world.width
        map_height = state.world.height
        
        polygon_mask1 = compute_polygon_mask(agent_corners_1, map_width, map_height)
        polygon_mask2 = compute_polygon_mask(agent_corners_2, map_width, map_height)
        traversability_mask = jnp.where(polygon_mask1, -1, traversability_mask)
        traversability_mask = jnp.where(polygon_mask2, -1, traversability_mask)

        padding_mask = state.world.padding_mask.map
        tm = jnp.where(padding_mask == 1, padding_mask, traversability_mask)

        return state._replace(
            world=state.world._replace(
                traversability_mask=state.world.traversability_mask._replace(
                    map=tm.astype(IntLowDim)
                )
            )
        )


class LocalMapWrapper:
    @staticmethod
    def _wrap(state: State, map_to_wrap: Array) -> Array:
        """
        Encodes the local map in the GridWorld.

        The local map is of dim angles_cabin, and encodes the cumulative
        sum of tiles to dig in the area spanned by the cyilindrical tile.
        """
        current_pos_idx = state._get_current_pos_vector_idx(
            pos_base=state.agent.agent_state_1.pos_base,
            map_height=state.env_cfg.maps.edge_length_px,
        )
        map_global_coords = state._map_to_flattened_global_coords(
            state.world.width, state.world.height, state.env_cfg.tile_size
        )
        current_pos = state._get_current_pos_from_flattened_map(
            map_global_coords, current_pos_idx
        )
        current_arm_angle = get_arm_angle_int(
            state.agent.agent_state_1.angle_base,
            state.agent.agent_state_1.angle_cabin,
            state.env_cfg.agent.angles_base,
            state.env_cfg.agent.angles_cabin,
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
    def wrap_target_map(state: State) -> State:
        target_map_pos = jnp.clip(state.world.target_map.map, a_min=0)
        target_map_neg = jnp.clip(state.world.target_map.map, a_max=0)
        local_map_target_pos = LocalMapWrapper._wrap(state, target_map_pos)
        local_map_target_neg = LocalMapWrapper._wrap(state, target_map_neg)
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
    def wrap_action_map(state: State) -> State:
        action_map_pos = jnp.clip(state.world.action_map.map, a_min=0)
        action_map_neg = jnp.clip(state.world.action_map.map, a_max=0)
        local_map_action_pos = LocalMapWrapper._wrap(state, action_map_pos)
        local_map_action_neg = LocalMapWrapper._wrap(state, action_map_neg)
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
    def wrap_dumpability_mask(state: State) -> State:
        dumpability_mask = state.world.dumpability_mask.map
        local_map_dumpability = LocalMapWrapper._wrap(state, dumpability_mask)
        return state._replace(
            world=state.world._replace(
                local_map_dumpability=state.world.local_map_dumpability._replace(
                    map=local_map_dumpability
                )
            )
        )

    @staticmethod
    def wrap_obstacles_mask(state: State) -> State:
        obstacles_mask = state.world.padding_mask.map
        local_map_obstacles = LocalMapWrapper._wrap(state, obstacles_mask)
        return state._replace(
            world=state.world._replace(
                local_map_obstacles=state.world.local_map_obstacles._replace(
                    map=local_map_obstacles
                )
            )
        )

    @staticmethod
    def wrap(state: State) -> State:
        state = LocalMapWrapper.wrap_target_map(state)
        state = LocalMapWrapper.wrap_action_map(state)
        state = LocalMapWrapper.wrap_dumpability_mask(state)
        state = LocalMapWrapper.wrap_obstacles_mask(state)
        return state
