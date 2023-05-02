import jax
import jax.numpy as jnp

from terra.config import EnvConfig
from terra.state import State
from terra.utils import angle_idx_to_rad
from terra.utils import apply_local_cartesian_to_cyl
from terra.utils import apply_rot_transl
from terra.utils import get_arm_angle_int
from terra.utils import IntLowDim


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
        agent_corners = state._get_agent_corners(
            state.agent.agent_state.pos_base,
            state.agent.agent_state.angle_base,
            state.env_cfg.agent.width,
            state.env_cfg.agent.height,
        )
        x, y = state._get_agent_corners_xy(agent_corners)

        map_width = state.world.width
        map_height = state.world.height
        traversability_mask = jnp.where(
            jnp.logical_or(
                jnp.logical_or(
                    (jnp.arange(map_width) > x[1])[:, None].repeat(map_height, axis=1),
                    (jnp.arange(map_width) < x[0])[:, None].repeat(map_height, axis=1),
                ),
                jnp.logical_or(
                    (jnp.arange(map_height) > y[1])[None].repeat(map_width, axis=0),
                    (jnp.arange(map_height) < y[0])[None].repeat(map_width, axis=0),
                ),
            ),
            traversability_mask,
            -1,
        )

        return state._replace(
            world=state.world._replace(
                traversability_mask=state.world.traversability_mask._replace(
                    map=traversability_mask.astype(IntLowDim)
                )
            )
        )


class LocalMapWrapper:
    @staticmethod
    def wrap(state: State) -> State:
        """
        Encodes the local map in the GridWorld.

        The local map is is (angles_cabin, max_arm_extension), and encodes
        the cumulative sum of tiles to dig in the area spanned by the cyilindrical tile.
        """
        current_pos_idx = state._get_current_pos_vector_idx(
            pos_base=state.agent.agent_state.pos_base,
            map_height=state.env_cfg.action_map.height,
        )
        map_global_coords = state._map_to_flattened_global_coords(
            state.world.width, state.world.height, state.env_cfg.tile_size
        )
        current_pos = state._get_current_pos_from_flattened_map(
            map_global_coords, current_pos_idx
        )
        current_arm_angle = get_arm_angle_int(
            state.agent.agent_state.angle_base,
            state.agent.agent_state.angle_cabin,
            state.env_cfg.agent.angles_base,
            state.env_cfg.agent.angles_cabin,
        )

        # Get the cumsum of the action height map in cyl coords, for every of [r, theta] portion of local space
        angles_cabin = (
            EnvConfig().agent.angles_cabin
        )  # TODO: state.env_cfg... does not work -- why?
        arm_angles_ints = jnp.arange(angles_cabin)
        arm_extensions = jnp.arange(EnvConfig().agent.max_arm_extension + 1)
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

        local_cartesian_masks = jax.vmap(
            lambda map: jax.vmap(
                lambda arm_extension: state._get_dig_dump_mask_cyl(map, arm_extension)
            )(arm_extensions)
        )(
            possible_maps_cyl_coords
        )  # (n_angles x n_arm_extensions x width*height)

        # Go from mask to masked height map to local cylindrical cumsum(s)
        local_cyl_height_map = jax.vmap(
            jax.vmap(
                lambda x: (
                    state.world.action_map.map
                    * x.reshape(state.world.width, state.world.height)
                ).sum()
            )
        )(local_cartesian_masks)

        # Roll it to bring it back in agent view
        local_cyl_height_map = jnp.roll(
            local_cyl_height_map, -current_arm_angle, axis=0
        )

        return state._replace(
            world=state.world._replace(
                local_map=state.world.local_map._replace(
                    map=local_cyl_height_map.astype(IntLowDim)
                )
            )
        )
