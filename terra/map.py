from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.map_generator import GridMap
from terra.settings import IntLowDim


# ``regularise_edges`` can add a one-cell raster fringe outside the ideal
# arm capsule.  The current generated bank's maximum measured excess is
# 1.354 tiles, so 1.5 admits that fringe without assigning arbitrarily distant
# negative targets to a trench section on a future mixed-purpose map.
TRENCH_MEMBERSHIP_FRINGE_TOLERANCE_TILES = 1.5


def _as_2d_map(x: Array) -> Array:
    x = jnp.asarray(x)
    return jnp.reshape(x, (-1,) + x.shape[-2:])[0]


def _as_axes_table(x: Array) -> Array:
    x = jnp.asarray(x)
    return jnp.reshape(x, (-1, x.shape[-2], x.shape[-1]))[0]


def _as_scalar_int(x: Array) -> Array:
    return jnp.ravel(jnp.asarray(x, dtype=jnp.int32))[0]


def compute_dynamic_dumpability(
    dumpability_mask_init: Array,
    action_map: Array,
    kernel_size: int = 5,
) -> Array:
    """Apply Terra's hole-clearance rule to a static dumpability mask."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    static_mask = _as_2d_map(dumpability_mask_init).astype(jnp.bool_)
    holes = (_as_2d_map(action_map) < 0).astype(jnp.float32)
    dilated_holes = (
        jax.lax.reduce_window(
            holes,
            jnp.float32(0.0),
            jax.lax.add,
            window_dimensions=(kernel_size, kernel_size),
            window_strides=(1, 1),
            padding="SAME",
        )
        > 0
    )
    return jnp.logical_and(static_mask, jnp.logical_not(dilated_holes))


def compute_trench_axis_membership(
    target_map: Array,
    trench_axes: Array,
    trench_type: Array,
) -> Array:
    """Encode generated finite-section ownership as one uint8 bit per axis."""

    target_map = _as_2d_map(target_map)
    records = _as_axes_table(trench_axes).astype(jnp.float32)
    axes = records[:, :3]
    max_axes = axes.shape[0]
    trench_type = jnp.clip(_as_scalar_int(trench_type), 0, max_axes)
    rows, cols = jnp.meshgrid(
        jnp.arange(target_map.shape[0], dtype=jnp.float32),
        jnp.arange(target_map.shape[1], dtype=jnp.float32),
        indexing="ij",
    )
    line_denominators = jnp.maximum(
        jnp.linalg.norm(axes[:, :2], axis=1),
        jnp.float32(1e-6),
    )
    line_distances = jnp.abs(
        axes[:, 0, None, None] * cols[None, :, :]
        + axes[:, 1, None, None] * rows[None, :, :]
        + axes[:, 2, None, None]
    ) / line_denominators[:, None, None]

    if records.shape[1] >= 7:
        points = jnp.stack([rows, cols], axis=-1)
        starts = records[:, 3:5]
        ends = records[:, 5:7]
        segment_vectors = ends - starts
        segment_norm_sq = jnp.sum(segment_vectors**2, axis=1)
        start_to_point = points[None, :, :, :] - starts[:, None, None, :]
        projection = jnp.sum(
            start_to_point * segment_vectors[:, None, None, :], axis=-1
        ) / jnp.maximum(segment_norm_sq[:, None, None], jnp.float32(1e-6))
        projection = jnp.clip(
            projection,
            a_min=jnp.float32(0.0),
            a_max=jnp.float32(1.0),
        )
        closest = (
            starts[:, None, None, :]
            + projection[:, :, :, None] * segment_vectors[:, None, None, :]
        )
        segment_distances = jnp.linalg.norm(
            points[None, :, :, :] - closest, axis=-1
        )
        has_segment = jnp.logical_and(
            segment_norm_sq > jnp.float32(1e-6),
            jnp.all(records[:, 3:7] > jnp.float32(-96.0), axis=1),
        )
        ownership_distances = jnp.where(
            has_segment[:, None, None],
            segment_distances,
            line_distances,
        )
    else:
        ownership_distances = line_distances

    valid_axes = jnp.arange(max_axes) < trench_type
    ownership_distances = jnp.where(
        valid_axes[:, None, None],
        ownership_distances,
        jnp.float32(jnp.inf),
    )
    minimum_distance = jnp.min(ownership_distances, axis=0)
    nearest_memberships = (
        ownership_distances
        <= minimum_distance[None, :, :] + jnp.float32(0.5)
    )
    if records.shape[1] >= 8:
        half_widths = records[:, 7]
        finite_width = half_widths > jnp.float32(0.0)
        generated_memberships = ownership_distances <= (
            half_widths[:, None, None] + jnp.float32(0.5)
        )
        generated_memberships = jnp.logical_and(
            finite_width[:, None, None], generated_memberships
        )
        fringe_memberships = jnp.logical_and(
            nearest_memberships,
            ownership_distances
            <= (
                half_widths[:, None, None]
                + jnp.float32(TRENCH_MEMBERSHIP_FRINGE_TOLERANCE_TILES)
            ),
        )
        memberships = jnp.where(
            jnp.any(generated_memberships, axis=0)[None, :, :],
            generated_memberships,
            fringe_memberships,
        )
    else:
        memberships = nearest_memberships
    memberships = jnp.logical_and(valid_axes[:, None, None], memberships)
    memberships = jnp.logical_and(
        memberships,
        target_map[None, :, :] < 0,
    )
    bit_values = jnp.left_shift(
        jnp.ones((max_axes,), dtype=jnp.uint8),
        jnp.arange(max_axes, dtype=jnp.uint8),
    )
    return jnp.sum(
        memberships.astype(jnp.uint8) * bit_values[:, None, None],
        axis=0,
        dtype=jnp.uint8,
    )


class GridWorld(NamedTuple):
    """
    Here we define the encoding of the maps.
    - target map
        - 1: must dump here to terminate the episode
        - 0: free
        - -1: must dig here
    - action map
        - -1: dug here during the episode
        - 0: free
        - greater than 0: dumped here
    - dumpability mask
        - 1: can dump
        - 0: can't dump
    - padding mask
        - 0: traversable
        - 1: non traversable
    - traversability mask
        - -1: agent occupancy
        - 0: traversable
        - 1: non traversable
    - last dig mask
        - 1: dug here during previous dig action
        - 0: not dug here during previous dig action
    - local map target positive (contains the sum of all the positive target map tiles in a given workspace)
    - local map target negative (contains the sum of all the negative target map tiles in a given workspace)
    - local map action positive (contains the sum of all the positive action map tiles in a given workspace)
    - local map action negative (contains the sum of all the negative action map tiles in a given workspace)
    - local obstacles map (contains the sum of all the padding mask tiles in a given workspace)
    - local dumpability mask (contains the sum of all the dumpability mask tiles in a given workspace)
    """

    target_map: GridMap
    action_map: GridMap
    padding_mask: GridMap
    dumpability_mask: GridMap
    dumpability_mask_init: GridMap
    last_dig_mask: GridMap
    interaction_mask: GridMap

    trench_axes: Array
    trench_type: jnp.int32  # type of trench (number of branches), or -1 if not a trench
    trench_axis_membership: Array  # uint8 bitmask over generated finite sections
    foundation_border_axes: Array
    foundation_border_type: jnp.int32  # number of foundation border segments, or -1 if unavailable

    # Dummies for wrappers
    static_traversability_base: GridMap = GridMap.dummy_map()
    traversability_mask: GridMap = GridMap.dummy_map()
    reachability_mask: GridMap = GridMap.dummy_map()
    local_map_target_pos: GridMap = GridMap.dummy_map()
    local_map_target_neg: GridMap = GridMap.dummy_map()
    local_map_action_pos: GridMap = GridMap.dummy_map()
    local_map_action_neg: GridMap = GridMap.dummy_map()
    local_map_dumpability: GridMap = GridMap.dummy_map()
    local_map_obstacles: GridMap = GridMap.dummy_map()
    local_map_border_workspace: GridMap = GridMap.dummy_map()
    local_map_edge_alignment_error: GridMap = GridMap.dummy_map()
    local_map_border_diggable: GridMap = GridMap.dummy_map()

    # Additional maps for second agent with "_2" suffix
    traversability_mask_2: GridMap = GridMap.dummy_map()
    local_map_target_pos_2: GridMap = GridMap.dummy_map()
    local_map_target_neg_2: GridMap = GridMap.dummy_map()
    local_map_action_pos_2: GridMap = GridMap.dummy_map()
    local_map_action_neg_2: GridMap = GridMap.dummy_map()
    local_map_dumpability_2: GridMap = GridMap.dummy_map()
    local_map_obstacles_2: GridMap = GridMap.dummy_map()

    # Cached per-episode map: distance to nearest designated dump zone (normalized); optional
    relocation_distance_map: Array = jnp.zeros((1, 1), dtype=jnp.float32)

    @property
    def width(self) -> int:
        return self.target_map.width

    @property
    def height(self) -> int:
        return self.target_map.height

    @property
    def max_traversable_x(self) -> int:
        return (self.padding_mask.map[:, 0] == 0).sum()

    @property
    def max_traversable_y(self) -> int:
        return (self.padding_mask.map[0] == 0).sum()

    @classmethod
    def new(
        cls,
        target_map: Array,
        padding_mask: Array,
        trench_axes: Array,
        trench_type: Array,
        foundation_border_axes: Array,
        foundation_border_type: Array,
        dumpability_mask_init: Array,
        action_map: Array,
        relocation_distance_map_override: Array | None = None,
    ) -> "GridWorld":
        target_map = _as_2d_map(target_map)
        padding_mask = _as_2d_map(padding_mask)
        dumpability_mask_init = _as_2d_map(dumpability_mask_init)
        action_map = _as_2d_map(action_map)
        trench_axes = _as_axes_table(trench_axes)
        trench_type = _as_scalar_int(trench_type)
        foundation_border_axes = _as_axes_table(foundation_border_axes)
        foundation_border_type = _as_scalar_int(foundation_border_type)
        if relocation_distance_map_override is not None:
            relocation_distance_map_override = _as_2d_map(relocation_distance_map_override)

        dynamic_dumpability = compute_dynamic_dumpability(
            dumpability_mask_init,
            action_map,
        )
        trench_axis_membership = compute_trench_axis_membership(
            target_map,
            trench_axes,
            trench_type,
        )
        action_map = GridMap.new(IntLowDim(action_map))
        target_map = GridMap.new(IntLowDim(target_map))
        padding_mask = GridMap.new(IntLowDim(padding_mask))
        static_traversability_base = GridMap.new((padding_mask.map == 1).astype(IntLowDim))
        dumpability_mask_init_gm = GridMap.new(dumpability_mask_init.astype(jnp.bool_))
        dumpability_mask = GridMap.new(dynamic_dumpability)
        last_dig_mask = GridMap.new(jnp.zeros_like(target_map.map, dtype=jnp.bool_))
        interaction_mask = GridMap.new(jnp.zeros_like(target_map.map, dtype=jnp.bool_))
        reachability_mask = GridMap.new(jnp.zeros_like(target_map.map, dtype=IntLowDim))
        relocation_distance_map = jnp.array(relocation_distance_map_override, dtype=jnp.float32) if relocation_distance_map_override is not None else jnp.zeros_like(target_map.map, dtype=jnp.float32)
        
        world = cls(
            target_map=target_map,
            action_map=action_map,
            padding_mask=padding_mask,
            trench_axes=trench_axes,
            trench_type=trench_type,
            trench_axis_membership=trench_axis_membership,
            foundation_border_axes=foundation_border_axes,
            foundation_border_type=foundation_border_type,
            static_traversability_base=static_traversability_base,
            dumpability_mask=dumpability_mask,
            dumpability_mask_init=dumpability_mask_init_gm,
            last_dig_mask=last_dig_mask,
            interaction_mask=interaction_mask,
            reachability_mask=reachability_mask,
            relocation_distance_map=relocation_distance_map,
        )

        return world
