from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from terra.map_generator import GridMap
from terra.utils import IntLowDim


class GridWorld(NamedTuple):
    target_map: GridMap
    action_map: GridMap
    padding_mask: GridMap
    dig_map: GridMap  # map where the dig action is applied before being applied to the action map (at dump time).
    dumpability_mask: GridMap
    dumpability_mask_init: GridMap

    trench_axes: Array
    trench_type: jnp.int32  # type of trench (number of branches), or -1 if not a trench

    # Dummies for wrappers
    traversability_mask: GridMap = GridMap.dummy_map()
    local_map_target: GridMap = GridMap.dummy_map()
    local_map_action: GridMap = GridMap.dummy_map()

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
        dumpability_mask_init: Array,
    ) -> "GridWorld":
        action_map = GridMap.new(jnp.zeros_like(target_map, dtype=IntLowDim))
        dig_map = GridMap.new(jnp.zeros_like(target_map, dtype=IntLowDim))

        target_map = GridMap.new(IntLowDim(target_map))

        padding_mask = GridMap.new(IntLowDim(padding_mask))

        dumpability_mask_init_gm = GridMap.new(dumpability_mask_init.astype(jnp.bool_))
        dumpability_mask = GridMap.new(dumpability_mask_init.astype(jnp.bool_))

        world = GridWorld(
            target_map=target_map,
            action_map=action_map,
            padding_mask=padding_mask,
            dig_map=dig_map,
            trench_axes=trench_axes,
            trench_type=trench_type,
            dumpability_mask=dumpability_mask,
            dumpability_mask_init=dumpability_mask_init_gm,
        )

        return world
