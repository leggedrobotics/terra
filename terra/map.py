from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from terra.map_generator import GridMap
from terra.settings import IntLowDim


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
    - dig map (same as action map but updated on the dig action & before the dump action is complete)
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

    trench_axes: Array
    trench_type: jnp.int32  # type of trench (number of branches), or -1 if not a trench

    # Dummies for wrappers
    traversability_mask: GridMap = GridMap.dummy_map()
    local_map_target_pos: GridMap = GridMap.dummy_map()
    local_map_target_neg: GridMap = GridMap.dummy_map()
    local_map_action_pos: GridMap = GridMap.dummy_map()
    local_map_action_neg: GridMap = GridMap.dummy_map()
    local_map_dumpability: GridMap = GridMap.dummy_map()
    local_map_obstacles: GridMap = GridMap.dummy_map()

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
        target_map = GridMap.new(IntLowDim(target_map))
        padding_mask = GridMap.new(IntLowDim(padding_mask))
        dumpability_mask_init_gm = GridMap.new(dumpability_mask_init.astype(jnp.bool_))
        dumpability_mask = GridMap.new(dumpability_mask_init.astype(jnp.bool_))

        world = cls(
            target_map=target_map,
            action_map=action_map,
            padding_mask=padding_mask,
            trench_axes=trench_axes,
            trench_type=trench_type,
            dumpability_mask=dumpability_mask,
            dumpability_mask_init=dumpability_mask_init_gm,
        )

        return world
