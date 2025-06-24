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

    interaction_mask_1: GridMap 
    interaction_mask_2: GridMap 

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

    # Additional maps for second agent with "_2" suffix
    traversability_mask_2: GridMap = GridMap.dummy_map()
    local_map_target_pos_2: GridMap = GridMap.dummy_map()
    local_map_target_neg_2: GridMap = GridMap.dummy_map()
    local_map_action_pos_2: GridMap = GridMap.dummy_map()
    local_map_action_neg_2: GridMap = GridMap.dummy_map()
    local_map_dumpability_2: GridMap = GridMap.dummy_map()
    local_map_obstacles_2: GridMap = GridMap.dummy_map()

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
        action_map: Array,
    ) -> "GridWorld":
        action_map = GridMap.new(IntLowDim(action_map))
        target_map = GridMap.new(IntLowDim(target_map))
        padding_mask = GridMap.new(IntLowDim(padding_mask))
        dumpability_mask_init_gm = GridMap.new(dumpability_mask_init.astype(jnp.bool_))
        dumpability_mask = GridMap.new(dumpability_mask_init.astype(jnp.bool_))
        last_dig_mask = GridMap.new(jnp.zeros_like(target_map.map, dtype=jnp.bool_))
        ineteraction_mask_1 = GridMap.new(jnp.zeros_like(target_map.map, dtype=IntLowDim))
        ineteraction_mask_2 = GridMap.new(jnp.zeros_like(target_map.map, dtype=IntLowDim))

        world = cls(
            target_map=target_map,
            action_map=action_map,
            padding_mask=padding_mask,
            trench_axes=trench_axes,
            trench_type=trench_type,
            dumpability_mask=dumpability_mask,
            dumpability_mask_init=dumpability_mask_init_gm,
            last_dig_mask=last_dig_mask,
            interaction_mask_1=ineteraction_mask_1,
            interaction_mask_2=ineteraction_mask_2,
        )

        return world
