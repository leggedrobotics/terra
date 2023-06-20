from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from terra.map_generator import GridMap
from terra.utils import IntMap


class GridWorld(NamedTuple):
    target_map: GridMap
    action_map: GridMap
    padding_mask: GridMap

    # Dummies for wrappers
    traversability_mask: GridMap = GridMap.dummy_map()
    local_map_target: GridMap = GridMap.dummy_map()
    local_map_action: GridMap = GridMap.dummy_map()

    @property
    def width(self) -> int:
        assert self.target_map.width == self.action_map.width
        return self.target_map.width

    @property
    def height(self) -> int:
        assert self.target_map.height == self.action_map.height
        return self.target_map.height

    @classmethod
    def new(cls, target_map: Array, padding_mask: Array) -> "GridWorld":
        action_map = GridMap.new(jnp.zeros_like(target_map, dtype=IntMap))

        target_map = GridMap.new(IntMap(target_map))

        padding_mask = GridMap.new(IntMap(padding_mask))  # TODO IntLowDim?

        world = GridWorld(
            target_map=target_map, action_map=action_map, padding_mask=padding_mask
        )

        return world
