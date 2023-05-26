from typing import NamedTuple

import jax
import jax.numpy as jnp

from terra.config import EnvConfig
from terra.map_generator import GridMap
from terra.utils import IntMap


class GridWorld(NamedTuple):
    key: jax.random.KeyArray

    target_map: GridMap
    action_map: GridMap

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
    def new(cls, key: jax.random.KeyArray, env_cfg: EnvConfig) -> "GridWorld":
        action_map = GridMap.new(
            map=jnp.zeros(
                (env_cfg.action_map.width, env_cfg.action_map.height), dtype=IntMap
            )
        )

        target_map, key = GridMap.random_map(
            key=key,
            map_params=env_cfg.target_map.params,
            width=env_cfg.target_map.width,
            height=env_cfg.target_map.height,
        )

        world = GridWorld(key=key, target_map=target_map, action_map=action_map)

        return world, key
