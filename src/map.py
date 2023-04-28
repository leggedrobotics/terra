from typing import NamedTuple

import jax.numpy as jnp

from src.config import EnvConfig
from src.map_generator import GridMap
from src.utils import IntMap


class GridWorld(NamedTuple):
    seed: jnp.uint32

    target_map: GridMap
    action_map: GridMap

    # Dummies for wrappers
    traversability_mask: GridMap = GridMap.dummy_map()
    local_map: GridMap = GridMap.dummy_map()

    @property
    def width(self) -> int:
        return self.target_map.width

    @property
    def height(self) -> int:
        return self.target_map.height

    @classmethod
    def new(cls, seed: int, env_cfg: EnvConfig) -> "GridWorld":
        assert env_cfg.target_map.width == env_cfg.action_map.width
        assert env_cfg.target_map.height == env_cfg.action_map.height

        target_map = GridMap.random_map(
            seed=seed,
            map_type=env_cfg.target_map.type,
            width=env_cfg.target_map.width,
            height=env_cfg.target_map.height,
        )

        action_map = GridMap.new(
            map=jnp.zeros(
                (env_cfg.action_map.width, env_cfg.action_map.height), dtype=IntMap
            )
        )

        return GridWorld(seed=seed, target_map=target_map, action_map=action_map)
