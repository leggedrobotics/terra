import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple, Type
from src.config import BufferConfig, EnvConfig
from src.utils import IntMap, INTMAP_MAX


class GridMap(NamedTuple):
    map: IntMap

    @property
    def width(self) -> int:
        return self.map.shape[0]
    
    @property
    def height(self) -> int:
        self.map.shape[1]
    
    @staticmethod
    def new(map: Array) -> "GridMap":
        assert len(map.shape) == 2
        
        return GridMap(
            map=map  # GridMap.__annotations__["map"](map)
        )

    @staticmethod
    def random_map_one_dig(seed: int, width: int, height: int) -> "GridMap":
        map = jnp.zeros((width, height), dtype=IntMap)
        key = jax.random.PRNGKey(seed)
        x = jax.random.randint(key, (1, ), minval=0, maxval=width - 1)

        key, _ = jax.random.split(key)
        y = jax.random.randint(key, (1, ), minval=0, maxval=height - 1)

        map = map.at[x, y].set(-1)
        return GridMap.new(map)


class GridWorld(NamedTuple):
    target_map: GridMap
    action_map: GridMap

    @property
    def width(self) -> int:
        return self.target_map.width

    @property
    def height(self) -> int:
        return self.target_map.height

    @classmethod
    def new(cls, seed: int, env_cfg: EnvConfig, buf_cfg: BufferConfig) -> "GridWorld":
        assert env_cfg.target_map.width == env_cfg.action_map.width
        assert env_cfg.target_map.height == env_cfg.action_map.height

        target_map = GridMap.random_map_one_dig(
            seed=seed,
            width=env_cfg.target_map.width,
            height=env_cfg.target_map.height
        )
        
        action_map = GridMap.new(
            map=jnp.zeros((env_cfg.action_map.width, env_cfg.action_map.height), dtype=IntMap)
        )
        
        return GridWorld(
            target_map=target_map,
            action_map=action_map
        )
