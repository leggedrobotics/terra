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
        map = jnp.zeros((width, height))
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
    def new(cls, env_cfg: EnvConfig, buf_cfg: BufferConfig) -> "GridWorld":
        assert env_cfg.target_map.dims == env_cfg.action_map.dims
        assert env_cfg.traversability_mask_map.dims == env_cfg.action_map.dims

        # TODO make the following more elegant (remove if statements)

        # Target map
        if env_cfg.target_map.type == "HeightMapSingleHole":
            target_map = HeightMapSingleHole.create(torch.tensor(env_cfg.target_map.dims))
        elif env_cfg.target_map.type == "HeightMap":
            target_map = HeightMap.random_map(
                torch.tensor(env_cfg.target_map.dims),
                env_cfg.target_map.low,
                env_cfg.target_map.high
                )
        else:
            raise ValueError(f"Map type {env_cfg.target_map.type} is not supported.")

        # Action map
        if env_cfg.action_map.type == "ZeroHeightMap":
            action_map = ZeroHeightMap.create(torch.tensor(env_cfg.action_map.dims))
        else:
            raise ValueError(f"Map type {env_cfg.action_map.type} is not supported.")
        
        # Traversability mask map
        if env_cfg.traversability_mask_map.type == "FreeTraversabilityMaskMap":
            traversability_mask_map = FreeTraversabilityMaskMap.create(
                torch.tensor(env_cfg.traversability_mask_map.dims)
                )

        return GridWorld(
            GridWorld.__annotations__["target_map"](target_map),
            GridWorld.__annotations__["action_map"](action_map),
            GridWorld.__annotations__["traversability_mask_map"](traversability_mask_map)
        )
