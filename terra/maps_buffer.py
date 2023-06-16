from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.map_generator import GridMap
from terra.map_generator import MapType

# from terra.config import EnvConfig


class TmpMapParams(NamedTuple):
    # TODO remove class
    type = 0
    depth = -1
    edge_min = 2
    edge_max = 2


class MapsBuffer(NamedTuple):
    """
    Handles the retrieval of maps saved on disk,
    and the generation of the procedurally-generated maps.
    """

    maps: list[Array]
    key_shuffle: jax.random.KeyArray

    map_params: TmpMapParams = TmpMapParams()  # TODO remove

    map_types_from_disk: Array = jnp.array(
        [
            MapType.OPENSTREET_2_DIG_DUMP,
            MapType.OPENSTREET_3_DIG_DIG_DUMP,
        ]
    )

    @classmethod
    def new(
        self,
        maps: list[Array],
        key: jax.random.KeyArray,
    ) -> "MapsBuffer":
        return MapsBuffer(
            maps=maps,
            key_shuffle=key,
        )

    @jax.jit
    def shuffle(self) -> "MapsBuffer":
        # TODO change logic to handle dict
        key_shuffle, subkey = jax.random.split(self.key_shuffle)
        maps = [jax.random.permutation(subkey, map, 0) for map in self.maps]
        return MapsBuffer.new(
            maps=maps,
            key_shuffle=key_shuffle,
        )

    # @partial(jax.jit, static_argnums=(1,))
    # def sample(self, n_envs: int) -> Array:
    #     # TODO change logic to handle dict
    #     maps = self.maps[:n_envs]
    #     return maps

    def _get_map_from_disk(self, key: jax.random.KeyArray, env_cfg, dof) -> Array:
        # dof = env_cfg.target_map.map_dof
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (), 0, self.maps[dof].shape[0])
        map = self.maps[dof][idx]
        return map, key

    def _procedurally_generate_map(self, key: jax.random.KeyArray, env_cfg) -> Array:
        key, subkey = jax.random.split(key)
        width = env_cfg.target_map.width
        height = env_cfg.target_map.height
        n_clusters = env_cfg.target_map.n_clusters
        n_tiles_per_cluster = env_cfg.target_map.n_tiles_per_cluster
        kernel_size_initial_sampling = env_cfg.target_map.kernel_size_initial_sampling
        map = GridMap.procedural_map(
            key=subkey,
            width=width,
            height=height,
            map_params=self.map_params,
            n_clusters=n_clusters,
            n_tiles_per_cluster=n_tiles_per_cluster,
            kernel_size_initial_sampling=kernel_size_initial_sampling,
        )
        return map, key

    def get_map(self, key: jax.random.KeyArray, env_cfg) -> Array:
        map_type = env_cfg.target_map.type
        map, key = jax.lax.cond(
            jnp.any(jnp.isin(jnp.array([map_type]), self.map_types_from_disk)),
            partial(self._get_map_from_disk, dof=env_cfg.target_map.map_dof),
            self._procedurally_generate_map,
            key,
            env_cfg,
        )

        # TODO include procedural maps
        # map, key = self._get_map_from_disk(key, env_cfg)
        return map, key

    def get_map_init(self, seed: int, env_cfg):
        key = jax.random.PRNGKey(seed)
        return self.get_map(key, env_cfg)

    def __hash__(self) -> int:
        return hash(self.key_shuffle)

    def __eq__(self, __o: "MapsBuffer") -> bool:
        return self.key_shuffle == __o.key
