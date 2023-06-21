from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.map_generator import GridMap
from terra.map_generator import MapType


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

    maps: Array  # [map_type, n_maps, W, H]
    padding_mask: Array  # [map_type, n_maps, W, H]
    n_maps: int  # number of maps for each map type

    map_params: TmpMapParams = TmpMapParams()  # TODO remove

    map_types_from_disk: Array = jnp.array(
        [
            MapType.OPENSTREET_2_DIG_DUMP,
            MapType.OPENSTREET_3_DIG_DIG_DUMP,
        ]
    )

    def __hash__(self) -> int:
        return hash((len(self.maps),))

    def __eq__(self, __o: "MapsBuffer") -> bool:
        return len(self.maps) == len(__o.maps)

    @classmethod
    def new(cls, maps: Array, padding_mask: Array) -> "MapsBuffer":
        return MapsBuffer(maps=maps, padding_mask=padding_mask, n_maps=maps.shape[1])

    @partial(jax.jit, static_argnums=(0,))
    def _get_map_from_disk(self, key: jax.random.KeyArray, env_cfg) -> Array:
        # maps_a = jax.lax.switch(env_cfg.target_map.map_dof, self.maps)
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (), 0, self.n_maps)
        map = self.maps[env_cfg.target_map.map_dof, idx]
        padding_mask = self.padding_mask[env_cfg.target_map.map_dof, idx]
        return map, padding_mask, key

    def _procedurally_generate_map(self, key: jax.random.KeyArray, env_cfg) -> Array:
        key, subkey = jax.random.split(key)
        width = env_cfg.maps.max_width
        height = env_cfg.maps.max_height
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

    @partial(jax.jit, static_argnums=(0,))
    def get_map(self, key: jax.random.KeyArray, env_cfg) -> Array:
        # map_type = env_cfg.target_map.type
        # map, key = jax.lax.cond(
        #     jnp.any(jnp.isin(jnp.array([map_type]), self.map_types_from_disk)),
        #     partial(self._get_map_from_disk, maps_a=self.maps[0]),
        #     # partial(self._get_map_from_disk, maps_a=self.maps[env_cfg.target_map.map_dof]),
        #     partial(self._procedurally_generate_map, env_cfg=env_cfg),
        #     key,
        # )

        # TODO include procedural maps
        map, padding_mask, key = self._get_map_from_disk(key, env_cfg)
        return map, padding_mask, key

    @partial(jax.jit, static_argnums=(0,))
    def get_map_init(self, seed: int, env_cfg):
        key = jax.random.PRNGKey(seed)
        return self.get_map(key, env_cfg)
