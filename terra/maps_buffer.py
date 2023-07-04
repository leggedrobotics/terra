import os
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from tqdm import tqdm

from terra.config import ImmutableMapsConfig
from terra.config import MapType
from terra.map_generator import GridMap
from terra.utils import IntMap


class MapsBuffer(NamedTuple):
    """
    Handles the retrieval of maps saved on disk,
    and the generation of the procedurally-generated maps.
    """

    maps: Array  # [map_type, n_maps, W, H]
    padding_mask: Array  # [map_type, n_maps, W, H]
    n_maps: int  # number of maps for each map type

    immutable_maps_cfg: ImmutableMapsConfig = ImmutableMapsConfig()

    # Set this array with the DOF you want to be considered (e.g. the first element will be considered as dof=0).
    map_types_from_disk: Array = jnp.array(
        [
            MapType.OPENSTREET_2_DIG_DUMP,
            MapType.OPENSTREET_3_DIG_DIG_DUMP,
            MapType.TRENCHES,
            MapType.FOUNDATIONS,
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

    def _procedurally_generate_map(
        self, key: jax.random.KeyArray, env_cfg, map_type
    ) -> Array:
        key, subkey = jax.random.split(key)
        min_width = self.immutable_maps_cfg.min_width
        min_height = self.immutable_maps_cfg.min_height
        max_width = self.immutable_maps_cfg.max_width
        max_height = self.immutable_maps_cfg.max_height
        element_edge_min = env_cfg.target_map.element_edge_min
        element_edge_max = env_cfg.target_map.element_edge_max
        map, padding_mask, key = GridMap.procedural_map(
            key=subkey,
            min_width=min_width,
            min_height=min_height,
            max_width=max_width,
            max_height=max_height,
            element_edge_min=element_edge_min,
            element_edge_max=element_edge_max,
            map_type=map_type,
        )
        return map, padding_mask, key

    @partial(jax.jit, static_argnums=(0,))
    def get_map(self, key: jax.random.KeyArray, env_cfg) -> Array:
        map_type = env_cfg.target_map.type
        map, padding_mask, key = jax.lax.cond(
            jnp.any(jnp.isin(jnp.array([map_type]), self.map_types_from_disk)),
            self._get_map_from_disk,
            partial(self._procedurally_generate_map, map_type=map_type),
            key,
            env_cfg,
        )

        # TODO include procedural maps
        # map, padding_mask, key = self._get_map_from_disk(key, env_cfg)
        return map, padding_mask, key

    @partial(jax.jit, static_argnums=(0,))
    def get_map_init(self, seed: int, env_cfg):
        key = jax.random.PRNGKey(seed)
        return self.get_map(key, env_cfg)


def load_maps_from_disk(folder_path: str) -> Array:
    dataset_size = int(os.getenv("DATASET_SIZE", -1))
    maps = []
    occupancies = []
    for i in tqdm(range(dataset_size), desc="Data Loader"):
        map = np.load(f"{folder_path}/images/img_{i}.npy")
        occupancy = np.load(f"{folder_path}/occupancy/img_{i}.npy")
        maps.append(map)
        occupancies.append(occupancy)
    print(f"Loaded {dataset_size} maps from {folder_path}.")
    return jnp.array(maps, dtype=IntMap), jnp.array(occupancies, dtype=IntMap)


def map_paths_to_idx(map_paths: list[str]) -> dict[str, int]:
    return {map_paths[idx]: idx for idx in range(len(map_paths))}


def _pad_map_array(m: Array, max_w: int, max_h: int) -> Array:
    """
    Pads the map array and returns the padded maps and the padding masks.
    """
    z = np.zeros((m.shape[0], max_w, max_h), dtype=IntMap)
    z_mask = np.ones((m.shape[0], max_w, max_h), dtype=IntMap)  # 1 for obstacles
    z[:, : m.shape[1], : m.shape[2]] = m
    z_mask[:, : m.shape[1], : m.shape[2]] = np.zeros_like(m)  # 0 for free
    return z, z_mask


def _pad_maps(maps: list[Array], occupancies: list[Array], batch_cfg):
    max_w = batch_cfg.maps.max_width
    max_h = batch_cfg.maps.max_height
    padding_mask = []
    maps_padded = []
    for m, o in zip(maps, occupancies):
        z, z_mask = _pad_map_array(m, max_w, max_h)
        z_mask[: o.shape[0], : o.shape[1]] = o  # use occupancies from dataset
        maps_padded.append(z)
        padding_mask.append(z_mask)

    return np.array(maps_padded, dtype=IntMap), np.array(padding_mask, dtype=IntMap)


def init_maps_buffer(batch_cfg):
    if os.getenv("DATASET_PATH", "") == "":
        print("DATASET_PATH not defined, skipping maps loading from disk...")
        return MapsBuffer.new(
            maps=jnp.zeros((1, batch_cfg.maps.max_width, batch_cfg.maps.max_height)),
            padding_mask=jnp.zeros(
                (1, batch_cfg.maps.max_width, batch_cfg.maps.max_height)
            ),
        )
    folder_paths = [
        str(Path(os.getenv("DATASET_PATH", "")) / el) for el in batch_cfg.maps_paths
    ]
    folder_paths_dict = map_paths_to_idx(folder_paths)
    maps_from_disk = []
    occupancies_from_disk = []
    for folder_path in folder_paths_dict.keys():
        maps, occupancies = load_maps_from_disk(folder_path)
        maps_from_disk.append(maps)
        occupancies_from_disk.append(occupancies)
    maps_from_disk_padded, padding_mask = _pad_maps(
        maps_from_disk, occupancies_from_disk, batch_cfg
    )
    maps_from_disk_padded = jnp.array(maps_from_disk_padded)
    padding_mask = jnp.array(padding_mask)
    return MapsBuffer.new(maps=maps_from_disk_padded, padding_mask=padding_mask)
