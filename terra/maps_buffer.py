import json
import os
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from tqdm import tqdm
from typing import Any

from terra.config import ImmutableMapsConfig, BatchConfig, EnvConfig
from terra.settings import IntMap
from terra.settings import IntLowDim


class MapsBuffer(NamedTuple):
    """
    Handles the retrieval of maps saved on disk,
    and the generation of the procedurally-generated maps.
    """

    maps: Array  # [map_type, n_maps, W, H]
    padding_mask: Array  # [map_type, n_maps, W, H]
    dumpability_masks_init: Array  # [map_type, n_maps, W, H]
    trench_axes: Array  # [map_type, n_maps, n_axes_per_map, 3] -- (A, B, C) coefficients for trench axes (-97 if not a trench)
    trench_types: Array  # type of trench (number of branches), or -1 if not a trench
    n_maps: int  # number of maps for each map type

    immutable_maps_cfg: ImmutableMapsConfig = ImmutableMapsConfig()

    def __hash__(self) -> int:
        return hash((len(self.maps),))

    def __eq__(self, __o: "MapsBuffer") -> bool:
        return len(self.maps) == len(__o.maps)

    @classmethod
    def new(
        cls, maps: Array, padding_mask: Array, trench_axes: Array, trench_types: Array, dumpability_masks_init: Array,
    ) -> "MapsBuffer":
        jax.debug.print("trench_axes.shape = {x}", x=trench_axes.shape)
        return MapsBuffer(
            maps=maps.astype(IntLowDim),
            padding_mask=padding_mask.astype(IntLowDim),
            dumpability_masks_init=dumpability_masks_init.astype(jnp.bool_),
            trench_axes=trench_axes.astype(jnp.float16),
            trench_types=trench_types,
            n_maps=maps.shape[1],
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_map_from_disk(self, key: jax.random.PRNGKey, env_cfg: EnvConfig) -> Array:
        curriculum_level = env_cfg.curriculum.level
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (), 0, self.n_maps)
        map = self.maps[curriculum_level, idx]
        padding_mask = self.padding_mask[curriculum_level, idx]
        trench_axes = self.trench_axes[curriculum_level, idx]
        trench_type = self.trench_types[curriculum_level]
        # make sure is int 32
        trench_type = trench_type.astype(jnp.int32)
        dumpability_mask_init = self.dumpability_masks_init[curriculum_level, idx]
        return map, padding_mask, trench_axes, trench_type, dumpability_mask_init, key

    @partial(jax.jit, static_argnums=(0,))
    def get_map(self, key: jax.random.PRNGKey, env_cfg) -> Array:
        map, padding_mask, trench_axes, trench_type, dumpability_mask_init, key = self._get_map_from_disk(key, env_cfg)
        # Ensure consistent dtypes for all return values
        trench_type = trench_type.astype(jnp.int32)
        return map, padding_mask, trench_axes, trench_type, dumpability_mask_init, key


    @partial(jax.jit, static_argnums=(0,))
    def get_map_init(self, key: int, env_cfg):
        return self.get_map(key, env_cfg)


def map_sanity_check(map: Array) -> None:
    valid = np.all((map == 0) | (map == 1) | (map == -1))
    if not valid:
        raise RuntimeError("Loaded target map is not valid.")
    
def occupancy_sanity_check(map: Array) -> None:
    valid = np.all((map == 0) | (map == 1))
    if not valid:
        raise RuntimeError("Loaded occupancy is not valid.")
    
def dumpability_sanity_check(map: Array) -> None:
    valid = np.all((map == 0) | (map == 1))
    if not valid:
        raise RuntimeError("Loaded dumpability mask is not valid.")
    
def metadata_sanity_check(metadata: dict[str, Any]) -> None:
    valid = True
    k = metadata.keys()
    valid &= "A" in k
    valid &= "B" in k
    valid &= "C" in k
    valid &= isinstance(metadata["A"], float)
    valid &= isinstance(metadata["B"], float)
    valid &= isinstance(metadata["C"], float)
    if not valid:
        raise RuntimeError("Loaded metadata is not valid.")

def load_maps_from_disk(folder_path: str) -> Array:
    # Set the max number of branches the trench has
    max_trench_type = 3

    dataset_size = int(os.getenv("DATASET_SIZE", -1))
    maps = []
    occupancies = []
    dumpability_masks_init = []
    trench_axes = []
    n_loaded_metadata = 0
    trench_type = -1
    for i in tqdm(range(1, dataset_size + 1), desc="Data Loader"):
        map = np.load(f"{folder_path}/images/img_{i}.npy")
        map_sanity_check(map)
        occupancy = np.load(f"{folder_path}/occupancy/img_{i}.npy")
        occupancy_sanity_check(occupancy)
        dumpability_mask_init = np.load(f"{folder_path}/dumpability/img_{i}.npy")
        dumpability_sanity_check(dumpability_mask_init)
        maps.append(map)
        occupancies.append(occupancy)
        dumpability_masks_init.append(dumpability_mask_init)

        try:
            # Metadata needs to be loaded only for trenches (A, B, C coefficients)
            with open(f"{folder_path}/metadata/trench_{i}.json") as f:
                trench_ax = json.load(f)["axes_ABC"]
            metadata_sanity_check(trench_ax[0])
            trench_ax = [[el["A"], el["B"], el["C"]] for el in trench_ax]
            trench_type = len(trench_ax)

            # Fill in with dummies the remaining metadata to reach the standard shape
            while len(trench_ax) < max_trench_type:
                trench_ax.append(
                    [
                        -97,
                        -97,
                        -97,
                    ]
                )

            trench_axes.append(trench_ax)
            n_loaded_metadata += 1
        except:
            if n_loaded_metadata > 0:
                raise (RuntimeError("Imported some trench metadata, but one failed."))
            continue
    print(f"Loaded {dataset_size} maps from {folder_path}.")
    if n_loaded_metadata > 0:
        print(f"Loaded {n_loaded_metadata} metadata files from {folder_path}.")
    else:
        trench_axes = -97.0 * jnp.ones(
            (
                dataset_size,
                3,
                3,
            )
        )
        print(f"Did NOT load any metadata file from {folder_path}.")
    return (
        jnp.array(maps, dtype=IntMap),
        jnp.array(occupancies, dtype=IntMap),
        jnp.array(trench_axes),
        trench_type,
        jnp.array(dumpability_masks_init, dtype=jnp.bool_),
    )


def _pad_map_array(m: Array, max_w: int, max_h: int) -> tuple[Array, Array]:
    """
    Pads the map array to dimensions (max_w, max_h) and returns the padded map and padding mask.

    Args:
        m (Array): The input map array.
        max_w (int): The maximum width for padding.
        max_h (int): The maximum height for padding.
    Returns:
        Tuple[Array, Array]: Padded map and padding mask.
    """

    z = np.zeros((m.shape[0], max_w, max_h), dtype=IntMap)
    z_mask = np.ones((m.shape[0], max_w, max_h), dtype=IntMap)  # 1 for obstacles
    z[:, : m.shape[1], : m.shape[2]] = m
    # Set mask to zero where original map is present
    z_mask[:, : m.shape[1], : m.shape[2]] = np.zeros_like(m)  # 0 for free
    return z, z_mask


def _pad_maps(
        maps: list[Array],
        occupancies: list[Array],
        dumpability_masks: list[Array],
        batch_cfg
        ):
    """
    Pads multiple maps along with their occupancies and dumpability masks.

    Args:
    maps (List[Array]): List of map arrays.
    occupancies (List[Array]): List of occupancy arrays.
    dumpability_masks (List[Array]): List of dumpability mask arrays.
    batch_cfg: Configuration object containing map dimensions.
    
    Returns:
    Tuple[Array, Array, Array]: Padded maps, padding masks, and padded dumpability masks.
    """
    max_w = batch_cfg.maps.max_width
    max_h = batch_cfg.maps.max_height
    padding_mask = []
    maps_padded = []
    dumpability_masks_padded = []
    for m, o, d in zip(maps, occupancies, dumpability_masks):
        z, z_mask = _pad_map_array(m, max_w, max_h)
        z_mask[:, : o.shape[1], : o.shape[2]] = o  # use occupancies from dataset
        d_padded = np.zeros((d.shape[0], max_w, max_h,), dtype=np.bool_)
        d_padded[:, : d.shape[1], : d.shape[2]] = d
        maps_padded.append(z)
        padding_mask.append(z_mask)
        dumpability_masks_padded.append(d_padded)

    return (
        np.array(maps_padded, dtype=IntMap),
        np.array(padding_mask, dtype=IntMap),
        np.array(dumpability_masks_padded, dtype=jnp.bool_),
    )


def init_maps_buffer(batch_cfg: BatchConfig):
    if os.getenv("DATASET_PATH", "") == "":
        raise RuntimeError("DATASET_PATH not defined, can't load maps from disk.")
    maps_paths = [el["maps_path"] for el in batch_cfg.curriculum_global.levels]
    folder_paths = [
        str(Path(os.getenv("DATASET_PATH", "")) / el) for el in maps_paths
    ]
    print(f"Loading maps from {folder_paths}.")
    maps_from_disk = []
    occupancies_from_disk = []
    dumpability_masks_init_from_disk = []
    trench_axes_list = []
    trench_types = []
    for folder_path in folder_paths:
        maps, occupancies, trench_axes, trench_type, dumpability_masks_init = load_maps_from_disk(folder_path)
        maps_from_disk.append(maps)
        occupancies_from_disk.append(occupancies)
        dumpability_masks_init_from_disk.append(dumpability_masks_init)
        trench_axes_list.append(trench_axes)
        trench_types.append(trench_type)
    maps_from_disk_padded, padding_mask, dumpability_masks_init_from_disk = _pad_maps(
        maps_from_disk, occupancies_from_disk, dumpability_masks_init_from_disk, batch_cfg
    )
    maps_from_disk_padded = jnp.array(maps_from_disk_padded)
    padding_mask = jnp.array(padding_mask)
    dumpability_masks_init_from_disk = jnp.array(dumpability_masks_init_from_disk)
    trench_axes_list = jnp.array(trench_axes_list)
    trench_types = jnp.array(trench_types)
    return MapsBuffer.new(
        maps=maps_from_disk_padded,
        padding_mask=padding_mask,
        trench_axes=trench_axes_list,
        trench_types=trench_types,
        dumpability_masks_init = dumpability_masks_init_from_disk,
    )
