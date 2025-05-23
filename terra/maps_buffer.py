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
    action_maps: Array  # [map_type, n_maps, W, H]
    n_maps: int  # number of maps for each map type

    immutable_maps_cfg: ImmutableMapsConfig = ImmutableMapsConfig()

    def __hash__(self) -> int:
        return hash((len(self.maps),))

    def __eq__(self, __o: "MapsBuffer") -> bool:
        return len(self.maps) == len(__o.maps)

    @classmethod
    def new(
        cls,
        maps: Array,
        padding_mask: Array,
        trench_axes: Array,
        trench_types: Array,
        dumpability_masks_init: Array,
        action_maps: Array,
    ) -> "MapsBuffer":
        return MapsBuffer(
            maps=maps.astype(IntLowDim),
            padding_mask=padding_mask.astype(IntLowDim),
            dumpability_masks_init=dumpability_masks_init.astype(jnp.bool_),
            trench_axes=trench_axes.astype(jnp.float16),
            trench_types=trench_types,
            n_maps=maps.shape[1],
            action_maps=action_maps.astype(IntLowDim),
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
        action_map = self.action_maps[curriculum_level, idx]
        return map, padding_mask, trench_axes, trench_type, dumpability_mask_init, action_map, key

    @partial(jax.jit, static_argnums=(0,))
    def get_map(self, key: jax.random.PRNGKey, env_cfg) -> Array:
        (
            map,
            padding_mask,
            trench_axes,
            trench_type,
            dumpability_mask_init,
            action_map,
            key,
        ) = self._get_map_from_disk(key, env_cfg)
        # Ensure consistent dtypes for all return values
        trench_type = trench_type.astype(jnp.int32)
        return map, padding_mask, trench_axes, trench_type, dumpability_mask_init, action_map, key

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

def actions_sanity_check(map: Array) -> None:
    valid = np.all((map == 0) | (map == 1) | (map == -1))
    if not valid:
        raise RuntimeError("Loaded actions map is not valid.")


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
    actions = []
    n_loaded_metadata = 0
    trench_type = -1
    # Check if the actions folder exists (only for relocations)
    actions_folder = Path(folder_path) / "actions"
    has_actions = actions_folder.exists()
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
        # Generate an actions map if present, otherwise initialize an empty one
        if has_actions:
            actions_map = np.load(actions_folder / f"img_{i}.npy")
            actions_sanity_check(actions_map)
            actions.append(actions_map)
        else:
            actions.append(np.zeros_like(map, dtype=IntMap))

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
        jnp.array(actions, dtype=IntMap)
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
    actions: list[Array],
    maps_width,
    maps_height,
):
    """
    Pads multiple maps along with their occupancies and dumpability masks.

    Args:
    maps (List[Array]): List of map arrays.
    occupancies (List[Array]): List of occupancy arrays.
    dumpability_masks (List[Array]): List of dumpability mask arrays.
    actions (List[Array]): List of action arrays.
    maps_width (int): Maximum width for padding.
    maps_height (int): Maximum height for padding.

    Returns:
    Tuple[Array, Array, Array]: Padded maps, padding masks, and padded dumpability masks.
    """
    max_w = maps_width
    max_h = maps_height
    padding_mask = []
    maps_padded = []
    dumpability_masks_padded = []
    actions_padded = []
    for m, o, d, a in zip(maps, occupancies, dumpability_masks, actions):
        z, z_mask = _pad_map_array(m, max_w, max_h)
        z_mask[:, : o.shape[1], : o.shape[2]] = o  # use occupancies from dataset
        d_padded = np.zeros(
            (d.shape[0], max_w, max_h),
            dtype=np.bool_,
        )
        d_padded[:, : d.shape[1], : d.shape[2]] = d

        a_padded = np.zeros(
            (a.shape[0], max_w, max_h),
            dtype=IntMap,
        )
        a_padded[:, : a.shape[1], : a.shape[2]] = a

        maps_padded.append(z)
        padding_mask.append(z_mask)
        dumpability_masks_padded.append(d_padded)
        actions_padded.append(a_padded)
    return (
        np.array(maps_padded, dtype=IntMap),
        np.array(padding_mask, dtype=IntMap),
        np.array(dumpability_masks_padded, dtype=jnp.bool_),
        np.array(actions_padded, dtype=IntMap),
    )


def _check_maps(maps: list[Array]) -> tuple[int, int]:
    """
    Checks if the maps have the same dimensions and returns them.

    Args:
    maps (List[Array]): List of map arrays.

    Returns:
    Tuple[int, int]: Width and height of the maps.
    """
    maps_width = maps[0].shape[1]
    maps_height = maps[0].shape[2]
    print(f"Maps width: {maps_width}")
    print(f"Maps height: {maps_height}")
    for m in maps:
        if m.shape[1] != maps_width or m.shape[2] != maps_height:
            raise ValueError("Maps have different dimensions.")
    return maps_width, maps_height


def init_maps_buffer(batch_cfg: BatchConfig, shuffle_maps: bool):
    if os.getenv("DATASET_PATH", "") == "":
        raise RuntimeError("DATASET_PATH not defined, can't load maps from disk.")
    maps_paths = [el["maps_path"] for el in batch_cfg.curriculum_global.levels]
    folder_paths = [str(Path(os.getenv("DATASET_PATH", "")) / el) for el in maps_paths]
    print(f"Loading maps from {folder_paths}.")
    maps_from_disk = []
    occupancies_from_disk = []
    dumpability_masks_init_from_disk = []
    trench_axes_list = []
    trench_types = []
    actions_from_disk = []
    for folder_path in folder_paths:
        (
            maps,
            occupancies,
            trench_axes,
            trench_type,
            dumpability_masks_init,
            actions,
        ) = load_maps_from_disk(folder_path)
        maps_from_disk.append(maps)
        occupancies_from_disk.append(occupancies)
        dumpability_masks_init_from_disk.append(dumpability_masks_init)
        trench_axes_list.append(trench_axes)
        trench_types.append(trench_type)
        actions_from_disk.append(actions)
    maps_width, maps_height = _check_maps(maps_from_disk)
    maps_from_disk_padded, padding_mask, dumpability_masks_init_from_disk_padded, actions_from_disk_padded = _pad_maps(
        maps_from_disk,
        occupancies_from_disk,
        dumpability_masks_init_from_disk,
        actions_from_disk,
        maps_width,
        maps_height,
    )
    unique_shapes = set([trench_axes.shape for trench_axes in trench_axes_list])
    print(f"Unique shapes of trench_axes_list: {unique_shapes}")

    maps_from_disk_padded = jnp.array(maps_from_disk_padded)
    padding_mask = jnp.array(padding_mask)
    dumpability_masks_init_from_disk = jnp.array(dumpability_masks_init_from_disk)
    trench_axes_list = jnp.array(trench_axes_list)
    trench_types = jnp.array(trench_types)
    actions_from_disk_padded = jnp.array(actions_from_disk_padded)
    print(f"Maps shape: {maps_from_disk_padded.shape}.")
    print(f"Padding mask shape: {padding_mask.shape}.")
    print(f"Dumpability mask shape: {dumpability_masks_init_from_disk.shape}.")
    print(f"Trench axes shape: {trench_axes_list.shape}.")
    print(f"Trench types shape: {trench_types.shape}.")
    print(f"Actions shape: {actions_from_disk_padded.shape}.")
    if shuffle_maps:
        # NOTE: this is only for visualization purposes (allows to visualize in a single gif every level of the curriculum)
        print("Shuffling maps between curriculum levels...")
        rng = jax.random.PRNGKey(3333)  # doesn't matter which key, it's used only once
        d0 = maps_from_disk_padded.shape[0]
        d1 = maps_from_disk_padded.shape[1]
        # Reshape
        maps_from_disk_padded = maps_from_disk_padded.reshape(
            (-1, *maps_from_disk_padded.shape[2:])
        )
        padding_mask = padding_mask.reshape((-1, *padding_mask.shape[2:]))
        dumpability_masks_init_from_disk = dumpability_masks_init_from_disk.reshape(
            (-1, *dumpability_masks_init_from_disk.shape[2:])
        )
        trench_axes_list = trench_axes_list.reshape((-1, *trench_axes_list.shape[2:]))
        actions_from_disk_padded = actions_from_disk_padded.reshape(
            (-1, *actions_from_disk_padded.shape[2:])
        )
        # Shuffle
        maps_from_disk_padded = jax.random.permutation(
            rng, maps_from_disk_padded, axis=0
        )
        padding_mask = jax.random.permutation(rng, padding_mask, axis=0)
        dumpability_masks_init_from_disk = jax.random.permutation(
            rng, dumpability_masks_init_from_disk, axis=0
        )
        trench_axes_list = jax.random.permutation(rng, trench_axes_list, axis=0)
        actions_from_disk_padded = jax.random.permutation(
            rng, actions_from_disk_padded, axis=0
        )
        # Reshape back
        maps_from_disk_padded = maps_from_disk_padded.reshape(
            (d0, d1, *maps_from_disk_padded.shape[1:])
        )
        padding_mask = padding_mask.reshape((d0, d1, *padding_mask.shape[1:]))
        dumpability_masks_init_from_disk = dumpability_masks_init_from_disk.reshape(
            (d0, d1, *dumpability_masks_init_from_disk.shape[1:])
        )
        trench_axes_list = trench_axes_list.reshape(
            (d0, d1, *trench_axes_list.shape[1:])
        )
        actions_from_disk_padded = actions_from_disk_padded.reshape(
            (d0, d1, *actions_from_disk_padded.shape[1:])
        )
        print("Maps shuffled.")
    maps_buffer = MapsBuffer.new(
        maps=maps_from_disk_padded,
        padding_mask=padding_mask,
        trench_axes=trench_axes_list,
        trench_types=trench_types,
        dumpability_masks_init=dumpability_masks_init_from_disk,
        action_maps=actions_from_disk_padded,
    )
    # Update batch config with the actual map dimensions
    maps_width = maps_from_disk_padded.shape[2]
    maps_height = maps_from_disk_padded.shape[3]
    assert maps_width == maps_height, "Maps are not square."
    batch_cfg = batch_cfg._replace(
        maps_dims=batch_cfg.maps_dims._replace(maps_edge_length=maps_width)
    )
    return maps_buffer, batch_cfg
