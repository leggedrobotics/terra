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
    trench_types: Array  # [map_type, n_maps], number of trench axes, or -1 if unavailable
    foundation_border_axes: Array  # [map_type, n_maps, n_border_axes_per_map, 3]
    foundation_border_types: Array  # [map_type, n_maps], number of border axes, or -1
    action_maps: Array  # [map_type, n_maps, W, H]
    n_maps: int  # number of maps for each map type
    distance_maps: Array  # [map_type, n_maps, W, H] normalized float32

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
        foundation_border_axes: Array,
        foundation_border_types: Array,
        dumpability_masks_init: Array,
        action_maps: Array,
        distance_maps: Array,
    ) -> "MapsBuffer":
        # PATCH: Set all action_map values of 1 to 5 at load time
        #action_maps = jnp.where(action_maps == 1, 5, action_maps)   #DELETE IF NOT NEEDED ANYMORE
        return MapsBuffer(
            maps=maps.astype(IntLowDim),
            padding_mask=padding_mask.astype(IntLowDim),
            dumpability_masks_init=dumpability_masks_init.astype(jnp.bool_),
            trench_axes=trench_axes.astype(jnp.float16),
            trench_types=trench_types,
            foundation_border_axes=foundation_border_axes.astype(jnp.float16),
            foundation_border_types=foundation_border_types,
            n_maps=maps.shape[1],
            action_maps=action_maps.astype(IntLowDim),
            distance_maps=distance_maps.astype(jnp.float32),
        )

    def _select_map(self, key: jax.random.PRNGKey, env_cfg: EnvConfig) -> Array:
        curriculum_level = env_cfg.curriculum.level
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (), 0, self.n_maps)
        map = self.maps[curriculum_level, idx]
        padding_mask = self.padding_mask[curriculum_level, idx]
        trench_axes = self.trench_axes[curriculum_level, idx]
        trench_type = self.trench_types[curriculum_level, idx]
        foundation_border_axes = self.foundation_border_axes[curriculum_level, idx]
        foundation_border_type = self.foundation_border_types[curriculum_level, idx]
        # make sure is int 32
        trench_type = trench_type.astype(jnp.int32)
        foundation_border_type = foundation_border_type.astype(jnp.int32)
        dumpability_mask_init = self.dumpability_masks_init[curriculum_level, idx]
        action_map = self.action_maps[curriculum_level, idx]
        distance_map = self.distance_maps[curriculum_level, idx]
        return map, padding_mask, trench_axes, trench_type, foundation_border_axes, foundation_border_type, dumpability_mask_init, action_map, distance_map, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_map_from_disk(self, key: jax.random.PRNGKey, env_cfg: EnvConfig) -> Array:
        return self._select_map(key, env_cfg)

    @partial(jax.jit, static_argnums=(0,))
    def get_map(self, key: jax.random.PRNGKey, env_cfg) -> Array:
        (
            map,
            padding_mask,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_map,
            distance_map,
            key,
        ) = self._get_map_from_disk(key, env_cfg)
        # Ensure consistent dtypes for all return values
        trench_type = trench_type.astype(jnp.int32)
        foundation_border_type = foundation_border_type.astype(jnp.int32)
        return map, padding_mask, trench_axes, trench_type, foundation_border_axes, foundation_border_type, dumpability_mask_init, action_map, distance_map, key

    def sample_map(self, key: jax.random.PRNGKey, env_cfg) -> Array:
        (
            map,
            padding_mask,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_map,
            distance_map,
            key,
        ) = self._select_map(key, env_cfg)
        trench_type = trench_type.astype(jnp.int32)
        foundation_border_type = foundation_border_type.astype(jnp.int32)
        return map, padding_mask, trench_axes, trench_type, foundation_border_axes, foundation_border_type, dumpability_mask_init, action_map, distance_map, key

    @partial(jax.jit, static_argnums=(0,))
    def get_map_init(self, key: int, env_cfg):
        return self.get_map(key, env_cfg)

    def sample_map_init(self, key: int, env_cfg):
        return self.sample_map(key, env_cfg)


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
    array = np.asarray(map)
    dtype = array.dtype
    if array.size == 0:
        raise RuntimeError(
            f"Loaded actions map must not be empty; got dtype={dtype}, shape={array.shape}."
        )
    if not np.issubdtype(dtype, np.integer):
        raise RuntimeError(
            "Loaded actions map must use an integer dtype; "
            f"got dtype={dtype}, min={array.min()}, max={array.max()}."
        )

    minimum = int(array.min())
    maximum = int(array.max())
    int_low_dim_max = int(np.iinfo(np.int8).max)
    if minimum < -1 or maximum > int_low_dim_max:
        raise RuntimeError(
            "Loaded actions map values must fit Terra's signed action-map "
            f"range [-1, {int_low_dim_max}]; got dtype={dtype}, "
            f"min={minimum}, max={maximum}."
        )


def _ensure_spatial_2d(array: Array, source: str) -> Array:
    """Normalize singleton-channel maps and reject true non-2D spatial data."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and 1 in (array.shape[0], array.shape[-1]):
        squeezed = np.squeeze(array)
        if squeezed.ndim == 2:
            return squeezed
    raise RuntimeError(
        f"Loaded map array from {source} has shape {array.shape}; expected a 2D "
        "grid or a singleton-channel 3D grid."
    )


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


def load_single_map(map_path: str) -> Array:
    """
    Load a single map and its associated files from the specified path.
    Supports two directory structures:
    
    1. Flat structure (original): Directory containing files directly:
       - image.npy: The map image
       - occupancy.npy: The occupancy map
       - dumpability.npy: The dumpability mask
       - distance.npy: The distance map
       - actions.npy: The actions map (optional)
       - metadata.json: Metadata file containing trench axes (optional)
    
    2. Subdirectory structure (like test_map2): Directory with subdirectories:
       - images/img_1.npy: The map image
       - occupancy/img_1.npy: The occupancy map
       - dumpability/img_1.npy: The dumpability mask
       - distance/img_1.npy: The distance map
       - actions/img_1.npy: The actions map (optional)
       - metadata/map.json: Metadata file containing trench axes (optional)

    Args: map_path: Path to the map files
    Returns: Tuple containing map data in the same format as load_maps_from_disk
    """
    map_path = Path(map_path)

    # Check if subdirectory structure exists
    images_dir = map_path / "images"
    has_subdir_structure = images_dir.exists() and (images_dir / "img_1.npy").exists()

    if has_subdir_structure:
        # Load from subdirectory structure
        image_file = map_path / "images" / "img_1.npy"
        occupancy_file = map_path / "occupancy" / "img_1.npy"
        dumpability_file = map_path / "dumpability" / "img_1.npy"
        distance_file = map_path / "distance" / "img_1.npy"
        actions_file = map_path / "actions" / "img_1.npy"
        metadata_file = map_path / "metadata" / "map.json"
    else:
        # Load from flat structure (original behavior)
        image_file = map_path / "image.npy"
        occupancy_file = map_path / "occupancy.npy"
        dumpability_file = map_path / "dumpability.npy"
        distance_file = map_path / "distance.npy"
        actions_file = map_path / "actions.npy"
        metadata_file = map_path / "metadata.json"

    # Load map
    image = _ensure_spatial_2d(np.load(image_file), str(image_file))
    map_sanity_check(image)

    # Load occupancy
    occupancy = _ensure_spatial_2d(np.load(occupancy_file), str(occupancy_file))
    occupancy_sanity_check(occupancy)

    # Load dumpability mask
    dumpability_mask_init = _ensure_spatial_2d(
        np.load(dumpability_file), str(dumpability_file)
    )
    dumpability_sanity_check(dumpability_mask_init)

    # Load distance map
    distance_map_init = _ensure_spatial_2d(np.load(distance_file), str(distance_file))

    # Check if actions map exists
    if actions_file.exists():
        actions_map = _ensure_spatial_2d(np.load(actions_file), str(actions_file))
        actions_sanity_check(actions_map)
    else:
        actions_map = np.zeros_like(image, dtype=IntMap)

    # Try to load metadata
    max_trench_type = 3
    max_foundation_border_type = 64
    trench_axes = -97.0 * np.ones((max_trench_type, 3))  # Default values
    trench_type = -1
    foundation_border_axes = -97.0 * np.ones((max_foundation_border_type, 3))
    foundation_border_type = -1
    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
        trench_ax = metadata.get("axes_ABC", [])
        if len(trench_ax) > 0:
            metadata_sanity_check(trench_ax[0])
            trench_ax = [[el["A"], el["B"], el["C"]] for el in trench_ax]
            trench_type = len(trench_ax)
        while len(trench_ax) < max_trench_type:
            trench_ax.append([-97, -97, -97])
        trench_axes = np.array(trench_ax)
        foundation_ax = metadata.get("foundation_border_axes_ABC", [])
        if len(foundation_ax) > 0:
            metadata_sanity_check(foundation_ax[0])
            foundation_ax = [[el["A"], el["B"], el["C"]] for el in foundation_ax]
            foundation_border_type = len(foundation_ax)
            if len(foundation_ax) > max_foundation_border_type:
                foundation_ax = foundation_ax[:max_foundation_border_type]
                foundation_border_type = max_foundation_border_type
            while len(foundation_ax) < max_foundation_border_type:
                foundation_ax.append([-97, -97, -97])
            foundation_border_axes = np.array(foundation_ax)
    except:
        print(f"No metadata found for given map: {map_path}.")

    # Convert to single-element arrays
    maps = jnp.array([image], dtype=IntMap)
    occupancies = jnp.array([occupancy], dtype=IntMap)
    trench_axes = jnp.array([trench_axes])
    trench_types = jnp.array([trench_type], dtype=jnp.int32)
    foundation_border_axes = jnp.array([foundation_border_axes])
    dumpability_masks_init = jnp.array([dumpability_mask_init], dtype=jnp.bool_)
    actions = jnp.array([actions_map], dtype=IntMap)
    distances = jnp.array([distance_map_init], dtype=jnp.float32)

    return (
        maps,
        occupancies,
        trench_axes,
        trench_types,
        foundation_border_axes,
        foundation_border_type,
        dumpability_masks_init,
        actions,
        distances,
    )


def load_maps_from_disk(folder_path: str, require_trench_metadata: bool = False) -> Array:
    # Set the max number of branches the trench has
    max_trench_type = 3
    max_foundation_border_type = 64

    dataset_size = int(os.getenv("DATASET_SIZE", -1))
    if dataset_size <= 0:
        raise RuntimeError("DATASET_SIZE must be > 0.")
    maps = []
    occupancies = []
    dumpability_masks_init = []
    trench_axes = []
    trench_types = []
    foundation_border_axes = []
    foundation_border_types = []
    actions = []
    distances = []
    n_loaded_metadata = 0
    # Check if the actions folder exists (only for relocations)
    actions_folder = Path(folder_path) / "actions"
    has_actions = actions_folder.exists()
    # Track if we found any distance maps at all
    found_any_distance = False
    images_dir = Path(folder_path) / "images"
    image_files = sorted(images_dir.glob("img_*.npy"))
    if not image_files:
        raise RuntimeError(f"No image files found in {images_dir}; expected img_*.npy.")

    available_indices = []
    for f in image_files:
        stem = f.stem
        try:
            available_indices.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    available_indices = sorted(available_indices)
    if not available_indices:
        raise RuntimeError(f"Could not parse any numeric indices from {images_dir}/img_*.npy.")

    selected_indices = available_indices[:dataset_size]
    if len(selected_indices) < dataset_size:
        print(
            f"Warning: requested DATASET_SIZE={dataset_size}, but found only "
            f"{len(selected_indices)} image files in {images_dir}. Using available files."
        )

    for i in tqdm(selected_indices, desc="Data Loader"):
        image_path = Path(folder_path) / "images" / f"img_{i}.npy"
        map = _ensure_spatial_2d(np.load(image_path), str(image_path))
        map_sanity_check(map)
        occupancy_path = Path(folder_path) / "occupancy" / f"img_{i}.npy"
        occupancy = _ensure_spatial_2d(np.load(occupancy_path), str(occupancy_path))
        occupancy_sanity_check(occupancy)
        dumpability_path = Path(folder_path) / "dumpability" / f"img_{i}.npy"
        dumpability_mask_init = _ensure_spatial_2d(
            np.load(dumpability_path), str(dumpability_path)
        )
        dumpability_sanity_check(dumpability_mask_init)
        maps.append(map)
        occupancies.append(occupancy)
        dumpability_masks_init.append(dumpability_mask_init)
        # Generate an actions map if present, otherwise initialize an empty one
        if has_actions:
            actions_path = actions_folder / f"img_{i}.npy"
            actions_map = _ensure_spatial_2d(np.load(actions_path), str(actions_path))
            actions_sanity_check(actions_map)
            actions.append(actions_map)
        else:
            actions.append(np.zeros_like(map, dtype=IntMap))

        # Load distance map (optional). Warn if missing and fill zeros so shapes match
        distance_file = Path(folder_path) / "distance" / f"img_{i}.npy"
        if distance_file.exists():
            try:
                dist_map = _ensure_spatial_2d(np.load(distance_file), str(distance_file))
                if dist_map.shape != map.shape:
                    print(f"Warning: distance map shape mismatch for {distance_file}, expected {map.shape}, got {dist_map.shape}; filling zeros.")
                    dist_map = np.zeros_like(map, dtype=np.float32)
                found_any_distance = True
            except Exception as e:
                print(f"Warning: failed to load distance map {distance_file}: {e}; filling zeros.")
                dist_map = np.zeros_like(map, dtype=np.float32)
        else:
            print(f"Warning: missing distance map {distance_file}, filling zeros.")
            dist_map = np.zeros_like(map, dtype=np.float32)
        distances.append(dist_map.astype(np.float32))

        # Metadata needs to be loaded only for trenches (A, B, C coefficients)
        metadata_path = Path(folder_path) / "metadata" / f"trench_{i}.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            trench_ax = metadata.get("axes_ABC", [])
            trench_type = -1
            if len(trench_ax) > 0:
                metadata_sanity_check(trench_ax[0])
                trench_ax = [[el["A"], el["B"], el["C"]] for el in trench_ax]
                trench_type = len(trench_ax)
            else:
                trench_ax = []

            # Fill in with dummies the remaining metadata to reach the standard shape
            while len(trench_ax) < max_trench_type:
                trench_ax.append([-97, -97, -97])

            trench_axes.append(trench_ax)
            trench_types.append(trench_type)
            foundation_ax = metadata.get("foundation_border_axes_ABC", [])
            foundation_border_type = -1
            if len(foundation_ax) > 0:
                metadata_sanity_check(foundation_ax[0])
                foundation_ax = [[el["A"], el["B"], el["C"]] for el in foundation_ax]
                foundation_border_type = len(foundation_ax)
            if len(foundation_ax) > max_foundation_border_type:
                foundation_ax = foundation_ax[:max_foundation_border_type]
                foundation_border_type = max_foundation_border_type
            while len(foundation_ax) < max_foundation_border_type:
                foundation_ax.append([-97, -97, -97])
            foundation_border_axes.append(foundation_ax)
            foundation_border_types.append(foundation_border_type)
            n_loaded_metadata += 1
        else:
            # Missing metadata for this map.
            if require_trench_metadata:
                # If the curriculum/level requires trench metadata (e.g. trench rewards),
                # raise immediately so the user can provide the metadata.
                raise RuntimeError(
                    f"Missing trench metadata file {metadata_path} but trench metadata is required for this level."
                )
            # Otherwise use defaults and continue. If some metadata were loaded
            # earlier this is allowed (we treat missing as defaults).
            if n_loaded_metadata > 0:
                print(f"Warning: missing metadata file {metadata_path}; using defaults for img_{i}.")
            trench_axes.append([[-97, -97, -97] for _ in range(max_trench_type)])
            trench_types.append(-1)
            foundation_border_axes.append(
                [[-97, -97, -97] for _ in range(max_foundation_border_type)]
            )
            foundation_border_types.append(-1)
            continue
    # If no distance maps were found at all, raise an error (strict behavior)
    if not found_any_distance:
        raise RuntimeError(f"No distance maps found in {Path(folder_path) / 'distance'}; please provide distance/img_*.npy files.")
    loaded_count = len(maps)
    print(f"Loaded {loaded_count} maps from {folder_path}.")
    if n_loaded_metadata > 0:
        print(f"Loaded {n_loaded_metadata} metadata files from {folder_path}.")
    else:
        trench_axes = -97.0 * jnp.ones(
            (
                loaded_count,
                3,
                3,
            )
        )
        trench_types = -1 * jnp.ones((loaded_count,), dtype=jnp.int32)
        foundation_border_axes = -97.0 * jnp.ones(
            (
                loaded_count,
                max_foundation_border_type,
                3,
            )
        )
        foundation_border_types = -1 * jnp.ones((loaded_count,), dtype=jnp.int32)
        print(f"Did NOT load any metadata file from {folder_path}.")
    return (
        jnp.array(maps, dtype=IntMap),
        jnp.array(occupancies, dtype=IntMap),
        jnp.array(trench_axes),
        jnp.array(trench_types, dtype=jnp.int32),
        jnp.array(foundation_border_axes),
        jnp.array(foundation_border_types, dtype=jnp.int32),
        jnp.array(dumpability_masks_init, dtype=jnp.bool_),
        jnp.array(actions, dtype=IntMap),
        jnp.array(distances, dtype=jnp.float32),
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


def init_maps_buffer(batch_cfg: BatchConfig, shuffle_maps: bool, single_map_path: str = None):
    if single_map_path is not None:
        print(f"Loading single map from {single_map_path}")
        maps_from_disk = []
        occupancies_from_disk = []
        dumpability_masks_init_from_disk = []
        trench_axes_list = []
        trench_types = []
        foundation_border_axes_list = []
        foundation_border_types = []
        actions_from_disk = []
        distances_from_disk = []

        # Load the single map (now also returns a distance map)
        (
            maps,
            occupancies,
            trench_axes,
            trench_type,
            foundation_border_axes,
            foundation_border_type,
            dumpability_masks_init,
            actions,
            distances,
        ) = load_single_map(single_map_path)

        # Repeat the map for each curriculum level
        num_levels = len(batch_cfg.curriculum_global.levels)
        maps_from_disk = [maps] * num_levels
        occupancies_from_disk = [occupancies] * num_levels
        dumpability_masks_init_from_disk = [dumpability_masks_init] * num_levels
        trench_axes_list = [trench_axes] * num_levels
        trench_types = [trench_type] * num_levels
        foundation_border_axes_list = [foundation_border_axes] * num_levels
        foundation_border_types = [jnp.array([foundation_border_type], dtype=jnp.int32)] * num_levels
        actions_from_disk = [actions] * num_levels
        distances_from_disk = [distances] * num_levels
    else:
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
        foundation_border_axes_list = []
        foundation_border_types = []
        actions_from_disk = []
        distances_from_disk = []
        for idx, folder_path in enumerate(folder_paths):
            (
                maps,
                occupancies,
                trench_axes,
                trench_types_for_level,
                foundation_border_axes,
                foundation_border_type,
                dumpability_masks_init,
                actions,
                distances,
            ) = load_maps_from_disk(
                folder_path,
                require_trench_metadata=batch_cfg.curriculum_global.levels[idx].get(
                    "apply_trench_rewards", False
                ),
            )
            maps_from_disk.append(maps)
            occupancies_from_disk.append(occupancies)
            dumpability_masks_init_from_disk.append(dumpability_masks_init)
            trench_axes_list.append(trench_axes)
            trench_types.append(trench_types_for_level)
            foundation_border_axes_list.append(foundation_border_axes)
            foundation_border_types.append(foundation_border_type)
            actions_from_disk.append(actions)
            distances_from_disk.append(distances)
    # Apply padding to ALL maps (unified logic like single-agent)
    maps_width, maps_height = _check_maps(maps_from_disk)
    maps_from_disk_padded, padding_mask, dumpability_masks_init_from_disk_padded, actions_from_disk_padded = _pad_maps(
        maps_from_disk,
        occupancies_from_disk,
        dumpability_masks_init_from_disk,
        actions_from_disk,
        maps_width,
        maps_height,
    )
    # Distance maps don't require padding masks; pad to same shape
    distances_padded = []
    for d in distances_from_disk:
        z = np.zeros((d.shape[0], maps_width, maps_height), dtype=np.float32)
        z[:, : d.shape[1], : d.shape[2]] = d
        distances_padded.append(z)

    unique_shapes = set([trench_axes.shape for trench_axes in trench_axes_list])
    print(f"Unique shapes of trench_axes_list: {unique_shapes}")

    maps_from_disk_padded = jnp.array(maps_from_disk_padded)
    padding_mask = jnp.array(padding_mask)
    dumpability_masks_init_from_disk = jnp.array(dumpability_masks_init_from_disk_padded)
    trench_axes_list = jnp.array(trench_axes_list)
    trench_types = jnp.array(trench_types)
    foundation_border_axes_list = jnp.array(foundation_border_axes_list)
    foundation_border_types = jnp.array(foundation_border_types)
    actions_from_disk_padded = jnp.array(actions_from_disk_padded)
    distances_padded = jnp.array(distances_padded)
    print(f"Maps shape: {maps_from_disk_padded.shape}.")
    print(f"Padding mask shape: {padding_mask.shape}.")
    print(f"Dumpability mask shape: {dumpability_masks_init_from_disk.shape}.")
    print(f"Trench axes shape: {trench_axes_list.shape}.")
    print(f"Trench types shape: {trench_types.shape}.")
    print(f"Foundation border axes shape: {foundation_border_axes_list.shape}.")
    print(f"Foundation border types shape: {foundation_border_types.shape}.")
    print(f"Actions shape: {actions_from_disk_padded.shape}.")
    print(f"Distance maps shape: {distances_padded.shape}.")
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
        trench_types = trench_types.reshape((-1,))
        foundation_border_axes_list = foundation_border_axes_list.reshape(
            (-1, *foundation_border_axes_list.shape[2:])
        )
        foundation_border_types = foundation_border_types.reshape((-1,))
        actions_from_disk_padded = actions_from_disk_padded.reshape(
            (-1, *actions_from_disk_padded.shape[2:])
        )
        distances_padded = distances_padded.reshape((-1, *distances_padded.shape[2:]))
        # Shuffle
        maps_from_disk_padded = jax.random.permutation(
            rng, maps_from_disk_padded, axis=0
        )
        padding_mask = jax.random.permutation(rng, padding_mask, axis=0)
        dumpability_masks_init_from_disk = jax.random.permutation(
            rng, dumpability_masks_init_from_disk, axis=0
        )
        trench_axes_list = jax.random.permutation(rng, trench_axes_list, axis=0)
        trench_types = jax.random.permutation(rng, trench_types, axis=0)
        foundation_border_axes_list = jax.random.permutation(
            rng, foundation_border_axes_list, axis=0
        )
        foundation_border_types = jax.random.permutation(
            rng, foundation_border_types, axis=0
        )
        actions_from_disk_padded = jax.random.permutation(
            rng, actions_from_disk_padded, axis=0
        )
        distances_padded = jax.random.permutation(
            rng, distances_padded, axis=0
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
        trench_types = trench_types.reshape((d0, d1))
        foundation_border_axes_list = foundation_border_axes_list.reshape(
            (d0, d1, *foundation_border_axes_list.shape[1:])
        )
        foundation_border_types = foundation_border_types.reshape((d0, d1))
        actions_from_disk_padded = actions_from_disk_padded.reshape(
            (d0, d1, *actions_from_disk_padded.shape[1:])
        )
        distances_padded = distances_padded.reshape(
            (d0, d1, *distances_padded.shape[1:])
        )
        print("Maps shuffled.")
    maps_buffer = MapsBuffer.new(
        maps=maps_from_disk_padded,
        padding_mask=padding_mask,
        trench_axes=trench_axes_list,
        trench_types=trench_types,
        foundation_border_axes=foundation_border_axes_list,
        foundation_border_types=foundation_border_types,
        dumpability_masks_init=dumpability_masks_init_from_disk,
        action_maps=actions_from_disk_padded,
        distance_maps=distances_padded,
    )
    # Update batch config with the actual map dimensions
    maps_width = maps_from_disk_padded.shape[2]
    maps_height = maps_from_disk_padded.shape[3]
    assert maps_width == maps_height, "Maps are not square."
    batch_cfg = batch_cfg._replace(
        maps_dims=batch_cfg.maps_dims._replace(maps_edge_length=maps_width)
    )
    return maps_buffer, batch_cfg
