import os
import yaml
import json
import math
import numpy as np
import cv2
import skimage
from pathlib import Path
from terra.env_generation.procedural_data import (
    generate_trenches_v2,
    add_obstacles,
    add_non_dumpables,
    save_or_display_image,
    convert_numpy,
    convert_terra_pad_to_color,
)
from terra.env_generation.convert_to_terra import (
    _convert_dumpability_to_terra,
    _convert_img_to_terra,
    _convert_occupancy_to_terra,
)
import terra.env_generation.convert_to_terra as convert_to_terra
from terra.env_generation.utils import _get_img_mask, color_dict
from terra.env_generation.procedural_squares import generate_squares
import os
import yaml

# Define package directory at module level
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_procedural_trenches(main_folder, config):
    # Load configurations from YAML
    resolution = config["resolution"]
    trenches_config = config["trenches"]
    difficulty_levels = trenches_config["difficulty_levels"]

    # Fix for loading tuples/lists correctly
    trenches_per_level = config["trenches"]["trenches_per_level"]
    corrected_trenches_per_level = [tuple(level) for level in trenches_per_level]

    n_imgs = config["n_imgs"]

    # Load new configurations for obstacles and non-dumpables
    n_obs_min = trenches_config["n_obs_min"]
    n_obs_max = trenches_config["n_obs_max"]
    size_obstacle_min = trenches_config["size_obstacle_min"]
    size_obstacle_max = trenches_config["size_obstacle_max"]
    n_nodump_min = trenches_config["n_nodump_min"]
    n_nodump_max = trenches_config["n_nodump_max"]
    size_nodump_min = trenches_config["size_nodump_min"]
    size_nodump_max = trenches_config["size_nodump_max"]

    for level, n_trenches in zip(difficulty_levels, corrected_trenches_per_level):
        save_folder = os.path.join(main_folder, "trenches", level)
        os.makedirs(save_folder, exist_ok=True)

        # Updated to use new configuration structure
        trench_dims_config = trenches_config["trench_dims"][level]
        trench_dims_min_ratio = trench_dims_config["min_ratio"]
        trench_dims_max_ratio = trench_dims_config["max_ratio"]

        trench_dims_min = (
            max(1, int(trench_dims_min_ratio[0] * trenches_config["img_edge_min"])),
            max(1, int(trench_dims_min_ratio[1] * trenches_config["img_edge_max"])),
        )
        trench_dims_max = (
            max(1, int(trench_dims_max_ratio[0] * trenches_config["img_edge_min"])),
            max(1, int(trench_dims_max_ratio[1] * trenches_config["img_edge_max"])),
        )

        generate_trenches_v2(
            n_imgs,
            trenches_config["img_edge_min"],
            trenches_config["img_edge_max"],
            trench_dims_min,
            trench_dims_max,
            n_trenches,  # Fixed to correctly pass the tuple/list
            resolution,
            save_folder,
            n_obs_min,
            n_obs_max,
            size_obstacle_min,
            size_obstacle_max,
            n_nodump_min,
            n_nodump_max,
            size_nodump_min,
            size_nodump_max,
        )


def create_foundations(config, 
                      n_obs_min=1,
                      n_obs_max=3,
                      size_obstacle_min=6,
                      size_obstacle_max=8,
                      n_nodump_min=1,
                      n_nodump_max=3,
                      size_nodump_min=8,
                      size_nodump_max=10,
                      expansion_factor=1,
                      all_dumpable=False,
                      copy_metadata=True,
                      downsample=True,
                      has_dumpability=False,
                      center_padding=True):
    """
    Creates foundation environments using configurations from a YAML file.

    Parameters:
    - config (dict): Configuration dictionary loaded from YAML file.
    - n_obs_min (int): Minimum number of obstacles to add.
    - n_obs_max (int): Maximum number of obstacles to add.
    - size_obstacle_min (int): Minimum size of obstacles.
    - size_obstacle_max (int): Maximum size of obstacles.
    - n_nodump_min (int): Minimum number of non-dumpable areas.
    - n_nodump_max (int): Maximum number of non-dumpable areas.
    - size_nodump_min (int): Minimum size of non-dumpable areas.
    - size_nodump_max (int): Maximum size of non-dumpable areas.
    - expansion_factor (int): Factor to expand the image by.
    - all_dumpable (bool): Whether all areas should be dumpable.
    - copy_metadata (bool): Whether to copy metadata.
    - downsample (bool): Whether to downsample the image.
    - has_dumpability (bool): Whether the image has dumpability information.
    - center_padding (bool): Whether to center the padding.
    """
    # Extract configuration parameters
    foundation_config = config["foundations"]
    size = foundation_config["max_size"]
    dataset_path = foundation_config["dataset_rel_path"]
    n_imgs = foundation_config["n_imgs"]
    
    # Define save folder using os.path.join
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "foundations")
    
    # Get the full dataset path using os.path.join
    full_dataset_path = os.path.join(PACKAGE_DIR, dataset_path)
    
    # Process foundation images
    max_size = size
    foundations_name = "foundations"
    img_folder = Path(full_dataset_path) / foundations_name / "images"
    metadata_folder = Path(full_dataset_path) / foundations_name / "metadata"
    occupancy_folder = Path(full_dataset_path) / foundations_name / "occupancy"
    dumpability_folder = Path(full_dataset_path) / foundations_name / "dumpability"
    filename_start = sorted(os.listdir(img_folder))[0].split("_")[0]

    for i, fn in enumerate(os.listdir(img_folder)):
        if i >= n_imgs:
            break

        print(f"Processing foundation nr {i + 1}")

        n = int(fn.split(".png")[0].split("_")[1])
        filename = filename_start + f"_{n}.png"
        file_path = img_folder / filename

        occupancy_path = occupancy_folder / filename
        img = cv2.imread(str(file_path))

        occupancy = cv2.imread(str(occupancy_path))

        if has_dumpability:
            dumpability_path = dumpability_folder / filename
            dumpability = cv2.imread(str(dumpability_path))

        if downsample:
            with open(
                metadata_folder / f"{filename.split('.png')[0]}.json"
            ) as json_file:
                metadata = json.load(json_file)

            # Calculate downsample factors based on max_size
            downsample_factor_w = int(max(1, math.ceil(img.shape[1] / max_size)) * 1.5)
            downsample_factor_h = int(max(1, math.ceil(img.shape[0] / max_size)) * 1.5)

            img_downsampled = skimage.measure.block_reduce(
                img, (downsample_factor_h, downsample_factor_w, 1), np.max
            )
            img = img_downsampled
            occupancy_downsampled = skimage.measure.block_reduce(
                occupancy, (downsample_factor_h, downsample_factor_w, 1), np.min, cval=0
            )
            occupancy = occupancy_downsampled
            if has_dumpability:
                dumpability_downsampled = skimage.measure.block_reduce(
                    dumpability,
                    (downsample_factor_h, downsample_factor_w, 1),
                    np.min,
                    cval=0,
                )
                dumpability = dumpability_downsampled

        # assert img_downsampled.shape[:-1] == occupancy_downsampled.shape
        img_terra = _convert_img_to_terra(img, all_dumpable)

        # Pad to max size
        if center_padding:
            xdim = max_size - img_terra.shape[0]
            ydim = max_size - img_terra.shape[1]
            # Note: applying full dumping tiles for the centered version
            img_terra_pad = np.ones((max_size, max_size), dtype=img_terra.dtype)
            img_terra_pad[
                xdim // 2 : max_size - (xdim - xdim // 2),
                ydim // 2 : max_size - (ydim - ydim // 2),
            ] = img_terra
            # Note: applying no occupancy for the centered version (mismatch with Terra env)
            img_terra_occupancy = np.zeros((max_size, max_size), dtype=np.bool_)
            img_terra_occupancy[
                xdim // 2 : max_size - (xdim - xdim // 2),
                ydim // 2 : max_size - (ydim - ydim // 2),
            ] = _convert_occupancy_to_terra(occupancy)
            if has_dumpability:
                img_terra_dumpability = np.zeros((max_size, max_size), dtype=np.bool_)
                img_terra_dumpability[
                    xdim // 2 : max_size - (xdim - xdim // 2),
                    ydim // 2 : max_size - (ydim - ydim // 2),
                ] = _convert_dumpability_to_terra(dumpability)
        else:
            img_terra_pad = np.zeros((max_size, max_size), dtype=img_terra.dtype)
            img_terra_pad[: img_terra.shape[0], : img_terra.shape[1]] = img_terra
            img_terra_occupancy = np.ones((max_size, max_size), dtype=np.bool_)
            img_terra_occupancy[: occupancy.shape[0], : occupancy.shape[1]] = (
                _convert_occupancy_to_terra(occupancy)
            )
            if has_dumpability:
                img_terra_dumpability = np.zeros((max_size, max_size), dtype=np.bool_)
                img_terra_dumpability[
                    : dumpability.shape[0], : dumpability.shape[1]
                ] = _convert_dumpability_to_terra(dumpability)

        img_terra_pad = img_terra_pad.repeat(expansion_factor, 0).repeat(
            expansion_factor, 1
        )
        img_terra_pad = convert_terra_pad_to_color(img_terra_pad, color_dict)
        dumping_image = np.zeros(
            (img_terra_pad.shape[0], img_terra_pad.shape[1], 3), dtype=np.uint8
        )
        corner_dump = np.random.randint(0, 4)
        w, h = img_terra_pad.shape[:2]
        if corner_dump == 0:
            dumping_image[0 : int(0.8 * w), :, :] = np.array(color_dict["dumping"])
        elif corner_dump == 1:
            dumping_image[int(0.2 * w) :, :, :] = np.array(color_dict["dumping"])
        elif corner_dump == 2:
            dumping_image[:, int(0.2 * h) :, :] = np.array(color_dict["dumping"])
        elif corner_dump == 3:
            dumping_image[:, : int(0.8 * h), :] = np.array(color_dict["dumping"])
        # add dumping to the image where it's not equal to color_dict["digging"]

        # Create a mask where img_terra_pad is not equal to color_dict["digging"]
        mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)

        # Use the mask to assign values from dumping_image to img_terra_pad
        img_terra_pad[mask] = dumping_image[mask]

        cumulative_mask = np.zeros_like(img_terra_pad, dtype=np.bool_)
        # where the img_terra_pad is [255, 255, 255] set to True across the three channels
        cumulative_mask[img_terra_pad == 255] = True
        occ, cumulative_mask = add_obstacles(
            img_terra_pad,
            cumulative_mask,
            n_obs_min,
            n_obs_max,
            size_obstacle_min,
            size_obstacle_max,
        )

        dmp, cumulative_mask = add_non_dumpables(
            img_terra_pad,
            occ,
            cumulative_mask,
            n_nodump_min,
            n_nodump_max,
            size_nodump_min,
            size_nodump_max,
        )
        save_or_display_image(img_terra_pad, occ, dmp, metadata, save_folder, n)
    
    print("Foundations created successfully.")
