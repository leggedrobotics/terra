import json
import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.signal import convolve2d
from skimage import measure
from tqdm import tqdm
import time

import terra.env_generation.utils as utils
from terra.env_generation.utils import _get_img_mask, color_dict
from terra.env_generation.distance import (
    DEFAULT_REALISTIC_MAX_DISTANCE,
    write_distance_maps,
)


def _convert_img_to_terra(img, all_dumpable=False):
    """
    Converts an image from color_dict RGB convention
    to [-1, 0, 1] Terra convention.
    """
    img = img.astype(np.int16)
    img = np.where(img == np.array(color_dict["digging"]), -1, img)
    img = np.where(img == np.array(color_dict["dumping"]), 1, img)
    if all_dumpable:
        img = np.where(img == np.array(color_dict["neutral"]), 1, img)
    else:
        img = np.where(img == np.array(color_dict["neutral"]), 0, img)
    img = np.where((img != -1) & (img != 1), 0, img)
    img = img[..., 0]  # take only 1 channel
    return img.astype(np.int8)


def _convert_occupancy_to_terra(img):
    img = img.astype(np.int16)
    mask = _get_img_mask(img, np.array(color_dict["obstacle"]))
    img = np.where(mask, 1, 0)
    return img.astype(np.bool_)


def _convert_dumpability_to_terra(img):
    img = img.astype(np.int16)
    mask = _get_img_mask(img, np.array(color_dict["nondumpable"]))
    img = np.where(mask, 0, 1)
    return img.astype(np.bool_)


def _convert_actions_to_terra(img):
    img = img.astype(np.int16)
    mask = _get_img_mask(img, np.array(color_dict["dirt"]))
    img = np.where(mask, 1, 0)
    return img.astype(np.int8)


def _convert_all_imgs_to_terra(
    img_folder,
    metadata_folder,
    occupancy_folder,
    dumpability_folder,
    destination_folder,
    size,
    n_imgs,
    expansion_factor=1,
    all_dumpable=False,
    copy_metadata=True,
    downsample=True,
    has_dumpability=True,
    center_padding=False,
    actions_folder=None,
    generate_distance_maps=True,
    distance_metric="manhattan",
    distance_connectivity=4,
    distance_realistic_max=DEFAULT_REALISTIC_MAX_DISTANCE,
    distance_obstacle_proximity_cost=False,
    distance_obstacle_proximity_radius=6,
    distance_obstacle_proximity_weight=0.35,
):
    max_size = size[1]
    print("max size: ", max_size)
    # try:

    def _image_index(filename):
        try:
            return (0, int(filename.split(".png")[0].split("_")[1]))
        except (IndexError, ValueError):
            return (1, filename)

    image_filenames = sorted(
        [fn for fn in os.listdir(img_folder) if fn.endswith(".png")],
        key=_image_index,
    )

    for i, fn in tqdm(enumerate(image_filenames)):
        if i >= n_imgs:
            break

        n = int(fn.split(".png")[0].split("_")[1])
        filename = fn
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
            downsample_factor_w = max(1, math.ceil(img.shape[1] / max_size))
            downsample_factor_h = max(1, math.ceil(img.shape[0] / max_size))

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

        img_terra = _convert_img_to_terra(img, all_dumpable)
        # Pad to max size
        if center_padding:
            xdim = max_size - img_terra.shape[0]
            ydim = max_size - img_terra.shape[1]
            # Note: applying full dumping tiles for the centered version
            img_terra_pad = np.ones((max_size, max_size), dtype=img_terra.dtype)
            print(
                "xdim:",
                xdim,
                "max_size:",
                max_size,
                "ydim:",
                ydim,
                "img_terra shape:",
                img_terra.shape,
            )
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
            img_terra_occupancy = _convert_occupancy_to_terra(occupancy)
            if has_dumpability:
                img_terra_dumpability = np.zeros((max_size, max_size), dtype=np.bool_)
                img_terra_dumpability = _convert_dumpability_to_terra(dumpability)

        destination_folder_images = destination_folder / "images"
        destination_folder_images.mkdir(parents=True, exist_ok=True)
        destination_folder_occupancy = destination_folder / "occupancy"
        destination_folder_occupancy.mkdir(parents=True, exist_ok=True)
        destination_folder_dumpability = destination_folder / "dumpability"
        destination_folder_dumpability.mkdir(parents=True, exist_ok=True)
        if copy_metadata:
            destination_folder_metadata = destination_folder / "metadata"
            destination_folder_metadata.mkdir(parents=True, exist_ok=True)

        img_terra_pad = img_terra_pad.repeat(expansion_factor, axis=0).repeat(
            expansion_factor, axis=1
        )
        img_terra_occupancy = img_terra_occupancy.repeat(
            expansion_factor, axis=0
        ).repeat(expansion_factor, axis=1)
        if has_dumpability:
            img_terra_dumpability = img_terra_dumpability.repeat(
                expansion_factor, 0
            ).repeat(expansion_factor, 1)

        np.save(destination_folder_images / f"img_{i + 1}", img_terra_pad)
        np.save(destination_folder_occupancy / f"img_{i + 1}", img_terra_occupancy)
        if has_dumpability:
            np.save(
                destination_folder_dumpability / f"img_{i + 1}", img_terra_dumpability
            )
        else:
            np.save(
                destination_folder_dumpability / f"img_{i + 1}",
                np.ones_like(img_terra_pad),
            )
        if actions_folder is not None:
            actions_path = actions_folder / f"trench_{n}.png"
            actions = cv2.imread(str(actions_path))
            actions_terra = _convert_actions_to_terra(actions)
            actions_terra = actions_terra.repeat(expansion_factor, axis=0).repeat(expansion_factor, axis=1)
            destination_folder_actions = destination_folder / "actions"
            destination_folder_actions.mkdir(parents=True, exist_ok=True)
            np.save(destination_folder_actions / f"img_{i + 1}", actions_terra)
        if copy_metadata:
            source_metadata_path = metadata_folder / f"{filename.split('.png')[0]}.json"
            destination_metadata_path = destination_folder_metadata / f"trench_{i + 1}.json"
            with open(source_metadata_path) as json_file:
                metadata = json.load(json_file)
            with open(destination_metadata_path, "w") as json_file:
                json.dump(metadata, json_file)

    # Distance maps are required by the reward system (see README "Distance maps").
    # Produce them here so every standalone generator emits a fully-ready Terra
    # dataset in one pass, instead of requiring a separate `tools/` invocation.
    if generate_distance_maps:
        write_distance_maps(
            destination_folder,
            metric=distance_metric,
            connectivity=distance_connectivity,
            realistic_max_distance=distance_realistic_max,
            obstacle_proximity_cost=distance_obstacle_proximity_cost,
            obstacle_proximity_radius=distance_obstacle_proximity_radius,
            obstacle_proximity_weight=distance_obstacle_proximity_weight,
        )


def generate_foundations_terra(dataset_folder, size, n_imgs, all_dumpable):
    print("Converting foundations...")
    # Check which foundation types exist and convert them
    possible_levels = ["foundations", "foundations_large", "foundations_dumpzones", "foundations_dumpzones_large"]
    foundations_levels = []
    
    for level in possible_levels:
        level_path = Path(dataset_folder) / level
        if level_path.exists():
            foundations_levels.append(level)
            print(f"  Found {level} folder - will convert to train/{level}")
    
    if not foundations_levels:
        print("  No foundation folders found - skipping foundation conversion")
        return
        
    for level in foundations_levels:
        img_folder = Path(dataset_folder) / level / "images"
        metadata_folder = Path(dataset_folder) / level / "metadata"
        occupancy_folder = Path(dataset_folder) / level/ "occupancy"
        dumpability_folder = Path(dataset_folder) / level / "dumpability"
        destination_folder = Path(dataset_folder) / "train" / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        
        # Determine the correct all_dumpable setting based on the level
        level_all_dumpable = all_dumpable
        if "dumpzones" in level:
            level_all_dumpable = False  # Foundations with dump zones use specific dump zones, not all dumpable
            print(f"  Converting {level} with all_dumpable=False (specific dump zones)")
        else:
            print(f"  Converting {level} with all_dumpable={all_dumpable}")
        
        _convert_all_imgs_to_terra(
            img_folder,
            metadata_folder,
            occupancy_folder,
            dumpability_folder,
            destination_folder,
            size,
            n_imgs,
            all_dumpable=level_all_dumpable,
            copy_metadata=True,
            downsample=False,
            has_dumpability=True,
            center_padding=False,
            actions_folder=None,
        )


def generate_trenches_terra(dataset_folder, size, n_imgs, expansion_factor, all_dumpable):
    print("Converting trenches...")
    trenches_name = "trenches"
    trenches_path = Path(dataset_folder) / trenches_name
    levels = [d.name for d in trenches_path.iterdir() if d.is_dir()]
    for level in levels:
        img_folder = trenches_path / level / "images"
        metadata_folder = trenches_path / level / "metadata"
        occupancy_folder = trenches_path / level / "occupancy"
        dumpability_folder = trenches_path / level / "dumpability"
        destination_folder = Path(dataset_folder) / "train" / trenches_name / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(
            img_folder,
            metadata_folder,
            occupancy_folder,
            dumpability_folder,
            destination_folder,
            size,
            n_imgs,
            expansion_factor=expansion_factor,
            all_dumpable=all_dumpable,
            actions_folder=None,
        )

def generate_relocations_terra(dataset_folder, size, n_imgs):
    print("Converting relocations...")
    img_folder = Path(dataset_folder) / "relocations" / "images"
    metadata_folder = Path(dataset_folder) / "relocations" / "metadata"
    occupancy_folder = Path(dataset_folder) / "relocations"/ "occupancy"
    dumpability_folder = Path(dataset_folder) / "relocations" / "dumpability"
    actions_folder = Path(dataset_folder) / "relocations" / "actions"
    destination_folder = Path(dataset_folder) / "train" / "relocations"
    destination_folder.mkdir(parents=True, exist_ok=True)
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,
        copy_metadata=False,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=actions_folder
    )

def generate_relocations_easy_terra(dataset_folder, size, n_imgs):
    print("Converting relocations_easy...")
    img_folder = Path(dataset_folder) / "relocations_easy" / "images"
    metadata_folder = Path(dataset_folder) / "relocations_easy" / "metadata"
    occupancy_folder = Path(dataset_folder) / "relocations_easy" / "occupancy"
    dumpability_folder = Path(dataset_folder) / "relocations_easy" / "dumpability"
    actions_folder = Path(dataset_folder) / "relocations_easy" / "actions"
    destination_folder = Path(dataset_folder) / "train" / "relocations_easy"
    destination_folder.mkdir(parents=True, exist_ok=True)
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,
        copy_metadata=False,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=actions_folder
    )

def generate_relocations_medium_terra(dataset_folder, size, n_imgs):
    print("Converting relocations_medium...")
    img_folder = Path(dataset_folder) / "relocations_medium" / "images"
    metadata_folder = Path(dataset_folder) / "relocations_medium" / "metadata"
    occupancy_folder = Path(dataset_folder) / "relocations_medium" / "occupancy"
    dumpability_folder = Path(dataset_folder) / "relocations_medium" / "dumpability"
    actions_folder = Path(dataset_folder) / "relocations_medium" / "actions"
    destination_folder = Path(dataset_folder) / "train" / "relocations_medium"
    destination_folder.mkdir(parents=True, exist_ok=True)
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,
        copy_metadata=False,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=actions_folder
    )

def generate_relocations_hard_terra(dataset_folder, size, n_imgs):
    print("Converting relocations_hard...")
    img_folder = Path(dataset_folder) / "relocations_hard" / "images"
    metadata_folder = Path(dataset_folder) / "relocations_hard" / "metadata"
    occupancy_folder = Path(dataset_folder) / "relocations_hard"/ "occupancy"
    dumpability_folder = Path(dataset_folder) / "relocations_hard" / "dumpability"
    actions_folder = Path(dataset_folder) / "relocations_hard" / "actions"
    destination_folder = Path(dataset_folder) / "train" / "relocations_hard"
    destination_folder.mkdir(parents=True, exist_ok=True)
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,
        copy_metadata=False,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=actions_folder
    )


def generate_relocations_harder_terra(dataset_folder, size, n_imgs):
    print("Converting relocations_harder...")
    img_folder = Path(dataset_folder) / "relocations_harder" / "images"
    metadata_folder = Path(dataset_folder) / "relocations_harder" / "metadata"
    occupancy_folder = Path(dataset_folder) / "relocations_harder"/ "occupancy"
    dumpability_folder = Path(dataset_folder) / "relocations_harder" / "dumpability"
    actions_folder = Path(dataset_folder) / "relocations_harder" / "actions"
    destination_folder = Path(dataset_folder) / "train" / "relocations_harder"
    destination_folder.mkdir(parents=True, exist_ok=True)
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,
        copy_metadata=False,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=actions_folder
    )

def generate_custom_terra(dataset_folder, size, n_imgs):
    print("Converting custom maps...")
    img_folder = Path(dataset_folder) / ".." / "custom" / "images"
    metadata_folder = Path(dataset_folder) / ".." / "custom" / "metadata"
    occupancy_folder = Path(dataset_folder) / ".." / "custom"/ "occupancy"
    dumpability_folder = Path(dataset_folder) / ".." / "custom" / "dumpability"
    destination_folder = Path(dataset_folder) / "train" / "custom"
    destination_folder.mkdir(parents=True, exist_ok=True)
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,
        copy_metadata=False,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=None,
    )


def generate_foundations_dumpzones_terra(dataset_folder, size, n_imgs):
    """Convert foundations with dump zones to Terra format."""
    print("Converting foundations with dump zones...")
    
    # Check if foundations_dumpzones exists
    foundations_dumpzones_dir = Path(dataset_folder) / "foundations_dumpzones"
    if not foundations_dumpzones_dir.exists():
        print(f"  foundations_dumpzones directory not found: {foundations_dumpzones_dir}")
        return
    
    print(f"  Found foundations_dumpzones folder - will convert to train/foundations_dumpzones")
    
    # Set up paths
    img_folder = foundations_dumpzones_dir / "images"
    metadata_folder = foundations_dumpzones_dir / "metadata"
    occupancy_folder = foundations_dumpzones_dir / "occupancy"
    dumpability_folder = foundations_dumpzones_dir / "dumpability"
    destination_folder = Path(dataset_folder) / "train" / "foundations_dumpzones"
    
    # Create destination directory
    destination_folder.mkdir(parents=True, exist_ok=True)
    print(f"  Created destination directory: {destination_folder}")
    
    # Convert with specific settings for foundations with dump zones
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,  # Foundations with dump zones use specific dump zones, not all dumpable
        copy_metadata=True,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=None,
    )
    
    print(f"  foundations_dumpzones conversion completed")


def generate_foundations_dumpzones_harder_terra(dataset_folder, size, n_imgs):
    """Convert foundations with dump zones (harder version) to Terra format."""
    print("Converting foundations with dump zones (harder version)...")
    
    # Check if foundations_dumpzones_harder exists
    foundations_dumpzones_harder_dir = Path(dataset_folder) / "foundations_dumpzones_harder_nodump"
    if not foundations_dumpzones_harder_dir.exists():
        print(f"  foundations_dumpzones_harder directory not found: {foundations_dumpzones_harder_dir}")
        return
    
    print(f"  Found foundations_dumpzones_harder folder - will convert to train/foundations_dumpzones_harder_nodump")
    
    # Set up paths
    img_folder = foundations_dumpzones_harder_dir / "images"
    metadata_folder = foundations_dumpzones_harder_dir / "metadata"
    occupancy_folder = foundations_dumpzones_harder_dir / "occupancy"
    dumpability_folder = foundations_dumpzones_harder_dir / "dumpability"
    destination_folder = Path(dataset_folder) / "train" / "foundations_dumpzones_harder_nodump"
    
    # Create destination directory
    destination_folder.mkdir(parents=True, exist_ok=True)
    print(f"  Created destination directory: {destination_folder}")
    
    # Convert with specific settings for foundations with dump zones (harder version)
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,  # Foundations with dump zones use specific dump zones, not all dumpable
        copy_metadata=True,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=None,
    )
    
    print(f"  foundations_dumpzones_harder conversion completed")


def generate_foundations_quarter_large_terra(dataset_folder, size, n_imgs):
    """Convert foundations with quarter dump zones (large version) to Terra format."""
    print("Converting foundations with quarter dump zones (large version)...")
    
    # Check if foundations_quarter_large exists
    foundations_quarter_large_dir = Path(dataset_folder) / "foundations_quarter_large"
    if not foundations_quarter_large_dir.exists():
        print(f"  foundations_quarter_large directory not found: {foundations_quarter_large_dir}")
        return
    
    print(f"  Found foundations_quarter_large folder - will convert to train/foundations_quarter_large")
    
    # Set up paths
    img_folder = foundations_quarter_large_dir / "images"
    metadata_folder = foundations_quarter_large_dir / "metadata"
    occupancy_folder = foundations_quarter_large_dir / "occupancy"
    dumpability_folder = foundations_quarter_large_dir / "dumpability"
    destination_folder = Path(dataset_folder) / "train" / "foundations_quarter_large"
    
    # Create destination directory
    destination_folder.mkdir(parents=True, exist_ok=True)
    print(f"  Created destination directory: {destination_folder}")
    
    # Convert with specific settings for quarter dump zone foundations
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,  # Quarter foundations use specific dump zones, not all dumpable
        copy_metadata=True,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=None,
    )
    
    print(f"  foundations_quarter_large conversion completed")


def generate_trenches_dumpzone_terra(dataset_folder, size, n_imgs):
    """Convert trenches with dump zones to Terra format."""
    print("Converting trenches with dump zones...")
    
    # Check if trenches directory exists
    trenches_dir = Path(dataset_folder) / "trenches"
    if not trenches_dir.exists():
        print(f"  trenches directory not found: {trenches_dir}")
        return
    
    # Look for subdirectories with "_dumpzone" suffix
    dumpzone_subdirs = []
    for subdir in trenches_dir.iterdir():
        if subdir.is_dir() and "_dumpzone" in subdir.name:
            dumpzone_subdirs.append(subdir)
    
    if not dumpzone_subdirs:
        print(f"  No trenches with dump zones found in: {trenches_single_dumpzone_dir}")
        return
    
    print(f"  Found {len(dumpzone_subdirs)} trench dump zone subdirectories")
    
    for subdir in dumpzone_subdirs:
        print(f"  Processing {subdir.name}...")
        
        # Set up paths
        img_folder = subdir / "images"
        metadata_folder = subdir / "metadata"
        occupancy_folder = subdir / "occupancy"
        dumpability_folder = subdir / "dumpability"
        destination_folder = Path(dataset_folder) / "train" / "trenches" / subdir.name
        
        # Create destination directory
        destination_folder.mkdir(parents=True, exist_ok=True)
        print(f"    Created destination directory: {destination_folder}")
        
        # Convert with specific settings for trenches with dump zones
        _convert_all_imgs_to_terra(
            img_folder,
            metadata_folder,
            occupancy_folder,
            dumpability_folder,
            destination_folder,
            size,
            n_imgs,
            all_dumpable=False,  # Trenches with dump zones use specific dump zones, not all dumpable
            copy_metadata=True,
            downsample=False,
            has_dumpability=True,
            center_padding=False,
            actions_folder=None,
        )
        
        print(f"    {subdir.name} conversion completed")
    
    print(f"  All trenches with dump zones conversion completed")


def generate_dataset_terra_format(dataset_folder, size, n_imgs=1000, map_types=None):
    print("dataset folder: ", dataset_folder)
    if map_types is None:
        map_types = [
            "foundations", "foundations_dumpzones", "foundations_dumpzones_harder", "trenches", "trenches_dumpzone", "relocations", "relocations_easy",
            "relocations_medium", "relocations_hard", "custom"
        ]
    if "foundations" in map_types:
        generate_foundations_terra(dataset_folder, size, n_imgs, all_dumpable=False)
        print("Foundations processed successfully.")
    if "foundations_dumpzones" in map_types:
        # Use the dedicated function for foundations with dump zones
        generate_foundations_dumpzones_terra(dataset_folder, size, n_imgs)
        print("Foundations with dump zones processed successfully.")
    if "foundations_dumpzones_harder" in map_types:
        # Use the dedicated function for foundations with dump zones (harder version)
        generate_foundations_dumpzones_harder_terra(dataset_folder, size, n_imgs)
        print("Foundations with dump zones (harder version) processed successfully.")
    if "foundations_quarter_large" in map_types:
        # Use the dedicated function for foundations with quarter dump zones (large version)
        generate_foundations_quarter_large_terra(dataset_folder, size, n_imgs)
        print("Foundations with quarter dump zones (large version) processed successfully.")
    if "trenches" in map_types:
        generate_trenches_terra(
            dataset_folder, size, n_imgs, expansion_factor=1, all_dumpable=False
        )
        print("Trenches processed successfully.")
    if "trenches_dumpzone" in map_types:
        # Use the dedicated function for trenches with dump zones
        generate_trenches_dumpzone_terra(dataset_folder, size, n_imgs)
        print("Trenches with dump zones processed successfully.")
    if "relocations" in map_types:
        generate_relocations_terra(dataset_folder, size, n_imgs)
        print("Relocations processed successfully.")
    if "relocations_easy" in map_types:
        generate_relocations_easy_terra(dataset_folder, size, n_imgs)
        print("Relocations_easy processed successfully.")
    if "relocations_medium" in map_types:
        generate_relocations_medium_terra(dataset_folder, size, n_imgs)
        print("Relocations_medium processed successfully.")
    if "relocations_hard" in map_types:
        generate_relocations_hard_terra(dataset_folder, size, n_imgs)
        print("Relocations_hard processed successfully.")
    if "relocations_harder" in map_types:
        generate_relocations_harder_terra(dataset_folder, size, n_imgs)
        print("Relocations_harder processed successfully.")
    if "custom" in map_types:
        generate_custom_terra(dataset_folder, size, n_imgs)
        print("Custom maps processed successfully.")
