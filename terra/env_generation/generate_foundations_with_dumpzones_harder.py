import os
import yaml
import json
import math
import numpy as np
import cv2
import skimage
import random
from pathlib import Path
from terra.env_generation.procedural_data import (
    add_obstacles,
    add_non_dumpables,
    initialize_image,
    save_or_display_image,
    convert_terra_pad_to_color,
)
from terra.env_generation.convert_to_terra import (
    _convert_dumpability_to_terra,
    _convert_img_to_terra,
    _convert_occupancy_to_terra,
)
import terra.env_generation.convert_to_terra as convert_to_terra
from terra.env_generation.utils import _get_img_mask, color_dict

# Define package directory at module level
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_single_dump_zone(img_terra_pad, size_dump_min, size_dump_max, foundation_mask):
    """
    Create exactly 1 dump zone that avoids foundation overlaps.
    
    Args:
        img_terra_pad: The image with foundation and dig zones
        size_dump_min: Minimum size of dump zone
        size_dump_max: Maximum size of dump zone  
        foundation_mask: Boolean mask where True indicates foundation areas to avoid
        
    Returns:
        img_terra_pad: Updated image with dump zone
        dump_cumulative_mask: Boolean mask of the dump zone
    """
    height, width = img_terra_pad.shape[:2]
    dump_cumulative_mask = np.zeros((height, width), dtype=np.bool_)
    
    # Offset from borders
    border_offset = 3
    
    # Try to place dump zone in valid areas
    max_attempts = 100
    for attempt in range(max_attempts):
        # Random dump zone size
        dump_size = random.randint(size_dump_min, size_dump_max)
        
        # Place dump zone at border with border_offset amount of offset
        # Randomly choose which border (top, bottom, left, right)
        border_choice = random.randint(0, 3)  # 0=top, 1=bottom, 2=left, 3=right
        
        if border_choice == 0:  # Top border
            x = random.randint(border_offset, width - dump_size - border_offset)
            y = border_offset
        elif border_choice == 1:  # Bottom border
            x = random.randint(border_offset, width - dump_size - border_offset)
            y = height - dump_size - border_offset
        elif border_choice == 2:  # Left border
            x = border_offset
            y = random.randint(border_offset, height - dump_size - border_offset)
        else:  # Right border
            x = width - dump_size - border_offset
            y = random.randint(border_offset, height - dump_size - border_offset)
        
        # Check if this area overlaps with foundations
        dump_area = foundation_mask[y:y+dump_size, x:x+dump_size]
        
        # Also check if it overlaps with dig zones
        dig_area = np.all(img_terra_pad[y:y+dump_size, x:x+dump_size] == color_dict["digging"], axis=-1)
        
        # If no overlaps, place the dump zone
        if not np.any(dump_area) and not np.any(dig_area):
            # Create dump zone
            img_terra_pad[y:y+dump_size, x:x+dump_size] = color_dict["dumping"]
            dump_cumulative_mask[y:y+dump_size, x:x+dump_size] = True
            print(f"Successfully placed dump zone of size {dump_size} at position ({x}, {y})")
            break
    else:
        print(f"Warning: Could not place dump zone after {max_attempts} attempts")
        # Fallback: try with smaller dump zone size (-3) for another 100 attempts
        print("Attempting fallback with smaller dump zone (size -3) for another 100 attempts...")
        fallback_size_min = max(5, size_dump_min - 3)  # Subtract 3 from min size
        fallback_size_max = max(8, size_dump_max - 3)  # Subtract 3 from max size
        
        for attempt in range(100):  # Try 100 more times with smaller size
            dump_size = random.randint(fallback_size_min, fallback_size_max)
            
            # Place dump zone at border with border_offset amount of offset
            border_choice = random.randint(0, 3)  # 0=top, 1=bottom, 2=left, 3=right
            
            if border_choice == 0:  # Top border
                x = random.randint(border_offset, width - dump_size - border_offset)
                y = border_offset
            elif border_choice == 1:  # Bottom border
                x = random.randint(border_offset, width - dump_size - border_offset)
                y = height - dump_size - border_offset
            elif border_choice == 2:  # Left border
                x = border_offset
                y = random.randint(border_offset, height - dump_size - border_offset)
            else:  # Right border
                x = width - dump_size - border_offset
                y = random.randint(border_offset, height - dump_size - border_offset)
            
            dump_area = foundation_mask[y:y+dump_size, x:x+dump_size]
            dig_area = np.all(img_terra_pad[y:y+dump_size, x:x+dump_size] == color_dict["digging"], axis=-1)
            
            if not np.any(dump_area) and not np.any(dig_area):
                img_terra_pad[y:y+dump_size, x:x+dump_size] = color_dict["dumping"]
                dump_cumulative_mask[y:y+dump_size, x:x+dump_size] = True
                print(f"Fallback successful: placed smaller dump zone of size {dump_size} at position ({x}, {y})")
                break
        else:
            # Second fallback: try with even smaller dump zone size (-2 more)
            print("Attempting second fallback with even smaller dump zone (size -5 total)...")
            fallback2_size_min = max(3, fallback_size_min - 2)  # Subtract 2 more from fallback min
            fallback2_size_max = max(6, fallback_size_max - 2)  # Subtract 2 more from fallback max
            
            for attempt in range(100):  # Try 100 more times with even smaller size
                dump_size = random.randint(fallback2_size_min, fallback2_size_max)
                
                # Place dump zone at border with border_offset amount of offset
                border_choice = random.randint(0, 3)  # 0=top, 1=bottom, 2=left, 3=right
                
                if border_choice == 0:  # Top border
                    x = random.randint(border_offset, width - dump_size - border_offset)
                    y = border_offset
                elif border_choice == 1:  # Bottom border
                    x = random.randint(border_offset, width - dump_size - border_offset)
                    y = height - dump_size - border_offset
                elif border_choice == 2:  # Left border
                    x = border_offset
                    y = random.randint(border_offset, height - dump_size - border_offset)
                else:  # Right border
                    x = width - dump_size - border_offset
                    y = random.randint(border_offset, height - dump_size - border_offset)
                
                dump_area = foundation_mask[y:y+dump_size, x:x+dump_size]
                dig_area = np.all(img_terra_pad[y:y+dump_size, x:x+dump_size] == color_dict["digging"], axis=-1)
                
                if not np.any(dump_area) and not np.any(dig_area):
                    img_terra_pad[y:y+dump_size, x:x+dump_size] = color_dict["dumping"]
                    dump_cumulative_mask[y:y+dump_size, x:x+dump_size] = True
                    print(f"Second fallback successful: placed even smaller dump zone of size {dump_size} at position ({x}, {y})")
                    break
            else:
                # If all fallbacks fail, throw an error
                raise RuntimeError(f"Failed to place dump zone after {max_attempts} attempts with original size ({size_dump_min}-{size_dump_max}), 100 attempts with first fallback size ({fallback_size_min}-{fallback_size_max}), and 100 attempts with second fallback size ({fallback2_size_min}-{fallback2_size_max}). Image may be too crowded or dump zone size too large.")
    
    return img_terra_pad, dump_cumulative_mask


def create_foundations_with_dumpzones_harder_defaults(
                                    n_imgs=1000,
                                    max_size=64,
                                    dataset_path="data/openstreet",
                                    n_obs_min=1,
                                    n_obs_max=2,
                                    size_obstacle_min=4,
                                    size_obstacle_max=7,
                                    n_nodump_min=0,
                                    n_nodump_max=1,
                                    size_nodump_min=5,
                                    size_nodump_max=8,
                                    expansion_factor=1,
                                    all_dumpable=False,
                                    copy_metadata=True,
                                    has_dumpability=False,
                                    center_padding=True,
                                    n_dump_min=1,
                                    n_dump_max=1,
                                    size_dump_min=10,
                                    size_dump_max=13):
    """
    Creates foundation environments with specific dump zones using default parameters.

    Parameters:
    - n_imgs (int): Number of images to generate (default: 1000)
    - max_size (int): Maximum size of the images (default: 50)
    - dataset_path (str): Path to the dataset (default: "data/openstreet")
    - n_obs_min (int): Minimum number of obstacles to add (default: 1)
    - n_obs_max (int): Maximum number of obstacles to add (default: 2)
    - size_obstacle_min (int): Minimum size of obstacles (default: 4)
    - size_obstacle_max (int): Maximum size of obstacles (default: 8)
    - n_nodump_min (int): Minimum number of non-dumpable areas (default: 0)
    - n_nodump_max (int): Maximum number of non-dumpable areas (default: 0)
    - size_nodump_min (int): Minimum size of non-dumpable areas (default: 8)
    - size_nodump_max (int): Maximum size of non-dumpable areas (default: 10)
    - expansion_factor (int): Factor to expand the image by (default: 1)
    - all_dumpable (bool): Whether all areas should be dumpable (default: False)
    - copy_metadata (bool): Whether to copy metadata (default: True)
    - has_dumpability (bool): Whether the image has dumpability information (default: False)
    - center_padding (bool): Whether to center the padding (default: True)
    - n_dump_min (int): Minimum number of specific dump zones (default: 1)
    - n_dump_max (int): Maximum number of specific dump zones (default: 1)
    - size_dump_min (int): Minimum size of specific dump zones (default: 13)
    - size_dump_max (int): Maximum size of specific dump zones (default: 13)
    """
    # Define save folder for the envs using os.path.join
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "foundations_dumpzones_harder_nodump")
    save_folder_large = os.path.join(PACKAGE_DIR, "data", "terra", "foundations_dumpzones_harder_large_nodump")
    print(f"Creating foundations with specific dump zones - saving to: foundations_dumpzones_harder_nodump/")

    # Choose different downsampling factors for different curriculum levels
    downsampling_factors = {
        save_folder: 2,
        save_folder_large: 1,
    }

    # Get the full dataset path using os.path.join
    full_dataset_path = os.path.join(PACKAGE_DIR, dataset_path)

    # Process foundation images
    foundations_name = "foundations"
    img_folder = Path(full_dataset_path) / foundations_name / "images"
    metadata_folder = Path(full_dataset_path) / foundations_name / "metadata"
    occupancy_folder = Path(full_dataset_path) / foundations_name / "occupancy"
    dumpability_folder = Path(full_dataset_path) / foundations_name / "dumpability"
    filename_start = sorted(os.listdir(img_folder))[0].split("_")[0]

    for curriculum_level, downsampling_factor in downsampling_factors.items():
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

            with open(
                metadata_folder / f"{filename.split('.png')[0]}.json"
            ) as json_file:
                metadata = json.load(json_file)

            # Calculate downsample factors based on max_size
            downsample_factor_w = int(max(1, math.ceil(img.shape[1] / max_size))) * downsampling_factor
            downsample_factor_h = int(max(1, math.ceil(img.shape[0] / max_size))) * downsampling_factor

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
            
            # Create specific dump zones like relocations (for skid steer training)
            # Start with neutral background everywhere except dig zones
            neutral_mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)
            img_terra_pad[neutral_mask] = color_dict["neutral"]
            
            # Create a mask for foundation buildings (non-neutral areas)
            foundation_mask = np.all(img_terra_pad != color_dict["neutral"], axis=-1) & np.all(img_terra_pad != color_dict["digging"], axis=-1)
            
            # Create exactly 1 dump zone with no overlaps
            img_terra_pad, dump_cumulative_mask = create_single_dump_zone(
                img_terra_pad, size_dump_min, size_dump_max, foundation_mask
            )

            # Initialize cumulative mask with dump zones and dig zones
            cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
            # Mark dig zones (white areas) and dump zones as occupied
            cumulative_mask[np.all(img_terra_pad == color_dict["digging"], axis=-1)] = True
            cumulative_mask = dump_cumulative_mask | cumulative_mask
            
            # Add obstacles
            occ, cumulative_mask = add_obstacles(
                img_terra_pad,
                cumulative_mask,
                n_obs_min,
                n_obs_max,
                size_obstacle_min,
                size_obstacle_max,
            )

            # Add non-dumpables
            dmp, cumulative_mask = add_non_dumpables(
                img_terra_pad,
                occ,
                cumulative_mask,
                n_nodump_min,
                n_nodump_max,
                size_nodump_min,
                size_nodump_max,
            )
            
            save_or_display_image(img_terra_pad, occ, dmp, metadata, curriculum_level, n)

        print("Foundations with dump zones created successfully.")


def create_foundations_with_dumpzones(config,
                                    n_obs_min=1,
                                    n_obs_max=2,
                                    size_obstacle_min=4,
                                    size_obstacle_max=8,
                                    n_nodump_min=0,
                                    n_nodump_max=0,
                                    size_nodump_min=8,
                                    size_nodump_max=10,
                                    expansion_factor=1,
                                    all_dumpable=False,
                                    copy_metadata=True,
                                    has_dumpability=False,
                                    center_padding=True,
                                    n_dump_min=1,
                                    n_dump_max=1,
                                    size_dump_min=13,
                                    size_dump_max=13):
    """
    Creates foundation environments with specific dump zones using configurations from a YAML file.

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
    - has_dumpability (bool): Whether the image has dumpability information.
    - center_padding (bool): Whether to center the padding.
    - n_dump_min (int): Minimum number of specific dump zones.
    - n_dump_max (int): Maximum number of specific dump zones.
    - size_dump_min (int): Minimum size of specific dump zones.
    - size_dump_max (int): Maximum size of specific dump zones.
    """
    # Extract configuration parameters
    foundation_config = config["foundations"]
    n_imgs = config["n_imgs"]
    size = foundation_config["max_size"]
    dataset_path = foundation_config["dataset_rel_path"]
    
    # Extract dump zone configuration if available
    if "use_specific_dump_zones" in foundation_config:
        n_dump_min = foundation_config.get("n_dump_min", n_dump_min)
        n_dump_max = foundation_config.get("n_dump_max", n_dump_max)
        size_dump_min = foundation_config.get("size_dump_min", size_dump_min)
        size_dump_max = foundation_config.get("size_dump_max", size_dump_max)

    # Define save folder for the envs using os.path.join
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "foundations_dumpzones")
    save_folder_large = os.path.join(PACKAGE_DIR, "data", "terra", "foundations_dumpzones_large")
    print(f"Creating foundations with specific dump zones - saving to: foundations_dumpzones/")

    # Choose different downsampling factors for different curriculum levels
    downsampling_factors = {
        save_folder: 2,
        save_folder_large: 1,
    }

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

    for curriculum_level, downsampling_factor in downsampling_factors.items():
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

            with open(
                metadata_folder / f"{filename.split('.png')[0]}.json"
            ) as json_file:
                metadata = json.load(json_file)

            # Calculate downsample factors based on max_size
            downsample_factor_w = int(max(1, math.ceil(img.shape[1] / max_size))) * downsampling_factor
            downsample_factor_h = int(max(1, math.ceil(img.shape[0] / max_size))) * downsampling_factor

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
            
            # Create specific dump zones like relocations (for skid steer training)
            # Start with neutral background everywhere except dig zones
            neutral_mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)
            img_terra_pad[neutral_mask] = color_dict["neutral"]
            
            # Create a mask for foundation buildings (non-neutral areas)
            foundation_mask = np.all(img_terra_pad != color_dict["neutral"], axis=-1) & np.all(img_terra_pad != color_dict["digging"], axis=-1)
            
            # Create exactly 1 dump zone with no overlaps
            img_terra_pad, dump_cumulative_mask = create_single_dump_zone(
                img_terra_pad, size_dump_min, size_dump_max, foundation_mask
            )

            # Initialize cumulative mask with dump zones and dig zones
            cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
            # Mark dig zones (white areas) and dump zones as occupied
            cumulative_mask[np.all(img_terra_pad == color_dict["digging"], axis=-1)] = True
            cumulative_mask = dump_cumulative_mask | cumulative_mask
            
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
            save_or_display_image(img_terra_pad, occ, dmp, metadata, curriculum_level, n)

        print("Foundations with dump zones created successfully.")


if __name__ == "__main__":
    # Example usage
    config_path = os.path.join(PACKAGE_DIR, "config", "config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    create_foundations_with_dumpzones(config) 