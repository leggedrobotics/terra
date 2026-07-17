import os
import shutil
import yaml
import json
import math
import numpy as np
import cv2
import skimage
import random
from pathlib import Path
from terra.env_generation.procedural_data import (
    generate_trenches_v2,
    generate_edges,
    generate_diagonal_edges,
    add_obstacles,
    add_non_dumpables,
    initialize_image,
    save_or_display_image,
    convert_terra_pad_to_color,
)
from terra.env_generation.generate_relocations import add_dump_zones
from terra.env_generation.convert_to_terra import (
    _convert_dumpability_to_terra,
    _convert_img_to_terra,
    _convert_occupancy_to_terra,
)
import terra.env_generation.convert_to_terra as convert_to_terra
from terra.env_generation.utils import _get_img_mask, color_dict
from terra.env_generation.generate_foundations_with_dumpzones import create_foundations_with_dumpzones

# Define package directory at module level
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_single_dump_zone_trenches(img_terra_pad, size_dump_min, size_dump_max, dig_mask):
    """
    Create exactly 1 dump zone that avoids dig zone overlaps.
    
    Args:
        img_terra_pad: The image with dig zones
        size_dump_min: Minimum size of dump zone
        size_dump_max: Maximum size of dump zone  
        dig_mask: Boolean mask where True indicates dig areas to avoid
        
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
        
        # Check if this area overlaps with dig zones
        dig_area = dig_mask[y:y+dump_size, x:x+dump_size]
        
        # If no overlaps, place the dump zone
        if not np.any(dig_area):
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
            
            dig_area = dig_mask[y:y+dump_size, x:x+dump_size]
            
            if not np.any(dig_area):
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
                
                dig_area = dig_mask[y:y+dump_size, x:x+dump_size]
                
                if not np.any(dig_area):
                    img_terra_pad[y:y+dump_size, x:x+dump_size] = color_dict["dumping"]
                    dump_cumulative_mask[y:y+dump_size, x:x+dump_size] = True
                    print(f"Second fallback successful: placed even smaller dump zone of size {dump_size} at position ({x}, {y})")
                    break
            else:
                # If all fallbacks fail, throw an error
                raise RuntimeError(f"Failed to place dump zone after {max_attempts} attempts with original size ({size_dump_min}-{size_dump_max}), 100 attempts with first fallback size ({fallback_size_min}-{fallback_size_max}), and 100 attempts with second fallback size ({fallback2_size_min}-{fallback2_size_max}). Image may be too crowded or dump zone size too large.")
    
    return img_terra_pad, dump_cumulative_mask


def create_procedural_trenches(config):
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
        #save_folder = os.path.join("data/terra", "trenches", level)
        save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "trenches", level)
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
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
        trench_width_tiles = trenches_config.get("trench_width_tiles")
        if trench_width_tiles is not None:
            trench_width_tiles = max(1, int(trench_width_tiles))
            trench_dims_min = (trench_width_tiles, trench_dims_min[1])
            trench_dims_max = (trench_width_tiles, trench_dims_max[1])

        diagonal = trench_dims_config["diagonal"]

        sizes_small = (trench_dims_min[0], trench_dims_max[0])
        sizes_long = (trench_dims_min[1], trench_dims_max[1])

        generate_trenches_v2(
            n_imgs,
            trenches_config["img_edge_min"],
            trenches_config["img_edge_max"],
            sizes_small,
            sizes_long,
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
            diagonal,
        )


def create_procedural_trenches_with_dumpzones(config):
    """
    Creates trench environments with specific dump zones using configurations from a YAML file.
    Similar to create_procedural_trenches but adds a single dump zone to each trench.
    """
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

    # Dump zone parameters (similar to foundations_with_dumpzones_harder)
    n_dump_min = 1
    n_dump_max = 1
    size_dump_min = 13
    size_dump_max = 15

    for level, n_trenches in zip(difficulty_levels, corrected_trenches_per_level):
        # Create new folder for trenches with dump zones
        save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "trenches", f"{level}_dumpzone")
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
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
        trench_width_tiles = trenches_config.get("trench_width_tiles")
        if trench_width_tiles is not None:
            trench_width_tiles = max(1, int(trench_width_tiles))
            trench_dims_min = (trench_width_tiles, trench_dims_min[1])
            trench_dims_max = (trench_width_tiles, trench_dims_max[1])

        diagonal = trench_dims_config["diagonal"]

        # Generate trenches with dump zones
        sizes_small = (trench_dims_min[0], trench_dims_max[0])
        sizes_long = (trench_dims_min[1], trench_dims_max[1])

        generate_trenches_with_dumpzones(
            n_imgs,
            trenches_config["img_edge_min"],
            trenches_config["img_edge_max"],
            sizes_small,
            sizes_long,
            n_trenches,
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
            diagonal,
            n_dump_min,
            n_dump_max,
            size_dump_min,
            size_dump_max,
        )


def generate_trenches_with_dumpzones(
    n_imgs,
    img_edge_min,
    img_edge_max,
    sizes_small,
    sizes_long,
    n_edges,
    resolution,
    save_folder=None,
    n_obs_min=1,
    n_obs_max=3,
    size_obstacle_min=2,
    size_obstacle_max=8,
    n_nodump_min=1,
    n_nodump_max=3,
    size_nodump_min=2,
    size_nodump_max=8,
    diagonal=False,
    n_dump_min=1,
    n_dump_max=1,
    size_dump_min=10,
    size_dump_max=13,
):
    """
    Generate trenches with specific dump zones added.
    Similar to generate_trenches_v2 but adds a single dump zone to each trench.
    """
    min_edges, max_edges = n_edges
    i = 0
    while i < n_imgs:
        print(f"Processing trench with dump zone nr {i + 1}")
        
        # Start with neutral background instead of dumping
        img = initialize_image(img_edge_min, img_edge_max, color_dict["neutral"])
        
        if diagonal:
            img, cumulative_mask, metadata = generate_diagonal_edges(
                img, (min_edges, max_edges), sizes_small, sizes_long, color_dict
            )
        else:
            img, cumulative_mask, metadata = generate_edges(
                img, (min_edges, max_edges), sizes_small, sizes_long, color_dict
            )
        
        if img is None:
            continue
            
        # Create a mask for dig zones (white areas)
        dig_mask = np.all(img == color_dict["digging"], axis=-1)
        
        # Add exactly 1 dump zone that avoids dig zones
        img, dump_cumulative_mask = create_single_dump_zone_trenches(
            img, size_dump_min, size_dump_max, dig_mask
        )
        
        # Update cumulative mask to include dump zones
        cumulative_mask = cumulative_mask | dump_cumulative_mask
        
        # Add obstacles
        occ, cumulative_mask = add_obstacles(
            img,
            cumulative_mask,
            n_obs_min,
            n_obs_max,
            size_obstacle_min,
            size_obstacle_max,
        )

        # Add non-dumpables
        dmp, cumulative_mask = add_non_dumpables(
            img,
            occ,
            cumulative_mask,
            n_nodump_min,
            n_nodump_max,
            size_nodump_min,
            size_nodump_max,
        )
        
        save_or_display_image(img, occ, dmp, metadata, save_folder, i)
        i += 1




def create_foundations(config,
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
                      use_specific_dump_zones=False,
                      n_dump_min=1,
                      n_dump_max=1,
                      size_dump_min=13,
                      size_dump_max=13):
    """
    Creates foundation environments using configurations from a YAML file.
    
    If use_specific_dump_zones is True, delegates to create_foundations_with_dumpzones.
    Otherwise, creates regular foundations with everything outside buildings dumpable.
    """
    if use_specific_dump_zones:
        # Delegate to the specialized function for dump zones
        create_foundations_with_dumpzones(
            config,
            n_obs_min=n_obs_min,
            n_obs_max=n_obs_max,
            size_obstacle_min=size_obstacle_min,
            size_obstacle_max=size_obstacle_max,
            n_nodump_min=n_nodump_min,
            n_nodump_max=n_nodump_max,
            size_nodump_min=size_nodump_min,
            size_nodump_max=size_nodump_max,
            expansion_factor=expansion_factor,
            all_dumpable=all_dumpable,
            copy_metadata=copy_metadata,
            has_dumpability=has_dumpability,
            center_padding=center_padding,
            n_dump_min=n_dump_min,
            n_dump_max=n_dump_max,
            size_dump_min=size_dump_min,
            size_dump_max=size_dump_max
        )
        return
    
    # Original behavior for regular foundations (everything outside buildings dumpable)
    # Extract configuration parameters
    foundation_config = config["foundations"]
    n_imgs = config["n_imgs"]
    size = foundation_config["max_size"]
    dataset_path = foundation_config["dataset_rel_path"]

    # Define save folder for the envs using os.path.join
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "foundations")
    save_folder_large = os.path.join(PACKAGE_DIR, "data", "terra", "foundations_large")
    print(f"Creating regular foundations - saving to: foundations/")

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
            
            # Make everything outside buildings dumpable (original behavior)
            dumping_image = initialize_image(size, size, color_dict["dumping"])
            # Create a mask where img_terra_pad is not equal to color_dict["digging"]
            mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)
            # Use the mask to assign values from dumping_image to img_terra_pad
            img_terra_pad[mask] = dumping_image[mask]

            # Original behavior: mark dig zones as occupied
            cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
            # where the img_terra_pad is [255, 255, 255] set to True across the three channels
            cumulative_mask[np.all(img_terra_pad == color_dict["digging"], axis=-1)] = True
            
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

        print("Foundations created successfully.")
