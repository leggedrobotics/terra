#!/usr/bin/env python
import os
import yaml
import json
import math
import numpy as np
import cv2
import skimage
import random
import argparse
import shutil
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
from terra.env_generation.foundation_border_metadata import (
    build_foundation_border_metadata,
)

# Define package directory at module level
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

name_string = ""

def create_single_dump_zone(img_terra_pad, size_dump_min, size_dump_max, foundation_mask):
    """
    Create exactly 1 dump zone on the border that avoids foundation overlaps.
    Uses the same border placement logic as foundations_with_dumpzones_harder.
    
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
    
    # Offset from borders (same as foundations_with_dumpzones_harder)
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


def add_dirt_tiles_hybrid(img, occ, dmp, cumulative_mask, total_dirt_tiles):
    """
    Add dirt tiles exactly like relocations_harder.py does.
    Creates 3 dirt spots with varying sizes, distributed across the map.
    
    Args:
        img: Main image
        occ: Occupancy mask  
        dmp: Dumpability mask
        cumulative_mask: Areas already occupied
        total_dirt_tiles: Total number of dirt tiles to place
        
    Returns:
        action_map: Image with dirt tiles for action map
        cumulative_mask: Updated mask with dirt locations
    """
    w, h = img.shape[:2]
    action_map = np.ones_like(img) * 255  # White background
    mask_occ = _get_img_mask(occ, color_dict["obstacle"])
    mask_dmp = _get_img_mask(dmp, color_dict["nondumpable"])
    
    # Fixed number of dirt spots (3 zones) - same as relocations_harder
    n_spots = 0
    
    # Distribute total dirt tiles across 3 spots
    remaining_dirt = total_dirt_tiles
    dirt_spots = []
    
    print(f"Placing {total_dirt_tiles} dirt tiles across {n_spots} spots...")
    
    # Create 3 dirt spots with varying sizes
    for spot in range(n_spots):
        if remaining_dirt <= 0:
            break
            
        # For the last spot, use all remaining dirt
        if spot == n_spots - 1:
            spot_size = remaining_dirt
        else:
            # Randomly assign dirt to this spot (at least 8, at most remaining/2)
            min_spot_size = max(8, remaining_dirt // 4)
            max_spot_size = min(remaining_dirt - (n_spots - spot - 1) * min_spot_size, remaining_dirt // 2)
            spot_size = np.random.randint(min_spot_size, max_spot_size + 1)
        
        # Calculate dirt patch dimensions (approximate square)
        patch_size = int(np.sqrt(spot_size)) + 1
        
        # Try to place this dirt patch
        placed = False
        for _ in range(30):  # Try 30 times to place without overlap
            x = np.random.randint(5, w - patch_size - 5)
            y = np.random.randint(5, h - patch_size - 5)
            
            if (
                np.all(cumulative_mask[x : x + patch_size, y : y + patch_size] == 0)
                and np.all(mask_occ[x : x + patch_size, y : y + patch_size] == 0)
                and np.all(mask_dmp[x : x + patch_size, y : y + patch_size] == 0)
            ):
                # Place dirt tiles in this patch
                tiles_placed = 0
                for dx in range(patch_size):
                    for dy in range(patch_size):
                        if tiles_placed < spot_size:
                            if (x + dx < w and y + dy < h and 
                                cumulative_mask[x + dx, y + dy] == 0 and
                                mask_occ[x + dx, y + dy] == 0 and
                                mask_dmp[x + dx, y + dy] == 0):
                                
                                action_map[x + dx, y + dy] = np.array(color_dict["dirt"])
                                cumulative_mask[x + dx, y + dy] = True
                                tiles_placed += 1
                
                dirt_spots.append((x, y, tiles_placed))
                remaining_dirt -= tiles_placed
                placed = True
                print(f"  Dirt spot {spot + 1}: placed {tiles_placed} tiles at ({x}, {y})")
                break
        
        if not placed:
            print(f"Warning: Could not place dirt spot {spot + 1}.")
    
    if remaining_dirt > 0:
        print(f"Warning: Could not place all {total_dirt_tiles} dirt tiles. Placed {total_dirt_tiles - remaining_dirt} tiles.")
    else:
        print(f"Successfully placed all {total_dirt_tiles} dirt tiles across {len(dirt_spots)} spots.")
    
    return action_map, cumulative_mask


def create_foundations_hybrid_standalone(
                                    n_imgs=600,
                                    max_size=64,
                                    dataset_path="data/openstreet",
                                    n_obs_min=0,
                                    n_obs_max=1,
                                    size_obstacle_min=3,
                                    size_obstacle_max=6,
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
                                    no_dump_zones=False,
                                    size_dump_min=15,  # Bigger dump zones
                                    size_dump_max=15,  # Bigger dump zones
                                    min_dirt_tiles=10,  # Minimum dirt tiles like relocations_harder
                                    max_dirt_tiles=20): # Maximum dirt tiles like relocations_harder
    """
    Creates hybrid foundation environments with pre-placed dirt (like relocations_harder) 
    AND dig zones (foundations) using 1.0 downsampling factor.

    This creates maps where:
    - Excavators can dig new dirt from foundations (dig zones)
    - Skidsteers can immediately start moving pre-placed dirt
    - Both agents have work from day 1!

    Parameters:
    - n_imgs (int): Number of images to generate (default: 600)
    - max_size (int): Maximum size of the images (default: 64)
    - dataset_path (str): Path to the dataset (default: "data/openstreet")
    - n_obs_min (int): Minimum number of obstacles to add (default: 1)
    - n_obs_max (int): Maximum number of obstacles to add (default: 2)
    - size_obstacle_min (int): Minimum size of obstacles (default: 4)
    - size_obstacle_max (int): Maximum size of obstacles (default: 7)
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
    - size_dump_min (int): Minimum size of specific dump zones (default: 16)
    - size_dump_max (int): Maximum size of specific dump zones (default: 16)
    - min_dirt_tiles (int): Minimum pre-placed dirt tiles (default: 40)
    - max_dirt_tiles (int): Maximum pre-placed dirt tiles (default: 50)
    """

    # Define save folder for the envs using os.path.join
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", name_string)
    print(f"Creating hybrid foundations with pre-placed dirt - saving to: {name_string}/")

    # Force clean the save folder to ensure overwrite
    if os.path.exists(save_folder):
        print(f"Cleaning existing output folder: {save_folder}")
        shutil.rmtree(save_folder)

    # Use 2x downsampling like other foundation scripts  
    downsampling_factors = {
        save_folder: 2,
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

            print(f"Processing hybrid foundation nr {i + 1}")

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
            # Use 2x downsampling like other foundation scripts
            downsample_factor_w = int(max(1, math.ceil(img.shape[1] / max_size)) * downsampling_factor)
            downsample_factor_h = int(max(1, math.ceil(img.shape[0] / max_size)) * downsampling_factor)

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
            dig_mask = np.all(img_terra_pad == color_dict["digging"], axis=-1)
            
            # Create exactly 1 dump zone with no overlaps
            img_terra_pad, dump_cumulative_mask = create_single_dump_zone(
                img_terra_pad, size_dump_min, size_dump_max, foundation_mask
            )

            # Initialize cumulative mask with dump zones, dig zones, and foundation buildings
            cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
            # Mark dig zones (white areas) and dump zones as occupied
            cumulative_mask[np.all(img_terra_pad == color_dict["digging"], axis=-1)] = True
            cumulative_mask = dump_cumulative_mask | cumulative_mask
            # Also mark foundation buildings as occupied to prevent dirt overlap
            cumulative_mask = foundation_mask | cumulative_mask
            
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

            # *** NEW: Add pre-placed dirt tiles like relocations_harder ***
            # Random total dirt tiles between min_dirt_tiles and max_dirt_tiles
            total_dirt_tiles = np.random.randint(min_dirt_tiles, max_dirt_tiles + 1)
            action_map, cumulative_mask = add_dirt_tiles_hybrid(
                img_terra_pad, occ, dmp, cumulative_mask, total_dirt_tiles
            )

            if (no_dump_zones):
                #neutral_mask = np.all(img_terra_pad == color_dict["neutral"], axis=-1)
                img_terra_pad[neutral_mask] = color_dict["dumping"]
                dump_cumulative_mask = neutral_mask
            
            # Save the image and action map containing pre-placed dirt
            # Save main image, occupancy, and dumpability
            Path(curriculum_level, "images").mkdir(parents=True, exist_ok=True)
            Path(curriculum_level, "metadata").mkdir(parents=True, exist_ok=True)
            Path(curriculum_level, "occupancy").mkdir(parents=True, exist_ok=True)
            Path(curriculum_level, "dumpability").mkdir(parents=True, exist_ok=True)
            Path(curriculum_level, "actions").mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(Path(curriculum_level, "images", f"trench_{n}.png")), img_terra_pad)
            cv2.imwrite(str(Path(curriculum_level, "occupancy", f"trench_{n}.png")), occ)
            cv2.imwrite(str(Path(curriculum_level, "dumpability", f"trench_{n}.png")), dmp)
            cv2.imwrite(str(Path(curriculum_level, "actions", f"trench_{n}.png")), action_map)

            # Modify metadata to include curriculum level information
            metadata_with_curriculum = metadata.copy()
            metadata_with_curriculum["curriculum_level"] = curriculum_level
            metadata_with_curriculum.update(build_foundation_border_metadata(dig_mask))

            with open(
                Path(curriculum_level, "metadata", f"trench_{n}.json"), "w"
            ) as json_file:
                json.dump(metadata_with_curriculum, json_file)

        print("Hybrid foundations with pre-placed dirt created successfully.")


def generate_foundations_hybrid_standalone(config_path="config/env_generation_config.yaml",
                                         generate_terra_format=True,
                                         no_dump_zones=False):
    """
    Generate hybrid foundation maps with pre-placed dirt using standalone configuration.

    Args:
        config_path: Path to the configuration file
        generate_terra_format: Whether to convert to Terra format
        no_dump_zones: Whether to skip dump zones
    """
    # Get the package directory
    package_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Load configuration
    with open(package_dir + "/" + config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create necessary directories
    os.makedirs("data/", exist_ok=True)
    os.makedirs("data/terra/", exist_ok=True)
    os.makedirs("data/openstreet/", exist_ok=True)

    n_imgs = config["n_imgs"]

    print("Generating HYBRID FOUNDATIONS WITH PRE-PLACED DIRT maps...")

    # Read foundation parameters from the config file
    foundations_config = config.get("foundations", {})
    if "min_size" in foundations_config and "max_size" in foundations_config:
        foundation_min_size = foundations_config.get("min_size")
        foundation_max_size = foundations_config.get("max_size")
    else:
        raise ValueError("min_size and max_size must be provided in the config file")
    max_buildings = 150  # Limit to a manageable number

    print(f"Foundation config - min_size: {foundation_min_size}, max_size: {foundation_max_size}, max_buildings: {max_buildings}")

    # Use existing foundation data (no new downloads here)
    dataset_folder = os.path.join(package_dir, "data", "openstreet")

    # Check if foundation data exists
    foundation_path = os.path.join(dataset_folder, "foundations", "images")
    if not os.path.exists(foundation_path):
        from terra.env_generation.generate_foundations import download_foundations, create_foundations
        print("Foundation data not found. Downloading and creating foundation data...")
        bbox = config.get("center_bbox", (47.5376, 47.6126, 7.5401, 7.6842))
        download_foundations(
            dataset_folder,
            min_size=(foundation_min_size, foundation_min_size),
            max_size=(foundation_max_size, foundation_max_size),
            center_bbox=bbox,
            max_buildings=max_buildings,
        )
        create_foundations(dataset_folder)
    else:
        print(f"Using existing foundation data from: {foundation_path}")
        foundation_count = len([f for f in os.listdir(foundation_path) if f.endswith('.png')])
        print(f"Found {foundation_count} existing foundation images")

    # Generate hybrid foundations with pre-placed dirt
    print("  → Generating HYBRID FOUNDATIONS maps...")
    create_foundations_hybrid_standalone(
        n_imgs=n_imgs,
        max_size=foundations_config.get("max_size", 64),
        no_dump_zones=no_dump_zones,
        
    )
    print(f"  ✓ Hybrid foundations maps saved to: data/terra/{name_string}")

    # === TERRA FORMAT CONVERSION ===
    if generate_terra_format:
        print("Converting hybrid foundations data to Terra format...")
        sizes = [(64, 64)]  # Default size
        npy_dataset_folder = package_dir + "/data/terra"
        for size in sizes:
            # Convert generated maps using internal converter
            foundations_dir = Path(npy_dataset_folder) / name_string
            if not foundations_dir.exists():
                print(f"  Skipping conversion; folder not found: {foundations_dir}")
                continue
            destination_folder = Path(npy_dataset_folder) / "train" / name_string
            # Force clean the destination folder to ensure overwrite
            if destination_folder.exists():
                print(f"  Cleaning existing Terra destination: {destination_folder}")
                shutil.rmtree(destination_folder)
            img_folder = foundations_dir / "images"
            metadata_folder = foundations_dir / "metadata"
            occupancy_folder = foundations_dir / "occupancy"
            dumpability_folder = foundations_dir / "dumpability"
            actions_folder = foundations_dir / "actions"  # NEW: Include actions folder for pre-placed dirt
            destination_folder.mkdir(parents=True, exist_ok=True)
            
            # Use the existing conversion function
            convert_to_terra._convert_all_imgs_to_terra(
                img_folder,
                metadata_folder,
                occupancy_folder,
                dumpability_folder,
                destination_folder,
                size,
                n_imgs,
                all_dumpable=False,
                copy_metadata=True,
                downsample=False,
                has_dumpability=True,
                center_padding=False,
                actions_folder=actions_folder,  # Include the actions folder with pre-placed dirt
            )
        print("  ✓ Terra format conversion complete")

    print("Hybrid foundations generation complete!")
    print(f"Data saved to {os.path.join(package_dir, 'data/terra/{name_string}')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hybrid foundation maps with pre-placed dirt.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/env_generation_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--no-terra-format",
        action="store_true",
        help="Skip Terra format conversion",
    )
    parser.add_argument(
        "--no-dump-zones",
        action="store_true",
        help="Skip dump zones",
    )

    args = parser.parse_args()

    # Determine Terra format conversion
    generate_terra_format = not args.no_terra_format
    if args.no_terra_format:
        print("Terra format conversion disabled by --no-terra-format flag")
    else:
        print("Terra format conversion enabled (use --no-terra-format to disable)")

    no_dump_zones = args.no_dump_zones
    name_string = "foundations_dumpzones_v3"

    generate_foundations_hybrid_standalone(
        args.config,
        generate_terra_format=generate_terra_format,
        no_dump_zones=no_dump_zones,
    )
