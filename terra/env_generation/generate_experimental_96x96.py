#!/usr/bin/env python3
"""
Generate experimental 96x96 maps with foundations, dump zones, obstacles, and non-dumpables.
This is an intermediate size between 64x64 and 128x128 for better memory efficiency.
Includes all features from 128x128: foundation buildings, specific dump zones, 3-tile borders, obstacles, non-dumpables, and experimental variations.
"""

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
    _convert_all_imgs_to_terra,
)
from terra.env_generation.utils import _get_img_mask, color_dict

# Define package directory at module level
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_single_dump_zone_96x96(img_terra_pad, size_dump_min, size_dump_max, foundation_mask):
    """
    Create exactly 1 dump zone that avoids foundation overlaps for 96x96 maps.
    
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
    
    # Offset from borders - 3 tiles thick no-dump zone border
    border_offset = 3
    
    # Try to place dump zone in valid areas
    max_attempts = 150  # Adjusted for 96x96 maps
    for attempt in range(max_attempts):
        # Random rectangular dump zone size
        dump_width = random.randint(size_dump_min, size_dump_max)
        dump_height = random.randint(size_dump_min, size_dump_max)
        
        # Place dump zone at border with border_offset amount of offset
        # Randomly choose which border (top, bottom, left, right)
        border_choice = random.randint(0, 3)  # 0=top, 1=bottom, 2=left, 3=right
        
        if border_choice == 0:  # Top border
            x = random.randint(border_offset, width - dump_width - border_offset)
            y = border_offset
        elif border_choice == 1:  # Bottom border
            x = random.randint(border_offset, width - dump_width - border_offset)
            y = height - dump_height - border_offset
        elif border_choice == 2:  # Left border
            x = border_offset
            y = random.randint(border_offset, height - dump_height - border_offset)
        else:  # Right border
            x = width - dump_width - border_offset
            y = random.randint(border_offset, height - dump_height - border_offset)
        
        # Check if this area overlaps with foundations
        dump_area = foundation_mask[y:y+dump_height, x:x+dump_width]
        
        # Also check if it overlaps with dig zones
        dig_area = np.all(img_terra_pad[y:y+dump_height, x:x+dump_width] == color_dict["digging"], axis=-1)
        
        # If no overlaps, place the dump zone
        if not np.any(dump_area) and not np.any(dig_area):
            # Create dump zone
            img_terra_pad[y:y+dump_height, x:x+dump_width] = color_dict["dumping"]
            dump_cumulative_mask[y:y+dump_height, x:x+dump_width] = True
            print(f"Successfully placed rectangular dump zone of size {dump_width}x{dump_height} at position ({x}, {y})")
            
            # Note: 3-tile non-dumpable border will be applied to dumpability mask, not image
            # The image shows visual representation, dumpability mask controls actual dumping
            
            break
    else:
        print(f"Warning: Could not place dump zone after {max_attempts} attempts")
        # Fallback: try with smaller dump zone size (-4) for another 150 attempts
        print("Attempting fallback with smaller dump zone (size -4) for another 150 attempts...")
        fallback_size_min = max(6, size_dump_min - 4)  # Subtract 4 from min size
        fallback_size_max = max(10, size_dump_max - 4)  # Subtract 4 from max size
        
        for attempt in range(150):  # Try 150 more times with smaller size
            dump_width = random.randint(fallback_size_min, fallback_size_max)
            dump_height = random.randint(fallback_size_min, fallback_size_max)
            
            # Place dump zone at border with border_offset amount of offset
            border_choice = random.randint(0, 3)  # 0=top, 1=bottom, 2=left, 3=right
            
            if border_choice == 0:  # Top border
                x = random.randint(border_offset, width - dump_width - border_offset)
                y = border_offset
            elif border_choice == 1:  # Bottom border
                x = random.randint(border_offset, width - dump_width - border_offset)
                y = height - dump_height - border_offset
            elif border_choice == 2:  # Left border
                x = border_offset
                y = random.randint(border_offset, height - dump_height - border_offset)
            else:  # Right border
                x = width - dump_width - border_offset
                y = random.randint(border_offset, height - dump_height - border_offset)
            
            dump_area = foundation_mask[y:y+dump_height, x:x+dump_width]
            dig_area = np.all(img_terra_pad[y:y+dump_height, x:x+dump_width] == color_dict["digging"], axis=-1)
            
            if not np.any(dump_area) and not np.any(dig_area):
                img_terra_pad[y:y+dump_height, x:x+dump_width] = color_dict["dumping"]
                dump_cumulative_mask[y:y+dump_height, x:x+dump_width] = True
                print(f"Fallback successful: placed smaller rectangular dump zone of size {dump_width}x{dump_height} at position ({x}, {y})")
                
                # Note: 3-tile non-dumpable border will be applied to dumpability mask, not image
                
                break
        else:
            # Second fallback: try with even smaller dump zone size (-6 total)
            print("Attempting second fallback with even smaller dump zone (size -6 total)...")
            fallback2_size_min = max(4, fallback_size_min - 2)  # Subtract 2 more from fallback min
            fallback2_size_max = max(8, fallback_size_max - 2)  # Subtract 2 more from fallback max
            
            for attempt in range(150):  # Try 150 more times with even smaller size
                dump_width = random.randint(fallback2_size_min, fallback2_size_max)
                dump_height = random.randint(fallback2_size_min, fallback2_size_max)
                
                # Place dump zone at border with border_offset amount of offset
                border_choice = random.randint(0, 3)  # 0=top, 1=bottom, 2=left, 3=right
                
                if border_choice == 0:  # Top border
                    x = random.randint(border_offset, width - dump_width - border_offset)
                    y = border_offset
                elif border_choice == 1:  # Bottom border
                    x = random.randint(border_offset, width - dump_width - border_offset)
                    y = height - dump_height - border_offset
                elif border_choice == 2:  # Left border
                    x = border_offset
                    y = random.randint(border_offset, height - dump_height - border_offset)
                else:  # Right border
                    x = width - dump_width - border_offset
                    y = random.randint(border_offset, height - dump_height - border_offset)
                
                dump_area = foundation_mask[y:y+dump_height, x:x+dump_width]
                dig_area = np.all(img_terra_pad[y:y+dump_height, x:x+dump_width] == color_dict["digging"], axis=-1)
                
                if not np.any(dump_area) and not np.any(dig_area):
                    img_terra_pad[y:y+dump_height, x:x+dump_width] = color_dict["dumping"]
                    dump_cumulative_mask[y:y+dump_height, x:x+dump_width] = True
                    print(f"Second fallback successful: placed even smaller rectangular dump zone of size {dump_width}x{dump_height} at position ({x}, {y})")
                    
                    # Note: 3-tile non-dumpable border will be applied to dumpability mask, not image
                    
                    break
            else:
                # If all fallbacks fail, throw an error
                raise RuntimeError(f"Failed to place dump zone after {max_attempts} attempts with original size ({size_dump_min}-{size_dump_max}), 150 attempts with first fallback size ({fallback_size_min}-{fallback_size_max}), and 150 attempts with second fallback size ({fallback2_size_min}-{fallback2_size_max}). Image may be too crowded or dump zone size too large.")
    
    return img_terra_pad, dump_cumulative_mask


def create_experimental_96x96_maps(
    n_imgs=300,
    max_size=96,
    dataset_path="data/openstreet",
    n_obs_min=2,
    n_obs_max=4,
    size_obstacle_min=4,
    size_obstacle_max=10,
    expansion_factor=1,
    all_dumpable=False,
    copy_metadata=True,
    has_dumpability=False,
    center_padding=True,
    n_dump_min=1,
    n_dump_max=1,
    size_dump_min=12,
    size_dump_max=18,
    experimental_variations=True):
    """
    Creates experimental 96x96 foundation environments with specific dump zones.
    Includes all features from 128x128: foundation buildings, specific dump zones, 3-tile borders, obstacles, custom roads, and experimental variations.

    Parameters:
    - n_imgs (int): Number of images to generate (default: 300)
    - max_size (int): Maximum size of the images (default: 96)
    - dataset_path (str): Path to the dataset (default: "data/openstreet")
    - n_obs_min (int): Minimum number of obstacles to add (default: 2)
    - n_obs_max (int): Maximum number of obstacles to add (default: 4)
    - size_obstacle_min (int): Minimum size of obstacles (default: 4)
    - size_obstacle_max (int): Maximum size of obstacles (default: 10)
    - expansion_factor (int): Factor to expand the image by (default: 1)
    - all_dumpable (bool): Whether all areas should be dumpable (default: False)
    - copy_metadata (bool): Whether to copy metadata (default: True)
    - has_dumpability (bool): Whether the image has dumpability information (default: False)
    - center_padding (bool): Whether to center the padding (default: True)
    - n_dump_min (int): Minimum number of specific dump zones (default: 1)
    - n_dump_max (int): Maximum number of specific dump zones (default: 1)
    - size_dump_min (int): Minimum size of specific dump zones (default: 12)
    - size_dump_max (int): Maximum size of specific dump zones (default: 18)
    - experimental_variations (bool): Whether to apply experimental variations (default: True)
    """
    # Define save folder for the experimental maps
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "experimental_96x96")
    print(f"Creating experimental 96x96 maps - saving to: experimental_96x96/")

    # Choose different downsampling factors for different curriculum levels
    downsampling_factors = {
        save_folder: 1,  # No downsampling for 96x96
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

            print(f"Processing experimental 96x96 map nr {i + 1}")

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

            # Calculate downsample factors to keep foundation at ~48x48 scale within 96x96 map
            # This means we want the foundation to be roughly half the map size
            target_foundation_size = 48
            downsample_factor_w = int(max(1, math.ceil(img.shape[1] / target_foundation_size))) * downsampling_factor
            downsample_factor_h = int(max(1, math.ceil(img.shape[0] / target_foundation_size))) * downsampling_factor

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

            # Pad to max size (96x96) with foundation centered
            if center_padding:
                xdim = max_size - img_terra.shape[0]
                ydim = max_size - img_terra.shape[1]
                # Start with neutral background (not dumping) for the 96x96 map
                img_terra_pad = np.zeros((max_size, max_size), dtype=img_terra.dtype)
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
            
            # Note: No dumpability mask needed - just like the working foundations script
            
            # Initialize cumulative mask with dig zones only (no dump zones yet)
            cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
            # Mark dig zones (white areas) as occupied
            cumulative_mask[np.all(img_terra_pad == color_dict["digging"], axis=-1)] = True
        
            # Add obstacles
            occ, cumulative_mask = add_obstacles(
                img_terra_pad,
                cumulative_mask,
                n_obs_min,
                n_obs_max,
                size_obstacle_min,
                size_obstacle_max,
            )
            
                        # Add custom roads first (but don't add them to cumulative_mask for occupancy)
            img_terra_pad, _, road_positions = create_roads_96x96(
                img_terra_pad,
                cumulative_mask,
                road_width=6
            )
            
            # Now create dump zones (can be placed on roads)
            img_terra_pad, dump_cumulative_mask = create_single_dump_zone_96x96(
                img_terra_pad, size_dump_min, size_dump_max, foundation_mask
            )
            
            # Update cumulative mask to include dump zones (but NOT roads)
            cumulative_mask = dump_cumulative_mask | cumulative_mask
            
            # Create dumpability mask - start with all areas as dumpable (white)
            dmp = np.ones_like(img_terra_pad) * 255
            
            # Mark road areas as non-dumpable (nondumpable color)
            # Roads are invisible in main image but we need to track them for dumpability
            road_mask = np.zeros_like(dump_cumulative_mask, dtype=bool)
            
            # Use actual road positions from create_roads_96x96
            for x, y, w, h in road_positions:
                road_mask[y:y+h, x:x+w] = True
            
            dmp[road_mask] = color_dict["nondumpable"]
            
            # Handle overlapping areas: where dump zones are placed on roads, make them neutral (dumpable)
            dump_mask = np.all(img_terra_pad == color_dict["dumping"], axis=-1)
            overlap_mask = road_mask & dump_mask
            
            if np.any(overlap_mask):
                # Make overlapping areas neutral (dumpable) in dumpability map
                dmp[overlap_mask] = 255  # White = dumpable
                print(f"Fixed {np.sum(overlap_mask)} overlapping tiles: made them dumpable in dumpability map")
            
            # Note: No need to update dumpability mask - just like the working foundations script
            
            # Apply experimental variations if enabled
            if experimental_variations:
                # Add some experimental features
                # 1. Randomly add some "restricted zones" (similar to non-dumpables but different)
                if random.random() < 0.3:  # 30% chance
                    n_restricted = random.randint(1, 2)
                    for _ in range(n_restricted):
                        # Create small restricted zones
                        size = random.randint(3, 6)
                        x = random.randint(8, max_size - size - 8)
                        y = random.randint(8, max_size - size - 8)
                        
                        # Check if area is free
                        area_mask = cumulative_mask[y:y+size, x:x+size]
                        if not np.any(area_mask):
                            # Mark as restricted (use a different color or just mark as occupied)
                            cumulative_mask[y:y+size, x:x+size] = True
                            print(f"Added experimental restricted zone of size {size} at ({x}, {y})")
                
                # 2. Randomly vary dump zone placement strategy
                if random.random() < 0.2:  # 20% chance
                    print("Applied experimental dump zone placement variation")
            
            # Save the image with obstacles and non-dumpables - just like the working foundations script
            save_or_display_image(img_terra_pad, occ, dmp, metadata, curriculum_level, n)

        print("Experimental 96x96 maps created successfully.")


def generate_experimental_96x96_terra(dataset_folder, size, n_imgs):
    """Convert experimental 96x96 maps to Terra format."""
    print("Converting experimental 96x96 maps to Terra format...")
    
    # Check if experimental_96x96 exists
    experimental_dir = Path(dataset_folder) / "experimental_96x96"
    if not experimental_dir.exists():
        print(f"  experimental_96x96 directory not found: {experimental_dir}")
        return
    
    print(f"  Found experimental_96x96 folder - will convert to train/experimental_96x96")
    
    # Set up paths
    img_folder = experimental_dir / "images"
    metadata_folder = experimental_dir / "metadata"
    occupancy_folder = experimental_dir / "occupancy"
    dumpability_folder = experimental_dir / "dumpability"
    destination_folder = Path(dataset_folder) / "train" / "experimental_96x96"
    
    # Create destination directory
    destination_folder.mkdir(parents=True, exist_ok=True)
    print(f"  Created destination directory: {destination_folder}")
    
    # Convert with specific settings for experimental 96x96 maps
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,  # Experimental maps use specific dump zones, not all dumpable
        copy_metadata=True,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=None,
    )
    
    print(f"  experimental_96x96 conversion completed")


def create_roads_96x96(img_terra_pad, cumulative_mask, road_width=6):
    """
    Create 1-2 roads with variety: single road, parallel roads, or crossing roads.
    Roads are placed near borders with consistent width, avoiding dump zones.
    
    Args:
        img_terra_pad: The image to add roads to
        cumulative_mask: Boolean mask of occupied areas
        road_width: Fixed width for all roads (default: 6)
        
    Returns:
        img_terra_pad: Updated image with roads
        cumulative_mask: Updated cumulative mask
        road_positions: List of road positions for dumpability map
    """
    height, width = img_terra_pad.shape[:2]
    
    # Randomly choose road configuration
    road_config = random.choice(["single", "parallel", "crossing"])
    
    roads_to_add = []
    
    if road_config == "single":
        # Single road - randomly choose direction and side
        if random.random() < 0.5:  # Horizontal road
            side = random.choice(["top", "bottom"])
            if side == "top":
                y_pos = random.randint(8, 20)
            else:  # bottom
                y_pos = random.randint(height-20, height-8)
            roads_to_add.append({"direction": "horizontal", "y": y_pos, "x": 0, "width": width, "height": road_width})
        else:  # Vertical road
            side = random.choice(["left", "right"])
            if side == "left":
                x_pos = random.randint(8, 20)
            else:  # right
                x_pos = random.randint(width-20, width-8)
            roads_to_add.append({"direction": "vertical", "x": x_pos, "y": 0, "width": road_width, "height": height})
    
    elif road_config == "parallel":
        # Two parallel roads (both horizontal or both vertical)
        if random.random() < 0.5:  # Two horizontal roads
            # First horizontal road
            y1 = random.randint(8, 20)
            roads_to_add.append({"direction": "horizontal", "y": y1, "x": 0, "width": width, "height": road_width})
            # Second horizontal road (different side)
            y2 = random.randint(height-20, height-8)
            roads_to_add.append({"direction": "horizontal", "y": y2, "x": 0, "width": width, "height": road_width})
        else:  # Two vertical roads
            # First vertical road
            x1 = random.randint(8, 20)
            roads_to_add.append({"direction": "vertical", "x": x1, "y": 0, "width": road_width, "height": height})
            # Second vertical road (different side)
            x2 = random.randint(width-20, width-8)
            roads_to_add.append({"direction": "vertical", "x": x2, "y": 0, "width": road_width, "height": height})
    
    else:  # crossing
        # Two crossing roads (1 horizontal + 1 vertical)
        # Horizontal road
        horizontal_side = random.choice(["top", "bottom"])
        if horizontal_side == "top":
            y_pos = random.randint(8, 20)
        else:  # bottom
            y_pos = random.randint(height-20, height-8)
        roads_to_add.append({"direction": "horizontal", "y": y_pos, "x": 0, "width": width, "height": road_width})
        
        # Vertical road
        vertical_side = random.choice(["left", "right"])
        if vertical_side == "left":
            x_pos = random.randint(8, 20)
        else:  # right
            x_pos = random.randint(width-20, width-8)
        roads_to_add.append({"direction": "vertical", "x": x_pos, "y": 0, "width": road_width, "height": height})
    
    roads_added = 0
    road_positions = []  # Track road positions for dumpability map
    
    for road in roads_to_add:
        x, y, w, h = road["x"], road["y"], road["width"], road["height"]
        
        # Check if area is free (roads can cross dump zones, so we only check for obstacles)
        obstacle_mask = np.all(img_terra_pad == color_dict["obstacle"], axis=-1)
        area_has_obstacles = np.any(obstacle_mask[y:y+h, x:x+w])
        
        if not area_has_obstacles:
            # Don't add road to main image - roads are only in dumpability map
            # Roads don't block digging, they're just non-dumpable areas
            road_positions.append((x, y, w, h))  # Store position for dumpability map
            roads_added += 1
            print(f"Added {road['direction']} road at ({x}, {y}) with size {w}x{h} (invisible, non-dumpable)")
        else:
            print(f"Warning: Could not place {road['direction']} road due to obstacles")
    
    return img_terra_pad, cumulative_mask, road_positions


def main():
    """
    Main function to run the experimental 96x96 map generation and Terra conversion.
    """
    print("Starting experimental 96x96 map generation...")
    
    # Generate experimental maps with default parameters
    create_experimental_96x96_maps(
        n_imgs=600,  # Start with fewer images for testing
        max_size=96,
        dataset_path="data/openstreet",
        n_obs_min=2,
        n_obs_max=4,
        size_obstacle_min=4,
        size_obstacle_max=10,
        size_dump_min=16,  # Bigger minimum size for 96x96
        size_dump_max=24,  # Bigger maximum size for 96x96
        experimental_variations=True
    )
    
    print("Experimental 96x96 map generation completed!")
    
    # Convert to Terra format
    print("\nStarting Terra format conversion...")
    dataset_folder = os.path.join(PACKAGE_DIR, "data", "terra")
    size = (96, 96)  # 96x96 resolution
    n_imgs = 600
    
    # Convert regular version
    generate_experimental_96x96_terra(dataset_folder, size, n_imgs)
    
    print("Terra format conversion completed!")
    print("Experimental 96x96 maps are now ready for use in Terra environment!")


if __name__ == "__main__":
    main() 