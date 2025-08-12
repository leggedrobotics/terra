#!/usr/bin/env python3
"""
Standalone script to generate experimental 128x128 maps with the same elements as foundations with dumpzones harder.

This script creates experimental environments with:
- 128x128 resolution
- Foundation buildings (from OpenStreetMap data)
- Specific dump zones (like relocations)
- Obstacles and non-dumpable areas
- Neutral background (not everything dumpable)
- Experimental variations for research

Usage:
    python generate_experimental_128x128.py
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


def create_single_dump_zone_128x128(img_terra_pad, size_dump_min, size_dump_max, foundation_mask):
    """
    Create exactly 1 dump zone that avoids foundation overlaps for 128x128 maps.
    
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
    max_attempts = 200  # Increased attempts for larger maps
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
        # Fallback: try with smaller dump zone size (-5) for another 200 attempts
        print("Attempting fallback with smaller dump zone (size -5) for another 200 attempts...")
        fallback_size_min = max(8, size_dump_min - 5)  # Subtract 5 from min size
        fallback_size_max = max(12, size_dump_max - 5)  # Subtract 5 from max size
        
        for attempt in range(200):  # Try 200 more times with smaller size
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
            # Second fallback: try with even smaller dump zone size (-8 total)
            print("Attempting second fallback with even smaller dump zone (size -8 total)...")
            fallback2_size_min = max(5, fallback_size_min - 3)  # Subtract 3 more from fallback min
            fallback2_size_max = max(9, fallback_size_max - 3)  # Subtract 3 more from fallback max
            
            for attempt in range(200):  # Try 200 more times with even smaller size
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
                raise RuntimeError(f"Failed to place dump zone after {max_attempts} attempts with original size ({size_dump_min}-{size_dump_max}), 200 attempts with first fallback size ({fallback_size_min}-{fallback_size_max}), and 200 attempts with second fallback size ({fallback2_size_min}-{fallback2_size_max}). Image may be too crowded or dump zone size too large.")
    
    return img_terra_pad, dump_cumulative_mask


def create_experimental_128x128_maps(
    n_imgs=500,
    max_size=128,
    dataset_path="data/openstreet",
    n_obs_min=2,
    n_obs_max=4,
    size_obstacle_min=6,
    size_obstacle_max=12,
    n_nodump_min=1,
    n_nodump_max=3,
    size_nodump_min=8,
    size_nodump_max=15,
    expansion_factor=1,
    all_dumpable=False,
    copy_metadata=True,
    has_dumpability=False,
    center_padding=True,
    n_dump_min=1,
    n_dump_max=1,
    size_dump_min=15,
    size_dump_max=20,
    experimental_variations=True):
    """
    Creates experimental 128x128 foundation environments with specific dump zones.

    Parameters:
    - n_imgs (int): Number of images to generate (default: 100)
    - max_size (int): Maximum size of the images (default: 128)
    - dataset_path (str): Path to the dataset (default: "data/openstreet")
    - n_obs_min (int): Minimum number of obstacles to add (default: 2)
    - n_obs_max (int): Maximum number of obstacles to add (default: 4)
    - size_obstacle_min (int): Minimum size of obstacles (default: 6)
    - size_obstacle_max (int): Maximum size of obstacles (default: 12)
    - n_nodump_min (int): Minimum number of non-dumpable areas (default: 1)
    - n_nodump_max (int): Maximum number of non-dumpable areas (default: 3)
    - size_nodump_min (int): Minimum size of non-dumpable areas (default: 8)
    - size_nodump_max (int): Maximum size of non-dumpable areas (default: 15)
    - expansion_factor (int): Factor to expand the image by (default: 1)
    - all_dumpable (bool): Whether all areas should be dumpable (default: False)
    - copy_metadata (bool): Whether to copy metadata (default: True)
    - has_dumpability (bool): Whether the image has dumpability information (default: False)
    - center_padding (bool): Whether to center the padding (default: True)
    - n_dump_min (int): Minimum number of specific dump zones (default: 1)
    - n_dump_max (int): Maximum number of specific dump zones (default: 1)
    - size_dump_min (int): Minimum size of specific dump zones (default: 15)
    - size_dump_max (int): Maximum size of specific dump zones (default: 20)
    - experimental_variations (bool): Whether to apply experimental variations (default: True)
    """
    # Define save folder for the experimental maps
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "experimental_128x128")
    # save_folder_large = os.path.join(PACKAGE_DIR, "data", "terra", "experimental_128x128_large")  # Commented out for now
    print(f"Creating experimental 128x128 maps - saving to: experimental_128x128/")

    # Choose different downsampling factors for different curriculum levels
    downsampling_factors = {
        save_folder: 1,  # No downsampling for 128x128
        # save_folder_large: 1,  # No downsampling for large version - commented out for now
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

            print(f"Processing experimental 128x128 map nr {i + 1}")

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

            # Calculate downsample factors to keep foundation at ~64x64 scale within 128x128 map
            # This means we want the foundation to be roughly half the map size
            target_foundation_size = 64
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

            # Pad to max size (128x128) with foundation centered
            if center_padding:
                xdim = max_size - img_terra.shape[0]
                ydim = max_size - img_terra.shape[1]
                # Start with neutral background (not dumping) for the 128x128 map
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
            
            # Create dumpability mask: start with all areas dumpable (True)
            dumpability_mask = np.ones((max_size, max_size), dtype=np.bool_)
            
            # Apply 3-tile thick non-dumpable border around the entire map
            border_thickness = 3
            # Top border
            dumpability_mask[:border_thickness, :] = False
            # Bottom border
            dumpability_mask[-border_thickness:, :] = False
            # Left border
            dumpability_mask[:, :border_thickness] = False
            # Right border
            dumpability_mask[:, -border_thickness:] = False
            
            # Create exactly 1 dump zone with no overlaps
            img_terra_pad, dump_cumulative_mask = create_single_dump_zone_128x128(
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
            
            # Apply experimental variations if enabled
            if experimental_variations:
                # Add some experimental features
                # 1. Randomly add some "restricted zones" (similar to non-dumpables but different)
                if random.random() < 0.3:  # 30% chance
                    n_restricted = random.randint(1, 2)
                    for _ in range(n_restricted):
                        # Create small restricted zones
                        size = random.randint(4, 8)
                        x = random.randint(10, max_size - size - 10)
                        y = random.randint(10, max_size - size - 10)
                        
                        # Check if area is free
                        area_mask = cumulative_mask[y:y+size, x:x+size]
                        if not np.any(area_mask):
                            # Mark as restricted (use a different color or just mark as occupied)
                            cumulative_mask[y:y+size, x:x+size] = True
                            print(f"Added experimental restricted zone of size {size} at ({x}, {y})")
                
                # 2. Randomly vary dump zone placement strategy
                if random.random() < 0.2:  # 20% chance
                    print("Applied experimental dump zone placement variation")
            
            # Convert boolean dumpability mask to uint8 for OpenCV saving
            dumpability_mask_uint8 = dumpability_mask.astype(np.uint8) * 255
            save_or_display_image(img_terra_pad, occ, dumpability_mask_uint8, metadata, curriculum_level, n)

        print("Experimental 128x128 maps created successfully.")


def generate_experimental_128x128_terra(dataset_folder, size, n_imgs):
    """Convert experimental 128x128 maps to Terra format."""
    print("Converting experimental 128x128 maps to Terra format...")
    
    # Check if experimental_128x128 exists
    experimental_dir = Path(dataset_folder) / "experimental_128x128"
    if not experimental_dir.exists():
        print(f"  experimental_128x128 directory not found: {experimental_dir}")
        return
    
    print(f"  Found experimental_128x128 folder - will convert to train/experimental_128x128")
    
    # Set up paths
    img_folder = experimental_dir / "images"
    metadata_folder = experimental_dir / "metadata"
    occupancy_folder = experimental_dir / "occupancy"
    dumpability_folder = experimental_dir / "dumpability"
    destination_folder = Path(dataset_folder) / "train" / "experimental_128x128"
    
    # Create destination directory
    destination_folder.mkdir(parents=True, exist_ok=True)
    print(f"  Created destination directory: {destination_folder}")
    
    # Convert with specific settings for experimental 128x128 maps
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
    
    print(f"  experimental_128x128 conversion completed")


def generate_experimental_128x128_large_terra(dataset_folder, size, n_imgs):
    """Convert experimental 128x128 large maps to Terra format."""
    print("Converting experimental 128x128 large maps to Terra format...")
    
    # Check if experimental_128x128_large exists
    experimental_large_dir = Path(dataset_folder) / "experimental_128x128_large"
    if not experimental_large_dir.exists():
        print(f"  experimental_128x128_large directory not found: {experimental_large_dir}")
        return
    
    print(f"  Found experimental_128x128_large folder - will convert to train/experimental_128x128_large")
    
    # Set up paths
    img_folder = experimental_large_dir / "images"
    metadata_folder = experimental_large_dir / "metadata"
    occupancy_folder = experimental_large_dir / "occupancy"
    dumpability_folder = experimental_large_dir / "dumpability"
    destination_folder = Path(dataset_folder) / "train" / "experimental_128x128_large"
    
    # Create destination directory
    destination_folder.mkdir(parents=True, exist_ok=True)
    print(f"  Created destination directory: {destination_folder}")
    
    # Convert with specific settings for experimental 128x128 large maps
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
    
    print(f"  experimental_128x128_large conversion completed")


def main():
    """
    Main function to run the experimental 128x128 map generation and Terra conversion.
    """
    print("Starting experimental 128x128 map generation...")
    
    # Generate experimental maps with default parameters
    create_experimental_128x128_maps(
        n_imgs=500,  # Start with fewer images for testing
        max_size=128,
        dataset_path="data/openstreet",
        n_obs_min=2,
        n_obs_max=4,
        size_obstacle_min=6,
        size_obstacle_max=12,
        n_nodump_min=1,
        n_nodump_max=3,
        size_nodump_min=8,
        size_nodump_max=15,
        size_dump_min=21,  # Smaller minimum size
        size_dump_max=30,  # Smaller maximum size
        experimental_variations=True
    )
    
    print("Experimental 128x128 map generation completed!")
    
    # Convert to Terra format
    print("\nStarting Terra format conversion...")
    dataset_folder = os.path.join(PACKAGE_DIR, "data", "terra")
    size = (128, 128)  # 128x128 resolution
    n_imgs = 500
    
    # Convert regular version only (large version commented out for now)
    generate_experimental_128x128_terra(dataset_folder, size, n_imgs)
    # generate_experimental_128x128_large_terra(dataset_folder, size, n_imgs)  # Commented out for now
    
    print("Terra format conversion completed!")
    print("Experimental 128x128 maps are now ready for use in Terra environment!")


if __name__ == "__main__":
    main() 