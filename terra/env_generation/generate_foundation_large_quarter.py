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
from pathlib import Path
from terra.env_generation.generate_foundations import download_foundations, create_foundations
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

def create_foundations_large_quarter(
                      n_imgs=100,
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
                      copy_metadata=True,
                      center_padding=True):
    """
    Creates foundation environments with quarter-sized dump zones (LARGE VERSION ONLY).
    Uses downsampling_factor=1 to create only large foundations.
    Quarter of the area becomes dumpable, rest becomes non-dumpable (except dig zones).
    
    Parameters:
    - n_imgs (int): Number of images to generate
    - max_size (int): Maximum size of the images
    - dataset_path (str): Path to the dataset
    - n_obs_min/max (int): Min/max number of obstacles
    - size_obstacle_min/max (int): Min/max size of obstacles  
    - n_nodump_min/max (int): Min/max number of non-dumpable areas
    - size_nodump_min/max (int): Min/max size of non-dumpable areas
    - expansion_factor (int): Factor to expand the image by
    - copy_metadata (bool): Whether to copy metadata
    - center_padding (bool): Whether to center the padding
    """
    
    # Define save folder for quarter dump zone foundations (large only)
    save_folder_large = os.path.join(PACKAGE_DIR, "data", "terra", "foundations_quarter_large")
    print(f"Using QUARTER DUMP ZONES mode (LARGE ONLY) - saving to: foundations_quarter_large/")
    
    # Only generate large version (downsampling_factor = 1)
    downsampling_factors = {
        save_folder_large: 1,  # Large version only
    }

    # Get the full dataset path
    full_dataset_path = os.path.join(PACKAGE_DIR, dataset_path)

    # Process foundation images
    size = max_size
    foundations_name = "foundations"
    img_folder = Path(full_dataset_path) / foundations_name / "images"
    metadata_folder = Path(full_dataset_path) / foundations_name / "metadata"
    occupancy_folder = Path(full_dataset_path) / foundations_name / "occupancy"
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

            with open(metadata_folder / f"{filename.split('.png')[0]}.json") as json_file:
                metadata = json.load(json_file)

            # Calculate downsample factors based on max_size
            downsample_factor_w = int(max(1, math.ceil(img.shape[1] / size))) * downsampling_factor
            downsample_factor_h = int(max(1, math.ceil(img.shape[0] / size))) * downsampling_factor

            img_downsampled = skimage.measure.block_reduce(
                img, (downsample_factor_h, downsample_factor_w, 1), np.max
            )
            img = img_downsampled
            occupancy_downsampled = skimage.measure.block_reduce(
                occupancy, (downsample_factor_h, downsample_factor_w, 1), np.min, cval=0
            )
            occupancy = occupancy_downsampled

            # Convert to terra format
            img_terra = _convert_img_to_terra(img, False)  # Not all dumpable

            # Pad to max size with center padding
            if center_padding:
                xdim = size - img_terra.shape[0]
                ydim = size - img_terra.shape[1]
                # Start with neutral background
                img_terra_pad = np.ones((size, size), dtype=img_terra.dtype)
                img_terra_pad[
                    xdim // 2 : size - (xdim - xdim // 2),
                    ydim // 2 : size - (ydim - ydim // 2),
                ] = img_terra
                img_terra_occupancy = np.zeros((size, size), dtype=np.bool_)
                img_terra_occupancy[
                    xdim // 2 : size - (xdim - xdim // 2),
                    ydim // 2 : size - (ydim - ydim // 2),
                ] = _convert_occupancy_to_terra(occupancy)
            else:
                img_terra_pad = np.zeros((size, size), dtype=img_terra.dtype)
                img_terra_pad[: img_terra.shape[0], : img_terra.shape[1]] = img_terra
                img_terra_occupancy = np.ones((size, size), dtype=np.bool_)
                img_terra_occupancy[: occupancy.shape[0], : occupancy.shape[1]] = (
                    _convert_occupancy_to_terra(occupancy)
                )

            # Apply expansion factor
            img_terra_pad = img_terra_pad.repeat(expansion_factor, 0).repeat(expansion_factor, 1)
            img_terra_pad = convert_terra_pad_to_color(img_terra_pad, color_dict)
            
            # QUARTER DUMP ZONE LOGIC
            # Start with neutral background everywhere except dig zones
            neutral_mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)
            img_terra_pad[neutral_mask] = color_dict["neutral"]
            
            # Create a quarter-sized dump zone in random corner
            quarter_size = size // 2  # Half the width/height = quarter of total area
            dump_cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
            
            # Randomly choose one of 4 corners: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
            corner = random.randint(0, 3)
            
            if corner == 0:  # Top-left
                x_start, x_end = 0, quarter_size
                y_start, y_end = 0, quarter_size
                corner_name = "top-left"
            elif corner == 1:  # Top-right
                x_start, x_end = quarter_size, size
                y_start, y_end = 0, quarter_size
                corner_name = "top-right"
            elif corner == 2:  # Bottom-left
                x_start, x_end = 0, quarter_size
                y_start, y_end = quarter_size, size
                corner_name = "bottom-left"
            else:  # Bottom-right
                x_start, x_end = quarter_size, size
                y_start, y_end = quarter_size, size
                corner_name = "bottom-right"
            
            # Create quarter dump zone in chosen corner, but exclude dig zones (foundations)
            quarter_area = img_terra_pad[y_start:y_end, x_start:x_end]
            dig_zone_in_quarter = np.all(quarter_area == color_dict["digging"], axis=-1)
            
            # Only place dump zone where there are no dig zones (foundations)
            dump_zone_mask = ~dig_zone_in_quarter  # Exclude dig zones
            quarter_area[dump_zone_mask] = color_dict["dumping"]
            img_terra_pad[y_start:y_end, x_start:x_end] = quarter_area
            
            dump_cumulative_mask[y_start:y_end, x_start:x_end] = dump_zone_mask
            print(f"Placed quarter dump zone in {corner_name} corner (excluding foundations)")
            
            # Keep the image neutral (don't mark non-dumpable areas visually)
            # Non-dumpable zones will be handled in the dumpability map instead

            # Initialize cumulative mask with dump zones and dig zones
            cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
            # Mark dig zones and dump zones as occupied
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
            
            # Create custom dumpability map for quarter dump zones
            # Priority order (highest to lowest): obstacles > foundations/dig zones > quarter dump zone > non-dumpable background
            # Foundations have priority over dump zones when they overlap
            
            # Start with all areas as non-dumpable (red)
            custom_dmp = np.full_like(img_terra_pad, color_dict["nondumpable"])
            
            # Make the chosen quarter dumpable (white) - but will be overridden by higher priorities
            custom_dmp[y_start:y_end, x_start:x_end] = [255, 255, 255]  # White = dumpable
            
            # HIGHER PRIORITY: Foundations/dig zones override dump zones when they overlap
            # Dig zones should be NON-DUMPABLE to completely exclude them from dump zones
            # This forces the agent to learn intermediate dumping strategies
            dig_mask = np.all(img_terra_pad == color_dict["digging"], axis=-1)
            custom_dmp[dig_mask] = [255, 255, 255]  
            
            # HIGHEST PRIORITY: Obstacles are always non-dumpable (override everything)
            obstacle_mask = np.all(img_terra_pad == color_dict["obstacle"], axis=-1)
            custom_dmp[obstacle_mask] = color_dict["nondumpable"]  # Red = non-dumpable
            
            # Save the result with custom dumpability map
            save_or_display_image(img_terra_pad, occ, custom_dmp, metadata, curriculum_level, n)

    print("Large quarter dump zone foundations created successfully.")

def generate_foundation_large_quarter(config_path="config/env_generation/config.yml", 
                                     generate_terra_format=True):
    """
    Generate foundation maps with quarter-sized dump zones (LARGE VERSION ONLY).
    The quarter area becomes dumpable, the rest becomes non-dumpable (except dig zones).
    
    Args:
        config_path: Path to the configuration file
        generate_terra_format: Whether to convert to Terra format
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

    print("Generating QUARTER DUMP ZONE foundation maps (LARGE VERSION ONLY)...")
    
    # Read foundation parameters from the config file
    foundations_config = config.get("foundations", {})
    if "min_size" in foundations_config and "max_size" in foundations_config:
        foundation_min_size = foundations_config.get("min_size")
        foundation_max_size = foundations_config.get("max_size")
    else:
        raise ValueError("min_size and max_size must be provided in the config file")
    max_buildings = 150  # Override config to only download 150 buildings for 100 maps
    
    print(f"Foundation config - min_size: {foundation_min_size}, max_size: {foundation_max_size}, max_buildings: {max_buildings} (overridden)")

    # Get bounding box from config, or use default
    bbox = config.get("center_bbox", (47.5376, 47.6126, 7.5401, 7.6842))

    # Use existing foundation data (no need to download again)
    dataset_folder = os.path.join(package_dir, "data", "openstreet")
    
    # Check if foundation data exists
    foundation_path = os.path.join(dataset_folder, "foundations", "images")
    if not os.path.exists(foundation_path):
        print("Foundation data not found. Downloading and creating foundation data...")
        download_foundations(
            dataset_folder,
            min_size=(foundation_min_size, foundation_min_size),
            max_size=(foundation_max_size, foundation_max_size),
            center_bbox=bbox,
            max_buildings=max_buildings
        )
        create_foundations(dataset_folder)
    else:
        print(f"Using existing foundation data from: {foundation_path}")
        foundation_count = len([f for f in os.listdir(foundation_path) if f.endswith('.png')])
        print(f"Found {foundation_count} existing foundation images")

    # Generate foundations with quarter-sized dump zones (LARGE ONLY)
    print("  → Generating QUARTER-SIZED DUMP ZONES foundation maps (LARGE ONLY)...")
    create_foundations_large_quarter(
        n_imgs=n_imgs,
        max_size=foundations_config.get("max_size", 64)
    )
    print("  ✓ Large quarter dump zone foundation maps saved to: data/terra/foundations_quarter_large/")

    # === TERRA FORMAT CONVERSION ===
    if generate_terra_format:
        print("Converting quarter dump zone data to Terra format...")
        sizes = [(64, 64)]  # Default size
        npy_dataset_folder = package_dir + "/data/terra"
        for size in sizes:
            # Convert the quarter dump zone foundations using dedicated function
            convert_to_terra.generate_foundations_quarter_large_terra(
                npy_dataset_folder, size, n_imgs
            )
        print("  ✓ Terra format conversion complete")

    print("Quarter dump zone foundation generation complete!")
    print(f"Data saved to {os.path.join(package_dir, 'data/terra/foundations_quarter_large')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate foundation maps with quarter-sized dump zones (harder version).")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/env_generation_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--no-terra-format", 
        action="store_true", 
        help="Skip Terra format conversion"
    )
    
    args = parser.parse_args()
    
    # Determine Terra format conversion
    generate_terra_format = not args.no_terra_format
    if args.no_terra_format:
        print("Terra format conversion disabled by --no-terra-format flag")
    else:
        print("Terra format conversion enabled (use --no-terra-format to disable)")

    generate_foundation_large_quarter(
        args.config,
        generate_terra_format=generate_terra_format
    ) 