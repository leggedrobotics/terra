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


name_string = ""

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
    
    # Try to place dump zone in valid areas
    max_attempts = 100
    for attempt in range(max_attempts):
        # Random dump zone size
        dump_size = random.randint(size_dump_min, size_dump_max)
        
        # Random position anywhere on the image
        x = random.randint(0, width - dump_size)
        y = random.randint(0, height - dump_size)
        
        # Check if it overlaps with dig zones (foundations)
        dig_area = np.all(img_terra_pad[y:y+dump_size, x:x+dump_size] == color_dict["digging"], axis=-1)
        
        # Place the dump zone if it doesn't overlap with dig zones (foundations)
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
            x = random.randint(0, width - dump_size)
            y = random.randint(0, height - dump_size)
            
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
                x = random.randint(0, width - dump_size)
                y = random.randint(0, height - dump_size)
                
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


def create_foundations_dumpzones_standalone(
                                    n_imgs=600,
                                    max_size=64,
                                    dataset_path="data/openstreet",
                                    n_obs_min=1,
                                    n_obs_max=2,
                                    size_obstacle_min=4,
                                    size_obstacle_max=7,
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
                                    size_dump_min=16,  # Bigger dump zones
                                    size_dump_max=16):  # Bigger dump zones
    """
    Creates foundation environments with specific dump zones using standalone parameters.
    Modified version with bigger dump zones (size 20) and downsampling factor 1.5.

    Parameters:
    - n_imgs (int): Number of images to generate (default: 100)
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
    - size_dump_min (int): Minimum size of specific dump zones (default: 20)
    - size_dump_max (int): Maximum size of specific dump zones (default: 20)
    """


    


    # Define save folder for the envs using os.path.join
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", name_string)
    print(f"Creating foundations with specific dump zones - saving to: {name_string}/")

    # Use downsampling factor 1.5 instead of 2 for larger foundations
    downsampling_factors = {
        save_folder: 1.5,
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


            if (no_dump_zones):
                #neutral_mask = np.all(img_terra_pad == color_dict["neutral"], axis=-1)
                img_terra_pad[neutral_mask] = color_dict["dumping"]
                dump_cumulative_mask = neutral_mask
            
            save_or_display_image(img_terra_pad, occ, dmp, metadata, curriculum_level, n)

        print("Foundations with dump zones created successfully.")


def generate_foundations_dumpzones_standalone(config_path="config/env_generation/config.yml",
                                            generate_terra_format=True,
                                            no_dump_zones=False):
    """
    Generate foundation maps with specific dump zones using standalone configuration.

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

    print("Generating FOUNDATIONS DUMPZONES STANDALONE maps...")

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

    # Generate foundations with dumpzones
    print("  → Generating FOUNDATIONS DUMPZONES 1.5 maps...")
    create_foundations_dumpzones_standalone(
        n_imgs=n_imgs,
        max_size=foundations_config.get("max_size", 64),
        no_dump_zones=no_dump_zones,
    )
    print("  ✓ Foundations dumpzones 1.5 maps saved to: data/terra/{name_string}")

    # === TERRA FORMAT CONVERSION ===
    if generate_terra_format:
        print("Converting foundations dumpzones standalone data to Terra format...")
        sizes = [(64, 64)]  # Default size
        npy_dataset_folder = package_dir + "/data/terra"
        for size in sizes:
            # Convert generated maps using internal converter
            foundations_dir = Path(npy_dataset_folder) / name_string
            if not foundations_dir.exists():
                print(f"  Skipping conversion; folder not found: {foundations_dir}")
                continue
            destination_folder = Path(npy_dataset_folder) / "train" / name_string
            img_folder = foundations_dir / "images"
            metadata_folder = foundations_dir / "metadata"
            occupancy_folder = foundations_dir / "occupancy"
            dumpability_folder = foundations_dir / "dumpability"
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
                actions_folder=None,
            )
        print("  ✓ Terra format conversion complete")

    print("Foundations 1.5 generation complete!")
    print(f"Data saved to {os.path.join(package_dir, 'data/terra/{name_string}')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate foundation maps with specific dump zones (standalone version).")
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
    name_string = "foundations_1.5" if no_dump_zones else "foundations_dumpzones_1.5"


    generate_foundations_dumpzones_standalone(
        args.config,
        generate_terra_format=generate_terra_format,
        no_dump_zones=no_dump_zones,
    ) 