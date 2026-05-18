#!/usr/bin/env python
import os
import yaml
import json
import math
import numpy as np
import cv2
import skimage
import argparse
from pathlib import Path
from terra.env_generation.procedural_data import (
    save_or_display_image,
    convert_terra_pad_to_color,
    add_obstacles,
)
from terra.env_generation.convert_to_terra import (
    _convert_img_to_terra,
    _convert_occupancy_to_terra,
    _convert_dumpability_to_terra,
    _convert_all_imgs_to_terra,
)
import terra.env_generation.convert_to_terra as convert_to_terra
from terra.env_generation.utils import color_dict

# Define package directory at module level
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_foundations_border_dumpzone(
                      n_imgs=100,
                      max_size=64,
                      dataset_path="data/openstreet",
                      expansion_factor=1,
                      copy_metadata=True,
                      center_padding=True):
    """
    Creates foundation environments with a 2-tile neutral border around foundations,
    and dumping (green) elsewhere.

    - Uses already downloaded buildings from foundations dataset
    - Generates corresponding dumpability maps

    Parameters:
    - n_imgs (int): Number of images to generate
    - max_size (int): Maximum size of the images
    - dataset_path (str): Path to the dataset
    - expansion_factor (int): Factor to expand the image by
    - copy_metadata (bool): Whether to copy metadata
    - center_padding (bool): Whether to center the padding
    """

    # Define save folder
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "foundations_border_dumpzone")
    print(f"Using BORDER DUMPZONE mode - saving to: foundations_border_dumpzone/")

    # Use small foundations (downsampling_factor = 2) like the original foundations script
    downsampling_factors = {
        save_folder: 2,
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

            # Convert to terra format (values -1,0,1)
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

            # Convert to color for painting
            img_color = convert_terra_pad_to_color(img_terra_pad, color_dict)

            # Build masks
            dig_mask = np.all(img_color == color_dict["digging"], axis=-1)

            # Start by making everything dumping
            img_color[:, :] = color_dict["dumping"]

            # Put the dig zones back
            img_color[dig_mask] = color_dict["digging"]

                        # Create a 16-tile neutral border around dig zones
            # Use 33x33 kernel for Chebyshev distance <= 16
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (33, 33))
            dilated = cv2.dilate(dig_mask.astype(np.uint8), kernel)
            border_mask = (dilated.astype(bool) & (~dig_mask))
            img_color[border_mask] = color_dict["neutral"]

            # Initialize cumulative mask with dig zones and border
            cumulative_mask = np.zeros(img_color.shape[:2], dtype=np.bool_)
            cumulative_mask[dig_mask] = True
            cumulative_mask[border_mask] = True
            
            # Add 1-2 obstacles
            occ, cumulative_mask = add_obstacles(
                img_color,
                cumulative_mask,
                n_obs_min=1,
                n_obs_max=2,
                size_obstacle_min=4,
                size_obstacle_max=8,
            )

            # Dumpability map: all white (dumpable) for now
            custom_dmp = np.full_like(img_color, [255, 255, 255])  # all areas dumpable (white)
            
            # Use the generated occupancy map with obstacles
            custom_occupancy = occ[:,:,0].astype(np.uint8)  # extract first channel and convert to uint8
            
            # Save the result
            save_or_display_image(img_color, custom_occupancy, custom_dmp, metadata, curriculum_level, n)

    print("Border dumpzone foundations created successfully.")


def generate_foundation_border_dumpzone(config_path="config/env_generation/config.yml",
                                       generate_terra_format=True):
    """
    Generate foundation maps with a 2-tile neutral border and dumping elsewhere.

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

    print("Generating BORDER DUMPZONE foundation maps...")

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

    # Generate foundations with border dumpzones
    print("  → Generating BORDER DUMPZONE foundation maps...")
    create_foundations_border_dumpzone(
        n_imgs=n_imgs,
        max_size=foundations_config.get("max_size", 64)
    )
    print("  ✓ Border dumpzone foundation maps saved to: data/terra/foundations_border_dumpzone/")

    # === TERRA FORMAT CONVERSION ===
    if generate_terra_format:
        print("Converting border dumpzone data to Terra format...")
        sizes = [(64, 64)]  # Default size
        npy_dataset_folder = package_dir + "/data/terra"
        for size in sizes:
            # Convert generated maps using internal converter
            foundations_dir = Path(npy_dataset_folder) / "foundations_border_dumpzone"
            if not foundations_dir.exists():
                print(f"  Skipping conversion; folder not found: {foundations_dir}")
                continue
            destination_folder = Path(npy_dataset_folder) / "train" / "foundations_border_dumpzone"
            img_folder = foundations_dir / "images"
            metadata_folder = foundations_dir / "metadata"
            occupancy_folder = foundations_dir / "occupancy"
            dumpability_folder = foundations_dir / "dumpability"
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
                copy_metadata=True,
                downsample=False,
                has_dumpability=True,
                center_padding=False,
                actions_folder=None,
            )
        print("  ✓ Terra format conversion complete")

    print("Border dumpzone foundation generation complete!")
    print(f"Data saved to {os.path.join(package_dir, 'data/terra/foundations_border_dumpzone')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate foundation maps with a 2-tile neutral border and dumping elsewhere.")
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

    args = parser.parse_args()

    # Determine Terra format conversion
    generate_terra_format = not args.no_terra_format
    if args.no_terra_format:
        print("Terra format conversion disabled by --no-terra-format flag")
    else:
        print("Terra format conversion enabled (use --no-terra-format to disable)")

    generate_foundation_border_dumpzone(
        args.config,
        generate_terra_format=generate_terra_format,
    ) 