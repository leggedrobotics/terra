#!/usr/bin/env python
import os
import yaml
import argparse
from terra.env_generation.generate_foundations import download_foundations, create_foundations
from terra.env_generation.create_train_data import (
    create_procedural_trenches, 
    create_foundations as create_train_foundations
)
from terra.env_generation.generate_relocations import create_relocations
import terra.env_generation.convert_to_terra as convert_to_terra

def generate_complete_dataset(config_path="config/env_generation/config.yml"):
    """
    Generate a complete dataset in one go - combining foundations generation and training data creation.
    
    Args:
        config_path: Path to the configuration file
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

    print("Step 1: Downloading and processing foundation maps...")
    # Download foundations
    # Read foundation parameters from the config file
    foundations_config = config.get("foundations", {})
    # Backup to using sizes list if not provided in foundations config
    if "min_size" in foundations_config and "max_size" in foundations_config:
        foundation_min_size = foundations_config.get("min_size")
        foundation_max_size = foundations_config.get("max_size")
    else:
        raise ValueError("min_size and max_size must be provided in the config file")
    max_buildings = foundations_config.get("max_buildings", 100)
    print(f"max_buildings: {max_buildings}")

    print(f"Foundation min_size: {foundation_min_size}, max_size: {foundation_max_size}, max_buildings: {max_buildings}")

    # Get bounding box from config, or use default
    bbox = config.get("center_bbox", (47.5376, 47.6126, 7.5401, 7.6842))

    # Download foundations
    dataset_folder = os.path.join(package_dir, "data", "openstreet")
    download_foundations(
        dataset_folder,
        min_size=(foundation_min_size, foundation_min_size),
        max_size=(foundation_max_size, foundation_max_size),
        center_bbox=bbox,
        max_buildings=max_buildings
    )
    create_foundations(dataset_folder)

    print("Step 2: Creating procedural trenches and processing training data...")
    # Create procedural trenches
    main_folder = "data/terra"
    create_procedural_trenches(main_folder, config)

    # Process foundation maps for training
    create_train_foundations(config)

    # Create relocations maps
    relocations_config = config.get("relocations", {})
    create_relocations(relocations_config, n_imgs)

    # Generate Terra format datasets
    print("Step 3: Converting data to Terra format...")
    sizes = [(size, size) for size in config["sizes"]]
    npy_dataset_folder = package_dir + "/data/terra"
    for size in sizes:
        convert_to_terra.generate_dataset_terra_format(npy_dataset_folder, size, n_imgs)

    print("Dataset generation complete!")
    print(f"Data saved to {os.path.join(package_dir, 'data/terra')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate complete Terra training dataset.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/env_generation_config.yaml",
        help="Path to the configuration file"
    )
    args = parser.parse_args()

    generate_complete_dataset(args.config)