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
from terra.env_generation.generate_relocations_easy import create_relocations_easy
import terra.env_generation.convert_to_terra as convert_to_terra

def generate_complete_dataset(config_path="config/env_generation/config.yml", 
                             generate_foundations=True,
                             generate_foundations_dumpzones=False,
                             generate_trenches=True,
                             generate_relocations=True,
                             generate_relocations_easy=False,
                             generate_terra_format=True):
    """
    Generate a complete dataset in one go - combining foundations generation and training data creation.
    
    Args:
        config_path: Path to the configuration file
        generate_foundations: Whether to generate standard foundation maps
        generate_foundations_dumpzones: Whether to generate foundation maps with specific dump zones
        generate_trenches: Whether to generate trench maps
        generate_relocations: Whether to generate relocation maps
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
    step_counter = 1

    # === FOUNDATION MAPS ===
    if generate_foundations or generate_foundations_dumpzones:
        print(f"Step {step_counter}: Downloading and processing foundation maps...")
        step_counter += 1
        
        # Read foundation parameters from the config file
        foundations_config = config.get("foundations", {})
        if "min_size" in foundations_config and "max_size" in foundations_config:
            foundation_min_size = foundations_config.get("min_size")
            foundation_max_size = foundations_config.get("max_size")
        else:
            raise ValueError("min_size and max_size must be provided in the config file")
        max_buildings = foundations_config.get("max_buildings", 100)
        
        print(f"Foundation config - min_size: {foundation_min_size}, max_size: {foundation_max_size}, max_buildings: {max_buildings}")

        # Get bounding box from config, or use default
        bbox = config.get("center_bbox", (47.5376, 47.6126, 7.5401, 7.6842))

        # Download and create base foundation data (needed for both types)
        dataset_folder = os.path.join(package_dir, "data", "openstreet")
        download_foundations(
            dataset_folder,
            min_size=(foundation_min_size, foundation_min_size),
            max_size=(foundation_max_size, foundation_max_size),
            center_bbox=bbox,
            max_buildings=max_buildings
        )
        create_foundations(dataset_folder)

        # Generate standard foundations (everything dumpable)
        if generate_foundations:
            print("  → Generating STANDARD foundation maps (everything dumpable)...")
            config_copy = config.copy()
            config_copy["foundations"]["use_specific_dump_zones"] = False
            create_train_foundations(config_copy)
            print("  ✓ Standard foundation maps saved to: data/terra/foundations/")

        # Generate foundations with specific dump zones
        if generate_foundations_dumpzones:
            print("  → Generating SPECIFIC DUMP ZONES foundation maps...")
            config_copy = config.copy()
            config_copy["foundations"]["use_specific_dump_zones"] = True
            create_train_foundations(config_copy)
            print("  ✓ Dump zone foundation maps saved to: data/terra/foundations_dumpzones/")

    # === TRENCH MAPS ===
    if generate_trenches:
        print(f"Step {step_counter}: Creating procedural trenches...")
        step_counter += 1
        create_procedural_trenches(config)
        print("  ✓ Trench maps saved to: data/terra/trenches/")

    # === RELOCATION MAPS ===
    if generate_relocations:
        print(f"Step {step_counter}: Creating relocation maps...")
        step_counter += 1
        relocations_config = config.get("relocations", {})
        create_relocations(relocations_config, n_imgs)
        print("  ✓ Relocation maps saved to: data/terra/relocations/")

    # === EASY RELOCATION MAPS ===
    if generate_relocations_easy:
        print(f"Step {step_counter}: Creating EASY relocation maps...")
        step_counter += 1
        save_folder = os.path.join(package_dir, "data", "terra", "relocations_easy")
        create_relocations_easy(n_imgs, save_folder)
        print("  ✓ Easy relocation maps saved to: data/terra/relocations_easy/")

    # === TERRA FORMAT CONVERSION ===
    if generate_terra_format:
        print(f"Step {step_counter}: Converting data to Terra format...")
        step_counter += 1
        sizes = [(size, size) for size in config["sizes"]]
        npy_dataset_folder = package_dir + "/data/terra"
        for size in sizes:
            convert_to_terra.generate_dataset_terra_format(npy_dataset_folder, size, n_imgs)
        print("  ✓ Terra format conversion complete")

    print("Dataset generation complete!")
    print(f"Data saved to {os.path.join(package_dir, 'data/terra')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Terra training dataset with selective options.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/env_generation_config.yaml",
        help="Path to the configuration file"
    )
    
    # Map type selection arguments
    parser.add_argument(
        "--foundations", 
        action="store_true", 
        help="Generate standard foundation maps (everything dumpable)"
    )
    parser.add_argument(
        "--foundations-dumpzones", 
        action="store_true", 
        help="Generate foundation maps with specific dump zones for mixed agent training"
    )
    parser.add_argument(
        "--trenches", 
        action="store_true", 
        help="Generate trench maps"
    )
    parser.add_argument(
        "--relocations", 
        action="store_true", 
        help="Generate relocation maps"
    )
    parser.add_argument(
        "--relocations-easy",
        action="store_true",
        help="Generate easy relocation maps (dirt and dump zones close, no obstacles)"
    )
    parser.add_argument(
        "--terra-format", 
        action="store_true", 
        help="Convert generated maps to Terra format (enabled by default)"
    )
    parser.add_argument(
        "--no-terra-format", 
        action="store_true", 
        help="Skip Terra format conversion"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Generate all map types (default behavior if no specific types are selected)"
    )
    
    args = parser.parse_args()
    
    # Determine what to generate
    if args.all or not any([args.foundations, args.foundations_dumpzones, args.trenches, args.relocations, args.relocations_easy, args.terra_format, args.no_terra_format]):
        # Generate everything (default behavior)
        generate_foundations = True
        generate_foundations_dumpzones = False  # Not generated by default
        generate_trenches = True
        generate_relocations = True
        generate_relocations_easy = False # Not generated by default
        generate_terra_format = True
        print("No specific options selected - generating all standard map types")
        print("(Use --foundations-dumpzones to also generate dump zone foundations)")
    else:
        # Generate only selected types
        generate_foundations = args.foundations
        generate_foundations_dumpzones = args.foundations_dumpzones
        generate_trenches = args.trenches
        generate_relocations = args.relocations
        generate_relocations_easy = args.relocations_easy
        # Handle Terra format conversion logic
        if args.no_terra_format:
            generate_terra_format = False
            print("Terra format conversion disabled by --no-terra-format flag")
        elif args.terra_format:
            generate_terra_format = True
        elif any([args.foundations, args.foundations_dumpzones, args.trenches, args.relocations, args.relocations_easy]):
            # If any map type is selected, default to True
            generate_terra_format = True
            print("Terra format conversion enabled by default (use --no-terra-format to disable)")
        else:
            generate_terra_format = False
        
        selected = []
        if generate_foundations: selected.append("standard foundations")
        if generate_foundations_dumpzones: selected.append("dump zone foundations")
        if generate_trenches: selected.append("trenches")
        if generate_relocations: selected.append("relocations")
        if generate_relocations_easy: selected.append("easy relocations")
        if generate_terra_format: selected.append("Terra format conversion")
        
        print(f"Generating selected map types: {', '.join(selected)}")

    generate_complete_dataset(
        args.config,
        generate_foundations=generate_foundations,
        generate_foundations_dumpzones=generate_foundations_dumpzones,
        generate_trenches=generate_trenches,
        generate_relocations=generate_relocations,
        generate_relocations_easy=generate_relocations_easy,
        generate_terra_format=generate_terra_format
    )