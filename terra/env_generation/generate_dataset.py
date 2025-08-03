#!/usr/bin/env python
import os
import yaml
import argparse
from terra.env_generation.generate_foundations import download_foundations, create_foundations
from terra.env_generation.create_train_data import (
    create_procedural_trenches, 
    create_foundations as create_train_foundations
)
from terra.env_generation.generate_foundations_with_dumpzones import create_foundations_with_dumpzones_defaults
from terra.env_generation.generate_foundations_with_dumpzones_harder import create_foundations_with_dumpzones_harder_defaults
from terra.env_generation.generate_relocations import create_relocations
from terra.env_generation.generate_relocations_easy import create_relocations_easy
from terra.env_generation.generate_relocations_medium import create_relocations_medium
from terra.env_generation.generate_relocations_hard import create_relocations_hard
from terra.env_generation.generate_relocations_harder import create_relocations_harder
import terra.env_generation.convert_to_terra as convert_to_terra
from terra.env_generation.convert_to_terra import generate_foundations_dumpzones_terra, generate_foundations_dumpzones_harder_terra

def generate_complete_dataset(config_path="config/env_generation/config.yml", 
                             generate_foundations=True,
                             generate_foundations_dumpzones=False,
                             generate_foundations_dumpzones_harder=False,
                             generate_trenches=True,
                             generate_relocations=True,
                             generate_relocations_easy=False,
                             generate_relocations_medium=False,
                             generate_relocations_hard=False,
                             generate_relocations_harder=False,
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
    if generate_foundations or generate_foundations_dumpzones or generate_foundations_dumpzones_harder:
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
            # Use default parameters - no config needed
            create_foundations_with_dumpzones_defaults()
            print("  ✓ Dump zone foundation maps saved to: data/terra/foundations_dumpzones/")

        # Generate foundations with specific dump zones (harder version)
        if generate_foundations_dumpzones_harder:
            print("  → Generating SPECIFIC DUMP ZONES foundation maps (HARDER VERSION)...")
            # Use default parameters - no config needed
            create_foundations_with_dumpzones_harder_defaults()
            print("  ✓ Harder dump zone foundation maps saved to: data/terra/foundations_dumpzones_harder/")

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

    # === MEDIUM RELOCATION MAPS ===
    if generate_relocations_medium:
        print(f"Step {step_counter}: Creating MEDIUM relocation maps...")
        step_counter += 1
        save_folder = os.path.join(package_dir, "data", "terra", "relocations_medium")
        create_relocations_medium(n_imgs, save_folder)
        print("  ✓ Medium relocation maps saved to: data/terra/relocations_medium/")

    # === HARD RELOCATION MAPS ===
    if generate_relocations_hard:
        print(f"Step {step_counter}: Creating HARD relocation maps...")
        step_counter += 1
        save_folder = os.path.join(package_dir, "data", "terra", "relocations_hard")
        create_relocations_hard(n_imgs, save_folder)
        print("  ✓ Hard relocation maps saved to: data/terra/relocations_hard/")

    if generate_relocations_harder:
        print(f"Step {step_counter}: Creating HARDER relocation maps...")
        step_counter += 1
        save_folder = os.path.join(package_dir, "data", "terra", "relocations_harder")
        create_relocations_harder(n_imgs, save_folder)
        print("  ✓ Harder relocation maps saved to: data/terra/relocations_harder/")

    # === TERRA FORMAT CONVERSION ===
    # Track which map types to convert
    map_types_to_convert = []
    if generate_foundations: map_types_to_convert.append("foundations")
    if generate_foundations_dumpzones: map_types_to_convert.append("foundations_dumpzones")
    if generate_foundations_dumpzones_harder: map_types_to_convert.append("foundations_dumpzones_harder")
    if generate_trenches: map_types_to_convert.append("trenches")
    if generate_relocations: map_types_to_convert.append("relocations")
    if generate_relocations_easy: map_types_to_convert.append("relocations_easy")
    if generate_relocations_medium: map_types_to_convert.append("relocations_medium")
    if generate_relocations_hard: map_types_to_convert.append("relocations_hard")
    if generate_relocations_harder: map_types_to_convert.append("relocations_harder")
    if generate_terra_format:
        print(f"Step {step_counter}: Converting data to Terra format...")
        step_counter += 1
        # Use default sizes instead of reading from config
        #sizes = [(50, 50), (100, 100)]  # Default sizes
        sizes = [(64, 64)]
        npy_dataset_folder = package_dir + "/data/terra"
        for size in sizes:
            # Use the dedicated function for foundations_dumpzones if it's in the map types
            if "foundations_dumpzones" in map_types_to_convert:
                print(f"  Converting foundations_dumpzones with size {size}...")
                convert_to_terra.generate_foundations_dumpzones_terra(npy_dataset_folder, size, n_imgs)
                # Remove foundations_dumpzones from the list to avoid double conversion
                map_types_to_convert = [mt for mt in map_types_to_convert if mt != "foundations_dumpzones"]
            
            # Use the dedicated function for foundations_dumpzones_harder if it's in the map types
            if "foundations_dumpzones_harder" in map_types_to_convert:
                print(f"  Converting foundations_dumpzones_harder with size {size}...")
                convert_to_terra.generate_foundations_dumpzones_harder_terra(npy_dataset_folder, size, n_imgs)
                # Remove foundations_dumpzones_harder from the list to avoid double conversion
                map_types_to_convert = [mt for mt in map_types_to_convert if mt != "foundations_dumpzones_harder"]
            
            # Convert remaining map types
            if map_types_to_convert:
                convert_to_terra.generate_dataset_terra_format(npy_dataset_folder, size, n_imgs, map_types_to_convert)
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
        "--foundations-dumpzones-harder", 
        action="store_true", 
        help="Generate foundation maps with specific dump zones (harder version) for mixed agent training"
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
        "--relocations-medium",
        action="store_true",
        help="Generate medium relocation maps (smaller dump zones, further dirt, and an obstacle)"
    )
    parser.add_argument(
        "--relocations-hard",
        action="store_true",
        help="Generate hard relocation maps (1-2 dirt, 1-2 dump zones)"
    )
    parser.add_argument(
        "--relocations-harder",
        action="store_true",
        help="Generate harder relocation maps (1-2 dirt, 1-2 dump zones, 1-2 obstacles)"
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
    if args.all or not any([args.foundations, args.foundations_dumpzones, args.foundations_dumpzones_harder, args.trenches, args.relocations, args.relocations_easy, args.relocations_medium, args.relocations_hard, args.relocations_harder, args.terra_format, args.no_terra_format]):
        # Generate everything (default behavior)
        generate_foundations = True
        generate_foundations_dumpzones = False  # Not generated by default
        generate_trenches = True
        generate_relocations = True
        generate_relocations_easy = False # Not generated by default
        generate_relocations_medium = False # Not generated by default
        generate_relocations_hard = False # Not generated by default
        generate_relocations_harder = False # Not generated by default
        generate_terra_format = True
        print("No specific options selected - generating all standard map types")
        print("(Use --foundations-dumpzones to also generate dump zone foundations)")
    else:
        # Generate only selected types
        generate_foundations = args.foundations
        generate_foundations_dumpzones = args.foundations_dumpzones
        generate_foundations_dumpzones_harder = args.foundations_dumpzones_harder
        generate_trenches = args.trenches
        generate_relocations = args.relocations
        generate_relocations_easy = args.relocations_easy
        generate_relocations_medium = args.relocations_medium
        generate_relocations_hard = args.relocations_hard
        generate_relocations_harder = args.relocations_harder
        # Handle Terra format conversion logic
        if args.no_terra_format:
            generate_terra_format = False
            print("Terra format conversion disabled by --no-terra-format flag")
        elif args.terra_format:
            generate_terra_format = True
        elif any([args.foundations, args.foundations_dumpzones, args.foundations_dumpzones_harder, args.trenches, args.relocations, args.relocations_easy, args.relocations_medium, args.relocations_hard, args.relocations_harder]):
            # If any map type is selected, default to True
            generate_terra_format = True
            print("Terra format conversion enabled by default (use --no-terra-format to disable)")
        else:
            generate_terra_format = False
        
        selected = []
        if generate_foundations: selected.append("standard foundations")
        if generate_foundations_dumpzones: selected.append("dump zone foundations")
        if generate_foundations_dumpzones_harder: selected.append("harder dump zone foundations")
        if generate_trenches: selected.append("trenches")
        if generate_relocations: selected.append("relocations")
        if generate_relocations_easy: selected.append("easy relocations")
        if generate_relocations_medium: selected.append("medium relocations")
        if generate_relocations_hard: selected.append("hard relocations")
        if generate_relocations_harder: selected.append("harder relocations")
        if generate_terra_format: selected.append("Terra format conversion")
        
        print(f"Generating selected map types: {', '.join(selected)}")

    generate_complete_dataset(
        args.config,
        generate_foundations=generate_foundations,
        generate_foundations_dumpzones=generate_foundations_dumpzones,
        generate_foundations_dumpzones_harder=generate_foundations_dumpzones_harder,
        generate_trenches=generate_trenches,
        generate_relocations=generate_relocations,
        generate_relocations_easy=generate_relocations_easy,
        generate_relocations_medium=generate_relocations_medium,
        generate_relocations_hard=generate_relocations_hard,
        generate_relocations_harder=generate_relocations_harder,
        generate_terra_format=generate_terra_format
    )