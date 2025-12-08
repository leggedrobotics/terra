#!/usr/bin/env python3
"""
Visualize a random map from the training dataset.
Uses the same loading functions as the training code (load_maps_from_disk).
Displays all map layers in a combined visualization.
"""

import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure the package root is on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PKG_PARENT not in sys.path:
    sys.path.insert(0, PKG_PARENT)

# Import validation functions from maps_buffer
# We'll import them inside the function to avoid circular dependencies


def clean_folder_name(folder_name: str) -> str:
    """Clean up folder name for display."""
    # Remove _v3, _v2, etc. suffixes
    if folder_name.endswith('_v3'):
        folder_name = folder_name[:-3]
    elif folder_name.endswith('_v2'):
        folder_name = folder_name[:-3]
    elif folder_name.endswith('_v1'):
        folder_name = folder_name[:-3]
    
    # Change relocations_harder to relocations
    if folder_name == 'relocations_harder':
        folder_name = 'relocations'
    
    return folder_name


def find_available_maps(train_dir: Path):
    """Find all available map indices in the train directory."""
    images_dir = train_dir / "images"
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Find all img_*.npy files
    map_files = sorted(images_dir.glob("img_*.npy"))
    if not map_files:
        raise ValueError(f"No map files found in {images_dir}")
    
    # Extract indices
    indices = []
    for f in map_files:
        try:
            idx = int(f.stem.split('_')[1])
            indices.append(idx)
        except (ValueError, IndexError):
            continue
    
    return sorted(indices)


def load_map_by_index_using_loader(train_dir: Path, map_idx: int):
    """Load a specific map using the same approach as load_maps_from_disk."""
    from terra.maps_buffer import map_sanity_check, occupancy_sanity_check, dumpability_sanity_check, actions_sanity_check
    from terra.settings import IntMap
    
    # Load files exactly like load_maps_from_disk does
    image_file = train_dir / "images" / f"img_{map_idx}.npy"
    occupancy_file = train_dir / "occupancy" / f"img_{map_idx}.npy"
    dumpability_file = train_dir / "dumpability" / f"img_{map_idx}.npy"
    distance_file = train_dir / "distance" / f"img_{map_idx}.npy"
    actions_file = train_dir / "actions" / f"img_{map_idx}.npy"
    
    if not image_file.exists():
        raise FileNotFoundError(f"Map {map_idx} not found: {image_file}")
    
    # Load and validate exactly like load_maps_from_disk
    image = np.load(str(image_file))
    map_sanity_check(image)
    
    occupancy = np.load(str(occupancy_file))
    occupancy_sanity_check(occupancy)
    
    dumpability_mask_init = np.load(str(dumpability_file))
    dumpability_sanity_check(dumpability_mask_init)
    
    # Load distance map
    if distance_file.exists():
        distance_map = np.load(str(distance_file))
        if distance_map.shape != image.shape:
            print(f"Warning: distance map shape mismatch, expected {image.shape}, got {distance_map.shape}; filling zeros.")
            distance_map = np.zeros_like(image, dtype=np.float32)
    else:
        print(f"Warning: missing distance map {distance_file}, filling zeros.")
        distance_map = np.zeros_like(image, dtype=np.float32)
    
    # Load actions map
    if actions_file.exists():
        actions_map = np.load(str(actions_file))
        actions_sanity_check(actions_map)
    else:
        actions_map = np.zeros_like(image, dtype=IntMap)
    
    return image, occupancy, dumpability_mask_init, distance_map.astype(np.float32), actions_map


def visualize_single_map(image, occupancy, dumpability, actions, ax, title: str = ""):
    """Visualize a single map on the given axes using original color scheme."""
    h, w = image.shape
    
    # Original color scheme from visualization settings
    COLOR_NEUTRAL = np.array([220, 220, 220]) / 255.0  # Light gray - natural background
    COLOR_DUMP_ZONE = np.array([204, 255, 204]) / 255.0  # Light green - dump zones (#ccffcc)
    COLOR_DIG_ZONE = np.array([136, 0, 255]) / 255.0  # Purple - dig zones (#8800ff)
    COLOR_OBSTACLE = np.array([0, 0, 0]) / 255.0  # Black - obstacles
    COLOR_NON_DUMPABLE = np.array([171, 159, 149]) / 255.0  # Dark grey - non-dumpable (#ab9f95)
    COLOR_ACTION_MASK = np.array([139, 69, 19]) / 255.0  # Brown - action mask (dirt)
    
    # Create RGB image
    rgb_image = np.zeros((h, w, 3))
    
    # Start with neutral background
    rgb_image[:, :] = COLOR_NEUTRAL
    
    # Apply dig zones (target_map < 0) - purple
    dig_mask = image < 0
    rgb_image[dig_mask] = COLOR_DIG_ZONE
    
    # Apply dump zones (target_map > 0) - light green
    dump_mask = image > 0
    rgb_image[dump_mask] = COLOR_DUMP_ZONE
    
    # Apply non-dumpable areas (dumpability == 0) - dark grey
    non_dumpable_mask = dumpability == 0
    rgb_image[non_dumpable_mask] = COLOR_NON_DUMPABLE
    
    # Apply action mask (actions > 0) - brown
    action_mask = actions > 0
    rgb_image[action_mask] = COLOR_ACTION_MASK
    
    # Apply obstacles (occupancy == 1) - black (overrides everything)
    obstacle_mask = occupancy == 1
    rgb_image[obstacle_mask] = COLOR_OBSTACLE
    
    # Check which layers are present
    has_dig_zones = np.any(image < 0)
    has_dump_zones = np.any(image > 0)
    has_non_dumpable = np.any(dumpability == 0)
    has_action_mask = np.any(actions > 0)
    has_obstacles = np.any(occupancy == 1)
    
    # Display the combined image
    ax.imshow(rgb_image, origin='upper', interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=8)
    ax.set_ylabel('Y (pixels)', fontsize=8)
    ax.grid(True, alpha=0.3, color='white', linewidth=0.3)
    ax.tick_params(labelsize=7)
    
    # Create legend only for layers present in this image
    legend_elements = []
    if has_dig_zones:
        legend_elements.append(mpatches.Patch(color=COLOR_DIG_ZONE, label='Dig Zone'))
    if has_dump_zones:
        legend_elements.append(mpatches.Patch(color=COLOR_DUMP_ZONE, label='Dump Zone'))
    if has_non_dumpable:
        legend_elements.append(mpatches.Patch(color=COLOR_NON_DUMPABLE, label='Non-dumpable'))
    if has_action_mask:
        legend_elements.append(mpatches.Patch(color=COLOR_ACTION_MASK, label='Action mask'))
    if has_obstacles:
        legend_elements.append(mpatches.Patch(color=COLOR_OBSTACLE, label='Obstacle'))
    # Always show neutral if no other layers (or show it if it's the only thing)
    if len(legend_elements) == 0 or (not has_dig_zones and not has_dump_zones and not has_obstacles):
        legend_elements.insert(0, mpatches.Patch(color=COLOR_NEUTRAL, label='Neutral'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)


def visualize_map_combined(image, occupancy, dumpability, distance, actions, output_path: str, map_idx: int):
    """Create a single combined visualization using original color scheme."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    h, w = image.shape
    
    # Original color scheme from visualization settings
    # Colors in RGB [0-255] format
    COLOR_NEUTRAL = np.array([220, 220, 220]) / 255.0  # Light gray - natural background
    COLOR_DUMP_ZONE = np.array([204, 255, 204]) / 255.0  # Light green - dump zones (#ccffcc)
    COLOR_DIG_ZONE = np.array([136, 0, 255]) / 255.0  # Purple - dig zones (#8800ff)
    COLOR_OBSTACLE = np.array([0, 0, 0]) / 255.0  # Black - obstacles
    COLOR_NON_DUMPABLE = np.array([171, 159, 149]) / 255.0  # Dark grey - non-dumpable (#ab9f95)
    COLOR_ACTION_MASK = np.array([139, 69, 19]) / 255.0  # Brown - action mask (dirt)
    
    # Create RGB image
    rgb_image = np.zeros((h, w, 3))
    
    # Start with neutral background
    rgb_image[:, :] = COLOR_NEUTRAL
    
    # Apply dig zones (target_map < 0) - purple
    dig_mask = image < 0
    rgb_image[dig_mask] = COLOR_DIG_ZONE
    
    # Apply dump zones (target_map > 0) - light green
    dump_mask = image > 0
    rgb_image[dump_mask] = COLOR_DUMP_ZONE
    
    # Apply non-dumpable areas (dumpability == 0) - dark grey
    non_dumpable_mask = dumpability == 0
    rgb_image[non_dumpable_mask] = COLOR_NON_DUMPABLE
    
    # Apply action mask (actions > 0) - brown
    action_mask = actions > 0
    rgb_image[action_mask] = COLOR_ACTION_MASK
    
    # Apply obstacles (occupancy == 1) - black (overrides everything)
    obstacle_mask = occupancy == 1
    rgb_image[obstacle_mask] = COLOR_OBSTACLE
    
    # Check which layers are present
    has_dig_zones = np.any(image < 0)
    has_dump_zones = np.any(image > 0)
    has_non_dumpable = np.any(dumpability == 0)
    has_action_mask = np.any(actions > 0)
    has_obstacles = np.any(occupancy == 1)
    
    # Display the combined image (no label to prevent auto-legend)
    ax.imshow(rgb_image, origin='upper', interpolation='nearest')
    ax.set_xlabel('X coordinate (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y coordinate (pixels)', fontsize=12, fontweight='bold')
    ax.set_title(f'Map Visualization\nShape: {image.shape}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Create legend only for layers present in this image
    # For single map visualization, we don't have folder_name, so default to "Non-dumpable"
    # (This function is typically used for single maps which are usually foundations)
    legend_elements = []
    if has_dig_zones:
        legend_elements.append(mpatches.Patch(color=COLOR_DIG_ZONE, label='Dig Zone'))
    if has_dump_zones:
        legend_elements.append(mpatches.Patch(color=COLOR_DUMP_ZONE, label='Dump Zone'))
    if has_non_dumpable:
        legend_elements.append(mpatches.Patch(color=COLOR_NON_DUMPABLE, label='Non-dumpable'))
    if has_action_mask:
        legend_elements.append(mpatches.Patch(color=COLOR_ACTION_MASK, label='Dirt Pile'))
    if has_obstacles:
        legend_elements.append(mpatches.Patch(color=COLOR_OBSTACLE, label='Obstacle'))
    # Always show neutral if no other layers (or show it if it's the only thing)
    if len(legend_elements) == 0 or (not has_dig_zones and not has_dump_zones and not has_obstacles):
        legend_elements.insert(0, mpatches.Patch(color=COLOR_NEUTRAL, label='Neutral'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    dig_zones = np.sum(image < 0)
    dump_zones = np.sum(image > 0)
    neutral_zones = np.sum(image == 0)
    print(f"Map shape: {image.shape}")
    print(f"Dig zones: {dig_zones} tiles (purple)")
    print(f"Dump zones: {dump_zones} tiles (light green)")
    print(f"Neutral: {neutral_zones} tiles (light gray)")
    print(f"Obstacles: {occupancy.sum()} tiles (black)")
    print(f"Dumpable tiles: {dumpability.sum()} out of {dumpability.size}")


def visualize_multiple_maps_grid(map_data_list, output_path: str):
    """Create a grid visualization of multiple maps.
    
    Args:
        map_data_list: List of tuples (image, occupancy, dumpability, folder_name, map_idx)
        output_path: Path to save the visualization
    """
    n_maps = len(map_data_list)
    if n_maps == 0:
        raise ValueError("No maps to visualize")
    
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(n_maps)))
    rows = int(np.ceil(n_maps / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_maps == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Original color scheme
    COLOR_NEUTRAL = np.array([220, 220, 220]) / 255.0
    COLOR_DUMP_ZONE = np.array([204, 255, 204]) / 255.0
    COLOR_DIG_ZONE = np.array([136, 0, 255]) / 255.0
    COLOR_OBSTACLE = np.array([0, 0, 0]) / 255.0
    COLOR_NON_DUMPABLE = np.array([171, 159, 149]) / 255.0  # Dark grey - non-dumpable
    COLOR_ACTION_MASK = np.array([139, 69, 19]) / 255.0  # Brown - action mask
    
    # Visualize each map
    for idx, (image, occupancy, dumpability, actions, folder_name, map_idx) in enumerate(map_data_list):
        ax = axes[idx]
        h, w = image.shape
        
        # Create RGB image
        rgb_image = np.zeros((h, w, 3))
        rgb_image[:, :] = COLOR_NEUTRAL
        
        # Apply zones and track which layers are present
        has_dig_zones = False
        has_dump_zones = False
        has_non_dumpable = False
        has_action_mask = False
        has_obstacles = False
        
        # Apply dig zones (target_map < 0) - purple
        dig_mask = image < 0
        if np.any(dig_mask):
            has_dig_zones = True
            rgb_image[dig_mask] = COLOR_DIG_ZONE
        
        # Apply dump zones (target_map > 0) - light green
        dump_mask = image > 0
        if np.any(dump_mask):
            has_dump_zones = True
            rgb_image[dump_mask] = COLOR_DUMP_ZONE
        
        # Apply non-dumpable areas (dumpability == 0) - dark grey
        non_dumpable_mask = dumpability == 0
        if np.any(non_dumpable_mask):
            has_non_dumpable = True
            rgb_image[non_dumpable_mask] = COLOR_NON_DUMPABLE
        
        # Apply action mask (actions > 0) - brown
        action_mask = actions > 0
        if np.any(action_mask):
            has_action_mask = True
            rgb_image[action_mask] = COLOR_ACTION_MASK
        
        # Apply obstacles (occupancy == 1) - black (overrides everything)
        obstacle_mask = occupancy == 1
        if np.any(obstacle_mask):
            has_obstacles = True
            rgb_image[obstacle_mask] = COLOR_OBSTACLE
        
        # Display
        ax.imshow(rgb_image, origin='upper', interpolation='nearest')
        display_name = clean_folder_name(folder_name)
        title = f"{display_name}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=8)
        ax.set_ylabel('Y (pixels)', fontsize=8)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.3)
        ax.tick_params(labelsize=7)
        
        # Create legend only for layers present in this image
        # Determine label for non-dumpable based on map type
        folder_lower = folder_name.lower()
        is_foundations_map = 'foundation' in folder_lower
        is_roads_map = 'road' in folder_lower
        # Use "Road" only for roads maps, "Non-dumpable" for foundations maps
        if is_roads_map:
            non_dumpable_label = 'Road'
        elif is_foundations_map:
            non_dumpable_label = 'Non-dumpable'
        else:
            # Default to "Non-dumpable" for other map types
            non_dumpable_label = 'Non-dumpable'
        
        legend_elements = []
        if has_dig_zones:
            legend_elements.append(mpatches.Patch(color=COLOR_DIG_ZONE, label='Dig Zone'))
        if has_dump_zones:
            legend_elements.append(mpatches.Patch(color=COLOR_DUMP_ZONE, label='Dump Zone'))
        if has_non_dumpable:
            legend_elements.append(mpatches.Patch(color=COLOR_NON_DUMPABLE, label=non_dumpable_label))
        if has_action_mask:
            legend_elements.append(mpatches.Patch(color=COLOR_ACTION_MASK, label='Dirt Pile'))
        if has_obstacles:
            legend_elements.append(mpatches.Patch(color=COLOR_OBSTACLE, label='Obstacle'))
        # Always show neutral if no other layers (or show it if it's the only thing)
        if len(legend_elements) == 0 or (not has_dig_zones and not has_dump_zones and not has_obstacles):
            legend_elements.insert(0, mpatches.Patch(color=COLOR_NEUTRAL, label='Neutral'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
    
    # Hide unused subplots
    for idx in range(n_maps, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid visualization with {n_maps} maps to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize a random map from the training dataset using proper loading functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default DATASET_PATH and random map
  python visualize_random_map.py
  
  # Specify train directory and map index
  python visualize_random_map.py --train-dir /path/to/train --map-idx 5
  
  # Use specific dataset path
  DATASET_PATH=/path/to/data python visualize_random_map.py --maps-path foundations_dumpzones_v3
        """
    )
    parser.add_argument('--train-dir', type=str, default=None,
                       help='Path to train directory (overrides DATASET_PATH/maps-path)')
    parser.add_argument('--maps-path', type=str, default=None, nargs='+',
                       help='Subdirectory name(s) under DATASET_PATH (e.g., "foundations_dumpzones_v3"). Can specify multiple for grid view.')
    parser.add_argument('--map-idx', type=int, default=None,
                       help='Specific map index to visualize (default: random). Applies to all folders if multiple specified.')
    parser.add_argument('--output', type=str, default='random_map_visualization.png',
                       help='Output file path for visualization')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Determine train directories
    train_dirs = []
    dataset_path = os.getenv("DATASET_PATH", "")
    
    if args.train_dir:
        # Single directory specified
        train_dirs = [Path(args.train_dir)]
    elif args.maps_path:
        # Multiple or single maps_path specified
        if not dataset_path:
            raise ValueError("DATASET_PATH environment variable not set. Use --train-dir or set DATASET_PATH.")
        train_dirs = [Path(dataset_path) / path for path in args.maps_path]
    else:
        # Try to use DATASET_PATH with default maps_path
        if not dataset_path:
            raise ValueError(
                "No train directory specified. Use --train-dir, --maps-path, or set DATASET_PATH environment variable."
            )
        # Try common map paths
        common_paths = ["foundations_dumpzones_v3", "trenches/easy", "relocations"]
        for path in common_paths:
            candidate = Path(dataset_path) / path
            if candidate.exists() and (candidate / "images").exists():
                train_dirs = [candidate]
                break
        
        if not train_dirs:
            raise ValueError(
                f"Could not find train directory. Checked DATASET_PATH={dataset_path} with common paths. "
                "Use --train-dir or --maps-path to specify."
            )
    
    # Validate all directories
    for train_dir in train_dirs:
        if not train_dir.exists():
            raise ValueError(f"Train directory does not exist: {train_dir}")
    
    # Load maps from each directory
    map_data_list = []
    for train_dir in train_dirs:
        folder_name = train_dir.name if train_dir.name else train_dir.parent.name
        
        print(f"Loading maps from: {train_dir}")
        available_indices = find_available_maps(train_dir)
        print(f"Found {len(available_indices)} maps (indices: {available_indices[0]} to {available_indices[-1]})")
        
        # Select map index
        if args.map_idx is not None:
            if args.map_idx not in available_indices:
                print(f"Warning: Map index {args.map_idx} not found in {train_dir}. Skipping.")
                continue
            map_idx = args.map_idx
        else:
            map_idx = random.choice(available_indices)
            print(f"Randomly selected map index: {map_idx}")
        
        # Load map
        print(f"Loading map {map_idx} from {folder_name}...")
        try:
            image, occupancy, dumpability, distance, actions = load_map_by_index_using_loader(train_dir, map_idx)
            map_data_list.append((image, occupancy, dumpability, actions, folder_name, map_idx))
        except Exception as e:
            print(f"Error loading map from {train_dir}: {e}")
            continue
    
    if not map_data_list:
        raise ValueError("No maps were successfully loaded.")
    
    # Visualize
    output_path = os.path.abspath(args.output)
    if len(map_data_list) == 1:
        # Single map: use original single visualization
        print(f"Creating single visualization...")
        image, occupancy, dumpability, folder_name, map_idx = map_data_list[0]
        # Load full data for single visualization
        train_dir = train_dirs[0]
        _, _, _, distance, actions = load_map_by_index_using_loader(train_dir, map_idx)
        visualize_map_combined(image, occupancy, dumpability, distance, actions, output_path, map_idx)
    else:
        # Multiple maps: use grid visualization
        print(f"Creating grid visualization with {len(map_data_list)} maps...")
        visualize_multiple_maps_grid(map_data_list, output_path)
    
    print(f"Saved visualization to: {output_path}")


if __name__ == '__main__':
    main()
