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
from skimage.morphology import binary_dilation, disk

# Define package directory at module level
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

name_string = "separated_v2"

def create_spatially_separated_trenches(img_terra_pad, n_trenches, sizes_small, sizes_long, separation_px, occupied_mask):
    """
    Create multiple trench dig zones that are spatially separated by at least separation_px.
    Uses rectangular trenches with the same size logic as the original trench generation.

    Args:
        img_terra_pad: RGB image holding current terra colors
        n_trenches: number of trench dig zones to place
        sizes_small: tuple (min_small, max_small) for small dimension range
        sizes_long: tuple (min_long, max_long) for long dimension range
        separation_px: minimum center-to-center distance between trenches
        occupied_mask: boolean mask of pixels that cannot be used (already occupied)

    Returns:
        img_terra_pad, occupied_mask updated with trench dig zones, trench_axes for metadata
    """
    h, w = img_terra_pad.shape[:2]
    centers = []
    trench_axes = []  # Store line equations for metadata
    attempts = 0
    max_attempts = 500

    while len(centers) < n_trenches and attempts < max_attempts:
        attempts += 1
        # Random trench dimensions using same logic as generate_edges
        min_ssmall, max_ssmall = sizes_small
        min_slong, max_slong = sizes_long
        
        current_size_small = random.randint(min_ssmall, max_ssmall)
        current_size_long = random.randint(min_slong, max_slong)
        
        # Randomly choose orientation (horizontal or vertical)
        is_horizontal = random.choice([True, False])
        
        if is_horizontal:
            width, height = current_size_long, current_size_small
        else:
            width, height = current_size_small, current_size_long
        
        # Random position with border padding
        x = random.randint(3, w - width - 3)
        y = random.randint(3, h - height - 3)

        # Check occupancy overlap
        if np.any(occupied_mask[y:y+height, x:x+width]):
            continue

        # Enforce separation from already placed trenches
        cx, cy = x + width // 2, y + height // 2
        ok = True
        for (px, py, pwidth, pheight) in centers:
            if (cx - px) ** 2 + (cy - py) ** 2 < separation_px ** 2:
                ok = False
                break
        if not ok:
            continue

        # Place trench
        img_terra_pad[y:y+height, x:x+width] = color_dict["digging"]
        occupied_mask[y:y+height, x:x+width] = True
        centers.append((cx, cy, width, height))
        
        # Generate line equation for this trench (simplified: use center line)
        if is_horizontal:
            # Horizontal trench: line through center y-coordinate
            A, B, C = 0.0, 1.0, float(cy)
        else:
            # Vertical trench: line through center x-coordinate  
            A, B, C = 1.0, 0.0, float(cx)
        
        trench_axes.append({"A": A, "B": B, "C": C})

    if len(centers) < n_trenches:
        print(f"Warning: requested {n_trenches} separated trenches, placed {len(centers)}")
    return img_terra_pad, occupied_mask, trench_axes


def create_single_dump_zone(img_terra_pad, size_dump_min, size_dump_max, foundation_mask, min_distance_from_dig: int = 5):
    """
    Create exactly 1 dump zone on the border that avoids foundation overlaps.
    """
    height, width = img_terra_pad.shape[:2]
    dump_cumulative_mask = np.zeros((height, width), dtype=np.bool_)
    border_offset = 3
    max_attempts = 100
    # Build a safety buffer around dig zones to keep dump zones not directly next to them
    dig_mask_global = np.all(img_terra_pad == color_dict["digging"], axis=-1)
    buffered_dig = binary_dilation(dig_mask_global, disk(max(1, min_distance_from_dig)))
    forbidden_mask = np.logical_or(foundation_mask, buffered_dig)

    for attempt in range(max_attempts):
        dump_size = random.randint(size_dump_min, size_dump_max)
        border_choice = random.randint(0, 3)
        if border_choice == 0:
            x = random.randint(border_offset, width - dump_size - border_offset)
            y = border_offset
        elif border_choice == 1:
            x = random.randint(border_offset, width - dump_size - border_offset)
            y = height - dump_size - border_offset
        elif border_choice == 2:
            x = border_offset
            y = random.randint(border_offset, height - dump_size - border_offset)
        else:
            x = width - dump_size - border_offset
            y = random.randint(border_offset, height - dump_size - border_offset)

        dump_area = forbidden_mask[y:y+dump_size, x:x+dump_size]
        dig_area = False  # covered by forbidden_mask already
        if not np.any(dump_area) and not np.any(dig_area):
            img_terra_pad[y:y+dump_size, x:x+dump_size] = color_dict["dumping"]
            dump_cumulative_mask[y:y+dump_size, x:x+dump_size] = True
            print(f"Successfully placed dump zone of size {dump_size} at position ({x}, {y})")
            break
    else:
        print("Warning: Could not place dump zone on first try; falling back to smaller size")
        fallback_size_min = max(5, size_dump_min - 3)
        fallback_size_max = max(8, size_dump_max - 3)
        placed = False
        for attempt in range(100):
            dump_size = random.randint(fallback_size_min, fallback_size_max)
            border_choice = random.randint(0, 3)
            if border_choice == 0:
                x = random.randint(border_offset, width - dump_size - border_offset)
                y = border_offset
            elif border_choice == 1:
                x = random.randint(border_offset, width - dump_size - border_offset)
                y = height - dump_size - border_offset
            elif border_choice == 2:
                x = border_offset
                y = random.randint(border_offset, height - dump_size - border_offset)
            else:
                x = width - dump_size - border_offset
                y = random.randint(border_offset, height - dump_size - border_offset)
            dump_area = forbidden_mask[y:y+dump_size, x:x+dump_size]
            dig_area = False
            if not np.any(dump_area) and not np.any(dig_area):
                img_terra_pad[y:y+dump_size, x:x+dump_size] = color_dict["dumping"]
                dump_cumulative_mask[y:y+dump_size, x:x+dump_size] = True
                print(f"Fallback successful: placed dump zone of size {dump_size} at position ({x}, {y})")
                placed = True
                break
        if not placed:
            print("Warning: Could not place dump zone even after fallback attempts")
    return img_terra_pad, dump_cumulative_mask


def add_dirt_tiles_hybrid(img, occ, dmp, cumulative_mask, total_dirt_tiles):
    """
    Add dirt tiles exactly like relocations_harder.py does.
    Kept identical to v3 for consistency; can be adapted if needed.
    """
    w, h = img.shape[:2]
    action_map = np.ones_like(img) * 255
    mask_occ = _get_img_mask(occ, color_dict["obstacle"]) 
    mask_dmp = _get_img_mask(dmp, color_dict["nondumpable"]) 

    n_spots = 0
    remaining_dirt = total_dirt_tiles
    dirt_spots = []
    print(f"Placing {total_dirt_tiles} dirt tiles across {n_spots} spots...")
    for spot in range(n_spots):
        if remaining_dirt <= 0:
            break
        if spot == n_spots - 1:
            spot_size = remaining_dirt
        else:
            min_spot_size = max(8, remaining_dirt // 4)
            max_spot_size = min(remaining_dirt - (n_spots - spot - 1) * min_spot_size, remaining_dirt // 2)
            spot_size = np.random.randint(min_spot_size, max_spot_size + 1)
        patch_size = int(np.sqrt(spot_size)) + 1
        placed = False
        for _ in range(30):
            x = np.random.randint(5, w - patch_size - 5)
            y = np.random.randint(5, h - patch_size - 5)
            if (
                np.all(cumulative_mask[x : x + patch_size, y : y + patch_size] == 0)
                and np.all(mask_occ[x : x + patch_size, y : y + patch_size] == 0)
                and np.all(mask_dmp[x : x + patch_size, y : y + patch_size] == 0)
            ):
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


def create_trenches_separated_standalone(
                                    n_imgs=600,
                                    max_size=64,
                                    n_obs_min=0,
                                    n_obs_max=1,
                                    size_obstacle_min=3,
                                    size_obstacle_max=5,
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
                                    size_dump_min=15,
                                    size_dump_max=15,
                                    min_dirt_tiles=10,
                                    max_dirt_tiles=20,
                                    n_separated_trenches=2,
                                    trench_sizes_small=(2, 3),  # (min_small, max_small) - matches original script
                                    trench_sizes_long=(12, 19), # (min_long, max_long) - matches original script
                                    trench_min_separation_px=23,
                                    no_dump_zones=False):
    """
    Creates separated trench environments with pre-placed dirt (like relocations_harder) 
    AND dig zones (trenches) using procedural generation.

    This creates maps where:
    - Excavators can dig new dirt from trenches (dig zones)
    - Skidsteers can immediately start moving pre-placed dirt
    - Both agents have work from day 1!
    - Two spatially separated trenches per map
    """
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "trenches", name_string)
    print(f"Creating separated trenches with pre-placed dirt - saving to: {name_string}/")

    if os.path.exists(save_folder):
        print(f"Cleaning existing output folder: {save_folder}")
        shutil.rmtree(save_folder)

    # Create base neutral map
    img_terra_pad = np.full((max_size, max_size, 3), color_dict["neutral"], dtype=np.uint8)
    
    # Create empty masks
    foundation_mask = np.zeros((max_size, max_size), dtype=np.bool_)
    dump_cumulative_mask = np.zeros((max_size, max_size), dtype=np.bool_)
    
    # Place one dump zone on border
    img_terra_pad, dump_cumulative_mask = create_single_dump_zone(
        img_terra_pad, size_dump_min, size_dump_max, foundation_mask, min_distance_from_dig=6
    )

    # Create two spatially separated trenches
    occupied_mask = foundation_mask | dump_cumulative_mask
    img_terra_pad, occupied_mask, trench_axes = create_spatially_separated_trenches(
        img_terra_pad,
        n_separated_trenches,
        trench_sizes_small,
        trench_sizes_long,
        trench_min_separation_px,
        occupied_mask,
    )

    # Initialize cumulative mask with dump zones, dig zones, and foundation buildings
    cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
    cumulative_mask[np.all(img_terra_pad == color_dict["digging"], axis=-1)] = True
    cumulative_mask = dump_cumulative_mask | cumulative_mask
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

    # Pre-placed dirt tiles like relocations_harder
    total_dirt_tiles = np.random.randint(min_dirt_tiles, max_dirt_tiles + 1)
    action_map, cumulative_mask = add_dirt_tiles_hybrid(
        img_terra_pad, occ, dmp, cumulative_mask, total_dirt_tiles
    )

    # Optional: make all neutral zones dumpable (match v3 behavior)
    if no_dump_zones:
        neutral_mask = np.all(img_terra_pad == color_dict["neutral"], axis=-1)
        img_terra_pad[neutral_mask] = color_dict["dumping"]
        dump_cumulative_mask = neutral_mask

    # Generate multiple maps
    for i in range(n_imgs):
        print(f"Processing separated trench nr {i + 1}")
        
        # Create fresh map for each iteration
        if i > 0:
            img_terra_pad = np.full((max_size, max_size, 3), color_dict["neutral"], dtype=np.uint8)
            foundation_mask = np.zeros((max_size, max_size), dtype=np.bool_)
            dump_cumulative_mask = np.zeros((max_size, max_size), dtype=np.bool_)
            
            # Place one dump zone on border
            img_terra_pad, dump_cumulative_mask = create_single_dump_zone(
                img_terra_pad, size_dump_min, size_dump_max, foundation_mask, min_distance_from_dig=6
            )

            # Create two spatially separated trenches
            occupied_mask = foundation_mask | dump_cumulative_mask
            img_terra_pad, occupied_mask, trench_axes = create_spatially_separated_trenches(
                img_terra_pad,
                n_separated_trenches,
                trench_sizes_small,
                trench_sizes_long,
                trench_min_separation_px,
                occupied_mask,
            )

            # Initialize cumulative mask with dump zones, dig zones, and foundation buildings
            cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
            cumulative_mask[np.all(img_terra_pad == color_dict["digging"], axis=-1)] = True
            cumulative_mask = dump_cumulative_mask | cumulative_mask
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

            # Pre-placed dirt tiles like relocations_harder
            total_dirt_tiles = np.random.randint(min_dirt_tiles, max_dirt_tiles + 1)
            action_map, cumulative_mask = add_dirt_tiles_hybrid(
                img_terra_pad, occ, dmp, cumulative_mask, total_dirt_tiles
            )

            # Optional: make all neutral zones dumpable (match v3 behavior)
            if no_dump_zones:
                neutral_mask = np.all(img_terra_pad == color_dict["neutral"], axis=-1)
                img_terra_pad[neutral_mask] = color_dict["dumping"]
                dump_cumulative_mask = neutral_mask

        # Save outputs
        Path(save_folder, "images").mkdir(parents=True, exist_ok=True)
        Path(save_folder, "metadata").mkdir(parents=True, exist_ok=True)
        Path(save_folder, "occupancy").mkdir(parents=True, exist_ok=True)
        Path(save_folder, "dumpability").mkdir(parents=True, exist_ok=True)
        Path(save_folder, "actions").mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(Path(save_folder, "images", f"trench_{i}.png")), img_terra_pad)
        cv2.imwrite(str(Path(save_folder, "occupancy", f"trench_{i}.png")), occ)
        cv2.imwrite(str(Path(save_folder, "dumpability", f"trench_{i}.png")), dmp)
        cv2.imwrite(str(Path(save_folder, "actions", f"trench_{i}.png")), action_map)

        # Create metadata
        metadata = {
            "curriculum_level": save_folder,
            "map_type": "separated_trenches",
            "n_trenches": n_separated_trenches,
            "trench_separation": trench_min_separation_px,
            "total_dirt_tiles": total_dirt_tiles,
            "axes_ABC": trench_axes,  # Add trench line equations for reward system
        }
        with open(Path(save_folder, "metadata", f"trench_{i}.json"), "w") as json_file:
            json.dump(metadata, json_file)

    print("Separated trenches with pre-placed dirt created successfully.")


def generate_trenches_separated_standalone(config_path="config/env_generation_config.yaml",
                                          generate_terra_format=True,
                                          no_dump_zones=False):
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(package_dir + "/" + config_path, "r") as file:
        config = yaml.safe_load(file)

    os.makedirs("data/", exist_ok=True)
    os.makedirs("data/terra/", exist_ok=True)

    n_imgs = config["n_imgs"]
    print("Generating TRENCHES (SEPARATED) maps...")

    print("  → Generating SEPARATED TRENCHES maps...")
    create_trenches_separated_standalone(
        n_imgs=n_imgs,
        max_size=config.get("max_size", 64),
        no_dump_zones=no_dump_zones,
    )
    print(f"  ✓ Separated trenches maps saved to: data/terra/trenches/{name_string}")

    if generate_terra_format:
        print("Converting separated trenches data to Terra format...")
        sizes = [(64, 64)]
        npy_dataset_folder = package_dir + "/data/terra"
        for size in sizes:
            trenches_dir = Path(npy_dataset_folder) / "trenches" / name_string
            if not trenches_dir.exists():
                print(f"  Skipping conversion; folder not found: {trenches_dir}")
                continue
            destination_folder = Path(npy_dataset_folder) / "train" / "trenches" / name_string
            if destination_folder.exists():
                print(f"  Cleaning existing Terra destination: {destination_folder}")
                shutil.rmtree(destination_folder)
            img_folder = trenches_dir / "images"
            metadata_folder = trenches_dir / "metadata"
            occupancy_folder = trenches_dir / "occupancy"
            dumpability_folder = trenches_dir / "dumpability"
            actions_folder = trenches_dir / "actions"
            destination_folder.mkdir(parents=True, exist_ok=True)
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
                actions_folder=actions_folder,
            )
        print("  ✓ Terra format conversion complete")

    print("Separated trenches generation complete!")
    print(f"Data saved to {os.path.join(package_dir, 'data/terra/trenches/{name_string}')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate separated trench maps with pre-placed dirt.")
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
    generate_terra_format = not args.no_terra_format
    if args.no_terra_format:
        print("Terra format conversion disabled by --no-terra-format flag")
    else:
        print("Terra format conversion enabled (use --no-terra-format to disable)")
    no_dump_zones = args.no_dump_zones
    generate_trenches_separated_standalone(
        args.config,
        generate_terra_format=generate_terra_format,
        no_dump_zones=no_dump_zones,
    )
