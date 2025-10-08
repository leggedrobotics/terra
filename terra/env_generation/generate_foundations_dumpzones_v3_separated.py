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

name_string = "foundations_dumpzones_v3_separated"

def create_spatially_separated_dig_zones(img_terra_pad, n_zones, size_min, size_max, separation_px, occupied_mask):
    """
    Create multiple dig zones (foundations) that are spatially separated by at least separation_px.

    Args:
        img_terra_pad: RGB image holding current terra colors
        n_zones: number of dig zones to place
        size_min, size_max: square side length range for each zone
        separation_px: minimum center-to-center distance between zones
        occupied_mask: boolean mask of pixels that cannot be used (already occupied)

    Returns:
        img_terra_pad, occupied_mask updated with dig zones
    """
    h, w = img_terra_pad.shape[:2]
    centers = []
    attempts = 0
    max_attempts = 500

    while len(centers) < n_zones and attempts < max_attempts:
        attempts += 1
        side = random.randint(size_min, size_max)
        x = random.randint(3, w - side - 3)
        y = random.randint(3, h - side - 3)

        # Check occupancy overlap
        if np.any(occupied_mask[y:y+side, x:x+side]):
            continue

        # Enforce separation from already placed dig zones
        cx, cy = x + side // 2, y + side // 2
        ok = True
        for (px, py, pside) in centers:
            if (cx - px) ** 2 + (cy - py) ** 2 < separation_px ** 2:
                ok = False
                break
        if not ok:
            continue

        # Place zone
        img_terra_pad[y:y+side, x:x+side] = color_dict["digging"]
        occupied_mask[y:y+side, x:x+side] = True
        centers.append((cx, cy, side))

    if len(centers) < n_zones:
        print(f"Warning: requested {n_zones} separated dig zones, placed {len(centers)}")
    return img_terra_pad, occupied_mask


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


def create_foundations_separated_standalone(
                                    n_imgs=600,
                                    max_size=64,
                                    dataset_path="data/openstreet",
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
                                    n_separated_dig_zones=2,
                                    dig_zone_size_min=10,
                                    dig_zone_size_max=14,
                                    dig_zone_min_separation_px=23,
                                    no_dump_zones=False):
    """
    Same as v3, but additionally forces multiple dig zones to be spatially separated.
    """
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", name_string)
    print(f"Creating separated foundations with pre-placed dirt - saving to: {name_string}/")

    if os.path.exists(save_folder):
        print(f"Cleaning existing output folder: {save_folder}")
        shutil.rmtree(save_folder)

    downsampling_factors = { save_folder: 2.2 }
    full_dataset_path = os.path.join(PACKAGE_DIR, dataset_path)

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
            print(f"Processing separated foundation nr {i + 1}")
            n = int(fn.split(".png")[0].split("_")[1])
            filename = filename_start + f"_{n}.png"
            file_path = img_folder / filename
            occupancy_path = occupancy_folder / filename
            img = cv2.imread(str(file_path))
            occupancy = cv2.imread(str(occupancy_path))
            if has_dumpability:
                dumpability_path = dumpability_folder / filename
                dumpability = cv2.imread(str(dumpability_path))
            with open(metadata_folder / f"{filename.split('.png')[0]}.json") as json_file:
                metadata = json.load(json_file)

            downsample_factor_w = int(max(1, math.ceil(img.shape[1] / max_size)) * downsampling_factor)
            downsample_factor_h = int(max(1, math.ceil(img.shape[0] / max_size)) * downsampling_factor)
            img_downsampled = skimage.measure.block_reduce(img, (downsample_factor_h, downsample_factor_w, 1), np.max)
            img = img_downsampled
            occupancy_downsampled = skimage.measure.block_reduce(occupancy, (downsample_factor_h, downsample_factor_w, 1), np.min, cval=0)
            occupancy = occupancy_downsampled
            if has_dumpability:
                dumpability_downsampled = skimage.measure.block_reduce(dumpability, (downsample_factor_h, downsample_factor_w, 1), np.min, cval=0)
                dumpability = dumpability_downsampled

            img_terra = _convert_img_to_terra(img, all_dumpable)

            if center_padding:
                xdim = max_size - img_terra.shape[0]
                ydim = max_size - img_terra.shape[1]
                img_terra_pad = np.ones((max_size, max_size), dtype=img_terra.dtype)
                img_terra_pad[
                    xdim // 2 : max_size - (xdim - xdim // 2),
                    ydim // 2 : max_size - (ydim - ydim // 2),
                ] = img_terra
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

            img_terra_pad = img_terra_pad.repeat(expansion_factor, 0).repeat(expansion_factor, 1)
            img_terra_pad = convert_terra_pad_to_color(img_terra_pad, color_dict)

            # Start neutral everywhere except pre-existing dig zones
            neutral_mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)
            img_terra_pad[neutral_mask] = color_dict["neutral"]

            # Build foundation mask (non-neutral non-digging)
            foundation_mask = np.all(img_terra_pad != color_dict["neutral"], axis=-1) & np.all(img_terra_pad != color_dict["digging"], axis=-1)

            # Place one dump zone on border
            img_terra_pad, dump_cumulative_mask = create_single_dump_zone(
                img_terra_pad, size_dump_min, size_dump_max, foundation_mask, min_distance_from_dig=6
            )

            # Ensure two different foundation shapes; first, randomly translate the primary dig shape (avoid always centered)
            dig_mask = np.all(img_terra_pad == color_dict["digging"], axis=-1)
            if np.any(dig_mask):
                rows0, cols0 = np.where(dig_mask)
                r0min, r0max = rows0.min(), rows0.max()
                c0min, c0max = cols0.min(), cols0.max()
                h, w = img_terra_pad.shape[:2]
                # Candidate shifts; shuffle to vary placement
                d = max(dig_zone_min_separation_px // 2, min(h, w) // 6)
                candidates_first = [
                    ( d,  d), ( d, -d), (-d,  d), (-d, -d),
                    ( 0,  d), ( d,  0), ( 0, -d), (-d,  0),
                    ( 2*d, 0), (0, 2*d), (-2*d, 0), (0, -2*d),
                ]
                random.shuffle(candidates_first)
                moved_first = False
                occupied_static = foundation_mask | dump_cumulative_mask
                # Clear current dig temporarily to test placement
                img_terra_pad[dig_mask] = color_dict["neutral"]
                for dr, dc in candidates_first:
                    rrmin = r0min + dr
                    rrmax = r0max + dr
                    ccmin = c0min + dc
                    ccmax = c0max + dc
                    if rrmin < 0 or ccmin < 0 or rrmax >= h or ccmax >= w:
                        continue
                    # Overlap with static occupied areas (do not collide with buildings/dump)
                    if np.any(occupied_static[rrmin:rrmax+1, ccmin:ccmax+1] & dig_mask[r0min:r0max+1, c0min:c0max+1]):
                        continue
                    trans_mask_first = np.zeros_like(dig_mask)
                    trans_mask_first[rrmin:rrmax+1, ccmin:ccmax+1] = dig_mask[r0min:r0max+1, c0min:c0max+1]
                    img_terra_pad[trans_mask_first] = color_dict["digging"]
                    moved_first = True
                    break
                if not moved_first:
                    # restore original if no valid move
                    img_terra_pad[dig_mask] = color_dict["digging"]
                # Recompute dig mask and centroid after possible move
                dig_mask = np.all(img_terra_pad == color_dict["digging"], axis=-1)
                rows0, cols0 = np.where(dig_mask)
                c0 = np.array([rows0.mean(), cols0.mean()], dtype=np.float32)

                # Try up to N attempts to sample a second foundation image and place its dig shape
                h, w = img_terra_pad.shape[:2]
                placed_second = False
                for attempt_img in range(30):
                    # Sample a different foundation index
                    try_idx = n
                    for _ in range(5):
                        rnd = random.choice(os.listdir(img_folder))
                        if rnd.endswith('.png'):
                            try_idx = int(rnd.split(".png")[0].split("_")[1])
                            if try_idx != n:
                                break
                    filename2 = filename_start + f"_{try_idx}.png"
                    file_path2 = img_folder / filename2
                    occupancy_path2 = occupancy_folder / filename2
                    if not file_path2.exists() or not occupancy_path2.exists():
                        continue

                    # Load and process second image using same downsampling/padding flow
                    img2 = cv2.imread(str(file_path2))
                    occupancy2 = cv2.imread(str(occupancy_path2))
                    img2_ds = skimage.measure.block_reduce(img2, (downsample_factor_h, downsample_factor_w, 1), np.max)
                    occ2_ds = skimage.measure.block_reduce(occupancy2, (downsample_factor_h, downsample_factor_w, 1), np.min, cval=0)
                    img2_terra = _convert_img_to_terra(img2_ds, all_dumpable)

                    h2, w2 = img2_terra.shape[:2]
                    # Guard: if second terra image exceeds canvas, crop to fit
                    if h2 > max_size or w2 > max_size:
                        h2c = min(h2, max_size)
                        w2c = min(w2, max_size)
                        img2_terra = img2_terra[:h2c, :w2c]
                        h2, w2 = img2_terra.shape[:2]
                    if center_padding:
                        img2_pad = np.ones((max_size, max_size), dtype=img2_terra.dtype)
                        x0 = max(0, (max_size - h2) // 2)
                        y0 = max(0, (max_size - w2) // 2)
                        img2_pad[x0:x0 + h2, y0:y0 + w2] = img2_terra
                    else:
                        img2_pad = np.zeros((max_size, max_size), dtype=img2_terra.dtype)
                        img2_pad[0:h2, 0:w2] = img2_terra

                    img2_col = convert_terra_pad_to_color(img2_pad, color_dict)
                    dig2 = np.all(img2_col == color_dict["digging"], axis=-1)
                    if not np.any(dig2):
                        continue

                    rows2, cols2 = np.where(dig2)
                    r2min, r2max = rows2.min(), rows2.max()
                    c2min, c2max = cols2.min(), cols2.max()

                    # Occupancy to avoid: original foundations, dump zones, and existing dig
                    occupied = foundation_mask | dump_cumulative_mask | dig_mask

                    # Candidate shifts to try
                    d = max(dig_zone_min_separation_px, min(h, w) // 3)
                    candidates = [
                        ( d,  d), ( d, -d), (-d,  d), (-d, -d),
                        ( 0,  d), ( d,  0), ( 0, -d), (-d,  0),
                    ]
                    # Add randomized candidate shifts to broaden search space
                    rng_candidates = []
                    max_r = max(d, min(h, w) // 2)
                    for _ in range(32):
                        r = random.randint(d, max_r)
                        theta = random.uniform(0, 2 * math.pi)
                        dr = int(round(r * math.sin(theta)))
                        dc = int(round(r * math.cos(theta)))
                        rng_candidates.append((dr, dc))
                    candidates.extend(rng_candidates)
                    random.shuffle(candidates)
                    for dr, dc in candidates:
                        rrmin = r2min + dr
                        rrmax = r2max + dr
                        ccmin = c2min + dc
                        ccmax = c2max + dc
                        # Bounds check
                        if rrmin < 0 or ccmin < 0 or rrmax >= h or ccmax >= w:
                            continue
                        # Overlap check on bbox region
                        if np.any(occupied[rrmin:rrmax+1, ccmin:ccmax+1] & dig2[r2min:r2max+1, c2min:c2max+1]):
                            continue
                        # Centroid separation check
                        c1 = np.array([rows2.mean() + dr, cols2.mean() + dc], dtype=np.float32)
                        if np.linalg.norm(c0 - c1) < dig_zone_min_separation_px:
                            continue

                        # Build translated mask for second shape and apply
                        trans_mask2 = np.zeros_like(dig2)
                        trans_mask2[rrmin:rrmax+1, ccmin:ccmax+1] = dig2[r2min:r2max+1, c2min:c2max+1]
                        img_terra_pad[trans_mask2] = color_dict["digging"]
                        placed_second = True
                        break
                    if placed_second:
                        break
                if not placed_second:
                    print("Warning: failed to place a second (different) foundation shape with required separation.")

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
                img_terra_pad[neutral_mask] = color_dict["dumping"]
                dump_cumulative_mask = neutral_mask

            # Save outputs
            Path(curriculum_level, "images").mkdir(parents=True, exist_ok=True)
            Path(curriculum_level, "metadata").mkdir(parents=True, exist_ok=True)
            Path(curriculum_level, "occupancy").mkdir(parents=True, exist_ok=True)
            Path(curriculum_level, "dumpability").mkdir(parents=True, exist_ok=True)
            Path(curriculum_level, "actions").mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(Path(curriculum_level, "images", f"trench_{n}.png")), img_terra_pad)
            cv2.imwrite(str(Path(curriculum_level, "occupancy", f"trench_{n}.png")), occ)
            cv2.imwrite(str(Path(curriculum_level, "dumpability", f"trench_{n}.png")), dmp)
            cv2.imwrite(str(Path(curriculum_level, "actions", f"trench_{n}.png")), action_map)

            metadata_with_curriculum = metadata.copy()
            metadata_with_curriculum["curriculum_level"] = curriculum_level
            with open(Path(curriculum_level, "metadata", f"trench_{n}.json"), "w") as json_file:
                json.dump(metadata_with_curriculum, json_file)

    print("Separated foundations with pre-placed dirt created successfully.")


def generate_foundations_separated_standalone(config_path="config/env_generation_config.yaml",
                                              generate_terra_format=True,
                                              no_dump_zones=False):
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(package_dir + "/" + config_path, "r") as file:
        config = yaml.safe_load(file)

    os.makedirs("data/", exist_ok=True)
    os.makedirs("data/terra/", exist_ok=True)
    os.makedirs("data/openstreet/", exist_ok=True)

    n_imgs = config["n_imgs"]
    print("Generating FOUNDATIONS (SEPARATED) maps...")

    foundations_config = config.get("foundations", {})
    if "min_size" in foundations_config and "max_size" in foundations_config:
        foundation_min_size = foundations_config.get("min_size")
        foundation_max_size = foundations_config.get("max_size")
    else:
        raise ValueError("min_size and max_size must be provided in the config file")
    max_buildings = 150
    print(f"Foundation config - min_size: {foundation_min_size}, max_size: {foundation_max_size}, max_buildings: {max_buildings}")

    dataset_folder = os.path.join(package_dir, "data", "openstreet")
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

    print("  → Generating SEPARATED FOUNDATIONS maps...")
    create_foundations_separated_standalone(
        n_imgs=n_imgs,
        max_size=foundations_config.get("max_size", 64),
        no_dump_zones=no_dump_zones,
    )
    print(f"  ✓ Separated foundations maps saved to: data/terra/{name_string}")

    if generate_terra_format:
        print("Converting separated foundations data to Terra format...")
        sizes = [(64, 64)]
        npy_dataset_folder = package_dir + "/data/terra"
        for size in sizes:
            foundations_dir = Path(npy_dataset_folder) / name_string
            if not foundations_dir.exists():
                print(f"  Skipping conversion; folder not found: {foundations_dir}")
                continue
            destination_folder = Path(npy_dataset_folder) / "train" / name_string
            if destination_folder.exists():
                print(f"  Cleaning existing Terra destination: {destination_folder}")
                shutil.rmtree(destination_folder)
            img_folder = foundations_dir / "images"
            metadata_folder = foundations_dir / "metadata"
            occupancy_folder = foundations_dir / "occupancy"
            dumpability_folder = foundations_dir / "dumpability"
            actions_folder = foundations_dir / "actions"
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

    print("Separated foundations generation complete!")
    print(f"Data saved to {os.path.join(package_dir, 'data/terra/{name_string}')}" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate separated foundation maps with pre-placed dirt.")
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
    generate_foundations_separated_standalone(
        args.config,
        generate_terra_format=generate_terra_format,
        no_dump_zones=no_dump_zones,
    )


