#!/usr/bin/env python3
"""
Generate experimental 64x64 maps with foundations, roads, obstacles, and a smaller dump zone flush with the border.
Differences to the standard generator:
 - Target resolution 64x64
 - No border offset for dump zone placement (flush with the map edge)
 - Smaller dump zone sizes by default
 - Roads are encoded via the dumpability map (non-dumpable), not drawn in the image
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


def create_single_dump_zone_64x64_flush(img_terra_pad, size_dump_min, size_dump_max, foundation_mask):
    """
    Create exactly 1 dump zone that is flush with the outer border (touching the map edge),
    avoiding overlaps with foundations and dig zones. Uses smaller sizes by default.
    """
    height, width = img_terra_pad.shape[:2]
    dump_cumulative_mask = np.zeros((height, width), dtype=np.bool_)

    border_offset = 0
    max_attempts = 200
    for attempt in range(max_attempts):
        # Force corner placement: choose strictly among the four corners
        corner_choice = random.randint(0, 3)
        dump_width = 16
        dump_height = 12
        if corner_choice in (0, 1):
            y = 0
            x = 0 if corner_choice == 0 else max(0, width - dump_width)
        else:
            y = max(0, height - dump_height)
            x = 0 if corner_choice == 2 else max(0, width - dump_width)

        dump_area_found = foundation_mask[y:y+dump_height, x:x+dump_width]
        dig_area = np.all(img_terra_pad[y:y+dump_height, x:x+dump_width] == color_dict["digging"], axis=-1)

        if not np.any(dump_area_found) and not np.any(dig_area):
            img_terra_pad[y:y+dump_height, x:x+dump_width] = color_dict["dumping"]
            dump_cumulative_mask[y:y+dump_height, x:x+dump_width] = True
            # Determine edge label without relying on border_choice
            edge_label = (
                "top" if y == 0 else (
                    "bottom" if y + dump_height >= height else (
                        "left" if x == 0 else "right"
                    )
                )
            )
            print(f"Placed flush dump zone {dump_width}x{dump_height} at ({x}, {y}) on border {edge_label}")
            break
    else:
        raise RuntimeError("Failed to place a flush dump zone after attempts. Try smaller sizes or fewer obstacles.")

    # Return dump rectangle (x, y, w, h) for downstream road placement
    return img_terra_pad, dump_cumulative_mask, (x, y, dump_width, dump_height)


def create_experimental_64x64_maps_roads_noborder(
    n_imgs=300,
    max_size=64,
    dataset_path="data/openstreet",
    n_obs_min=2,
    n_obs_max=4,
    size_obstacle_min=3,
    size_obstacle_max=8,
    expansion_factor=1,
    all_dumpable=False,
    copy_metadata=True,
    has_dumpability=False,
    center_padding=True,
    n_dump_min=1,
    n_dump_max=1,
    size_dump_min=8,
    size_dump_max=14,
    experimental_variations=True,
    road_width=10):
    """
    Creates experimental 64x64 foundation environments with roads and a smaller dump zone flush to the border.
    """
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", "foundations_dumpzones_roads")
    print(f"Creating experimental 64x64 maps (roads, no border) - saving to: foundations_dumpzones_roads/")

    downsampling_factors = {
        save_folder: 2,
    }

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

            print(f"Processing experimental 64x64 (roads, no border) map nr {i + 1}")

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

            # Match downsampling strategy from foundations_dumpzones_v3 for consistent foundation sizing
            downsample_factor_w = int(max(1, math.ceil(img.shape[1] / max_size))) * downsampling_factor
            downsample_factor_h = int(max(1, math.ceil(img.shape[0] / max_size))) * downsampling_factor

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

            if center_padding:
                xdim = max_size - img_terra.shape[0]
                ydim = max_size - img_terra.shape[1]
                img_terra_pad = np.zeros((max_size, max_size), dtype=img_terra.dtype)
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

            img_terra_pad = img_terra_pad.repeat(expansion_factor, 0).repeat(
                expansion_factor, 1
            )
            img_terra_pad = convert_terra_pad_to_color(img_terra_pad, color_dict)

            # Neutral everywhere except dig zones
            neutral_mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)
            img_terra_pad[neutral_mask] = color_dict["neutral"]

            # Foundation mask (non-neutral, non-dig)
            foundation_mask = np.all(img_terra_pad != color_dict["neutral"], axis=-1) & np.all(img_terra_pad != color_dict["digging"], axis=-1)

            cumulative_mask = np.zeros(img_terra_pad.shape[:2], dtype=np.bool_)
            cumulative_mask[np.all(img_terra_pad == color_dict["digging"], axis=-1)] = True

            # Create a single dump zone flush with border (smaller sizes) BEFORE obstacles
            img_terra_pad, dump_cumulative_mask, dump_rect = create_single_dump_zone_64x64_flush(
                img_terra_pad, size_dump_min, size_dump_max, foundation_mask
            )
            cumulative_mask = dump_cumulative_mask | cumulative_mask

            # Add roads BEFORE obstacles, enforcing at least one road touches the dump zone
            img_terra_pad, _, road_positions = create_roads_64x64(
                img_terra_pad,
                cumulative_mask,
                road_width=road_width,
                must_touch_rect=dump_rect,
            )
            # Reserve road area so obstacles can't overwrite it
            for x, y, w, h in road_positions:
                cumulative_mask[y:y+h, x:x+w] = True

            # After dump zone and road are placed, add obstacles (cap to max 2)
            _n_obs_max = min(n_obs_max, 2)
            _n_obs_min = min(n_obs_min, _n_obs_max)
            occ, cumulative_mask = add_obstacles(
                img_terra_pad,
                cumulative_mask,
                _n_obs_min,
                _n_obs_max,
                size_obstacle_min,
                size_obstacle_max,
            )

            # Dumpability map: start dumpable everywhere
            dmp = np.ones_like(img_terra_pad) * 255

            # Mark road areas as non-dumpable
            road_mask = np.zeros_like(dump_cumulative_mask, dtype=bool)
            for x, y, w, h in road_positions:
                road_mask[y:y+h, x:x+w] = True
            dmp[road_mask] = color_dict["nondumpable"]

            # If dump zones overlap roads, make those overlapping tiles dumpable
            dump_mask = np.all(img_terra_pad == color_dict["dumping"], axis=-1)
            overlap_mask = road_mask & dump_mask
            if np.any(overlap_mask):
                dmp[overlap_mask] = 255
                print(f"Fixed {np.sum(overlap_mask)} overlapping tiles: made them dumpable in dumpability map")

            if experimental_variations:
                if random.random() < 0.2:
                    n_restricted = random.randint(1, 2)
                    for _ in range(n_restricted):
                        size = random.randint(3, 5)
                        x = random.randint(6, max_size - size - 6)
                        y = random.randint(6, max_size - size - 6)
                        area_mask = cumulative_mask[y:y+size, x:x+size]
                        if not np.any(area_mask):
                            cumulative_mask[y:y+size, x:x+size] = True
                            print(f"Added restricted zone of size {size} at ({x}, {y})")

            save_or_display_image(img_terra_pad, occ, dmp, metadata, curriculum_level, n)

    print("Experimental 64x64 (roads, no border) maps created successfully.")


def generate_experimental_64x64_roads_noborder_terra(dataset_folder, size, n_imgs):
    """Convert roads-noborder experimental 64x64 maps to Terra format."""
    print("Converting experimental 64x64 (roads, no border) maps to Terra format...")

    experimental_dir = Path(dataset_folder) / "foundations_dumpzones_roads"
    if not experimental_dir.exists():
        print(f"  foundations_dumpzones_roads directory not found: {experimental_dir}")
        return

    print(f"  Found foundations_dumpzones_roads folder - will convert to train/foundations_dumpzones_roads")

    img_folder = experimental_dir / "images"
    metadata_folder = experimental_dir / "metadata"
    occupancy_folder = experimental_dir / "occupancy"
    dumpability_folder = experimental_dir / "dumpability"
    destination_folder = Path(dataset_folder) / "train" / "foundations_dumpzones_roads"

    destination_folder.mkdir(parents=True, exist_ok=True)
    print(f"  Created destination directory: {destination_folder}")

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

    print(f"  foundations_dumpzones_roads conversion completed")


def create_roads_64x64(img_terra_pad, cumulative_mask, road_width=10, must_touch_rect=None):
    """
    Create 1-2 roads with variety: single road, parallel roads, or crossing roads.
    Roads are placed near borders with consistent width, avoiding obstacles. Roads are not drawn on the main image;
    they are only recorded for the dumpability map as non-dumpable regions.
    """
    height, width = img_terra_pad.shape[:2]

    road_config = "single"
    roads_to_add = []

    # If we must touch a given rectangle (dump zone), place a border-aligned road on the same edge
    def add_forced_touching_road(rect):
        x0, y0, w0, h0 = rect
        obstacle_mask = np.all(img_terra_pad == color_dict["obstacle"], axis=-1)
        # Top edge
        if y0 == 0:
            y_try = 0
            x, y, w, h = 0, y_try, width, road_width
            if not np.any(obstacle_mask[y:y+h, x:x+w]):
                return {"direction": "horizontal", "y": y_try, "x": 0, "width": width, "height": road_width}
        # Bottom edge
        if y0 + h0 >= height:
            y_try = max(0, height - road_width)
            x, y, w, h = 0, y_try, width, road_width
            if not np.any(obstacle_mask[y:y+h, x:x+w]):
                return {"direction": "horizontal", "y": y_try, "x": 0, "width": width, "height": road_width}
        # Left edge
        if x0 == 0:
            x_try = 0
            x, y, w, h = x_try, 0, road_width, height
            if not np.any(obstacle_mask[y:y+h, x:x+w]):
                return {"direction": "vertical", "x": x_try, "y": 0, "width": road_width, "height": height}
        # Right edge
        if x0 + w0 >= width:
            x_try = max(0, width - road_width)
            x, y, w, h = x_try, 0, road_width, height
            if not np.any(obstacle_mask[y:y+h, x:x+w]):
                return {"direction": "vertical", "x": x_try, "y": 0, "width": road_width, "height": height}
        return None

    forced = None
    if must_touch_rect is not None:
        # Randomly choose one road aligned with one of the dump zone edges (dump zones are in corners, so they touch 2 edges)
        x0, y0, w0, h0 = must_touch_rect
        dump_on_top = (y0 == 0)
        dump_on_bottom = (y0 + h0 >= height)
        dump_on_left = (x0 == 0)
        dump_on_right = (x0 + w0 >= width)

        roads_to_add = []
        # Collect all edges the dump zone touches (corners touch 2 edges)
        possible_roads = []
        
        if dump_on_top:
            possible_roads.append({"direction": "horizontal", "y": 0, "x": 0, "width": width, "height": road_width})
        if dump_on_bottom:
            possible_roads.append({"direction": "horizontal", "y": max(0, height - road_width), "x": 0, "width": width, "height": road_width})
        if dump_on_left:
            possible_roads.append({"direction": "vertical", "x": 0, "y": 0, "width": road_width, "height": height})
        if dump_on_right:
            possible_roads.append({"direction": "vertical", "x": max(0, width - road_width), "y": 0, "width": road_width, "height": height})
        
        # Randomly choose one of the possible roads
        if len(possible_roads) > 0:
            roads_to_add.append(random.choice(possible_roads))
        else:
            # Fallback to previous forced touching when ambiguous
            forced = add_forced_touching_road(must_touch_rect)
            if forced is not None:
                roads_to_add.append(forced)

    if must_touch_rect is None:
        if random.random() < 0.5:
            side = random.choice(["top", "bottom"])
            y_pos = 0 if side == "top" else max(0, height - road_width)
            roads_to_add.append({"direction": "horizontal", "y": y_pos, "x": 0, "width": width, "height": road_width})
        else:
            side = random.choice(["left", "right"])
            x_pos = 0 if side == "left" else max(0, width - road_width)
            roads_to_add.append({"direction": "vertical", "x": x_pos, "y": 0, "width": road_width, "height": height})

    # Enforce exactly one road
    if must_touch_rect is not None and forced is not None:
        roads_to_add = [forced]
    if len(roads_to_add) > 1:
        roads_to_add = roads_to_add[:1]
    if len(roads_to_add) == 0:
        fallback = {"direction": "horizontal", "y": 0, "x": 0, "width": width, "height": road_width}
        roads_to_add = [fallback]

    road_positions = []
    for road in roads_to_add:
        x, y, w, h = road.get("x", 0), road.get("y", 0), road.get("width", 0), road.get("height", 0)
        obstacle_mask = np.all(img_terra_pad == color_dict["obstacle"], axis=-1)
        area_has_obstacles = np.any(obstacle_mask[y:y+h, x:x+w])
        if area_has_obstacles:
            print(f"Warning: Road at ({x}, {y}) intersects obstacles; marking as road anyway.")
        road_positions.append((x, y, w, h))
        print(f"Added {road['direction']} road at ({x}, {y}) with size {w}x{h} (invisible, non-dumpable)")

    # If we required a touching road but failed to add any, try once more with horizontal preference
    if must_touch_rect is not None and len(road_positions) == 0:
        fallback = (0, 0, width, road_width)
        road_positions.append(fallback)
        print("Added fallback horizontal road at top edge to touch dump zone")

    return img_terra_pad, cumulative_mask, road_positions


def main():
    print("Starting experimental 64x64 (roads, no border) map generation...")

    # Generate maps
    create_experimental_64x64_maps_roads_noborder(
        n_imgs=600,
        max_size=64,
        dataset_path="data/openstreet",
        n_obs_min=2,
        n_obs_max=4,
        size_obstacle_min=3,
        size_obstacle_max=8,
        size_dump_min=4,   # smaller for 64x64
        size_dump_max=10,
        experimental_variations=True,
        road_width=10,
    )

    print("Experimental 64x64 (roads, no border) map generation completed!")

    # Convert to Terra format
    print("\nStarting Terra format conversion...")
    dataset_folder = os.path.join(PACKAGE_DIR, "data", "terra")
    size = (64, 64)
    n_imgs = 600

    generate_experimental_64x64_roads_noborder_terra(dataset_folder, size, n_imgs)
    print("Terra format conversion completed!")
    print("Experimental 64x64 (roads, no border) maps are now ready for use in Terra environment!")


if __name__ == "__main__":
    main()


