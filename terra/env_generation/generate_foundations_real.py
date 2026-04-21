#!/usr/bin/env python
import argparse
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import skimage
import yaml

from terra.env_generation.convert_to_terra import (
    _convert_dumpability_to_terra,
    _convert_img_to_terra,
    _convert_occupancy_to_terra,
)
import terra.env_generation.convert_to_terra as convert_to_terra
from terra.env_generation.procedural_data import (
    add_obstacles,
    convert_terra_pad_to_color,
    save_or_display_image,
)
from terra.env_generation.utils import color_dict


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
name_string = "foundations_real"


def _empty_layer_like(img):
    return np.ones_like(img, dtype=np.uint8) * 255


def _foundation_side_masks_from_obstacle(dig_mask, obstacle_mask, vertical):
    rows, cols = dig_mask.shape
    if vertical:
        obstacle_cols = np.where(np.any(obstacle_mask, axis=0))[0]
        if obstacle_cols.size == 0:
            return None
        left_end = int(obstacle_cols.min())
        right_start = int(obstacle_cols.max()) + 1
        left_mask = np.zeros((rows, cols), dtype=bool)
        right_mask = np.zeros((rows, cols), dtype=bool)
        left_mask[:, :left_end] = True
        right_mask[:, right_start:] = True
        left_has_foundation = np.any(dig_mask & left_mask)
        right_has_foundation = np.any(dig_mask & right_mask)
        if left_has_foundation == right_has_foundation:
            return None
        foundation_side = left_mask if left_has_foundation else right_mask
        blocked_side = right_mask if left_has_foundation else left_mask
        return foundation_side, blocked_side

    obstacle_rows = np.where(np.any(obstacle_mask, axis=1))[0]
    if obstacle_rows.size == 0:
        return None
    top_end = int(obstacle_rows.min())
    bottom_start = int(obstacle_rows.max()) + 1
    top_mask = np.zeros((rows, cols), dtype=bool)
    bottom_mask = np.zeros((rows, cols), dtype=bool)
    top_mask[:top_end, :] = True
    bottom_mask[bottom_start:, :] = True
    top_has_foundation = np.any(dig_mask & top_mask)
    bottom_has_foundation = np.any(dig_mask & bottom_mask)
    if top_has_foundation == bottom_has_foundation:
        return None
    foundation_side = top_mask if top_has_foundation else bottom_mask
    blocked_side = bottom_mask if top_has_foundation else top_mask
    return foundation_side, blocked_side


def add_full_span_obstacles_and_nondump(
    img_terra_pad,
    dig_mask,
    n_obs_min,
    n_obs_max,
    obstacle_width=2,
    max_attempts=400,
):
    height, width = dig_mask.shape
    occ = _empty_layer_like(img_terra_pad)
    dmp = _empty_layer_like(img_terra_pad)
    obstacle_mask_total = np.zeros((height, width), dtype=bool)
    nondump_mask_total = np.zeros((height, width), dtype=bool)
    foundation_side_mask = np.ones((height, width), dtype=bool)

    target_obstacles = random.randint(n_obs_min, n_obs_max)
    placed = 0
    attempts = 0

    while placed < target_obstacles and attempts < max_attempts:
        attempts += 1
        vertical = random.choice([True, False])
        obstacle_mask = np.zeros((height, width), dtype=bool)

        # Keep at least one free row/column on each side so that row 0 and
        # column 0 of the padding/occupancy mask are never fully blocked.
        # Otherwise GridWorld.max_traversable_x/y collapse to 0, which causes
        # the agent-spawn rejection sampler (jax.lax.while_loop) to hang
        # forever at the first env reset (see terra/agent.py).
        edge_margin = max(1, obstacle_width)
        if vertical:
            if width <= obstacle_width + 2 * edge_margin:
                continue
            start = random.randint(edge_margin, width - obstacle_width - edge_margin)
            obstacle_mask[:, start : start + obstacle_width] = True
        else:
            if height <= obstacle_width + 2 * edge_margin:
                continue
            start = random.randint(edge_margin, height - obstacle_width - edge_margin)
            obstacle_mask[start : start + obstacle_width, :] = True

        if np.any(obstacle_mask & dig_mask):
            continue
        if np.any(obstacle_mask & obstacle_mask_total):
            continue

        side_masks = _foundation_side_masks_from_obstacle(dig_mask, obstacle_mask, vertical)
        if side_masks is None:
            continue

        candidate_foundation_side, blocked_side = side_masks
        obstacle_mask_total |= obstacle_mask
        nondump_mask_total |= blocked_side
        foundation_side_mask &= candidate_foundation_side | dig_mask
        placed += 1

    if placed < target_obstacles:
        print(
            f"Warning: requested {target_obstacles} full-span obstacles, placed {placed}."
        )

    occ[obstacle_mask_total] = np.array(color_dict["obstacle"], dtype=np.uint8)
    dmp[nondump_mask_total] = np.array(color_dict["nondumpable"], dtype=np.uint8)
    return occ, dmp, obstacle_mask_total, nondump_mask_total, foundation_side_mask


def create_single_dump_zone_on_foundation_side(
    img_terra_pad,
    allowed_mask,
    size_dump_min,
    size_dump_max,
):
    height, width = allowed_mask.shape
    dump_cumulative_mask = np.zeros((height, width), dtype=bool)
    dig_mask = np.all(img_terra_pad == color_dict["digging"], axis=-1)

    for size_offset in (0, -3, -5):
        current_min = max(3, size_dump_min + size_offset)
        current_max = max(current_min, size_dump_max + size_offset)
        for _ in range(120):
            dump_size = random.randint(current_min, current_max)
            if dump_size > height or dump_size > width:
                continue
            x = random.randint(0, width - dump_size)
            y = random.randint(0, height - dump_size)
            candidate = np.zeros((height, width), dtype=bool)
            candidate[y : y + dump_size, x : x + dump_size] = True
            if np.any(candidate & dig_mask):
                continue
            if np.all(allowed_mask[y : y + dump_size, x : x + dump_size]):
                img_terra_pad[y : y + dump_size, x : x + dump_size] = color_dict["dumping"]
                dump_cumulative_mask[y : y + dump_size, x : x + dump_size] = True
                print(
                    f"Successfully placed dump zone of size {dump_size} at position ({x}, {y})"
                )
                return img_terra_pad, dump_cumulative_mask

    raise RuntimeError(
        "Failed to place a dump zone on the foundation side of the full-span obstacle."
    )


def add_small_rectangle_obstacles(
    img_terra_pad,
    base_occ,
    base_mask,
    n_obs_min=1,
    n_obs_max=3,
    size_obstacle_min=3,
    size_obstacle_max=6,
):
    small_occ, updated_mask = add_obstacles(
        img_terra_pad,
        base_mask.copy(),
        n_obs_min,
        n_obs_max,
        size_obstacle_min,
        size_obstacle_max,
    )
    combined_occ = base_occ.copy()
    small_occ_mask = np.all(small_occ == color_dict["obstacle"], axis=-1)
    combined_occ[small_occ_mask] = np.array(color_dict["obstacle"], dtype=np.uint8)
    return combined_occ, updated_mask


def create_foundations_dumpzones_standalone(
    n_imgs=600,
    max_size=64,
    dataset_path="data/openstreet",
    n_obs_min=1,
    n_obs_max=1,
    obstacle_width=2,
    n_small_obs_min=1,
    n_small_obs_max=3,
    size_small_obstacle_min=3,
    size_small_obstacle_max=6,
    expansion_factor=1,
    all_dumpable=False,
    copy_metadata=True,
    has_dumpability=False,
    center_padding=True,
    no_dump_zones=False,
    full_dumpzone=False,
    size_dump_min=16,
    size_dump_max=16,
):
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", name_string)
    print(f"Creating foundations with dump zones and full-span obstacles - saving to: {name_string}/")

    downsampling_factors = {
        save_folder: 2.0,
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

            with open(metadata_folder / f"{filename.split('.png')[0]}.json") as json_file:
                metadata = json.load(json_file)

            downsample_factor_w = int(
                max(1, math.ceil(img.shape[1] / max_size)) * downsampling_factor
            )
            downsample_factor_h = int(
                max(1, math.ceil(img.shape[0] / max_size)) * downsampling_factor
            )

            img = skimage.measure.block_reduce(
                img, (downsample_factor_h, downsample_factor_w, 1), np.max
            )
            occupancy = skimage.measure.block_reduce(
                occupancy, (downsample_factor_h, downsample_factor_w, 1), np.min, cval=0
            )
            if has_dumpability:
                dumpability = skimage.measure.block_reduce(
                    dumpability, (downsample_factor_h, downsample_factor_w, 1), np.min, cval=0
                )

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

            # Keep non-dig terrain dumpable by default so the generated target maps
            # stay bi-valued like standard foundations (-1 digging, +1 dumping).
            default_dumpable_mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)
            img_terra_pad[default_dumpable_mask] = color_dict["dumping"]
            dig_mask = np.all(img_terra_pad == color_dict["digging"], axis=-1)

            occ, dmp, obstacle_mask, nondump_mask, foundation_side_mask = (
                add_full_span_obstacles_and_nondump(
                    img_terra_pad,
                    dig_mask,
                    n_obs_min=n_obs_min,
                    n_obs_max=n_obs_max,
                    obstacle_width=obstacle_width,
                )
            )

            cumulative_mask = dig_mask | obstacle_mask | nondump_mask
            occ, cumulative_mask = add_small_rectangle_obstacles(
                img_terra_pad,
                occ,
                cumulative_mask,
                n_obs_min=n_small_obs_min,
                n_obs_max=n_small_obs_max,
                size_obstacle_min=size_small_obstacle_min,
                size_obstacle_max=size_small_obstacle_max,
            )
            obstacle_mask = np.all(occ == color_dict["obstacle"], axis=-1)

            allowed_dump_mask = foundation_side_mask & ~dig_mask & ~obstacle_mask & ~nondump_mask

            if full_dumpzone or no_dump_zones:
                img_terra_pad[allowed_dump_mask] = color_dict["dumping"]
            else:
                img_terra_pad, _ = create_single_dump_zone_on_foundation_side(
                    img_terra_pad,
                    allowed_dump_mask,
                    size_dump_min=size_dump_min,
                    size_dump_max=size_dump_max,
                )

            save_or_display_image(img_terra_pad, occ, dmp, metadata, curriculum_level, n)

    print("Foundations with full-span obstacles created successfully.")


def generate_foundations_dumpzones_standalone(
    config_path="config/env_generation_config.yaml",
    generate_terra_format=True,
    no_dump_zones=False,
    full_dumpzone=False,
):
    package_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    with open(os.path.join(package_dir, config_path), "r") as file:
        config = yaml.safe_load(file)

    os.makedirs("data/", exist_ok=True)
    os.makedirs("data/terra/", exist_ok=True)
    os.makedirs("data/openstreet/", exist_ok=True)

    n_imgs = config["n_imgs"]
    foundations_config = config.get("foundations", {})
    if "min_size" not in foundations_config or "max_size" not in foundations_config:
        raise ValueError("min_size and max_size must be provided in the config file")

    foundation_min_size = foundations_config.get("min_size")
    foundation_max_size = foundations_config.get("max_size")
    max_buildings = 150

    print("Generating FOUNDATIONS DUMPZONES 2.0 STANDALONE maps...")
    print(
        f"Foundation config - min_size: {foundation_min_size}, max_size: {foundation_max_size}, max_buildings: {max_buildings}"
    )

    dataset_folder = os.path.join(package_dir, "data", "openstreet")
    foundation_path = os.path.join(dataset_folder, "foundations", "images")
    if not os.path.exists(foundation_path):
        from terra.env_generation.generate_foundations import (
            create_foundations,
            download_foundations,
        )

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

    create_foundations_dumpzones_standalone(
        n_imgs=n_imgs,
        max_size=foundations_config.get("max_size", 64),
        no_dump_zones=no_dump_zones,
        full_dumpzone=full_dumpzone,
    )
    print(f"  ✓ Foundations dumpzones 2.0 maps saved to: data/terra/{name_string}")

    if generate_terra_format:
        print("Converting foundations dumpzones 2.0 standalone data to Terra format...")
        npy_dataset_folder = os.path.join(package_dir, "data", "terra")
        foundations_dir = Path(npy_dataset_folder) / name_string
        destination_folder = Path(npy_dataset_folder) / "train" / name_string

        if foundations_dir.exists():
            destination_folder.mkdir(parents=True, exist_ok=True)
            convert_to_terra._convert_all_imgs_to_terra(
                foundations_dir / "images",
                foundations_dir / "metadata",
                foundations_dir / "occupancy",
                foundations_dir / "dumpability",
                destination_folder,
                (64, 64),
                n_imgs,
                all_dumpable=False,
                copy_metadata=True,
                downsample=False,
                has_dumpability=True,
                center_padding=False,
                actions_folder=None,
            )
            print("  ✓ Terra format conversion complete")
        else:
            print(f"  Skipping conversion; folder not found: {foundations_dir}")

    print("Foundations 2.0 generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate foundation maps with a single dump zone, full-span 2-wide obstacles, "
            "and no-dump regions on the side without foundations."
        )
    )
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
        help="Make all allowed foundation-side free space dumpable instead of creating a specific dump zone",
    )
    parser.add_argument(
        "--full-dumpzone",
        action="store_true",
        help="Make all allowed foundation-side free space dumpable",
    )

    args = parser.parse_args()

    generate_terra_format = not args.no_terra_format
    no_dump_zones = True if not args.full_dumpzone else args.no_dump_zones
    full_dumpzone = args.full_dumpzone
    if full_dumpzone and no_dump_zones:
        raise ValueError("Use either --full-dumpzone or --no-dump-zones, not both.")


    generate_foundations_dumpzones_standalone(
        args.config,
        generate_terra_format=generate_terra_format,
        no_dump_zones=no_dump_zones,
        full_dumpzone=full_dumpzone,
    )
