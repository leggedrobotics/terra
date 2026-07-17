#!/usr/bin/env python
import os
import argparse
from pathlib import Path

from terra.env_generation.create_train_data import (
    generate_trenches_with_dumpzones,
)
import terra.env_generation.convert_to_terra as convert_to_terra


# Module-level name variable (mirrors pattern in foundations v2 script)
# This determines the output subdirectory under data/terra/trenches/
name_string = "single_dumpzone_v2"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate procedural trench maps with a single dump zone per map, "
            "then optionally convert them to Terra format. This script is fully "
            "standalone and does not read any YAML config."
        )
    )
    # Minimal controls; rest are fixed to sensible defaults
    parser.add_argument("--n-imgs", type=int, default=600, help="Number of maps to generate")
    parser.add_argument("--size", type=int, default=64, help="Square size for Terra conversion, e.g., 64 -> (64, 64)")
    parser.add_argument("--name", type=str, default=None, help="Optional name to override the default output folder name")
    parser.add_argument("--no-generate", action="store_true", help="Skip generating trench dumpzone maps (only convert existing)")
    parser.add_argument("--no-convert", action="store_true", help="Skip Terra format conversion (only generate raw maps)")

    args = parser.parse_args()

    # Compute the package dir (root of the terra package repo)
    package_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Ensure base folders exist
    os.makedirs(os.path.join(package_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(package_dir, "data", "terra"), exist_ok=True)

    # Resolve final output folder name using module-level name_string
    global name_string
    if args.name:
        name_string = args.name
    out_subdir = name_string
    save_folder = os.path.join(package_dir, "data", "terra", "trenches", out_subdir)
    os.makedirs(save_folder, exist_ok=True)

    # 1) Generate trenches with dump zones
    if not args.no_generate:
        print("Creating procedural trenches with dump zones...")
        # Keep trench width fixed at 2 tiles; preserve the existing length range.
        img_edge_min_val = 64
        img_edge_max_val = 64
        max_ratio_w, max_ratio_h = 0.20, 0.30

        sizes_small = (2, 2)
        sizes_long = (
            max(1, int(max_ratio_w * img_edge_min_val)),
            max(1, int(max_ratio_h * img_edge_max_val)),
        )
        generate_trenches_with_dumpzones(
            n_imgs=args.n_imgs,
            img_edge_min=64,
            img_edge_max=64,
            sizes_small=sizes_small,
            sizes_long=sizes_long,
            n_edges=(1, 1),
            resolution=1.0,
            save_folder=save_folder,
            n_obs_min=0,
            n_obs_max=2,
            size_obstacle_min=3,
            size_obstacle_max=6,
            n_nodump_min=0,
            n_nodump_max=1,
            size_nodump_min=4,
            size_nodump_max=8,
            diagonal=False,
            n_dump_min=1,
            n_dump_max=1,
            size_dump_min=15,
            size_dump_max=15,
        )
        print(f"\u2713 Trench maps with dump zones saved to: {save_folder}")
    else:
        print("Skipping generation per --no-generate")

    # 2) Convert to Terra format into train/trenches/<name>
    if not args.no_convert:
        print("Converting trench dumpzone maps to Terra format...")
        npy_dataset_folder = os.path.join(package_dir, "data", "terra")
        size_tuple = (args.size, args.size)

        src_root = Path(npy_dataset_folder) / "trenches" / out_subdir
        dst_root = Path(npy_dataset_folder) / "train" / "trenches" / out_subdir
        dst_root.mkdir(parents=True, exist_ok=True)

        img_folder = src_root / "images"
        metadata_folder = src_root / "metadata"
        occupancy_folder = src_root / "occupancy"
        dumpability_folder = src_root / "dumpability"

        convert_to_terra._convert_all_imgs_to_terra(
            img_folder,
            metadata_folder,
            occupancy_folder,
            dumpability_folder,
            dst_root,
            size_tuple,
            args.n_imgs,
            expansion_factor=1,
            all_dumpable=False,
            copy_metadata=True,
            downsample=False,
            has_dumpability=True,
            center_padding=False,
        )
        print("\u2713 Terra format conversion complete")
        print(f"Note: Converted data saved to: train/trenches/{out_subdir}")
    else:
        print("Skipping Terra conversion per --no-convert")

    print("Done.")


if __name__ == "__main__":
    main()
