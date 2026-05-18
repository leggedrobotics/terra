#!/usr/bin/env python
import os
import argparse
from pathlib import Path

import numpy as np

from terra.env_generation import convert_to_terra
from terra.env_generation.procedural_data import (
    initialize_image,
    generate_edges,
    add_obstacles,
    add_non_dumpables,
    save_or_display_image,
)
from terra.env_generation.utils import color_dict


# Output subdirectory name under: data/terra/trenches/<name_string>
name_string = "single_masked_ring"


def _uneven_circle_mask_around_trench_from_img(
    img: np.ndarray,
    trench_color: tuple[int, int, int],
    radius_min: int,
    radius_max: int,
    border_roughness: int,
    n_sectors: int,
    ring_thickness_tiles: int = 2,
    safety_margin_tiles: int = 5,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Compute a coarse uneven circular ring directly from the trench pixels in the image.
    Returns boolean masks for the obstacle ring, the area outside the ring,
    and the circle center (cy, cx).
    """
    # Extract trench pixels from image
    trench_mask = np.all(img == trench_color, axis=2)
    rows, cols = trench_mask.shape
    ys, xs = np.nonzero(trench_mask)
    if ys.size == 0:
        empty = np.zeros((rows, cols), dtype=bool)
        return empty, empty, (rows / 2, cols / 2)

    # Circle center = centroid of trench pixels
    cy = float(ys.mean())
    cx = float(xs.mean())

    rr = np.arange(rows, dtype=np.float32)[:, None]
    cc = np.arange(cols, dtype=np.float32)[None, :]

    dist = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2).astype(np.float32)

    # Polar angle
    ang = np.arctan2(rr - cy, cc - cx).astype(np.float32)
    ang = (ang + np.pi) % (2.0 * np.pi)

    # Base radius ensures trench fully covered
    r0 = np.random.randint(radius_min, radius_max + 1)
    max_trench_r = float(dist[trench_mask].max())
    r0 = max(r0, max_trench_r + float(safety_margin_tiles))

    # Coarse uneven boundary
    n_sectors = int(max(8, n_sectors))
    offsets = np.random.randint(
        -int(border_roughness),
        int(border_roughness) + 1,
        size=(n_sectors,),
        dtype=np.int32,
    ).astype(np.float32)
    sector_idx = np.floor(ang / (2.0 * np.pi) * n_sectors).astype(np.int32)
    sector_idx = np.clip(sector_idx, 0, n_sectors - 1)
    r_eff = r0 + offsets[sector_idx]

    ring_thickness_tiles = max(1, int(ring_thickness_tiles))
    inner_r = np.maximum(r_eff - float(ring_thickness_tiles), 0.0)
    ring = np.logical_and(dist <= r_eff, dist >= inner_r)
    ring = np.logical_and(ring, np.logical_not(trench_mask))
    outside = dist > r_eff
    outside = np.logical_and(outside, np.logical_not(trench_mask))
    return ring, outside, (cy, cx)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate procedural single-trench maps with a masked obstacle circle "
            "around the trench. Produces: data/terra/trenches/single_masked and "
            "converts it to: data/terra/train/trenches/single_masked."
        )
    )
    parser.add_argument("--n-imgs", type=int, default=600, help="Number of maps to generate")
    parser.add_argument("--size", type=int, default=64, help="Square size for Terra conversion")
    parser.add_argument("--name", type=str, default=None, help="Optional override for output folder name")
    parser.add_argument("--no-generate", action="store_true", help="Skip raw map generation")
    parser.add_argument("--no-convert", action="store_true", help="Skip Terra conversion")
    parser.add_argument("--radius-min", type=int, default=35)
    parser.add_argument("--radius-max", type=int, default=40)
    parser.add_argument("--border-roughness", type=int, default=1)
    parser.add_argument("--n-sectors", type=int, default=6)
    parser.add_argument("--ring-thickness", type=int, default=2)
    parser.add_argument(
        "--disable-circle-mask",
        action="store_true",
        help="Disable the round obstacle mask around the trench.",
    )

    args = parser.parse_args()

    global name_string
    if args.name:
        name_string = args.name

    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(package_dir, "data", "terra", "trenches", name_string), exist_ok=True)
    save_folder = os.path.join(package_dir, "data", "terra", "trenches", name_string)

    # Parameters
    img_edge_min = img_edge_max = 64
    sizes_small = (max(1, int(0.04 * img_edge_min)), max(1, int(0.05 * img_edge_min)))
    sizes_long = (max(1, int(0.2 * img_edge_min)), max(1, int(0.3 * img_edge_min)))
    n_edges = (1, 1)
    n_obs_min, n_obs_max = 0, 2
    size_obstacle_min, size_obstacle_max = 3, 7
    n_nodump_min, n_nodump_max = 1, 2
    size_nodump_min, size_nodump_max = 6, 8

    # Some data paths use "trench" but the procedural generator writes trenches as "digging".
    # Fall back to digging color for compatibility.
    trench_color = color_dict.get("trench", color_dict["digging"])

    j = 0  # Counter for successfully generated maps
    i = 0  # Attempt counter
    while j < args.n_imgs:
        img = initialize_image(img_edge_min, img_edge_max, color_dict["dumping"])
        img, cumulative_mask, metadata = generate_edges(img, n_edges, sizes_small, sizes_long, color_dict)
        if img is None:
            i += 1
            continue
        occ, _ = add_obstacles(img, cumulative_mask, n_obs_min, n_obs_max, size_obstacle_min, size_obstacle_max)
        dmp, _ = add_non_dumpables(img, occ, cumulative_mask, n_nodump_min, n_nodump_max, size_nodump_min, size_nodump_max)

        if not args.disable_circle_mask:
            # --- Compute hollow obstacle ring from actual image trench pixels ---
            circle_ring, outside_ring, _ = _uneven_circle_mask_around_trench_from_img(
                img,
                trench_color,
                radius_min=args.radius_min,
                radius_max=args.radius_max,
                border_roughness=args.border_roughness,
                n_sectors=args.n_sectors,
                ring_thickness_tiles=args.ring_thickness,
                safety_margin_tiles=3,
            )
            occ[circle_ring] = np.array(color_dict["obstacle"])
            dmp[outside_ring] = np.array(color_dict["nondumpable"])

        save_or_display_image(img, occ, dmp, metadata, save_folder, j)
        j += 1
        i += 1

    # Convert to Terra format
    if not args.no_convert:
        npy_dataset_folder = os.path.join(package_dir, "data", "terra")
        dst_root = Path(npy_dataset_folder) / "train" / "trenches" / name_string
        dst_root.mkdir(parents=True, exist_ok=True)
        convert_to_terra._convert_all_imgs_to_terra(
            Path(npy_dataset_folder) / "trenches" / name_string / "images",
            Path(npy_dataset_folder) / "trenches" / name_string / "metadata",
            Path(npy_dataset_folder) / "trenches" / name_string / "occupancy",
            Path(npy_dataset_folder) / "trenches" / name_string / "dumpability",
            dst_root,
            (args.size, args.size),
            j,  # Use actual number of generated maps
            expansion_factor=1,
            all_dumpable=False,
            copy_metadata=True,
            downsample=False,
            has_dumpability=True,
            center_padding=False,
            actions_folder=None,
        )
        print(f"✓ Converted data saved to: train/trenches/{name_string}")
    else:
        print("Skipping Terra conversion per --no-convert")


if __name__ == "__main__":
    main()
