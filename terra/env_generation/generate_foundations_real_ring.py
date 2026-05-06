#!/usr/bin/env python
"""
Variant of `generate_foundations_real.py` that places the dump zone as a
rectangular ring around the foundation (simplified extruded shape), rather
than as a full-map dumpable region or a single square on one side.

For each map:
  1. The foundation's axis-aligned bounding box is computed.
  2. The box is expanded by `ring_margin` cells in each direction (clamped to
     the map) to form an outer rectangle.
  3. The ring = (outer rect) \\ (inner foundation bbox).
  4. Cells of the ring that overlap the foundation, obstacles, or the
     non-dumpable band behind a full-span obstacle are removed.
  5. The remaining ring cells become the dump zone.

Everything else (full-span obstacles, non-dumpable band, small obstacles,
output folder conventions) is identical to `generate_foundations_real.py`.
"""
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
from terra.env_generation.generate_foundations_real import (
    add_full_span_obstacles_and_nondump,
    add_small_rectangle_obstacles,
)
from terra.env_generation.procedural_data import (
    convert_terra_pad_to_color,
    save_or_display_image,
)
from terra.env_generation.utils import color_dict


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
name_string = "foundations_real_ring"


def _to_abc_from_points(p1, p2):
    """Return line coefficients A,B,C for Ax + By + C = 0 from two points."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return {"A": float(a), "B": float(b), "C": float(c)}


def _point_to_segment_distance(point, start, end):
    point = np.asarray(point, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-6:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _get_border_pixels(dig_mask: np.ndarray):
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(dig_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    border = np.logical_and(dig_mask, np.logical_not(eroded))
    rows, cols = np.where(border)
    return np.stack([cols, rows], axis=1).astype(np.float32)


def _polygon_covers_border(pts: np.ndarray, border_pixels: np.ndarray, max_dist: float):
    if len(pts) < 3 or len(border_pixels) == 0:
        return False
    max_seen_dist = 0.0
    for pixel in border_pixels:
        nearest = min(
            _point_to_segment_distance(pixel, pts[i], pts[(i + 1) % len(pts)])
            for i in range(len(pts))
        )
        max_seen_dist = max(max_seen_dist, nearest)
        if max_seen_dist > max_dist:
            return False
    return True


def _approx_foundation_border_polygon(dig_mask: np.ndarray):
    """
    Approximate the rasterized foundation border with the fewest useful segments.
    """
    mask_u8 = (dig_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    border_pixels = _get_border_pixels(dig_mask)
    approx = None
    max_border_error_px = 1.8
    for eps_fraction in (0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01):
        candidate = cv2.approxPolyDP(contour, eps_fraction * peri, True).reshape(-1, 2)
        if _polygon_covers_border(candidate, border_pixels, max_border_error_px):
            approx = candidate
            break
    if approx is None:
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True).reshape(-1, 2)
    return approx.reshape(-1, 2).astype(np.float32)


def _build_foundation_border_axes_from_mask(dig_mask: np.ndarray):
    """
    Build border axes from the rasterized foundation mask.
    Uses contour extraction and adaptive polygon approximation to get the fewest
    edge segments that still cover the raster border.
    """
    pts = _approx_foundation_border_polygon(dig_mask)
    if pts.shape[0] < 2:
        return []
    axes = []
    for i in range(pts.shape[0]):
        p1 = pts[i]
        p2 = pts[(i + 1) % pts.shape[0]]
        if np.all(p1 == p2):
            continue
        axes.append(_to_abc_from_points(p1, p2))
    return axes


def _build_foundation_border_lines_from_mask(dig_mask: np.ndarray):
    pts = _approx_foundation_border_polygon(dig_mask)
    if pts.shape[0] < 2:
        return []
    lines = []
    for i in range(pts.shape[0]):
        p1 = pts[i]
        p2 = pts[(i + 1) % pts.shape[0]]
        if np.all(p1 == p2):
            continue
        lines.append(
            [
                [float(p1[0]), float(p1[1])],
                [float(p2[0]), float(p2[1])],
            ]
        )
    return lines


def _build_foundation_border_metadata(dig_mask: np.ndarray):
    """
    Build border-axis metadata for the final Terra foundation mask.

    Source metadata may be in the pre-downsampled/pre-padded image frame. Rebuild
    from the final raster mask so the generated axes match the playable map.
    """
    return {
        "foundation_border_axes_ABC": _build_foundation_border_axes_from_mask(dig_mask),
        "foundation_border_lines_pts": _build_foundation_border_lines_from_mask(dig_mask),
    }


def _save_image_with_polygon_overlay(img, metadata, save_folder, i):
    overlay_folder = Path(save_folder) / "images_with_polygon"
    overlay_folder.mkdir(parents=True, exist_ok=True)

    overlay = img.copy()
    colors = [
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (0, 128, 255),
        (128, 0, 255),
        (0, 255, 128),
        (255, 0, 128),
    ]

    lines = metadata.get("foundation_border_lines_pts", [])
    for line_idx, line in enumerate(lines):
        p1 = tuple(np.rint(line[0]).astype(int))
        p2 = tuple(np.rint(line[1]).astype(int))
        color = colors[line_idx % len(colors)]
        cv2.line(overlay, p1, p2, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(overlay_folder / f"trench_{i}.png"), overlay)


def create_ring_dump_zone_around_foundation(
    img_terra_pad,
    allowed_mask,
    dig_mask,
    ring_margin=8,
):
    """
    Build a ring around the foundation whose inner border is flush with the
    foundation outline (no neutral gap), and whose outer border is the
    Chebyshev-dilated foundation shape by `ring_margin` cells. The ring is
    then intersected with `allowed_mask` (not foundation, not obstacle,
    not non-dumpable) so obstacles and the non-dumpable band carve it up.

    Args:
        img_terra_pad: HxWx3 color map (mutated: dumping color is written).
        allowed_mask: HxW bool where dumping is permitted.
        dig_mask: HxW bool of foundation / dig cells.
        ring_margin: Chebyshev distance to extrude the foundation by.
    Returns:
        (img_terra_pad, dump_mask). Raises if no dumpable cells remain.
    """
    if not np.any(dig_mask):
        raise RuntimeError("No foundation cells found; cannot build a ring dump zone.")

    # Outer border: rectangular bounding box of the foundation, extruded by
    # `ring_margin` cells on every side and clamped to the map.
    h, w = dig_mask.shape
    rows = np.any(dig_mask, axis=1)
    cols = np.any(dig_mask, axis=0)
    y0, y1 = int(np.argmax(rows)), int(h - 1 - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(w - 1 - np.argmax(cols[::-1]))
    yo0 = max(0, y0 - ring_margin)
    yo1 = min(h - 1, y1 + ring_margin)
    xo0 = max(0, x0 - ring_margin)
    xo1 = min(w - 1, x1 + ring_margin)
    outer = np.zeros((h, w), dtype=bool)
    outer[yo0 : yo1 + 1, xo0 : xo1 + 1] = True

    # Inner border: flush with the actual foundation outline (no neutral gap
    # in concave corners). So the ring = outer rectangle minus the foundation.
    ring = outer & ~dig_mask
    dump_mask = ring & allowed_mask

    if not np.any(dump_mask):
        raise RuntimeError(
            "Ring around foundation contains no dumpable cells "
            "(blocked by obstacles / nondumpable band / map edges)."
        )

    img_terra_pad[dump_mask] = color_dict["dumping"]
    return img_terra_pad, dump_mask


def create_foundations_ring_standalone(
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
    ring_margin=8,
):
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", name_string)
    print(
        f"Creating foundations with ring dump zones and full-span obstacles "
        f"- saving to: {name_string}/"
    )

    downsampling_factors = {
        save_folder: 2.0,
    }

    full_dataset_path = os.path.join(PACKAGE_DIR, dataset_path)
    foundations_name = "foundations"
    img_folder = Path(full_dataset_path) / foundations_name / "images"
    metadata_folder = Path(full_dataset_path) / foundations_name / "metadata"
    occupancy_folder = Path(full_dataset_path) / foundations_name / "occupancy"
    dumpability_folder = Path(full_dataset_path) / foundations_name / "dumpability"
    image_filenames = sorted(
        [f for f in os.listdir(img_folder) if f.endswith(".png")]
    )

    for curriculum_level, downsampling_factor in downsampling_factors.items():
        for i, fn in enumerate(image_filenames):
            if i >= n_imgs:
                break

            print(f"Processing foundation nr {i + 1}")
            filename = fn
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

            # Start from a neutral (non-dump) background so only the ring ends
            # up as dumpable. Keep the foundation cells as dig.
            neutral_mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)
            img_terra_pad[neutral_mask] = color_dict["neutral"]
            dig_mask = np.all(img_terra_pad == color_dict["digging"], axis=-1)

            occ, dmp, obstacle_mask, nondump_mask, _foundation_side_mask = (
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

            allowed_dump_mask = ~dig_mask & ~obstacle_mask & ~nondump_mask

            img_terra_pad, _ = create_ring_dump_zone_around_foundation(
                img_terra_pad,
                allowed_dump_mask,
                dig_mask,
                ring_margin=ring_margin,
            )

            metadata_out = dict(metadata)
            metadata_out.update(_build_foundation_border_metadata(dig_mask))
            # Save with contiguous output ids starting at 0, independent of source ids.
            save_or_display_image(img_terra_pad, occ, dmp, metadata_out, curriculum_level, i)
            _save_image_with_polygon_overlay(img_terra_pad, metadata_out, curriculum_level, i)

    print("Foundations with ring dump zones created successfully.")


def generate_foundations_ring_standalone(
    config_path="config/env_generation_config.yaml",
    generate_terra_format=True,
    ring_margin=8,
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

    print("Generating FOUNDATIONS REAL RING STANDALONE maps...")
    print(
        f"Foundation config - min_size: {foundation_min_size}, "
        f"max_size: {foundation_max_size}, max_buildings: {max_buildings}, "
        f"ring_margin: {ring_margin}"
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
        foundation_count = len([f for f in os.listdir(foundation_path) if f.endswith(".png")])
        print(f"Found {foundation_count} existing foundation images")

    create_foundations_ring_standalone(
        n_imgs=n_imgs,
        max_size=foundations_config.get("max_size", 64),
        ring_margin=ring_margin,
    )
    print(f"  ✓ Foundations real ring maps saved to: data/terra/{name_string}")

    if generate_terra_format:
        print("Converting foundations real ring standalone data to Terra format...")
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

    print("Foundations real ring generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate foundation maps with a rectangular dump-zone ring around "
            "the foundation (simplified extruded shape), full-span 2-wide "
            "obstacles, and no-dump regions on the side without foundations."
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
        "--ring-margin",
        type=int,
        default=8,
        help="How many cells to extrude the foundation bbox by (ring thickness).",
    )

    args = parser.parse_args()

    generate_terra_format = not args.no_terra_format

    generate_foundations_ring_standalone(
        args.config,
        generate_terra_format=generate_terra_format,
        ring_margin=args.ring_margin,
    )
