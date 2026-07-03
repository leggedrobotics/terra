#!/usr/bin/env python
"""
Variant of `generate_foundations_rectangles_real_ring.py` that keeps the simple
axis-aligned rectangle foundations but replaces the surrounding dump-zone ring
with one rectangular dump zone placed near a randomly chosen map border, using
the same placement style as `generate_foundations_dumpzones_v3.py`.

For each map:
  1. A simple rectangle foundation is sampled with different side lengths,
     placed at least `foundation_border_offset` pixels from the map border.
  2. A rectangular dump zone is sampled with independent width/height.
  3. The dump zone is placed near one map border with a random inset from
     `dump_border_offset` up to `dump_border_offset_max`.
  4. Candidates overlapping the foundation or small obstacles are rejected.

Everything else (small obstacles, output folder conventions) follows
`generate_foundations_rectangles_real_ring.py`. Full-span obstacles are disabled
for now.
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

import terra.env_generation.convert_to_terra as convert_to_terra
from terra.env_generation.generate_foundations_real import (
    # add_full_span_obstacles_and_nondump,
    add_small_rectangle_obstacles,
)
from terra.env_generation.procedural_data import (
    convert_terra_pad_to_color,
    save_or_display_image,
)
from terra.env_generation.utils import color_dict


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
name_string = "foundations_rectangles_real_dumpzone"


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


def _sample_border_inset(border_offset_min, border_offset_max, max_allowed):
    """Sample how far a dump zone sits inward from the chosen map border."""
    hi = min(border_offset_max, max_allowed)
    lo = border_offset_min
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def create_rectangular_dump_zone_at_border(
    img_terra_pad,
    allowed_mask,
    dig_mask,
    dump_width_min=14,
    dump_width_max=23,
    dump_height_min=14,
    dump_height_max=23,
    border_offset_min=3,
    border_offset_max=16,
    max_attempts=200,
):
    """
    Place one rectangular dump zone near a random map border.

    This follows the dump-zone placement style from
    `generate_foundations_dumpzones_v3.py`: choose one of the four borders,
    place the zone with a random inset from the border (between
    `border_offset_min` and `border_offset_max`), and reject overlaps. Width
    and height are sampled independently so the zone may be rectangular.

    Args:
        img_terra_pad: HxWx3 color map (mutated: dumping color is written).
        allowed_mask: HxW bool where dumping is permitted.
        dig_mask: HxW bool of foundation / dig cells.
        border_offset_min: Minimum pixels between dump zone and map edge.
        border_offset_max: Maximum inset from the chosen border (random per map).
    Returns:
        (img_terra_pad, dump_mask, dump_metadata). Raises if no dumpable
        placement is found.
    """
    border_offset_max = max(border_offset_min, border_offset_max)
    if not np.any(dig_mask):
        raise RuntimeError("No foundation cells found; cannot place a dump zone.")

    h, w = dig_mask.shape
    foundation_area = int(np.sum(dig_mask))
    sides = ("top", "bottom", "left", "right")

    placement_passes = (
        ("primary", "border", dump_width_min, dump_width_max, dump_height_min, dump_height_max),
        (
            "position_retry_1",
            "anywhere",
            dump_width_min,
            dump_width_max,
            dump_height_min,
            dump_height_max,
        ),
        (
            "position_retry_2",
            "anywhere",
            dump_width_min,
            dump_width_max,
            dump_height_min,
            dump_height_max,
        ),
        (
            "fallback_size_minus_3",
            "anywhere",
            11,
            15,
            11,
            15,
        ),
        (
            "fallback_size_8_10",
            "anywhere",
            9,
            12,
            9,
            12,
        ),
    )
    attempted_ranges = []

    for pass_name, placement_mode, width_min, width_max, height_min, height_max in placement_passes:
        width_max = max(width_min, width_max)
        height_max = max(height_min, height_max)
        attempted_ranges.append(
            f"{pass_name} ({placement_mode}): width {width_min}-{width_max}, "
            f"height {height_min}-{height_max}"
        )

        for _ in range(max_attempts):
            dump_width = random.randint(width_min, width_max)
            dump_height = random.randint(height_min, height_max)
            # Temporarily disabled: allow larger dump zones, including zones whose
            # area is equal to or larger than the foundation area.
            # if dump_width * dump_height >= foundation_area:
            #     continue
            if (
                dump_width > w - 2 * border_offset_min
                or dump_height > h - 2 * border_offset_min
            ):
                continue

            map_margin = border_offset_min

            if placement_mode == "anywhere":
                side = "anywhere"
                x = random.randint(map_margin, w - dump_width - map_margin)
                y = random.randint(map_margin, h - dump_height - map_margin)
                border_inset = None
            else:
                side = random.choice(sides)
                if side == "top":
                    max_inset = h - dump_height - map_margin
                    if max_inset < border_offset_min:
                        continue
                    border_inset = _sample_border_inset(
                        border_offset_min, border_offset_max, max_inset
                    )
                    x = random.randint(map_margin, w - dump_width - map_margin)
                    y = border_inset
                elif side == "bottom":
                    max_inset = h - dump_height - map_margin
                    if max_inset < border_offset_min:
                        continue
                    border_inset = _sample_border_inset(
                        border_offset_min, border_offset_max, max_inset
                    )
                    x = random.randint(map_margin, w - dump_width - map_margin)
                    y = h - dump_height - border_inset
                elif side == "left":
                    max_inset = w - dump_width - map_margin
                    if max_inset < border_offset_min:
                        continue
                    border_inset = _sample_border_inset(
                        border_offset_min, border_offset_max, max_inset
                    )
                    x = border_inset
                    y = random.randint(map_margin, h - dump_height - map_margin)
                else:
                    max_inset = w - dump_width - map_margin
                    if max_inset < border_offset_min:
                        continue
                    border_inset = _sample_border_inset(
                        border_offset_min, border_offset_max, max_inset
                    )
                    x = w - dump_width - border_inset
                    y = random.randint(map_margin, h - dump_height - map_margin)

            if not np.all(allowed_mask[y : y + dump_height, x : x + dump_width]):
                continue

            dump_mask = np.zeros((h, w), dtype=bool)
            dump_mask[y : y + dump_height, x : x + dump_width] = True
            img_terra_pad[dump_mask] = color_dict["dumping"]
            dump_metadata = {
                "rectangular_dump_zone": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(dump_width),
                    "height": int(dump_height),
                    "side": side,
                    "placement_mode": placement_mode,
                    "border_offset_min": int(border_offset_min),
                    "border_offset_max": int(border_offset_max),
                    "border_inset": (
                        None if border_inset is None else int(border_inset)
                    ),
                    "placement_pass": pass_name,
                    "area": int(dump_width * dump_height),
                    "foundation_area": foundation_area,
                }
            }
            if pass_name != "primary":
                print(
                    f"{pass_name}: placed rectangular dump zone "
                    f"{dump_width}x{dump_height} at ({x}, {y})"
                )
            return img_terra_pad, dump_mask, dump_metadata

    raise RuntimeError(
        "Failed to place a rectangular border dump zone after "
        f"{max_attempts} attempts for each placement pass: {attempted_ranges}. "
        f"Map size={w}x{h}, border_offset_min={border_offset_min}, "
        f"border_offset_max={border_offset_max}, "
        f"foundation_area={foundation_area}. Try smaller border offsets, "
        "fewer obstacles, or larger foundation rectangles."
    )


def _sample_rectangle_side_lengths(min_side, max_side):
    width = random.randint(min_side, max_side)
    height = random.randint(min_side, max_side)
    if min_side < max_side:
        while height == width:
            height = random.randint(min_side, max_side)
    return width, height


def _create_rectangular_foundation_terra_pad(
    max_size,
    rectangle_min_side,
    rectangle_max_side,
    expansion_factor=1,
    placement_margin=6,
):
    placement_margin = max(1, int(placement_margin))
    max_side_that_fits = max_size - 2 * placement_margin
    rectangle_max_side = min(rectangle_max_side, max_side_that_fits)
    if rectangle_max_side < rectangle_min_side:
        raise ValueError(
            "Rectangle side limits leave no room for placement. "
            f"min={rectangle_min_side}, max={rectangle_max_side}, "
            f"map={max_size}, placement_margin={placement_margin}"
        )

    width, height = _sample_rectangle_side_lengths(
        rectangle_min_side,
        rectangle_max_side,
    )
    x0 = random.randint(placement_margin, max_size - placement_margin - width)
    y0 = random.randint(placement_margin, max_size - placement_margin - height)

    img_terra_pad = np.zeros((max_size, max_size), dtype=np.int8)
    img_terra_pad[y0 : y0 + height, x0 : x0 + width] = -1
    img_terra_pad = img_terra_pad.repeat(expansion_factor, 0).repeat(expansion_factor, 1)
    metadata = {
        "rectangle_foundation": {
            "x": x0 * expansion_factor,
            "y": y0 * expansion_factor,
            "width": width * expansion_factor,
            "height": height * expansion_factor,
            "border_offset": placement_margin * expansion_factor,
        }
    }
    return convert_terra_pad_to_color(img_terra_pad, color_dict), metadata


def create_foundations_rectangles_dumpzone_standalone(
    n_imgs=600,
    max_size=64,
    n_obs_min=1,
    n_obs_max=1,
    obstacle_width=2,
    n_small_obs_min=0,
    n_small_obs_max=2,
    size_small_obstacle_min=3,
    size_small_obstacle_max=6,
    expansion_factor=1,
    all_dumpable=False,
    copy_metadata=True,
    has_dumpability=False,
    center_padding=True,
    dump_width_min=14,
    dump_width_max=20,
    dump_height_min=14,
    dump_height_max=20,
    dump_border_offset=3,
    dump_border_offset_max=16,
    foundation_border_offset=8,
    rectangle_min_side=12,
    rectangle_max_side=24,
):
    save_folder = os.path.join(PACKAGE_DIR, "data", "terra", name_string)
    _ = (all_dumpable, copy_metadata, has_dumpability, center_padding)
    print(
        f"Creating rectangular foundations with rectangular border dump zones "
        f"and small obstacles "
        f"- saving to: {name_string}/"
    )

    downsampling_factors = {
        save_folder: 2.0,
    }

    for curriculum_level, _downsampling_factor in downsampling_factors.items():
        for i in range(n_imgs):
            print(f"Processing foundation nr {i + 1}")
            img_terra_pad, metadata = _create_rectangular_foundation_terra_pad(
                max_size=max_size,
                rectangle_min_side=rectangle_min_side,
                rectangle_max_side=rectangle_max_side,
                expansion_factor=expansion_factor,
                placement_margin=foundation_border_offset,
            )

            # Start from a neutral background so only the border rectangle
            # becomes a positive dump target. Keep the foundation cells as dig.
            neutral_mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)
            img_terra_pad[neutral_mask] = color_dict["neutral"]
            dig_mask = np.all(img_terra_pad == color_dict["digging"], axis=-1)

            # Full-span obstacles are disabled for the simple-rectangle curriculum
            # for now. Keep empty layers so the downstream writer still emits the
            # expected occupancy and dumpability images.
            occ = np.ones_like(img_terra_pad, dtype=np.uint8) * 255
            dmp = np.ones_like(img_terra_pad, dtype=np.uint8) * 255
            obstacle_mask = np.zeros(dig_mask.shape, dtype=bool)
            nondump_mask = np.zeros(dig_mask.shape, dtype=bool)
            # occ, dmp, obstacle_mask, nondump_mask, _foundation_side_mask = (
            #     add_full_span_obstacles_and_nondump(
            #         img_terra_pad,
            #         dig_mask,
            #         n_obs_min=n_obs_min,
            #         n_obs_max=n_obs_max,
            #         obstacle_width=obstacle_width,
            #     )
            # )

            allowed_dump_mask = ~dig_mask & ~obstacle_mask & ~nondump_mask

            img_terra_pad, dump_mask, dump_metadata = create_rectangular_dump_zone_at_border(
                img_terra_pad,
                allowed_dump_mask,
                dig_mask,
                dump_width_min=dump_width_min,
                dump_width_max=dump_width_max,
                dump_height_min=dump_height_min,
                dump_height_max=dump_height_max,
                border_offset_min=dump_border_offset,
                border_offset_max=dump_border_offset_max,
            )

            cumulative_mask = dig_mask | dump_mask | obstacle_mask | nondump_mask
            occ, cumulative_mask = add_small_rectangle_obstacles(
                img_terra_pad,
                occ,
                cumulative_mask,
                n_obs_min=n_small_obs_min,
                n_obs_max=n_small_obs_max,
                size_obstacle_min=size_small_obstacle_min,
                size_obstacle_max=size_small_obstacle_max,
            )

            metadata_out = dict(metadata)
            metadata_out.update(dump_metadata)
            metadata_out.update(_build_foundation_border_metadata(dig_mask))
            # Save with contiguous output ids starting at 0, independent of source ids.
            save_or_display_image(img_terra_pad, occ, dmp, metadata_out, curriculum_level, i)
            _save_image_with_polygon_overlay(img_terra_pad, metadata_out, curriculum_level, i)

    print("Rectangular foundations with border dump zones created successfully.")


def generate_foundations_rectangles_dumpzone_standalone(
    config_path="config/env_generation_config.yaml",
    generate_terra_format=True,
    dump_width_min=14,
    dump_width_max=20,
    dump_height_min=14,
    dump_height_max=20,
    dump_border_offset=3,
    dump_border_offset_max=16,
    foundation_border_offset=8,
    rectangle_min_side=None,
    rectangle_max_side=None,
):
    package_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    with open(os.path.join(package_dir, config_path), "r") as file:
        config = yaml.safe_load(file)

    os.makedirs("data/", exist_ok=True)
    os.makedirs("data/terra/", exist_ok=True)

    n_imgs = config["n_imgs"]
    foundations_config = config.get("foundations", {})
    if "min_size" not in foundations_config or "max_size" not in foundations_config:
        raise ValueError("min_size and max_size must be provided in the config file")

    map_size = foundations_config.get("max_size", 64)
    rectangle_min_side = rectangle_min_side or foundations_config.get(
        "rectangle_min_side",
        foundations_config.get("min_size"),
    )
    rectangle_max_side = rectangle_max_side or foundations_config.get(
        "rectangle_max_side",
        24,
    )

    print("Generating FOUNDATIONS RECTANGLES REAL DUMPZONE maps...")
    print(
        f"Rectangle foundation config - min_side: {rectangle_min_side}, "
        f"max_side: {rectangle_max_side}, map_size: {map_size}, "
        f"dump_size: {dump_width_min}-{dump_width_max} x "
        f"{dump_height_min}-{dump_height_max}, "
        f"dump_border_offset: {dump_border_offset}-{dump_border_offset_max}, "
        f"foundation_border_offset: {foundation_border_offset}"
    )

    create_foundations_rectangles_dumpzone_standalone(
        n_imgs=n_imgs,
        max_size=map_size,
        dump_width_min=dump_width_min,
        dump_width_max=dump_width_max,
        dump_height_min=dump_height_min,
        dump_height_max=dump_height_max,
        dump_border_offset=dump_border_offset,
        dump_border_offset_max=dump_border_offset_max,
        foundation_border_offset=foundation_border_offset,
        rectangle_min_side=rectangle_min_side,
        rectangle_max_side=rectangle_max_side,
    )
    print(
        f"  OK Rectangular foundations dump-zone maps saved to: "
        f"data/terra/{name_string}"
    )

    if generate_terra_format:
        print(
            "Converting rectangular foundations border dump-zone data to Terra format..."
        )
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
            print("  OK Terra format conversion complete")
        else:
            print(f"  Skipping conversion; folder not found: {foundations_dir}")

    print("Rectangular foundations border dump-zone generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate simple rectangular foundation maps with a rectangular "
            "border dump zone and reduced small obstacles."
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
        "--dump-width-min",
        type=int,
        default=14,
        help="Minimum dump-zone rectangle width.",
    )
    parser.add_argument(
        "--dump-width-max",
        type=int,
        default=20,
        help="Maximum dump-zone rectangle width.",
    )
    parser.add_argument(
        "--dump-height-min",
        type=int,
        default=14,
        help="Minimum dump-zone rectangle height.",
    )
    parser.add_argument(
        "--dump-height-max",
        type=int,
        default=20,
        help="Maximum dump-zone rectangle height.",
    )
    parser.add_argument(
        "--dump-border-offset",
        type=int,
        default=3,
        help="Minimum distance between the dump zone and the map border.",
    )
    parser.add_argument(
        "--dump-border-offset-max",
        type=int,
        default=16,
        help=(
            "Maximum random inset from the chosen border when placing the dump "
            "zone (>= --dump-border-offset)."
        ),
    )
    parser.add_argument(
        "--foundation-border-offset",
        type=int,
        default=8,
        help="Minimum distance between the foundation rectangle and the map border.",
    )
    parser.add_argument(
        "--rectangle-min-side",
        type=int,
        default=None,
        help="Minimum rectangle foundation side length. Defaults to foundations.min_size.",
    )
    parser.add_argument(
        "--rectangle-max-side",
        type=int,
        default=None,
        help="Maximum rectangle foundation side length. Defaults to foundations.max_size.",
    )

    args = parser.parse_args()

    generate_terra_format = not args.no_terra_format

    generate_foundations_rectangles_dumpzone_standalone(
        args.config,
        generate_terra_format=generate_terra_format,
        dump_width_min=args.dump_width_min,
        dump_width_max=args.dump_width_max,
        dump_height_min=args.dump_height_min,
        dump_height_max=args.dump_height_max,
        dump_border_offset=args.dump_border_offset,
        dump_border_offset_max=max(
            args.dump_border_offset, args.dump_border_offset_max
        ),
        foundation_border_offset=args.foundation_border_offset,
        rectangle_min_side=args.rectangle_min_side,
        rectangle_max_side=args.rectangle_max_side,
    )
