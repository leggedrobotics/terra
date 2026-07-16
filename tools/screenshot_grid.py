#!/usr/bin/env python3
"""Create a marked grid view from a directory of screenshot frames."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "This script needs Pillow. Install it with `pip install pillow` or run it "
        "inside the repo's `terra` conda environment."
    ) from exc


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def parse_color(value: str) -> tuple[int, int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) not in (6, 8):
        raise ValueError("Colors must be hex RGB or RGBA, e.g. ffffff or ffffffff")
    channels = [int(value[i : i + 2], 16) for i in range(0, len(value), 2)]
    if len(channels) == 3:
        channels.append(255)
    return tuple(channels)  # type: ignore[return-value]


def parse_crop(crop: str | None) -> tuple[int, int, int, int] | None:
    if crop is None:
        return None
    values = [int(value.strip()) for value in crop.split(",")]
    if len(values) != 4:
        raise ValueError("--crop must have four comma-separated integers: x,y,w,h")
    x, y, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("--crop width and height must be positive")
    return x, y, x + width, y + height


def parse_grid(grid: str | None) -> tuple[int, int] | None:
    if grid is None:
        return None
    normalized = grid.lower().replace(" ", "")
    if "x" not in normalized:
        raise ValueError("--grid must be formatted as columnsxrows, e.g. 4x2")
    columns_s, rows_s = normalized.split("x", 1)
    columns = int(columns_s)
    rows = int(rows_s)
    if columns <= 0 or rows <= 0:
        raise ValueError("--grid columns and rows must be positive")
    return columns, rows


def parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise ValueError("--indices must contain at least one 1-based index")
    if any(index <= 0 for index in indices):
        raise ValueError("--indices values must be positive 1-based integers")
    return indices


def find_images(input_dir: Path, pattern: str | None) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    if pattern is None:
        paths = [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    else:
        paths = [
            path
            for path in input_dir.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    return sorted(paths, key=natural_key)


def load_images(
    paths: Iterable[Path],
    *,
    crop_box: tuple[int, int, int, int] | None,
    scale: float,
) -> list[tuple[int, Image.Image]]:
    if scale <= 0:
        raise ValueError("--scale must be greater than zero")

    frames: list[tuple[int, Image.Image]] = []
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as source:
            image = source.convert("RGBA")
            if crop_box is not None:
                image = image.crop(crop_box)
            if scale != 1.0:
                new_size = (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            frames.append((index, image.copy()))
    return frames


def normalize_frame_sizes(
    indexed_frames: Iterable[tuple[int, Image.Image]],
    background: tuple[int, int, int, int],
) -> list[tuple[int, Image.Image]]:
    indexed_frames = list(indexed_frames)
    if not indexed_frames:
        return []

    width = max(frame.width for _, frame in indexed_frames)
    height = max(frame.height for _, frame in indexed_frames)
    normalized: list[tuple[int, Image.Image]] = []
    for index, frame in indexed_frames:
        canvas = Image.new("RGBA", (width, height), background)
        x = (width - frame.width) // 2
        y = (height - frame.height) // 2
        canvas.alpha_composite(frame, (x, y))
        normalized.append((index, canvas))
    return normalized


def get_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def get_label_font(size: int | None) -> ImageFont.ImageFont:
    if size is None:
        return ImageFont.load_default()
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_waypoint_mark(
    image: Image.Image,
    *,
    text: str,
    xy: tuple[int, int],
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    text_fill: tuple[int, int, int, int],
    scale: float,
) -> None:
    draw = ImageDraw.Draw(image)
    radius = max(12, round(21 * scale))
    outline_width = max(2, round(3 * scale))
    x, y = xy
    box = (x, y, x + 2 * radius, y + 2 * radius)
    draw.ellipse(box, fill=fill, outline=outline, width=outline_width)

    font = get_font(max(10, round(18 * scale)))
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_x = x + radius - text_width / 2
    text_y = y + radius - text_height / 2 - text_box[1]
    draw.text((text_x, text_y), text, fill=text_fill, font=font)


def make_grid(
    indexed_frames: list[tuple[int, Image.Image]],
    *,
    columns: int | None,
    padding: int,
    margin: int,
    background: tuple[int, int, int, int],
    mark: bool,
    mark_prefix: str,
    mark_start: int,
    mark_step: int,
    mark_fill: tuple[int, int, int, int],
    mark_outline: tuple[int, int, int, int],
    mark_text: tuple[int, int, int, int],
    label: bool,
    label_template: str,
    label_position: str,
    label_shape: str,
    label_circle_scale: float,
    label_reference_size: int,
    label_base_font_size: float,
) -> Image.Image:
    if not indexed_frames:
        raise ValueError("No screenshots found")
    if padding < 0 or margin < 0:
        raise ValueError("--padding and --margin must be non-negative")

    indexed_frames = normalize_frame_sizes(indexed_frames, background)
    frame_width = indexed_frames[0][1].width
    frame_height = indexed_frames[0][1].height
    label_height = 18 if label and label_position == "below" else 0

    if columns is None:
        columns = math.ceil(math.sqrt(len(indexed_frames)))
    if columns <= 0:
        raise ValueError("--columns must be greater than zero")
    rows = math.ceil(len(indexed_frames) / columns)

    sheet_width = 2 * margin + columns * frame_width + (columns - 1) * padding
    sheet_height = (
        2 * margin + rows * (frame_height + label_height) + (rows - 1) * padding
    )
    sheet = Image.new("RGBA", (sheet_width, sheet_height), background)
    draw = ImageDraw.Draw(sheet)
    if label_position == "below":
        font = get_label_font(None)
    else:
        label_scale = min(frame_width, frame_height) / label_reference_size
        font = get_label_font(max(1, round(label_base_font_size * label_scale)))

    mark_scale = max(0.55, min(frame_width, frame_height) / 700)
    for output_index, (source_index, frame) in enumerate(indexed_frames):
        row = output_index // columns
        col = output_index % columns
        x = margin + col * (frame_width + padding)
        y = margin + row * (frame_height + label_height + padding)
        mark_value = mark_start + (source_index - 1) * mark_step
        sheet.alpha_composite(frame, (x, y))
        if mark:
            mark_text_value = f"{mark_prefix}{mark_value}"
            offset = max(10, round(14 * mark_scale))
            draw_waypoint_mark(
                sheet,
                text=mark_text_value,
                xy=(x + offset, y + offset),
                fill=mark_fill,
                outline=mark_outline,
                text_fill=mark_text,
                scale=mark_scale,
            )
        if label:
            label_text = label_template.format(
                index=output_index + 1,
                mark=mark_value,
                source=source_index,
            )
            if label_position == "below":
                draw.text(
                    (x + 3, y + frame_height + 3),
                    label_text,
                    fill=(20, 20, 20, 255),
                    font=font,
                )
            else:
                text_box = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_box[2] - text_box[0]
                text_height = text_box[3] - text_box[1]
                box_padding = max(3, round(text_height * 0.12))
                if label_shape == "circle":
                    box_size = round(
                        (max(text_width, text_height) + 2 * box_padding)
                        * label_circle_scale
                    )
                    if label_position == "top-right":
                        box_x = x + frame_width - box_size - 4
                    else:
                        box_x = x + 4
                    box_y = y + 4
                    draw.ellipse(
                        (box_x, box_y, box_x + box_size, box_y + box_size),
                        fill=(255, 255, 255, 230),
                    )
                    text_x = box_x + (box_size - text_width) / 2
                    text_y = box_y + (box_size - text_height) / 2 - text_box[1]
                else:
                    box_width = text_width + 2 * box_padding
                    box_height = text_height + 2 * box_padding
                    if label_position == "top-right":
                        box_x = x + frame_width - box_width - 4
                    else:
                        box_x = x + 4
                    box_y = y + 4
                    if label_shape == "rectangle":
                        draw.rectangle(
                            (box_x, box_y, box_x + box_width, box_y + box_height),
                            fill=(255, 255, 255, 230),
                        )
                    else:
                        draw.rounded_rectangle(
                            (box_x, box_y, box_x + box_width, box_y + box_height),
                            radius=max(3, round(text_height * 0.18)),
                            fill=(255, 255, 255, 230),
                        )
                    text_x = box_x + box_padding
                    text_y = box_y + box_padding - text_box[1]
                draw.text(
                    (text_x, text_y),
                    label_text,
                    fill=(20, 20, 20, 255),
                    font=font,
                )

    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a marked PNG grid from screenshot frames in a directory."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing screenshots")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional glob pattern within input_dir, e.g. 'Screenshot*.png'",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Set the output grid as columnsxrows, e.g. 4x2",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=None,
        help="Number of columns. Defaults to a near-square layout.",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop each screenshot before layout as x,y,w,h",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale each screenshot before layout (default: 1.0)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=12,
        help="Pixels between screenshots (default: 12)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=12,
        help="Outer margin in pixels (default: 12)",
    )
    parser.add_argument(
        "--background",
        default="ffffff",
        help="Hex background color, RGB/RGBA (default: ffffff)",
    )
    parser.add_argument(
        "--no-mark",
        action="store_true",
        help="Do not draw waypoint-style numbered marks",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Add GIF-contact-sheet-style text labels below each cell",
    )
    parser.add_argument(
        "--label-template",
        default="waypoint {mark}",
        help=(
            "Label text template. Available fields: {index}, {mark}, {source} "
            "(default: 'waypoint {mark}')."
        ),
    )
    parser.add_argument(
        "--label-position",
        choices=("top-left", "top-right", "below"),
        default="below",
        help="Where to draw --label text (default: below)",
    )
    parser.add_argument(
        "--label-shape",
        choices=("rectangle", "rounded", "circle"),
        default="rounded",
        help="Shape for overlaid label backgrounds (default: rounded)",
    )
    parser.add_argument(
        "--label-circle-scale",
        type=float,
        default=1.0,
        help="Scale circular label backgrounds without changing text (default: 1.0)",
    )
    parser.add_argument(
        "--label-reference-size",
        type=int,
        default=64,
        help=(
            "Reference frame size used to scale overlaid label text. The default "
            "matches the 64x64 GIF contact-sheet cells."
        ),
    )
    parser.add_argument(
        "--label-base-font-size",
        type=float,
        default=10,
        help="Font size at --label-reference-size for overlaid labels (default: 10)",
    )
    parser.add_argument(
        "--mark-prefix",
        default="",
        help="Text prefix for the numbered marks (default: none)",
    )
    parser.add_argument(
        "--mark-start",
        type=int,
        default=1,
        help="First mark number (default: 1)",
    )
    parser.add_argument(
        "--mark-step",
        type=int,
        default=1,
        help=(
            "Increment between mark numbers. Use 2 for dig-only frames from "
            "alternating dig/dump waypoints (default: 1)."
        ),
    )
    parser.add_argument(
        "--mark-fill",
        default="ffcc00",
        help="Hex fill color for numbered marks (default: ffcc00)",
    )
    parser.add_argument(
        "--mark-outline",
        default="141414",
        help="Hex outline color for numbered marks (default: 141414)",
    )
    parser.add_argument(
        "--mark-text",
        default="141414",
        help="Hex text color for numbered marks (default: 141414)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth screenshot after sorting (default: 1)",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help=(
            "Comma-separated 1-based screenshot indices after sorting, e.g. "
            "'1,3,5,6'. Overrides --stride."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mark_step <= 0:
        raise ValueError("--mark-step must be greater than zero")
    if args.label_reference_size <= 0:
        raise ValueError("--label-reference-size must be greater than zero")
    if args.label_base_font_size <= 0:
        raise ValueError("--label-base-font-size must be greater than zero")
    if args.label_circle_scale <= 0:
        raise ValueError("--label-circle-scale must be greater than zero")
    if args.stride <= 0:
        raise ValueError("--stride must be greater than zero")

    grid = parse_grid(args.grid)
    if grid is not None and args.columns is not None:
        raise ValueError("Use either --grid or --columns, not both")

    paths = find_images(args.input_dir, args.pattern)
    indices = parse_indices(args.indices)
    if indices is not None:
        max_index = len(paths)
        if any(index > max_index for index in indices):
            raise ValueError(
                f"--indices contains a value greater than screenshot count {max_index}"
            )
        paths = [paths[index - 1] for index in indices]
    else:
        paths = paths[:: args.stride]
    if grid is not None:
        columns, rows = grid
        frame_count = columns * rows
        if len(paths) > frame_count:
            paths = paths[:frame_count]
    else:
        columns = args.columns

    frames = load_images(paths, crop_box=parse_crop(args.crop), scale=args.scale)
    sheet = make_grid(
        frames,
        columns=columns,
        padding=args.padding,
        margin=args.margin,
        background=parse_color(args.background),
        mark=not args.no_mark,
        mark_prefix=args.mark_prefix,
        mark_start=args.mark_start,
        mark_step=args.mark_step,
        mark_fill=parse_color(args.mark_fill),
        mark_outline=parse_color(args.mark_outline),
        mark_text=parse_color(args.mark_text),
        label=args.label,
        label_template=args.label_template,
        label_position=args.label_position,
        label_shape=args.label_shape,
        label_circle_scale=args.label_circle_scale,
        label_reference_size=args.label_reference_size,
        label_base_font_size=args.label_base_font_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if sheet.mode == "RGBA" and parse_color(args.background)[3] == 255:
        sheet = sheet.convert("RGB")
    sheet.save(args.output)
    print(
        f"Saved {args.output} with {len(frames)} screenshots ({sheet.width}x{sheet.height})."
    )


if __name__ == "__main__":
    main()
