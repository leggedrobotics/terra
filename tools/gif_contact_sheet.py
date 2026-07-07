#!/usr/bin/env python3
"""Convert an animated GIF into a single raster contact-sheet PNG."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageDraw, ImageFont, ImageSequence
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "This script needs Pillow. Install it with `pip install pillow` or run it "
        "inside the repo's `terra` conda environment."
    ) from exc


def parse_frame_spec(spec: str) -> set[int]:
    """Parse frame specs such as '0,4,8-12,20' into zero-based frame ids."""
    frame_ids: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending frame range: {part}")
            frame_ids.update(range(start, end + 1))
        else:
            frame_ids.add(int(part))
    return frame_ids


def evenly_limit_frames(
    frames: list[Image.Image], max_frames: int | None
) -> list[Image.Image]:
    if max_frames is None or len(frames) <= max_frames:
        return frames
    if max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero")

    indices = {
        round(i * (len(frames) - 1) / max(1, max_frames - 1)) for i in range(max_frames)
    }
    return [frame for i, frame in enumerate(frames) if i in indices]


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


def load_frames(
    gif_path: Path,
    *,
    start: int,
    end: int | None,
    stride: int,
    frame_spec: set[int] | None,
    crop_box: tuple[int, int, int, int] | None,
    scale: float,
) -> list[tuple[int, Image.Image]]:
    if stride <= 0:
        raise ValueError("--stride must be greater than zero")
    if scale <= 0:
        raise ValueError("--scale must be greater than zero")

    frames: list[tuple[int, Image.Image]] = []
    with Image.open(gif_path) as gif:
        for index, frame in enumerate(ImageSequence.Iterator(gif)):
            if index < start:
                continue
            if end is not None and index > end:
                break
            if frame_spec is not None and index not in frame_spec:
                continue
            if frame_spec is None and (index - start) % stride != 0:
                continue

            image = frame.convert("RGBA")
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


def parse_color(value: str) -> tuple[int, int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) not in (6, 8):
        raise ValueError("Colors must be hex RGB or RGBA, e.g. ffffff or ffffffff")
    channels = [int(value[i : i + 2], 16) for i in range(0, len(value), 2)]
    if len(channels) == 3:
        channels.append(255)
    return tuple(channels)  # type: ignore[return-value]


def make_contact_sheet(
    indexed_frames: list[tuple[int, Image.Image]],
    *,
    columns: int | None,
    padding: int,
    margin: int,
    background: tuple[int, int, int, int],
    label: bool,
) -> Image.Image:
    if not indexed_frames:
        raise ValueError("No frames selected")
    if padding < 0 or margin < 0:
        raise ValueError("--padding and --margin must be non-negative")

    indexed_frames = normalize_frame_sizes(indexed_frames, background)
    frame_width = indexed_frames[0][1].width
    frame_height = indexed_frames[0][1].height
    label_height = 18 if label else 0

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
    font = ImageFont.load_default()

    for output_index, (source_index, frame) in enumerate(indexed_frames):
        row = output_index // columns
        col = output_index % columns
        x = margin + col * (frame_width + padding)
        y = margin + row * (frame_height + label_height + padding)
        sheet.alpha_composite(frame, (x, y))
        if label:
            draw.text(
                (x + 3, y + frame_height + 3),
                f"frame {source_index}",
                fill=(20, 20, 20, 255),
                font=font,
            )

    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unroll an animated GIF into a single PNG contact sheet for papers, "
            "slides, and static PDFs."
        )
    )
    parser.add_argument("gif", type=Path, help="Input GIF path")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First source frame to consider, zero-based (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last source frame to consider, inclusive and zero-based",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth frame after --start (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Evenly downsample the selected frames to at most this many frames",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Explicit source frames/ranges to keep, e.g. '3,8,12-20'. Overrides --stride.",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop each frame before layout as x,y,w,h",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale each selected frame before layout (default: 1.0)",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=None,
        help="Number of columns in the raster. Defaults to a near-square layout.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=12,
        help="Pixels between frames (default: 12)",
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
        "--label",
        action="store_true",
        help="Add source frame numbers below each cell",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame_spec = parse_frame_spec(args.frames) if args.frames else None
    crop_box = parse_crop(args.crop)
    background = parse_color(args.background)

    indexed_frames = load_frames(
        args.gif,
        start=args.start,
        end=args.end,
        stride=args.stride,
        frame_spec=frame_spec,
        crop_box=crop_box,
        scale=args.scale,
    )
    limited_images = evenly_limit_frames(
        [frame for _, frame in indexed_frames], args.max_frames
    )
    limited_ids = [index for index, _ in indexed_frames]
    if args.max_frames is not None and len(indexed_frames) > args.max_frames:
        kept_positions = {
            round(i * (len(indexed_frames) - 1) / max(1, args.max_frames - 1))
            for i in range(args.max_frames)
        }
        limited_ids = [
            index
            for position, (index, _) in enumerate(indexed_frames)
            if position in kept_positions
        ]
    limited_frames = list(zip(limited_ids, limited_images, strict=True))

    sheet = make_contact_sheet(
        limited_frames,
        columns=args.columns,
        padding=args.padding,
        margin=args.margin,
        background=background,
        label=args.label,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if sheet.mode == "RGBA" and background[3] == 255:
        sheet = sheet.convert("RGB")
    sheet.save(args.output)
    print(
        f"Saved {args.output} with {len(limited_frames)} frames "
        f"({sheet.width}x{sheet.height})."
    )


if __name__ == "__main__":
    main()
