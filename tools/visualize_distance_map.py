#!/usr/bin/env python3
import os
import sys
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Ensure the package root (directory containing the 'terra' package) is on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.abspath(os.path.join(THIS_DIR, '..'))  # parent dir that contains 'terra' package
if PKG_PARENT not in sys.path:
    sys.path.insert(0, PKG_PARENT)

from terra.map import GridWorld  # noqa: E402


def _clip_rect(x0, y0, x1, y1, width, height):
    x0 = int(max(0, min(width - 1, x0)))
    x1 = int(max(0, min(width, x1)))
    y0 = int(max(0, min(height - 1, y0)))
    y1 = int(max(0, min(height, y1)))
    return x0, y0, x1, y1


def build_custom_maps(width: int, height: int,
                      rects: list[tuple[int, int, int, int]] | None,
                      default_rect: tuple[int, int, int, int]):
    """
    Create simple maps with one or more rectangular dump zones in target_map.
    - target_map: 1 in dump zone(s), 0 elsewhere
    - action_map: all zeros
    - padding_mask: all zeros (fully traversable)
    - dumpability_mask_init: all ones (dumpable everywhere for simplicity)
    - trench placeholders minimal
    """
    target_map = jnp.zeros((width, height), dtype=jnp.int8)

    rects_to_use = rects if rects and len(rects) > 0 else [default_rect]
    for (x0, y0, x1, y1) in rects_to_use:
        x0, y0, x1, y1 = _clip_rect(x0, y0, x1, y1, width, height)
        target_map = target_map.at[x0:x1, y0:y1].set(1)

    action_map = jnp.zeros((width, height), dtype=jnp.int8)
    padding_mask = jnp.zeros((width, height), dtype=jnp.int8)
    dumpability_mask_init = jnp.ones((width, height), dtype=jnp.bool_)

    # Trench placeholders (unused for this visualization)
    trench_axes = jnp.zeros((1, 3), dtype=jnp.float32)
    trench_type = jnp.int32(0)

    return target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init, action_map


def visualize_distance_map(dist_map: jnp.ndarray, output_path: str):
    # Convert to numpy if it's a JAX array
    dist_map_np = np.array(dist_map)
    
    plt.figure(figsize=(12, 8))
    
    # Main distance map
    plt.subplot(2, 2, 1)
    im1 = plt.imshow(dist_map_np, cmap='viridis', origin='upper')
    plt.colorbar(im1, label='Normalized distance to nearest dump zone')
    plt.title('Relocation Distance Map')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # Distance histogram
    plt.subplot(2, 2, 2)
    plt.hist(dist_map_np.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Distance value')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.grid(True, alpha=0.3)
    
    # Distance map with different colormap for better contrast
    plt.subplot(2, 2, 3)
    im2 = plt.imshow(dist_map_np, cmap='plasma', origin='upper')
    plt.colorbar(im2, label='Normalized distance to nearest dump zone')
    plt.title('Distance Map (Plasma colormap)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # Distance map with log scale for better visualization of small differences
    plt.subplot(2, 2, 4)
    # Add small epsilon to avoid log(0)
    dist_map_log = np.log(dist_map_np + 1e-6)
    im3 = plt.imshow(dist_map_log, cmap='inferno', origin='upper')
    plt.colorbar(im3, label='Log(distance + ε)')
    plt.title('Distance Map (Log scale)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _parse_rect(rect_str: str) -> tuple[int, int, int, int]:
    parts = rect_str.split(',')
    if len(parts) != 4:
        raise ValueError(f"--dump-rect must be x0,y0,x1,y1; got: {rect_str}")
    x0, y0, x1, y1 = map(int, parts)
    return x0, y0, x1, y1


def main():
    parser = argparse.ArgumentParser(description='Visualize relocation distance map for custom dump zone(s) or load pre-computed maps.')
    parser.add_argument('--load-distance', type=str, help='Load pre-computed distance map from .npy file')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--dump-x0', type=int, default=44)
    parser.add_argument('--dump-y0', type=int, default=44)
    parser.add_argument('--dump-x1', type=int, default=62)
    parser.add_argument('--dump-y1', type=int, default=62)
    parser.add_argument('--dump-rect', action='append', help='Add a dump rectangle x0,y0,x1,y1; can be provided multiple times')
    parser.add_argument('--output', type=str, default='distance_map.png')

    args = parser.parse_args()

    if args.load_distance:
        # Load pre-computed distance map
        print(f"Loading pre-computed distance map from: {args.load_distance}")
        dist_map = np.load(args.load_distance)
        print(f"Distance map shape: {dist_map.shape}, min={dist_map.min():.4f}, max={dist_map.max():.4f}")
        print(f"Distance map dtype: {dist_map.dtype}")
    else:
        # Generate custom distance map
        rects = []
        if args.dump_rect:
            for r in args.dump_rect:
                rects.append(_parse_rect(r))

        default_rect = (args.dump_x0, args.dump_y0, args.dump_x1, args.dump_y1)

        target_map, padding_mask, trench_axes, trench_type, dumpability_mask_init, action_map = build_custom_maps(
            args.width, args.height, rects, default_rect
        )

        world = GridWorld.new(
            target_map=target_map,
            padding_mask=padding_mask,
            trench_axes=trench_axes,
            trench_type=trench_type,
            dumpability_mask_init=dumpability_mask_init,
            action_map=action_map,
        )

        dist_map = world.relocation_distance_map
        print(f"Distance map shape: {dist_map.shape}, min={dist_map.min():.4f}, max={dist_map.max():.4f}")

    out_path = os.path.abspath(args.output)
    visualize_distance_map(dist_map, out_path)
    print(f"Saved visualization to {out_path}")


if __name__ == '__main__':
    main() 