#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import numpy as np
try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
from collections import deque

def compute_geodesic_distance_map(target_map: np.ndarray, obstacle_mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """
    Compute a normalized distance-to-dump-zone map respecting obstacles.
    
    Args:
        target_map: 2D array; dump zone where target_map > 0
        obstacle_mask: 2D bool/uint8 array; obstacles where obstacle_mask != 0
        connectivity: 4 or 8 (grid neighbors)
    Returns:
        dist: np.ndarray, float32 map in [0, 1] (1 = max resolvable distance or unreachable)
    """
    h, w = target_map.shape
    dump = (target_map > 0)
    obstacle = (obstacle_mask != 0)
    dist = np.full((h, w), np.inf, dtype=np.float32)
    # Set 0 distance for dump zones (but not obstacles)
    frontier = deque()
    for i in range(h):
        for j in range(w):
            if dump[i, j] and not obstacle[i, j]:
                dist[i, j] = 0.0
                frontier.append((i, j))
    # Define neighbor moves
    if connectivity == 8:
        moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
    # BFS: shortest path in grid, respecting obstacles
    while frontier:
        i, j = frontier.popleft()
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                if obstacle[ni, nj]:
                    continue
                proposed = dist[i, j] + 1
                if dist[ni, nj] > proposed:
                    dist[ni, nj] = proposed
                    frontier.append((ni, nj))
    # Normalize: set unreachable (np.inf) cells to max value
    mask = np.isfinite(dist)
    if mask.any():
        maxd = dist[mask].max()
        if maxd > 0:
            dist[mask] = dist[mask] / maxd
    dist[~mask] = 1.0
    return dist.astype(np.float32)

def find_image_indices(images_dir: Path) -> list[int]:
    indices = []
    for p in images_dir.glob('img_*.npy'):
        name = p.stem
        try:
            idx = int(name.split('_')[1])
            indices.append(idx)
        except Exception:
            continue
    indices.sort()
    return indices

def process_dataset_folder(dataset_folder: Path):
    images_dir = dataset_folder / 'images'
    distance_dir = dataset_folder / 'distance_geodesic'
    occupancy_dir = dataset_folder / 'occupancy'
    distance_dir.mkdir(parents=True, exist_ok=True)
    if not images_dir.exists() or not occupancy_dir.exists():
        raise RuntimeError(f"Missing images or occupancy folder: {images_dir}, {occupancy_dir}")
    indices = find_image_indices(images_dir)
    if not indices:
        print(f"No images found under {images_dir}")
        return
    print(f"Generating OBS-aware geodesic distance maps for {len(indices)} images in {dataset_folder} (overwrite enabled)")
    for idx in indices:
        img_path = images_dir / f'img_{idx}.npy'
        occ_path = occupancy_dir / f'img_{idx}.npy'
        target_map = np.load(img_path)
        occupancy_mask = np.load(occ_path)
        dist_map = compute_geodesic_distance_map(target_map, occupancy_mask)
        out_path = distance_dir / f'img_{idx}.npy'
        np.save(out_path, dist_map)
    print(f"Done: {dataset_folder}")

def process_root_folder(root: Path, recursive: bool):
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            images = Path(dirpath) / 'images'
            occupancy = Path(dirpath) / 'occupancy'
            if images.exists() and occupancy.exists():
                try:
                    process_dataset_folder(Path(dirpath))
                except Exception as e:
                    print(f"Skipping {dirpath} due to error: {e}")
    else:
        process_dataset_folder(root)

def main():
    parser = argparse.ArgumentParser(description='Generate geodesic (obstacle-aware) relocation distance maps for dataset maps.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to a dataset folder containing images/ and occupancy/, or a root folder when --recursive is set')
    parser.add_argument('--recursive', action='store_true', help='If set, walk subfolders and process all datasets discovered')
    args = parser.parse_args()
    root = Path(args.dataset)
    if not root.exists():
        raise RuntimeError(f"Dataset path does not exist: {root}")
    process_root_folder(root, args.recursive)

if __name__ == '__main__':
    main()
