"""
Distance-map generation for Terra datasets.

This module is the single source of truth for distance-to-dump-zone map
generation. It is invoked automatically by
`convert_to_terra._convert_all_imgs_to_terra` at the end of conversion, and is
also exposed via the thin CLI wrappers in `tools/generate_distance_maps.py`
and `tools/generate_geodesic_distance_maps.py`.

Two metrics are supported:
    - "manhattan": cityblock (taxicab) distance transform, no obstacle awareness.
                   Fast, used by default.
    - "geodesic":  BFS distance respecting the occupancy mask (obstacle-aware).

Normalization mirrors the previous `tools/` scripts so existing datasets and
training semantics remain unchanged:
    - manhattan: divide by `realistic_max_distance` (defaults to 24, which is
      the sensible value for 64x64 maps with central dump zones).
    - geodesic:  divide by the per-map finite max distance; unreachable cells
      are set to 1.0.
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


DEFAULT_REALISTIC_MAX_DISTANCE = 24


def compute_distance_map_taxicab(
    target_map: np.ndarray,
    realistic_max_distance: int = DEFAULT_REALISTIC_MAX_DISTANCE,
) -> np.ndarray:
    """
    Normalized Manhattan (taxicab) distance to nearest dump zone
    (`target_map > 0`). Returns float32 in [0, ~1].
    """
    dump = target_map > 0
    if _HAS_SCIPY:
        dist = ndi.distance_transform_cdt(~dump, metric="taxicab").astype(np.float32)
    else:
        h, w = target_map.shape
        big = np.int32(10**6)
        dist = np.where(dump, 0, big).astype(np.int32)
        for i in range(h):
            for j in range(w):
                if dist[i, j] == 0:
                    continue
                left = dist[i - 1, j] + 1 if i > 0 else big
                up = dist[i, j - 1] + 1 if j > 0 else big
                dist[i, j] = min(dist[i, j], left, up)
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                current = dist[i, j]
                right = dist[i + 1, j] + 1 if i < h - 1 else big
                down = dist[i, j + 1] + 1 if j < h - 1 else big
                dist[i, j] = min(current, right, down)
        dist = dist.astype(np.float32)

    norm = float(realistic_max_distance) if realistic_max_distance > 0 else 1.0
    return dist / norm


def compute_geodesic_distance_map(
    target_map: np.ndarray,
    obstacle_mask: np.ndarray,
    connectivity: int = 4,
) -> np.ndarray:
    """
    Normalized obstacle-aware geodesic distance to nearest reachable dump zone.

    Args:
        target_map: 2D array; dump zone where target_map > 0.
        obstacle_mask: 2D bool/uint8; obstacles where value != 0.
        connectivity: 4 or 8.
    Returns:
        float32 in [0, 1]. Unreachable cells are set to 1.0.
    """
    h, w = target_map.shape
    dump = target_map > 0
    obstacle = obstacle_mask != 0
    dist = np.full((h, w), np.inf, dtype=np.float32)

    frontier: deque = deque()
    for i in range(h):
        for j in range(w):
            if dump[i, j] and not obstacle[i, j]:
                dist[i, j] = 0.0
                frontier.append((i, j))

    if connectivity == 8:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while frontier:
        i, j = frontier.popleft()
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and not obstacle[ni, nj]:
                proposed = dist[i, j] + 1
                if dist[ni, nj] > proposed:
                    dist[ni, nj] = proposed
                    frontier.append((ni, nj))

    finite = np.isfinite(dist)
    if finite.any():
        maxd = dist[finite].max()
        if maxd > 0:
            dist[finite] = dist[finite] / maxd
    dist[~finite] = 1.0
    return dist.astype(np.float32)


def _find_image_indices(images_dir: Path) -> list[int]:
    indices: list[int] = []
    for p in images_dir.glob("img_*.npy"):
        try:
            indices.append(int(p.stem.split("_")[1]))
        except Exception:
            continue
    indices.sort()
    return indices


def write_distance_maps(
    dataset_folder: os.PathLike | str,
    metric: str = "manhattan",
    *,
    connectivity: int = 4,
    realistic_max_distance: int = DEFAULT_REALISTIC_MAX_DISTANCE,
    distance_subfolder: str = "distance",
    verbose: bool = True,
) -> None:
    """
    Write a `distance/` subfolder inside a Terra-format dataset directory.

    Expects `<dataset_folder>/images/img_*.npy` (always) and
    `<dataset_folder>/occupancy/img_*.npy` (only when metric == "geodesic").

    Args:
        dataset_folder: Terra dataset directory (must already contain images/).
        metric: "manhattan" (default) or "geodesic".
        connectivity: 4 or 8 (only used for geodesic).
        realistic_max_distance: normalization constant for manhattan metric.
        distance_subfolder: output subfolder name (default "distance").
        verbose: print progress.
    """
    dataset_folder = Path(dataset_folder)
    images_dir = dataset_folder / "images"
    if not images_dir.exists():
        raise RuntimeError(f"Missing images folder: {images_dir}")

    metric_norm = metric.lower()
    if metric_norm not in ("manhattan", "geodesic"):
        raise ValueError(f"Unknown distance metric: {metric!r}")

    occupancy_dir = dataset_folder / "occupancy"
    if metric_norm == "geodesic" and not occupancy_dir.exists():
        raise RuntimeError(
            f"Geodesic metric requires occupancy folder: {occupancy_dir}"
        )

    distance_dir = dataset_folder / distance_subfolder
    distance_dir.mkdir(parents=True, exist_ok=True)

    indices = _find_image_indices(images_dir)
    if not indices:
        if verbose:
            print(f"No images found under {images_dir}")
        return

    if verbose:
        print(
            f"Generating {metric_norm} distance maps for {len(indices)} images "
            f"in {dataset_folder} (overwrite enabled)"
        )

    for idx in indices:
        img_path = images_dir / f"img_{idx}.npy"
        target_map = np.load(img_path)
        if metric_norm == "manhattan":
            dist_map = compute_distance_map_taxicab(
                target_map, realistic_max_distance=realistic_max_distance
            )
        else:
            occupancy_mask = np.load(occupancy_dir / f"img_{idx}.npy")
            dist_map = compute_geodesic_distance_map(
                target_map, occupancy_mask, connectivity=connectivity
            )
        np.save(distance_dir / f"img_{idx}.npy", dist_map)

    if verbose:
        print(f"Done: {dataset_folder}")


def write_distance_maps_recursive(
    root: os.PathLike | str,
    metric: str = "manhattan",
    **kwargs,
) -> None:
    """
    Walk subfolders under `root` and run `write_distance_maps` for every
    folder that looks like a Terra dataset (contains an images/ dir,
    plus occupancy/ when metric == "geodesic").
    """
    root = Path(root)
    metric_norm = metric.lower()
    for dirpath, _dirnames, _filenames in os.walk(root):
        p = Path(dirpath)
        if not (p / "images").exists():
            continue
        if metric_norm == "geodesic" and not (p / "occupancy").exists():
            continue
        try:
            write_distance_maps(p, metric=metric_norm, **kwargs)
        except Exception as e:
            print(f"Skipping {p} due to error: {e}")


__all__ = [
    "DEFAULT_REALISTIC_MAX_DISTANCE",
    "compute_distance_map_taxicab",
    "compute_geodesic_distance_map",
    "write_distance_maps",
    "write_distance_maps_recursive",
]
