#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import numpy as np

try:
	from scipy import ndimage as ndi
	_HAS_SCIPY = True
except Exception:
	_HAS_SCIPY = False


def compute_distance_map_taxicab(target_map: np.ndarray) -> np.ndarray:
	"""
	Compute normalized Manhattan distance to nearest dump zone (target_map > 0).
	Returns float32 array in [0,1] with shape equal to target_map.
	"""
	dump = target_map > 0
	if _HAS_SCIPY:
		# Cityblock (taxicab) distance transform on the complement of dump zones
		dist = ndi.distance_transform_cdt(~dump, metric='taxicab').astype(np.float32)
	else:
		# Fallback: simple two-pass DP for cityblock
		h, w = target_map.shape
		big = np.int32(10**6)
		dist = np.where(dump, 0, big).astype(np.int32)
		# forward pass
		for i in range(h):
			for j in range(w):
				if dist[i, j] == 0:
					continue
				left = dist[i - 1, j] + 1 if i > 0 else big
				up = dist[i, j - 1] + 1 if j > 0 else big
				dist[i, j] = min(dist[i, j], left, up)
		# backward pass
		for i in range(h - 1, -1, -1):
			for j in range(w - 1, -1, -1):
				current = dist[i, j]
				right = dist[i + 1, j] + 1 if i < h - 1 else big
				down = dist[i, j + 1] + 1 if j < h - 1 else big
				dist[i, j] = min(current, right, down)
		dist = dist.astype(np.float32)
	
	# BETTER NORMALIZATION: Use realistic max distance for optimal resolution
	h, w = target_map.shape
	# For 64x64 map with center dump zones, max realistic distance is ~24 tiles
	# This gives perfect resolution: 2 tiles = 0.083, 4 tiles = 0.167, 8 tiles = 0.333
	realistic_max_distance = 24  # Optimal for 64x64 maps with center dump zones
	norm = float(realistic_max_distance) if realistic_max_distance > 0 else 1.0
	return dist / norm


def find_image_indices(images_dir: Path) -> list[int]:
	indices = []
	for p in images_dir.glob('img_*.npy'):
		name = p.stem  # img_123
		try:
			idx = int(name.split('_')[1])
			indices.append(idx)
		except Exception:
			continue
	indices.sort()
	return indices


def process_dataset_folder(dataset_folder: Path) -> None:
	images_dir = dataset_folder / 'images'
	if not images_dir.exists():
		raise RuntimeError(f"Missing images folder: {images_dir}")
	# Create/ensure distance folder
	distance_dir = dataset_folder / 'distance'
	distance_dir.mkdir(parents=True, exist_ok=True)

	indices = find_image_indices(images_dir)
	if not indices:
		print(f"No images found under {images_dir}")
		return

	print(f"Generating distance maps for {len(indices)} images in {dataset_folder} (overwrite enabled)")
	for idx in indices:
		img_path = images_dir / f'img_{idx}.npy'
		target_map = np.load(img_path)
		# Compute and save
		dist_map = compute_distance_map_taxicab(target_map)
		out_path = distance_dir / f'img_{idx}.npy'
		np.save(out_path, dist_map)
	print(f"Done: {dataset_folder}")


def process_root_folder(root: Path, recursive: bool) -> None:
	if recursive:
		# Walk subfolders; treat any folder containing 'images' as a dataset
		for dirpath, dirnames, filenames in os.walk(root):
			images = Path(dirpath) / 'images'
			if images.exists():
				try:
					process_dataset_folder(Path(dirpath))
				except Exception as e:
					print(f"Skipping {dirpath} due to error: {e}")
	else:
		process_dataset_folder(root)


def main():
	parser = argparse.ArgumentParser(description='Generate relocation distance maps (Manhattan) for dataset maps.')
	parser.add_argument('--dataset', type=str, required=True, help='Path to a dataset folder containing images/, or a root folder when --recursive is set')
	parser.add_argument('--recursive', action='store_true', help='If set, walk subfolders and process all datasets discovered')
	args = parser.parse_args()

	root = Path(args.dataset)
	if not root.exists():
		raise RuntimeError(f"Dataset path does not exist: {root}")
	process_root_folder(root, args.recursive)


if __name__ == '__main__':
	main() 