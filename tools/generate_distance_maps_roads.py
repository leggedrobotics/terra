#!/usr/bin/env python3
"""
Generate distance maps where both dump zones and non-dumpable tiles are treated
as zero-distance targets. This is useful for training agents that should
approach either a dump zone or any restricted (non-dumpable) region.
"""

import argparse
import os
from pathlib import Path

import numpy as np

try:
	from scipy import ndimage as ndi
	_HAS_SCIPY = True
except Exception:
	_HAS_SCIPY = False


def _infer_non_dumpable_mask(dumpability_map: np.ndarray) -> np.ndarray:
	"""
	Convert an arbitrary dumpability map to a boolean mask where True indicates
	a non-dumpable tile.
	"""
	if dumpability_map.dtype == np.bool_:
		return ~dumpability_map
	# Common encodings: {0,1} or {0,255}. Everything strictly greater than zero
	# is considered dumpable.
	return dumpability_map <= 0


def compute_distance_map_dump_or_nodump(
	target_map: np.ndarray,
	dumpability_map: np.ndarray,
	realistic_max_distance: int = 24,
) -> np.ndarray:
	"""
	Compute a normalized Manhattan distance to the nearest dump zone OR
	non-dumpable tile (both act as sinks).

	Args:
		target_map: 2D array; dump zones where target_map > 0
		dumpability_map: 2D array; dumpable tiles (1/True) vs non-dumpable (0/False)
		realistic_max_distance: normalization divisor (default tuned for 64x64 maps)
	Returns:
		float32 numpy array with shape equal to the input maps, values in [0, 1]
	"""
	if target_map.shape != dumpability_map.shape:
		raise ValueError(
			f"Shape mismatch between target map {target_map.shape} and dumpability map {dumpability_map.shape}"
		)

	target_map = np.asarray(target_map)
	dumpability_map = np.asarray(dumpability_map)

	dump_mask = target_map > 0
	non_dumpable_mask = _infer_non_dumpable_mask(dumpability_map)
	sinks = dump_mask | non_dumpable_mask

	if sinks.all():
		return np.zeros_like(target_map, dtype=np.float32)

	if _HAS_SCIPY:
		dist = ndi.distance_transform_cdt(~sinks, metric="taxicab").astype(np.float32)
	else:
		# Fallback: manual 2-pass cityblock distance
		h, w = target_map.shape
		big = np.int32(10**6)
		dist = np.where(sinks, 0, big).astype(np.int32)
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
				down = dist[i, j + 1] + 1 if j < w - 1 else big
				dist[i, j] = min(current, right, down)
		dist = dist.astype(np.float32)

	norm = float(realistic_max_distance) if realistic_max_distance > 0 else 1.0
	return dist / norm


def find_image_indices(images_dir: Path) -> list[int]:
	indices: list[int] = []
	for p in images_dir.glob("img_*.npy"):
		name = p.stem
		try:
			idx = int(name.split("_")[1])
			indices.append(idx)
		except Exception:
			continue
	indices.sort()
	return indices


def _resolve_dumpability_dir(dataset_folder: Path) -> Path:
	candidates = [
		dataset_folder / "dumpability",
		dataset_folder / "dumpability_mask",
		dataset_folder / "dumpability_masks",
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise RuntimeError(f"Missing dumpability folder under {dataset_folder} (looked for {', '.join(str(c) for c in candidates)})")


def process_dataset_folder(dataset_folder: Path) -> None:
	images_dir = dataset_folder / "images"
	if not images_dir.exists():
		raise RuntimeError(f"Missing images folder: {images_dir}")

	dumpability_dir = _resolve_dumpability_dir(dataset_folder)
	distance_dir = dataset_folder / "distance_dump_or_nodump"
	distance_dir.mkdir(parents=True, exist_ok=True)

	indices = find_image_indices(images_dir)
	if not indices:
		print(f"No images found under {images_dir}")
		return

	print(
		f"Generating dump-or-nondump distance maps for {len(indices)} images in {dataset_folder} (overwrite enabled)"
	)
	for idx in indices:
		img_path = images_dir / f"img_{idx}.npy"
		dumpability_path = dumpability_dir / f"img_{idx}.npy"
		if not dumpability_path.exists():
			raise RuntimeError(f"Missing dumpability map for index {idx}: {dumpability_path}")
		target_map = np.load(img_path)
		dumpability_map = np.load(dumpability_path)
		dist_map = compute_distance_map_dump_or_nodump(target_map, dumpability_map)
		out_path = distance_dir / f"img_{idx}.npy"
		np.save(out_path, dist_map)
	print(f"Done: {dataset_folder}")


def process_root_folder(root: Path, recursive: bool) -> None:
	if recursive:
		for dirpath, dirnames, filenames in os.walk(root):
			dirpath = Path(dirpath)
			images = dirpath / "images"
			if images.exists():
				try:
					_resolve_dumpability_dir(dirpath)
				except RuntimeError:
					continue
				try:
					process_dataset_folder(dirpath)
				except Exception as e:
					print(f"Skipping {dirpath} due to error: {e}")
	else:
		process_dataset_folder(root)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Generate Manhattan distance maps where both dump zones and non-dumpable tiles are treated as sinks."
		)
	)
	parser.add_argument(
		"--dataset",
		type=str,
		required=True,
		help="Path to a dataset folder containing images/ and dumpability/, or a root folder when --recursive is set",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="If set, walk subfolders and process all datasets discovered",
	)
	args = parser.parse_args()

	root = Path(args.dataset)
	if not root.exists():
		raise RuntimeError(f"Dataset path does not exist: {root}")
	process_root_folder(root, args.recursive)


if __name__ == "__main__":
	main()

