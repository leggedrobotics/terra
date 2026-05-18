#!/usr/bin/env python
import os
import numpy as np
import cv2
import random
import json
import argparse
import shutil
from pathlib import Path
import queue
from scipy.ndimage import binary_erosion
from terra.env_generation.utils import color_dict
from terra.env_generation.convert_to_terra import _convert_all_imgs_to_terra
import skimage
from skimage.measure import block_reduce

PACKAGE_DIR = Path(__file__).resolve().parents[2]
name_string = ""

FOUND_IMG_DIR = PACKAGE_DIR / 'data' / 'terra' / 'foundations_dumpzones_v3' / 'images'
FOUND_OCC_DIR = PACKAGE_DIR / 'data' / 'terra' / 'foundations_dumpzones_v3' / 'occupancy'
MIN_GAP = 10

def load_and_place_single_foundation(map_size, margin, img_dir, occ_dir):
    found_files = list(img_dir.glob('trench_*.png'))
    max_trials = 1000
    for trial in range(max_trials):
        filename = random.choice(found_files).name
        img = cv2.imread(str(img_dir / filename))
        occ = cv2.imread(str(occ_dir / filename), cv2.IMREAD_GRAYSCALE)
        mask_orig = (occ == 0)
        if not np.any(mask_orig):
            print(f"[skip {trial}] {filename} is empty dig mask before downsample.")
            continue
        ys, xs = np.where(mask_orig)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        bh, bw = y1 - y0, x1 - x0
        max_for_this = map_size - 2 * margin
        factor = max(1, int(np.ceil(max(bh, bw) / max_for_this)))
        img_ds = block_reduce(img, (factor, factor, 1), np.max)
        occ_ds = block_reduce(occ, (factor, factor), np.min)
        mask = (occ_ds == 0)
        if not np.any(mask):
            print(f"[skip {trial}] {filename} is empty dig after downsampling factor {factor}.")
            continue
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        dig_shape = mask[y0:y1, x0:x1]
        h, w = dig_shape.shape
        maxx = map_size - w - margin
        maxy = map_size - h - margin
        if maxx < margin or maxy < margin:
            print(f"[skip {trial}] {filename} downsampled bbox too large {h}x{w}")
            continue
        for attempt in range(30):
            px = random.randint(margin, maxx)
            py = random.randint(margin, maxy)
            placed_mask = np.zeros((map_size, map_size), dtype=bool)
            placed_mask[py:py + h, px:px + w] = dig_shape
            return placed_mask, (px, py, px + w, py + h)
    raise RuntimeError(f'Could not find/fit valid foundation shape after {max_trials} trials. See printed skips for details.')

def try_place_foundation_mask(map_size: int, dig_mask: np.ndarray, margin: int, other_masks=None):
    mh, mw = dig_mask.shape
    max_trials = 150
    if other_masks is None: other_masks = []
    for _ in range(max_trials):
        x0 = random.randint(margin, map_size - mw - margin)
        y0 = random.randint(margin, map_size - mh - margin)
        # Compose mask for collision check
        fits = True
        for m in other_masks:
            if np.any(dig_mask & m[y0:y0+mh, x0:x0+mw]):
                fits = False
                break
        if fits:
            placed_mask = np.zeros((map_size, map_size), dtype=bool)
            placed_mask[y0:y0+mh, x0:x0+mw] = dig_mask
            return placed_mask, (x0, y0, x0+mw, y0+mh)
    raise RuntimeError('Failed to place dig zone with required margin/mask')

def has_minimum_gap(mask, targets, min_gap):
    # mask: True=obstacle, False=free
    # targets: binary mask for targets (dump/foundation)
    # returns True iff exists path from some edge to targets with corridor >=min_gap wide
    h, w = mask.shape
    # Erode the mask by (min_gap//2) so that only wide enough channels remain
    eroded_free = ~binary_erosion(mask, structure=np.ones((min_gap, min_gap)))
    # Start flood from each edge point
    visited = np.zeros((h, w), dtype=bool)
    q = queue.Queue()
    for x in range(w):
        if eroded_free[0, x]: q.put((0, x)); visited[0, x]=True
        if eroded_free[h-1, x]: q.put((h-1, x)); visited[h-1, x]=True
    for y in range(h):
        if eroded_free[y, 0]: q.put((y, 0)); visited[y, 0]=True
        if eroded_free[y, w-1]: q.put((y, w-1)); visited[y, w-1]=True
    # BFS
    while not q.empty():
        y, x = q.get()
        if targets[y, x]:
            return True
        for dy, dx in [(-1, 0), (1, 0), (0,-1), (0, 1)]:
            yy, xx = y+dy, x+dx
            if 0<=yy<h and 0<=xx<w and eroded_free[yy,xx] and not visited[yy,xx]:
                q.put((yy,xx)); visited[yy,xx]=True
    return False

def add_fences_with_channel(map_size, n_fence, fence_width, fence_minlen, fence_maxlen, forbidden_masks, dump_mask, found_mask, min_gap=10, margin=3):
    fences = []
    all_mask = np.zeros((map_size, map_size), dtype=bool)
    for m in forbidden_masks:
        all_mask |= m
    max_trials = 100*n_fence
    for _ in range(max_trials):
        vert = random.choice([True, False])
        if vert:
            length = random.randint(fence_minlen, fence_maxlen)
            wx = random.randint(margin, map_size-fence_width-margin)
            wy = random.randint(margin, map_size-length-margin)
            bx0, by0, bx1, by1 = wx, wy, wx+fence_width, wy+length
        else:
            length = random.randint(fence_minlen, fence_maxlen)
            wx = random.randint(margin, map_size-length-margin)
            wy = random.randint(margin, map_size-fence_width-margin)
            bx0, by0, bx1, by1 = wx, wy, wx+length, wy+fence_width
        box_mask = np.zeros((map_size, map_size), dtype=bool)
        box_mask[by0:by1, bx0:bx1] = True
        if np.any(all_mask & box_mask):
            continue
        # Test channel gap
        mask_proposed = all_mask | box_mask
        if has_minimum_gap(mask_proposed, dump_mask, min_gap) and has_minimum_gap(mask_proposed, found_mask, min_gap):
            fences.append((bx0, by0, bx1, by1))
            all_mask = mask_proposed
            if len(fences) == n_fence:
                break
    return fences

def dumpability_map_with_obstacles(map_size, fence_mask):
    # White everywhere, black where obstacle
    array = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
    array[fence_mask] = color_dict['nondumpable']
    return array

def main():
    global name_string
    parser = argparse.ArgumentParser(description="Generate fence maps with real foundation (from v3 separated logic), dumpzone, fences, and Terra conversion.")
    parser.add_argument('--n_imgs', type=int, default=30)
    parser.add_argument('--map_size', type=int, default=64)
    parser.add_argument('--fence_width', type=int, default=2)
    parser.add_argument('--fence_minlen', type=int, default=32)
    parser.add_argument('--fence_maxlen', type=int, default=54)
    parser.add_argument('--name', type=str, default="foundations_fence_v2")
    args = parser.parse_args()
    name_string = args.name
    save_dir = PACKAGE_DIR / "data" / "terra" / name_string
    if save_dir.exists():
        print(f"Cleaning old output folder: {save_dir}")
        shutil.rmtree(save_dir)
    image_folder = save_dir / "images"
    occupancy_folder = save_dir / "occupancy"
    dumpability_folder = save_dir / "dumpability"
    actions_folder = save_dir / "actions"
    metadata_folder = save_dir / "metadata"
    for folder in [image_folder, occupancy_folder, dumpability_folder, actions_folder, metadata_folder]:
        folder.mkdir(parents=True, exist_ok=True)
    margin = 3
    min_gap = 10  # for fence channel
    for i in range(args.n_imgs):
        # --- Place foundation using v3_separated system ---
        foundation_mask, foundation_box = load_and_place_single_foundation(args.map_size, margin, FOUND_IMG_DIR, FOUND_OCC_DIR)
        # --- Place dump zone ---
        dump_size = random.randint(10, 18)
        dump_placed = False
        for _ in range(100):
            side = random.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                dx = random.randint(margin, args.map_size-dump_size-margin)
                dy = margin if side=="top" else args.map_size-dump_size-margin
            else:
                dx = margin if side=="left" else args.map_size-dump_size-margin
                dy = random.randint(margin, args.map_size-dump_size-margin)
            dump_box = (dx, dy, dx+dump_size, dy+dump_size)
            dump_mask = np.zeros((args.map_size, args.map_size), dtype=bool)
            dump_mask[dy:dy+dump_size, dx:dx+dump_size] = True
            # enforce margin to foundation
            expand_found = cv2.dilate(foundation_mask.astype(np.uint8), np.ones((2*margin+1,2*margin+1), np.uint8)).astype(bool)
            if not np.any(dump_mask & expand_found):
                dump_placed = True
                break
        if not dump_placed: raise RuntimeError('Failed to place dump zone with margin to foundation')
        # --- Place fences ---
        fences = add_fences_with_channel(
            args.map_size, random.randint(1,2), args.fence_width, args.fence_minlen, args.fence_maxlen, [foundation_mask, dump_mask], dump_mask, foundation_mask, min_gap=min_gap, margin=margin)
        fence_mask = np.zeros((args.map_size, args.map_size), dtype=bool)
        for bx0, by0, bx1, by1 in fences:
            fence_mask[by0:by1, bx0:bx1] = True
        # --- Compose all maps ---
        rgb = np.ones((args.map_size, args.map_size, 3), dtype=np.uint8) * color_dict['neutral']
        rgb[foundation_mask] = color_dict['digging']
        rgb[dump_mask] = color_dict['dumping']
        rgb[fence_mask] = color_dict['obstacle']
        occ = np.zeros((args.map_size, args.map_size, 3), dtype=np.uint8)
        occ[fence_mask] = color_dict['obstacle']
        dum = dumpability_map_with_obstacles(args.map_size, fence_mask)
        action_img = np.ones((args.map_size, args.map_size, 3), dtype=np.uint8) * 255
        img_name = f"trench_{i}.png"
        cv2.imwrite(str(image_folder / img_name), rgb)
        cv2.imwrite(str(occupancy_folder / img_name), occ)
        cv2.imwrite(str(dumpability_folder / img_name), dum)
        cv2.imwrite(str(actions_folder / img_name), action_img)
        meta_struct = {
            'foundation_box': foundation_box,
            'dump_box': dump_box,
            'fences': fences,
            'map_size': args.map_size
        }
        json_path = metadata_folder / f"trench_{i}.json"
        with open(json_path, 'w') as f:
            json.dump(meta_struct, f)
    destination_folder = save_dir / 'terra_npy'
    destination_folder.mkdir(exist_ok=True)
    _convert_all_imgs_to_terra(
        img_folder=image_folder,
        metadata_folder=metadata_folder,
        occupancy_folder=occupancy_folder,
        dumpability_folder=dumpability_folder,
        destination_folder=destination_folder,
        size=(args.map_size,args.map_size),
        n_imgs=args.n_imgs,
        all_dumpable=False,
        copy_metadata=True,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
        actions_folder=actions_folder
    )
    print(f"All maps and Terra npy data saved in {save_dir}")

if __name__ == "__main__":
    main()
