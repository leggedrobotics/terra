import cv2
import numpy as np
import os
from pathlib import Path
import random
from terra.env_generation.procedural_data import (
    add_non_dumpables,
    add_obstacles,
    initialize_image,
    save_or_display_image
)
from terra.env_generation.utils import color_dict, _get_img_mask

def get_random_close_positions(img_edge, size, min_dist=10, max_dist=20):
    # Randomly pick a position for the dump zone
    margin = 5
    x1 = random.randint(margin, img_edge - size - margin)
    y1 = random.randint(margin, img_edge - size - margin)
    # Now pick a position for the dirt pile within a certain distance, but not overlapping
    while True:
        angle = random.uniform(0, 2 * np.pi)
        dist = random.randint(min_dist, max_dist)
        x2 = int(x1 + np.cos(angle) * dist)
        y2 = int(y1 + np.sin(angle) * dist)
        # Ensure dirt pile is within bounds and does not overlap dump zone
        if (
            margin <= x2 <= img_edge - size - margin and
            margin <= y2 <= img_edge - size - margin and
            (abs(x2 - x1) >= size or abs(y2 - y1) >= size)
        ):
            break
    return (x1, y1), (x2, y2)

def add_dump_zone_easy(img, size, pos):
    x, y = pos
    img[x:x+size, y:y+size] = np.array(color_dict["dumping"])
    cumulative_mask = np.zeros_like(img[..., 0], dtype=bool)
    cumulative_mask[x:x+size, y:y+size] = True
    return img, cumulative_mask

def add_dirt_tile_easy(img, occ, dmp, cumulative_mask, size, pos):
    x, y = pos
    img[x:x+size, y:y+size] = np.array(color_dict["dirt"])
    cumulative_mask[x:x+size, y:y+size] = True
    return img, cumulative_mask

def save_action_image(drt, save_folder, i):
    save_folder_action = Path(save_folder) / "actions"
    save_folder_action.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_folder_action, "trench_" + str(i) + ".png"), drt
    )

def create_relocations_easy(n_imgs, save_folder):
    img_edge = 64
    size = 7
    for i in range(n_imgs):
        img = initialize_image(img_edge, img_edge, color_dict["neutral"])
        (dump_x, dump_y), (dirt_x, dirt_y) = get_random_close_positions(img_edge, size, min_dist=12, max_dist=20)
        img, cumulative_mask = add_dump_zone_easy(img, size=size, pos=(dump_x, dump_y))
        occ = img.copy()
        dmp = img.copy()
        drt, cumulative_mask = add_dirt_tile_easy(img, occ, dmp, cumulative_mask, size=size, pos=(dirt_x, dirt_y))
        # No obstacles or non-dumpables for easy maps
        save_or_display_image(img, occ, dmp, {}, save_folder, i)
        save_action_image(drt, save_folder, i)
    print(f"Easy relocation maps created in {save_folder}")

if __name__ == "__main__":
    save_folder = "terra/terra/env_generation/data/terra/relocations_easy"
    os.makedirs(save_folder, exist_ok=True)
    create_relocations_easy(10, save_folder)
