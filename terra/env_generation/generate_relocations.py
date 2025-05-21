import numpy as np
import os

from terra.env_generation.procedural_data import (
    add_non_dumpables,
    add_obstacles,
    initialize_image,
    save_or_display_image
)

from terra.env_generation.utils import color_dict

def add_dumpables(img, n_dump_min, n_dump_max, size_dump_min, size_dump_max):
    w, h = img.shape[:2]
    n_dump = 0
    n_dump_todo = np.random.randint(n_dump_min, n_dump_max + 1)
    cumulative_mask = np.zeros_like(img[..., 0], dtype=bool)

    while n_dump < n_dump_todo:
        # Randomly determine the size of the dumping zone
        sizeox = np.random.randint(size_dump_min, size_dump_max + 1)
        sizeoy = np.random.randint(size_dump_min, size_dump_max + 1)
        # Randomly select a position for the dumping zone
        x = np.random.randint(0, w - sizeox)
        y = np.random.randint(0, h - sizeoy)
        # Update the dumping zone layer
        img[x : x + sizeox, y : y + sizeoy] = np.array(color_dict["dumping"])
        cumulative_mask[x : x + sizeox, y : y + sizeoy] = True
        n_dump += 1  # Increment the count of zones added

    return img, cumulative_mask

def generate_relocations(
    n_imgs,
    img_edge_min,
    img_edge_max,
    n_dump_min,
    n_dump_max,
    size_dump_min,
    size_dump_max,
    n_obs_min,
    n_obs_max,
    size_obstacle_min,
    size_obstacle_max
    n_nodump_min,
    n_nodump_max,
    size_nodump_min,
    size_nodump_max,
):
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_folder = os.path.join(package_dir, "data", "terra", "foundations")

    i = 0
    while i < n_imgs:
        img = initialize_image(img_edge_min, img_edge_max, color_dict["neutral"])
        img, cumulative_mask = add_dumpables(img, n_dump_min, n_dump_max, size_dump_min, size_dump_max)
        occ, cumulative_mask = add_obstacles(img, cumulative_mask, n_obs_min, n_obs_max, size_obstacle_min, size_obstacle_max)
        dmp, cumulative_mask = add_non_dumpables(img, occ, cumulative_mask, n_nodump_min, n_nodump_max, size_nodump_min, size_nodump_max)
        # TODO: Add spilled soil to the image
        save_or_display_image(img, occ, dmp, {}, save_folder, i)
