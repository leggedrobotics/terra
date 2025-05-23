import cv2
import numpy as np
import os

from pathlib import Path
from terra.env_generation.procedural_data import (
    add_non_dumpables,
    add_obstacles,
    initialize_image,
    save_or_display_image
)

from terra.env_generation.utils import color_dict

def add_dump_zones(img, n_dump_min, n_dump_max, size_dump_min, size_dump_max):
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

def add_dirt_tiles(img, cumulative_mask, n_dirt_min, n_dirt_max, size_dirt_min, size_dirt_max):
    w, h = img.shape[:2]
    n_dirt = 0
    n_dirt_todo = np.random.randint(n_dirt_min, n_dirt_max + 1)
    cumulative_mask = np.zeros_like(img[..., 0], dtype=bool)

    while n_dirt < n_dirt_todo:
        # Randomly determine the size of the dirt pile
        sizeox = np.random.randint(size_dirt_min, size_dirt_max + 1)
        sizeoy = np.random.randint(size_dirt_min, size_dirt_max + 1)
        # Randomly select a position for the dirt pile
        x = np.random.randint(0, w - sizeox)
        y = np.random.randint(0, h - sizeoy)
        # Update the dumping zone layer
        img[x : x + sizeox, y : y + sizeoy] = np.array(color_dict["dirt"])
        cumulative_mask[x : x + sizeox, y : y + sizeoy] = True
        n_dirt += 1

    return img, cumulative_mask

def save_action_image(drt, save_folder, i):
    # make dir if does not exist
    os.makedirs(save_folder, exist_ok=True)
    save_folder_action = Path(save_folder) / "actions"
    save_folder_action.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_folder_action, "trench_" + str(i) + ".png"), drt
    )  # Added .png extension

def create_relocations(config, n_imgs):
    img_edge_min = config["img_edge_min"]
    img_edge_max = config["img_edge_max"]
    n_dump_min = config["n_dump_min"]
    n_dump_max = config["n_dump_max"]
    size_dump_min = config["size_dump_min"]
    size_dump_max = config["size_dump_max"]
    n_obs_min = config["n_obs_min"]
    n_obs_max = config["n_obs_max"]
    size_obstacle_min = config["size_obstacle_min"]
    size_obstacle_max = config["size_obstacle_max"]
    n_nodump_min = config["n_nodump_min"]
    n_nodump_max = config["n_nodump_max"]
    size_nodump_min = config["size_nodump_min"]
    size_nodump_max = config["size_nodump_max"]
    n_dirt_min = config["n_dirt_min"]
    n_dirt_max = config["n_dirt_max"]
    size_dirt_min = config["size_dirt_min"]
    size_dirt_max = config["size_dirt_max"]

    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_folder = os.path.join(package_dir, "data", "terra", "relocations")

    i = 0
    while i < n_imgs:
        img = initialize_image(img_edge_min, img_edge_max, color_dict["neutral"])
        img, cumulative_mask = add_dump_zones(img, n_dump_min, n_dump_max, size_dump_min, size_dump_max)
        occ, cumulative_mask = add_obstacles(img, cumulative_mask, n_obs_min, n_obs_max, size_obstacle_min, size_obstacle_max)
        dmp, cumulative_mask = add_non_dumpables(img, occ, cumulative_mask, n_nodump_min, n_nodump_max, size_nodump_min, size_nodump_max)
        drt, cumulative_mask = add_dirt_tiles(img, cumulative_mask, n_dirt_min, n_dirt_max, size_dirt_min, size_dirt_max)
        save_or_display_image(img, occ, dmp, {}, save_folder, i)
        save_action_image(drt, save_folder, i)
        i += 1

    print("Relocations created successfully.")
