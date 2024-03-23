import numpy as np
import os
from pathlib import Path
import yaml
from procedural_data import add_obstacles, add_non_dumpables


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# def add_obstacles_to_occupancy(occ, n_obs_min, n_obs_max, size_obstacle_min, size_obstacle_max):
#     """
#     Adds obstacles to an image using the occupancy array to ensure they do not overlap with existing features.
#     This function is adapted to work with a 2D occupancy array instead of a 3D cumulative mask.

#     Parameters:
#     - img (np.ndarray): The image array where obstacles will be added.
#     - occ (np.ndarray): A 2D boolean occupancy array indicating areas where obstacles or other features already exist.
#     - n_obs_min (int): Minimum number of obstacles to add.
#     - n_obs_max (int): Maximum number of obstacles to add.
#     - size_obstacle_min (int): Minimum size of the obstacles.
#     - size_obstacle_max (int): Maximum size of the obstacles.

#     Returns:
#     - np.ndarray: The updated image array with obstacles added.
#     - np.ndarray: The updated occupancy array including the new obstacles.
#     """
#     w, h = occ.shape  # Extract width and height from the image dimensions
#     n_occ = 0  # Initialize the count of obstacles added
#     n_obs_now = np.random.randint(n_obs_min, n_obs_max + 1)  # Randomly decide the number of obstacles to add
#     occ_new = np.zeros_like(occ)  # Initialize a new occupancy array to store the updated obstacles
#     while n_occ < n_obs_now:
#         # Randomly determine the size of the obstacle
#         sizeox = np.random.randint(size_obstacle_min, size_obstacle_max + 1)
#         sizeoy = np.random.randint(size_obstacle_min, size_obstacle_max + 1)
#         # Randomly select a position for the obstacle
#         x = np.random.randint(0, w - sizeox)
#         y = np.random.randint(0, h - sizeoy)
#         # Check if the selected area overlaps with existing features in the occupancy array
#         if not occ[x:x+sizeox, y:y+sizeoy].any():
#             # Update the image and the occupancy array to include the new obstacle
#             occ_new[x:x+sizeox, y:y+sizeoy] = 1  # Mark the area as occupied
#             occ[x:x+sizeox, y:y+sizeoy] = 1  # Mark the area as occupied
#             n_occ += 1  # Increment the count of obstacles added

#     return occ_new, occ


def add_obstacles_to_occupancy(
    occ, n_obs_min, n_obs_max, size_obstacle_min, size_obstacle_max, occupied_value=1
):
    """
    Adds obstacles to an image using the occupancy array to ensure they do not overlap with existing features.
    This function is adapted to work with a 2D occupancy array instead of a 3D cumulative mask.
    It now accepts an additional parameter to specify the value that represents an occupied space.

    Parameters:
    - occ (np.ndarray): A 2D boolean occupancy array indicating areas where obstacles or other features already exist.
    - n_obs_min (int): Minimum number of obstacles to add.
    - n_obs_max (int): Maximum number of obstacles to add.
    - size_obstacle_min (int): Minimum size of the obstacles.
    - size_obstacle_max (int): Maximum size of the obstacles.
    - occupied_value (int, optional): The value that represents an occupied space in the occupancy array. Defaults to 1.

    Returns:
    - np.ndarray: The updated occupancy array with obstacles added.
    """
    w, h = occ.shape  # Extract width and height from the image dimensions
    n_occ = 0  # Initialize the count of obstacles added
    n_obs_now = np.random.randint(
        n_obs_min, n_obs_max + 1
    )  # Randomly decide the number of obstacles to add
    occ_new = np.ones_like(occ) * (
        1 - occupied_value
    )  # Initialize a new occupancy array to store the updated obstacles
    while n_occ < n_obs_now:
        # Randomly determine the size of the obstacle
        sizeox = np.random.randint(size_obstacle_min, size_obstacle_max + 1)
        sizeoy = np.random.randint(size_obstacle_min, size_obstacle_max + 1)
        # Randomly select a position for the obstacle
        x = np.random.randint(0, w - sizeox)
        y = np.random.randint(0, h - sizeoy)
        # Check if the selected area overlaps with existing features in the occupancy array
        if not np.any(occ[x : x + sizeox, y : y + sizeoy] == 1):
            # Update the occupancy array to include the new obstacle
            occ_new[
                x : x + sizeox, y : y + sizeoy
            ] = occupied_value  # Mark the area as occupied in the new occupancy array
            occ[
                x : x + sizeox, y : y + sizeoy
            ] = 1  # Also mark the area as occupied in the original occupancy array
            n_occ += 1  # Increment the count of obstacles added

    return occ_new, occ


def generate_squares(num_images, config_values, save_folder_base):
    x_dim = config_values["x_dim"]
    y_dim = config_values["y_dim"]
    side_lens = config_values["side_lens"]
    min_margin = config_values["min_margin"]
    max_margin = config_values["max_margin"]
    n_obs_min = config_values["n_obs_min"]
    n_obs_max = config_values["n_obs_max"]
    size_obstacle_min = config_values["size_obstacle_min"]
    size_obstacle_max = config_values["size_obstacle_max"]
    n_nodump_min = config_values["n_nodump_min"]
    n_nodump_max = config_values["n_nodump_max"]
    size_nodump_min = config_values["size_nodump_min"]
    size_nodump_max = config_values["size_nodump_max"]

    for side_len in side_lens:
        for i in range(1, num_images + 1):
            # Ensure the total size with margins does not exceed the dimensions
            total_max_side = min(x_dim, y_dim) - side_len
            max_margin_adjusted = min(max_margin, (total_max_side - side_len) // 2)

            if max_margin_adjusted < min_margin:
                print(
                    f"Skipping generation for side length {side_len} due to insufficient space."
                )
                continue

            # Ensuring at least two edges have max_margin
            margins = [
                np.random.randint(min_margin, max_margin_adjusted + 1) for _ in range(4)
            ]
            # Randomly choose two edges to set to max_margin_adjusted
            max_margin_edges = np.random.choice(range(4), 2, replace=False)
            for edge in max_margin_edges:
                margins[edge] = max_margin_adjusted

            left_margin, right_margin, top_margin, bottom_margin = margins

            # Calculate total dimensions including variable margins
            total_x = side_len + left_margin + right_margin
            total_y = side_len + top_margin + bottom_margin

            if x_dim < total_x or y_dim < total_y:
                print(
                    f"Skipping generation for side length {side_len} as it cannot fit within the given dimensions."
                )
                continue

            img = np.zeros((x_dim, y_dim))
            occ = np.zeros_like(img)
            dmp = np.ones_like(img)

            # Randomly position the entire structure within the dimensions
            x = np.random.randint(0, x_dim - total_x + 1)
            y = np.random.randint(0, y_dim - total_y + 1)

            # Set the dumpable area
            img[x : x + total_x, y : y + total_y] = 1  # must dump

            # Calculate and set the dig area inside the dumpable area
            x_dig = x + left_margin
            y_dig = y + top_margin
            img[x_dig : x_dig + side_len, y_dig : y_dig + side_len] = -1  # dig
            img = img.astype(np.int8)

            # Adjusting non-dumpable and occupation squares placement
            for _ in range(2):  # Non-dumpable squares
                x_nd = np.random.randint(0, max(1, x_dim - 3))
                y_nd = np.random.randint(0, max(1, y_dim - 3))
                dmp[x_nd : x_nd + 3, y_nd : y_nd + 3] = 0

            dmp = np.where(img == 1, 1, dmp).astype(np.bool_)

            occ_dig = np.where(img == -1, 1, occ).astype(np.bool_)
            occ, occ_cm = add_obstacles_to_occupancy(
                occ_dig, n_obs_min, n_obs_max, size_obstacle_min, size_obstacle_max
            )
            dmp, occ_cm = add_obstacles_to_occupancy(
                occ_cm,
                n_nodump_min,
                n_nodump_max,
                size_nodump_min,
                size_nodump_max,
                occupied_value=0,
            )

            # Saving
            save_folder = Path(save_folder_base) / f"{side_len}"
            save_folder_images = save_folder / "images"
            save_folder_occupancy = save_folder / "occupancy"
            save_folder_dumpability = save_folder / "dumpability"
            save_folder_images.mkdir(parents=True, exist_ok=True)
            save_folder_occupancy.mkdir(parents=True, exist_ok=True)
            save_folder_dumpability.mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(save_folder_images, f"img_{i}"), img)
            np.save(os.path.join(save_folder_occupancy, f"img_{i}"), occ)
            np.save(os.path.join(save_folder_dumpability, f"img_{i}"), dmp)
