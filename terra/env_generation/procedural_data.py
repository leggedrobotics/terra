import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import skimage

from terra.env_generation.postprocessing import (
    _convert_dumpability_to_terra,
    _convert_img_to_terra,
    _convert_occupancy_to_terra,
)
from terra.env_generation.utils import _get_img_mask, color_dict


def distance_point_to_line(x, y, A, B, C):
    """
    Calculate the distance from a point (x, y) to a line represented by Ax + By + C = 0.

    Parameters:
    - x, y (float): Coordinates of the point.
    - A, B, C (float): Coefficients of the line equation.

    Returns:
    - float: The distance from the point to the line.
    """
    return abs(A * x + B * y + C) / math.sqrt(A**2 + B**2)


def initialize_image(img_edge_min, img_edge_max, color_dict):
    """
    Initializes an image array with dimensions within given range, applying a color scheme based on a corner dump strategy.

    Parameters:
    - img_edge_min, img_edge_max (int): Minimum and maximum edge sizes for the image.
    - color_dict (dict): A dictionary containing color mappings.

    Returns:
    - img (np.array): Initialized image array.
    - w, h (int): Width and height of the image.
    """
    # Randomly select dimensions within the specified range
    w, h = np.random.randint(img_edge_min, img_edge_max + 1, size=2, dtype=np.int32)
    img = np.ones((w, h, 3)) * np.array(color_dict["neutral"])

    # Randomly select a corner to dump
    corner_dump = np.random.randint(0, 4)
    if corner_dump == 0:
        img[0 : int(0.8 * w), :, :] = np.array(color_dict["dumping"])
    elif corner_dump == 1:
        img[int(0.2 * w) :, :, :] = np.array(color_dict["dumping"])
    elif corner_dump == 2:
        img[:, int(0.2 * h) :, :] = np.array(color_dict["dumping"])
    elif corner_dump == 3:
        img[:, : int(0.8 * h), :] = np.array(color_dict["dumping"])

    return img


def generate_edges(img, edges_range, sizes_small, sizes_long, color_dict):
    # Generate edges based on 'n_edges', 'sizes_small', and 'sizes_long'
    min_ssmall, max_ssmall = sizes_small
    min_slong, max_slong = sizes_long
    min_edges, max_edges = edges_range

    n_edges = np.random.randint(min_edges, max_edges + 1)

    prev_horizontal = True if np.random.choice([0, 1]).astype(np.bool_) else False
    w, h = img.shape[0], img.shape[1]
    lines_abc = []
    lines_pts = []
    for edge_i in range(n_edges):
        if edge_i == 0:
            mask = np.ones_like(img[..., 0], dtype=np.bool_)
            cumulative_mask = np.zeros_like(img[..., 0], dtype=np.bool_)
        fmask = mask.reshape(-1)
        fmask_idx_set = list(set((np.arange(w * h) * fmask).tolist()))[
            1:
        ]  # remove idx 0 as it's always going to be present
        fidxs = np.array(fmask_idx_set)
        idx = np.random.choice(fidxs)
        x = idx // h
        y = idx % h
        size_small = np.random.randint(min_ssmall, max_ssmall + 1)
        size_long = np.random.randint(min_slong, max_slong + 1)
        if prev_horizontal:
            size_x = size_long
            size_y = size_small

            # Compute axes
            y_coord = (2 * y + size_y - 1) / 2
            axis_pt1 = (float(y_coord), float(x))
            axis_pt2 = (float(y_coord), float(x) + size_x - 1)
        else:
            size_x = size_small
            size_y = size_long

            # Compute axes
            x_coord = (2 * x + size_x - 1) / 2
            axis_pt1 = (float(y), float(x_coord))
            axis_pt2 = (float(y) + size_y - 1, float(x_coord))
        prev_horizontal = not prev_horizontal
        lines_pts.append([axis_pt1, axis_pt2])

        size_x = min(size_x, w - x)
        size_y = min(size_y, h - y)

        img[x : x + size_x, y : y + size_y] = np.array(color_dict["digging"])

        A = axis_pt2[1] - axis_pt1[1]
        B = axis_pt1[0] - axis_pt2[0]
        C = axis_pt2[0] * axis_pt1[1] - axis_pt1[0] * axis_pt2[1]
        lines_abc.append(
            {
                "A": float(A),
                "B": float(B),
                "C": float(C),
            }
        )

        mask = np.zeros_like(img[..., 0], dtype=np.bool_)
        mask[x : x + size_x, y : y + size_y] = np.ones((size_x, size_y), dtype=np.bool_)
        cumulative_mask = cumulative_mask | mask

    ixts = img.shape[0]
    iyts = img.shape[1]
    # Set margin % here
    ixt = int(ixts * 0.15)
    iyt = int(iyts * 0.15)
    img_test = img.copy()
    img_test[ixt : ixts - ixt, iyt : iyts - iyt] = np.array(color_dict["neutral"])
    if np.any(_get_img_mask(img_test, color_dict["digging"])):
        return None, None, None

    metadata = {
        "real_dimensions": {"width": float(h), "height": float(w)},
        "axes_ABC": lines_abc,
        "lines_pts": lines_pts,
    }
    return img, cumulative_mask, metadata


def calculate_line_eq(pt1, pt2):
    """
    Calculates the coefficients of the line equation Ax + By + C = 0 given two points.

    Parameters:
    - pt1, pt2 (tuple): Points on the line.

    Returns:
    - A, B, C (float): Coefficients of the line equation.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    A = y2 - y1
    B = x1 - x2
    C = (x2 * y1) - (x1 * y2)
    return A, B, C


def add_obstacles_to_image(
    img, cumulative_mask, n_obs_min, n_obs_max, size_obstacle_min, size_obstacle_max
):
    """
    Adds obstacles to an image within specified parameters, ensuring they do not overlap with existing features.

    Parameters:
    - img (np.ndarray): The image array where obstacles will be added.
    - cumulative_mask (np.ndarray): A boolean mask indicating areas where obstacles or other features already exist.
    - n_obs_min (int): Minimum number of obstacles to add.
    - n_obs_max (int): Maximum number of obstacles to add.
    - size_obstacle_min (int): Minimum size of the obstacles.
    - size_obstacle_max (int): Maximum size of the obstacles.

    Returns:
    - np.ndarray: The updated image array with obstacles added.
    - np.ndarray: The updated cumulative mask including the new obstacles.
    """
    w, h = img.shape[:2]  # Extract width and height from the image dimensions
    n_occ = 0  # Initialize the count of obstacles added
    occ = (
        np.ones_like(img) * 255
    )  # Initialize an obstacle layer with the same dimensions as the input image
    n_obs_now = np.random.randint(
        n_obs_min, n_obs_max + 1
    )  # Randomly decide the number of obstacles to add

    while n_occ < n_obs_now:
        # Randomly determine the size of the obstacle
        sizeox = np.random.randint(size_obstacle_min, size_obstacle_max + 1)
        sizeoy = np.random.randint(size_obstacle_min, size_obstacle_max + 1)
        # Randomly select a position for the obstacle
        x = np.random.randint(0, w - sizeox)
        y = np.random.randint(0, h - sizeoy)
        # Check if the selected area overlaps with existing features
        if not cumulative_mask[x : x + sizeox, y : y + sizeoy].any():
            # Update the obstacle layer and the cumulative mask
            occ[x : x + sizeox, y : y + sizeoy] = np.array(color_dict["obstacle"])
            cumulative_mask[x : x + sizeox, y : y + sizeoy] = True
            n_occ += 1  # Increment the count of obstacles added

    return occ, cumulative_mask


def add_non_dumpables_to_image(
    img,
    occ,
    cumulative_mask,
    n_nodump_min,
    n_nodump_max,
    size_nodump_min,
    size_nodump_max,
):
    """
    Add non-dumpable but traversable objects to an image.

    Parameters:
    - img: numpy array representing the image to be modified.
    - occ: numpy array representing the occupancy grid.
    - cumulative_mask: numpy array that keeps track of all modifications to the image.
    - n_nodump_min: minimum number of non-dumpable objects to add.
    - n_nodump_max: maximum number of non-dumpable objects to add.
    - size_nodump_min: minimum size of non-dumpable objects.
    - size_nodump_max: maximum size of non-dumpable objects.

    Returns:
    - dmp: numpy array with non-dumpable objects added.
    - cumulative_mask: updated cumulative mask with non-dumpable objects added.
    """
    # Image dimensions
    w, h = img.shape[:2]
    # Number of nondumpable objects to add
    n_dmp_to_add = np.random.randint(n_nodump_min, n_nodump_max + 1)
    # Placeholder for non-dumpable objects
    dmp = np.ones_like(img) * 255
    # Track the number of added non-dumpable objects
    n_dmp_added = 0
    # Mask for occupied areas
    mask_occ = _get_img_mask(occ, color_dict["obstacle"])

    # Add non-dumpable objects until the target is reached
    while n_dmp_added < n_dmp_to_add:
        # Randomly choose the type of object (0: Square, 1: Road)
        dmp_type = np.random.randint(0, 2)
        # Square object
        if dmp_type == 0:
            sizeox, sizeoy = np.random.randint(
                size_nodump_min, size_nodump_max + 1, size=2
            )
            x, y = np.random.randint(0, w - sizeox, ()), np.random.randint(
                0, h - sizeoy, ()
            )
        # Road object
        else:
            road_direction = np.random.randint(0, 2)
            if road_direction == 0:  # Horizontal road
                sizeox, sizeoy = w, np.random.randint(
                    size_nodump_min, size_nodump_max + 1
                )
                x, y = 0, np.random.randint(0, h - sizeoy, ())
            else:  # Vertical road
                sizeox, sizeoy = (
                    np.random.randint(size_nodump_min, size_nodump_max + 1),
                    h,
                )
                x, y = np.random.randint(0, w - sizeox, ()), 0

        # Ensure the selected area is not already occupied or marked
        if np.all(cumulative_mask[x : x + sizeox, y : y + sizeoy] == 0) and np.all(
            mask_occ[x : x + sizeox, y : y + sizeoy] == 0
        ):
            dmp[x : x + sizeox, y : y + sizeoy] = (
                np.ones((3,)) * color_dict["nondumpable"]
            )
            n_dmp_added += 1

    return dmp, cumulative_mask


def save_or_display_image(img, occ, dmp, metadata, save_folder, i):
    # make dir if does not exist
    os.makedirs(save_folder, exist_ok=True)
    save_folder_images = Path(save_folder) / "images"
    save_folder_metadata = Path(save_folder) / "metadata"
    save_folder_occupancy = Path(save_folder) / "occupancy"
    save_folder_dumpability = Path(save_folder) / "dumpability"
    save_folder_images.mkdir(parents=True, exist_ok=True)
    save_folder_metadata.mkdir(parents=True, exist_ok=True)
    save_folder_occupancy.mkdir(parents=True, exist_ok=True)
    save_folder_dumpability.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_folder_images, "trench_" + str(i) + ".png"), img
    )  # Added .png extension
    cv2.imwrite(
        os.path.join(save_folder_occupancy, "trench_" + str(i) + ".png"), occ
    )  # Added .png extension
    cv2.imwrite(
        os.path.join(save_folder_dumpability, "trench_" + str(i) + ".png"), dmp
    )  # Added .png extension
    metadata_converted = convert_numpy(metadata)

    with open(
        os.path.join(save_folder_metadata, "trench_" + str(i) + ".json"), "w"
    ) as outfile:
        json.dump(metadata_converted, outfile)  # flipped convention


def generate_trenches_v2(
    n_imgs,
    img_edge_min,
    img_edge_max,
    sizes_small,
    sizes_long,
    n_edges,
    resolution,
    save_folder=None,
    n_obs_min=1,
    n_obs_max=3,
    size_obstacle_min=2,
    size_obstacle_max=8,
    n_nodump_min=1,
    n_nodump_max=3,
    size_nodump_min=2,
    size_nodump_max=8,
    add_obstacles=True,
    add_non_dumpables=True,
):
    min_edges, max_edges = n_edges
    i = 0
    while i < n_imgs:
        img = initialize_image(img_edge_min, img_edge_max, color_dict)
        img, cumulative_mask, metadata = generate_edges(
            img, (min_edges, max_edges), sizes_small, sizes_long, color_dict
        )
        if img is None:
            continue
        if add_obstacles:
            occ, cumulative_mask = add_obstacles_to_image(
                img,
                cumulative_mask,
                n_obs_min,
                n_obs_max,
                size_obstacle_min,
                size_obstacle_max,
            )
        if add_non_dumpables:
            dmp, cumulative_mask = add_non_dumpables_to_image(
                img,
                occ,
                cumulative_mask,
                n_nodump_min,
                n_nodump_max,
                size_nodump_min,
                size_nodump_max,
            )
        save_or_display_image(img, occ, dmp, metadata, save_folder, i)
        i += 1


def convert_terra_pad_to_color(img_terra_pad, color_dict):
    # Create an empty (N, N, 3) array filled with zeros
    img_color = np.zeros(
        (img_terra_pad.shape[0], img_terra_pad.shape[1], 3), dtype=np.uint8
    )

    # Mapping from terra_pad values to color_dict keys
    value_to_color = {-1: "digging", 1: "dumping"}

    # Iterate over the unique values in img_terra_pad (-1 and 1)
    for value in np.unique(img_terra_pad):
        # Get the corresponding color from color_dict
        color = color_dict[value_to_color[value]]
        # Apply the color to the positions where img_terra_pad equals the current value
        img_color[img_terra_pad == value] = color

    return img_color


def generate_foundations_v2(
    size,
    dataset_path,
    save_folder=None,
    n_obs_min=1,
    n_obs_max=3,
    size_obstacle_min=2,
    size_obstacle_max=8,
    n_nodump_min=1,
    n_nodump_max=3,
    size_nodump_min=2,
    size_nodump_max=8,
    expansion_factor=1,
    all_dumpable=False,
    copy_metadata=True,
    downsample=True,
    has_dumpability=False,
    center_padding=True,
):
    max_size = size
    # try:
    foundations_name = "foundations"
    img_folder = Path(dataset_path) / foundations_name / "images"
    metadata_folder = Path(dataset_path) / foundations_name / "metadata"
    occupancy_folder = Path(dataset_path) / foundations_name / "occupancy"
    dumpability_folder = Path(dataset_path) / foundations_name / "dumpability"
    filename_start = sorted(os.listdir(img_folder))[0].split("_")[0]

    for i, fn in enumerate(os.listdir(img_folder)):
        if i >= 1000:
            break

        n = int(fn.split(".png")[0].split("_")[1])
        filename = filename_start + f"_{n}.png"
        file_path = img_folder / filename

        occupancy_path = occupancy_folder / filename
        img = cv2.imread(str(file_path))

        occupancy = cv2.imread(str(occupancy_path))

        if has_dumpability:
            dumpability_path = dumpability_folder / filename
            dumpability = cv2.imread(str(dumpability_path))

        if downsample:
            with open(
                metadata_folder / f"{filename.split('.png')[0]}.json"
            ) as json_file:
                metadata = json.load(json_file)

            # Calculate downsample factors based on max_size
            downsample_factor_w = max(1, math.ceil(img.shape[1] / max_size)) * 2
            downsample_factor_h = max(1, math.ceil(img.shape[0] / max_size)) * 2

            img_downsampled = skimage.measure.block_reduce(
                img, (downsample_factor_h, downsample_factor_w, 1), np.max
            )
            img = img_downsampled
            occupancy_downsampled = skimage.measure.block_reduce(
                occupancy, (downsample_factor_h, downsample_factor_w, 1), np.min, cval=0
            )
            occupancy = occupancy_downsampled
            if has_dumpability:
                dumpability_downsampled = skimage.measure.block_reduce(
                    dumpability,
                    (downsample_factor_h, downsample_factor_w, 1),
                    np.min,
                    cval=0,
                )
                dumpability = dumpability_downsampled

        # assert img_downsampled.shape[:-1] == occupancy_downsampled.shape
        img_terra = _convert_img_to_terra(img, all_dumpable)

        # Pad to max size
        if center_padding:
            xdim = max_size - img_terra.shape[0]
            ydim = max_size - img_terra.shape[1]
            # Note: applying full dumping tiles for the centered version
            img_terra_pad = np.ones((max_size, max_size), dtype=img_terra.dtype)
            img_terra_pad[
                xdim // 2 : max_size - (xdim - xdim // 2),
                ydim // 2 : max_size - (ydim - ydim // 2),
            ] = img_terra
            # Note: applying no occupancy for the centered version (mismatch with Terra env)
            img_terra_occupancy = np.zeros((max_size, max_size), dtype=np.bool_)
            img_terra_occupancy[
                xdim // 2 : max_size - (xdim - xdim // 2),
                ydim // 2 : max_size - (ydim - ydim // 2),
            ] = _convert_occupancy_to_terra(occupancy)
            if has_dumpability:
                img_terra_dumpability = np.zeros((max_size, max_size), dtype=np.bool_)
                img_terra_dumpability[
                    xdim // 2 : max_size - (xdim - xdim // 2),
                    ydim // 2 : max_size - (ydim - ydim // 2),
                ] = _convert_dumpability_to_terra(dumpability)
        else:
            img_terra_pad = np.zeros((max_size, max_size), dtype=img_terra.dtype)
            img_terra_pad[: img_terra.shape[0], : img_terra.shape[1]] = img_terra
            img_terra_occupancy = np.ones((max_size, max_size), dtype=np.bool_)
            img_terra_occupancy[: occupancy.shape[0], : occupancy.shape[1]] = (
                _convert_occupancy_to_terra(occupancy)
            )
            if has_dumpability:
                img_terra_dumpability = np.zeros((max_size, max_size), dtype=np.bool_)
                img_terra_dumpability[
                    : dumpability.shape[0], : dumpability.shape[1]
                ] = _convert_dumpability_to_terra(dumpability)

        img_terra_pad = img_terra_pad.repeat(expansion_factor, 0).repeat(
            expansion_factor, 1
        )
        img_terra_pad = convert_terra_pad_to_color(img_terra_pad, color_dict)
        dumping_image = np.zeros(
            (img_terra_pad.shape[0], img_terra_pad.shape[1], 3), dtype=np.uint8
        )
        corner_dump = np.random.randint(0, 4)
        w, h = img_terra_pad.shape[:2]
        if corner_dump == 0:
            dumping_image[0 : int(0.8 * w), :, :] = np.array(color_dict["dumping"])
        elif corner_dump == 1:
            dumping_image[int(0.2 * w) :, :, :] = np.array(color_dict["dumping"])
        elif corner_dump == 2:
            dumping_image[:, int(0.2 * h) :, :] = np.array(color_dict["dumping"])
        elif corner_dump == 3:
            dumping_image[:, : int(0.8 * h), :] = np.array(color_dict["dumping"])
        # add dumping to the image where it's not equal to color_dict["digging"]

        # Create a mask where img_terra_pad is not equal to color_dict["digging"]
        mask = np.all(img_terra_pad != color_dict["digging"], axis=-1)

        # Use the mask to assign values from dumping_image to img_terra_pad
        img_terra_pad[mask] = dumping_image[mask]

        cumulative_mask = np.zeros_like(img_terra_pad, dtype=np.bool_)
        # where the img_terra_pad is [255, 255, 255] set to True across the three channels
        cumulative_mask[img_terra_pad == 255] = True
        occ, cumulative_mask = add_obstacles_to_image(
            img_terra_pad,
            cumulative_mask,
            n_obs_min,
            n_obs_max,
            size_obstacle_min,
            size_obstacle_max,
        )

        dmp, cumulative_mask = add_non_dumpables_to_image(
            img_terra_pad,
            occ,
            cumulative_mask,
            n_nodump_min,
            n_nodump_max,
            size_nodump_min,
            size_nodump_max,
        )
        save_folder = save_folder
        save_or_display_image(img_terra_pad, occ, dmp, metadata, save_folder, n)


def convert_numpy(obj):
    """
    Recursively convert numpy types in an object to their native Python equivalents.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy(value) for value in obj)
    else:
        return obj
