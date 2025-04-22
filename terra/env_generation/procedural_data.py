import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import random

from terra.env_generation.convert_to_terra import (
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
        img[0 : int(0.75 * w), :, :] = np.array(color_dict["dumping"])
    elif corner_dump == 1:
        img[int(0.25 * w) :, :, :] = np.array(color_dict["dumping"])
    elif corner_dump == 2:
        img[:, int(0.25 * h) :, :] = np.array(color_dict["dumping"])
    elif corner_dump == 3:
        img[:, : int(0.75 * h), :] = np.array(color_dict["dumping"])

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
        lines_abc.append({"A": float(A), "B": float(B), "C": float(C)})

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


def generate_diagonal_edges(img, edges_range, sizes_small, sizes_long, color_dict):
    """
    Wrapper function to generate trenches using generate_edges() and optionally rotate.

    Args:
        img (np.ndarray): Base image.
        edges_range, sizes_small, sizes_long: Parameters for generate_edges.
        color_dict (dict): Color dictionary including 'neutral' and 'digging'.

    Returns:
        tuple: (final_img, final_mask, final_metadata) or (None, None, None)
               Metadata includes rotation_angle (0.0 if not rotated or angle was 0).
    """

    img_background = img.copy()
    h, w = img_background.shape[:2]

    # Call generate_edges on a throwaway copy to get the mask and metadata
    # The returned image 'img_generated_with_bg' won't be used directly for rotation
    img_generated_with_bg, mask_generated, metadata_generated = generate_edges(
        img_background.copy(), # Pass a copy so original isn't modified here
        edges_range,
        sizes_small,
        sizes_long,
        color_dict
    )

    # Check if the original generation succeeded
    if img_generated_with_bg is None:
        return None, None, None

    # --- Isolate the orthogonal trenches onto a new layer ---
    # Start with a neutral layer
    trench_layer_ortho = np.ones_like(img_background) * np.array(color_dict["neutral"])
    # Where the mask is true, put the digging color
    trench_layer_ortho[mask_generated] = np.array(color_dict["digging"])

    # --- Rotation ---
    possible_angles = np.arange(30, 360, 30)
    angle = random.choice(possible_angles)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate ONLY the trench layer
    border_val_neutral = tuple(map(int, color_dict["neutral"]))
    rotated_trench_layer = cv2.warpAffine(trench_layer_ortho, M, (w, h),
                                          flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=border_val_neutral)

    # Rotate the MASK
    mask_uint8 = mask_generated.astype(np.uint8)
    rotated_mask_interpolated = cv2.warpAffine(mask_uint8, M, (w, h),
                                               flags=cv2.INTER_NEAREST,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=0)
    rotated_mask = rotated_mask_interpolated > 0

    # --- Rotate Metadata Points ---
    rotated_lines_pts = []
    rotated_lines_abc = []
    final_metadata = metadata_generated.copy()

    if 'lines_pts' in metadata_generated:
        center_w, center_h = center
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        for pt1_orig, pt2_orig in metadata_generated['lines_pts']:
            pt1_col, pt1_row = pt1_orig
            pt2_col, pt2_row = pt2_orig

            # Rotate pt1
            temp_col1 = pt1_col - center_w
            temp_row1 = pt1_row - center_h
            new_pt1_col = temp_col1 * cos_a - temp_row1 * sin_a + center_w
            new_pt1_row = temp_col1 * sin_a + temp_row1 * cos_a + center_h
            new_pt1 = (float(new_pt1_col), float(new_pt1_row))

            # Point 2
            temp_col2 = pt2_col - center_w
            temp_row2 = pt2_row - center_h
            new_pt2_col = temp_col2 * cos_a - temp_row2 * sin_a + center_w
            new_pt2_row = temp_col2 * sin_a + temp_row2 * cos_a + center_h
            new_pt2 = (float(new_pt2_col), float(new_pt2_row))

            rotated_lines_pts.append([new_pt1, new_pt2])
            # Recalculate A, B, C
            A = new_pt2[1] - new_pt1[1]
            B = new_pt1[0] - new_pt2[0]
            C = new_pt2[0] * new_pt1[1] - new_pt1[0] * new_pt2[1]
            rotated_lines_abc.append({'A':float(A),'B':float(B),'C':float(C)})

        final_metadata["axes_ABC"] = rotated_lines_abc
        final_metadata["lines_pts"] = rotated_lines_pts


    # --- Combine original background with rotated trenches ---
    # Find where the rotated trench layer has the digging color
    final_trench_mask = ~np.all(rotated_trench_layer == np.array(color_dict["neutral"]), axis=-1)
    # Create the final image starting from the original background
    final_img = img_background.copy()
    # Paste the rotated digging color pixels onto the original background
    final_img[final_trench_mask] = np.array(color_dict["digging"])

    return final_img, rotated_mask, final_metadata


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


def add_obstacles(
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
    h, w = img.shape[:2] # Use height (rows), width (cols)
    n_occ = 0
    # Initialize obstacle layer with white background (as per original function)
    occ_layer = np.ones_like(img, dtype=np.uint8) * 255
    obstacle_color = np.array(color_dict["obstacle"])
    # Make a copy of cumulative_mask to update and return
    updated_cumulative_mask = cumulative_mask.copy()

    n_obs_now = np.random.randint(n_obs_min, n_obs_max + 1)
    attempts = 0
    max_attempts = n_obs_now * 50

    while n_occ < n_obs_now:
        # Choose size and center position
        size_h = np.random.randint(size_obstacle_min, size_obstacle_max + 1)
        size_w = np.random.randint(size_obstacle_min, size_obstacle_max + 1)
        # Ensure center is chosen such that even a rotated box has a chance to fit
        margin_h = int(np.ceil(np.sqrt(size_h ** 2 + size_w ** 2) / 2)) + 1
        margin_w = margin_h
        if h <= 2 * margin_h or w <= 2 * margin_w: continue

        center_x = np.random.randint(margin_h, h - margin_h)
        center_y = np.random.randint(margin_w, w - margin_w)

        # Choose rotation angle
        angle_deg = np.random.choice(np.arange(0, 360, 30))
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Define corners relative to center (0,0)
        half_w = size_w / 2.0
        half_h = size_h / 2.0
        local_corners = np.array([
            [-half_w, -half_h], [ half_w, -half_h],
            [ half_w,  half_h], [-half_w,  half_h]
        ])

        # Rotate corners
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_local_corners = local_corners @ R.T

        # Translate corners to global center and make integer
        global_corners = rotated_local_corners + np.array([center_y, center_x])
        # Ensure corners are within bounds before creating mask
        global_corners[:, 0] = np.clip(global_corners[:, 0], 0, w - 1)
        global_corners[:, 1] = np.clip(global_corners[:, 1], 0, h - 1)
        global_corners_int = global_corners.astype(np.int32)

        # Create mask for the proposed obstacle
        proposed_mask = np.zeros((h, w), dtype=np.uint8)
        # cv2.fillPoly requires list of arrays, points should be (col, row) format
        cv2.fillPoly(proposed_mask, [global_corners_int], 1)
        proposed_mask_bool = proposed_mask.astype(bool)

        # Check collision using the proposed mask
        cumulative_mask_2d_view = np.any(updated_cumulative_mask, axis=-1)
        if not np.any(cumulative_mask_2d_view & proposed_mask_bool):
            # Draw the rotated obstacle on the 'occ' layer
            # cv2.fillPoly modifies the array in place
            cv2.fillPoly(occ_layer, [global_corners_int], tuple(map(int, obstacle_color))) # Use tuple for color
            updated_cumulative_mask |= proposed_mask_bool
            n_occ += 1

    # Return the layer with obstacles on white, and the updated mask
    return occ_layer, updated_cumulative_mask


def add_non_dumpables(
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
    diagonal=False,
    should_add_obstacles=True,
    should_add_non_dumpables=True,
):
    min_edges, max_edges = n_edges
    i = 0
    while i < n_imgs:
        print(f"Processing trench nr {i + 1}")
        img = initialize_image(img_edge_min, img_edge_max, color_dict)
        if diagonal:
            img, cumulative_mask, metadata = generate_diagonal_edges(
                img, (min_edges, max_edges), sizes_small, sizes_long, color_dict
            )
        else:
            img, cumulative_mask, metadata = generate_edges(
                img, (min_edges, max_edges), sizes_small, sizes_long, color_dict
            )
        if img is None:
            continue
        if should_add_obstacles:
            occ, cumulative_mask = add_obstacles(
                img,
                cumulative_mask,
                n_obs_min,
                n_obs_max,
                size_obstacle_min,
                size_obstacle_max,
            )
        else:
            # Initialize occ with default values if obstacles aren't added
            occ = np.ones_like(img) * 255

        if should_add_non_dumpables:
            dmp, cumulative_mask = add_non_dumpables(
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
