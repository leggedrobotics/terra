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
    Initializes an image array with dimensions within given range, applying a color scheme.

    Parameters:
    - img_edge_min, img_edge_max (int): Minimum and maximum edge sizes for the image.
    - color_dict (dict): A dictionary containing color mappings.

    Returns:
    - img (np.array): Initialized image array.
    - w, h (int): Width and height of the image.
    """
    # Randomly select dimensions within the specified range
    w, h = np.random.randint(img_edge_min, img_edge_max + 1, size=2, dtype=np.int32)
    img = np.ones((w, h, 3)) * np.array(color_dict["dumping"])
    return img


def generate_edges(img, edges_range, sizes_small, sizes_long, color_dict):
    min_ssmall, max_ssmall = sizes_small
    min_slong, max_slong = sizes_long
    min_edges, max_edges = edges_range
    n_edges = np.random.randint(min_edges, max_edges + 1)

    if n_edges == 0:
        # If no edges, return the original image and empty metadata,
        # but it likely won't pass the margin check later.
        # Consider what an appropriate empty/neutral state should be.
        # For now, let's ensure it returns something that doesn't break downstream.
        metadata = {
            "real_dimensions": {"width": float(img.shape[1]), "height": float(img.shape[0])},
            "axes_ABC": [],
            "lines_pts": [],
        }
        return img.copy(), np.zeros_like(img[..., 0], dtype=bool), metadata


    img_rows, img_cols = img.shape[0], img.shape[1]
    lines_abc = [] # Renamed from lines_abc_list
    lines_pts = [] # Renamed from lines_pts_list
    
    # Stores details of generated edges:
    # {'axis_pt1': (col,row), 'axis_pt2': (col,row), 'is_horizontal': bool,
    #  'rect_origin_row': int, 'rect_origin_col': int,
    #  'rect_height_rows': int, 'rect_width_cols': int}
    generated_edge_details = []
    
    cumulative_mask = np.zeros_like(img[..., 0], dtype=bool)

    for edge_i in range(n_edges):
        current_size_small = np.random.randint(min_ssmall, max_ssmall + 1)
        current_size_long = np.random.randint(min_slong, max_slong + 1)

        # Ensure sizes are at least 1
        current_size_small = max(1, current_size_small)
        current_size_long = max(1, current_size_long)

        new_edge_is_horizontal = False
        axis_pt1, axis_pt2 = None, None
        
        # Declare rect properties that will be defined in if/else
        rect_origin_row, rect_origin_col = 0, 0
        rect_height_rows, rect_width_cols = 0, 0


        if not generated_edge_details: # First edge
            new_edge_is_horizontal = random.choice([True, False])
            
            ideal_rect_height_rows, ideal_rect_width_cols = \
                (current_size_small, current_size_long) if new_edge_is_horizontal else \
                (current_size_long, current_size_small)

            # Max starting row/col ensures the ideal rect *could* start there
            # It doesn't guarantee it fits, that's handled by clipping.
            # Ensure upper bound of randint is at least 0.
            max_start_row = max(0, img_rows - ideal_rect_height_rows)
            max_start_col = max(0, img_cols - ideal_rect_width_cols)
            
            # If image is smaller than ideal rect, start at 0,0
            start_row_candidate = np.random.randint(0, max_start_row + 1) if max_start_row >=0 else 0
            start_col_candidate = np.random.randint(0, max_start_col + 1) if max_start_col >=0 else 0
            
            rect_origin_row = start_row_candidate
            rect_origin_col = start_col_candidate

            rect_height_rows = min(ideal_rect_height_rows, img_rows - rect_origin_row)
            rect_width_cols = min(ideal_rect_width_cols, img_cols - rect_origin_col)

            if rect_height_rows <= 0 or rect_width_cols <= 0:
                # First edge couldn't be placed (e.g. image too small for min_sizes)
                return None, None, None 

            if new_edge_is_horizontal:
                axis_row_coord = rect_origin_row + (rect_height_rows - 1) / 2.0
                axis_pt1 = (float(rect_origin_col), axis_row_coord)
                axis_pt2 = (float(rect_origin_col + rect_width_cols - 1), axis_row_coord)
            else: # Vertical
                axis_col_coord = rect_origin_col + (rect_width_cols - 1) / 2.0
                axis_pt1 = (axis_col_coord, float(rect_origin_row))
                axis_pt2 = (axis_col_coord, float(rect_origin_row + rect_height_rows - 1))
            
        else: # Subsequent edges
            if not generated_edge_details: # Should not happen if n_edges > 0
                 return None, None, None # Safety break

            parent_edge = random.choice(generated_edge_details)
            new_edge_is_horizontal = not parent_edge['is_horizontal']

            parent_ax_p1_col, parent_ax_p1_row = parent_edge['axis_pt1']
            parent_ax_p2_col, parent_ax_p2_row = parent_edge['axis_pt2']

            attach_choice = random.randint(0, 2) # 0: midpoint, 1: endpoint1, 2: endpoint2
            if attach_choice == 0:
                attach_col = (parent_ax_p1_col + parent_ax_p2_col) / 2.0
                attach_row = (parent_ax_p1_row + parent_ax_p2_row) / 2.0
            elif attach_choice == 1:
                attach_col, attach_row = parent_ax_p1_col, parent_ax_p1_row
            else:
                attach_col, attach_row = parent_ax_p2_col, parent_ax_p2_row

            ideal_rect_height_rows, ideal_rect_width_cols = \
                (current_size_small, current_size_long) if new_edge_is_horizontal else \
                (current_size_long, current_size_small)
            
            extension_direction = random.choice([-1, 1])

            # Calculate initial unclipped rectangle origin and axis points
            unclipped_rect_origin_row, unclipped_rect_origin_col = 0,0
            unclipped_axis_pt1, unclipped_axis_pt2 = (0,0),(0,0)

            if new_edge_is_horizontal:
                # New edge's axis is horizontal, passing through attach_row
                new_axis_row_coord = attach_row
                unclipped_rect_origin_row = round(new_axis_row_coord - (ideal_rect_height_rows - 1) / 2.0)
                
                if extension_direction == 1: # Extends "positively" (right for horizontal)
                    unclipped_rect_origin_col = round(attach_col)
                    unclipped_axis_pt1 = (float(attach_col), new_axis_row_coord)
                    unclipped_axis_pt2 = (float(attach_col + ideal_rect_width_cols - 1), new_axis_row_coord)
                else: # Extends "negatively" (left for horizontal)
                    unclipped_rect_origin_col = round(attach_col - (ideal_rect_width_cols - 1))
                    unclipped_axis_pt1 = (float(attach_col - (ideal_rect_width_cols - 1)), new_axis_row_coord)
                    unclipped_axis_pt2 = (float(attach_col), new_axis_row_coord)
            else: # New edge is vertical
                # New edge's axis is vertical, passing through attach_col
                new_axis_col_coord = attach_col
                unclipped_rect_origin_col = round(new_axis_col_coord - (ideal_rect_width_cols - 1) / 2.0)

                if extension_direction == 1: # Extends "positively" (down for vertical)
                    unclipped_rect_origin_row = round(attach_row)
                    unclipped_axis_pt1 = (new_axis_col_coord, float(attach_row))
                    unclipped_axis_pt2 = (new_axis_col_coord, float(attach_row + ideal_rect_height_rows - 1))
                else: # Extends "negatively" (up for vertical)
                    unclipped_rect_origin_row = round(attach_row - (ideal_rect_height_rows - 1))
                    unclipped_axis_pt1 = (new_axis_col_coord, float(attach_row - (ideal_rect_height_rows - 1)))
                    unclipped_axis_pt2 = (new_axis_col_coord, float(attach_row))

            # Clip to image boundaries
            rect_origin_row = int(max(0, unclipped_rect_origin_row))
            rect_origin_col = int(max(0, unclipped_rect_origin_col))
            
            rect_height_rows = min(ideal_rect_height_rows, img_rows - rect_origin_row)
            rect_width_cols = min(ideal_rect_width_cols, img_cols - rect_origin_col)
            
            # Ensure dimensions are positive after clipping
            rect_height_rows = max(0, rect_height_rows)
            rect_width_cols = max(0, rect_width_cols)

            if rect_height_rows <= 0 or rect_width_cols <= 0:
                continue # Skip this edge if it cannot be placed

            # Recalculate axis points for the *actual* (clipped) rectangle
            if new_edge_is_horizontal:
                final_axis_row = rect_origin_row + (rect_height_rows - 1) / 2.0
                # The axis starts at the rect_origin_col and spans rect_width_cols
                axis_pt1 = (float(rect_origin_col), final_axis_row)
                axis_pt2 = (float(rect_origin_col + rect_width_cols - 1), final_axis_row)
            else: # New edge is vertical
                final_axis_col = rect_origin_col + (rect_width_cols - 1) / 2.0
                axis_pt1 = (final_axis_col, float(rect_origin_row))
                axis_pt2 = (final_axis_col, float(rect_origin_row + rect_height_rows - 1))

        # Ensure axis points are ordered (pt1's coord <= pt2's coord) for consistency
        if axis_pt1[0] > axis_pt2[0] or axis_pt1[1] > axis_pt2[1]:
            axis_pt1, axis_pt2 = axis_pt2, axis_pt1
        
        # Draw the edge
        # Ensure rect_origin and dimensions are integers for slicing
        final_rect_origin_row = int(rect_origin_row)
        final_rect_origin_col = int(rect_origin_col)
        final_rect_height = int(rect_height_rows)
        final_rect_width = int(rect_width_cols)

        if final_rect_height <= 0 or final_rect_width <= 0: # Should be caught by earlier continue
            continue

        img[
            final_rect_origin_row : final_rect_origin_row + final_rect_height,
            final_rect_origin_col : final_rect_origin_col + final_rect_width,
        ] = np.array(color_dict["digging"])

        A_coeff = axis_pt2[1] - axis_pt1[1]  # row2 - row1
        B_coeff = axis_pt1[0] - axis_pt2[0]  # col1 - col2
        C_coeff = axis_pt2[0] * axis_pt1[1] - axis_pt1[0] * axis_pt2[1] # col2*row1 - col1*row2
        lines_abc.append({"A": float(A_coeff), "B": float(B_coeff), "C": float(C_coeff)})
        lines_pts.append([axis_pt1, axis_pt2])

        mask_for_edge = np.ones((final_rect_height, final_rect_width), dtype=bool)
        cumulative_mask[
            final_rect_origin_row : final_rect_origin_row + final_rect_height,
            final_rect_origin_col : final_rect_origin_col + final_rect_width,
        ] = cumulative_mask[
            final_rect_origin_row : final_rect_origin_row + final_rect_height,
            final_rect_origin_col : final_rect_origin_col + final_rect_width,
        ] | mask_for_edge

        generated_edge_details.append({
            'axis_pt1': axis_pt1, 'axis_pt2': axis_pt2,
            'is_horizontal': new_edge_is_horizontal,
            'rect_origin_row': final_rect_origin_row,
            'rect_origin_col': final_rect_origin_col,
            'rect_height_rows': final_rect_height,
            'rect_width_cols': final_rect_width
        })

    if not generated_edge_details and n_edges > 0 : # All edges failed to be placed
        return None, None, None

    # Margin check (same as original)
    ixts = img.shape[0]
    iyts = img.shape[1]
    ixt = int(ixts * 0.15)
    iyt = int(iyts * 0.15)
    img_test = img.copy()
    
    neutral_color_arr = np.array(color_dict["neutral"])
    # Ensure the slice for neutral color is valid
    slice_row_start, slice_row_end = ixt, ixts - ixt
    slice_col_start, slice_col_end = iyt, iyts - iyt

    if slice_row_start < slice_row_end and slice_col_start < slice_col_end:
        if img_test.ndim == 3 and neutral_color_arr.ndim == 1 and neutral_color_arr.shape[0] == img_test.shape[2]:
            img_test[slice_row_start:slice_row_end, slice_col_start:slice_col_end, :] = neutral_color_arr
        elif img_test.ndim == neutral_color_arr.ndim: # e.g. grayscale or already matching
             img_test[slice_row_start:slice_row_end, slice_col_start:slice_col_end] = neutral_color_arr
        # Add more sophisticated handling if needed, or ensure color_dict["neutral"] is always appropriate
    
    if np.any(_get_img_mask(img_test, color_dict["digging"])):
        return None, None, None

    metadata = {
        "real_dimensions": {"width": float(img_cols), "height": float(img_rows)},
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
