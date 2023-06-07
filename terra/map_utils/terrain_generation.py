"""
This file contains a set of function the make use of opencv to create a set of geometric primitives that can be used to
create a map.
These primitives include triangles, trapezoids, rectangles, pentagons, hexagons, L shape, C (or U) shapes.
Use open These objects can either represent obstacles to be avoided or areas to be dug out by the robot.
Each shape can be randomly placed on the map and intersections are allowed, more complex way of placing them is
welcome.
The map is starts out as a blank image and each shape is drawn on the image.
Let's use a dictionary to specify the number of shapes and their type.
The dictionary will be of the form:
    map_dict = {
        'num_shapes': 10,
        'shapes': {
            'triangle': 2,
            'trapezoid': 2,
            'rectangle': 2,
            'pentagon': 2,
            'hexagon': 2
        }
        dimensions: (600, 400)
    }
The map_dict will be passed to the function that will generate the map.
The function will return the map as a numpy array.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from cv2 import convexHull
from cv2 import fillPoly
from cv2 import polylines
from shapely.geometry import Polygon

# from skimage.measure import block_reduce


BORDER_COLOR = (255, 255, 255)


def _get_search_bounds(vertices, max_edge, w, h):
    if isinstance(vertices, list):
        if len(vertices) == 0:
            minw = 0
            maxw = w
            minh = 0
            maxh = h
        elif len(vertices) >= 1:
            minw_a = []
            maxw_a = []
            minh_a = []
            maxh_a = []
            for v in vertices:
                minw_a.append(max(0, v[0] - max_edge))
                maxw_a.append(min(w, v[0] + max_edge))
                minh_a.append(max(0, v[1] - max_edge))
                maxh_a.append(min(w, v[1] + max_edge))
            minw = min(minw_a)
            maxw = max(maxw_a)
            minh = min(minh_a)
            maxh = max(maxh_a)
    else:
        minw = jnp.min(jnp.clip(vertices[..., 0] - max_edge, a_min=0))
        minh = jnp.min(jnp.clip(vertices[..., 1] - max_edge, a_min=0))
        maxw = jnp.max(jnp.clip(vertices[..., 0] + max_edge, a_max=w))
        maxh = jnp.max(jnp.clip(vertices[..., 1] + max_edge, a_max=h))
    return minw, maxw, minh, maxh


def _loop(i, value):
    key, vertices, xy_set, xy_set_mask, min_edge, max_edge = value
    key, subkey = jax.random.split(key)
    valid_numbers = xy_set_mask.sum()
    p = xy_set_mask * (1.0 / valid_numbers)
    xy_idx = jax.random.choice(subkey, jnp.arange(0, xy_set_mask.shape[0]), p=p)
    xy = xy_set[xy_idx]
    vertices = vertices.at[i].set(xy)

    # Mask out unavailable choices in sets
    xy_norm = jnp.linalg.norm(xy_set - vertices[:, None], axis=-1)
    xy_set_mask = (xy_norm >= min_edge) * (xy_norm <= max_edge)
    # Exclude placeholder rows
    xy_set_mask = jnp.where(
        jnp.arange(vertices.shape[0])[:, None].repeat(xy_set.shape[0], 1) > i,
        1,
        xy_set_mask,
    )
    xy_set_mask = xy_set_mask.prod(axis=0)
    value = key, vertices, xy_set, xy_set_mask.astype(jnp.bool_), min_edge, max_edge
    return value


@partial(jax.jit, static_argnums=(1, 2, 3))
def _get_vertices_polygon(key, n_sides, w, h, min_edge, max_edge, image_mask):
    x_set = jnp.arange(0, w)[None].repeat(h, axis=-2).reshape(-1)
    y_set = jnp.arange(0, h)[:, None].repeat(w, axis=-1).reshape(-1)
    xy_set = jnp.concatenate((x_set[..., None], y_set[..., None]), axis=-1)
    xy_set_mask = image_mask
    vertices = jnp.empty((n_sides, 2))

    value = key, vertices, xy_set, xy_set_mask, min_edge, max_edge
    value = jax.lax.fori_loop(lower=0, upper=n_sides, body_fun=_loop, init_val=value)
    key, vertices, xy_set, xy_set_mask, min_edge, max_edge = value

    # close the shape
    vertices = jnp.array(vertices)
    vertices = jnp.concatenate((vertices, vertices[0][None]), axis=0)
    return vertices, key


def _get_vertices_polygon_numpy(key, n_sides, w, h, min_edge, max_edge, image_mask):
    min_edge = min_edge.item() if not isinstance(min_edge, int) else min_edge
    max_edge = max_edge.item() if not isinstance(max_edge, int) else max_edge
    vertices, key = _get_vertices_polygon(
        key, n_sides, w, h, min_edge, max_edge, image_mask
    )
    return np.array(vertices), key


def get_triangle(key, image, image_mask, min_edge, max_edge, min_area, min_angle=1):
    n_sides = 3
    h, w = image.shape[:2]
    while True:
        vertices, key = _get_vertices_polygon_numpy(
            key, n_sides, w, h, min_edge, max_edge, image_mask
        )
        area = get_shape_area(vertices)
        if area >= min_area:
            break
    return vertices, key


def get_shape_area(vertices):
    polygon = Polygon(vertices)
    return polygon.area


def get_trapezoid(key, image, min_edge, max_edge, min_area, min_angle=30):
    h, w = image.shape[:2]
    while True:
        vertices = []
        for _ in range(4):
            while True:
                minw, maxw, minh, maxh = _get_search_bounds(vertices, max_edge, w, h)
                key, subkey = jax.random.split(key)
                x = jax.random.randint(subkey, (1,), minval=minw, maxval=maxw)
                key, subkey = jax.random.split(key)
                y = jax.random.randint(subkey, (1,), minval=minh, maxval=maxh)
                # Calculate the distances to the other vertices
                xy = jnp.concatenate((x, y), axis=-1)
                distances = jnp.array([jnp.linalg.norm(xy - v) for v in vertices])
                # Check if the distances are within the desired range
                if jnp.all((distances >= min_edge) * (distances <= max_edge)):
                    # Check the angle between all pairs of edges
                    for i in range(len(vertices)):
                        v1 = jnp.array(vertices[i - 1]) - jnp.array(vertices[i])
                        v2 = jnp.concatenate((x, y), axis=-1) - jnp.array(vertices[i])
                        cosine_angle = jnp.dot(v1, v2) / (
                            jnp.linalg.norm(v1) * jnp.linalg.norm(v2)
                        )
                        # cosine_angle = np.clip(cosine_angle, -1, 1)
                        angle = jnp.arccos(cosine_angle) * 180 / jnp.pi

                        if angle < min_angle or angle > 0.9 * 180 or angle is jnp.nan:
                            break
                    else:
                        vertices.append(jnp.concatenate((x, y), axis=-1))
                        break

        vertices = jnp.array(vertices)

        # Calculate the centroid of the vertices
        centroid = jnp.mean(vertices, axis=0)

        # Sort the vertices based on their angle from the centroid
        angles = jnp.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        vertices = vertices[jnp.argsort(angles)]

        # Calculate the area of the trapezoid
        vertices = np.array(vertices)
        area = get_shape_area(vertices)
        if area >= min_area:
            return vertices, key


def get_rectangle(key, image, min_edge, max_edge, min_area):
    h, w = image.shape[:2]
    while True:
        key, *subkeys = jax.random.split(key, 3)
        pt1 = jnp.concatenate(
            (
                jax.random.randint(subkeys[0], (1,), minval=0, maxval=w),
                jax.random.randint(subkeys[1], (1,), minval=0, maxval=h),
            ),
            axis=-1,
        )
        key, *subkeys = jax.random.split(key, 3)
        pt2 = jnp.concatenate(
            (
                jax.random.randint(subkeys[0], (1,), minval=0, maxval=w),
                jax.random.randint(subkeys[1], (1,), minval=0, maxval=h),
            ),
            axis=-1,
        )
        # Calculate the edge lengths
        edge_lengths = jnp.array([jnp.abs(pt1[0] - pt2[0]), jnp.abs(pt1[1] - pt2[1])])
        # Check if the edge lengths are within the desired range
        if jnp.all((edge_lengths >= min_edge) * (edge_lengths <= max_edge)):
            # cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)

            # Fill the rectangle with blue color
            vertices = jnp.array([pt1, (pt1[0], pt2[1]), pt2, (pt2[0], pt1[1])])

            # Calculate the area of the rectangle
            area = get_shape_area(vertices)
            if area >= min_area:
                return np.array(vertices), key


def get_pentagon(key, image, image_mask, min_edge, max_edge, min_area):
    h, w = image.shape[:2]
    n_sides = 5
    while True:
        vertices, key = _get_vertices_polygon_numpy(
            key, n_sides, w, h, min_edge, max_edge, image_mask
        )

        # Calculate the centroid of the vertices
        centroid = jnp.mean(vertices, axis=0)

        # Sort the vertices based on their angle with the centroid
        angles = jnp.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        sorted_indices = jnp.argsort(angles)
        ordered_vertices = vertices[sorted_indices]

        # Calculate the area of the pentagon
        vertices = np.array(vertices)
        area = get_shape_area(vertices)
        if area >= min_area:
            return ordered_vertices, key


def get_hexagon(key, image, image_mask, min_edge, max_edge, min_area):
    h, w = image.shape[:2]
    n_sides = 6
    while True:
        vertices, key = _get_vertices_polygon_numpy(
            key, n_sides, w, h, min_edge, max_edge, image_mask
        )

        # Calculate the centroid of the vertices
        centroid = jnp.mean(vertices, axis=0)

        # Sort the vertices based on their angle with the centroid
        angles = jnp.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        sorted_indices = jnp.argsort(angles)
        ordered_vertices = vertices[sorted_indices]

        # Calculate the area of the hexagon
        vertices = np.array(vertices)
        area = get_shape_area(ordered_vertices)
        # print(area)
        if area >= min_area:
            return ordered_vertices, key


def draw_regular_polygon(key, image, num_sides: list[int], radius: int):
    """
    Creates a regular polygon with the specified number of sides.
    :param image: The image to draw the polygon on.
    :param num_sides: list of possible number of sides of the polygon.
    :param min_edge: The minimum length of the edges of the polygon.
    :param max_edge: The maximum length of the edges of the polygon.
    :param min_area: The minimum area of the polygon.

    :return: The vertices of the polygon.
    """
    # sample the number of sides
    key, subkey = jax.random.split(key)
    num_sides = jax.random.choice(subkey, num_sides)
    # center is randomly chosen but not too close to the edges (radius)
    h, w = image.shape[:2]
    key, *subkeys = jax.random.split(key, 3)
    center = jnp.concatenate(
        (
            jax.random.randint(subkeys[0], (1,), minval=radius, maxval=w - radius),
            jax.random.randint(subkeys[1], (1,), minval=radius, maxval=h - radius),
        ),
        axis=-1,
    )

    vertices = jnp.array(
        [
            [
                center[0] + radius * jnp.cos(2 * jnp.pi * i / num_sides),
                center[1] + radius * jnp.sin(2 * jnp.pi * i / num_sides),
            ]
            for i in range(num_sides)
        ],
        dtype=jnp.int32,
    )
    vertices = np.array(vertices)
    return vertices, key


def draw_L(key, image, min_edge, max_edge, min_area):
    """
    Creates a L shape by creating two intersecting rectangles. we use the function
    get_rectangle to create the rectangles but we ensure that the rectangles
    intersect.
    """
    # generate the first rectangle
    vertices, key = get_rectangle(key, image, min_edge, max_edge, min_area)
    vertices2 = []
    while True:
        # generate the second rectangle
        vertices2, key = get_rectangle(key, image, min_edge, max_edge, min_area)
        # check if the rectangles intersect
        if find_intersection_vertices(vertices, vertices2):
            break
    # get the exclusion of the intersection from the first rectangle
    vertices = get_excluded_shape(vertices, vertices2)
    # list to np array
    vertices = np.array(vertices).astype(np.int64)
    return vertices, key


def draw_Z(key, image, min_edge, max_edge, min_area):
    # generate the first rectangle
    vertices, key = get_rectangle(key, image, min_edge, max_edge, min_area)
    vertices2 = []
    while True:
        # generate the second rectangle
        vertices2, key = get_rectangle(key, image, min_edge, max_edge, min_area)
        # check if the rectangles intersect
        if find_intersection_vertices(vertices, vertices2):
            break
    # get the exclusion of the intersection from the first rectangle
    vertices = get_union(vertices, vertices2)
    # list to np array
    vertices = np.array(vertices).astype(np.int64)
    return vertices, key


def draw_O(key, image, min_edge, max_edge):
    h, w = image.shape[:2]
    # Ensure the size of the C shape
    while True:
        key, *subkeys = jax.random.split(key, 9)
        pt1 = jnp.concatenate(
            (
                jax.random.randint(subkeys[0], (1,), minval=0, maxval=w - max_edge),
                jax.random.randint(subkeys[1], (1,), minval=0, maxval=h - max_edge),
            ),
            axis=-1,
        )
        pt2 = jnp.concatenate(
            (
                pt1[0]
                + jax.random.randint(
                    subkeys[2], (1,), minval=min_edge, maxval=max_edge
                ),
                pt1[1]
                + jax.random.randint(
                    subkeys[3], (1,), minval=min_edge, maxval=max_edge
                ),
            ),
            axis=-1,
        )

        inner_width = jax.random.randint(
            subkeys[4], (1,), minval=min_edge, maxval=pt2[0] - pt1[0] - 1
        )
        inner_height = jax.random.randint(
            subkeys[5], (1,), minval=min_edge, maxval=pt2[1] - pt1[1] - 1
        )
        inner_pt1 = jnp.concatenate(
            (
                jax.random.randint(
                    subkeys[6], (1,), minval=pt1[0] + 1, maxval=pt2[0] - inner_width
                ),
                jax.random.randint(
                    subkeys[7], (1,), minval=pt1[1] + 1, maxval=pt2[1] - inner_height
                ),
            ),
            axis=-1,
        )

        inner_pt2 = jnp.concatenate(
            (inner_pt1[0] + inner_width, inner_pt1[1] + inner_height), axis=-1
        )

        if (
            pt1[0] < inner_pt1[0] < pt2[0]
            and pt1[1] < inner_pt1[1] < pt2[1]
            and pt1[0] < inner_pt2[0] < pt2[0]
            and pt1[1] < inner_pt2[1] < pt2[1]
        ):
            break
    # cv2.rectangle(image, pt1, pt2, (0, 0, 0), -1)
    # cv2.rectangle(image, inner_pt1, inner_pt2, (255, 255, 255), -1)
    return image, key


def draw_shape(
    image,
    vertices,
    color=(0, 0, 0),
    countour_color=(255, 255, 255),
    countour_thickness=0,
):
    fillPoly(image, [vertices], color=color)
    # drow countour in grey
    polylines(image, [vertices], True, countour_color, countour_thickness)
    return image


def scale_shape(vertices, scale_factor=1.0):
    hull = convexHull(vertices)
    centroid = np.mean(hull, axis=0)
    scaled_vertices = (vertices - centroid) * scale_factor + centroid
    # make sure they are uint8
    scaled_vertices = np.round(scaled_vertices).astype(int)

    # Find the minimum and maximum values for each axis
    min_values = np.min(vertices, axis=0)
    max_values = np.max(vertices, axis=0)

    # Check if any of the scaled vertices fall outside the bounds
    outside_indices = np.any(
        (scaled_vertices < min_values) | (scaled_vertices > max_values), axis=1
    )

    if np.any(outside_indices):
        # If any vertices are outside the bounds, shrink the shape until it fits
        shrink_factor = 1.0
        while np.any(outside_indices):
            shrink_factor -= 0.01
            scaled_vertices = (vertices - centroid) * shrink_factor + centroid
            scaled_vertices = np.round(scaled_vertices).astype(int)
            outside_indices = np.any(
                (scaled_vertices < min_values) | (scaled_vertices > max_values), axis=1
            )

    return scaled_vertices


def close_poligon(vertices):
    if not np.array_equal(vertices[0], vertices[-1]):
        vertices = np.vstack((vertices, vertices[0]))
    return vertices


def is_overlap(shape1, shape2):
    # if any of the shape is empty, return False
    if not shape1 or not shape2:
        return False
    x_min1, y_min1, x_max1, y_max1 = shape1
    x_min2, y_min2, x_max2, y_max2 = shape2

    # If one rectangle is on left side of other
    if x_max1 < x_min2 or x_max2 < x_min1:
        return False

    # If one rectangle is above other
    if y_max1 < y_min2 or y_max2 < y_min1:
        return False

    return True


def find_intersection_vertices(rect1, rect2):
    # Create polygons from rectangles
    polygon1 = Polygon(rect1)
    polygon2 = Polygon(rect2)

    # Find the intersection polygon
    intersection = polygon1.intersection(polygon2)
    # if intersection is a LineString, return None
    if intersection.geom_type == "LineString" or intersection.geom_type == "Point":
        return []
    # Get the vertices of the intersection polygon
    vertices = list(intersection.exterior.coords)

    return vertices


def get_excluded_shape(rect1, rect2):
    # Create polygons from rectangles
    polygon1 = Polygon(rect1)
    polygon2 = Polygon(rect2)

    # Find the intersection polygon
    intersection = polygon1.intersection(polygon2)

    # Get the resulting shape by excluding the intersection from rect1
    excluded_shape = polygon1.difference(intersection)
    # if type MultiPolygon, return None
    if excluded_shape.geom_type == "MultiPolygon":
        return []
    vertices = list(excluded_shape.exterior.coords)
    return vertices


def get_union(rect1, rect2):
    # Create polygons from rectangles
    polygon1 = Polygon(rect1)
    polygon2 = Polygon(rect2)

    # Find the intersection polygon
    intersection = polygon1.intersection(polygon2)

    # Get the resulting shape by excluding the intersection from rect1
    excluded_shape = polygon1.difference(intersection)

    # Get the union of all shapes
    union_shape = excluded_shape.union(polygon2)
    vertices = list(union_shape.exterior.coords)
    return vertices


def increase_image_size(image, factor=1.0, color=(255, 255, 255)):
    """
    Increase image size by adding 0s around the image
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    new_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    new_image[:, :] = color
    new_image[
        int((new_h - h) / 2) : int((new_h - h) / 2) + h,
        int((new_w - w) / 2) : int((new_w - w) / 2) + w,
    ] = image
    return new_image


# Main function to generate the map
def generate_map(key, map_dict):
    # Create a blank image
    dimensions = map_dict.get("dimensions", (600, 400))
    image = np.ones((dimensions[1], dimensions[0], 3), dtype=np.uint8) * 255
    image_mask = np.ones((dimensions[1], dimensions[0]), dtype=np.bool_).reshape(-1)
    shapes_list = []  # List of all shapes drawn so far
    # Draw the shapes
    color = (0, 0, 0)
    color_idx = 0
    for shape, count in map_dict.get("shapes", {}).items():
        for _ in range(count):
            for _ in range(map_dict["max_trial_per_shape"]):
                # print(shape)
                if shape == "triangle":
                    vertices, key = get_triangle(
                        key,
                        image,
                        image_mask,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "trapezoid":
                    vertices, key = get_trapezoid(
                        key,
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "rectangle":
                    vertices, key = get_rectangle(
                        key,
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "pentagon":
                    vertices, key = get_pentagon(
                        key,
                        image,
                        image_mask,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "hexagon":
                    vertices, key = get_hexagon(
                        key,
                        image,
                        image_mask,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "regular_polygon":
                    vertices, key = draw_regular_polygon(
                        key,
                        image,
                        map_dict.get("regular_num_sides", 5),
                        map_dict.get("radius", 100),
                    )
                elif shape == "L":
                    vertices, key = draw_L(
                        key,
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "Z":
                    vertices, key = draw_Z(
                        key,
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                else:
                    continue
                if (
                    vertices is None
                    or len(vertices.shape) == 0
                    or vertices.shape[0] == 0
                ):
                    continue
                # Get the bounding rectangle of the shape
                x_min, y_min = np.min(vertices, axis=0)
                x_max, y_max = np.max(vertices, axis=0)
                new_shape = (x_min, y_min, x_max, y_max)

                # Check if it overlaps with any existing shape
                if any(is_overlap(new_shape, shape) for shape in shapes_list):
                    continue  # This shape overlaps, try again
                # If we reached this point, the shape doesn't overlap, so we can draw it
                # print('Drawing shape {}'.format(shape))
                shapes_list.append(new_shape)
                vertices = np.array(vertices)
                vertices_shrinked = scale_shape(
                    vertices, map_dict.get("scale_factor", 1.0)
                )
                # image = draw_shape(image, vertices, color=(128, 128, 128))
                image_old = image.copy()
                image = draw_shape(image, vertices_shrinked, color=color)
                if (image_old - image).sum().item() != 0:
                    # something has been added to the image
                    image_mask = image_mask.reshape(dimensions[1], dimensions[0])
                    image_mask[int(y_min) : int(y_max), int(x_min) : int(x_max)] = False
                    image_mask = image_mask.reshape(-1)
                    color_idx += 1
                    break  # Break the while loop and move to the next shape
            if color_idx % 2 != 0:
                color = (100, 100, 100)
            else:
                color = (0, 0, 0)

    # increase image size
    image = increase_image_size(image, factor=1.0, color=(255, 255, 255))
    return image, key


# def invert_map(image: np.ndarray):
#     """
#     Flips the image in this way:
#     - turns white (255, 255, 255) to BORDER_COLOR, which means traversable but should not dig there
#     - turns black (0, 0, 0) to white (255, 255, 255), which means that previous obstacles become dig zones
#     """
#     # Define the color for the traversable but non-diggable areas

#     # Create a copy of the image to avoid modifying the original array
#     flipped_image = np.copy(image)

#     # Replace white pixels with BORDER_COLOR
#     white_pixels = np.all(image == [255, 255, 255], axis=2)
#     flipped_image[white_pixels] = BORDER_COLOR

#     # Replace black pixels with white pixels
#     black_pixels = np.all(image == [0, 0, 0], axis=2)
#     flipped_image[black_pixels] = [255, 255, 255]

#     return flipped_image


def generate_occupancy(image: np.ndarray):
    """
    Generates the occupancy grid from the workspace image. Where ever the image is (0, 0, 0), it is considered an
    obstacle and it get registered as white (255, 255, 255) in the occupancy grid. The rest of the pixels are
    considered traversable and get registered as black (0, 0, 0).
    """
    # Create a blank image
    occupancy_grid = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Define the color for the traversable but non-diggable areas

    # Replace white pixels with BORDER_COLOR
    white_pixels = np.all(image == (0, 0, 0), axis=2)
    occupancy_grid[white_pixels] = 255

    return occupancy_grid


# def downsample(image, final_edge_size: int):
#     div = int(np.ceil(image.shape[0] / final_edge_size).item())
#     image = block_reduce(
#         image,
#         block_size=div,
#         func=np.max,
#     )
#     return image


def image_to_bitmap(image):
    """
    Converts an RGB image to a single-channel (WxH) bitmap.

    The following conversions are performed:
    255 --> 0 (nothing there)
    1 to 254 --> 1 (dump area)
    0 --> -1 (dig area)
    """
    image = image.mean(axis=-1, keepdims=False)

    image = np.where(image == 0, -1, image)
    image = np.where((image > 0) & (image < 255), 1, image)
    image = np.where(image == 255, 0, image)
    return image.astype(np.int8)


def generate_polygonal_bitmap(key, map_dict):
    image, key = generate_map(key, map_dict)
    image = image_to_bitmap(image)
    return image, key


if __name__ == "__main__":
    width, height = 40, 40
    map_dict = {
        "shapes": {
            "triangle": 2,
            "trapezoid": 0,
            "rectangle": 0,
            "pentagon": 2,
            "hexagon": 1,
            "L": 0,
            "Z": 0,
            # 'regular_polygon': 6,
        },
        "dimensions": (width, height),
        "max_edge": max(3, min(width, height) // 4),
        "min_edge": max(1, min(width, height) // 20),
        "radius": 300,
        "regular_num_sides": [3, 4, 5],
        "scale_factor": 1,
        "min_area": max(1, width * height // 55),
        "max_trial_per_shape": 5,
    }
    n_images = 20
    key = jax.random.PRNGKey(11)

    from tqdm import tqdm
    import cv2

    for _ in tqdm(range(n_images)):
        image, key = generate_map(key, map_dict)
        # inverted_image = invert_map(image)
        # bitmap = image_to_bitmap(image)

        div = 600 // 40

        cv2.imshow("Map", image.repeat(div, axis=0).repeat(div, axis=1))
        # cv2.imshow("Map", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
