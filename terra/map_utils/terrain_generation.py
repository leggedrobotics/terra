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
import random

import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.measure import block_reduce


BORDER_COLOR = (255, 255, 255)


def get_triangle(image, min_edge, max_edge, min_area, min_angle=1):
    h, w = image.shape[:2]
    while True:
        vertices = []
        for _ in range(3):
            while True:
                x = random.randint(0, w)
                y = random.randint(0, h)
                # Calculate the distances to the other vertices
                distances = [np.linalg.norm(np.array([x, y]) - v) for v in vertices]
                # Check if the distances are within the desired range
                if all(d >= min_edge and d <= max_edge for d in distances):
                    # Check the angle between all pairs of edges
                    for i in range(len(vertices)):
                        try:
                            v1 = np.array(vertices[i - 1]) - np.array(vertices[i])
                            v2 = np.array([x, y]) - np.array(vertices[i])
                            cosine_angle = np.dot(v1, v2) / (
                                np.linalg.norm(v1) * np.linalg.norm(v2)
                            )
                            cosine_angle = np.clip(cosine_angle, -1, 1)
                            angle = np.arccos(cosine_angle) * 180 / np.pi
                        except RuntimeWarning:
                            print("Runtime Warning encountered in get_triangle.")
                            break
                        if angle < min_angle:
                            break
                    else:
                        vertices.append([x, y])
                        break
        vertices = np.array(vertices)
        # close the shape
        vertices = np.append(vertices, [vertices[0]], axis=0)
        area = get_shape_area(vertices)
        if area >= min_area:
            return vertices


def get_shape_area(vertices):
    polygon = Polygon(vertices)
    return polygon.area


def get_trapezoid(image, min_edge, max_edge, min_area, min_angle=30):
    h, w = image.shape[:2]
    while True:
        vertices = []
        for _ in range(4):
            while True:
                x = random.randint(0, w)
                y = random.randint(0, h)
                # Calculate the distances to the other vertices
                distances = [np.linalg.norm(np.array([x, y]) - v) for v in vertices]
                # Check if the distances are within the desired range
                if all(d >= min_edge and d <= max_edge for d in distances):
                    # Check the angle between all pairs of edges
                    for i in range(len(vertices)):
                        try:
                            v1 = np.array(vertices[i - 1]) - np.array(vertices[i])
                            v2 = np.array([x, y]) - np.array(vertices[i])
                            cosine_angle = np.dot(v1, v2) / (
                                np.linalg.norm(v1) * np.linalg.norm(v2)
                            )
                            # cosine_angle = np.clip(cosine_angle, -1, 1)
                            angle = np.arccos(cosine_angle) * 180 / np.pi
                        except RuntimeWarning:
                            print("Runtime Warning encountered in get_trapezoid.")
                            break

                        if angle < min_angle or angle > 0.9 * 180 or angle is np.nan:
                            break
                    else:
                        vertices.append([x, y])
                        break

        vertices = np.array(vertices)

        # Calculate the centroid of the vertices
        centroid = np.mean(vertices, axis=0)

        # Sort the vertices based on their angle from the centroid
        angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        vertices = vertices[np.argsort(angles)]

        # Calculate the area of the trapezoid
        area = get_shape_area(vertices)
        if area >= min_area:
            return vertices


def get_rectangle(image, min_edge, max_edge, min_area):
    h, w = image.shape[:2]
    while True:
        pt1 = (random.randint(0, w), random.randint(0, h))
        pt2 = (random.randint(0, w), random.randint(0, h))
        # Calculate the edge lengths
        edge_lengths = [abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1])]
        # Check if the edge lengths are within the desired range
        if all(edge >= min_edge and edge <= max_edge for edge in edge_lengths):
            # cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)

            # Fill the rectangle with blue color
            vertices = np.array([pt1, (pt1[0], pt2[1]), pt2, (pt2[0], pt1[1])])

            # Calculate the area of the rectangle
            area = get_shape_area(vertices)
            if area >= min_area:
                return vertices


def get_pentagon(image, min_edge, max_edge, min_area):
    h, w = image.shape[:2]
    while True:
        vertices = []
        for _ in range(5):
            while True:
                x = random.randint(0, w)
                y = random.randint(0, h)
                # Calculate the distances to the other vertices
                distances = [np.linalg.norm(np.array([x, y]) - v) for v in vertices]
                # Check if the distances are within the desired range
                if all(d >= min_edge and d <= max_edge for d in distances):
                    vertices.append([x, y])
                    break
        vertices = np.array(vertices)

        # Calculate the centroid of the vertices
        centroid = np.mean(vertices, axis=0)

        # Sort the vertices based on their angle with the centroid
        angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        ordered_vertices = vertices[sorted_indices]

        # Calculate the area of the pentagon
        area = get_shape_area(vertices)
        if area >= min_area:
            return ordered_vertices


def get_hexagon(image, min_edge, max_edge, min_area):
    h, w = image.shape[:2]
    while True:
        vertices = []
        for _ in range(6):
            while True:
                x = random.randint(0, w)
                y = random.randint(0, h)
                # Calculate the distances to the other vertices
                distances = [np.linalg.norm(np.array([x, y]) - v) for v in vertices]
                # Check if the distances are within the desired range
                if all(d >= min_edge and d <= max_edge for d in distances):
                    vertices.append([x, y])
                    break
        vertices = np.array(vertices)

        # Calculate the centroid of the vertices
        centroid = np.mean(vertices, axis=0)

        # Sort the vertices based on their angle with the centroid
        angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        ordered_vertices = vertices[sorted_indices]

        # Calculate the area of the hexagon
        area = get_shape_area(ordered_vertices)
        # print(area)
        if area >= min_area:
            return ordered_vertices


def draw_regular_polygon(image, num_sides: list[int], radius: int):
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
    num_sides = random.choice(num_sides)
    # center is randomly chosen but not too close to the edges (radius)
    h, w = image.shape[:2]
    center = (random.randint(radius, w - radius), random.randint(radius, h - radius))

    vertices = np.array(
        [
            [
                center[0] + radius * np.cos(2 * np.pi * i / num_sides),
                center[1] + radius * np.sin(2 * np.pi * i / num_sides),
            ]
            for i in range(num_sides)
        ],
        dtype=np.int32,
    )
    return vertices


def draw_L(image, min_edge, max_edge, min_area):
    """
    Creates a L shape by creating two intersecting rectangles. we use the function
    get_rectangle to create the rectangles but we ensure that the rectangles
    intersect.
    """
    # generate the first rectangle
    vertices = get_rectangle(image, min_edge, max_edge, min_area)
    vertices2 = []
    while True:
        # generate the second rectangle
        vertices2 = get_rectangle(image, min_edge, max_edge, min_area)
        # check if the rectangles intersect
        if find_intersection_vertices(vertices, vertices2):
            break
    # get the exclusion of the intersection from the first rectangle
    vertices = get_excluded_shape(vertices, vertices2)
    # list to np array
    vertices = np.array(vertices).astype(np.int64)
    return vertices


def draw_Z(image, min_edge, max_edge, min_area):
    # generate the first rectangle
    vertices = get_rectangle(image, min_edge, max_edge, min_area)
    vertices2 = []
    while True:
        # generate the second rectangle
        vertices2 = get_rectangle(image, min_edge, max_edge, min_area)
        # check if the rectangles intersect
        if find_intersection_vertices(vertices, vertices2):
            break
    # get the exclusion of the intersection from the first rectangle
    vertices = get_union(vertices, vertices2)
    # list to np array
    vertices = np.array(vertices).astype(np.int64)
    return vertices


def draw_O(image, min_edge, max_edge):
    h, w = image.shape[:2]
    # Ensure the size of the C shape
    while True:
        pt1 = (random.randint(0, w - max_edge), random.randint(0, h - max_edge))
        pt2 = (
            pt1[0] + random.randint(min_edge, max_edge),
            pt1[1] + random.randint(min_edge, max_edge),
        )
        inner_width = random.randint(min_edge, pt2[0] - pt1[0] - 1)
        inner_height = random.randint(min_edge, pt2[1] - pt1[1] - 1)
        inner_pt1 = (
            random.randint(pt1[0] + 1, pt2[0] - inner_width),
            random.randint(pt1[1] + 1, pt2[1] - inner_height),
        )
        inner_pt2 = (inner_pt1[0] + inner_width, inner_pt1[1] + inner_height)
        if (
            pt1[0] < inner_pt1[0] < pt2[0]
            and pt1[1] < inner_pt1[1] < pt2[1]
            and pt1[0] < inner_pt2[0] < pt2[0]
            and pt1[1] < inner_pt2[1] < pt2[1]
        ):
            break
    # cv2.rectangle(image, pt1, pt2, (0, 0, 0), -1)
    # cv2.rectangle(image, inner_pt1, inner_pt2, (255, 255, 255), -1)
    return image


def draw_shape(
    image,
    vertices,
    color=(0, 0, 0),
    countour_color=(255, 255, 255),
    countour_thickness=10,
):
    cv2.fillPoly(image, [vertices], color=color)
    # drow countour in grey
    cv2.polylines(image, [vertices], True, countour_color, countour_thickness)
    return image


def scale_shape(vertices, scale_factor=1.0):
    hull = cv2.convexHull(vertices)
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
def generate_map(map_dict):
    # Create a blank image
    dimensions = map_dict.get("dimensions", (600, 400))
    image = np.ones((dimensions[1], dimensions[0], 3), dtype=np.uint8) * 255
    shapes_list = []  # List of all shapes drawn so far
    # Draw the shapes
    odd = False
    for shape, count in map_dict.get("shapes", {}).items():
        for _ in range(count):
            while True:
                if shape == "triangle":
                    vertices = get_triangle(
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "trapezoid":
                    vertices = get_trapezoid(
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "rectangle":
                    vertices = get_rectangle(
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "pentagon":
                    vertices = get_pentagon(
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "hexagon":
                    vertices = get_hexagon(
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "regular_polygon":
                    vertices = draw_regular_polygon(
                        image,
                        map_dict.get("regular_num_sides", 5),
                        map_dict.get("radius", 100),
                    )
                elif shape == "L":
                    vertices = draw_L(
                        image,
                        map_dict.get("min_edge", 50),
                        map_dict.get("max_edge", 100),
                        map_dict.get("min_area", 1000),
                    )
                elif shape == "Z":
                    vertices = draw_Z(
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
                vertices_shrinked = scale_shape(
                    vertices, map_dict.get("scale_factor", 1.0)
                )
                # image = draw_shape(image, vertices, color=(128, 128, 128))
                if odd:
                    odd = False
                    color = (100, 100, 100)
                elif not odd:
                    odd = True
                    color = (0, 0, 0)
                image = draw_shape(image, vertices_shrinked, color=color)
                break  # Break the while loop and move to the next shape
    # increase image size
    image = increase_image_size(image, factor=1.1, color=(255, 255, 255))
    return image


def invert_map(image: np.ndarray):
    """
    Flips the image in this way:
    - turns white (255, 255, 255) to BORDER_COLOR, which means traversable but should not dig there
    - turns black (0, 0, 0) to white (255, 255, 255), which means that previous obstacles become dig zones
    """
    # Define the color for the traversable but non-diggable areas

    # Create a copy of the image to avoid modifying the original array
    flipped_image = np.copy(image)

    # Replace white pixels with BORDER_COLOR
    white_pixels = np.all(image == [255, 255, 255], axis=2)
    flipped_image[white_pixels] = BORDER_COLOR

    # Replace black pixels with white pixels
    black_pixels = np.all(image == [0, 0, 0], axis=2)
    flipped_image[black_pixels] = [255, 255, 255]

    return flipped_image


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


def downsample(image, final_edge_size: int):
    div = int(np.ceil(image.shape[0] / final_edge_size).item())
    image = block_reduce(
        image,
        block_size=div,
        func=np.max,
    )
    return image


if __name__ == "__main__":
    map_dict = {
        "shapes": {
            "triangle": 1,
            "trapezoid": 1,
            "rectangle": 1,
            "pentagon": 1,
            "hexagon": 1,
            "L": 1,
            "Z": 1,
            # 'regular_polygon': 6,
        },
        "dimensions": (1000, 1000),
        "max_edge": 400,
        "min_edge": 50,
        "radius": 300,
        "regular_num_sides": [3, 4, 5],
        "scale_factor": 0.7,
        "area_threshold": 1000,
    }
    n_images = 100

    from tqdm import tqdm
    import sys

    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("error")

    for _ in tqdm(range(n_images)):
        image = generate_map(map_dict)
        # inverted_image = invert_map(image)
        final_edge_size = 40
        image = downsample(image, final_edge_size)

    print("image.shape", image.shape)
    div = 1000 // final_edge_size
    cv2.imshow("Map", image.repeat(div, axis=0).repeat(div, axis=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
