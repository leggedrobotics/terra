import json
import matplotlib.pyplot as plt
import osmnx as ox
import cv2
from PIL import Image
from pyproj import Proj, transform
import os
from shapely.geometry import Polygon, Point


def get_building_shapes_from_OSM(
    north, south, east, west, option=1, save_folder="data/"
):
    """
    Extracts building shapes from OpenStreetMap given a bounding box of coordinates.

    Parameters:
    north (float): northern boundary of bounding box
    south (float): southern boundary of bounding box
    east (float): eastern boundary of bounding box
    west (float): western boundary of bounding box
    option (int, optional): Option for operation. 1 for saving a binary map, 2 for saving individual buildings. Defaults to 1.
    save_path (str, optional): File path to save output if option is 1. Defaults to 'output.png'.
    folder_path (str, optional): Folder path to save output if option is 2. Defaults to None.

    Returns:
    None
    """
    # make sure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Fetch buildings from OSM\
    bbox = (north, south, east, west)
    try:
        buildings = ox.features.features_from_bbox(*bbox, tags={"building": True})
    except Exception as e:
        return
    print("got buildings")

    # Define coordinate reference systems
    wgs84 = Proj("EPSG:4326")  # WGS84 (lat-long) coordinate system
    utm = Proj("EPSG:32633")  # UTM zone 33N (covers central Europe)

    # Check option
    if option == 1:
        extract_crop(buildings, wgs84, utm, north, south, east, west, save_folder)
    elif option == 2:
        extract_single_buildings(buildings, wgs84, utm, folder_path=save_folder)
    else:
        print("Invalid option selected. Choose either 1 or 2.")


def extract_crop(buildings, wgs84, utm, north, south, east, west, save_folder):
    # Convert the bounding box to UTM
    west_utm, south_utm = transform(wgs84, utm, west, south)
    east_utm, north_utm = transform(wgs84, utm, east, north)
    print("width crop in meters", east_utm - west_utm)
    print("height crop in meters", north_utm - south_utm)
    # Create a Polygon that represents the bounding box
    bbox_polygon = Polygon(
        [
            (west_utm, south_utm),
            (east_utm, south_utm),
            (east_utm, north_utm),
            (west_utm, north_utm),
        ]
    )

    # Count how many buildings have their centroid within the bounding box
    try:
        buildings_in_bbox = sum(
            bbox_polygon.contains(
                Point(transform(wgs84, utm, *building.centroid.coords[0]))
            )
            for building in buildings.geometry
        )
    except Exception as e:
        buildings_in_bbox = 0

    # Check the number of buildings. Return if less than 2.
    if buildings_in_bbox < 2:
        print("Less than 2 buildings found in the given bounding box")
        return

    try:
        # Convert total bounds to UTM
        total_bounds_utm = [
            transform(
                wgs84, utm, buildings.total_bounds[i], buildings.total_bounds[i + 1]
            )
            for i in range(0, 4, 2)
        ]
        aspect_ratio = (
            (total_bounds_utm[1][0] - total_bounds_utm[0][0])
            / (total_bounds_utm[1][1] - total_bounds_utm[0][1])
        ) ** (-1)
        dpi = 50
        max_pixels = 600
        max_inches = int(max_pixels / dpi)
        if aspect_ratio > 1:
            figsize = (max_inches, int(max_inches / aspect_ratio))
        else:
            figsize = (int(max_inches * aspect_ratio), max_inches)
        # if 0 set it to 1
        if figsize[0] == 0:
            figsize = (1, figsize[1])
        if figsize[1] == 0:
            figsize = (figsize[0], 1)

        # Plot and save the binary map
        fig, ax = plt.subplots(figsize=figsize, dpi=20)
        ax.axis("off")
        print("Plotting binary map...")
        ox.plot_footprints(buildings, ax=ax, color="black", bgcolor="white", show=False)
        print("Done")
        # save path include the coordinates of the bounding box
        save_path = f"{save_folder}/output_{north}_{south}_{east}_{west}"
        plt.savefig(
            save_path + ".png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        print("Saved binary map in {}".format(save_path))
        plt.close()
    except ValueError:
        print("No buildings found in the given bounding box")
        return

        # Post-process the image with OpenCV to count the number of distinct buildings
    image = cv2.imread(f"{save_path}.png", cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(thresh)

    # Check the number of distinct buildings (subtract 1 because the background is also considered a label)
    num_buildings = num_labels - 1
    if num_buildings < 2:
        print("Less than 2 distinct buildings found in the image")
        os.remove(
            f"{save_path}.png"
        )  # Delete the image if it has less than 2 buildings
        return

    # Open the image file with PIL and get its size in pixels
    with Image.open(f"{save_path}.png") as img:
        width_px, height_px = img.size

    # Convert the size from pixels to inches
    new_size = (width_px / dpi, height_px / dpi)

    print("old size {} and new size {}".format(figsize, new_size))

    width = total_bounds_utm[1][0] - total_bounds_utm[0][0]
    height = total_bounds_utm[1][1] - total_bounds_utm[0][1]
    new_height = new_size[0] / figsize[0] * height
    new_width = new_size[1] / figsize[1] * width
    print("old width {} and new width {}".format(width, new_width))
    print("old height {} and new height {}".format(height, new_height))

    # Save metadata file with real dimensions in meters
    metadata = {
        "coordinates": {"north": north, "south": south, "east": east, "west": west},
        "real_dimensions": {"height": new_width, "width": new_height},
    }
    with open(f"{save_path}.json", "w") as file:
        json.dump(metadata, file)


def extract_single_buildings(buildings, wgs84, utm, folder_path=None):
    if folder_path is None:
        raise ValueError("No folder path provided")

    for i, building in enumerate(buildings.geometry):
        if building.area < 1e-9:
            continue

        bounds_utm = [
            transform(wgs84, utm, building.bounds[i], building.bounds[i + 1])
            for i in range(0, 4, 2)
        ]
        aspect_ratio = (
            (bounds_utm[1][0] - bounds_utm[0][0])
            / (bounds_utm[1][1] - bounds_utm[0][1])
        ) ** (-1)
        dpi = 50
        max_pixels = 600
        max_inches = int(max_pixels / dpi)
        if aspect_ratio > 1:
            figsize = (max_inches, int(max_inches / aspect_ratio))
        else:
            figsize = (int(max_inches * aspect_ratio), max_inches)
        # if 0 set it to 1
        if figsize[0] == 0:
            figsize = (1, figsize[1])
        if figsize[1] == 0:
            figsize = (figsize[0], 1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.set_xlim(bounds_utm[0][0], bounds_utm[1][0])
        ax.set_ylim(bounds_utm[0][1], bounds_utm[1][1])
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ox.plot_footprints(
            buildings.iloc[i : i + 1], ax=ax, color="black", bgcolor="white", show=False
        )
        # plt.tight_layout()
        # new_size = fig.get_size_inches()
        # create image folder if it does not exist
        os.makedirs(f"{folder_path}/images", exist_ok=True)
        # apply padding
        plt.savefig(
            f"{folder_path}/images/building_{i}.png",
            dpi=dpi,
            pad_inches=1.0,
            bbox_inches="tight",
        )
        plt.close()
        # Open the image file with PIL and get its size in pixels
        with Image.open(f"{folder_path}/images/building_{i}.png") as img:
            width_px, height_px = img.size

        # Convert the size from pixels to inches
        new_size = (width_px / dpi, height_px / dpi)

        print(
            "old size {} and new size {} for building {}".format(figsize, new_size, i)
        )

        width = bounds_utm[1][0] - bounds_utm[0][0]
        height = bounds_utm[1][1] - bounds_utm[0][1]
        new_height = new_size[0] / figsize[0] * height
        new_width = new_size[1] / figsize[1] * width
        print(
            "old width {} and new width {} for building {}".format(width, new_width, i)
        )
        print(
            "old height {} and new height {} for building {}".format(
                height, new_height, i
            )
        )

        metadata = {
            "building_index": i,
            "real_dimensions": {"width": new_height, "height": new_width},
        }
        # make metadata folder if it does not exist
        os.makedirs(f"{folder_path}/metadata", exist_ok=True)
        with open(f"{folder_path}/metadata/building_{i}.json", "w") as file:
            json.dump(metadata, file)


import random
from typing import Tuple


# def collect_random_crops_old(bbox: Tuple, crop_size: Tuple[float, float], save_folder: str):
#     """
#     Collects random crops of the map using get_building_shapes_from_OSM option 1.
#     Select a random crop of inside the main bbox. The size of the random crop is expressed as a fraction
#     of the total bbox size by scale_factor.
#
#     Parameters:
#     bbox (Tuple): Bounding box coordinates (north, south, east, west).
#     scale_factor (float): Scale factor to determine the size of the random crop.
#
#     Returns:
#     None
#     """
#     # Load the metadata file to obtain real dimensions
#     # Convert total bounds to UTM
#     height = bbox[0] - bbox[1]
#     width = bbox[3] - bbox[2]
#
#     # Calculate the size of the random crop
#     # crop_length = height * scale_factor[0]
#     # crop_width = width * scale_factor[1]
#     # check that the aspect ratio is the same as the original
#     # if crop_length / crop_width != length / width:
#     #     print("scale factor {}".format(scale_factor))
#     #     print("original length {} and original width {}".format(length, width))
#     #     raise ValueError("The aspect ratio of the random crop is different from the original {} {}".format(
#     #         crop_length / crop_width, length / width
#     #     ))
#
#     print("crop length {} and crop width {}".format(crop_length, crop_width))
#
#     # Calculate the range of valid crop positions
#     # min_x = bbox[3] + crop_width
#     # max_x = bbox[2] - crop_width
#     # min_y = bbox[1] + crop_length
#     # max_y = bbox[0] - crop_length
#     north, south, east, west = bbox
#
#     # Generate random crop position
#     random_x = random.uniform(south, max_x)
#     random_y = random.uniform(min_y, max_y)
#
#     # Define the new crop bounding box
#     crop_bbox = (
#         random_y + crop_length,
#         random_y - crop_length,
#         random_x + crop_width,
#         random_x - crop_width
#     )
#
#
#     # Call get_building_shapes_from_OSM with option 2 to save the random crop
#     get_building_shapes_from_OSM(*crop_bbox, option=1, save_folder=save_folder)


from pyproj import Proj, transform
import random
from typing import Tuple
import os


def collect_random_crops(
    outer_bbox_wgs84: Tuple, crop_size_utm: Tuple[float, float], save_folder: str
):
    """
    Creates a random crop in wgs84 coordinates and using the outer_bbox_wgs84 as a boundary and the crops_size_utm in meters
    as the size of the crop.

    Args:
        outer_bbox_wgs84: (north, south, east, west)
        crop_size_utm: (width, height)
        save_folder: str
    """
    north, south, east, west = outer_bbox_wgs84
    width_crop, height_crop = crop_size_utm

    utm = Proj("EPSG:32633")  # UTM zone 33N (covers central Europe)
    wgs84 = Proj("EPSG:4326")  # WGS84 (covers the entire globe)

    west_utm, south_utm = transform(wgs84, utm, west, south)
    east_utm, north_utm = transform(wgs84, utm, east, north)

    # Get width and height of the outer box in UTM coordinates
    width_utm = east_utm - west_utm
    height_utm = north_utm - south_utm

    # Choose a random point within the UTM coordinate box as the lower left corner of the new box
    west_crop = west_utm + random.uniform(0, width_utm - width_crop)
    south_crop = south_utm + random.uniform(0, height_utm - height_crop)

    # Create a new box with the given size around this point
    east_crop = west_crop + width_crop
    north_crop = south_crop + height_crop

    # Convert the UTM coordinates of this box back to WGS84 coordinates
    west_wgs84, south_wgs84 = transform(utm, wgs84, west_crop, south_crop)
    east_wgs84, north_wgs84 = transform(utm, wgs84, east_crop, north_crop)

    crop_wgs84 = (north_wgs84, south_wgs84, east_wgs84, west_wgs84)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Call the get_building_shapes_from_OSM function with these new coordinates and the given save folder
    get_building_shapes_from_OSM(*crop_wgs84, option=1, save_folder=save_folder)


if __name__ == "__main__":
    # to get the bbox use https://colab.research.google.com/github/opengeos/segment-geospatial/blob/main/docs/examples/satellite.ipynb#scrollTo=kvB16LLk0qPY
    center_bbox = (47.378177, 47.364622, 8.526535, 8.544894)
    zurich_bbox = (47.3458, 47.409, 8.5065, 8.5814)
    folder_path = "data/openstreet/crops"
    # get_building_shapes_from_OSM(*zurich_bbox, option=1, folder_path=folder_path, save_folder=folder_path + "/" +
    #                                                                                         "random_crops")
    num_crop = 100
    for i in range(num_crop):
        print(f"crop {i} of {num_crop}")
        collect_random_crops(center_bbox, 0.02, folder_path)
    # get_building_shapes_from_OSM(*center_bbox, option=2, folder_path=folder_path, save_folder=folder_path +
    #                                                                                           "/foundations")
