import json
import matplotlib.pyplot as plt
import osmnx as ox
import cv2
from PIL import Image
from pyproj import CRS, Transformer
import os
from shapely.geometry import Polygon, Point
import random
from typing import Tuple
import concurrent.futures
import geopandas as gpd

def get_building_shapes_from_OSM(
    north, south, east, west, option=1, save_folder="data/", max_buildings=None
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
    wgs84 = CRS("EPSG:4326")  # WGS84 (lat-long) coordinate system
    utm = CRS("EPSG:32633")  # UTM zone 33N (covers central Europe)
    
    # Create transformer
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)

    # Check option
    if option == 1:
        extract_crop(buildings, wgs84, utm, north, south, east, west, save_folder, transformer)
    elif option == 2:
        extract_single_buildings_parallel(buildings, wgs84, utm, folder_path=save_folder, transformer=transformer, max_buildings=max_buildings)
    else:
        print("Invalid option selected. Choose either 1 or 2.")


def process_single_building(i, building, wgs84, transformer, folder_path):
    """
    Worker function to process one building. This creates a single-row GeoDataFrame,
    sets up the plot, saves the image, computes its dimensions, and writes metadata.
    """
    if building.area < 1e-9:
        return

    # Create a GeoDataFrame for the single building.
    gdf = gpd.GeoDataFrame({'geometry': [building]}, crs=wgs84)
    
    # Calculate UTM bounds from the building's bounding box.
    b = building.bounds  # (minx, miny, maxx, maxy)
    bounds_utm = [
        transformer.transform(b[0], b[1]),
        transformer.transform(b[2], b[3])
    ]
    width_utm = bounds_utm[1][0] - bounds_utm[0][0]
    height_utm = bounds_utm[1][1] - bounds_utm[0][1]
    if height_utm == 0 or width_utm == 0:
        return
    # Compute aspect ratio as in the original code.
    aspect_ratio = (width_utm / height_utm) ** (-1)
    dpi = 50
    max_pixels = 600
    max_inches = int(max_pixels / dpi)
    if aspect_ratio > 1:
        figsize = (max_inches, int(max_inches / aspect_ratio))
    else:
        figsize = (int(max_inches * aspect_ratio), max_inches)
    if figsize[0] == 0:
        figsize = (1, figsize[1])
    if figsize[1] == 0:
        figsize = (figsize[0], 1)

    # Create a matplotlib figure.
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_xlim(bounds_utm[0][0], bounds_utm[1][0])
    ax.set_ylim(bounds_utm[0][1], bounds_utm[1][1])
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ox.plot_footprints(gdf, ax=ax, color="black", bgcolor="white", show=False)

    # Ensure the images folder exists.
    images_folder = os.path.join(folder_path, "images")
    os.makedirs(images_folder, exist_ok=True)
    image_path = os.path.join(images_folder, f"building_{i}.png")
    plt.savefig(image_path, dpi=dpi, pad_inches=1.0, bbox_inches="tight")
    plt.close(fig)

    # Open the saved image and compute its dimensions.
    try:
        with Image.open(image_path) as img:
            width_px, height_px = img.size
    except Exception as e:
        print(f"Error opening image for building {i}: {e}")
        return

    new_size = (width_px / dpi, height_px / dpi)
    print(f"Building {i}: old size {figsize} and new size {new_size}")

    width = width_utm
    height = height_utm
    new_height = new_size[0] / figsize[0] * height
    new_width = new_size[1] / figsize[1] * width
    print(f"Building {i}: old width {width} and new width {new_width}")
    print(f"Building {i}: old height {height} and new height {new_height}")

    metadata = {
        "building_index": i,
        "real_dimensions": {"width": new_height, "height": new_width},
    }
    metadata_folder = os.path.join(folder_path, "metadata")
    os.makedirs(metadata_folder, exist_ok=True)
    metadata_path = os.path.join(metadata_folder, f"building_{i}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

def extract_single_buildings_parallel(buildings, wgs84, utm, folder_path=None, transformer=None, max_buildings=None):
    """
    Processes building extraction in parallel using multiple CPU cores.
    Each building is handled independently.
    """
    if folder_path is None:
        raise ValueError("No folder path provided")
    if transformer is None:
        raise ValueError("No transformer provided")
    
    tasks = []
    for i, building in enumerate(buildings.geometry):
        if max_buildings is not None and i >= max_buildings:
            break
        if building.area < 1e-9:
            continue
        tasks.append((i, building, wgs84, transformer, folder_path))
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_building, *args) for args in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Error processing building:", e)


def extract_crop(buildings, wgs84, utm, north, south, east, west, save_folder, transformer):
    # Convert the bounding box to UTM
    west_utm, south_utm = transformer.transform(west, south)
    east_utm, north_utm = transformer.transform(east, north)
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
                Point(transformer.transform(*building.centroid.coords[0]))
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
            transformer.transform(
                buildings.total_bounds[i], buildings.total_bounds[i + 1]
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

    # Define coordinate reference systems
    utm = CRS("EPSG:32633")  # UTM zone 33N (covers central Europe)
    wgs84 = CRS("EPSG:4326")  # WGS84 (covers the entire globe)
 
    # Create transformers
    transformer_to_utm = Transformer.from_crs(wgs84, utm, always_xy=True)
    transformer_from_utm = Transformer.from_crs(utm, wgs84, always_xy=True)

    west_utm, south_utm = transformer_to_utm.transform(west, south)
    east_utm, north_utm = transformer_to_utm.transform(east, north)

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
    west_wgs84, south_wgs84 = transformer_from_utm.transform(west_crop, south_crop)
    east_wgs84, north_wgs84 = transformer_from_utm.transform(east_crop, north_crop)

    crop_wgs84 = (north_wgs84, south_wgs84, east_wgs84, west_wgs84)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Call the get_building_shapes_from_OSM function with these new coordinates and the given save folder
    get_building_shapes_from_OSM(*crop_wgs84, option=1, save_folder=save_folder)
