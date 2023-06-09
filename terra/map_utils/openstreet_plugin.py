import json

import matplotlib.pyplot as plt
import osmnx as ox
from pyproj import Proj
from pyproj import transform


def get_building_shapes_from_OSM(
    north, south, east, west, option=1, save_path="output.png", folder_path=None
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
    # Fetch buildings from OSM
    bbox = (north, south, east, west)
    buildings = ox.geometries.geometries_from_bbox(*bbox, tags={"building": True})

    # Define coordinate reference systems
    wgs84 = Proj("EPSG:4326")  # WGS84 (lat-long) coordinate system
    utm = Proj("EPSG:32633")  # UTM zone 33N (covers central Europe)

    # Check option
    if option == 1:
        # Plot and save the binary map
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        ax.axis("off")
        ox.plot_footprints(buildings, ax=ax, color="black", bgcolor="white", show=False)
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True
        )
        plt.close()

        # Convert total bounds to UTM
        total_bounds_utm = [
            transform(
                wgs84, utm, buildings.total_bounds[i], buildings.total_bounds[i + 1]
            )
            for i in range(0, 4, 2)
        ]

        # Save metadata file with real dimensions in meters
        metadata = {
            "real_dimensions": {
                "length": total_bounds_utm[1][0] - total_bounds_utm[0][0],
                "width": total_bounds_utm[1][1] - total_bounds_utm[0][1],
            }
        }
        with open(f"{save_path}.json", "w") as file:
            json.dump(metadata, file)

    elif option == 2:
        # Check if folder path is provided
        if folder_path is None:
            raise ValueError("No folder path provided")

        # Iterate over the buildings
        print(f"total number of buildings = {len(buildings.geometry)}")
        for i, building in enumerate(buildings.geometry):
            # Filter out very small shapes
            if building.area < 1e-9:
                continue

            print(i)

            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            ax.axis("off")
            ox.plot_footprints(
                buildings.iloc[i : i + 1],
                ax=ax,
                color="black",
                bgcolor="white",
                show=False,
            )
            plt.savefig(
                f"{folder_path}/images/building_{i}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )
            plt.close()

            # Convert building bounds to UTM
            bounds_utm = [
                transform(wgs84, utm, building.bounds[i], building.bounds[i + 1])
                for i in range(0, 4, 2)
            ]

            # Save metadata file with real dimensions in meters
            metadata = {
                "building_index": i,
                "real_dimensions": {
                    "length": bounds_utm[1][0] - bounds_utm[0][0],
                    "width": bounds_utm[1][1] - bounds_utm[0][1],
                },
            }
            with open(f"{folder_path}/metadata/building_{i}.json", "w") as file:
                json.dump(metadata, file)

    else:
        print("Invalid option selected. Choose either 1 or 2.")


if __name__ == "__main__":
    # to get the bbox use https://colab.research.google.com/github/opengeos/segment-geospatial/blob/main/docs/examples/satellite.ipynb#scrollTo=kvB16LLk0qPY
    zurich_bbox = (47.378177, 47.364622, 8.526535, 8.544894)
    folder_path = "/home/antonio/Downloads"
    get_building_shapes_from_OSM(*zurich_bbox, option=2, folder_path=folder_path)
