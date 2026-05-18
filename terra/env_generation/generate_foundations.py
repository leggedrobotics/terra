from terra.env_generation import openstreet
from terra.env_generation import utils
import os
import shutil

# set seed
from pyproj import CRS, Transformer
import random
import pathlib

random.seed(42)


def download_foundations(
    main_folder,
    center_bbox=(47.378177, 47.364622, 8.526535, 8.544894),
    min_size=(20, 20),
    max_size=(100, 100),
    padding=3,
    resolution=0.05,
    max_buildings=None,
):
    dataset_folder = main_folder + "/foundations_raw"
    # if it does not exist
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        os.makedirs(dataset_folder + "/images", exist_ok=True)
        os.makedirs(dataset_folder + "/metadata", exist_ok=True)
        try:
            openstreet.get_building_shapes_from_OSM(
                *center_bbox, option=2, save_folder=dataset_folder, max_buildings=max_buildings
            )
        except Exception as e:
            print(e)

    # filter out small cases
    image_folder = main_folder + "/foundations_raw/images"
    save_folder = main_folder + "/foundations_filtered/images"
    metadata_folder = main_folder + "/foundations_raw/metadata"
    utils.size_filter(image_folder, save_folder, metadata_folder, min_size, max_size)

    # pad the edges
    image_folder = main_folder + "/foundations_filtered/images"
    save_folder = main_folder + "/foundations_filtered_padded"
    metadata_folder = main_folder + "/foundations_raw/metadata"
    utils.pad_images_and_update_metadata(
        image_folder, metadata_folder, padding, (255, 255, 255), save_folder
    )

    # set resolution
    image_folder = main_folder + "/foundations_filtered_padded"
    metadata_folder = main_folder + "/foundations_filtered_padded"
    image_resized_folder = main_folder + "/foundations_filtered_padded_resized"
    utils.preprocess_dataset_fixed_resolution(
        image_folder, metadata_folder, image_resized_folder, resolution
    )

    # filter out small cases again, after resizing and all
    image_folder = main_folder + "/foundations_filtered_padded_resized"
    save_folder = main_folder + "/foundations_filtered_padded_resized_refiltered"
    metadata_folder = main_folder + "/foundations_filtered_padded_resized"
    utils.size_filter(
        image_folder,
        save_folder,
        metadata_folder,
        min_size,
        max_size,
        copy_metadata=True,
    )


def create_exterior_foundations(main_folder, padding=5):
    # fill holes
    image_folder = main_folder + "/foundations_filtered_padded_resized_refiltered"
    dataset_folder = main_folder + "/exterior_foundations_filled"
    # make it if it doesn't exist
    os.makedirs(dataset_folder, exist_ok=True)
    save_folder = dataset_folder + "/images"
    metadata_folder = main_folder + "/foundations_filtered_padded"
    utils.fill_dataset(image_folder, save_folder, copy_metadata=False)
    # copy metadata folder to save folder and change its name to metadata
    utils.copy_metadata(metadata_folder, dataset_folder + "/metadata")
    # make occupancy, in this case is the same as the images folder
    # copy folder but change name
    shutil.copytree(
        main_folder + "/exterior_foundations_filled/images",
        main_folder + "/exterior_foundations_filled/occupancy",
        dirs_exist_ok=True,
    )
    # pad the edges for navigation
    image_folder = main_folder + "/exterior_foundations_filled/images"
    save_folder = main_folder + "/exterior_foundations"
    metadata_folder = main_folder + "/foundations_filtered_padded"
    utils.pad_images_and_update_metadata(
        image_folder, metadata_folder, padding, (220, 220, 200), save_folder
    )
    # restructure folder format
    image_folder = main_folder + "/exterior_foundations/images"
    metadata_folder = main_folder + "/exterior_foundations/metadata"
    utils.copy_metadata(save_folder, metadata_folder)
    # remove all json files from save_folder
    os.system("rm " + save_folder + "/*.json")
    # move all remaining images in /exterior_foundations inside /exterior_foundations/images
    # Create the destination directory if it doesn't exist
    os.makedirs(image_folder, exist_ok=True)

    # Get a list of all files in the source directory
    files = os.listdir(save_folder)

    # Filter the files to only include PNG images
    png_files = [file for file in files if file.lower().endswith(".png")]

    # Move each PNG image to the destination directory
    for file in png_files:
        source_path = os.path.join(save_folder, file)
        destination_path = os.path.join(image_folder, file)
        shutil.move(source_path, destination_path)

    shutil.copytree(
        main_folder + "/exterior_foundations/images",
        main_folder + "/exterior_foundations/occupancy",
        dirs_exist_ok=True,
    )


def create_exterior_foundations_traversable(main_folder):
    dataset_folder = main_folder + "/exterior_foundations"
    save_folder = main_folder + "/exterior_foundations_traversable"
    utils.make_obstacles_traversable(
        dataset_folder + "/images", save_folder + "/images"
    )
    # copy metadata folder to save folder and change its name to metadata
    utils.copy_metadata(dataset_folder + "/metadata", save_folder + "/metadata")
    # generate empty occupancy
    utils.generate_empty_occupancy(
        dataset_folder + "/images", save_folder + "/occupancy"
    )


def create_foundations(main_folder):
    dataset_folder = main_folder + "/foundations_filtered_padded_resized_refiltered"
    save_folder = main_folder + "/foundations"
    utils.invert_dataset_apply_dump_foundations(dataset_folder, save_folder)
    utils.copy_metadata(dataset_folder, save_folder + f"/metadata")
    utils.generate_empty_occupancy(dataset_folder, save_folder + f"/occupancy")


if __name__ == "__main__":
    # Basel center_bbox = (47.5376, 47.6126, 7.5401, 7.6842)
    # Basel center_bbox small = (47.5645, 47.572, 7.5867, 7.5979)
    # Zurich center_bbox small (benchmark) = (47.378177, 47.364622, 8.526535, 8.544894)
    sizes = [(20, 60)]  # , (40, 80), (80, 160), (160, 320), (320, 640)]
    package_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    for size in sizes:
        dataset_folder = os.path.join(package_dir, "data", "openstreet")
        download_foundations(
            dataset_folder,
            min_size=(size[0], size[0]),
            max_size=(size[1], size[1]),
            center_bbox=(47.5376, 47.6126, 7.5401, 7.6842),
            max_buildings=100
        )
        create_foundations(dataset_folder)
