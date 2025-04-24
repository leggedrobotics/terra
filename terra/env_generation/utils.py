import os
import shutil

import numpy as np
from pathlib import Path
from scipy.signal import convolve2d
import re

from terra.env_generation import terrain_generation

# Colors represented in RGB format

color_dict = {
    "neutral": [220, 220, 220],  # Light Gray
    "digging": [255, 255, 255],  # White
    "dumping": [90, 191, 20],  # Green
    "nondumpable": [255, 0, 0],  # Red
    "obstacle": [0, 0, 255],  # Blue
}


def _get_img_mask(img, color_triplet):
    return (
        (img[..., 0] == color_triplet[0])
        & (img[..., 1] == color_triplet[1])
        & (img[..., 2] == color_triplet[2])
    )


def shrink_obstacles(image: np.ndarray, shrink_factor: int = 1):
    """
    This function takes a binary image, 1s indicate areas to be dug out and 0s areas to be avoided.
    This function shrinks areas of 0s down by a factor. The difference between the new image and the old image
    should be marked with 0.5 (signifying the new borders of the obstacles)
    """
    # Make a copy of the image
    image_copy = image.copy()
    # Convert the image to uint8 format for OpenCV contour detection
    image_copy = (image_copy).astype(np.uint8)

    # Find contours in the image
    contours, _ = cv2.findContours(
        image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Shrink the contours by the specified factor
    for contour in contours:
        epsilon = shrink_factor * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(image_copy, [approx_contour], -1, (127,), thickness=cv2.FILLED)

    return image_copy


def shrink_obstacles_erosion(image: np.ndarray, shrink_factor: int = 1):
    """
    This function takes a binary image, 1s indicate areas to be dug out and 0s areas to be avoided.
    This function shrinks areas of 0s down by a factor. The difference between the new image and the old image
    should be marked with 0.5 (signifying the new borders of the obstacles)
    """
    # Invert the image, since we want to shrink the obstacles (represented as 0s)
    inverted_image = 1 - image

    # Define the structuring element (you can modify the size depending on your needs)
    struct_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * shrink_factor + 1, 2 * shrink_factor + 1)
    )

    # Perform erosion
    eroded_image = cv2.erode(inverted_image.astype(np.uint8), struct_element)

    # Subtract the eroded image from the original inverted image to find the borders
    border_image = inverted_image - eroded_image

    # Mark the borders with 0.5 in the original image
    image[border_image == 1] = 0.5

    # Invert the eroded image back to the original representation (1s for free space, 0s for obstacles)
    final_image = 1 - eroded_image

    return final_image


import cv2


def pad_images(
    image_folder, pad_size_factor, color, save_folder, obstacles_traversable=False
):
    """
    This function takes a folder of images and pads them by the specified factor (as fraction of the total dimension)
    The padding takes place on all sides of the image and has the specified color
    """
    # Get the list of images in the folder
    file_list = os.listdir(image_folder)
    image_list = [file for file in file_list if file.endswith(".png")]

    # Loop through all images
    for image_name in image_list:
        # Read the image
        image = cv2.imread(os.path.join(image_folder, image_name))
        if obstacles_traversable:
            image = terrain_generation.make_obstacles_traversable(image)
        # Get the dimensions of the image
        height, width = image.shape[:2]
        # Calculate the padding size
        pad_size = int(pad_size_factor * max(height, width))
        # Pad the image
        padded_image = cv2.copyMakeBorder(
            image,
            pad_size,
            pad_size,
            pad_size,
            pad_size,
            cv2.BORDER_CONSTANT,
            value=color,
        )
        # add a another 2% grey border
        pad_size = int(0.02 * max(height, width))
        padded_image = cv2.copyMakeBorder(
            padded_image,
            pad_size,
            pad_size,
            pad_size,
            pad_size,
            cv2.BORDER_CONSTANT,
            value=(color),
        )
        # Save the image
        cv2.imwrite(os.path.join(save_folder, image_name), padded_image)


def make_obstacles_traversable(image_folder, save_folder):
    """
    This function takes a folder of images and makes the obstacles traversable by setting the obstacle pixels to 0
    """
    # if save folder does not exist create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # Get the list of images in the folder
    file_list = os.listdir(image_folder)
    image_list = [file for file in file_list if file.endswith(".png")]

    # Loop through all images
    for image_name in image_list:
        # Read the image
        image = cv2.imread(os.path.join(image_folder, image_name))
        # Make the obstacles traversable
        image = terrain_generation.make_obstacles_traversable(image)
        # Save the image
        cv2.imwrite(os.path.join(save_folder, image_name), image)


def pad_images_and_update_metadata(
    image_folder,
    metadata_folder,
    pad_dims: float,
    color,
    save_folder,
    occupancy_folder: str = None,
):
    """
    This function takes a folder of images and pads them by the specified factor (as fraction of the total dimension)
    The padding takes place on all sides of the image and has the specified color.
    It fist looks up the real dimensions of the image in the metadata folder and then pads the image accordingly using
    the function pad_images. The metadata is then updated with the new dimensions.

    Args:
        image_folder: folder containing the images
        metadata_folder: folder containing the metadata
        pad_dims: float the padding size in meters

    """
    # create save folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_list = os.listdir(image_folder)
    image_list = [file for file in file_list if file.endswith(".png")]

    # Loop through all images
    for image_name in image_list:
        image = cv2.imread(os.path.join(image_folder, image_name))
        # load the json file of the metadata
        with open(
            os.path.join(metadata_folder, image_name[:-4] + ".json")
        ) as json_file:
            metadata = json.load(json_file)
        width = metadata["real_dimensions"]["width"]
        height = metadata["real_dimensions"]["height"]

        # make a new metadata file with the new dimensions
        metadata["real_dimensions"]["width"] = width + 2 * pad_dims
        metadata["real_dimensions"]["height"] = height + 2 * pad_dims
        with open(os.path.join(save_folder, image_name[:-4] + ".json"), "w") as outfile:
            json.dump(metadata, outfile)

        # use cv2 copyMakeBorder to pad the image
        resolution = image.shape[0] / height
        pad_size_pixels = int(pad_dims * resolution)

        padded_image = cv2.copyMakeBorder(
            image,
            pad_size_pixels,
            pad_size_pixels,
            pad_size_pixels,
            pad_size_pixels,
            cv2.BORDER_CONSTANT,
            value=color,
        )

        # save image
        cv2.imwrite(os.path.join(save_folder, image_name), padded_image)

        if occupancy_folder:
            occupancy_image = cv2.imread(os.path.join(occupancy_folder, image_name))
            padded_occupancy_image = cv2.copyMakeBorder(
                occupancy_image,
                pad_size_pixels,
                pad_size_pixels,
                pad_size_pixels,
                pad_size_pixels,
                cv2.BORDER_CONSTANT,
                value=color,
            )
            cv2.imwrite(os.path.join(save_folder, image_name), padded_occupancy_image)


import json


def preprocess_dataset_fixed_resolution(
    image_folder,
    image_metadata_folder,
    image_resized_folder,
    min_resolution,
    flip=False,
):
    """
    This function preprocesses the dataset by resizing the images to a minimum resolution and potentially flipping the
    color. If it flips the color of the shape from black to white, it adds a grey background.
    The images are black shapes with no background.

    This function increases the amount of pixels in the image according to the min_resolution but preserves the
    aspect ratio. There should not be any color aliasing.
    """
    # Create the resized folder if it doesn't exist
    if not os.path.exists(image_resized_folder):
        os.makedirs(image_resized_folder)

    # Iterate over the images in the image folder, only png images
    for filename in os.listdir(image_folder):
        if not filename.endswith(".png"):
            # copy the file
            shutil.copy(
                os.path.join(image_folder, filename),
                os.path.join(image_resized_folder, filename),
            )
        image_path = os.path.join(image_folder, filename)
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        # if not skip
        if image is None:
            print(f"Unable to read or process image '{filename}'. Skipping...")
            continue
        image_metadata_path = os.path.join(
            image_metadata_folder, filename[:-4] + ".json"
        )

        # Read metadata
        with open(image_metadata_path, "r") as f:
            image_metadata = json.load(f)

        real_dimensions = image_metadata["real_dimensions"]
        resolution = real_dimensions["width"] / image.shape[0]

        # Check if resolution is already enough
        if resolution > min_resolution:
            print(f"Image '{filename}' already has the required resolution.")
            resized_image = image
        else:
            # Check if image is None or has no shape
            if image is None or image.shape[0] is None or image.shape[1] is None:
                print(f"Unable to read or process image '{filename}'. Skipping...")
                continue

            # Get the original width and height
            height, width, _ = image.shape

            # Calculate the new dimensions based on the min_resolution and the aspect ratio
            scale_factor = resolution / min_resolution
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)

            # Resize the image using OpenCV's resize function
            resized_image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            )

        # Apply flip if enabled
        if flip:
            flipped_image = terrain_generation.invert_map(resized_image)
        else:
            flipped_image = resized_image

        # Save the flipped or resized image to the resized folder
        resized_image_path = os.path.join(image_resized_folder, filename)
        cv2.imwrite(resized_image_path, flipped_image)

        print(f"Image '{filename}' resized successfully.")


def preprocess_dataset(
    image_folder, metadata_folder, image_resized_folder, max_resolution, flip=False
):
    """
    This function preprocesses the dataset by resizing the images to a maximum resolution and potentially flipping the
    color. If it flips the color of the shape from black to white, it adds a grey background.
    The images are black shapes with no background.

    This function reduces the amount of pixels in the image according to the max_resolution but preserves the
    aspect ratio. There should not be any color aliasing.
    """
    # Create the resized folder if it doesn't exist
    if not os.path.exists(image_resized_folder):
        os.makedirs(image_resized_folder)

    # Iterate over the images in the image folder
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        image_metadata_path = os.path.join(metadata_folder, filename[:-4] + ".json")
        with open(image_metadata_path, "r") as f:
            image_metadata = json.load(f)
        real_dimensions = image_metadata["real_dimensions"]

        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Check if image is None or has no shape
        if image is None or image.shape[0] is None or image.shape[1] is None:
            print(f"Unable to read or process image '{filename}'. Skipping...")
            continue

        # Get the original width and height
        height, width, _ = image.shape

        # Calculate the new dimensions based on the max_resolution and the aspect ratio
        if width > height:
            new_width = max_resolution
            new_height = int((height / width) * max_resolution)
        else:
            new_width = int((width / height) * max_resolution)
            new_height = max_resolution

        # Resize the image using OpenCV's resize function
        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

        # Apply flip if enabled
        if flip:
            flipped_image = terrain_generation.invert_map(resized_image)
        else:
            flipped_image = resized_image

        # Save the flipped or resized image to the resized folder
        resized_image_path = os.path.join(image_resized_folder, filename)
        cv2.imwrite(resized_image_path, flipped_image)

        print(f"Image '{filename}' resized successfully.")


def invert_dataset(image_folder, image_inverted_folder):
    """
    This function inverts the color of the images in the image folder and saves them to the inverted folder.
    """
    # Create the inverted folder if it doesn't exist
    if not os.path.exists(image_inverted_folder):
        os.makedirs(image_inverted_folder)

    # Iterate over the images in the image folder
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Check if image is None or has no shape
        if image is None or image.shape[0] is None or image.shape[1] is None:
            print(f"Unable to read or process image '{filename}'. Skipping...")
            continue

        # Invert the image using the invert_map function
        inverted_image = terrain_generation.invert_map(image)

        # Save the inverted image to the inverted folder
        inverted_image_path = os.path.join(image_inverted_folder, filename)
        cv2.imwrite(inverted_image_path, inverted_image)

        print(f"Image '{filename}' inverted successfully.")


def invert_dataset_apply_dump_foundations(image_folder, image_inverted_folder):
    """
    This function inverts the color of the images in the image folder.
    Also, it applies 3 different dump patterns (easy, medium, hard),
    and saves the image to 3 different subfolders of image_inverted_folder.
    """
    # Create the inverted folder if it doesn't exist
    if not os.path.exists(image_inverted_folder):
        os.makedirs(image_inverted_folder)

    # Iterate over the images in the image folder
    i = -1
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Check if image is None or has no shape
        if image is None or image.shape[0] is None or image.shape[1] is None:
            print(f"Unable to read or process image '{filename}'. Skipping...")
            continue

        # Invert the image using the invert_map function
        inverted_image = terrain_generation.invert_map(image)

        # Fix image such that it only has color_dict colors
        inverted_image = np.where(
            (
                ~(
                    _get_img_mask(inverted_image, color_dict["digging"])
                    | _get_img_mask(inverted_image, color_dict["dumping"])
                    | _get_img_mask(inverted_image, color_dict["neutral"])
                )
            )[..., None].repeat(3, -1),
            np.array(color_dict["digging"])[None, None]
            .repeat(inverted_image.shape[0], 0)
            .repeat(inverted_image.shape[1], 1),
            inverted_image,
        ).astype(inverted_image.dtype)

        # Fully dumpable
        img = np.where(
            _get_img_mask(inverted_image, color_dict["neutral"])[..., None].repeat(
                3, -1
            ),
            color_dict["dumping"],
            inverted_image,
        )

        # Get the outer profile of the image
        # inverted_image_black = np.where(
        #     _get_img_mask(inverted_image[..., None].repeat(3, -1), color_dict["neutral"]),
        #     0,
        #     inverted_image
        # )
        # kernel_dim = int(min(inverted_image_black.shape[:2]) * 0.25)
        # kernel = np.ones((kernel_dim, kernel_dim))
        # expanded_img = convolve2d(inverted_image_black[..., 0], kernel, mode="same")
        # contoured_img = np.where(
        #     (expanded_img > 0) & (inverted_image_black[..., 0] == 0),
        #     1,
        #     inverted_image_black[..., 0]
        # )
        # tmp = _get_img_mask(contoured_img[..., None].repeat(3, -1), [1, 1, 1])[..., None] * color_dict["dumping"]
        # contoured_img = np.where(
        #     _get_img_mask(contoured_img[..., None].repeat(3, -1), [1, 1, 1])[..., None].repeat(3, -1),
        #     tmp,
        #     contoured_img[..., None].repeat(3, -1)
        # ).astype(np.uint8)
        # contoured_img = np.where(
        #     contoured_img == 0,
        #     color_dict["neutral"],
        #     contoured_img
        # ).astype(np.uint8)

        # Apply 3 dumping levels
        i += 1
        # w, h, _ = contoured_img.shape
        # for level in ["easy", "medium", "hard"]:
        #     if level == "easy":
        #         img = contoured_img
        #     elif level == "medium":
        #         side_constraints_medium = [
        #             (np.arange(w) < w // 2)[:, None].repeat(h, 1),
        #             (np.arange(w) > w // 2)[:, None].repeat(h, 1),
        #             (np.arange(h) < h // 2)[:, None].repeat(w, 1).T,
        #             (np.arange(h) > h // 2)[:, None].repeat(w, 1).T,
        #         ]
        #         img = np.where(
        #             (_get_img_mask(contoured_img, color_dict["dumping"]) * side_constraints_medium[i % 4])[..., None].repeat(3, -1),
        #             np.array(color_dict["neutral"])[None, None].repeat(w, 0).repeat(h, 1),
        #             contoured_img,
        #         ).astype(np.uint8)
        #     elif level == "hard":
        #         side_constraints_hard = [
        #             (np.arange(w) < w // 2)[:, None].repeat(h, 1) | (np.arange(h) < h // 2)[:, None].repeat(w, 1).T,
        #             (np.arange(w) < w // 2)[:, None].repeat(h, 1) | (np.arange(h) > h // 2)[:, None].repeat(w, 1).T,
        #             (np.arange(w) > w // 2)[:, None].repeat(h, 1) | (np.arange(h) < h // 2)[:, None].repeat(w, 1).T,
        #             (np.arange(w) > w // 2)[:, None].repeat(h, 1) | (np.arange(h) > h // 2)[:, None].repeat(w, 1).T,
        #         ]
        #         img = np.where(
        #             (_get_img_mask(contoured_img, color_dict["dumping"]) * side_constraints_hard[i % 4])[..., None].repeat(3, -1),
        #             np.array(color_dict["neutral"])[None, None].repeat(w, 0).repeat(h, 1),
        #             contoured_img,
        #         ).astype(np.uint8)

        # Save the inverted image to the inverted folder
        p = Path(f"{image_inverted_folder}/images")
        p.mkdir(parents=True, exist_ok=True)
        inverted_image_path = p / filename
        cv2.imwrite(str(inverted_image_path), img.astype(np.uint8))

        print(f"Foundations '{filename}' created successfully.")


def generate_empty_occupancy(image_folder: str, save_folder: str):
    """
    Goes through all the images in image_folder and generate an empty image of the same shape as the
    images in image_folder. The empty image is saved in save_folder.
    """
    # if the folder does not exist, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for filename in os.listdir(image_folder):
        # only png
        if not filename.endswith(".png"):
            continue
        image = cv2.imread(image_folder + "/" + filename, cv2.IMREAD_GRAYSCALE)
        image = image / 255
        empty_image = 255 * np.ones(image.shape)
        cv2.imwrite(save_folder + "/" + filename, empty_image)


def size_filter(
    image_folder,
    save_folder,
    metadata_folder,
    min_size=(20, 20),
    max_size=(1920, 1080),
    copy_metadata=False,
):
    """
    Goes through all the images, checks the real size in the metadata, and saves the image only if it's size is within
    min_size and max_size. The sizes correspond to the size of the whole image (same as reported in the metadata).
    """
    # if the folder does not exist, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # get the metadata
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image = cv2.imread(image_folder + "/" + filename, cv2.IMREAD_GRAYSCALE)
            metadata_path = metadata_folder + "/" + filename[:-4] + ".json"
            with open(metadata_path) as json_file:
                metadata = json.load(json_file)
            # get the real size
            real_dimensions = (
                metadata["real_dimensions"]["width"],
                metadata["real_dimensions"]["height"],
            )
            if (
                min_size[0] <= real_dimensions[0] <= max_size[0]
                and min_size[1] <= real_dimensions[1] <= max_size[1]
            ):
                cv2.imwrite(save_folder + "/" + filename, image)
                if copy_metadata:
                    with open(
                        os.path.join(save_folder + "/" + filename[:-4] + ".json"), "w"
                    ) as outfile:
                        json.dump(metadata, outfile)


def fill_holes(image: np.array):
    """
    Fills the holes in the image. The holes are white area in the image surrounded by black pixels.
    """
    # Threshold the image, let's assume that white is the color of the holes
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image
    im_floodfill = thresh.copy()

    # Mask used for flood filling.
    # Notice the size needs to be 2 pixels larger than the image.
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert the floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground
    filled_image = thresh | im_floodfill_inv

    # obtain the image in the right format by flipping the colors
    filled_image = 255 - filled_image

    return filled_image


def fill_dataset(image_folder, save_folder, copy_metadata=True):
    """
    Goes through all the images in image_folder and fill the holes in the images. The filled images are saved in
    save_folder.
    """
    # if the folder does not exist, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for filename in os.listdir(image_folder):
        if copy_metadata:
            # if json just copy
            if filename[-4:] == "json":
                shutil.copy(image_folder + "/" + filename, save_folder + "/" + filename)
                continue
        if filename[-3:] == "png":
            image = cv2.imread(image_folder + "/" + filename, cv2.IMREAD_GRAYSCALE)
            filled_image = fill_holes(image)
            cv2.imwrite(save_folder + "/" + filename, filled_image)
        else:
            continue


def copy_metadata(folder, target_folder):
    """
    Copies only *json files from folder to target_folder
    """
    # if the folder does not exist, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(folder):
        if filename[-4:] == "json":
            shutil.copy(folder + "/" + filename, target_folder + "/" + filename)
        else:
            continue


def copy_metadata_individual(path_input, path_output, target_folder):
    # if the folder does not exist, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    shutil.copy(path_input, path_output)



def copy_and_increment_filenames(source_folder, destination_folder):
    """
    Copies JSON files from the source folder to the destination folder.
    Files named like 'trench_{i}.json' are renamed to 'trench_{i+1}.json' during the copy.
    The copy operation starts from the highest numbered file to prevent overwriting.
    
    :param source_folder: Path to the source folder
    :param destination_folder: Path to the destination folder
    """
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Compile a regex pattern to match 'trench_{i}.json' and capture 'i'
    pattern = re.compile(r"trench_(\d+)\.json")

    # Create a list to store (filename, number) tuples
    files_with_numbers = []

    # Identify all matching files and their numbers
    for filename in os.listdir(source_folder):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            files_with_numbers.append((filename, number))
    
    # Sort the list based on numbers in descending order
    files_with_numbers.sort(key=lambda x: x[1], reverse=True)

    # Process files in descending order
    for filename, number in files_with_numbers:
        new_filename = f"trench_{number + 1}.json"
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, new_filename)

        # Check if the destination file already exists to avoid overwriting
        if not os.path.exists(destination_path):
            shutil.copy2(source_path, destination_path)
        else:
            print(f"File {new_filename} already exists in the destination folder. Skipping.")
