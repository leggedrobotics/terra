import json
import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.signal import convolve2d
from skimage import measure
from tqdm import tqdm
import time

import terra.env_generation.utils as utils
from terra.env_generation.utils import _get_img_mask, color_dict


def _convert_img_to_terra(img, all_dumpable=False):
    """
    Converts an image from color_dict RGB convention
    to [-1, 0, 1] Terra convention.
    """
    img = img.astype(np.int16)
    img = np.where(img == np.array(color_dict["digging"]), -1, img)
    img = np.where(img == np.array(color_dict["dumping"]), 1, img)
    if all_dumpable:
        img = np.where(img == np.array(color_dict["neutral"]), 1, img)
    else:
        img = np.where(img == np.array(color_dict["neutral"]), 0, img)
    img = np.where((img != -1) & (img != 1), 0, img)
    img = img[..., 0]  # take only 1 channel
    return img.astype(np.int8)


def _convert_occupancy_to_terra(img):
    img = img.astype(np.int16)
    mask = _get_img_mask(img, np.array(color_dict["obstacle"]))
    img = np.where(mask, 1, 0)
    return img.astype(np.bool_)


def _convert_dumpability_to_terra(img):
    img = img.astype(np.int16)
    mask = _get_img_mask(img, np.array(color_dict["nondumpable"]))
    img = np.where(mask, 0, 1)
    return img.astype(np.bool_)


def _convert_all_imgs_to_terra(
    img_folder,
    metadata_folder,
    occupancy_folder,
    dumpability_folder,
    destination_folder,
    size,
    n_imgs,
    expansion_factor=1,
    all_dumpable=False,
    copy_metadata=True,
    downsample=True,
    has_dumpability=True,
    center_padding=False,
):
    max_size = size[1]
    print("max size: ", max_size)
    # try:

    filename_start = sorted(os.listdir(img_folder))[0].split("_")[0]

    for i, fn in tqdm(enumerate(os.listdir(img_folder))):
        if i >= n_imgs:
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
            # plt.imshow(dumpability)
            # plt.show()

        if downsample:
            with open(
                metadata_folder / f"{filename.split('.png')[0]}.json"
            ) as json_file:
                metadata = json.load(json_file)

            # Calculate downsample factors based on max_size
            downsample_factor_w = max(1, math.ceil(img.shape[1] / max_size))
            downsample_factor_h = max(1, math.ceil(img.shape[0] / max_size))

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
                # plt.imshow(dumpability)
                # plt.show()

        # assert img_downsampled.shape[:-1] == occupancy_downsampled.shape
        img_terra = _convert_img_to_terra(img, all_dumpable)
        # Pad to max size
        if center_padding:
            xdim = max_size - img_terra.shape[0]
            ydim = max_size - img_terra.shape[1]
            # Note: applying full dumping tiles for the centered version
            img_terra_pad = np.ones((max_size, max_size), dtype=img_terra.dtype)
            print(
                "xdim:",
                xdim,
                "max_size:",
                max_size,
                "ydim:",
                ydim,
                "img_terra shape:",
                img_terra.shape,
            )
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
            img_terra_occupancy = _convert_occupancy_to_terra(occupancy)
            if has_dumpability:
                img_terra_dumpability = np.zeros((max_size, max_size), dtype=np.bool_)
                img_terra_dumpability = _convert_dumpability_to_terra(dumpability)

        destination_folder_images = destination_folder / "images"
        destination_folder_occupancy = destination_folder / "occupancy"
        destination_folder_images.mkdir(parents=True, exist_ok=True)
        destination_folder_occupancy.mkdir(parents=True, exist_ok=True)
        destination_folder_dumpability = destination_folder / "dumpability"
        destination_folder_dumpability.mkdir(parents=True, exist_ok=True)
        if copy_metadata:
            destination_folder_metadata = destination_folder / "metadata"
            destination_folder_metadata.mkdir(parents=True, exist_ok=True)

        img_terra_pad = img_terra_pad.repeat(expansion_factor, axis=0).repeat(
            expansion_factor, axis=1
        )
        img_terra_occupancy = img_terra_occupancy.repeat(
            expansion_factor, axis=0
        ).repeat(expansion_factor, axis=1)
        if has_dumpability:
            img_terra_dumpability = img_terra_dumpability.repeat(
                expansion_factor, 0
            ).repeat(expansion_factor, 1)

        np.save(destination_folder_images / f"img_{i + 1}", img_terra_pad)
        np.save(destination_folder_occupancy / f"img_{i + 1}", img_terra_occupancy)
        if has_dumpability:
            np.save(
                destination_folder_dumpability / f"img_{i + 1}", img_terra_dumpability
            )
        else:
            np.save(
                destination_folder_dumpability / f"img_{i + 1}",
                np.ones_like(img_terra_pad),
            )
    if copy_metadata:
        utils.copy_and_increment_filenames(str(metadata_folder), str(destination_folder_metadata))
        # we increase the index by 1 for consistency


def generate_foundations_terra(dataset_folder, size, n_imgs, all_dumpable):
    print("Converting foundations...")
    foundations_levels = ["foundations", "foundations_large"]
    for level in foundations_levels:
        img_folder = Path(dataset_folder) / level / "images"
        metadata_folder = Path(dataset_folder) / level / "metadata"
        occupancy_folder = Path(dataset_folder) / level/ "occupancy"
        dumpability_folder = Path(dataset_folder) / level / "dumpability"
        destination_folder = Path(dataset_folder) / "train" / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(
            img_folder,
            metadata_folder,
            occupancy_folder,
            dumpability_folder,
            destination_folder,
            size,
            n_imgs,
            all_dumpable=all_dumpable,
            copy_metadata=True,
            downsample=False,
            has_dumpability=True,
            center_padding=False,
        )


def generate_trenches_terra(dataset_folder, size, n_imgs, expansion_factor, all_dumpable):
    print("Converting trenches...")
    trenches_name = "trenches"
    trenches_path = Path(dataset_folder) / trenches_name
    levels = [d.name for d in trenches_path.iterdir() if d.is_dir()]
    for level in levels:
        img_folder = trenches_path / level / "images"
        metadata_folder = trenches_path / level / "metadata"
        occupancy_folder = trenches_path / level / "occupancy"
        dumpability_folder = trenches_path / level / "dumpability"
        destination_folder = Path(dataset_folder) / "train" / trenches_name / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(
            img_folder,
            metadata_folder,
            occupancy_folder,
            dumpability_folder,
            destination_folder,
            size,
            n_imgs,
            expansion_factor=expansion_factor,
            all_dumpable=all_dumpable,
        )

def generate_custom_terra(dataset_folder, size, n_imgs, all_dumpable):
    print("Converting custom maps...")
    img_folder = Path(dataset_folder) / ".." / "custom" / "images"
    metadata_folder = Path(dataset_folder) / ".." / "custom" / "metadata"
    occupancy_folder = Path(dataset_folder) / ".." / "custom"/ "occupancy"
    dumpability_folder = Path(dataset_folder) / ".." / "custom" / "dumpability"
    destination_folder = Path(dataset_folder) / "train" / "custom"
    destination_folder.mkdir(parents=True, exist_ok=True)
    _convert_all_imgs_to_terra(
        img_folder,
        metadata_folder,
        occupancy_folder,
        dumpability_folder,
        destination_folder,
        size,
        n_imgs,
        all_dumpable=False,
        copy_metadata=False,
        downsample=False,
        has_dumpability=True,
        center_padding=False,
    )


def generate_dataset_terra_format(dataset_folder, size, n_imgs=1000):
    print("dataset folder: ", dataset_folder)
    generate_foundations_terra(dataset_folder, size, n_imgs, all_dumpable=False)
    print("Foundations processed successfully.")
    generate_trenches_terra(
        dataset_folder, size, n_imgs, expansion_factor=1, all_dumpable=False
    )
    print("Trenches processed successfully.")
    generate_custom_terra(dataset_folder, size, n_imgs)
    print("Custom maps processed successfully.")
