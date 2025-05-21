import numpy as np

from terra.env_generation.procedural_data import (
    initialize_image
)

from terra.env_generation.utils import color_dict

def add_dumpables(img, n_dump_min, n_dump_max, size_dump_min, size_dump_max):
    w, h = img.shape[:2]
    n_dump = 0
    n_dump_todo = np.random.randint(n_dump_min, n_dump_max + 1)

    while n_dump < n_dump_todo:
        # Randomly determine the size of the dumping zone
        sizeox = np.random.randint(size_dump_min, size_dump_max + 1)
        sizeoy = np.random.randint(size_dump_min, size_dump_max + 1)
        # Randomly select a position for the dumping zone
        x = np.random.randint(0, w - sizeox)
        y = np.random.randint(0, h - sizeoy)
        # Update the dumping zone layer
        img[x : x + sizeox, y : y + sizeoy] = np.array(color_dict["dumping"])
        n_dump += 1  # Increment the count of zones added

    return img

def generate_relocations(
    n_imgs,
    img_edge_min,
    img_edge_max,
    n_dump_min,
    n_dump_max,
    size_dump_min,
    size_dump_max,
):
    i = 0
    while i < n_imgs:
        img = initialize_image(img_edge_min, img_edge_max, color_dict["neutral"])
        img = add_dumpables(img, n_dump_min, n_dump_max, size_dump_min, size_dump_max)
        # TODO: Add spilled soil to the image
        # TODO: Add non-dumpable zones to the image
        # TODO: Add obstacles to the image
        # TODO: Save the image to a file
        