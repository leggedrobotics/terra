import numpy as np
import os
from pathlib import Path


def generate_onetile(n_imgs, x_dim, y_dim, save_folder=None):
    """
    n_imgs, img_edge_min, img_edge_max, resolution=0.1, option=1, save_folder=save_folder
    option 1: visualize
    option 2: save to disk
    """

    for i in range(1, n_imgs+1):
        img = np.ones((x_dim, y_dim))
        x = np.random.randint(0, x_dim, ())
        y = np.random.randint(0, y_dim, ())
        img[x, y] = -1
        img = img.astype(np.int8)

        occ = np.zeros_like(img)

        save_folder_images = Path(save_folder) / "images"
        save_folder_occupancy = Path(save_folder) / "occupancy"
        save_folder_images.mkdir(parents=True, exist_ok=True)
        save_folder_occupancy.mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_folder_images, "img_" + str(i)), img)
        np.save(os.path.join(save_folder_occupancy, "img_" + str(i)), occ)
        print(f"Generated onetile {i}")


if __name__ == "__main__":
    n_imgs = 1000
    x_dim = 60
    y_dim = 60
    package_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = package_dir + '/../data/openstreet/train/onetile/terra'
    generate_onetile(n_imgs, x_dim, y_dim, save_folder=save_folder)
