import numpy as np
import os
from pathlib import Path
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def generate_squares(n_imgs, x_dim, y_dim, side_lens, save_folder_base):
    for side_len in side_lens:
        # Dynamic calculation of maximum margin and contour based on map size and side length
        max_possible_margin = min(x_dim, y_dim) - (3 * side_len)  # Initial assumption
        margin = max(1, max_possible_margin // 8)  # Ensure some margin, but limit to available space
        side_len_contour = 3 * side_len + 2 * margin  # Adjust contour size based on margin
        
        for i in range(1, n_imgs+1):
            if x_dim < side_len_contour or y_dim < side_len_contour:
                print(f"Skipping generation for side length {side_len} as it cannot fit within the given dimensions.")
                continue
            img = np.zeros((x_dim, y_dim))
            occ = np.zeros_like(img)
            dmp = np.ones_like(img)

            x = np.random.randint(0, x_dim - side_len_contour + 1)
            y = np.random.randint(0, y_dim - side_len_contour + 1)

            img[x:x+side_len_contour, y:y+side_len_contour] = 1  # must dump
            x_dig = x + margin
            y_dig = y + margin
            img[x_dig:x_dig+side_len, y_dig:y_dig+side_len] = -1  # dig
            img = img.astype(np.int8)
            
            # Adjusting non-dumpable and occupation squares placement
            for _ in range(2):  # Non-dumpable squares
                x = np.random.randint(0, max(1, x_dim - 3))
                y = np.random.randint(0, max(1, y_dim - 3))
                dmp[x:x+3, y:y+3] = 0
            
            dmp = np.where(img == 1, 1, dmp).astype(np.bool_)

            for _ in range(2):  # Occupation squares
                x = np.random.randint(0, max(1, x_dim - 3))
                y = np.random.randint(0, max(1, y_dim - 3))
                occ[x:x+3, y:y+3] = 1
            
            occ = np.where((img == 1) | (img == -1), 0, occ).astype(np.bool_)

            # Saving
            save_folder = Path(save_folder_base) / f'squares_final_{side_len}'
            save_folder_images = save_folder / "images"
            save_folder_occupancy = save_folder / "occupancy"
            save_folder_dumpability = save_folder / "dumpability"
            save_folder_images.mkdir(parents=True, exist_ok=True)
            save_folder_occupancy.mkdir(parents=True, exist_ok=True)
            save_folder_dumpability.mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(save_folder_images, f"img_{i}"), img)
            np.save(os.path.join(save_folder_occupancy, f"img_{i}"), occ)
            np.save(os.path.join(save_folder_dumpability, f"img_{i}"), dmp)
            print(f"Generated square {i} with side length {side_len}, margin {margin}")