import cv2
import numpy as np
import os
from pathlib import Path
from terra.env_generation.procedural_data import (
    add_obstacles,
    add_non_dumpables,
    initialize_image,
)
from terra.env_generation.utils import color_dict, _get_img_mask

def add_dump_zones_harder(img, size, n_zones):
    w, h = img.shape[:2]
    cumulative_mask = np.zeros_like(img[..., 0], dtype=bool)
    min_edge_dist = 9  # Increased from 5 to 9 to place dump zones further from edges
    for _ in range(n_zones):
        sizeox = size
        sizeoy = size
        placed = False
        for _ in range(20):  # Try 20 times to place without overlap
            x = np.random.randint(min_edge_dist, w - sizeox - min_edge_dist)
            y = np.random.randint(min_edge_dist, h - sizeoy - min_edge_dist)
            if np.all(cumulative_mask[x : x + sizeox, y : y + sizeoy] == 0):
                img[x : x + sizeox, y : y + sizeoy] = np.array(color_dict["dumping"])
                cumulative_mask[x : x + sizeox, y : y + sizeoy] = True
                placed = True
                break
        if not placed:
            print("Warning: Could not place all dump zones without overlap.")
    return img, cumulative_mask

def add_dirt_tiles_harder(img, occ, dmp, cumulative_mask, size, total_dirt_tiles):
    w, h = img.shape[:2]
    drt = np.ones_like(img) * 255
    mask_occ = _get_img_mask(occ, color_dict["obstacle"])
    mask_dmp = _get_img_mask(dmp, color_dict["nondumpable"])
    
    # Fixed number of dirt spots (3 zones)
    n_spots = 3
    
    # Distribute total dirt tiles across 3 spots
    remaining_dirt = total_dirt_tiles
    dirt_spots = []
    
    # Create 3 dirt spots with varying sizes
    for spot in range(n_spots):
        if remaining_dirt <= 0:
            break
            
        # For the last spot, use all remaining dirt
        if spot == n_spots - 1:
            spot_size = remaining_dirt
        else:
            # Randomly assign dirt to this spot (at least 8, at most remaining/2)
            min_spot_size = max(8, remaining_dirt // 4)
            max_spot_size = min(remaining_dirt - (n_spots - spot - 1) * min_spot_size, remaining_dirt // 2)
            spot_size = np.random.randint(min_spot_size, max_spot_size + 1)
        
        # Calculate dirt patch dimensions (approximate square)
        patch_size = int(np.sqrt(spot_size)) + 1
        
        # Try to place this dirt patch
        placed = False
        for _ in range(30):  # Try 30 times to place without overlap
            x = np.random.randint(5, w - patch_size - 5)
            y = np.random.randint(5, h - patch_size - 5)
            
            if (
                np.all(cumulative_mask[x : x + patch_size, y : y + patch_size] == 0)
                and np.all(mask_occ[x : x + patch_size, y : y + patch_size] == 0)
                and np.all(mask_dmp[x : x + patch_size, y : y + patch_size] == 0)
            ):
                # Place dirt tiles in this patch
                tiles_placed = 0
                for dx in range(patch_size):
                    for dy in range(patch_size):
                        if tiles_placed < spot_size:
                            if (x + dx < w and y + dy < h and 
                                cumulative_mask[x + dx, y + dy] == 0 and
                                mask_occ[x + dx, y + dy] == 0 and
                                mask_dmp[x + dx, y + dy] == 0):
                                
                                drt[x + dx, y + dy] = np.array(color_dict["dirt"])
                                cumulative_mask[x + dx, y + dy] = True
                                tiles_placed += 1
                
                dirt_spots.append((x, y, tiles_placed))
                remaining_dirt -= tiles_placed
                placed = True
                break
        
        if not placed:
            print(f"Warning: Could not place dirt spot {spot + 1}.")
    
    if remaining_dirt > 0:
        print(f"Warning: Could not place all {total_dirt_tiles} dirt tiles. Placed {total_dirt_tiles - remaining_dirt} tiles.")
    
    return drt, cumulative_mask

def save_action_image(drt, save_folder, i):
    os.makedirs(save_folder, exist_ok=True)
    save_folder_action = Path(save_folder) / "actions"
    save_folder_action.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_folder_action, "trench_" + str(i) + ".png"), drt
    )

def create_relocations_harder(n_imgs, save_folder):
    img_edge = 64
    dump_zone_size = 12
    dirt_zone_size = 7
    n_obs_min = 1
    n_obs_max = 2
    size_obstacle_min = 4
    size_obstacle_max = 8
    n_nodump_min = 0
    n_nodump_max = 0
    size_nodump_min = 4
    size_nodump_max = 8
    for i in range(n_imgs):
        img = initialize_image(img_edge, img_edge, color_dict["neutral"])
        n_dumps = 1#np.random.randint(1, 3)  # 1 or 2 dump zones
        img, cumulative_mask = add_dump_zones_harder(img, dump_zone_size, n_dumps)
        occ, cumulative_mask = add_obstacles(
            img=np.ones_like(img) * np.array(color_dict["neutral"], dtype=np.uint8),
            cumulative_mask=cumulative_mask,
            n_obs_min=n_obs_min,
            n_obs_max=n_obs_max,
            size_obstacle_min=size_obstacle_min,
            size_obstacle_max=size_obstacle_max
        )
        dmp, cumulative_mask = add_non_dumpables(
            img=np.ones_like(img) * np.array(color_dict["neutral"], dtype=np.uint8),
            occ=occ,
            cumulative_mask=cumulative_mask,
            n_nodump_min=n_nodump_min,
            n_nodump_max=n_nodump_max,
            size_nodump_min=size_nodump_min,
            size_nodump_max=size_nodump_max
        )
        # Random total dirt tiles between 40-50
        total_dirt_tiles = np.random.randint(40, 51)  # 40 to 50 dirt tiles
        drt, cumulative_mask = add_dirt_tiles_harder(img, occ, dmp, cumulative_mask, dirt_zone_size, total_dirt_tiles)
        # Save main image (only dump zone and background)
        Path(save_folder, "images").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(save_folder, "images", f"img_{i+1}.png")), img)
        # Save action map (dirt only, in actions folder)
        save_action_image(drt, save_folder, i+1)
        # Save occupancy map (obstacle only, in occupancy folder)
        save_folder_occ = Path(save_folder) / "occupancy"
        save_folder_occ.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(save_folder_occ, f"img_{i+1}.png"), occ)
        # Save dumpability map (nondumpables only, in dumpability folder)
        save_folder_dump = Path(save_folder) / "dumpability"
        save_folder_dump.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(save_folder_dump, f"img_{i+1}.png"), dmp)
    print(f"Hard relocation maps created in {save_folder}") 