import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_rectangles(map_size, n_maps, destination_folder, small, all_dumpable):
    for i in tqdm(range(1, n_maps + 1)):
        p = -np.ones((4, 2))
        while ~(np.all(p > 0) & np.all(p[:, 0] < map_size[0]) & np.all(p[:, 1] < map_size[1])):
            imgc = np.zeros((map_size[0], map_size[1], 3), np.uint8)
            if small:
                margin_x = int(0.4 * map_size[0])
                margin_y = int(0.48 * map_size[1])
            else:
                margin_x = int(0.3 * map_size[0])
                margin_y = int(0.4 * map_size[1])

            contour = 6
            border = 3
            x = np.array([margin_x, map_size[0] - margin_x])
            y = np.array([margin_y, map_size[1] - margin_y])
            xc = np.array([margin_x - contour, map_size[0] - margin_x + contour])
            yc = np.array([margin_y - contour, map_size[1] - margin_y + contour])
            xb = np.array([margin_x - border, map_size[0] - margin_x + border])
            yb = np.array([margin_y - border, map_size[1] - margin_y + border])
            # x = np.random.randint(margin_x, map_size[0] - margin_x + 1, (2,))
            # y = np.random.randint(margin_y, map_size[1] - margin_y + 1, (2,))

            def get_rotated_rectangle(x, y, theta):
                min_x = x.min()
                max_x = x.max()
                min_y = y.min()
                max_y = y.max()
                p = np.array(
                    [
                        [min_x, min_y],
                        [min_x, max_y],
                        [max_x, max_y],
                        [max_x, min_y],
                    ],
                    dtype=np.int32
                )

                center = np.array([np.mean(x), np.mean(y)])
                R = np.array(
                    [
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                    ]
                )
                p = ((R @ (p - center).T).T + center).astype(np.int32)
                return p
            
            theta = np.random.uniform(0, 2.*np.pi)
            p = get_rotated_rectangle(x, y, theta)
            pc = get_rotated_rectangle(xc, yc, theta)
            pb = get_rotated_rectangle(xb, yb, theta)
            cv2.drawContours(imgc, [pc], 0, (120, 120, 120), -1, cv2.LINE_AA)
            imgc = np.where(imgc != 0, 120, imgc)
            cv2.drawContours(imgc, [pb], 0, (0, 0, 0), -1, cv2.LINE_AA)
            imgc = np.where(imgc != 0, 120, imgc)
            cv2.drawContours(imgc, [p], 0, (255, 255, 255), -1, cv2.LINE_AA)
            imgc = np.where((imgc != 0) & (imgc != 120), 255, imgc)
            # cv2.imshow("imgc", imgc)
            # cv2.waitKey(0)

        imgc = imgc.astype(np.int32)[..., 2]
        img = np.where(imgc == 255, -1, imgc).astype(np.int8)
        img = np.where(img == 120, 1, img).astype(np.int8)

        if all_dumpable:
            img = np.where(img == 0, 1, img).astype(np.int8)

        occ = np.zeros_like(img)
        Path(destination_folder + "/images").mkdir(parents=True, exist_ok=True)
        Path(destination_folder + "/occupancy").mkdir(parents=True, exist_ok=True)
        np.save(f"{destination_folder}/images/img_{i}", img)
        np.save(f"{destination_folder}/occupancy/img_{i}", occ)

if __name__ == "__main__":
    map_size = (60, 60)
    n_maps = 1000
    destination_folder = f"/home/antonio/thesis/digbench/data/openstreet/train/benchmark_{map_size[0]}_{map_size[1]}/terra/rectangles_60"
    generate_rectangles(map_size, n_maps, destination_folder, small=False, all_dumpable=False)
