import json

import cv2
import numpy as np
import skimage.measure


def _filter_buildings_on_dims(w_max: float, h_max: float) -> list[int]:
    idx_list = []
    for i in range(2700):
        try:
            with open(f"/home/antonio/Downloads/metadata/building_{i}.json") as f:
                meta1 = json.load(f)
            w1 = int(meta1["real_dimensions"]["length"])
            h1 = int(meta1["real_dimensions"]["width"])
            if w1 < w_max and h1 < h_max:
                idx_list.append(i)
        except:
            continue
    return idx_list


def _get_building_h_w(idx: int) -> tuple[float, float]:
    with open(f"/home/antonio/Downloads/metadata/building_{idx}.json") as f:
        meta1 = json.load(f)
    w1 = int(meta1["real_dimensions"]["length"])
    h1 = int(meta1["real_dimensions"]["width"])
    return w1, h1


def generate_openstreet_2(wm, hm: int, div: int):
    w, h = div * wm, div * hm

    idx_list = _filter_buildings_on_dims(40, 40)
    n = len(idx_list)
    idx_list = np.array(idx_list)
    idx_list_a = idx_list[None].repeat(n, 0).reshape(-1)
    idx_list_b = idx_list[:, None].repeat(n, 1).reshape(-1)
    idx_list = np.vstack([idx_list_a, idx_list_b])  # 2x(N^2) -> all combinations
    np.random.shuffle(idx_list.swapaxes(0, 1))
    n_combinations = idx_list.shape[-1]

    n_overlapping = 0
    for i in range(n_combinations):
        idx1 = idx_list[0, i]
        idx2 = idx_list[1, i]

        building_img_path1 = f"/home/antonio/Downloads/images/building_{idx1}.png"
        building_img_path2 = f"/home/antonio/Downloads/images/building_{idx2}.png"

        w1, h1 = _get_building_h_w(idx1)
        w2, h2 = _get_building_h_w(idx2)

        pic1 = cv2.imread(building_img_path1)
        pic2 = cv2.imread(building_img_path2)

        p = np.ones((w, h, 3)) * 255
        p1 = p.copy()

        pic1 = cv2.resize(pic1, (w1 * div, h1 * div)).astype(
            np.uint8
        )  # , interpolation=cv2.INTER_AREA)
        pic2 = cv2.resize(pic2, (w2 * div, h2 * div)).astype(
            np.uint8
        )  # , interpolation=cv2.INTER_AREA)

        for _ in range(40):
            x1 = np.random.randint(0, w - pic1.shape[0])
            y1 = np.random.randint(0, h - pic1.shape[1])
            p[x1 : x1 + pic1.shape[0], y1 : y1 + pic1.shape[1]] = pic1
            mask1 = p == 0

            x2 = np.random.randint(0, w - pic2.shape[0])
            y2 = np.random.randint(0, h - pic2.shape[1])
            p1[x2 : x2 + pic2.shape[0], y2 : y2 + pic2.shape[1]] = pic2
            mask2 = p1 == 0
            overlapping = np.any(mask1 * mask2)
            if not overlapping:
                p[x2 : x2 + pic2.shape[0], y2 : y2 + pic2.shape[1]] = pic2
                break

        if not overlapping:
            # pd = cv2.resize(p, (40, 40)).repeat(10, 0).repeat(10, 1)
            pd = (
                skimage.measure.block_reduce(
                    p, (p.shape[0] // wm, p.shape[1] // hm, 1), np.min
                )
                .astype(np.uint8)
                .repeat(10, 0)
                .repeat(10, 1)
            )

            pd = np.where(pd == 255, 255, 0).astype(np.uint8)
            p_grey = np.where(p < 255, 100, p).astype(np.uint8)
            # p += pic1 + pic2

            # cv2.imshow("buildings", p)
            # cv2.imshow("building downsampled", pd)
            cv2.imshow(
                "building comparison", np.where(p_grey == 100, 100, pd).astype(np.uint8)
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue
        else:
            n_overlapping += 1
            print(f"{n_overlapping=}")


def generate_openstreet_3(wm, hm: int, div: int):
    w, h = div * wm, div * hm

    idx_list = _filter_buildings_on_dims(40, 40)
    n = len(idx_list)
    idx_list = np.array(idx_list)
    idx_list_plain = idx_list.copy()
    idx_list_a = idx_list[None].repeat(n, 0).reshape(-1)
    idx_list_b = idx_list[:, None].repeat(n, 1).reshape(-1)
    idx_list = np.vstack([idx_list_a, idx_list_b])  # 2x(N^2) -> all combinations of 2

    for idx3 in idx_list_plain.tolist():
        np.random.shuffle(idx_list.swapaxes(0, 1))
        n_combinations = idx_list.shape[-1]

        n_overlapping = 0
        for i in range(n_combinations):
            idx1 = idx_list[0, i]
            idx2 = idx_list[1, i]

            building_img_path1 = f"/home/antonio/Downloads/images/building_{idx1}.png"
            building_img_path2 = f"/home/antonio/Downloads/images/building_{idx2}.png"
            building_img_path3 = f"/home/antonio/Downloads/images/building_{idx3}.png"

            w1, h1 = _get_building_h_w(idx1)
            w2, h2 = _get_building_h_w(idx2)
            w3, h3 = _get_building_h_w(idx3)

            pic1 = cv2.imread(building_img_path1)
            pic2 = cv2.imread(building_img_path2)
            pic3 = cv2.imread(building_img_path3)

            p = np.ones((w, h, 3)) * 255
            p1 = p.copy()
            p2 = p.copy()

            pic1 = cv2.resize(pic1, (w1 * div, h1 * div)).astype(np.uint8)
            pic2 = cv2.resize(pic2, (w2 * div, h2 * div)).astype(np.uint8)
            pic3 = cv2.resize(pic3, (w3 * div, h3 * div)).astype(np.uint8)

            for _ in range(40):
                x1 = np.random.randint(0, w - pic1.shape[0])
                y1 = np.random.randint(0, h - pic1.shape[1])
                p[x1 : x1 + pic1.shape[0], y1 : y1 + pic1.shape[1]] = pic1
                mask1 = p == 0

                x2 = np.random.randint(0, w - pic2.shape[0])
                y2 = np.random.randint(0, h - pic2.shape[1])
                p1[x2 : x2 + pic2.shape[0], y2 : y2 + pic2.shape[1]] = pic2
                mask2 = p1 == 0

                overlapping = np.any(mask1 * mask2)
                overlapping2 = True
                if not overlapping:
                    x3 = np.random.randint(0, w - pic3.shape[0])
                    y3 = np.random.randint(0, h - pic3.shape[1])
                    p2[x3 : x3 + pic3.shape[0], y3 : y3 + pic3.shape[1]] = pic3
                    mask3 = p2 == 0

                    overlapping2 = np.any(mask1 * mask2 * mask3)
                    if not overlapping2:
                        p[x2 : x2 + pic2.shape[0], y2 : y2 + pic2.shape[1]] = pic2
                        p[x3 : x3 + pic3.shape[0], y3 : y3 + pic3.shape[1]] = pic3
                        break

            if not overlapping2:
                # pd = cv2.resize(p, (40, 40)).repeat(10, 0).repeat(10, 1)
                pd = (
                    skimage.measure.block_reduce(
                        p, (p.shape[0] // wm, p.shape[1] // hm, 1), np.min
                    )
                    .astype(np.uint8)
                    .repeat(10, 0)
                    .repeat(10, 1)
                )

                pd = np.where(pd == 255, 255, 0).astype(np.uint8)
                p_grey = np.where(p < 255, 100, p).astype(np.uint8)
                # p += pic1 + pic2

                # cv2.imshow("buildings", p)
                # cv2.imshow("building downsampled", pd)
                cv2.imshow(
                    "building comparison",
                    np.where(p_grey == 100, 100, pd).astype(np.uint8),
                )
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                continue
            else:
                n_overlapping += 1
                print(f"{n_overlapping=}")


if __name__ == "__main__":
    # TODO implement tile size
    wm, hm = 60, 60  # meters
    div = 10
    generate_openstreet_3(wm, hm, div)
