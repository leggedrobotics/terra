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


def _convert_img_to_terra(img):
    """
    Converts an image from [0, 255] convention
    to [-1, 0, 1] convention.
    """
    img = img[..., 0].astype(np.int16)  # squeeze to 1 channel
    img = np.where(img == 0, -1, img)
    img = np.where(img == 100, 1, img)
    img = np.where(img == 255, 0, img)
    return img.astype(np.int8)


def _handle_options(img, option: int, img_path: str | None):
    """
    The input image is a NxMx3 uint8 image, where:
    0 --> dig
    100 --> dump
    255 --> nothing

    option: [1, 2]
    1 --> save image at path (converted to [-1, 0, 1])
    2 --> show img
    """
    if option == 1:
        img = _convert_img_to_terra(img)
        np.save(img_path, img)
    elif option == 2:
        img = img.astype(np.uint8).repeat(10, 0).repeat(10, 1)
        cv2.imshow("buildings", img.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise RuntimeError(f"Option {option} doesn't exist.")


def generate_openstreet_2(
    wm, hm: int, div: int, option: int = 1, max_n_imgs: int | None = None
):
    """
    option: [1, 2]
    1 --> save image at path
    2 --> show imgs
    """
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
        if max_n_imgs is not None and i >= max_n_imgs:
            break

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
            p[x1 : x1 + pic1.shape[0], y1 : y1 + pic1.shape[1]] = np.where(
                pic1 < 255, 0, pic1
            )
            mask1 = p == 0

            x2 = np.random.randint(0, w - pic2.shape[0])
            y2 = np.random.randint(0, h - pic2.shape[1])
            p1[x2 : x2 + pic2.shape[0], y2 : y2 + pic2.shape[1]] = pic2
            mask2 = p1 == 0
            overlapping = np.any(mask1 * mask2)
            if not overlapping:
                p[x2 : x2 + pic2.shape[0], y2 : y2 + pic2.shape[1]] = np.where(
                    pic2 < 255, 100, pic2
                )
                break

        if not overlapping:
            # pd = cv2.resize(p, (40, 40)).repeat(10, 0).repeat(10, 1)
            pd = skimage.measure.block_reduce(
                p, (p.shape[0] // wm, p.shape[1] // hm, 1), np.min
            )

            _handle_options(
                pd,
                option,
                f"/home/antonio/Downloads/img_generator/2_buildings/{wm}x{hm}/img_{i}.npy",
            )

            continue
        else:
            n_overlapping += 1
            print(f"{n_overlapping=}")

    print(f"Generated {i + 1} images.")


def generate_openstreet_3(
    wm, hm: int, div: int, option: int = 1, max_n_imgs: int | None = None
):
    """
    option: [1, 2]
    1 --> save image at path
    2 --> show imgs
    """
    w, h = div * wm, div * hm

    idx_list = _filter_buildings_on_dims(40, 40)
    n = len(idx_list)
    idx_list = np.array(idx_list)
    idx_list_plain = idx_list.copy()
    idx_list_a = idx_list[None].repeat(n, 0).reshape(-1)
    idx_list_b = idx_list[:, None].repeat(n, 1).reshape(-1)
    idx_list = np.vstack([idx_list_a, idx_list_b])  # 2x(N^2) -> all combinations of 2

    img_idx = 0
    for idx3 in idx_list_plain.tolist():
        if max_n_imgs is not None and img_idx >= max_n_imgs:
            break
        np.random.shuffle(idx_list.swapaxes(0, 1))
        n_combinations = idx_list.shape[-1]

        n_overlapping = 0
        for i in range(n_combinations):
            if max_n_imgs is not None and img_idx >= max_n_imgs:
                break

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
                p[x1 : x1 + pic1.shape[0], y1 : y1 + pic1.shape[1]] = np.where(
                    pic1 < 255, 0, pic1
                )
                mask1 = p < 255

                x2 = np.random.randint(0, w - pic2.shape[0])
                y2 = np.random.randint(0, h - pic2.shape[1])
                p1[x2 : x2 + pic2.shape[0], y2 : y2 + pic2.shape[1]] = pic2
                mask2 = p1 < 255

                overlapping = np.any(mask1 * mask2)
                overlapping2 = True
                if not overlapping:
                    x3 = np.random.randint(0, w - pic3.shape[0])
                    y3 = np.random.randint(0, h - pic3.shape[1])
                    p2[x3 : x3 + pic3.shape[0], y3 : y3 + pic3.shape[1]] = pic3
                    mask3 = p2 < 255

                    overlapping2 = np.any(mask1 * mask2 * mask3)
                    if not overlapping2:
                        p[x2 : x2 + pic2.shape[0], y2 : y2 + pic2.shape[1]] = np.where(
                            pic2 < 255, 100, pic2
                        )
                        p[x3 : x3 + pic3.shape[0], y3 : y3 + pic3.shape[1]] = np.where(
                            pic3 < 255, 0, pic3
                        )
                        break

            if not overlapping2:
                # pd = cv2.resize(p, (40, 40)).repeat(10, 0).repeat(10, 1)
                pd = skimage.measure.block_reduce(
                    p, (p.shape[0] // wm, p.shape[1] // hm, 1), np.min
                )

                # p_grey = np.where(p < 255, 100, p).astype(np.uint8)

                _handle_options(
                    pd,
                    option,
                    f"/home/antonio/Downloads/img_generator/3_buildings/{wm}x{hm}/img_{img_idx}.npy",
                )
                img_idx += 1
                continue
            else:
                n_overlapping += 1
                print(f"{n_overlapping=}")

    print(f"Generated {img_idx + 1} images.")


if __name__ == "__main__":
    # TODO implement tile size
    wm, hm = 60, 60  # meters
    div = 10
    generate_openstreet_2(wm, hm, div, option=1, max_n_imgs=5)
