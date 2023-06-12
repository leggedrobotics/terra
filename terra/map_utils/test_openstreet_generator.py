import cv2
import numpy as np


def _convert_terra_img_to_cv2(img):
    img = img.astype(np.int8)
    img = np.where(img == 0, 255, img)
    img = np.where(img == -1, 0, img)
    img = np.where(img == 1, 100, img)
    img = img[..., None].repeat(3, -1)
    return img.astype(np.uint8)


if __name__ == "__main__":
    img_idx = 1
    path = f"/home/antonio/Downloads/img_generator/3_buildings/img_{img_idx}.npy"
    img = np.load(path)
    print(img)
    print(img.shape)

    print((img == 0).sum())
    img = _convert_terra_img_to_cv2(img)
    print((img == 255).sum())
    cv2.imshow("img", img.repeat(10, 0).repeat(10, 1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
