import json

import cv2
import numpy as np
import skimage.measure

w, h = 300, 300  # meters and pixels
wm, hm = w // 10, h // 10
i, j = 4, 5

building_img_path1 = f"/home/antonio/Downloads/images/building_{i}.png"
building_img_path2 = f"/home/antonio/Downloads/images/building_{j}.png"

with open(f"/home/antonio/Downloads/metadata/building_{i}.json") as f:
    meta1 = json.load(f)
w1 = int(meta1["real_dimensions"]["length"])
h1 = int(meta1["real_dimensions"]["width"])

with open(f"/home/antonio/Downloads/metadata/building_{j}.json") as f:
    meta2 = json.load(f)
w2 = int(meta2["real_dimensions"]["length"])
h2 = int(meta2["real_dimensions"]["width"])

pic1 = cv2.imread(building_img_path1)
pic2 = cv2.imread(building_img_path2)
print(f"{pic1.shape=}")
print(f"{pic2.shape=}")

p = np.ones((w, h, 3)) * 255

pic1 = cv2.resize(pic1, (w1, h1)).astype(np.uint8)  # , interpolation=cv2.INTER_AREA)
pic2 = cv2.resize(pic2, (w2, h2)).astype(np.uint8)  # , interpolation=cv2.INTER_AREA)


print(f"{pic1.shape=}")
print(f"{pic2.shape=}")

p[: pic1.shape[0], : pic1.shape[1]] = pic1
p[-pic2.shape[0] :, -pic2.shape[1] :] = pic2

# pd = cv2.resize(p, (40, 40)).repeat(10, 0).repeat(10, 1)
pd = (
    skimage.measure.block_reduce(p, (p.shape[0] // wm, p.shape[1] // hm, 1), np.min)
    .astype(np.uint8)
    .repeat(10, 0)
    .repeat(10, 1)
)
print(f"{pd.shape=}")

pd = np.where(pd == 255, 255, 0).astype(np.uint8)
p_grey = np.where(p < 255, 100, p)
# p += pic1 + pic2

# cv2.imshow("building1", pic1)
# cv2.imshow("building2", pic2)
cv2.imshow("buildings", p)
cv2.imshow("building downsampled", pd)
cv2.imshow("building comparison", np.where(p_grey == 100, 100, pd).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
