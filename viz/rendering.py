import math
import numpy as np
from abc import abstractmethod, ABCMeta

def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape([img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img


def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout


def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn


def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn


def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """

    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img


OBJECT_TO_IDX = {
    "empty": 0,
    "goal": -1,
    "wall": 2,
    "ramp": 3,
    "agent": 1,
}

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))


class GridObject(metaclass=ABCMeta):
    """Base class for objects are present in the environment"""

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color

        self.type = type
        self.color = color
        # self.orientable = False
        # self.current_pos = None

    @abstractmethod
    def can_overlap(self):
        raise NotImplementedError

    def encode(self):
        """Encode the object type into and integer"""
        return (
            OBJECT_TO_IDX[self.type],
            IDX_TO_COLOR[OBJECT_TO_IDX[self.type] % 10],
            32,
        )

class AgentObject(GridObject):
    def __init__(self, type="agent", color="red"):
        super().__init__(type, color)

    def can_overlap(self):
        return True


class RenderingEngine:
    def __init__(self, x_dim, y_dim) -> None:
        self.tile_cache = {}
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.grid_object = [None] * (self.x_dim * self.y_dim)

    # @classmethod
    def render_tile(
            self, obj, height, base_dir=None, cabin_dir=None, tile_size=32, subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache

        key = str(tile_size) + "h" + str(height)
        key = obj.type + key if obj else key
        # key = obj.encode() if obj else key

        if key in self.tile_cache and obj is None:
            return self.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)

        # if obj != None:
        #     obj.render(img)
        # else:
        fill_coords(
            img,
            point_in_rect(0, 1, 0, 1),
            np.array([255, 255, 255]) * (height + 3) / 7,
        )
        fill_coords(
            img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100)
        )
        fill_coords(
            img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100)
        )

        # Overlay the agent on top
        if base_dir is not None and cabin_dir is not None:
            # draw the base a yellow rectangle with one side longer than the other
            # to make it easier to see the direction
            back_base_fn = point_in_rect(
                0.25, 0.75, 0.0, 0.25
            )

            back_base_fn = rotate_fn(
                back_base_fn, cx=0.5, cy=0.5, theta=-np.pi / 2 + np.pi / 2 * base_dir
            )
            # render in black
            fill_coords(
                img, back_base_fn, (0, 0, 0))

            base_fn = point_in_rect(
                0.25, 0.75, 0.25, 1
            )

            base_fn = rotate_fn(
                base_fn, cx=0.5, cy=0.5, theta=-np.pi / 2 + np.pi / 2 * base_dir
            )

            fill_coords(img, base_fn, (255, 255, 0))

            tri_fn = point_in_triangle(
                (0.12, 0.81),
                (0.12, 0.19),
                (0.87, 0.50),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=np.pi / 4 * cabin_dir
            )
            fill_coords(img, tri_fn, (255, 0, 0))

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile

        self.tile_cache[key] = img

        return img
    
    def get(self, i: int, j: int) -> GridObject:
        """Retrieve object at location (i, j)

        Args:
            i (int): index of the x location
            j (int): index of the y location
        """
        assert i <= self.x_dim, "Grid index i out of bound"
        assert j <= self.y_dim, "Grid index j out of boudns"
        return self.grid_object[i + self.x_dim * j]
    
    # def _agent_occupancy_from_pos(agent_pos, agent_width, agent_height, base_dir):
    #     pass


    def render_grid(
            self,
            tile_size,
            height_grid,
            agent_pos=None,
            base_dir=None,
            cabin_dir=None,
            agent_width=None,
            agent_height=None,
            render_objects=True,
            target_height=False
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        # Compute the total grid size
        width_px = self.x_dim * tile_size
        height_px = self.y_dim * tile_size

        # img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        img = np.zeros(shape=(width_px, height_px, 3), dtype=np.uint8)

        # Render the grid
        for i in range(0, self.x_dim):
            for j in range(0, self.y_dim):

                # if render_objects:
                #     cell = self.get(i, j)
                # else:
                #     cell = None

                # print(f"{cell=}")

                if target_height:
                    agent_here = False
                else:
                    # agent_occupancy = _agent_occupancy_from_pos(agent_pos, agent_width, agent_height, base_dir)
                    agent_here = np.array_equal(agent_pos, (i, j))
                
                tile_img = self.render_tile(
                    AgentObject() if agent_here else None,
                    height_grid[i, j],
                    base_dir=base_dir if agent_here else None,
                    cabin_dir=cabin_dir if agent_here else None,
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size

                # img[ymin:ymax, xmin:xmax, :] = tile_img
                img[xmin:xmax, ymin:ymax, :] = tile_img

        return img.transpose(0, 1, 2)

# def render(
#         self,
#         mode="human",
#         close=False,
#         block=False,
#         key_handler=None,
#         highlight=False,
#         tile_size=SIZE_TILE_PIXELS,
# ):
#     """
#     Render the whole-grid human view
#     """
#     self.place_obj_at_pos(AgentObj(), self.agent_pos)

#     if close:
#         if self.window:
#             self.window.close()
#         if self.window_target:
#             self.window_target.close()
#         return

#     if mode == "human" and not self.window:
#         self.window = heightgrid.window.Window("heightgrid")

#     # Render the whole grid
#     img = self.render_grid(
#         tile_size, self.image_obs[:, :, 0], self.agent_pos, self.base_dir, self.cabin_dir
#     )

#     img_target = self.render_grid(
#         tile_size,
#         self.image_obs[:, :, 1],
#         self.agent_pos,
#         self.base_dir,
#         self.cabin_dir,
#         render_objects=True,
#         target_height=True
#     )

#     # white row of pixels
#     img_white = np.ones(shape=(tile_size * self.x_dim, tile_size, 3), dtype=np.uint8) * 255

#     img = np.concatenate((img_white, img, img_white, img_target), axis=1)
#     # add a row of white pixels at the bottom
#     white_row = np.ones(shape=(tile_size, tile_size * self.y_dim * 2 + 2 * tile_size, 3), dtype=np.uint8) * 255
#     img = np.concatenate((img, white_row), axis=0)

#     if key_handler:
#         if mode == "human":
#             # self.window.set_caption(self.mission)
#             self.window.show_img(img)
#             # self.window_target.show_img(img_target)
#             # manually controlled
#             self.window.reg_key_handler(key_handler)
#             # self.window_target.reg_key_handler(key_handler)
#             self.window.show(block=block)
#             # self.window_target.show(block=block)

#     return img
