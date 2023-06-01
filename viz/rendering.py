from abc import ABCMeta
from abc import abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from terra.state import State


def fill_coords(img, fn, color, batch_size):
    """
    Fill pixels of an image with coordinates matching a filter function
    """
    x = np.arange(img.shape[1], dtype=np.int16)
    y = np.arange(img.shape[2], dtype=np.int16)
    xf = (x.copy() + 0.5) / img.shape[1]
    yf = (y.copy() + 0.5) / img.shape[2]

    xf = xf[None]
    yf = yf[None]

    cond = fn(xf, yf).reshape(-1, img.shape[1], img.shape[2])

    img = np.where(cond[..., None], color, img)
    return img


def generate_combinations_xy(x, y):
    xa = x[:, :, None].repeat(y.shape[1], axis=2).reshape(1, -1)
    ya = y[:, None].repeat(x.shape[1], axis=1).reshape(1, -1)
    xya = np.concatenate([xa[:, None], ya[:, None]], axis=1)
    return xya


def rotate_fn(fin, cx, cy, theta):
    theta = -theta

    def fout(x, y):
        xya = generate_combinations_xy(x, y)
        xya[:, 0] -= cx
        xya[:, 1] -= cy
        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        ).transpose(2, 0, 1, 3)
        xya = R[..., 0] @ xya
        xya[:, 0] += cx
        xya[:, 1] += cy

        # x = x - cx
        # y = y - cy

        # x2 = cx + x * np.cos(theta) - y * np.sin(theta)
        # y2 = cy + y * np.cos(theta) + x * np.sin(theta)

        return fin(xya[:, 0], xya[:, 1])

    return fout


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        # return x >= xmin and x <= xmax and y >= ymin and y <= ymax
        x_cond = np.logical_and(x >= xmin, x <= xmax)
        y_cond = np.logical_and(y >= ymin, y <= ymax)
        return np.logical_and(x_cond, y_cond)

    return fn


def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = (c - a)[None]  # (1, 2)
        v1 = (b - a)[None]  # (1, 2)
        v2 = np.concatenate((x[..., None], y[..., None]), -1) - a  # (N, 2)

        # Compute dot products
        dot00 = (v0**2).sum(axis=-1)
        dot01 = np.multiply(v0, v1).sum(axis=-1)
        dot02 = np.multiply(v0, v2).sum(axis=-1)
        dot11 = (v1**2).sum(axis=-1)
        dot12 = np.multiply(v1, v2).sum(axis=-1)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return np.logical_and(np.logical_and(u >= 0, v >= 0), (u + v) < 1)

    return fn


def point_in_triangle_naive(a, b, c):
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


# def point_in_line(x0, y0, x1, y1, r):
#     p0 = np.array([x0, y0])
#     p1 = np.array([x1, y1])
#     dir = p1 - p0
#     dist = np.linalg.norm(dir)
#     dir = dir / dist

#     xmin = min(x0, x1) - r
#     xmax = max(x0, x1) + r
#     ymin = min(y0, y1) - r
#     ymax = max(y0, y1) + r

#     def fn(x, y):
#         # Fast, early escape test
#         if x < xmin or x > xmax or y < ymin or y > ymax:
#             return False

#         q = np.array([x, y])
#         pq = q - p0

#         # Closest point on line
#         a = np.dot(pq, dir)
#         a = np.clip(a, 0, dist)
#         p = p0 + a * dir

#         dist_to_line = np.linalg.norm(q - p)
#         return dist_to_line <= r

#     return fn


# def point_in_circle(cx, cy, r):
#     def fn(x, y):
#         return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

#     return fn


# def highlight_img(img, color=(255, 255, 255), alpha=0.30):
#     """
#     Add highlighting to an image
#     """

#     blend_img = img + alpha * (np.array(color, dtype=np.uint16) - img)
#     blend_img = blend_img.clip(0, 255).astype(np.uint16)
#     img[:, :, :] = blend_img


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
    def __init__(self) -> None:
        self.tile_cache = {}
        self.first_render = True

    def render_agent(
        self,
        agent_width,
        agent_height,
        tile_size,
        base_dir,
        cabin_dir=None,
        batch_size=1,
    ):
        imgs = [
            jnp.zeros(
                shape=(1, tile_size * agent_width[i], tile_size * agent_height[i], 3),
                dtype=jnp.uint16,
            )
            for i in range(batch_size)
        ]

        for i in range(batch_size):
            # base
            back_base_fn = point_in_rect(0.25, 0.75, 0.0, 0.25)

            back_base_fn = rotate_fn(
                back_base_fn,
                cx=0.5,
                cy=0.5,
                theta=-np.pi / 2 + np.pi / 2 * base_dir[[i]],
            )
            imgs[i] = fill_coords(
                imgs[i], back_base_fn, np.array([0, 0, 0]), batch_size
            )

            # yellow of the base
            base_fn = point_in_rect(0.25, 0.75, 0.0, 0.25)

            base_fn = rotate_fn(
                base_fn, cx=0.5, cy=0.5, theta=np.pi / 2 * base_dir[[i]] + np.pi
            )

            imgs[i] = fill_coords(imgs[i], base_fn, np.array([255, 255, 0]), batch_size)

            # red triangle
            tri_fn = point_in_triangle(
                (0.12, 0.81),
                (0.12, 0.19),
                (0.87, 0.50),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn,
                cx=0.5,
                cy=0.5,
                theta=np.pi / 4 * cabin_dir[[i]]
                + np.pi / 2 * base_dir[[i]]
                + np.pi / 2,
            )
            imgs[i] = fill_coords(imgs[i], tri_fn, (255, 0, 0), batch_size)

        return imgs

    def _render_grids(self, tile_size, height_grid):
        x_dim = height_grid.shape[-2]
        y_dim = height_grid.shape[-1]
        width_px = x_dim * tile_size
        height_px = y_dim * tile_size

        height_grid = np.array(height_grid)

        # img = np.zeros(shape=(width_px, height_px, 3), dtype=np.uint16)
        x = (
            (height_grid.repeat(tile_size, axis=-2).repeat(tile_size, axis=-1) + 3) / 7
        )[..., None]
        img = (np.array([[[255, 255, 255]]]) * x).astype(np.int16)

        # apply grid
        grid_idx_x = np.arange(start=0, stop=width_px, step=tile_size)
        grid_idx_y = np.arange(start=0, stop=height_px, step=tile_size)
        img[:, grid_idx_x] = np.array([100, 100, 100])
        img[:, :, grid_idx_y] = np.array([100, 100, 100])
        img = img.astype(np.int16)
        return img

    def render_active_grid(
        self,
        tile_size,
        height_grid,
        agent_pos=None,
        base_dir=None,
        cabin_dir=None,
        agent_width=None,
        agent_height=None,
        batch_size=1,
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        # s1 = time.time()

        imgs = self._render_grids(tile_size, height_grid)

        # s2 = time.time()

        # Render agent
        agent_corners = jax.vmap(
            lambda agent_pos, base_dir: State._get_agent_corners(
                pos_base=agent_pos,
                base_orientation=base_dir,
                agent_width=agent_width,
                agent_height=agent_height,
            )
        )(agent_pos, base_dir)

        ay_min = jax.vmap(lambda x: np.min(x[:, 1]))(agent_corners).astype(np.int16)
        ax_min = jax.vmap(lambda x: np.min(x[:, 0]))(agent_corners).astype(np.int16)
        ay_max = jax.vmap(lambda x: np.max(x[:, 1]))(agent_corners).astype(np.int16)
        ax_max = jax.vmap(lambda x: np.max(x[:, 0]))(agent_corners).astype(np.int16)

        agent_ymin = ay_min * tile_size
        agent_ymax = (ay_max + 1) * tile_size
        agent_xmin = ax_min * tile_size
        agent_xmax = (ax_max + 1) * tile_size

        agent_imgs = self.render_agent(
            ax_max - ax_min + 1,
            ay_max - ay_min + 1,
            tile_size,
            np.array(base_dir),
            np.array(cabin_dir),
            batch_size,
        )
        for i in range(batch_size):
            imgs[
                i, agent_xmin[i] : agent_xmax[i], agent_ymin[i] : agent_ymax[i], :
            ] = agent_imgs[i]

        # e = time.time()

        # print(f"grid = {100 * (s2-s1)/(e-s1)}%")
        # print(f"agent = {100 * (e-s2)/(e-s1)}%")

        return imgs

    def render_target_grids(self, tile_size, target_grid):
        return self._render_grids(tile_size, target_grid)

    def render_global(
        self,
        tile_size,
        active_grid,
        target_grid,
        agent_pos=None,
        base_dir=None,
        cabin_dir=None,
        agent_width=None,
        agent_height=None,
    ):
        # Add batch dim in case it's not there
        if len(active_grid.shape) < 3:
            active_grid = active_grid[None]
        if len(target_grid.shape) < 3:
            target_grid = target_grid[None]
        if agent_pos is not None and len(agent_pos.shape) < 2:
            agent_pos = agent_pos[None]
        if base_dir is not None and len(base_dir.shape) < 2:
            base_dir = base_dir[None]
        if cabin_dir is not None and len(cabin_dir.shape) < 2:
            cabin_dir = cabin_dir[None]

        white_margin = 0.05  # percentage
        batch_size = active_grid.shape[0]

        imgs_active_grid = self.render_active_grid(
            tile_size,
            active_grid,
            agent_pos,
            base_dir,
            cabin_dir,
            agent_width,
            agent_height,
            batch_size,
        )

        imgs_target_grid = self.render_target_grids(tile_size, target_grid)

        imgs = [
            np.hstack(
                [
                    imgs_active_grid[i],
                    255
                    * np.ones(
                        (
                            imgs_active_grid[i].shape[0],
                            int(white_margin * imgs_active_grid[i].shape[1]),
                            3,
                        )
                    ).astype(np.int16),
                    imgs_target_grid[i],
                ]
            )
            for i in range(batch_size)
        ]

        return imgs

    def render_local(self):
        pass
