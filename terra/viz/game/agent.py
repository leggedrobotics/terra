import numpy as np
import math
from .settings import COLORS
from .utils import agent_base_to_angle
from .utils import agent_cabin_to_angle
from .utils import rotate_triangle


class Agent:
    def __init__(self, width, height, tile_size) -> None:
        self.width = width
        self.height = height
        self.tile_size = tile_size

    def create_agent(self, px_center, py_center, angle_base, angle_cabin, loaded):
        px = px_center - self.width // 2
        py = py_center - self.height // 2

        if angle_base in (0, 2):
            px = px_center - self.height // 2
            py = py_center - self.width // 2
            agent_body = [
                (py * self.tile_size, px * self.tile_size),
            ]
            w = self.width
            h = self.height
        elif angle_base in (1, 3):
            px = px_center - self.width // 2
            py = py_center - self.height // 2
            agent_body = [
                (py * self.tile_size, px * self.tile_size),
            ]
            w = self.height
            h = self.width

        angle_cabin = (angle_cabin + 2 * angle_base) % 8

        # Cabin (triangle)
        deg_angle_cabin = agent_cabin_to_angle(angle_cabin)
        scaling = self.tile_size / 3
        points = [
            (3 / scaling, 0),
            (-1.5 / scaling, -1.5 / scaling),
            (-1.5 / scaling, 1.5 / scaling),
        ]
        a_center_x = agent_body[0][0] + w * self.tile_size // 2
        a_center_y = agent_body[0][1] + h * self.tile_size // 2
        agent_cabin = rotate_triangle(
            (a_center_x, a_center_y), points, self.tile_size, deg_angle_cabin
        )

        out = {
            "body": {
                "vertices": agent_body,
                "width": w,
                "height": h,
                "color": COLORS["agent_body"],
            },
            "cabin": {
                "vertices": agent_cabin,
                "color": COLORS["agent_cabin"]["loaded"]
                if loaded
                else COLORS["agent_cabin"]["not_loaded"],
            },
        }
        return out

    def update(self, agent_pos, base_dir, cabin_dir, loaded):
        agent_pos = np.asarray(agent_pos, dtype=np.int32)
        base_dir = np.asarray(base_dir, dtype=np.int32)
        cabin_dir = np.asarray(cabin_dir, dtype=np.int32)
        loaded = np.asarray(loaded, dtype=bool)
        self.agent = self.create_agent(
            agent_pos[0].item(),
            agent_pos[1].item(),
            base_dir.item(),
            cabin_dir.item(),
            loaded.item(),
        )
