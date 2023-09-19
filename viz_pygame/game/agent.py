import numpy as np
import math
from .settings import TILE_SIZE
from .settings import COLORS
from .utils import agent_base_to_angle
from .utils import agent_cabin_to_angle
from .utils import rotate_triangle

class Agent:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

    def create_agent(self, px_center, py_center, angle_base, angle_cabin):
        # px = px_center - math.ceil(self.width / 2)
        # py = py_center - math.ceil(self.height / 2)
        px = px_center - self.width // 2
        py = py_center - self.height // 2


        if angle_base in (0, 2):
            px = px_center - self.height // 2
            py = py_center - self.width // 2
            agent_body = [
                (py * TILE_SIZE, px * TILE_SIZE),
            ]
            w = self.width
            h = self.height
        elif angle_base in (1, 3):
            px = px_center - self.width // 2
            py = py_center - self.height // 2
            agent_body = [
                (py * TILE_SIZE, px * TILE_SIZE),
            ]
            w = self.height
            h = self.width

        angle_cabin = (angle_cabin + 2*angle_base) % 8

        # Cabin (triangle)
        deg_angle_cabin = agent_cabin_to_angle(angle_cabin)
        points = [(3, 0), (-1.5, -1.5), (-1.5, 1.5)]
        a_center_x = agent_body[0][0] + w * TILE_SIZE // 2
        a_center_y = agent_body[0][1] + h * TILE_SIZE // 2
        agent_cabin = rotate_triangle((a_center_x, a_center_y), points, TILE_SIZE, deg_angle_cabin)

        loaded = False

        out = {
            "body": {
                "vertices": agent_body,
                "width": w,
                "height": h,
                "color": COLORS["agent_body"],
            },
            "cabin": {
                "vertices": agent_cabin,
                "color": COLORS["agent_cabin"]["loaded"] if loaded else COLORS["agent_cabin"]["not_loaded"]
            },
        }
        return out

    def update(self, agent_pos, base_dir, cabin_dir):
        agent_pos = np.asarray(agent_pos, dtype=np.int32)
        base_dir = np.asarray(base_dir, dtype=np.int32)
        cabin_dir = np.asarray(cabin_dir, dtype=np.int32)
        self.agent = self.create_agent(agent_pos[0].item(), agent_pos[1].item(), base_dir.item(), cabin_dir.item(),)
