import pygame as pg
from .settings import TILE_SIZE
from .settings import COLORS
from .utils import agent_base_to_angle
from .utils import agent_cabin_to_angle
from .utils import rotate_triangle

class Agent:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

        self.agent = self.create_agent(2, 3, 0, 2,)

    def create_agent(self, px, py, angle_base, angle_cabin):
        agent_body = [
            (px * TILE_SIZE, py * TILE_SIZE),
            (px * TILE_SIZE + TILE_SIZE, py * TILE_SIZE),
            (px * TILE_SIZE + TILE_SIZE, py * TILE_SIZE + TILE_SIZE),
            (px * TILE_SIZE, py * TILE_SIZE + TILE_SIZE)
        ]

        deg_angle_base = agent_base_to_angle(angle_base)
        deg_angle_cabin = agent_cabin_to_angle(angle_cabin)

        # Cabin (triangle)
        points = [(1, 0), (-0.5, -0.5), (-0.5, 0.5)]
        a_center_x = agent_body[0][0] + self.width * TILE_SIZE // 2
        a_center_y = agent_body[0][1] + self.height * TILE_SIZE // 2
        agent_cabin = rotate_triangle((a_center_x, a_center_y), points, TILE_SIZE, deg_angle_cabin)

        loaded = False

        out = {
            "body": {
                "vertices": agent_body,
                "color": COLORS["agent_body"],
            },
            "cabin": {
                "vertices": agent_cabin,
                "color": COLORS["agent_cabin"]["loaded"] if loaded else COLORS["agent_cabin"]["not_loaded"]
            },
        }
        return out
