
import pygame as pg
from .settings import TILE_SIZE
from .settings import COLORS
import numpy as np

class World:

    def __init__(self, grid_length_x, grid_length_y, width, height):
        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y
        self.width = width
        self.height = height

        self.world = self.create_world()

    def create_world(self):

        world = []

        # TODO replace with actual map
        bitmap_codes = np.random.choice((-1, 0, 1), (10, 10))

        for grid_x in range(self.grid_length_x):
            world.append([])
            for grid_y in range(self.grid_length_y):
                world_tile = self.grid_to_world(grid_x, grid_y, bitmap_codes[grid_x, grid_y])
                world[grid_x].append(world_tile)

        return world

    def grid_to_world(self, grid_x, grid_y, bitmap_code):

        rect = [
            (grid_x * TILE_SIZE, grid_y * TILE_SIZE),
            (grid_x * TILE_SIZE + TILE_SIZE, grid_y * TILE_SIZE),
            (grid_x * TILE_SIZE + TILE_SIZE, grid_y * TILE_SIZE + TILE_SIZE),
            (grid_x * TILE_SIZE, grid_y * TILE_SIZE + TILE_SIZE)
        ]

        out = {
            "grid": [grid_x, grid_y],
            "cart_rect": rect,
            "color": COLORS[bitmap_code]
        }

        return out
