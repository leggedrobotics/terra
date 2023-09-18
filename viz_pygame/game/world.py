
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

    def create_map(self, map):
        map = np.asarray(map, dtype=np.int32)
        map = map.swapaxes(0, 1)
        world = []

        for grid_x in range(self.grid_length_x):
            world.append([])
            for grid_y in range(self.grid_length_y):
                world_tile = self.grid_to_world(grid_x, grid_y, map[grid_x, grid_y])
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

    def update(self, action_map, target_map):
        self.action_map = self.create_map(action_map)
        self.target_map = self.create_map(target_map)
