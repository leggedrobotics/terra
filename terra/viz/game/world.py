
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

    def create_map(self, map_type, map, obstacles_mask, dumpability_mask, secondary_map):
        map = np.asarray(map, dtype=np.int32)
        map = map.swapaxes(0, 1)
        if obstacles_mask is not None:
            obstacles_mask = np.asarray(obstacles_mask, dtype=np.bool_)
            obstacles_mask = obstacles_mask.swapaxes(0, 1)
        if dumpability_mask is not None:
            dumpability_mask = np.asarray(dumpability_mask, dtype=np.bool_)
            dumpability_mask = dumpability_mask.swapaxes(0, 1)
        if secondary_map is not None:
            secondary_map = np.asarray(secondary_map, dtype=np.int32)
            secondary_map = secondary_map.swapaxes(0, 1)

        world = []

        for grid_x in range(self.grid_length_x):
            world.append([])
            for grid_y in range(self.grid_length_y):
                if obstacles_mask is not None and obstacles_mask[grid_x, grid_y] == 1:
                    tile = 2
                elif dumpability_mask is not None and dumpability_mask[grid_x, grid_y] == 0:
                    tile = 3
                elif map[grid_x, grid_y] > 0:
                    tile = 1
                elif map[grid_x, grid_y] < 0:
                    tile = -1
                else:
                    if map_type == "action" and secondary_map[grid_x, grid_y] == -1:
                        tile = 4
                    else:
                        tile = 0

                
                world_tile = self.grid_to_world(grid_x, grid_y, tile)
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
    
    def update(self, action_map, target_map, obstacles_mask, dumpability_mask):
        # self.target_map = self.create_map("target", target_map, obstacles_mask, dumpability_mask, None)
        self.action_map = self.create_map("action", action_map, obstacles_mask, None, target_map)
