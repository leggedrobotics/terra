
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
        action_map = np.asarray(action_map, dtype=np.int32)
        action_map = action_map.swapaxes(0, 1)
        target_map = np.asarray(target_map, dtype=np.int32)
        target_map = target_map.swapaxes(0, 1)
        if obstacles_mask is not None:
            obstacles_mask = np.asarray(obstacles_mask, dtype=np.bool_)
            obstacles_mask = obstacles_mask.swapaxes(0, 1)
        if dumpability_mask is not None:
            dumpability_mask = np.asarray(dumpability_mask, dtype=np.bool_)
            dumpability_mask = dumpability_mask.swapaxes(0, 1)

        world = []

        for grid_x in range(self.grid_length_x):
            world.append([])
            for grid_y in range(self.grid_length_y):
                if target_map[grid_x, grid_y] == -1:
                    # to dig
                    tile = 4
                elif target_map[grid_x, grid_y] == 1:
                    # final dumping area to terminate the episode
                    tile = 5
                else:
                    # neutral
                    tile = 0
                
                if obstacles_mask is not None and obstacles_mask[grid_x, grid_y] == 1:
                    # obstacle
                    tile = 2
                if dumpability_mask is not None and dumpability_mask[grid_x, grid_y] == 0:
                    # non-dumpable (e.g. road)
                    tile = 3
                if action_map[grid_x, grid_y] > 0:
                    # action map dump
                    tile = 1
                if action_map[grid_x, grid_y] < 0:
                    # action map dug
                    tile = -1
                
                world_tile = self.grid_to_world(grid_x, grid_y, tile)
                world[grid_x].append(world_tile)

        self.action_map = world
