import pygame as pg
from .settings import COLORS
import numpy as np


class World:
    def __init__(self, grid_length_x, grid_length_y, width, height, tile_size):
        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y
        self.width = width
        self.height = height
        self.tile_size = tile_size

    def grid_to_world(self, grid_x, grid_y, bitmap_code):
        rect = [
            (grid_x * self.tile_size, grid_y * self.tile_size),
            (grid_x * self.tile_size + self.tile_size, grid_y * self.tile_size),
            (
                grid_x * self.tile_size + self.tile_size,
                grid_y * self.tile_size + self.tile_size,
            ),
            (grid_x * self.tile_size, grid_y * self.tile_size + self.tile_size),
        ]

        out = {
            "grid": [grid_x, grid_y],
            "cart_rect": rect,
            "color": COLORS[bitmap_code],
        }

        return out

    def update(self, action_map, target_map, obstacles_mask, dumpability_mask):
        # Store the maps as attributes of the World object
        self.target_map = np.asarray(target_map, dtype=np.int32).swapaxes(0, 1)
        self.action_map = np.asarray(action_map, dtype=np.int32).swapaxes(0, 1)
    
        if obstacles_mask is not None:
            self.obstacles_mask = np.asarray(obstacles_mask, dtype=np.bool_).swapaxes(0, 1)
        else:
            self.obstacles_mask = None

        if dumpability_mask is not None:
            self.dumpability_mask = np.asarray(dumpability_mask, dtype=np.bool_).swapaxes(0, 1)
        else:
            self.dumpability_mask = None

        world = []

        for grid_x in range(self.grid_length_x):
            world.append([])
            for grid_y in range(self.grid_length_y):
                if self.target_map[grid_x, grid_y] == -1:
                    # to dig
                    tile = 4
                elif self.target_map[grid_x, grid_y] == 1:
                    # final dumping area to terminate the episode
                    tile = 5
                else:
                    # neutral
                    tile = 0

                if self.obstacles_mask is not None and self.obstacles_mask[grid_x, grid_y] == 1:
                    # obstacle
                    tile = 2
                if self.dumpability_mask is not None and self.dumpability_mask[grid_x, grid_y] == 0:
                    # non-dumpable (e.g. road)
                    tile = 3
                if self.action_map[grid_x, grid_y] > 0:
                    # action map dump
                    tile = 1
                if self.action_map[grid_x, grid_y] < 0:
                    # action map dug
                    tile = -1

                world_tile = self.grid_to_world(grid_x, grid_y, tile)
                world[grid_x].append(world_tile)

        self.action_map = world
