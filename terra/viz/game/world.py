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

        # Handle custom colors (hex strings) or standard COLORS dictionary lookup
        if isinstance(bitmap_code, str) and bitmap_code.startswith('#'):
            # Custom color (hex string)
            color = bitmap_code
        elif isinstance(bitmap_code, (int, str)) and bitmap_code in COLORS:
            # Standard color from dictionary
            color = COLORS[bitmap_code]
        else:
            # Fallback to neutral color
            color = COLORS[0]

        out = {
            "grid": [grid_x, grid_y],
            "cart_rect": rect,
            "color": color,
        }

        return out

    def _get_dirt_gradient_color(self, dirt_amount, max_dirt_amount):
        """
        Generate a gradient color for dirt amount.
        Light blue for small amounts, dark blue for large amounts.
        
        Args:
            dirt_amount: Current dirt amount on this tile
            max_dirt_amount: Maximum dirt amount across all tiles for normalization
        Returns:
            Hex color string
        """
        if dirt_amount <= 0 or max_dirt_amount <= 0:
            return COLORS[0]  # neutral color
            
        # Normalize dirt amount to 0-1 range
        intensity = min(dirt_amount / max_dirt_amount, 1.0)
        
        # Define gradient from light blue to dark blue
        # Light blue: RGB(173, 216, 230) -> #ADD8E6
        # Dark blue: RGB(0, 43, 91) -> #002B5B (original dumped dirt color)
        light_blue = (173, 216, 230)  # Light blue
        dark_blue = (0, 43, 91)       # Dark blue (original)
        
        # Interpolate between light and dark blue
        r = int(light_blue[0] + (dark_blue[0] - light_blue[0]) * intensity)
        g = int(light_blue[1] + (dark_blue[1] - light_blue[1]) * intensity)
        b = int(light_blue[2] + (dark_blue[2] - light_blue[2]) * intensity)
        
        # Convert to hex
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        
        return hex_color

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

        # Find max dirt amount for gradient normalization (per-frame)
        max_dirt_amount = np.max(action_map[action_map > 0]) if np.any(action_map > 0) else 1

        world = []

        for grid_x in range(self.grid_length_x):
            world.append([])
            for grid_y in range(self.grid_length_y):
                dirt_amount = action_map[grid_x, grid_y]
                
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
                if (
                    dumpability_mask is not None
                    and dumpability_mask[grid_x, grid_y] == 0
                ):
                    # non-dumpable (e.g. road)
                    tile = 3
                if dirt_amount > 0:
                    # action map dump - use gradient based on dirt amount
                    tile = self._get_dirt_gradient_color(dirt_amount, max_dirt_amount)
                if dirt_amount < 0:
                    # action map dug
                    tile = -1

                world_tile = self.grid_to_world(grid_x, grid_y, tile)
                world[grid_x].append(world_tile)

        self.action_map = world
