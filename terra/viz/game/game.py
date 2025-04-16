import pygame as pg
import sys
from PIL import Image
from .world import World
from .agent import Agent
from .settings import MAP_TILES, COLORS
from terra.config import ExcavatorDims, ImmutableMapsConfig
import threading
import numpy as np


# Define colors for boolean masks (can be moved to settings.py)
BOOL_COLORS = {
    "padding_true": (255, 255, 255), # White for valid area
    "padding_false": (100, 100, 100), # Gray for padding/obstacle
    "dumpability_true": (173, 216, 230), # Light blue for dumpable
    "dumpability_false": (0, 0, 139),    # Dark blue for non-dumpable
    "traversability_true": (0, 255, 0),   # Green for traversable
    "traversability_false": (255, 0, 0),  # Red for non-traversable
}

def get_agent_dims(agent_w_m, agent_h_m, tile_size_m):
    """TODO repeated function, move to utils and share with env."""
    agent_height = (
        round(agent_w_m / tile_size_m)
        if (round(agent_w_m / tile_size_m)) % 2 != 0
        else round(agent_w_m / tile_size_m) + 1
    )
    agent_width = (
        round(agent_h_m / tile_size_m)
        if (round(agent_h_m / tile_size_m)) % 2 != 0
        else round(agent_h_m / tile_size_m) + 1
    )
    return agent_width, agent_height


class Game:
    def __init__(
        self,
        screen,
        surface,
        clock,
        maps_size_px,
        n_envs_x=1,
        n_envs_y=1,
        display=True,
        progressive_gif=False,
    ):
        self.screen = screen
        self.surface = surface
        self.clock = clock
        self.display = display
        self.progressive_gif = progressive_gif
        self.width, self.height = self.screen.get_size()

        # Initialize font for labels
        pg.font.init()
        self.font = pg.font.SysFont(None, 24) # Use default system font, size 24

        self.n_envs_x = n_envs_x
        self.n_envs_y = n_envs_y
        self.n_envs = n_envs_x * n_envs_y
        self.worlds = []
        self.agents_1 = []
        self.agents_2 = []

        tile_size_m = ImmutableMapsConfig().edge_length_m / maps_size_px
        self.maps_size_px = maps_size_px
        tile_size = MAP_TILES // maps_size_px
        self.tile_size = tile_size
        excavator_dims = ExcavatorDims()
        agent_h, agent_w = get_agent_dims(
            excavator_dims.WIDTH, excavator_dims.HEIGHT, tile_size_m
        )
        print(f"Agent size (in rendering): {agent_w}x{agent_h}")
        print(f"Tile size (in rendering): {tile_size_m}")
        print(f"Rendering tile size: {tile_size}")
        for _ in range(self.n_envs):
            self.worlds.append(
                World(maps_size_px, maps_size_px, self.width, self.height, tile_size)
            )
            self.agents_1.append(Agent(agent_w, agent_h, tile_size))
            self.agents_2.append(Agent(agent_w, agent_h, tile_size))

        self.frames = []

        self.old_agents_1 = []
        self.old_agents_2 = []
        self.count = 0

    def run(
        self,
        active_grid,
        target_grid,
        padding_mask,
        dumpability_mask,
        traversability_mask,
        agent_pos_1,
        base_dir_1,
        cabin_dir_1,
        loaded_1,
        agent_pos_2,
        base_dir_2,
        cabin_dir_2,
        loaded_2,
        generate_gif,
        target_tiles=None,
    ):
        # Process inputs for update (ensure correct types/orientation if needed by update logic)
        action_map_update = np.asarray(active_grid, dtype=np.int32)
        target_map_update = np.asarray(target_grid, dtype=np.int32)
        padding_mask_update = np.asarray(padding_mask, dtype=np.bool_)
        dumpability_mask_update = np.asarray(dumpability_mask, dtype=np.bool_)
        traversability_mask_update = np.asarray(traversability_mask, dtype=np.bool_)

        self.update(
            action_map_update,
            target_map_update,
            padding_mask_update,
            dumpability_mask_update,
            traversability_mask_update,
            agent_pos_1,
            base_dir_1,
            cabin_dir_1,
            loaded_1,
            agent_pos_2,
            base_dir_2,
            cabin_dir_2,
            loaded_2,
            target_tiles,
        )

        # Pass raw grids directly to draw
        self.draw(
            active_grid,
            target_grid,
            padding_mask,
            dumpability_mask,
            traversability_mask,
            agent_pos_1,
            base_dir_1,
            cabin_dir_1,
            loaded_1,
            agent_pos_2,
            base_dir_2,
            cabin_dir_2,
            loaded_2,
            target_tiles,
        )

        if generate_gif:
            frame = pg.surfarray.array3d(pg.display.get_surface())
            self.frames.append(frame.swapaxes(0, 1))

    def create_gif(self, gif_path="/Terra.gif"):
        image_frames = [Image.fromarray(frame) for frame in self.frames]
        image_frames[0].save(
            gif_path,
            save_all=True,
            append_images=image_frames[1:],
            loop=0,
            duration=100,
        )
        print(f"GIF generated at {gif_path}")
        self.frames = []

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()

    def update(
        self,
        active_grid,
        target_grid,
        padding_mask,
        dumpability_mask,
        traversability_mask,
        agent_pos_1,
        base_dir_1,
        cabin_dir_1,
        loaded_1,
        agent_pos_2,
        base_dir_2,
        cabin_dir_2,
        loaded_2,
        target_tiles=None,
    ):
        def update_world_agent(
            world,
            agent_1,
            agent_2,
            active_grid,
            target_grid,
            padding_mask,
            dumpability_mask,
            traversability_mask,
            agent_pos_1,
            base_dir_1,
            cabin_dir_1,
            loaded_1,
            agent_pos_2,
            base_dir_2,
            cabin_dir_2,
            loaded_2,
            target_tiles=None,
        ):
            world.update(active_grid, target_grid, padding_mask, dumpability_mask)
            agent_1.update(agent_pos_1, base_dir_1, cabin_dir_1, loaded_1)
            agent_2.update(agent_pos_2, base_dir_2, cabin_dir_2, loaded_2)
            if target_tiles is not None:
                world.target_tiles = target_tiles

        threads = []
        for i in range(self.n_envs):
            ag = active_grid[i]
            tg = target_grid[i]
            pm = padding_mask[i]
            dm = dumpability_mask[i]
            tm = traversability_mask[i]
            ap1 = agent_pos_1[i]
            bd1 = base_dir_1[i]
            cd1 = cabin_dir_1[i]
            ld1 = loaded_1[i]
            ap2 = agent_pos_2[i]
            bd2 = base_dir_2[i]
            cd2 = cabin_dir_2[i]
            ld2 = loaded_2[i]
            tt = None if target_tiles is None else target_tiles[i]
            thread = threading.Thread(
                target=update_world_agent,
                args=(self.worlds[i], self.agents_1[i], self.agents_2[i], ag, tg, pm, dm, tm, ap1, bd1, cd1, ld1, ap2, bd2, cd2, ld2, tt),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def _draw_map_in_subwindow(self, map_data, map_type, subwindow_rect, label):
        map_grid_x = map_data.shape[0]
        map_grid_y = map_data.shape[1]
        sub_x, sub_y, sub_w, sub_h = subwindow_rect

        # Draw map background / border (optional)
        # pg.draw.rect(self.surface, (50, 50, 50), subwindow_rect, 1)

        for x in range(map_grid_x):
            for y in range(map_grid_y):
                val = map_data[x, y]
                color = COLORS[0] # Default: neutral

                if map_type == "active":
                    if val > 0: color = COLORS[1]  # Dumped
                    elif val < 0: color = COLORS[-1] # Dug
                elif map_type == "target":
                    if val == -1: color = COLORS[4] # To Dig
                    elif val == 1: color = COLORS[5] # Final Dump
                elif map_type == "padding":
                    color = BOOL_COLORS["padding_true"] if val else BOOL_COLORS["padding_false"]
                elif map_type == "dumpability":
                    color = BOOL_COLORS["dumpability_true"] if val else BOOL_COLORS["dumpability_false"]
                elif map_type == "traversability":
                    color = BOOL_COLORS["traversability_true"] if val else BOOL_COLORS["traversability_false"]

                # Calculate tile position relative to subwindow
                rect_x = sub_x + x * self.tile_size
                rect_y = sub_y + y * self.tile_size
                tile_rect = pg.Rect(rect_x, rect_y, self.tile_size, self.tile_size)

                # Draw the tile
                pg.draw.rect(self.surface, color, tile_rect, 0)

        # Draw label
        #label_surface = self.font.render(label, True, (0, 0, 0)) # Black text
        #label_rect = label_surface.get_rect(center=(sub_x + sub_w // 2, sub_y - 15)) # Position above subwindow
        #self.surface.blit(label_surface, label_rect)

    def draw(
        self,
        active_grid_batch,
        target_grid_batch,
        padding_mask_batch,
        dumpability_mask_batch,
        traversability_mask_batch,
        agent_pos_1_batch,
        base_dir_1_batch,
        cabin_dir_1_batch,
        loaded_1_batch,
        agent_pos_2_batch,
        base_dir_2_batch,
        cabin_dir_2_batch,
        loaded_2_batch,
        target_tiles_batch=None,
    ):
        self.surface.fill("#F0F0F0")

        # --- Get data for the first environment (i=0) ---
        # We'll visualize only this one in the subwindows
        env_idx = 0
        if active_grid_batch.shape[0] <= env_idx:
            print(f"Warning: Cannot display environment {env_idx}, batch size is {active_grid_batch.shape[0]}")
            return # Or display a message on screen

        active_grid = np.asarray(active_grid_batch[env_idx], dtype=np.int32).swapaxes(0, 1)
        target_grid = np.asarray(target_grid_batch[env_idx], dtype=np.int32).swapaxes(0, 1)
        padding_mask = np.asarray(padding_mask_batch[env_idx], dtype=np.bool_).swapaxes(0, 1)
        dumpability_mask = np.asarray(dumpability_mask_batch[env_idx], dtype=np.bool_).swapaxes(0, 1)
        traversability_mask = None
        if traversability_mask_batch is not None:
            traversability_mask = np.asarray(traversability_mask_batch[env_idx], dtype=np.bool_).swapaxes(0, 1)

        target_tiles = None
        if target_tiles_batch is not None and target_tiles_batch.ndim > 1 and target_tiles_batch.shape[0] > env_idx:
            tt_flat = target_tiles_batch[env_idx]
            grid_dim = self.maps_size_px
            if tt_flat.size == grid_dim * grid_dim:
                target_tiles = tt_flat.reshape(grid_dim, grid_dim).swapaxes(0, 1)
            # else: print warning already handled in previous edit

        agent1 = self.agents_1[env_idx]
        agent2 = self.agents_2[env_idx]

        # --- Define Subwindow Layout --- 
        map_render_width = self.maps_size_px * self.tile_size
        map_render_height = self.maps_size_px * self.tile_size
        padding_px = 4 * self.tile_size # Use the same unit as in env.py

        # Top-left: Active Grid
        rect_active = pg.Rect(padding_px, padding_px + 30, map_render_width, map_render_height) # Add space for label
        # Top-right: Target Grid
        rect_target = pg.Rect(padding_px * 2 + map_render_width, padding_px + 30, map_render_width, map_render_height)
        # Bottom-left: Padding Mask
        rect_padding = pg.Rect(padding_px, padding_px * 2 + map_render_height + 30, map_render_width, map_render_height)
        # Bottom-right: Dumpability Mask
        rect_dumpability = pg.Rect(padding_px * 2 + map_render_width, padding_px * 2 + map_render_height + 30, map_render_width, map_render_height)

        # --- Draw each map in its subwindow --- 
        self._draw_map_in_subwindow(active_grid, "active", rect_active, "Active Map")
        self._draw_map_in_subwindow(target_grid, "target", rect_target, "Target Map")
        if traversability_mask is not None:
            self._draw_map_in_subwindow(traversability_mask, "traversability", rect_padding, "Traversability (True=Pass)")
        else:
            self._draw_map_in_subwindow(padding_mask, "padding", rect_padding, "Padding Mask (True=Valid)")
        self._draw_map_in_subwindow(dumpability_mask, "dumpability", rect_dumpability, "Dumpability Mask (True=Dumpable)")

        # --- Highlight Target Tiles (on Active Map subwindow) --- 
        if target_tiles is not None:
            grid_length_x = active_grid.shape[0]
            grid_length_y = active_grid.shape[1]
            for x in range(grid_length_x):
                for y in range(grid_length_y):
                    if target_tiles[x, y]:
                         # Calculate position relative to the active map subwindow
                         rect_x = rect_active.left + x * self.tile_size
                         rect_y = rect_active.top + y * self.tile_size
                         highlight_rect = pg.Rect(rect_x, rect_y, self.tile_size, self.tile_size)
                         pg.draw.rect(self.surface, "#FF3300", highlight_rect, 2) # Red border

        # --- Agent Drawing (on Active Map subwindow) --- 
        agent_surfaces_1 = []
        agent_positions_1 = []
        agent_surfaces_2 = []
        agent_positions_2 = []

        # Agent 1
        a1 = agent1.agent["body"]["vertices"]
        w1 = agent1.agent["body"]["width"]
        h1 = agent1.agent["body"]["height"]
        ca1 = agent1.agent["body"]["color"]
        # Calculate agent position relative to the Active Map subwindow top-left
        agent_offset_x1 = rect_active.left 
        agent_offset_y1 = rect_active.top
        agent_x1 = a1[0][0] + agent_offset_x1
        agent_y1 = a1[0][1] + agent_offset_y1
        a_rect1 = pg.Rect(0, 0, w1 * self.tile_size, h1 * self.tile_size)
        agent_surf1 = pg.Surface((w1 * self.tile_size, h1 * self.tile_size), pg.SRCALPHA)
        pg.draw.rect(agent_surf1, ca1, a_rect1, 0, 3)
        cabin1 = agent1.agent["cabin"]["vertices"]
        # Adjust cabin vertices relative to agent surface top-left (0,0)
        cabin1 = [(el[0] - a1[0][0], el[1] - a1[0][1]) for el in cabin1]
        cabin_color1 = agent1.agent["cabin"]["color"]
        pg.draw.polygon(agent_surf1, cabin_color1, cabin1)
        agent_surfaces_1.append(agent_surf1)
        agent_positions_1.append((agent_x1, agent_y1))

        # Agent 2
        a2 = agent2.agent["body"]["vertices"]
        w2 = agent2.agent["body"]["width"]
        h2 = agent2.agent["body"]["height"]
        ca2 = agent2.agent["body"]["color"]
        # Calculate agent position relative to the Active Map subwindow top-left
        agent_offset_x2 = rect_active.left
        agent_offset_y2 = rect_active.top
        agent_x2 = a2[0][0] + agent_offset_x2
        agent_y2 = a2[0][1] + agent_offset_y2
        a_rect2 = pg.Rect(0, 0, w2 * self.tile_size, h2 * self.tile_size)
        agent_surf2 = pg.Surface((w2 * self.tile_size, h2 * self.tile_size), pg.SRCALPHA)
        pg.draw.rect(agent_surf2, ca2, a_rect2, 0, 3)
        cabin2 = agent2.agent["cabin"]["vertices"]
        cabin2 = [(el[0] - a2[0][0], el[1] - a2[0][1]) for el in cabin2]
        cabin_color2 = agent2.agent["cabin"]["color"]
        pg.draw.polygon(agent_surf2, cabin_color2, cabin2)
        agent_surfaces_2.append(agent_surf2)
        agent_positions_2.append((agent_x2, agent_y2))


        # --- Blitting Logic --- 
        # Blit the main surface (with maps drawn) to the screen first
        self.screen.blit(self.surface, (0, 0))

        # Then blit agents on top (no progressive GIF logic here for simplicity)
        # The progressive GIF logic would need rethinking for the subwindow layout
        for agent_surface1, agent_position1 in zip(agent_surfaces_1, agent_positions_1):
            self.screen.blit(agent_surface1, agent_position1)
        for agent_surface2, agent_position2 in zip(agent_surfaces_2, agent_positions_2):
            self.screen.blit(agent_surface2, agent_position2)

        # Old blitting logic (commented out/removed)
        # if self.progressive_gif:
        #     if self.count % 5 == 0:
        #         self.old_agents_1.append((agent_surfaces_1, agent_positions_1))
        #     for s in self.old_agents_1:
        #         for agent_surface, agent_position in zip(s[0], s[1]):
        #             self.screen.blit(agent_surface, agent_position)
        #     self.count += 1
        # else:
        #     for agent_surface1, agent_position1 in zip(agent_surfaces_1, agent_positions_1):
        #         self.screen.blit(agent_surface1, agent_position1)
        #     for agent_surface2, agent_position2 in zip(agent_surfaces_2, agent_positions_2):
        #         self.screen.blit(agent_surface2, agent_position2)

        if self.display:
            pg.display.flip() 