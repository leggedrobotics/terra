import pygame as pg
import sys
from PIL import Image
from .world import World
from .agent import Agent
from .settings import MAP_TILES
from terra.config import ExcavatorDims, ImmutableMapsConfig, ImmutableAgentConfig
import threading
import math


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
        angles_base = ImmutableAgentConfig().angles_base
        angles_cabin = ImmutableAgentConfig().angles_cabin
        print(f"Agent size (in rendering): {agent_w}x{agent_h}")
        print(f"Tile size (in rendering): {tile_size_m}")
        print(f"Rendering tile size: {tile_size}")
        print(f"Number of possible base rotations: {angles_base}")
        print(f"Number of possible cabin rotations: {angles_cabin}")
        for _ in range(self.n_envs):
            self.worlds.append(
                World(maps_size_px, maps_size_px, self.width, self.height, tile_size)
            )
            self.agents_1.append(Agent(agent_w, agent_h, tile_size, angles_base, angles_cabin))
            self.agents_2.append(Agent(agent_w, agent_h, tile_size, angles_base, angles_cabin))

        self.frames = []

        self.old_agents_1 = []
        self.old_agents_2 = []
        self.count = 0
        self.num_agents = 1

    def run(
        self,
        active_grid,
        target_grid,
        padding_mask,
        dumpability_mask,
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
        num_agents=1,
    ):
        # self.events()
        self.update(
            active_grid,
            target_grid,
            padding_mask,
            dumpability_mask,
            agent_pos_1,
            base_dir_1,
            cabin_dir_1,
            loaded_1,
            agent_pos_2,
            base_dir_2,
            cabin_dir_2,
            loaded_2,
            target_tiles,
            num_agents,
        )
        self.draw()
        if generate_gif:
            frame = pg.surfarray.array3d(pg.display.get_surface())
            self.frames.append(frame.swapaxes(0, 1))

    def create_gif(self, gif_path="/home/antonio/Downloads/Terra.gif"):
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
        agent_pos_1,
        base_dir_1,
        cabin_dir_1,
        loaded_1,
        agent_pos_2,
        base_dir_2,
        cabin_dir_2,
        loaded_2,
        target_tiles=None,
        num_agents=1,
    ):
        self.num_agents = num_agents
        def update_world_agent(
            world,
            agent_1,
            agent_2,
            active_grid,
            target_grid,
            padding_mask,
            dumpability_mask,
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
            if self.num_agents == 2:
                agent_2.update(agent_pos_2, base_dir_2, cabin_dir_2, loaded_2)
            if target_tiles is not None:
                world.target_tiles = target_tiles

        threads = []
        for i in range(self.n_envs):
            ag = active_grid[i]
            tg = target_grid[i]
            pm = padding_mask[i]
            dm = dumpability_mask[i]
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
                args=(self.worlds[i], self.agents_1[i], self.agents_2[i], ag, tg, pm, dm, ap1, bd1, cd1, ld1, ap2, bd2, cd2, ld2, tt),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def draw(self):
        self.surface.fill("#F0F0F0")
        agent_surfaces_1 = []
        agent_positions_1 = []
        agent_surfaces_2 = []
        agent_positions_2 = []

        for i, (world, agent1, agent2) in enumerate(zip(self.worlds, self.agents_1, self.agents_2)):
            ix = i % self.n_envs_y
            iy = i // self.n_envs_y

            total_offset_x = (
                ix * (self.maps_size_px + 4) * self.tile_size + 4 * self.tile_size
            )
            total_offset_y = (
                iy * (self.maps_size_px + 4) * self.tile_size + 4 * self.tile_size
            )

            # Draw terrain
            for x in range(world.grid_length_x):
                for y in range(world.grid_length_y):
                    sq = world.action_map[x][y]["cart_rect"]
                    c = world.action_map[x][y]["color"]
                    rect = pg.Rect(
                        sq[0][0] + total_offset_x,
                        sq[0][1] + total_offset_y,
                        self.tile_size,
                        self.tile_size,
                    )
                    pg.draw.rect(self.surface, c, rect, 0)

                    # Highlight target tiles (where the digger will dig / dump)
                    if hasattr(world, 'target_tiles') and world.target_tiles is not None:
                        flat_idx = y * world.grid_length_x + x
                        if flat_idx < len(world.target_tiles) and world.target_tiles[flat_idx]:
                            pg.draw.rect(self.surface, "#FF3300", rect, 2)

            # Agent 1
            body_vertices_1 = agent1.agent["body"]["vertices"]
            body_color_1 = agent1.agent["body"]["color"]
            
            min_x1 = min(v[0] for v in body_vertices_1)
            min_y1 = min(v[1] for v in body_vertices_1)
            max_x1 = max(v[0] for v in body_vertices_1)
            max_y1 = max(v[1] for v in body_vertices_1)
            
            surface_width1 = math.ceil(max_x1 - min_x1) + 2
            surface_height1 = math.ceil(max_y1 - min_y1) + 2
            
            agent_surface1 = pg.Surface((surface_width1, surface_height1), pg.SRCALPHA)
            if self.progressive_gif:
                agent_surface1.set_alpha(50)
            
            agent_x1_pos = min_x1 + total_offset_x
            agent_y1_pos = min_y1 + total_offset_y
            agent_positions_1.append((agent_x1_pos, agent_y1_pos))
            
            offset_vertices_1 = [(v[0] - min_x1, v[1] - min_y1) for v in body_vertices_1]
            pg.draw.polygon(agent_surface1, body_color_1, offset_vertices_1)
            
            cabin_vertices_1 = agent1.agent["cabin"]["vertices"]
            cabin_offset_1 = [(v[0] - min_x1, v[1] - min_y1) for v in cabin_vertices_1]
            cabin_color_1 = agent1.agent["cabin"]["color"]
            pg.draw.polygon(agent_surface1, cabin_color_1, cabin_offset_1)
            agent_surfaces_1.append(agent_surface1)

            if self.num_agents == 2:
                # Agent 2
                body_vertices_2 = agent2.agent["body"]["vertices"]
                body_color_2 = agent2.agent["body"]["color"]

                min_x2 = min(v[0] for v in body_vertices_2)
                min_y2 = min(v[1] for v in body_vertices_2)
                max_x2 = max(v[0] for v in body_vertices_2)
                max_y2 = max(v[1] for v in body_vertices_2)

                surface_width2 = math.ceil(max_x2 - min_x2) + 2
                surface_height2 = math.ceil(max_y2 - min_y2) + 2

                agent_surface2 = pg.Surface((surface_width2, surface_height2), pg.SRCALPHA)
                if self.progressive_gif:
                    agent_surface2.set_alpha(50)

                agent_x2_pos = min_x2 + total_offset_x
                agent_y2_pos = min_y2 + total_offset_y
                agent_positions_2.append((agent_x2_pos, agent_y2_pos))

                offset_vertices_2 = [(v[0] - min_x2, v[1] - min_y2) for v in body_vertices_2]
                pg.draw.polygon(agent_surface2, body_color_2, offset_vertices_2)

                cabin_vertices_2 = agent2.agent["cabin"]["vertices"]
                cabin_offset_2 = [(v[0] - min_x2, v[1] - min_y2) for v in cabin_vertices_2]
                cabin_color_2 = agent2.agent["cabin"]["color"]
                pg.draw.polygon(agent_surface2, cabin_color_2, cabin_offset_2)
                agent_surfaces_2.append(agent_surface2)

        self.screen.blit(self.surface, (0, 0))

        if self.progressive_gif:
            if self.count % 5 == 0:
                self.old_agents_1.append((list(agent_surfaces_1), list(agent_positions_1)))
                self.old_agents_2.append((list(agent_surfaces_2), list(agent_positions_2)))
            
            for history_surfaces, history_positions in self.old_agents_1:
                for agent_surface, agent_position in zip(history_surfaces, history_positions):
                    self.screen.blit(agent_surface, agent_position)
            if self.num_agents == 2:
                for history_surfaces, history_positions in self.old_agents_2:
                    for agent_surface, agent_position in zip(history_surfaces, history_positions):
                        self.screen.blit(agent_surface, agent_position)
            self.count += 1
        else:
            for agent_surface, agent_position in zip(agent_surfaces_1, agent_positions_1):
                self.screen.blit(agent_surface, agent_position)
            if self.num_agents == 2:
                for agent_surface, agent_position in zip(agent_surfaces_2, agent_positions_2):
                    self.screen.blit(agent_surface, agent_position)

        if self.display:
            pg.display.flip()
