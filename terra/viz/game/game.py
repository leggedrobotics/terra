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
        self.agents = []

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
            self.agents.append(Agent(agent_w, agent_h, tile_size, angles_base, angles_cabin))

        self.frames = []

        self.old_agents = []
        self.count = 0

    def run(
        self,
        active_grid,
        target_grid,
        padding_mask,
        dumpability_mask,
        agent_pos,
        base_dir,
        cabin_dir,
        generate_gif,
        target_tiles=None,
    ):
        # self.events()
        self.update(
            active_grid,
            target_grid,
            padding_mask,
            dumpability_mask,
            agent_pos,
            base_dir,
            cabin_dir,
            target_tiles,
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
        agent_pos,
        base_dir,
        cabin_dir,
        target_tiles=None,
    ):
        def update_world_agent(
            world,
            agent,
            active_grid,
            target_grid,
            padding_mask,
            dumpability_mask,
            agent_pos,
            base_dir,
            cabin_dir,
            target_tiles=None,
        ):
            world.update(active_grid, target_grid, padding_mask, dumpability_mask)
            agent.update(agent_pos, base_dir, cabin_dir)
            if target_tiles is not None:
                world.target_tiles = target_tiles

        threads = []
        for i in range(self.n_envs):
            ag = active_grid[i]
            tg = target_grid[i]
            pm = padding_mask[i]
            dm = dumpability_mask[i]
            ap = agent_pos[i]
            bd = base_dir[i]
            cd = cabin_dir[i]
            tt = None if target_tiles is None else target_tiles[i]
            thread = threading.Thread(
                target=update_world_agent,
                args=(self.worlds[i], self.agents[i], ag, tg, pm, dm, ap, bd, cd, tt),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def draw(self):
        self.surface.fill("#F0F0F0")
        agent_surfaces = []
        agent_positions = []

        for i, (world, agent) in enumerate(zip(self.worlds, self.agents)):
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

            body_vertices = agent.agent["body"]["vertices"]
            ca = agent.agent["body"]["color"]
            
            # Calculate the bounding box
            min_x = min(v[0] for v in body_vertices)
            min_y = min(v[1] for v in body_vertices)
            max_x = max(v[0] for v in body_vertices)
            max_y = max(v[1] for v in body_vertices)
            
            # Calculate surface size with a small padding
            surface_width = math.ceil(max_x - min_x) + 2
            surface_height = math.ceil(max_y - min_y) + 2
            
            # Create surface for the agent
            agent_surfaces.append(
                pg.Surface((surface_width, surface_height), pg.SRCALPHA)
            )
            if self.progressive_gif:
                agent_surfaces[-1].set_alpha(50)
            
            # Calculate surface position
            agent_x = min_x + total_offset_x
            agent_y = min_y + total_offset_y
            agent_positions.append((agent_x, agent_y))
            
            # Adjust vertices for the agent's surface
            offset_vertices = [(v[0] - min_x, v[1] - min_y) for v in body_vertices]
            
            # Draw agent body as polygon
            pg.draw.polygon(agent_surfaces[-1], ca, offset_vertices)
            
            # Get cabin vertices and adjust for agent surface
            cabin = agent.agent["cabin"]["vertices"]
            cabin_offset = [(v[0] - min_x, v[1] - min_y) for v in cabin]
            cabin_color = agent.agent["cabin"]["color"]
            pg.draw.polygon(agent_surfaces[-1], cabin_color, cabin_offset)

        self.screen.blit(self.surface, (0, 0))

        if self.progressive_gif:
            if self.count % 5 == 0:
                self.old_agents.append((agent_surfaces, agent_positions))
            for s in self.old_agents:
                for agent_surface, agent_position in zip(s[0], s[1]):
                    self.screen.blit(agent_surface, agent_position)
            self.count += 1
        else:
            for agent_surface, agent_position in zip(agent_surfaces, agent_positions):
                self.screen.blit(agent_surface, agent_position)

        if self.display:
            pg.display.flip()
