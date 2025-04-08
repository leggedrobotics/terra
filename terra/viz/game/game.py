import pygame as pg
import sys
from PIL import Image
from .world import World
from .agent import Agent
from .settings import MAP_TILES
from terra.config import ExcavatorDims, ImmutableMapsConfig
import threading


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
        self.path = None
        self.path2 = None
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
        print(f"Agent size (in rendering): {agent_w}x{agent_h}")
        print(f"Tile size (in rendering): {tile_size_m}")
        print(f"Rendering tile size: {tile_size}")
        for _ in range(self.n_envs):
            self.worlds.append(
                World(maps_size_px, maps_size_px, self.width, self.height, tile_size)
            )
            self.agents.append(Agent(agent_w, agent_h, tile_size))

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
        loaded,
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
            loaded,
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
        loaded,
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
            loaded,
            target_tiles=None,
        ):
            world.update(active_grid, target_grid, padding_mask, dumpability_mask)
            agent.update(agent_pos, base_dir, cabin_dir, loaded)
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
            ld = loaded[i]
            tt = None if target_tiles is None else target_tiles[i]
            thread = threading.Thread(
                target=update_world_agent,
                args=(self.worlds[i], self.agents[i], ag, tg, pm, dm, ap, bd, cd, ld, tt),
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

            # Action map
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

            a = agent.agent["body"]["vertices"]
            w = agent.agent["body"]["width"]
            h = agent.agent["body"]["height"]

            # Get vertices for the agent body
            ca = agent.agent["body"]["color"]
            agent_x = a[0][0] + total_offset_x
            agent_y = a[0][1] + total_offset_y
            a_rect = pg.Rect(0, 0, w * self.tile_size, h * self.tile_size)
            agent_surfaces.append(
                pg.Surface((w * self.tile_size, h * self.tile_size), pg.SRCALPHA)
            )
            if self.progressive_gif:
                agent_surfaces[-1].set_alpha(50)

            agent_positions.append((agent_x, agent_y))
            pg.draw.rect(agent_surfaces[-1], ca, a_rect, 0, 3)

            cabin = agent.agent["cabin"]["vertices"]
            cabin = [(el[0] - a[0][0], el[1] - a[0][1]) for el in cabin]
            cabin_color = agent.agent["cabin"]["color"]
            pg.draw.polygon(agent_surfaces[-1], cabin_color, cabin)

            DRAW_PATH = False

            if DRAW_PATH and self.path is not None:
                line_points = []
                for x, y in self.path:
                    world_coords = world.grid_to_world(y , x , 0)  # Convert grid to world coordinates
                    line_points.append((world_coords["cart_rect"][0][0]+ total_offset_x, world_coords["cart_rect"][0][1] + total_offset_y))  # Add the top-left corner of the tile

                if len(line_points) > 1:
                    pg.draw.lines(self.surface, (255, 0, 0), False, line_points, 2)  # Draw the path as a red line

                # line_points2 = []
                # for x, y in self.path2:
                #     world_coords = world.grid_to_world(y , x , 0)  # Convert grid to world coordinates
                #     line_points2.append((world_coords["cart_rect"][0][0]+ total_offset_x, world_coords["cart_rect"][0][1] + total_offset_y))  # Add the top-left corner of the tile

                # if len(line_points) > 1:
                #     pg.draw.lines(self.surface, (255, 255, 0), False, line_points2, 2)  # Draw the path as a  line

            DRAW_FRONT_MARKER = False
            
            if DRAW_FRONT_MARKER:
                front_pos = agent.agent["front_marker"]["vertices"][0]
                front_x = front_pos[0] + total_offset_x
                front_y = front_pos[1] + total_offset_y

                # Determine the offset based on base orientation
                offset = 20  # Adjust this value for better visibility
                if agent.agent["angle_base"] == 0:  # Right
                    front_x += offset
                elif agent.agent["angle_base"] == 1:  # Up
                    front_y -= offset
                elif agent.agent["angle_base"] == 2:  # Left
                    front_x -= offset
                elif agent.agent["angle_base"] == 3: # Down
                    front_y += offset

                # Draw a small yellow dot at the front of the agent
                pg.draw.circle(self.surface, (255, 255, 0), (front_x, front_y), 4)  # Radius = 8 pixels

            OVERLAY_PIXEL_COORDS = False

            if OVERLAY_PIXEL_COORDS:
                skip_step = 20  # Process every 5th pixel (adjust as needed)
                font = pg.font.Font(None, 18)  # Default font, size 15

                for grid_x in range(0, world.grid_length_x, skip_step):  # Step through grid_x with skip_step
                    for grid_y in range(0, world.grid_length_y, skip_step):  # Step through grid_y with skip_step
                         # Determine the terrain type
                        if world.target_map[grid_x, grid_y] == -1:  # Area to dig
                            color = (200, 0, 200)  # bright Purple
                            terrain_value = "D"  # Label for "dig"
                        elif world.obstacles_mask is not None and world.obstacles_mask[grid_x, grid_y]:  # Obstacle
                            color = (255, 0, 0)  # bright Red
                            terrain_value = "O"  # Label for "obstacle"
                        elif world.dumpability_mask is not None and not world.dumpability_mask[grid_x, grid_y]:  # Street (non-dumpable)
                            color = (255, 255, 0)  # Yellow
                            terrain_value = "S"  # Label for "street"
                        else:  # Free area
                            color = (0, 0, 255)  # bright Blue
                            terrain_value = ""  # Label for "free"

                        # Convert grid coordinates to world coordinates
                        world_tile = world.grid_to_world(grid_x, grid_y, 0)  # 0 is a placeholder for neutral terrain
                        cart_rect = world_tile["cart_rect"]

                        coord_x, coord_y = cart_rect[0]  # Top-left corner of the tile

                        # Calculate the center of the pixel (tile)
                        center_x = coord_x + world.tile_size / 2
                        center_y = coord_y + world.tile_size / 2

                        # Render the terrain value as text directly over the pixel
                        #text_surface = font.render(terrain_value, True, color)
                        text_surface = font.render(f"({center_x}, {center_y})", True, color)
                        text_rect = text_surface.get_rect()  # Get the rectangle of the text
                        text_rect.center = (center_x + total_offset_x, center_y + total_offset_y)  # Set the center of the text to the center of the tile
                        self.surface.blit(text_surface, text_rect)

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
