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
        agent_config=None,
    ):
        self.screen = screen
        self.surface = surface
        self.clock = clock
        self.display = display
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

        if maps_size_px == 128:
            # Extract dimensions from config
            # Use agent_config argument if provided, else default values
            if agent_config is not None:
                agent_h = int(agent_config.get('height', 5))
                agent_w = int(agent_config.get('width', 9))
            else:
                agent_h = 5
                agent_w = 9
            print(f"Using agent config: {agent_w}x{agent_h}")
        
        else:
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
        loaded,
        generate_gif,
        target_tiles=None,
        info=None,
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
        self.draw(info)
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


    def _create_agent_surface(self, agent, base_dir, cabin_dir, loaded):
            """Create a temporary agent with specific angles and return its surface."""

            temp_agent = Agent(
                agent.width if hasattr(agent, 'width') else agent.w, 
                agent.height if hasattr(agent, 'height') else agent.h, 
                agent.tile_size, 
                agent.angles_base, 
                agent.angles_cabin
            )
            
            # Update it with the specific position and angles
            temp_agent.update([0, 0], base_dir, cabin_dir, loaded)  # Position doesn't matter for surface creation

            # Get vertices for the body and cabin
            body_vertices = temp_agent.agent["body"]["vertices"]
            cabin_vertices = temp_agent.agent["cabin"]["vertices"]
            body_color = temp_agent.agent["body"]["color"]
            cabin_color = temp_agent.agent["cabin"]["color"]
        
            # Calculate bounding box
            all_vertices = body_vertices + cabin_vertices
            min_x = min(v[0] for v in all_vertices)
            min_y = min(v[1] for v in all_vertices)
            max_x = max(v[0] for v in all_vertices)
            max_y = max(v[1] for v in all_vertices)
        
            # Create surface
            surface_width = math.ceil(max_x - min_x) + 2
            surface_height = math.ceil(max_y - min_y) + 2
            agent_surface = pg.Surface((surface_width, surface_height), pg.SRCALPHA)
            
            # Adjust vertices for the surface coordinate system
            body_offset = [(v[0] - min_x, v[1] - min_y) for v in body_vertices]
            cabin_offset = [(v[0] - min_x, v[1] - min_y) for v in cabin_vertices]
            
            # Draw agent parts
            pg.draw.polygon(agent_surface, body_color, body_offset)
            pg.draw.polygon(agent_surface, cabin_color, cabin_offset)
            
            return agent_surface, (min_x, min_y)
    
    def _render_agents_for_env(self, env_idx, world, agent, total_offset_x, total_offset_y, info):
        """Render all agents for a specific environment and return surfaces and positions."""
        agent_surfaces = []
        agent_positions = []
                
        # Check if we have additional agents in info - if so, skip primary agent to avoid duplication
        has_additional_agents = (info and 'additional_agents' in info and 
                            'positions' in info['additional_agents'] and 
                            len(info['additional_agents']['positions']) > 0)
                
        # Only render the primary agent if we don't have additional agents
        if not has_additional_agents:
            # Render the primary agent (from the original arrays)
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
            
            # Create surface for the primary agent
            primary_surface = pg.Surface((surface_width, surface_height), pg.SRCALPHA)

            
            # Calculate surface position
            agent_x = min_x + total_offset_x
            agent_y = min_y + total_offset_y
            
            # Adjust vertices for the agent's surface
            offset_vertices = [(v[0] - min_x, v[1] - min_y) for v in body_vertices]
            
            # Draw primary agent body as polygon
            pg.draw.polygon(primary_surface, ca, offset_vertices)
            
            # Get cabin vertices and adjust for agent surface
            cabin = agent.agent["cabin"]["vertices"]
            cabin_offset = [(v[0] - min_x, v[1] - min_y) for v in cabin]
            cabin_color = agent.agent["cabin"]["color"]
            pg.draw.polygon(primary_surface, cabin_color, cabin_offset)
            
            agent_surfaces.append(primary_surface)
            agent_positions.append((agent_x, agent_y))
            #print(f"Primary agent rendered at position: ({agent_x}, {agent_y})")
        
        # Render additional agents if they exist in info
        if info and 'additional_agents' in info:
            additional_agents = info['additional_agents']
            
            # Check for the required keys in your specific format
            if ('positions' in additional_agents and 
                'angles base' in additional_agents and 
                'angles cabin' in additional_agents and
                'loaded' in additional_agents):
                
                positions = additional_agents['positions']
                angles_base = additional_agents['angles base']
                angles_cabin = additional_agents['angles cabin']
                loaded_states = additional_agents['loaded']
                                
                # Handle both single environment and multi-environment cases
                if env_idx == 0:  # For single environment or first environment
                    #print(f"Processing additional agents for env_idx 0")
                    
                    # Convert numpy arrays/lists to Python lists if needed
                    if hasattr(positions, 'tolist'):
                        positions = positions.tolist()
                    if hasattr(angles_base, 'tolist'):
                        angles_base = angles_base.tolist()
                    if hasattr(angles_cabin, 'tolist'):
                        angles_cabin = angles_cabin.tolist()
                    if hasattr(loaded_states, 'tolist'):
                        loaded_states = loaded_states.tolist()
                    
                    # Render each additional agent
                    for i, pos in enumerate(positions):
                        
                        if i < len(angles_base) and i < len(angles_cabin) and i < len(loaded_states):
                            base_dir = angles_base[i]
                            cabin_dir = angles_cabin[i]
                            loaded = bool(loaded_states[i])
                            
                            # Create agent surface with specific angles
                            additional_surface, offset = self._create_agent_surface(
                                agent, base_dir, cabin_dir, loaded
                            )
                                                        
                            # Calculate position on screen
                            # pos is already in pixel coordinates based on your example
                            screen_x = pos[0] * self.tile_size + total_offset_x + offset[0]
                            screen_y = pos[1] * self.tile_size + total_offset_y + offset[1]
                                                        
                            agent_surfaces.append(additional_surface)
                            agent_positions.append((screen_x, screen_y))
                else:
                    print(f"Skipping additional agents for env_idx {env_idx}")

        return agent_surfaces, agent_positions
        
    def draw(self, info=None):
        if info is None or "additional_agents" not in info or info["additional_agents"] is None:
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

            for agent_surface, agent_position in zip(agent_surfaces, agent_positions):
                self.screen.blit(agent_surface, agent_position)

            if self.display:
                pg.display.flip()
        else:
            self.surface.fill("#F0F0F0")
            all_agent_surfaces = []
            all_agent_positions = []
            print("Map size in px:", self.maps_size_px)
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

                        # # Highlight target tiles (where the digger will dig / dump)
                        # if hasattr(world, 'target_tiles') and world.target_tiles is not None:
                        #     flat_idx = y * world.grid_length_x + x
                        #     if flat_idx < len(world.target_tiles) and world.target_tiles[flat_idx]:
                         #         pg.draw.rect(self.surface, "#FF3300", rect, 2)
                # Draw partition rectangles (only for the first environment)
                if info and info.get('show_partitions', False) and i == 0:
                    partitions = info.get('partitions', [])
                    self._draw_partition_rectangles(partitions, total_offset_x, total_offset_y)

                # Render all agents for this environment
                env_agent_surfaces, env_agent_positions = self._render_agents_for_env(
                    i, world, agent, total_offset_x, total_offset_y, info
                )
                        
                all_agent_surfaces.extend(env_agent_surfaces)
                all_agent_positions.extend(env_agent_positions)

            self.screen.blit(self.surface, (0, 0))


            for agent_surface, agent_position in zip(all_agent_surfaces, all_agent_positions):
                self.screen.blit(agent_surface, agent_position)

            if self.display:
                pg.display.flip()

    def _draw_partition_rectangles(self, partitions, total_offset_x, total_offset_y):
        """Draw simple rectangles around each partition."""
        
        for i, partition in enumerate(partitions):
            y_start, x_start, y_end, x_end = partition['region_coords']
            status = partition['status']
            
            # Convert to screen coordinates
            rect_x = x_start * self.tile_size + total_offset_x
            rect_y = y_start * self.tile_size + total_offset_y
            rect_width = (x_end - x_start + 1) * self.tile_size
            rect_height = (y_end - y_start + 1) * self.tile_size
            
            # Choose color based on status
            if status == 'active':
                color = (0, 255, 0)  # Green
                width = 3
            elif status == 'completed':
                color = (0, 0, 255)  # Blue  
                width = 2
            elif status == 'failed':
                color = (255, 0, 0)  # Red
                width = 2
            else:  # pending
                color = (255, 255, 0)  # Yellow
                width = 1
                
            # Draw the rectangle
            pg.draw.rect(self.surface, color, 
                        (rect_x, rect_y, rect_width, rect_height), width)