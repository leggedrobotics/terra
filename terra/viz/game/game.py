import pygame as pg
import sys
from PIL import Image
from .world import World
from .agent import Agent
from .settings import MAP_TILES
from terra.config import ExcavatorDims, ImmutableMapsConfig, ImmutableAgentConfig
import threading
import math

import pygame as pg
import sys
from PIL import Image
import threading
import math
import pygame as pg
import sys
from PIL import Image
import threading
import math

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
        agent_h = 9
        agent_w = 5
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
        
        # Pre-allocate agent surface to avoid repeated allocation
        self.agent_surface_cache = {}
        self.max_agent_surface_size = (50, 50)  # Reasonable max size

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
        additional_agents=None,
    ):
        if additional_agents and len(self.worlds) > 0:
            self.worlds[0].additional_agents = additional_agents
            print(f"Game.run: Set additional_agents with {len(additional_agents.get('positions', []))} agents")
    
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
        self.draw_optimized()  # Use optimized draw method
        if generate_gif:
            frame = pg.surfarray.array3d(pg.display.get_surface())
            self.frames.append(frame.swapaxes(0, 1))

    def create_gif(self, gif_path="/home/antonio/Downloads/Terra.gif"):
        if self.frames:
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
            #f target_tiles is not None:
            world.target_tiles = target_tiles

        # Avoid threading for small numbers of environments to reduce overhead
        if self.n_envs <= 2:
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
                update_world_agent(self.worlds[i], self.agents[i], ag, tg, pm, dm, ap, bd, cd, ld, tt)
        else:
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

    def get_agent_surface(self, agent, cache_key=None):
        """
        Get or create an agent surface, with caching to avoid repeated allocations.
        """
        if cache_key and cache_key in self.agent_surface_cache:
            cached_surface, cached_offset = self.agent_surface_cache[cache_key]
            return cached_surface, cached_offset
        
        body_vertices = agent.agent["body"]["vertices"]
        
        # Calculate the bounding box
        min_x = min(v[0] for v in body_vertices)
        min_y = min(v[1] for v in body_vertices)
        max_x = max(v[0] for v in body_vertices)
        max_y = max(v[1] for v in body_vertices)
        
        # Calculate surface size with a small padding
        surface_width = min(math.ceil(max_x - min_x) + 2, self.max_agent_surface_size[0])
        surface_height = min(math.ceil(max_y - min_y) + 2, self.max_agent_surface_size[1])
        
        # Create surface for the agent
        agent_surface = pg.Surface((surface_width, surface_height), pg.SRCALPHA)
        
        if self.progressive_gif:
            agent_surface.set_alpha(50)
        
        # Adjust vertices for the agent's surface
        offset_vertices = [(max(0, min(surface_width-1, v[0] - min_x)), 
                           max(0, min(surface_height-1, v[1] - min_y))) for v in body_vertices]
        
        # Draw agent body as polygon
        if len(offset_vertices) >= 3:  # Need at least 3 points for polygon
            ca = agent.agent["body"]["color"]
            try:
                pg.draw.polygon(agent_surface, ca, offset_vertices)
            except Exception as e:
                print(f"Warning: Could not draw agent body polygon: {e}")
        
        # Get cabin vertices and adjust for agent surface
        cabin = agent.agent["cabin"]["vertices"]
        cabin_offset = [(max(0, min(surface_width-1, v[0] - min_x)), 
                        max(0, min(surface_height-1, v[1] - min_y))) for v in cabin]
        
        if len(cabin_offset) >= 3:  # Need at least 3 points for polygon
            cabin_color = agent.agent["cabin"]["color"]
            try:
                pg.draw.polygon(agent_surface, cabin_color, cabin_offset)
            except Exception as e:
                print(f"Warning: Could not draw cabin polygon: {e}")
        
        # Cache the surface and offset if a key is provided
        if cache_key:
            # Limit cache size to prevent memory bloat
            if len(self.agent_surface_cache) > 20:
                # Remove oldest entry
                oldest_key = next(iter(self.agent_surface_cache))
                del self.agent_surface_cache[oldest_key]
            self.agent_surface_cache[cache_key] = (agent_surface, (min_x, min_y))
        
        return agent_surface, (min_x, min_y)

    def render_obs_pygame(self, obs, info):
        """
        Enhanced render method that handles additional agents from info.
        This should be called from your environment's render method.
        """
        # Extract additional agents from info if present
        additional_agents = info.get('additional_agents', None)
        
        # Set additional agents data on the world for the draw method to use
        if additional_agents and len(self.worlds) > 0:
            self.worlds[0].additional_agents = additional_agents
        
        # Call the standard rendering pipeline
        # This assumes you have the standard render data available
        # You may need to adapt this based on your actual rendering setup
        pass  # The actual rendering will be handled by draw_optimized()

    def draw_optimized(self):
        """
        Memory-optimized draw method that handles multiple agents efficiently.
        """
        self.surface.fill("#F0F0F0")
        
        # Collect agent data first to minimize surface creation
        agent_data = []

        for i, (world, agent) in enumerate(zip(self.worlds, self.agents)):
            ix = i % self.n_envs_y
            iy = i // self.n_envs_y

            total_offset_x = (
                ix * (self.maps_size_px + 4) * self.tile_size + 4 * self.tile_size
            )
            total_offset_y = (
                iy * (self.maps_size_px + 4) * self.tile_size + 4 * self.tile_size
            )

            # Draw terrain efficiently
            self.draw_terrain_optimized(world, total_offset_x, total_offset_y)
            
            # Handle multiple agents if present
            additional_agents = getattr(world, 'additional_agents', None)
            if additional_agents and 'positions' in additional_agents:
                positions = additional_agents['positions']
                angles = additional_agents.get('angles', [0] * len(positions))
                
                print(f"Drawing {len(positions)} additional agents")
                
                # Draw all agents with the same appearance
                for agent_idx, (pos, angle) in enumerate(zip(positions, angles)):
                    # Create a cache key based on angle to reuse surfaces
                    # Use the agent's actual dimensions from constructor
                    agent_w = getattr(agent, 'agent_width', 5)  # Default to 5 if not found
                    agent_h = getattr(agent, 'agent_height', 9)  # Default to 9 if not found
                    cache_key = f"agent_{angle}_{agent_w}_{agent_h}"
                    
                    try:
                        agent_surface, (min_x, min_y) = self.get_agent_surface(agent, cache_key)
                        
                        # Calculate position based on the provided position
                        # Note: you may need to adjust this based on your coordinate system
                        agent_x = pos[1] * self.tile_size + total_offset_x  # pos[1] is x coordinate
                        agent_y = pos[0] * self.tile_size + total_offset_y  # pos[0] is y coordinate
                        
                        agent_data.append((agent_surface, (agent_x, agent_y)))
                        print(f"Agent {agent_idx} surface created at ({agent_x}, {agent_y})")
                        
                    except Exception as e:
                        print(f"Warning: Could not create surface for agent {agent_idx}: {e}")
                        continue
            else:
                # Default: draw single agent
                try:
                    # Use the agent's actual dimensions or defaults
                    agent_w = getattr(agent, 'agent_width', 5)
                    agent_h = getattr(agent, 'agent_height', 9)
                    angle_base = agent.agent.get('angle_base', 0)
                    cache_key = f"single_agent_{angle_base}_{agent_w}_{agent_h}"
                    agent_surface, (min_x, min_y) = self.get_agent_surface(agent, cache_key)
                    
                    agent_x = min_x + total_offset_x
                    agent_y = min_y + total_offset_y
                    
                    agent_data.append((agent_surface, (agent_x, agent_y)))
                    
                except Exception as e:
                    print(f"Warning: Could not create surface for single agent: {e}")
                    continue

        # Blit the main surface
        self.screen.blit(self.surface, (0, 0))

        # Progressive gif handling (memory intensive - limit old agents)
        if self.progressive_gif:
            if self.count % 5 == 0:
                self.old_agents.append(agent_data[:])  # Copy current agent data
                # Limit old agents to prevent memory bloat
                if len(self.old_agents) > 10:
                    self.old_agents.pop(0)  # Remove oldest
            
            # Draw old agents with reduced opacity
            for old_agent_surfaces in self.old_agents:
                for agent_surface, agent_position in old_agent_surfaces:
                    try:
                        self.screen.blit(agent_surface, agent_position)
                    except Exception as e:
                        print(f"Warning: Could not blit old agent surface: {e}")
            self.count += 1

        # Draw current agents
        print(f"Drawing {len(agent_data)} agent surfaces")
        for agent_surface, agent_position in agent_data:
            try:
                self.screen.blit(agent_surface, agent_position)
            except Exception as e:
                print(f"Warning: Could not blit agent surface: {e}")

        if self.display:
            pg.display.flip()

    def draw_terrain_optimized(self, world, total_offset_x, total_offset_y):
        """
        Optimized terrain rendering to reduce memory allocations.
        """
        try:
            # Batch terrain rendering
            terrain_rects = []
            terrain_colors = []
            
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
                    terrain_rects.append(rect)
                    terrain_colors.append(c)

            # Draw all terrain rectangles
            for rect, color in zip(terrain_rects, terrain_colors):
                pg.draw.rect(self.surface, color, rect, 0)

            # FIXED: Handle target tiles highlighting with proper None check
            if hasattr(world, 'target_tiles') and world.target_tiles is not None:
                for x in range(world.grid_length_x):
                    for y in range(world.grid_length_y):
                        flat_idx = y * world.grid_length_x + x
                        # Check if we have target tiles and if this tile should be highlighted
                        if (flat_idx < len(world.target_tiles) and 
                            world.target_tiles[flat_idx] is not None and 
                            world.target_tiles[flat_idx]):
                            
                            sq = world.action_map[x][y]["cart_rect"]
                            rect = pg.Rect(
                                sq[0][0] + total_offset_x,
                                sq[0][1] + total_offset_y,
                                self.tile_size,
                                self.tile_size,
                            )
                            
                            # Create a small temporary surface for highlighting
                            temp_surface = pg.Surface((rect.width, rect.height), pg.SRCALPHA)
                            transparent_orange = (255, 51, 0, 100)
                            pg.draw.rect(temp_surface, transparent_orange, 
                                       pg.Rect(0, 0, rect.width, rect.height), 2)
                            
                            self.surface.blit(temp_surface, rect.topleft)
            else:
                # Debug: print when target_tiles is None or missing
                if not hasattr(world, 'target_tiles'):
                    print(f"World {id(world)} missing target_tiles attribute")
                elif world.target_tiles is None:
                    print(f"World {id(world)} has target_tiles = None")
                            
        except Exception as e:
            print(f"Warning: Error in terrain rendering: {e}")

    def cleanup_memory(self):
        """
        Clean up memory-intensive objects.
        """
        # Clear surface cache
        self.agent_surface_cache.clear()
        
        # Clear old agents if using progressive gif
        if self.progressive_gif:
            self.old_agents.clear()
        
        # Clear frames if they're taking too much memory
        if len(self.frames) > 100:
            self.frames = self.frames[-50:]  # Keep only last 50 frames
            
        print("Memory cleanup completed.")

    def __del__(self):
        """
        Cleanup when object is destroyed.
        """
        self.cleanup_memory()