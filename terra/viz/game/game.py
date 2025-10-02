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
        self.agents = []  # Support up to MAX_AGENTS agents per environment

        tile_size_m = ImmutableMapsConfig().edge_length_m / maps_size_px
        self.maps_size_px = maps_size_px
        
        # Dynamic display scaling to maintain consistent visual scale
        # 64x64 = 44m, so each tile = 44m/64 = 0.6875m
        # We want to maintain this scale across all map sizes
        baseline_map_size = 64   # Reference map size (44m)
        baseline_tile_size = MAP_TILES // baseline_map_size  # 192 // 64 = 3 pixels
        
        # Keep tile size constant so larger maps occupy proportionally more pixels
        tile_size = baseline_tile_size
        
        # Calculate total display size per map (maps_size_px * tile_size)
        total_display_size = maps_size_px * tile_size
        self.total_display_size = total_display_size
        
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
        print(f"Total display size: {self.total_display_size}x{self.total_display_size}")
        print(f"Map size: {maps_size_px}x{maps_size_px}")
        print(f"Number of possible base rotations: {angles_base}")
        print(f"Number of possible cabin rotations: {angles_cabin}")
        MAX_AGENTS = 4
        for _ in range(self.n_envs):
            self.worlds.append(
                World(maps_size_px, maps_size_px, self.total_display_size, self.total_display_size, tile_size)
            )
            # Create up to MAX_AGENTS agents per environment
            env_agents = []
            for _ in range(MAX_AGENTS):
                env_agents.append(Agent(agent_w, agent_h, tile_size, angles_base, angles_cabin))
            self.agents.append(env_agents)

        self.frames = []

        self.old_agents = []  # Support multi-agent history
        self.count = 0

    def run(
        self,
        active_grid,
        target_grid,
        padding_mask,
        dumpability_mask,
        interaction_mask,  # [H, W] - dig/dump cones for all active agents
        agent_states,  # [MAX_AGENTS, 8] - all agent states with active agent at index 0
        agent_active,  # [MAX_AGENTS] - which agents are active
        num_agents,    # scalar - number of active agents
        generate_gif,
        target_tiles=None,
    ):
        # self.events()
        self.update(
            active_grid,
            target_grid,
            padding_mask,
            dumpability_mask,
            interaction_mask,
            agent_states,
            agent_active,
            num_agents,
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
        interaction_mask,  # [H, W] - dig/dump cones for all active agents
        agent_states,  # [MAX_AGENTS, 8] - all agent states with active agent at index 0
        agent_active,  # [MAX_AGENTS] - which agents are active
        num_agents,    # scalar - number of active agents
        target_tiles=None,
    ):
        def update_world_agent(
            world,
            agents,  # List of up to MAX_AGENTS agents
            active_grid,
            target_grid,
            padding_mask,
            dumpability_mask,
            interaction_mask,  # [H, W] - dig/dump cones for all active agents
            agent_states,  # [MAX_AGENTS, 8] for this environment
            agent_active,  # [MAX_AGENTS] for this environment
            num_agents,    # scalar for this environment
            target_tiles=None,
        ):
            world.update(active_grid, target_grid, padding_mask, dumpability_mask, interaction_mask)
            
            # Update all agents (up to MAX_AGENTS)
            MAX_AGENTS = 4
            for i in range(MAX_AGENTS):
                if i < num_agents and agent_active[i]:  # Only update active agents
                    agent_state = agent_states[i]  # [8] - pos_x, pos_y, angle_base, angle_cabin, wheel_angle, loaded, agent_type, shovel_lifted
                    agent_pos = agent_state[:2]  # pos_x, pos_y
                    base_dir = agent_state[2:3]  # angle_base
                    cabin_dir = agent_state[3:4]  # angle_cabin
                    loaded = agent_state[5:6]  # loaded
                    agent_type = agent_state[6:7]  # agent_type
                    shovel_lifted = agent_state[7:8]  # shovel_lifted
                    agents[i].update(agent_pos, base_dir, cabin_dir, loaded, agent_type, shovel_lifted)
            
            if target_tiles is not None:
                world.target_tiles = target_tiles

        threads = []
        for i in range(self.n_envs):
            ag = active_grid[i]
            tg = target_grid[i]
            pm = padding_mask[i]
            dm = dumpability_mask[i]
            im = interaction_mask[i]  # [H, W] - dig/dump cones for all active agents
            ast = agent_states[i]  # [MAX_AGENTS, 8] for this environment
            aac = agent_active[i]  # [MAX_AGENTS] for this environment
            nag = num_agents[i] if hasattr(num_agents, '__len__') else num_agents  # scalar for this environment
            tt = None if target_tiles is None else target_tiles[i]
            thread = threading.Thread(
                target=update_world_agent,
                args=(self.worlds[i], self.agents[i], ag, tg, pm, dm, im, ast, aac, nag, tt),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def draw(self):
        self.surface.fill("#F0F0F0")
        agent_surfaces = []  # List of lists: [env][agent] -> surface
        agent_positions = []  # List of lists: [env][agent] -> position

        for i, (world, agents) in enumerate(zip(self.worlds, self.agents)):
            ix = i % self.n_envs_y
            iy = i // self.n_envs_y

            # Offsets based on pixel size per map (robust for dynamic scaling)
            map_px = self.total_display_size  # = self.maps_size_px * self.tile_size
            border_px = 4 * self.tile_size
            total_offset_x = ix * (map_px + border_px) + border_px
            total_offset_y = iy * (map_px + border_px) + border_px

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

            # Render all agents for this environment
            env_surfaces = []
            env_positions = []
            
            for agent_idx, agent in enumerate(agents):
                # Only render agents that have been updated (have valid state)
                if hasattr(agent, 'agent') and agent.agent is not None:
                    body_vertices = agent.agent["body"]["vertices"]
                    body_color = agent.agent["body"]["color"]
                    
                    # Calculate bounding box for all agent components (body, cabin, and optionally shovel)
                    all_vertices = body_vertices + agent.agent["cabin"]["vertices"]
                    if "shovel" in agent.agent:
                        all_vertices += agent.agent["shovel"]["vertices"]
                    
                    min_x = min(v[0] for v in all_vertices)
                    min_y = min(v[1] for v in all_vertices)
                    max_x = max(v[0] for v in all_vertices)
                    max_y = max(v[1] for v in all_vertices)
                    
                    surface_width = math.ceil(max_x - min_x) + 2
                    surface_height = math.ceil(max_y - min_y) + 2
                    
                    agent_surface = pg.Surface((surface_width, surface_height), pg.SRCALPHA)
                    if self.progressive_gif:
                        agent_surface.set_alpha(50)
                    
                    agent_x_pos = min_x + total_offset_x
                    agent_y_pos = min_y + total_offset_y
                    env_positions.append((agent_x_pos, agent_y_pos))
                    
                    # Draw agent body
                    offset_vertices = [(v[0] - min_x, v[1] - min_y) for v in body_vertices]
                    pg.draw.polygon(agent_surface, body_color, offset_vertices)
                    
                    # Draw shovel first (behind the body) if it exists
                    if "shovel" in agent.agent:
                        shovel_vertices = agent.agent["shovel"]["vertices"]
                        shovel_offset = [(v[0] - min_x, v[1] - min_y) for v in shovel_vertices]
                        shovel_color = agent.agent["shovel"]["color"]
                        pg.draw.polygon(agent_surface, shovel_color, shovel_offset)
                    
                    # Draw cabin on top
                    cabin_vertices = agent.agent["cabin"]["vertices"]
                    cabin_offset = [(v[0] - min_x, v[1] - min_y) for v in cabin_vertices]
                    cabin_color = agent.agent["cabin"]["color"]
                    pg.draw.polygon(agent_surface, cabin_color, cabin_offset)
                    env_surfaces.append(agent_surface)
            
            agent_surfaces.append(env_surfaces)
            agent_positions.append(env_positions)

        self.screen.blit(self.surface, (0, 0))

        if self.progressive_gif:
            if self.count % 5 == 0:
                self.old_agents.append((agent_surfaces, agent_positions))
            
            for history_surfaces, history_positions in self.old_agents:
                for env_surfaces, env_positions in zip(history_surfaces, history_positions):
                    for agent_surface, agent_position in zip(env_surfaces, env_positions):
                        self.screen.blit(agent_surface, agent_position)
            self.count += 1
        else:
            for env_surfaces, env_positions in zip(agent_surfaces, agent_positions):
                for agent_surface, agent_position in zip(env_surfaces, env_positions):
                    self.screen.blit(agent_surface, agent_position)

        if self.display:
            pg.display.flip()
