import pygame as pg
import sys
from PIL import Image
from .world import World
from .agent import Agent
from .settings import TILE_SIZE
from .settings import MAP_EDGE
from .settings import AGENT_DIMS

class Game:

    def __init__(self, screen, surface, clock, n_envs_x=1, n_envs_y=1, display=True, progressive_gif=False):
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
        for _ in range(self.n_envs):
            self.worlds.append(World(MAP_EDGE, MAP_EDGE, self.width, self.height))
            self.agents.append(Agent(AGENT_DIMS[0], AGENT_DIMS[1]))

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
        )
        self.draw()
        if generate_gif:
            frame = pg.surfarray.array3d(pg.display.get_surface())
            self.frames.append(frame.swapaxes(0, 1))

    def create_gif(self, gif_path = '/home/antonio/Downloads/Terra.gif'):
        image_frames = [Image.fromarray(frame) for frame in self.frames]
        image_frames[0].save(gif_path, save_all=True, append_images=image_frames[1:], loop=0, duration=100)
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
    ):
        for i in range(self.n_envs):
            ag = active_grid[i]
            tg = target_grid[i]
            pm = padding_mask[i]
            dm = dumpability_mask[i]
            ap = agent_pos[i]
            bd = base_dir[i]
            cd = cabin_dir[i]
            self.worlds[i].update(ag, tg, pm, dm)
            self.agents[i].update(ap, bd, cd)

    def draw(self):
        self.surface.fill("#F0F0F0")
        agent_surfaces = []
        agent_positions = []
        cabin_positions = []

        for i, (world, agent) in enumerate(zip(self.worlds, self.agents)):
            ix = i % self.n_envs_y
            iy = i // self.n_envs_y

            print(f"{i=}, {ix=}, {iy=}")

            total_offset_x = ix * (MAP_EDGE + 4) * TILE_SIZE + 4*TILE_SIZE
            total_offset_y = iy * (MAP_EDGE + 4) * TILE_SIZE + 4*TILE_SIZE

            # Target map
            # for x in range(world.grid_length_x):
            #     for y in range(world.grid_length_y):

            #         sq = world.target_map[x][y]["cart_rect"]
            #         c = world.target_map[x][y]["color"]
            #         rect = pg.Rect(sq[0][0]+ total_offset_x, sq[0][1]+ total_offset_y, TILE_SIZE, TILE_SIZE)
            #         pg.draw.rect(self.surface, c, rect, 0)
            #         # pg.draw.rect(self.screen, (255, 255, 255), rect, 1)


            # Action map
            # offset = 61 * TILE_SIZE
            for x in range(world.grid_length_x):
                for y in range(world.grid_length_y):

                    sq = world.action_map[x][y]["cart_rect"]
                    c = world.action_map[x][y]["color"]
                    rect = pg.Rect(sq[0][0] + total_offset_x, sq[0][1] + total_offset_y, TILE_SIZE, TILE_SIZE)
                    pg.draw.rect(self.surface, c, rect, 0)
                    # pg.draw.rect(self.screen, (255, 255, 255), rect, 1)

            a = agent.agent["body"]["vertices"]
            w = agent.agent["body"]["width"]
            h = agent.agent["body"]["height"]

            ca = agent.agent["body"]["color"]
            agent_x = a[0][0] + total_offset_x
            agent_y = a[0][1] + total_offset_y
            a_rect = pg.Rect(0, 0, w * TILE_SIZE, h * TILE_SIZE)
            agent_surfaces.append(pg.Surface((w*TILE_SIZE, h*TILE_SIZE), pg.SRCALPHA))
            if self.progressive_gif:
                agent_surfaces[-1].set_alpha(50)
            
            agent_positions.append((agent_x, agent_y))
            pg.draw.rect(agent_surfaces[-1], ca, a_rect, 0, 3)

            cabin = agent.agent["cabin"]["vertices"]
            cabin = [(el[0] - a[0][0], el[1] - a[0][1]) for el in cabin]
            cabin_color = agent.agent["cabin"]["color"]
            pg.draw.polygon(agent_surfaces[-1], cabin_color, cabin)

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
