import pygame as pg
import sys
from PIL import Image
from .world import World
from .agent import Agent
from .settings import TILE_SIZE

class Game:

    def __init__(self, screen, clock, n_envs_x=1, n_envs_y=1, display=True):
        self.screen = screen
        self.clock = clock
        self.display = display
        self.width, self.height = self.screen.get_size()

        self.n_envs_x = n_envs_x
        self.n_envs_y = n_envs_y
        self.n_envs = n_envs_x * n_envs_y
        self.worlds = []
        self.agents = []
        for _ in range(self.n_envs):
            self.worlds.append(World(60, 60, self.width, self.height))
            self.agents.append(Agent(9, 5))

        self.frames = []
        

    def run(
            self,
            active_grid,
            target_grid,
            agent_pos,
            base_dir,
            cabin_dir,
            generate_gif,
    ):
        # self.events()
        self.update(
            active_grid,
            target_grid,
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
            agent_pos,
            base_dir,
            cabin_dir,
    ):
        for i in range(self.n_envs):
            ag = active_grid[i]
            tg = target_grid[i]
            ap = agent_pos[i]
            bd = base_dir[i]
            cd = cabin_dir[i]
            self.worlds[i].update(ag, tg)
            self.agents[i].update(ap, bd, cd)

    def draw(self):
        self.screen.fill((255, 255, 255,))

        for i, (world, agent) in enumerate(zip(self.worlds, self.agents)):
            
            ix = i % self.n_envs_x
            iy = i // self.n_envs_y
            total_offset_x = 2 * ix * 61 * TILE_SIZE
            total_offset_y = iy * 61 * TILE_SIZE

            # Target map
            for x in range(world.grid_length_x):
                for y in range(world.grid_length_y):

                    sq = world.target_map[x][y]["cart_rect"]
                    c = world.target_map[x][y]["color"]
                    rect = pg.Rect(sq[0][0]+ total_offset_x, sq[0][1]+ total_offset_y, TILE_SIZE, TILE_SIZE)
                    pg.draw.rect(self.screen, c, rect)
                    # pg.draw.rect(self.screen, (255, 255, 255), rect, 1)


            # Action map
            offset = 61 * TILE_SIZE
            for x in range(world.grid_length_x):
                for y in range(world.grid_length_y):

                    sq = world.action_map[x][y]["cart_rect"]
                    c = world.action_map[x][y]["color"]
                    rect = pg.Rect(sq[0][0] + offset + total_offset_x, sq[0][1] + total_offset_y, TILE_SIZE, TILE_SIZE)
                    pg.draw.rect(self.screen, c, rect)
                    # pg.draw.rect(self.screen, (255, 255, 255), rect, 1)

            a = agent.agent["body"]["vertices"]
            w = agent.agent["body"]["width"]
            h = agent.agent["body"]["height"]

            ca = agent.agent["body"]["color"]
            a_rect = pg.Rect(a[0][0] + offset + total_offset_x, a[0][1] + total_offset_y, w * TILE_SIZE, h * TILE_SIZE)
            pg.draw.rect(self.screen, ca, a_rect)

            cabin = agent.agent["cabin"]["vertices"]
            cabin = [(el[0] + offset + total_offset_x, el[1] + total_offset_y) for el in cabin]
            cabin_color = agent.agent["cabin"]["color"]
            pg.draw.polygon(self.screen, cabin_color, cabin)

        if self.display:
            pg.display.flip()
