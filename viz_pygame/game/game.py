import pygame as pg
import sys
from .world import World
from .agent import Agent
from .settings import TILE_SIZE


class Game:

    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.width, self.height = self.screen.get_size()

        self.world = World(10, 10, self.width, self.height)
        self.agent = Agent(2, 3)

    def run(self):
        self.playing = True
        while self.playing:
            self.clock.tick(60)
            self.events()
            self.update()
            self.draw()

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()

    def update(self):
        pass

    def draw(self):
        self.screen.fill((255, 255, 255,))

        for x in range(self.world.grid_length_x):
            for y in range(self.world.grid_length_y):

                sq = self.world.world[x][y]["cart_rect"]
                c = self.world.world[x][y]["color"]
                rect = pg.Rect(sq[0][0], sq[0][1], TILE_SIZE, TILE_SIZE)
                pg.draw.rect(self.screen, c, rect)

        a = self.agent.agent["body"]["vertices"]
        ca = self.agent.agent["body"]["color"]
        a_rect = pg.Rect(a[0][0], a[0][1], self.agent.width * TILE_SIZE, self.agent.height * TILE_SIZE)
        pg.draw.rect(self.screen, ca, a_rect)

        cabin = self.agent.agent["cabin"]["vertices"]
        cabin_color = self.agent.agent["cabin"]["color"]
        pg.draw.polygon(self.screen, cabin_color, cabin)

        pg.display.flip()
