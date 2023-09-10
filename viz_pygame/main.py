
import pygame as pg
from game.game import Game


def main():

    running = True
    playing = True

    pg.init()
    pg.mixer.init()
    screen = pg.display.set_mode((1920, 1080))
    clock = pg.time.Clock()

    # implement menus

    # implement game
    game = Game(screen, clock)

    while running:

        # start menu goes here

        while playing:
            # game loop here
            game.run()

if __name__ == "__main__":
    main()
