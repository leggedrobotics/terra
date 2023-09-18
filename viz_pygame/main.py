
import pygame as pg
import numpy as np
from game.game import Game


def main():
    pg.init()
    pg.mixer.init()
    screen = pg.display.set_mode((1920 // 2, 1080 // 2))
    clock = pg.time.Clock()

    game = Game(screen, clock)

    playing = True
    mock_target_map = np.random.randint(-1, 2, (1,60,60), dtype=np.int32)
    while playing:
        game.run(
            active_grid=np.ones((1,60,60), dtype=np.int32),
            target_grid=mock_target_map,
            agent_pos=np.array([[6, 6]], dtype=np.int32),
            base_dir=np.array([[1]], dtype=np.int32),
            cabin_dir=np.array([[2]], dtype=np.int32),
        )

if __name__ == "__main__":
    main()
