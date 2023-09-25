
import pygame as pg
import numpy as np
from game.game import Game
from game.settings import TILE_SIZE


def main():
    nx = 2
    ny = 3

    pg.init()
    pg.mixer.init()
    clock = pg.time.Clock()
    x = 2 * ny * 65 * TILE_SIZE + 8*TILE_SIZE
    y = nx * 69 * TILE_SIZE + 8*TILE_SIZE
    screen = pg.display.set_mode((x, y))
    surface = pg.Surface((x, y), pg.SRCALPHA)
    
    game = Game(screen, surface, clock, n_envs_x=nx, n_envs_y=ny, progressive_gif=False)
    n_envs = nx*ny

    mock_target_map = np.ones((1, 60, 60))
    mock_target_map[:, 10:20, 10:40] = -np.ones((1, 10, 30))

    mock_action_map = np.zeros((1, 60, 60))
    mock_action_map[:, 10:20, 10:40] = -np.ones((1, 10, 30))
    mock_action_map[:, 25:35, 10:40] = np.ones((1, 10, 30))

    mock_obstacles = np.zeros((1, 60, 60))
    mock_obstacles[:, 50:55, 50:55] = np.ones((1, 5, 5))

    mock_dumpability = np.ones((1, 60, 60))
    mock_dumpability[:, 30:35, 30:35] = np.zeros((1, 5, 5))

    mock_target_map = mock_target_map.repeat(n_envs, 0)
    mock_action_map = mock_action_map.repeat(n_envs, 0)
    mock_obstacles = mock_obstacles.repeat(n_envs, 0)
    mock_dumpability = mock_dumpability.repeat(n_envs, 0)
    
    playing = True
    while playing:
        playing = False
        game.run(
            active_grid=mock_action_map,
            target_grid=mock_target_map,
            padding_mask=mock_obstacles,
            dumpability_mask=mock_dumpability,
            agent_pos=np.array([[6, 6]], dtype=np.int32).repeat(n_envs, 0),
            base_dir=np.array([[1]], dtype=np.int32).repeat(n_envs, 0),
            cabin_dir=np.array([[2]], dtype=np.int32).repeat(n_envs, 0),
            generate_gif=False,
        )
        import time
        time.sleep(5)

if __name__ == "__main__":
    main()
