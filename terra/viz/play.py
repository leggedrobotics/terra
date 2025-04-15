import time
import jax
import jax.numpy as jnp
import pygame as pg
from pygame.locals import (
    KEYDOWN,
    QUIT,
)
from terra.config import EnvConfig
from terra.env import TerraEnvBatch


def main():
    n_envs_x = 1
    n_envs_y = 1
    n_envs = n_envs_x * n_envs_y
    seed = 24
    rng = jax.random.PRNGKey(seed)
    shuffle_maps = True
    env = TerraEnvBatch(
        rendering=True,
        display=True,
        n_envs_x_rendering=n_envs_x,
        n_envs_y_rendering=n_envs_y,
        shuffle_maps=shuffle_maps,
    )

    print("Starting the environment...")
    start_time = time.time()
    env_cfgs = jax.vmap(lambda x: EnvConfig.new())(jnp.arange(n_envs))
    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(_rng, n_envs)
    timestep = env.reset(env_cfgs, _rng)
    end_time = time.time()
    print(f"Environment started. Compilation time: {end_time - start_time} seconds.")
    print("Press any key to query the next set of environments.")
    playing = True
    while playing:
        for event in pg.event.get():
            if event.type == KEYDOWN:
                rng, _rng = jax.random.split(rng)
                _rng = jax.random.split(_rng, n_envs)
                timestep = env.reset(env_cfgs, _rng)

            elif event.type == QUIT:
                playing = False

        env.terra_env.render_obs_pygame(
            timestep.observation,
        )


if __name__ == "__main__":
    main()
