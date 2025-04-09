import time
import jax
import jax.numpy as jnp
import pygame as pg
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_a,
    K_d,
    K_i,
    K_o,
    K_k,
    K_l,
    K_SPACE,
    KEYDOWN,
    QUIT,
)
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch
from terra.viz.llms_utils import *
from terra.viz.a_star import compute_path, simplify_path


def main():
    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type
    n_envs_x = 1
    n_envs_y = 1
    n_envs = n_envs_x * n_envs_y
    seed = 24
    rng = jax.random.PRNGKey(seed)
    env = TerraEnvBatch(
        rendering=True,
        display=True,
        n_envs_x_rendering=n_envs_x,
        n_envs_y_rendering=n_envs_y,
    )

    print("Starting the environment...")
    start_time = time.time()
    env_cfgs = jax.vmap(lambda x: EnvConfig.new())(jnp.arange(n_envs))
    rng, _rng = jax.random.split(rng)
    _rng = _rng[None]
    timestep = env.reset(env_cfgs, _rng)

    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))

    # Trigger the JIT compilation
    timestep = env.step(timestep, repeat_action(action_type.do_nothing()), _rng)
    end_time = time.time()
    print(f"Environment started. Compilation time: {end_time - start_time} seconds.")
    
    current_map = timestep.state.world.target_map.map[0]  # Extract the target map
    previous_map = current_map.copy()  # Initialize the previous map
    count_map_change = 0
    DETERMINISTIC = True
    
    playing = True
    while playing:
        for event in pg.event.get():
            current_map = timestep.state.world.target_map.map[0]  # Extract the target map

            # Check if the map has changed
            if previous_map is None or not jnp.array_equal(previous_map, current_map):
                print("Map has changed.")
                count_map_change += 1
                previous_map = current_map.copy()  # Update the previous map

            if event.type == KEYDOWN:
                action = None
                if event.key == K_UP:
                    action = action_type.forward()
                elif event.key == K_DOWN:
                    action = action_type.backward()
                elif event.key == K_LEFT:
                    action = action_type.anticlock()
                elif event.key == K_RIGHT:
                    action = action_type.clock()
                elif event.key == K_a:
                    action = action_type.cabin_anticlock()
                elif event.key == K_d:
                    action = action_type.cabin_clock()
                elif event.key == K_o:
                    action = action_type.clock_forward()
                elif event.key == K_k:
                    action = action_type.clock_backward()
                elif event.key == K_i:
                    action = action_type.anticlock_forward()
                elif event.key == K_l:
                    action = action_type.anticlock_backward()
                elif event.key == K_SPACE:
                    action = action_type.do()

                if action is not None:
                    print("Action: ", action)
                    #print("count_map_change: ", count_map_change)
                     
                    if DETERMINISTIC:
                        key = jnp.array([[count_map_change, count_map_change]], dtype=jnp.uint32)  # Convert to a JAX array

                        timestep = env.step(
                            timestep,
                            repeat_action(action),
                            key,
                        )
                    else:
                        rng, _rng = jax.random.split(rng)
                        _rng = _rng[None]

                        timestep = env.step(
                            timestep,
                            repeat_action(action),
                            _rng,
                        )

                    print("Reward: ", timestep.reward.item())

            elif event.type == QUIT:
                playing = False


        env.terra_env.render_obs_pygame(
            timestep.observation,
            timestep.info,
        )


if __name__ == "__main__":
    main()

