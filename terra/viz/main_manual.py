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
    K_e,
    K_r,
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
from terra.viz.a_star import a_star

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
    #print(f"{timestep.state.agent.width=}")
    #print(f"{timestep.state.agent.height=}")

    rng, _rng = jax.random.split(rng)
    _rng = _rng[None]

    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))

    # Trigger the JIT compilation
    timestep = env.step(timestep, repeat_action(action_type.do_nothing()), _rng)
    end_time = time.time()
    print(f"Environment started. Compilation time: {end_time - start_time} seconds.")
    
    state = timestep.state
    current_position, target_position = extract_positions(state)

    # Extract current and target positions
    state = timestep.state
    current_position, target_position = extract_positions(state)

    # Convert positions to tuples
    start = (int(current_position["x"]), int(current_position["y"]))
    target = (int(target_position["x"]), int(target_position["y"])) if target_position else None

    print(f"Current Position: {start}")
    print(f"Target Position: {target}")

    # Run the A* algorithm if the target exists
    if target:
        grid = state.world.target_map.map[0]  # Extract the 2D grid
        grid = grid.at[start[0], start[1]].set(7)
        grid = grid.at[target[0], target[1]].set(8)
        print("\nOriginal Grid:")
        print(grid)

        # Adjust the grid for A* logic:
        # - Convert `-1` (target) to `1` (traversable) for pathfinding
        # - Keep `0` as non-traversable
        # - Keep `1` as free (traversable)
        adjusted_grid = grid.copy()
        adjusted_grid = adjusted_grid.at[adjusted_grid == -1].set(1)  # Treat target as traversable
        adjusted_grid = adjusted_grid.at[adjusted_grid == 7].set(1)  
        adjusted_grid = adjusted_grid.at[adjusted_grid == 8].set(1)

        # neighbors = [
        #     (start[0] + dx, start[1] + dy)
        #     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # ]
        # print(f"Neighbors of {start}:")
        # for neighbor in neighbors:
        #     x, y = neighbor
        #     if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
        #         print(f"Neighbor {neighbor}: {grid[x, y]}")

        print("\n Adjusted Grid:")
        print(adjusted_grid)
        path = a_star(adjusted_grid, start, target)

        if path:
            print(f"\n Computed Path: {path}")

            # Highlight the path in the grid
            highlighted_grid = grid.copy()
            for x, y in path:
                #print(f"Path: {x}, {y}")
                #print(f"Type of grid: {type(grid)}")
                #print(f"Type of highlighted_grid: {type(highlighted_grid)}")
                highlighted_grid = highlighted_grid.at[x, y].set(9)  # Mark the path with 9
                #print(highlighted_grid)

            print("\nHighlighted Grid (Path marked with 9):")
            print(highlighted_grid)
            
            game = env.terra_env.rendering_engine
            game.path = path
        else:
            print("No path found.")
    else:
        print("Target position not available.")

    playing = True
    while playing:
        for event in pg.event.get():
            state = timestep.state
            base_orientation = extract_base_orientation(state)
            bucket_status = extract_bucket_status(state)  # Extract bucket status

            #print(base_orientation)
            #print(bucket_status)        

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
                elif event.key == K_e:
                    action = action_type.extend_arm()
                elif event.key == K_r:
                    action = action_type.retract_arm()
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
