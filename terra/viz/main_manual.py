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
    K_k,
    K_l,
    K_SPACE,
    KEYDOWN,
    QUIT,
)
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch


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
    print(f"{timestep.state.agent.width=}")
    print(f"{timestep.state.agent.height=}")

    rng, _rng = jax.random.split(rng)
    _rng = _rng[None]

    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))

    # Trigger the JIT compilation
    timestep = env.step(timestep, repeat_action(action_type.do_nothing()), _rng)
    end_time = time.time()
    print(f"Environment started. Compilation time: {end_time - start_time} seconds.")

    playing = True
    while playing:
        for event in pg.event.get():
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
                elif event.key == K_k:
                    action = action_type.wheels_left()
                elif event.key == K_l:
                    action = action_type.wheels_right()
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

                    # DEBUG: Show maximum dirt on any single tile
                    action_map = timestep.state.world.action_map.map[0]  # First environment
                    max_dirt = jnp.max(action_map)
                    min_dirt = jnp.min(action_map)
                    total_dirt = jnp.sum(action_map)
                    print(f"🏔️ MAX DIRT ON TILE: {max_dirt}")
                    print(f"⛳ MIN DIRT ON TILE: {min_dirt}")
                    print(f"🌍 TOTAL DIRT: {total_dirt}")
                    
                    # Show agent loading state for context
                    agent_loaded = timestep.state.agent.agent_state.loaded[0]
                    total_dirt_with_agent = total_dirt + agent_loaded
                    print(f"🚜 AGENT: Loaded={agent_loaded}")
                    print(f"🧮 TOTAL DIRT (map + agent): {total_dirt_with_agent}")
                    print("-" * 40)

            elif event.type == QUIT:
                playing = False

        env.terra_env.render_obs_pygame(
            timestep.observation,
            timestep.info,
        )


if __name__ == "__main__":
    main()
