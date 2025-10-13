import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
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
    K_RETURN,
    KEYDOWN,
    QUIT,
)
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch


def test_x11_availability():
    """Test if X11 is available for pygame"""
    try:
        # Save current environment
        old_display = os.environ.get('DISPLAY')
        old_videodriver = os.environ.get('SDL_VIDEODRIVER')
        
        # Try to initialize pygame with X11
        pg.init()
        screen = pg.display.set_mode((100, 100))
        pg.quit()
        
        print("X11 is available for pygame")
        return True
    except pg.error as e:
        if "x11 not available" in str(e):
            print("X11 not available for pygame, using dummy driver")
            # Set dummy driver for headless mode
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            os.environ['DISPLAY'] = ':0'
            return False
        else:
            raise e


def main():
    # Test X11 availability and set up display accordingly
    x11_available = test_x11_availability()
    
    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type
    n_envs_x = 1
    n_envs_y = 1
    n_envs = n_envs_x * n_envs_y
    seed = 24
    rng = jax.random.PRNGKey(seed)
    env = TerraEnvBatch(
        rendering=True,
        display=True, #x11_available
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
                elif event.key == K_RETURN:
                    action = action_type.do_nothing()

                if action is not None:
                    print("Action: ", action)
                    rng, _rng = jax.random.split(rng)
                    _rng = _rng[None]
                    timestep = env.step(
                        timestep,
                        repeat_action(action),
                        _rng,
                    )
                    # DEBUG: Show reward with agent type and reward function used
                    # Get the current active agent (index 0 in the reordered observation)
                    current_agent_idx = timestep.state.agent.current_agent.item()  # Convert JAX array to Python int
                    current_agent_state = timestep.state.agent.agent_states[current_agent_idx]
                    agent_type = current_agent_state.agent_type[0].item()  # Convert JAX array to Python int
                    reward_value = timestep.reward.item()
                    action_num = action.action[0].item()  # Convert JAX array to Python int
                    
                    agent_type_names = {0: "Excavator", 1: "Truck", 2: "Skid Steer"}
                    reward_function_names = {0: "_get_rewards_tracked()", 1: "_get_rewards_truck()", 2: "_get_rewards_skidsteer()"}
                    
                    agent_name = agent_type_names.get(agent_type, f"Unknown({agent_type})")
                    reward_func = reward_function_names.get(agent_type, f"unknown_function({agent_type})")
                    
                    print(f"🎯 REWARD: {reward_value:.4f} | Agent: {agent_name} | Function: {reward_func}")
                    
                    # For skid steers, show additional debugging for specific reward components
                    if agent_type == 2:  # Skid steer
                        if action_num == 0:  # Forward action
                            print("   💡 Skid Steer Forward: Auto-loading checked")
                        elif action_num == 6:  # DO action  
                            print("   💡 Skid Steer DO: Dump/Lift/Shovel rewards applied")
                    
                    # Show action details
                    action_names = {
                        0: "FORWARD", 1: "BACKWARD", 2: "CLOCK", 3: "ANTICLOCK", 
                        4: "CABIN_CLOCK", 5: "CABIN_ANTICLOCK", 6: "DO", 7: "DO_NOTHING"
                    }
                    action_name = action_names.get(action_num, f"ACTION_{action_num}")
                    print(f"⚡ ACTION: {action_name} ({action_num})")
                    
                    # DEBUG: Show maximum dirt on any single tile
                    action_map = timestep.state.world.action_map.map[0]  # First environment
                    max_dirt = jnp.max(action_map)
                    min_dirt = jnp.min(action_map)
                    total_dirt = jnp.sum(action_map)
                    print(f"🏔️  MAX DIRT ON TILE: {max_dirt}")
                    print(f"⛳ MIN DIRT ON TILE: {min_dirt}")
                    print(f"🌍 TOTAL DIRT: {total_dirt}")
                    
                    # Show agent loading state for context
                    agent_loaded = current_agent_state.loaded[0]
                    shovel_lifted = current_agent_state.shovel_lifted[0]
                    total_dirt_with_agent = total_dirt + agent_loaded
                    print(f"🚜 AGENT: Loaded={agent_loaded}, Shovel={'UP' if shovel_lifted else 'DOWN'}")
                    print(f"🧮 TOTAL DIRT (map + agent): {total_dirt_with_agent}")
                    
                    # Show multi-agent debug info
                    num_agents = timestep.state.agent.num_agents.item()  # Convert JAX array to Python int
                    agent_active = timestep.state.agent.agent_active
                    print(f"🤖 MULTI-AGENT: Current={current_agent_idx}, Total={num_agents}, Active={agent_active}")
                    print("-" * 40)

            elif event.type == QUIT:
                playing = False

        env.terra_env.render_obs_pygame(
            timestep.observation,
            timestep.info,
            generate_gif=False,
        )


if __name__ == "__main__":
    main()
