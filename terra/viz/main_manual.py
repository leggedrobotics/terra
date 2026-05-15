import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import time
import jax
import jax.numpy as jnp
import numpy as np
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
    K_r,
    K_SPACE,
    K_RETURN,
    KEYDOWN,
    QUIT,
)
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch


def _foundation_border_pixels(target_map):
    dig = target_map < 0
    padded = np.pad(dig, 1, constant_values=False)
    inner = dig.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            inner &= padded[1 + dr : 1 + dr + dig.shape[0], 1 + dc : 1 + dc + dig.shape[1]]
    border = dig & ~inner
    rows, cols = np.where(border)
    return cols.astype(np.float32), rows.astype(np.float32)


def _axis_border_segment(axis, border_cols, border_rows, nearest_axis, axis_idx):
    if border_cols.size == 0:
        return None

    a, b, c = (float(axis[0]), float(axis[1]), float(axis[2]))
    denom = np.hypot(a, b)
    if denom < 1e-6:
        return None

    selected = nearest_axis == axis_idx
    if np.count_nonzero(selected) < 2:
        return None

    x = border_cols[selected]
    y = border_rows[selected]
    signed = (a * x + b * y + c) / denom
    proj_x = x - signed * (a / denom)
    proj_y = y - signed * (b / denom)

    dir_x = b / denom
    dir_y = -a / denom
    t = proj_x * dir_x + proj_y * dir_y
    start = int(np.argmin(t))
    end = int(np.argmax(t))
    return (proj_x[start], proj_y[start]), (proj_x[end], proj_y[end])


def draw_foundation_axis_overlay(env, timestep):
    renderer = env.terra_env.rendering_engine
    if renderer is None or renderer.screen is None:
        return

    axes_batch = np.asarray(timestep.state.world.foundation_border_axes)
    types_batch = np.asarray(timestep.state.world.foundation_border_type)
    target_batch = np.asarray(timestep.state.world.target_map.map)
    if axes_batch.ndim == 2:
        axes_batch = axes_batch[None, ...]
    if types_batch.ndim == 0:
        types_batch = np.asarray([types_batch])
    if target_batch.ndim == 2:
        target_batch = target_batch[None, ...]

    tile_size = renderer.tile_size
    map_size = renderer.maps_size_px
    map_px = renderer.total_display_size
    border_px = 4 * tile_size

    colors = [
        "#ffff00",
        "#00ffff",
        "#ff66ff",
        "#ff9900",
        "#66ff66",
        "#ffffff",
    ]

    for env_idx in range(min(renderer.n_envs, axes_batch.shape[0])):
        ix = env_idx % renderer.n_envs_y
        iy = env_idx // renderer.n_envs_y
        offset_x = ix * (map_px + border_px) + border_px
        offset_y = iy * (map_px + border_px) + border_px
        border_type = int(types_batch[min(env_idx, len(types_batch) - 1)])
        valid_axes = axes_batch[env_idx, : max(border_type, 0)]
        valid_axes = valid_axes[valid_axes[:, 0] > -96.0]
        border_cols, border_rows = _foundation_border_pixels(target_batch[env_idx])
        if valid_axes.size == 0 or border_cols.size == 0:
            continue

        dists = []
        for a, b, c in valid_axes:
            denom = np.hypot(float(a), float(b)) + 1e-6
            dists.append(np.abs(float(a) * border_cols + float(b) * border_rows + float(c)) / denom)
        nearest_axis = np.argmin(np.stack(dists, axis=0), axis=0)

        for axis_idx, axis in enumerate(valid_axes):
            segment = _axis_border_segment(axis, border_cols, border_rows, nearest_axis, axis_idx)
            if segment is None:
                continue
            p0, p1 = segment
            screen_p0 = (
                int(offset_x + p0[0] * tile_size),
                int(offset_y + p0[1] * tile_size),
            )
            screen_p1 = (
                int(offset_x + p1[0] * tile_size),
                int(offset_y + p1[1] * tile_size),
            )
            color = colors[axis_idx % len(colors)]
            pg.draw.line(renderer.screen, color, screen_p0, screen_p1, 1)

    valid_axes = axes_batch[0][axes_batch[0, :, 0] > -96.0] if axes_batch.shape[0] else np.empty((0, 3))
    axis_signature = tuple(np.round(valid_axes[:6].reshape(-1), 2).tolist())
    target_signature = int(np.sum(target_batch[0] * np.arange(1, target_batch[0].size + 1).reshape(target_batch[0].shape)))
    signature = (target_signature, axis_signature)
    if getattr(draw_foundation_axis_overlay, "_last_signature", None) != signature:
        draw_foundation_axis_overlay._last_signature = signature
        print(
            "[overlay] target_sig="
            f"{target_signature} axes={len(valid_axes)} first_axes={axis_signature}"
        )

    if renderer.display:
        pg.display.flip()


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
    
    # Example: Configure different agent and action types
    # env_cfgs = jax.vmap(lambda x: EnvConfig(agent_types=(0, 1), action_types=(0, 1)))(jnp.arange(n_envs))  # Excavator+Tracked, Truck+Wheeled
    # env_cfgs = jax.vmap(lambda x: EnvConfig(agent_types=(0, 2), action_types=(1, 1)))(jnp.arange(n_envs))  # Excavator+Wheeled, SkidSteer+Wheeled
    # env_cfgs = jax.vmap(lambda x: EnvConfig(agent_types=(0, 1), action_types=(1, 1)))(jnp.arange(n_envs))  # Excavator+Wheeled, Truck+Wheeled
    env_cfgs = jax.vmap(lambda x: EnvConfig.new())(
        jnp.arange(n_envs)
    )  # Default: (0,2) agents, (0,0) actions
    rng, _rng = jax.random.split(rng)
    _rng = _rng[None]
    timestep = env.reset(env_cfgs, _rng)
    
    # Action type will be determined dynamically based on current agent
    print(f"{timestep.state.agent.width=}")
    print(f"{timestep.state.agent.height=}")
    
    # Show agent and action type configuration
    agent_types = timestep.state.env_cfg.agent_types
    action_types = timestep.state.env_cfg.action_types
    print(f"🤖 Agent Types: {agent_types}")
    print(f"🚗 Action Types: {action_types}")
    
    # Show which action type is currently being used
    current_action_type = timestep.state.agent.agent_states[0].action_type[0].item()
    action_type_names = {0: "Tracked", 1: "Wheeled"}
    print(f"🎮 Current Action Type: {current_action_type} ({action_type_names.get(current_action_type, 'Unknown')})")

    rng, _rng = jax.random.split(rng)
    _rng = _rng[None]

    def get_current_action_type():
        """Get the action type for the current active agent"""
        current_agent_idx = timestep.state.agent.current_agent.item()
        current_agent_state = timestep.state.agent.agent_states[current_agent_idx]
        action_type_val = current_agent_state.action_type[0].item()
        
        # Import action types
        from terra.config import TrackedAction, WheeledAction
        if action_type_val == 0:
            return TrackedAction()
        else:
            return WheeledAction()
    
    def repeat_action(action, n_times=n_envs):
        return action.new(action.action[None].repeat(n_times, 0))

    # Trigger the JIT compilation
    current_action_type = get_current_action_type()
    timestep = env.step(timestep, repeat_action(current_action_type.do_nothing()), _rng)
    end_time = time.time()
    print(f"Environment started. Compilation time: {end_time - start_time} seconds.")

    playing = True
    while playing:
        for event in pg.event.get():
            if event.type == KEYDOWN:
                # Get the current action type for the active agent
                current_action_type = get_current_action_type()
                action = None
                if event.key == K_UP:
                    action = current_action_type.forward()
                elif event.key == K_DOWN:
                    action = current_action_type.backward()
                elif event.key == K_LEFT:
                    action = current_action_type.anticlock()
                elif event.key == K_RIGHT:
                    action = current_action_type.clock()
                elif event.key == K_a:
                    action = current_action_type.cabin_anticlock()
                elif event.key == K_d:
                    action = current_action_type.cabin_clock()
                elif event.key == K_k:
                    action = current_action_type.wheels_left()
                elif event.key == K_l:
                    action = current_action_type.wheels_right()
                elif event.key == K_SPACE:
                    action = current_action_type.do()
                elif event.key == K_RETURN:
                    action = current_action_type.do_nothing()
                elif event.key == K_r:
                    rng, _rng = jax.random.split(rng)
                    _rng = _rng[None]
                    timestep = env.reset(env_cfgs, _rng)
                    print("Manual reset: sampled a fresh map.")

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
                    action_type = current_agent_state.action_type[0].item()  # Convert JAX array to Python int
                    reward_value = timestep.reward.item()
                    action_num = action.action[0].item()  # Convert JAX array to Python int
                    
                    agent_type_names = {0: "Excavator", 1: "Truck", 2: "Skid Steer"}
                    action_type_names = {0: "Tracked", 1: "Wheeled"}
                    reward_function_names = {0: "_get_rewards_tracked()", 1: "_get_rewards_truck()", 2: "_get_rewards_skidsteer()"}
                    
                    agent_name = agent_type_names.get(agent_type, f"Unknown({agent_type})")
                    action_name_type = action_type_names.get(action_type, f"Unknown({action_type})")
                    reward_func = reward_function_names.get(agent_type, f"unknown_function({agent_type})")
                    
                    print(f"🎯 REWARD: {reward_value:.4f} | Agent: {agent_name} | Action Type: {action_name_type} | Function: {reward_func}")
                    
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
                    
                    # Show action types for all agents
                    print("🚗 Action Types for all agents:")
                    for i in range(num_agents):
                        # Check if agent is active - use JAX-compatible approach
                        try:
                            # Try to get the scalar value directly
                            is_active = agent_active[i].item() == 1
                        except ValueError:
                            # If that fails, use JAX comparison and check if any element is True
                            comparison_result = agent_active[i] == 1
                            is_active = jnp.any(comparison_result).item()
                        
                        if is_active:
                            agent_state = timestep.state.agent.agent_states[i]
                            agent_type = agent_state.agent_type[0].item()
                            action_type = agent_state.action_type[0].item()
                            agent_name = agent_type_names.get(agent_type, f"Unknown({agent_type})")
                            action_name_type = action_type_names.get(action_type, f"Unknown({action_type})")
                            print(f"   Agent {i}: {agent_name} ({action_name_type})")
                    
                    print("-" * 40)

            elif event.type == QUIT:
                playing = False

        env.terra_env.render_obs_pygame(
            timestep.observation,
            timestep.info,
            generate_gif=False,
        )
        draw_foundation_axis_overlay(env, timestep)


if __name__ == "__main__":
    main()
