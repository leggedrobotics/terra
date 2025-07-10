import time
from time import gmtime
from time import strftime
import sys
import jax.numpy as jnp
import numpy as np
import pandas as pd
import jax

from terra.actions import TrackedActionType, TrackedAction
from terra.config import EnvConfig
from terra.env import TerraEnvBatch

# Import the global flag from the state module
import terra.state as terra_state

def debug_soil_mechanics_step(step_num, timestep, new_timestep, enable_soil_mechanics):
    """Detailed debugging of a single step to understand soil mechanics behavior"""
    print(f"\n--- STEP {step_num} DEBUG ---")
    
    # Agent state analysis
    was_loaded = timestep.state.agent.agent_state.loaded
    is_loaded = new_timestep.state.agent.agent_state.loaded
    
    print(f"Agent loaded: {was_loaded} -> {is_loaded}")
    
    # Action map analysis (dirt on ground)
    old_dirt = jnp.sum(timestep.state.world.action_map.map > 0)
    new_dirt = jnp.sum(new_timestep.state.world.action_map.map > 0)
    print(f"Ground dirt: {old_dirt} -> {new_dirt}")
    
    # Analyze dirt spread patterns
    old_map = timestep.state.world.action_map.map
    new_map = new_timestep.state.world.action_map.map
    
    # Count tiles with different dirt amounts
    old_dirt_tiles = jnp.sum(old_map > 0)
    new_dirt_tiles = jnp.sum(new_map > 0)
    
    # Calculate dirt concentration (average dirt per tile)
    old_concentration = jnp.where(old_dirt_tiles > 0, old_dirt / old_dirt_tiles, 0)
    new_concentration = jnp.where(new_dirt_tiles > 0, new_dirt / new_dirt_tiles, 0)
    
    # Find max dirt height
    old_max_height = jnp.max(old_map)
    new_max_height = jnp.max(new_map)
    
    print(f"Dirt tiles: {old_dirt_tiles} -> {new_dirt_tiles}")
    print(f"Dirt concentration: {old_concentration:.2f} -> {new_concentration:.2f}")
    print(f"Max dirt height: {old_max_height} -> {new_max_height}")
    
    # Check if this was a successful dump
    successful_dump = jnp.logical_and(was_loaded > 0, is_loaded == 0)
    print(f"Successful dump: {successful_dump}")
    
    # Count successful dumps (this is what we actually want to track)
    successful_dump_count = jnp.sum(successful_dump)
    print(f"Successful dumps this step: {successful_dump_count}")
    
    # For soil mechanics events, we need to check if soil mechanics actually triggered
    # Since we can't easily access the internal soil mechanics state from here,
    # we'll just return the successful dump count and let the caller decide
    # whether to count them as soil mechanics events based on the enable_soil_mechanics flag
    return successful_dump_count

def analyze_dirt_spread(timestep):
    """Analyze dirt spread patterns in the action map"""
    action_map = timestep.state.world.action_map.map
    
    # Count tiles with different dirt amounts
    total_dirt = jnp.sum(action_map > 0)
    dirt_tiles = jnp.sum(action_map > 0)
    
    # Calculate statistics
    concentration = jnp.where(dirt_tiles > 0, total_dirt / dirt_tiles, 0)
    max_height = jnp.max(action_map)
    min_height = jnp.min(action_map)
    
    # Count tiles by dirt height
    height_1 = jnp.sum(action_map == 1)
    height_2 = jnp.sum(action_map == 2)
    height_3 = jnp.sum(action_map == 3)
    height_4 = jnp.sum(action_map == 4)
    height_5_plus = jnp.sum(action_map >= 5)
    
    return {
        'total_dirt': total_dirt,
        'dirt_tiles': dirt_tiles,
        'concentration': concentration,
        'max_height': max_height,
        'min_height': min_height,
        'height_1': height_1,
        'height_2': height_2,
        'height_3': height_3,
        'height_4': height_4,
        'height_5_plus': height_5_plus
    }

def run_detailed_benchmark(enable_soil_mechanics, batch_size=5, episode_length=10):
    """Run a detailed benchmark with extensive debugging"""
    print(f"\n{'='*60}")
    print(f"DETAILED SOIL MECHANICS DEBUG")
    print(f"Soil mechanics: {'ENABLED' if enable_soil_mechanics else 'DISABLED'}")
    print(f"Batch size: {batch_size}")
    print(f"Episode length: {episode_length}")
    print(f"{'='*60}")
    
    # Note: ENABLE_SOIL_MECHANICS is set manually in terra/terra/state.py
    print(f"DEBUG: Current test with soil mechanics: {'ENABLED' if enable_soil_mechanics else 'DISABLED'}")
    print(f"DEBUG: Make sure to set ENABLE_SOIL_MECHANICS in state.py and restart process between runs!")
    
    env_batch = TerraEnvBatch()
    env_cfgs = jax.vmap(lambda x: EnvConfig())(jnp.arange(batch_size))
    rng = jax.random.PRNGKey(0)
    rng_keys = jax.random.split(rng, batch_size)
    timestep = env_batch.reset(env_cfgs, rng_keys)
    
    # Directly load agents with dirt for testing
    print(f"\nLoading agents with dirt for testing...")
    loaded_agents = timestep.state.agent.agent_state.loaded.at[:].set(40)  # Set all agents to have 40 dirt
    timestep = timestep._replace(
        state=timestep.state._replace(
            agent=timestep.state.agent._replace(
                agent_state=timestep.state.agent.agent_state._replace(
                    loaded=loaded_agents
                )
            )
        )
    )
    
    print(f"\nInitial state:")
    print(f"  Agents loaded: {timestep.state.agent.agent_state.loaded}")
    print(f"  Ground dirt: {jnp.sum(timestep.state.world.action_map.map > 0)}")
    print(f"  Soil mechanics flag: {terra_state.ENABLE_SOIL_MECHANICS}")
    
    # WARMUP
    print(f"\nRunning warmup...")
    warmup_actions = jax.vmap(lambda x: TrackedAction.do())(jnp.arange(batch_size))
    warmup_keys = jax.random.split(jax.random.PRNGKey(999), batch_size)
    for _ in range(3):
        timestep = env_batch.step(timestep, warmup_actions, warmup_keys)
    print(f"Warmup complete.")
    
    # Reload agents after warmup (warmup might have cleared them)
    print(f"Reloading agents after warmup...")
    loaded_agents = timestep.state.agent.agent_state.loaded.at[:].set(40)
    timestep = timestep._replace(
        state=timestep.state._replace(
            agent=timestep.state.agent._replace(
                agent_state=timestep.state.agent.agent_state._replace(
                    loaded=loaded_agents
                )
            )
        )
    )
    print(f"Agents after reload: {timestep.state.agent.agent_state.loaded}")
    
    duration = 0
    successful_dumps = 0
    soil_mechanics_events = 0
    detailed_events = []
    dirt_spread_stats = []
    
    for i in range(episode_length):
        print(f"\n--- EPISODE STEP {i+1} ---")
        
        # Use dump actions to trigger soil mechanics (agents are already loaded)
        actions = jax.vmap(lambda x: TrackedAction.do())(jnp.arange(batch_size))
        step_rng = jax.random.PRNGKey(i)
        maps_buffer_keys = jax.random.split(step_rng, batch_size)
        
        # Check pre-step state
        agents_with_dirt = jnp.sum(timestep.state.agent.agent_state.loaded > 0)
        ground_dirt = jnp.sum(timestep.state.world.action_map.map > 0)
        
        print(f"Pre-step: {agents_with_dirt} agents loaded, {ground_dirt} dirt on ground")
        print(f"  Agent dirt amounts: {timestep.state.agent.agent_state.loaded}")
        
        # Time the step
        s = time.time()
        new_timestep = env_batch.step(timestep, actions, maps_buffer_keys)
        e = time.time()
        step_time = e - s
        duration += step_time
        
        # Debug this step
        successful_dump_count = debug_soil_mechanics_step(i+1, timestep, new_timestep, enable_soil_mechanics)
        
        # Count events
        was_loaded = timestep.state.agent.agent_state.loaded > 0
        is_loaded = new_timestep.state.agent.agent_state.loaded > 0
        step_dumps = jnp.sum(jnp.logical_and(was_loaded, ~is_loaded))
        successful_dumps += step_dumps
        
        # Handle soil mechanics events - only count if soil mechanics is enabled
        if isinstance(successful_dump_count, (jnp.ndarray, np.ndarray)):
            soil_mechanics_count = jnp.sum(successful_dump_count) if enable_soil_mechanics else 0
        else:
            soil_mechanics_count = successful_dump_count if enable_soil_mechanics else 0
        soil_mechanics_events += soil_mechanics_count
        
        # Record detailed event
        event = {
            'step': i+1,
            'time': step_time,
            'agents_loaded_before': agents_with_dirt,
            'ground_dirt_before': ground_dirt,
            'successful_dumps': step_dumps,
            'soil_mechanics_triggered': soil_mechanics_count,
            'agents_loaded_after': jnp.sum(new_timestep.state.agent.agent_state.loaded > 0),
            'ground_dirt_after': jnp.sum(new_timestep.state.world.action_map.map > 0),
        }
        detailed_events.append(event)
        
        # Analyze dirt spread after the step
        dirt_stats = analyze_dirt_spread(new_timestep)
        dirt_spread_stats.append(dirt_stats)
        
        print(f"Step time: {step_time:.6f}s")
        print(f"Successful dumps this step: {step_dumps}")
        print(f"Soil mechanics triggered this step: {soil_mechanics_count}")
        
        # Reload agents with dirt and clear action map for next step (so we can continue testing dumps)
        if i < episode_length - 1:  # Don't reload on the last step
            # Reload agents with dirt
            loaded_agents = new_timestep.state.agent.agent_state.loaded.at[:].set(40)
            
            # Clear the action map (remove all dumped dirt) to avoid accumulation
            cleared_action_map = new_timestep.state.world.action_map.map.at[:].set(0)
            
            new_timestep = new_timestep._replace(
                state=new_timestep.state._replace(
                    agent=new_timestep.state.agent._replace(
                        agent_state=new_timestep.state.agent.agent_state._replace(
                            loaded=loaded_agents
                        )
                    ),
                    world=new_timestep.state.world._replace(
                        action_map=new_timestep.state.world.action_map._replace(
                            map=cleared_action_map
                        )
                    )
                )
            )
        
        timestep = new_timestep
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {duration:.6f}s")
    print(f"Average step time: {duration/episode_length:.6f}s")
    print(f"Total successful dumps: {successful_dumps}")
    print(f"Total soil mechanics events: {soil_mechanics_events}")
    print(f"Final agents loaded: {timestep.state.agent.agent_state.loaded}")
    print(f"Final ground dirt: {jnp.sum(timestep.state.world.action_map.map > 0)}")
    
    # Analyze dirt spread patterns
    if dirt_spread_stats:
        print(f"\nDIRT SPREAD ANALYSIS:")
        print(f"{'='*40}")
        
        # Calculate averages across all steps
        avg_concentration = np.mean([s['concentration'] for s in dirt_spread_stats])
        avg_max_height = np.mean([s['max_height'] for s in dirt_spread_stats])
        avg_dirt_tiles = np.mean([s['dirt_tiles'] for s in dirt_spread_stats])
        
        # Calculate height distribution
        total_height_1 = sum(s['height_1'] for s in dirt_spread_stats)
        total_height_2 = sum(s['height_2'] for s in dirt_spread_stats)
        total_height_3 = sum(s['height_3'] for s in dirt_spread_stats)
        total_height_4 = sum(s['height_4'] for s in dirt_spread_stats)
        total_height_5_plus = sum(s['height_5_plus'] for s in dirt_spread_stats)
        
        print(f"Average dirt concentration: {avg_concentration:.2f}")
        print(f"Average max dirt height: {avg_max_height:.2f}")
        print(f"Average dirt tiles: {avg_dirt_tiles:.2f}")
        print(f"\nHeight distribution (total across all steps):")
        print(f"  Height 1: {total_height_1}")
        print(f"  Height 2: {total_height_2}")
        print(f"  Height 3: {total_height_3}")
        print(f"  Height 4: {total_height_4}")
        print(f"  Height 5+: {total_height_5_plus}")
        
        # Store for comparison
        dirt_analysis = {
            'avg_concentration': avg_concentration,
            'avg_max_height': avg_max_height,
            'avg_dirt_tiles': avg_dirt_tiles,
            'height_distribution': {
                'height_1': total_height_1,
                'height_2': total_height_2,
                'height_3': total_height_3,
                'height_4': total_height_4,
                'height_5_plus': total_height_5_plus
            }
        }
    else:
        dirt_analysis = None
    
    # Detailed events table
    print(f"\nDETAILED EVENTS:")
    print(f"{'Step':<4} {'Time(s)':<8} {'Loaded':<7} {'Dirt':<4} {'Dumps':<6} {'Soil':<5} {'Loaded':<7} {'Dirt':<4}")
    print(f"{'':<4} {'':<8} {'Before':<7} {'Before':<4} {'':<6} {'Mech':<5} {'After':<7} {'After':<4}")
    print(f"{'-'*50}")
    
    for event in detailed_events:
        print(f"{event['step']:<4} {event['time']:<8.6f} {event['agents_loaded_before']:<7} "
              f"{event['ground_dirt_before']:<4} {event['successful_dumps']:<6} "
              f"{event['soil_mechanics_triggered']:<5} {event['agents_loaded_after']:<7} "
              f"{event['ground_dirt_after']:<4}")
    
    return {
        'total_time': duration,
        'avg_step_time': duration/episode_length,
        'successful_dumps': successful_dumps,
        'soil_mechanics_events': soil_mechanics_events,
        'detailed_events': detailed_events,
        'dirt_analysis': dirt_analysis
    }

if __name__ == "__main__":
    print("=== MANUAL SOIL MECHANICS TESTING ===")
    print("This script requires manual flag setting in terra/terra/state.py")
    print()
    print("INSTRUCTIONS:")
    print("1. Set ENABLE_SOIL_MECHANICS = True in terra/terra/state.py")
    print("2. Run this script and record results")
    print("3. Set ENABLE_SOIL_MECHANICS = False in terra/terra/state.py")
    print("4. Restart Python process and run this script again")
    print("5. Compare the results manually")
    print()
    print("Current run will use whatever ENABLE_SOIL_MECHANICS is set to in state.py")
    print("="*60)
    
    # Run the benchmark with the current flag setting
    # The enable_soil_mechanics parameter is now just for display purposes
    print("\nRunning benchmark with current ENABLE_SOIL_MECHANICS setting...")
    
    # Run multiple trials for statistical significance
    num_trials = 3
    batch_size = 16
    episode_length = 50
    results_list = []
    
    for trial in range(num_trials):
        print(f"\n{'='*60}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'='*60}")
        
        # Run detailed benchmark with current flag setting
        print(f"\nTrial {trial + 1}:")
        results = run_detailed_benchmark(terra_state.ENABLE_SOIL_MECHANICS, batch_size=batch_size, episode_length=episode_length)
        results_list.append(results)
    
    # Calculate statistics across trials
    times = [r['avg_step_time'] for r in results_list]
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Trials: {num_trials}")
    print(f"Batch size: {batch_size}")
    print(f"Episode length: {episode_length}")
    print(f"Total steps per trial: {batch_size * episode_length}")
    print(f"Total steps across all trials: {batch_size * episode_length * num_trials}")
    
    print(f"\nAverage step time: {avg_time:.6f}s ± {std_time:.6f}s")
    
    # Total dumps across all trials
    total_dumps = sum(r['successful_dumps'] for r in results_list)
    total_soil_events = sum(r['soil_mechanics_events'] for r in results_list)
    
    print(f"\nTotal successful dumps: {total_dumps}")
    print(f"Total soil mechanics events: {total_soil_events}")
    
    # Dirt analysis from first trial
    if results_list and results_list[0]['dirt_analysis']:
        dirt_analysis = results_list[0]['dirt_analysis']
        print(f"\nDIRT SPREAD ANALYSIS:")
        print(f"Average dirt concentration: {dirt_analysis['avg_concentration']:.2f}")
        print(f"Average max dirt height: {dirt_analysis['avg_max_height']:.2f}")
        print(f"Average dirt tiles: {dirt_analysis['avg_dirt_tiles']:.2f}")
        
        print(f"\nHeight distribution:")
        for height, count in dirt_analysis['height_distribution'].items():
            print(f"  {height}: {count}")
    
   
    print(f"Detailed analysis complete!") 