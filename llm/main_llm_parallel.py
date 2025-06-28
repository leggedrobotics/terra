"""
Parallelized version of the main LLM experiment script
"""

import numpy as np
import jax
jax.config.update('jax_num_cpu_devices', 16)

from utils.helpers import load_pkl_object

import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input

from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints
from terra.config import EnvConfig
from terra.config import BatchConfig

from llm.utils_llm import *
from terra.viz.llms_adk import *
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)

import asyncio
import os
import argparse
import datetime
import json
import pygame as pg
import multiprocessing as mp
from multiprocessing import Pool, Manager
import functools
import traceback

from pygame.locals import (
    K_q,
    QUIT,
)

from llm.eval_llm import compute_stats_llm
from llm.env_manager_llm import EnvironmentsManager

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# Configuration constants
FORCE_DELEGATE_TO_RL = False
FORCE_DELEGATE_TO_LLM = False
LLM_CALL_FREQUENCY = 15
USE_MANUAL_PARTITIONING = False
NUM_PARTITIONS = 4
VISUALIZE_PARTITIONS = True
USE_IMAGE_PROMPT = True
USE_LOCAL_MAP = True
USE_PATH = False
APP_NAME = "ExcavatorGameApp"
USER_ID = "user_1"
SESSION_ID = "session_001"
GRID_RENDERING = False
ORIGINAL_MAP_SIZE = 128
COMPUTE_BENCH_STATS = True
USE_RENDERING = False  # Disable rendering for parallel execution
USE_DISPLAY = False   # Disable display for parallel execution


def run_single_experiment(args_tuple):
    """
    Wrapper function to run a single experiment.
    This function will be called by each worker process.
    """
    try:
        experiment_id, llm_model_name, llm_model_key, num_timesteps, seed, run_name = args_tuple
        
        print(f"Process {os.getpid()}: Starting experiment {experiment_id} with seed {seed}")

        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['JAX_ENABLE_X64'] = 'False'
        
        # XLA configuration
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16 --xla_cpu_multi_thread_eigen=false'
        
        # BLAS configuration for multiprocessing
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['BLIS_NUM_THREADS'] = '1'
        
        # TensorFlow configuration
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
        
        # Initialize pygame for this process (if needed)
        if USE_RENDERING:
            pg.init()
            pg.display.set_mode((800, 600), pg.NOFRAME)

        print(jax.devices(backend='cpu'))

        
        info = run_experiment(
            llm_model_name=llm_model_name,
            llm_model_key=llm_model_key,
            num_timesteps=num_timesteps,
            seed=seed,
            run=run_name,
            experiment_id=experiment_id
        )
        
        # Extract results
        result = {
            'experiment_id': experiment_id,
            'episode_done_once': info["episode_done_once"].item() if hasattr(info["episode_done_once"], 'item') else info["episode_done_once"],
            'episode_length': info["episode_length"].item() if hasattr(info["episode_length"], 'item') else info["episode_length"],
            'move_cumsum': info["move_cumsum"].item() if hasattr(info["move_cumsum"], 'item') else info["move_cumsum"],
            'do_cumsum': info["do_cumsum"].item() if hasattr(info["do_cumsum"], 'item') else info["do_cumsum"],
            'areas': info["areas"].item() if hasattr(info["areas"], 'item') else info["areas"],
            'dig_tiles_per_target_map_init': info["dig_tiles_per_target_map_init"].item() if hasattr(info["dig_tiles_per_target_map_init"], 'item') else info["dig_tiles_per_target_map_init"],
            'dug_tiles_per_action_map': info["dug_tiles_per_action_map"].item() if hasattr(info["dug_tiles_per_action_map"], 'item') else info["dug_tiles_per_action_map"],
            'success': True
        }
        
        print(f"Process {os.getpid()}: Completed experiment {experiment_id}")
        return result
        
    except Exception as e:
        print(f"Process {os.getpid()}: Error in experiment {experiment_id}: {e}")
        traceback.print_exc()
        return {
            'experiment_id': experiment_id,
            'error': str(e),
            'success': False
        }
    finally:
        # Clean up pygame for this process
        if USE_RENDERING:
            pg.quit()


def run_experiment(llm_model_name, llm_model_key, num_timesteps, seed, 
                run, small_env_config=None, experiment_id=0):
    """
    Modified run_experiment function with experiment_id parameter
    """
    agent_checkpoint_path = run
    model_params = None
    config = None

    print(f"Experiment {experiment_id}: Loading RL agent configuration from: {agent_checkpoint_path}")
    log = load_pkl_object(agent_checkpoint_path)
    config = log["train_config"]
    model_params = log["model"]

    # Create the original environment configs for the full map
    global_env_config = jax.tree_map(
        lambda x: x[0][None, ...].repeat(1, 0), log["env_config"]
    ) 

    config.num_test_rollouts = 1
    config.num_devices = 1
    config.num_embeddings_agent_min = 60

    # Initialize the environment manager ONCE with all maps
    print(f"Experiment {experiment_id}: Initializing environment manager with all maps...")
    env_manager = EnvironmentsManager(
        seed=seed,
        global_env_config=global_env_config,
        small_env_config=small_env_config,
        shuffle_maps=False,
        rendering=USE_RENDERING,
        display=USE_DISPLAY
    )
    print(f"Experiment {experiment_id}: Environment manager initialized.")

    # Initialize once with proper batching
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset_initial = jax.random.split(_rng, 1)

    initial_custom_pos = None
    initial_custom_angle = None
    
    # Initial setup
    env_manager.global_env.timestep = env_manager.global_env.reset(
        global_env_config, rng_reset_initial, initial_custom_pos, initial_custom_angle
    )

    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type

    def repeat_action(action, n_times=1):
        return action_type.new(action.action[None].repeat(n_times, 0))
    
    # Trigger the JIT compilation
    env_manager.global_env.timestep = env_manager.global_env.step(
        env_manager.global_env.timestep, repeat_action(action_type.do_nothing()), rng_reset_initial
    )

    if USE_RENDERING:
        env_manager.global_env.terra_env.render_obs_pygame(
            env_manager.global_env.timestep.observation, env_manager.global_env.timestep.info
        )

    # Initialize variables for tracking progress across all maps
    global_step = 0
    playing = True
    current_map_index = 0
    max_maps = 10  # Set a reasonable limit for number of maps to process
    
    # For visualization and metrics across all maps
    all_frames = []
    all_reward_seq = []
    all_global_step_rewards = []
    all_obs_seq = []
    all_action_list = []
    
    tile_size = global_env_config.tile_size[0].item()
    move_tiles = global_env_config.agent.move_tiles[0].item()

    action_type = batch_cfg.action_type
    if action_type == TrackedAction:
        move_actions = (TrackedActionType.FORWARD, TrackedActionType.BACKWARD)
        l_actions = ()
        do_action = TrackedActionType.DO
    elif action_type == WheeledAction:
        move_actions = (WheeledActionType.FORWARD, WheeledActionType.BACKWARD)
        l_actions = (WheeledActionType.CLOCK, WheeledActionType.ANTICLOCK)
        do_action = WheeledActionType.DO
    else:
        raise (ValueError(f"{action_type=}"))

    obs = env_manager.global_env.timestep.observation
    areas = (obs["target_map"] == -1).sum(
        tuple([i for i in range(len(obs["target_map"].shape))][1:])
            ) * (tile_size**2)
    target_maps_init = obs["target_map"].copy()
    dig_tiles_per_target_map_init = (target_maps_init == -1).sum(
        tuple([i for i in range(len(target_maps_init.shape))][1:])
    )
    reward_seq = []
    episode_done_once = None
    episode_length = None
    move_cumsum = None
    do_cumsum = None
    dug_tiles_per_action_map = 0
    screen = None
    if USE_RENDERING:
        screen = pg.display.get_surface()

    # MAIN LOOP - PROCESS MULTIPLE MAPS
    while playing and global_step < num_timesteps and current_map_index < max_maps:
        print(f"\nExperiment {experiment_id}: {'='*60}")
        print(f"Experiment {experiment_id}: STARTING MAP {current_map_index}")
        print(f"Experiment {experiment_id}: {'='*60}")
        
        # Reset to next map (reusing the same environment)
        try:
            if current_map_index > 0:  # Don't reset on first map since it's already initialized
                reset_to_next_map(current_map_index, seed, env_manager, global_env_config,
                       initial_custom_pos, initial_custom_angle)

            if USE_RENDERING:
                env_manager.global_env.terra_env.render_obs_pygame(
                    env_manager.global_env.timestep.observation, 
                    env_manager.global_env.timestep.info
                )
                screen = pg.display.get_surface()
                game_state_image = capture_screen(screen)
            else:
                screen = None
                game_state_image = None
            
            llm_query, runner_partitioning, runner_delegation, system_message_master, session_manager, prompts = setup_partitions_and_llm(
                        current_map_index, ORIGINAL_MAP_SIZE, env_manager, 
                        config, llm_model_name, llm_model_key,
                        USE_PATH, APP_NAME, f"{USER_ID}_exp_{experiment_id}", f"{SESSION_ID}_exp_{experiment_id}", screen,
                        USE_MANUAL_PARTITIONING, USE_IMAGE_PROMPT)
            partition_states, partition_models, active_partitions = initialize_partitions_for_current_map(env_manager, config, model_params)
            
            if partition_states is None:
                print(f"Experiment {experiment_id}: Failed to initialize map {current_map_index}, moving to next map")
                current_map_index += 1
                continue
                
        except Exception as e:
            print(f"Experiment {experiment_id}: Error setting up map {current_map_index}: {e}")
            current_map_index += 1
            continue

        # Track metrics for this map
        map_frames = []
        map_reward_seq = []
        map_global_step_rewards = []
        map_obs_seq = []
        map_action_list = []
        
        # First step delegate to RL agent
        llm_decision = "delegate_to_rl"

        # MAP-SPECIFIC GAME LOOP
        map_step = 0
        max_steps_per_map = num_timesteps
        map_done = False

        while playing and active_partitions and map_step < max_steps_per_map and global_step < num_timesteps:
            
            print(f"Experiment {experiment_id}: Map {current_map_index}, Step {map_step} (Global {global_step}) - "
                  f"Processing {len(active_partitions)} active partitions")

            if USE_RENDERING:
                screen = pg.display.get_surface()
                game_state_image = capture_screen(screen)
                map_frames.append(game_state_image)
            else:
                screen = None
                game_state_image = None

            # Step all active partitions simultaneously
            partitions_to_remove = []
            current_step_reward = 0.0

            for partition_idx in active_partitions:
                partition_state = partition_states[partition_idx]
            
                print(f"Experiment {experiment_id}: Processing partition {partition_idx} (partition step {partition_state['step_count']})")

                try:
                    # Set the small environment to the current partition's state
                    env_manager.small_env_timestep = partition_state['timestep']
                    env_manager.current_partition_idx = partition_idx

                    # Get the current observation
                    current_observation = env_manager.small_env_timestep.observation
                    map_obs_seq.append(current_observation)

                    # Extract partition info and create subsurface
                    partition_info = env_manager.partitions[partition_idx]
                    region_coords = partition_info['region_coords']
                    y_start, x_start, y_end, x_end = region_coords
                    width = x_end - x_start + 1
                    height = y_end - y_start + 1

                    if USE_RENDERING:
                        subsurface = extract_subsurface(screen, x_start, y_start, width, height, ORIGINAL_MAP_SIZE, global_env_config, partition_idx)
                        game_state_image_small = capture_screen(subsurface)
                    else:
                        game_state_image_small = None

                    state = env_manager.small_env_timestep.state
                    base_orientation = extract_base_orientation(state)
                    bucket_status = extract_bucket_status(state)

                    # LLM decision making 
                    if global_step % LLM_CALL_FREQUENCY == 0 and global_step > 0 and \
                        FORCE_DELEGATE_TO_RL is False and \
                        FORCE_DELEGATE_TO_LLM is False:

                        print(f"Experiment {experiment_id}: Calling LLM agent for decision...")
                        try:
                            obs_dict = {k: v.tolist() for k, v in current_observation.items()}
                            observation_str = json.dumps(obs_dict)

                        except AttributeError:
                            observation_str = str(current_observation)

                        if USE_IMAGE_PROMPT:
                            delegation_prompt = get_delegation_prompt(prompts, "See image", 
                                                                    context=f"Map {current_map_index}, Step {map_step}")
                        else:
                            delegation_prompt = get_delegation_prompt(prompts, observation_str, 
                                                                    context=f"Map {current_map_index}, Step {map_step}")

                        delegation_session_id = f"{SESSION_ID}_exp_{experiment_id}_map_{current_map_index}_delegation"
                        delegation_user_id = f"{USER_ID}_exp_{experiment_id}_delegation"

                        try:
                            if USE_IMAGE_PROMPT:
                                response = asyncio.run(call_agent_async_master(
                                    delegation_prompt, 
                                    game_state_image_small, 
                                    runner_delegation,               
                                    delegation_user_id,
                                    delegation_session_id,
                                    session_manager
                                ))
                            else:
                                response = asyncio.run(call_agent_async_master(
                                    delegation_prompt, 
                                    None, 
                                    runner_delegation,               
                                    delegation_user_id,
                                    delegation_session_id,
                                    session_manager
                                ))
                                
                            llm_response_text = response
                            print(f"Experiment {experiment_id}: LLM response: {llm_response_text}")
                                
                            if "delegate_to_rl" in llm_response_text.lower():
                                llm_decision = "delegate_to_rl"
                                print(f"Experiment {experiment_id}: Delegating to RL agent based on LLM response.")
                            elif "delegate_to_llm" in llm_response_text.lower():
                                llm_decision = "delegate_to_llm"
                                print(f"Experiment {experiment_id}: Delegating to LLM agent based on LLM response.")
                            else:
                                llm_decision = "STOP"                    

                        except Exception as adk_err:
                            print(f"Experiment {experiment_id}: Error during ADK agent communication: {adk_err}")
                            print(f"Experiment {experiment_id}: Defaulting to fallback action due to ADK error.")
                            llm_decision = "fallback"

                    if FORCE_DELEGATE_TO_LLM:
                        llm_decision = "delegate_to_llm"
                    elif FORCE_DELEGATE_TO_RL:
                        llm_decision = "delegate_to_rl"

                    # Action selection
                    if llm_decision == "delegate_to_rl":
                        print(f"Experiment {experiment_id}: Partition {partition_idx} - Delegating to RL agent")
                        try:
                            current_observation = env_manager.small_env_timestep.observation
                            batched_observation = add_batch_dimension_to_observation(current_observation)
                            obs = obs_to_model_input(batched_observation, partition_state['prev_actions_rl'], config)

                            current_model = partition_models[partition_idx]
                            _, logits_pi = current_model['model'].apply(current_model['params'], obs)
                            pi = tfp.distributions.Categorical(logits=logits_pi)

                            # Use experiment-specific random key
                            action_rng = jax.random.PRNGKey(seed + global_step * len(env_manager.partitions) + partition_idx + current_map_index * 10000 + experiment_id * 100000)
                            action_rng, action_key, step_key = jax.random.split(action_rng, 3)
                            action_rl = pi.sample(seed=action_key)
                        
                            partition_state['actions'].append(action_rl)
                            map_action_list.append(action_rl)

                        except Exception as rl_error:
                            print(f"Experiment {experiment_id}: ERROR getting action from RL model for partition {partition_idx}: {rl_error}")
                            action_rl = jnp.array(0)
                            partition_state['actions'].append(action_rl)
                            map_action_list.append(action_rl)

                    elif llm_decision == "delegate_to_llm":
                        print(f"Experiment {experiment_id}: Partition {partition_idx} - Delegating to LLM agent")
                        
                        start = env_manager.small_env_timestep.state.agent.agent_state.pos_base

                        msg = get_excavator_prompt(prompts, 
                          base_orientation['direction'], 
                          bucket_status, 
                          start)

                        llm_query.add_user_message(frame=game_state_image_small, user_msg=msg, local_map=None)
                        action_output, reasoning = llm_query.generate_response("./")
                        print(f"Experiment {experiment_id}: Action output: {action_output}, Reasoning: {reasoning}")
                        llm_query.add_assistant_message()

                        action_rl = jnp.array([action_output], dtype=jnp.int32)
                        map_action_list.append(action_rl)
                    
                    else:
                        print(f"Experiment {experiment_id}: Master Agent stop.")
                        action_rl = jnp.array([-1], dtype=jnp.int32)
                        map_action_list.append(action_rl)

                    # Clear LLM messages periodically
                    if len(llm_query.messages) > 3:
                        llm_query.delete_messages()

                    # Update action history and step environment
                    partition_state['prev_actions_rl'] = jnp.roll(partition_state['prev_actions_rl'], shift=1, axis=1)
                    partition_state['prev_actions_rl'] = partition_state['prev_actions_rl'].at[:, 0].set(action_rl)

                    wrapped_action = wrap_action2(action_rl, action_type)
                    new_timestep = env_manager.step_simple(partition_idx, wrapped_action, partition_states)
                    partition_states[partition_idx]['timestep'] = new_timestep
                    partition_state['step_count'] += 1
                
                    # Process reward
                    reward = new_timestep.reward
                    if isinstance(reward, jnp.ndarray):
                        if reward.shape == ():
                            reward_val = float(reward)
                        elif len(reward.shape) > 0:
                            reward_val = float(reward.flatten()[0])
                        else:
                            reward_val = float(reward)
                    else:
                        reward_val = float(reward)
                    
                    if not (jnp.isnan(reward_val) or jnp.isinf(reward_val)):
                        partition_state['rewards'].append(reward_val)
                        partition_state['total_reward'] += reward_val
                        map_reward_seq.append(reward_val)
                        current_step_reward += reward_val
                        print(f"Experiment {experiment_id}: Partition {partition_idx} - reward: {reward_val:.4f}, action: {action_rl}, done: {new_timestep.done}")
                    else:
                        print(f"Experiment {experiment_id}: Partition {partition_idx} - INVALID reward: {reward_val}, action: {action_rl}, done: {new_timestep.done}")

                    # Check completion conditions
                    partition_completed = False
                
                    if env_manager.is_small_task_completed():
                        print(f"Experiment {experiment_id}: Partition {partition_idx} COMPLETED after {partition_state['step_count']} steps!")
                        print(f"Experiment {experiment_id}: Total reward for partition {partition_idx}: {partition_state['total_reward']:.4f}")
                        env_manager.partitions[partition_idx]['status'] = 'completed'
                        partition_state['status'] = 'completed'
                        partition_completed = True
                
                    elif partition_state['step_count'] >= max_steps_per_map:
                        print(f"Experiment {experiment_id}: Partition {partition_idx} TIMED OUT")
                        env_manager.partitions[partition_idx]['status'] = 'failed'
                        partition_state['status'] = 'failed'
                        partition_completed = True
                
                    elif jnp.isnan(reward):
                        print(f"Experiment {experiment_id}: Partition {partition_idx} FAILED due to NaN reward")
                        env_manager.partitions[partition_idx]['status'] = 'failed'
                        partition_state['status'] = 'failed'
                        partition_completed = True
                
                    if partition_completed:
                        partitions_to_remove.append(partition_idx)

                except Exception as e:
                    print(f"Experiment {experiment_id}: ERROR stepping partition {partition_idx}: {e}")
                    if partition_idx < len(env_manager.partitions):
                        env_manager.partitions[partition_idx]['status'] = 'failed'
                    partition_state['status'] = 'failed'
                    partitions_to_remove.append(partition_idx)

            # Synchronize environments
            env_manager.complete_synchronization_with_full_agents(partition_states)

            # Remove completed/failed partitions
            for partition_idx in partitions_to_remove:
                if partition_idx in active_partitions:
                    active_partitions.remove(partition_idx)
                    print(f"Experiment {experiment_id}: Removed partition {partition_idx} from active list")

            print(f"Experiment {experiment_id}: Remaining active partitions: {active_partitions}")
            map_global_step_rewards.append(current_step_reward)
            print(f"Experiment {experiment_id}: Map {current_map_index} step {map_step} reward: {current_step_reward:.4f}")

            # Render (only if enabled)
            if USE_RENDERING:
                if GRID_RENDERING:
                    env_manager.render_all_partition_views_grid(partition_states)
                else:
                    env_manager.render_global_environment_with_multiple_agents(partition_states, VISUALIZE_PARTITIONS)

            # After processing all partitions, check if map is complete
            map_metrics = calculate_map_completion_metrics(partition_states, env_manager)
            map_done = map_metrics['done']
            
            # Update done flag for this step
            done = jnp.array(map_done)
            
            map_step += 1
            global_step += 1

            reward_seq.append(current_step_reward)

            if episode_done_once is None:
                episode_done_once = done
            if episode_length is None:
                episode_length = jnp.zeros_like(done, dtype=jnp.int32)
            if move_cumsum is None:
                move_cumsum = jnp.zeros_like(done, dtype=jnp.int32)
            if do_cumsum is None:
                do_cumsum = jnp.zeros_like(done, dtype=jnp.int32)

            episode_done_once = episode_done_once | done
            episode_length += ~episode_done_once

            move_cumsum_tmp = jnp.zeros_like(done, dtype=jnp.int32)
            for move_action in move_actions:
                move_mask = (action_rl == move_action) * (~episode_done_once)
                move_cumsum_tmp += move_tiles * tile_size * move_mask.astype(jnp.int32)
            for l_action in l_actions:
                l_mask = (action_rl == l_action) * (~episode_done_once)
                move_cumsum_tmp += 2 * move_tiles * tile_size * l_mask.astype(jnp.int32)
            move_cumsum += move_cumsum_tmp

            do_cumsum += (action_rl == do_action) * (~episode_done_once)

            dug_tiles_per_action_map = (env_manager.global_maps['action_map'] == -1).sum()
  
        # Add map data to global collections
        all_frames.extend(map_frames)
        all_reward_seq.extend(map_reward_seq)
        all_global_step_rewards.extend(map_global_step_rewards)
        all_obs_seq.extend(map_obs_seq)
        all_action_list.extend(map_action_list)
        
        # Move to next map
        current_map_index += 1
        
        # Check if we should continue
        if not playing or global_step >= num_timesteps:
            break
            
        print(f"Experiment {experiment_id}: Transitioning to map {current_map_index}...")

    # FINAL SUMMARY ACROSS ALL MAPS
    print(f"Experiment {experiment_id}: {'='*60}")
    print(f"Experiment {experiment_id}: COMPLETED - PROCESSED {current_map_index} MAPS")
    print(f"Experiment {experiment_id}: {'='*60}")

    # Save results (only save video if rendering is enabled)
    if USE_RENDERING:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_model_name = llm_model_name.replace('/', '_')
        output_dir = os.path.join("experiments", f"{safe_model_name}_{current_time}_exp_{experiment_id}")
        os.makedirs(output_dir, exist_ok=True)

        video_path = os.path.join(output_dir, "gameplay_all_maps.mp4")
        save_video(all_frames, video_path)
        print(f"Experiment {experiment_id}: Results saved to: {output_dir}")

    info = {
        "episode_done_once": episode_done_once,
        "episode_length": episode_length,
        "move_cumsum": move_cumsum,
        "do_cumsum": do_cumsum,
        "areas": areas,
        "dig_tiles_per_target_map_init": dig_tiles_per_target_map_init,
        "dug_tiles_per_action_map": dug_tiles_per_action_map,
    }

    return info


def main():
    """Main function to handle argument parsing and experiment execution"""
    parser = argparse.ArgumentParser(description="Run parallel LLM-based simulation experiments with RL agents.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        choices=["gpt-4o", 
                 "gpt-4.1", 
                 "o4-mini", 
                 "o3", 
                 "o3-mini", 
                 "gemini-1.5-flash-latest", 
                 "gemini-2.0-flash", 
                 "gemini-2.5-pro-exp-03-25", 
                 "gemini-2.5-pro-preview-03-25",
                 "gemini-2.5-pro-preview-05-06",
                 "gemini-2.5-flash-preview-04-17", 
                 "gemini-2.5-flash-preview-05-20",
                 "claude-3-haiku-20240307", 
                 "claude-3-7-sonnet-20250219",
                 "claude-opus-4-20250514",
                 "claude-sonnet-4-20250514",		
                 ], 
        help="Name of the LLM model to use."
    )
    parser.add_argument(
        "--model_key", 
        type=str, 
        required=True, 
        choices=["gpt", 
                 "gemini", 
                 "claude"], 
        help="Name of the LLM model key to use."
    )
    parser.add_argument(
        "--num_timesteps", 
        type=int, 
        default=100, 
        help="Number of timesteps to run."
    )
    parser.add_argument(
        "-n",
        "--n_envs",
        type=int,
        default=1,
        help="Number of environments",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        default="/home/gioelemo/Documents/terra/no-action-map.pkl",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--enable_rendering",
        action="store_true",
        help="Enable rendering and display (disabled by default for parallel execution)",
    )

    args = parser.parse_args()
    
    # Override rendering settings if specified
    global USE_RENDERING, USE_DISPLAY
    if args.enable_rendering:
        USE_RENDERING = True
        USE_DISPLAY = True
    
    NUM_ENVS = args.n_envs
    base_seed = args.seed

    mp.set_start_method("spawn", force=True)
    
    # Determine number of processes
    if args.num_processes is None:
        num_processes = min(NUM_ENVS, mp.cpu_count())
    else:
        num_processes = min(args.num_processes, NUM_ENVS, mp.cpu_count())
    
    print(f"Running {NUM_ENVS} experiments using {num_processes} parallel processes")
    print(f"Model: {args.model_name}")
    print(f"Timesteps: {args.num_timesteps}")
    print(f"Base seed: {base_seed}")
    print(f"Rendering enabled: {USE_RENDERING}")
    
    # Prepare arguments for each experiment
    experiment_args = []
    for i in range(NUM_ENVS):
        experiment_args.append((
            i,  # experiment_id
            args.model_name,
            args.model_key,
            args.num_timesteps,
            base_seed + i * 1000,  # Ensure different seeds
            args.run_name
        ))
    
    # Run experiments in parallel
    start_time = datetime.datetime.now()
    print(f"\nStarting parallel execution at {start_time}")
    
    try:
        if num_processes == 1:
            # Run sequentially if only one process
            print("Running experiments sequentially...")
            results = []
            for args_tuple in experiment_args:
                result = run_single_experiment(args_tuple)
                results.append(result)
        else:
            # Run in parallel
            print(f"Running experiments in parallel with {num_processes} processes...")
            with Pool(processes=num_processes) as pool:
                results = pool.map(run_single_experiment, experiment_args)
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Terminating processes...")
        if 'pool' in locals():
            pool.terminate()
            pool.join()
        return
    except Exception as e:
        print(f"Error during parallel execution: {e}")
        if 'pool' in locals():
            pool.terminate()
            pool.join()
        return
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nParallel execution completed at {end_time}")
    print(f"Total duration: {duration}")
    
    # Process results
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    print(f"\nExperiment Summary:")
    print(f"Total experiments: {NUM_ENVS}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed experiments:")
        for result in failed_results:
            print(f"  Experiment {result['experiment_id']}: {result.get('error', 'Unknown error')}")
    
    if successful_results:
        # Collect statistics from successful experiments
        episode_done_once_list = []
        episode_length_list = []
        move_cumsum_list = []
        do_cumsum_list = []
        areas_list = []
        dig_tiles_per_target_map_init_list = []
        dug_tiles_per_action_map_list = []
        
        for result in successful_results:
            episode_done_once_list.append(result['episode_done_once'])
            episode_length_list.append(result['episode_length'])
            move_cumsum_list.append(result['move_cumsum'])
            do_cumsum_list.append(result['do_cumsum'])
            areas_list.append(result['areas'])
            dig_tiles_per_target_map_init_list.append(result['dig_tiles_per_target_map_init'])
            dug_tiles_per_action_map_list.append(result['dug_tiles_per_action_map'])
        
        print(f"\nComputing statistics from {len(successful_results)} successful experiments...")
        
        if COMPUTE_BENCH_STATS:
            compute_stats_llm(
                episode_done_once_list, 
                episode_length_list, 
                move_cumsum_list,
                do_cumsum_list, 
                areas_list, 
                dig_tiles_per_target_map_init_list,
                dug_tiles_per_action_map_list
            )
        
        # Save aggregated results
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_model_name = args.model_name.replace('/', '_')
        results_dir = os.path.join("experiments", f"parallel_{safe_model_name}_{current_time}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(results_dir, "experiment_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'experiment_config': {
                    'model_name': args.model_name,
                    'model_key': args.model_key,
                    'num_timesteps': args.num_timesteps,
                    'num_envs': NUM_ENVS,
                    'base_seed': base_seed,
                    'num_processes': num_processes,
                    'duration_seconds': duration.total_seconds()
                },
                'results': results,
                'summary': {
                    'total_experiments': NUM_ENVS,
                    'successful': len(successful_results),
                    'failed': len(failed_results)
                }
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
    
    else:
        print("\nNo successful experiments to analyze.")
    
    print(f"\nAll experiments completed. Duration: {duration}")


if __name__ == "__main__":
    main()