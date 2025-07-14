"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

import numpy as np
import jax
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

from pygame.locals import (
    K_q,
    QUIT,
)

from llm.eval_llm import compute_stats_llm
from llm.env_manager_llm import EnvironmentsManager

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

def run_experiment(llm_model_name, llm_model_key, num_timesteps, seed, 
                run, current_experiment_number, env_manager, global_env_config,small_env_config=None):
    """
    Run an experiment with completely separate environments for global and small maps.
    """


    (FORCE_DELEGATE_TO_RL, FORCE_DELEGATE_TO_LLM, LLM_CALL_FREQUENCY,
     USE_MANUAL_PARTITIONING, MAX_NUM_PARTITIONS, VISUALIZE_PARTITIONS,
     USE_IMAGE_PROMPT , APP_NAME, USER_ID, SESSION_ID,
     GRID_RENDERING, ORIGINAL_MAP_SIZE, 
     USE_RENDERING, USE_DISPLAY,
    ENABLE_INTERVENTION, INTERVENTION_FREQUENCY, STUCK_WINDOW, MIN_REWARD, USE_RANDOM_PARTITIONING) = setup_experiment_config()

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
    max_maps = 10 # Set a reasonable limit for number of maps to process
    
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
    #obs = env_manager.global_maps
    areas = (obs["target_map"] == -1).sum(
        tuple([i for i in range(len(obs["target_map"].shape))][1:])
            ) * (tile_size**2)
    target_maps_init = obs["target_map"].copy()
    #target_maps_init = env_manager.global_maps['target_map'].copy()
    dig_tiles_per_target_map_init = (target_maps_init == -1).sum(
        tuple([i for i in range(len(target_maps_init.shape))][1:])
    )
    reward_seq = []
    episode_done_once = None
    episode_length = None
    move_cumsum = None
    do_cumsum = None
    sub_task_seed = current_experiment_number
    
    screen = pg.display.get_surface()

    # MAIN LOOP - PROCESS MULTIPLE MAPS
    while playing and global_step < num_timesteps and current_map_index < max_maps:
        print(f"\n{'='*80}")
        print(f"STARTING MAP {current_map_index}")
        print(f"{'='*80}")
        
        # Reset to next map (reusing the same environment)
        try:
            #if current_map_index > 0:  # Don't reset on first map since it's already initialized
            reset_to_next_map(current_map_index, seed, env_manager, global_env_config,
                       initial_custom_pos, initial_custom_angle)

            env_manager.global_env.terra_env.render_obs_pygame(
                env_manager.global_env.timestep.observation, 
                env_manager.global_env.timestep.info
            )
            screen = pg.display.get_surface()
            game_state_image = capture_screen(screen)
            
            llm_query, runner_delegation, session_manager, prompts = setup_partitions_and_llm(
                        current_map_index, ORIGINAL_MAP_SIZE, env_manager, 
                        config, llm_model_name, llm_model_key,
                        APP_NAME, USER_ID, SESSION_ID, screen,
                        USE_MANUAL_PARTITIONING, USE_IMAGE_PROMPT, MAX_NUM_PARTITIONS, USE_RANDOM_PARTITIONING, sub_task_seed)
            partition_states, partition_models, active_partitions = initialize_partitions_for_current_map(env_manager, config, model_params)

            env_manager.initialize_partition_specific_target_maps(partition_states)


            if partition_states is None:
                print(f"Failed to initialize map {current_map_index}, moving to next map")
                current_map_index += 1
                continue
                
        except Exception as e:
            print(f"Error setting up map {current_map_index}: {e}")
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
        map_done = False  # Track map completion


        while playing and active_partitions and map_step < max_steps_per_map and global_step < num_timesteps:
            # Handle quit events
            if USE_RENDERING:
                for event in pg.event.get():
                    if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                        playing = False

            print(f"\nMap {current_map_index}, Step {map_step} (Global {global_step}) - "
                  f"Processing {len(active_partitions)} active partitions")
            
            if USE_RENDERING:
                #Capture screen state
                screen = pg.display.get_surface()
                game_state_image = capture_screen(screen)

            else:
                screen = None
                game_state_image = None
            map_frames.append(game_state_image)

            # Step all active partitions simultaneously
            partitions_to_remove = []
            current_step_reward = 0.0

            for partition_idx in active_partitions:
                partition_state = partition_states[partition_idx]
                print(f"  Processing partition {partition_idx} (partition step {partition_state['step_count']})")

                try:
                    # Set the small environment to the current partition's state
                    env_manager.small_env_timestep = partition_state['timestep']
                    env_manager.current_partition_idx = partition_idx

                    current_observation = partition_state['timestep'].observation

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

                        print("    Calling LLM agent for decision...")

                        # Check if intervention is enabled and needed
                        needs_intervention = False
                        stuck_info = {'is_stuck': False, 'reason': 'not_checked'}

                        if ENABLE_INTERVENTION:
                            stuck_info = detect_stuck_excavator(
                                partition_state,
                                threshold_steps=STUCK_WINDOW,
                                min_reward_threshold=MIN_REWARD
                            )
                            needs_intervention = should_intervene(
                                partition_state, 
                                active_partitions,
                                intervention_frequency=INTERVENTION_FREQUENCY
                            )
    
                            if needs_intervention:
                                print(f"    🚨 Partition {partition_idx} appears stuck: {stuck_info['reason']}")
                                print(f"    Details: {stuck_info['details']}")

                        try:
                            obs_dict = {k: v.tolist() for k, v in current_observation.items()}
                            observation_str = json.dumps(obs_dict)

                        except AttributeError:
                            # Handle the case where current_observation is not a dictionary
                            observation_str = str(current_observation)

                        # Enhanced context with stuck information
                        base_context = f"Map {current_map_index}, Step {map_step}"
                        if needs_intervention and ENABLE_INTERVENTION:  # ← USED HERE
                            stuck_context = f" | STUCK: {stuck_info['reason']} - {stuck_info['details']}"
                            context = base_context + stuck_context
                        else:
                            context = base_context

                        if USE_IMAGE_PROMPT:
                            delegation_prompt = get_delegation_prompt(
                                prompts, 
                                "See image", 
                                context=context
                            )
                        else:
                            delegation_prompt = get_delegation_prompt(
                                prompts, 
                                observation_str, 
                                context=context
                            )
                        delegation_session_id = f"{SESSION_ID}_map_{current_map_index}_delegation"  # This creates "session_001_map_0_delegation"
                        delegation_user_id = f"{USER_ID}_delegation"  # This creates "user_1_delegation"

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
                            print(f"LLM response: {llm_response_text}")
                                
                            if "delegate_to_rl" in llm_response_text.lower():
                                llm_decision = "delegate_to_rl"
                                print("Delegating to RL agent based on LLM response.")
                            elif "delegate_to_llm" in llm_response_text.lower():
                                llm_decision = "delegate_to_llm"
                                print("Delegating to LLM agent based on LLM response.")
                            elif "intervention" in llm_response_text.lower():
                                llm_decision = "intervention"
                                print("INTERVENTION mode activated based on LLM response.")
                            else:
                                # Default fallback
                                if needs_intervention:
                                    llm_decision = "intervention"
                                    print("INTERVENTION mode activated due to detected stuck condition.")
                                else:
                                    llm_decision = "delegate_to_rl"           

                        except Exception as adk_err:
                            print(f"Error during ADK agent communication: {adk_err}")
                            if needs_intervention:
                                llm_decision = "intervention"
                                print("INTERVENTION mode activated due to communication error and stuck condition.")
                            else:
                                llm_decision = "delegate_to_rl"

                    if FORCE_DELEGATE_TO_LLM:
                        llm_decision = "delegate_to_llm"
                    elif FORCE_DELEGATE_TO_RL:
                        llm_decision = "delegate_to_rl"

                    # Action selection
                    if llm_decision == "delegate_to_rl":
                        print(f"    Partition {partition_idx} - Delegating to RL agent")
                        try:
                            current_observation = partition_state['timestep'].observation

                            batched_observation = add_batch_dimension_to_observation(current_observation)
                            obs = obs_to_model_input(batched_observation, partition_state['prev_actions_rl'], config)

                            current_model = partition_models[partition_idx]
                            _, logits_pi = current_model['model'].apply(current_model['params'], obs)
                            pi = tfp.distributions.Categorical(logits=logits_pi)

                            # Use map-specific random key
                            action_rng = jax.random.PRNGKey(seed + global_step * len(env_manager.partitions) + partition_idx + current_map_index * 10000)
                            action_rng, action_key, step_key = jax.random.split(action_rng, 3)
                            action_rl = pi.sample(seed=action_key)
                        
                            partition_state['actions'].append(action_rl)
                            map_action_list.append(action_rl)

                        except Exception as rl_error:
                            print(f"    ERROR getting action from RL model for partition {partition_idx}: {rl_error}")
                            action_rl = jnp.array(0)
                            partition_state['actions'].append(action_rl)
                            map_action_list.append(action_rl)

                    elif llm_decision == "delegate_to_llm":
                        print(f"    Partition {partition_idx} - Delegating to LLM agent")
                        
                        start = env_manager.small_env_timestep.state.agent.agent_state.pos_base

                        msg = get_excavator_prompt(prompts, 
                          base_orientation['direction'], 
                          bucket_status, 
                          start)

                        llm_query.add_user_message(frame=game_state_image_small, user_msg=msg, local_map=None)
                        action_output, reasoning = llm_query.generate_response("./")
                        print(f"\n    Action output: {action_output}, Reasoning: {reasoning}")
                        llm_query.add_assistant_message()

                        action_rl = jnp.array([action_output], dtype=jnp.int32)
                        map_action_list.append(action_rl)
                    
                    elif llm_decision == "intervention" and ENABLE_INTERVENTION:
                        print(f"    Partition {partition_idx} - INTERVENTION MODE")
                        total_interventions += 1
                        if partition_idx not in partition_interventions:
                            partition_interventions[partition_idx] = 0
                        partition_interventions[partition_idx] += 1

                        try:
                            stuck_info = detect_stuck_excavator(
                                partition_state,
                                threshold_steps=STUCK_WINDOW,
                                min_reward_threshold=MIN_REWARD
                            )
                            action_rl = get_intervention_action(partition_state, stuck_info, action_type)
                            partition_state['actions'].append(action_rl)
                            map_action_list.append(action_rl)
                            
                            # Log intervention details
                            print(f"    🔧 Intervention #{total_interventions} for partition {partition_idx}")
                            print(f"    🔧 Reason: {stuck_info['reason']} | Action: {action_rl}")
                        except Exception as intervention_error:
                            print(f"    ERROR during intervention for partition {partition_idx}: {intervention_error}")
                            # Fallback to a safe action
                            action_rl = jnp.array([0], dtype=jnp.int32)  # Forward movement
                            partition_state['actions'].append(action_rl)
                            map_action_list.append(action_rl)
                    
                    else:
                        print("    Master Agent stop.")
                        action_rl = jnp.array([-1], dtype=jnp.int32)
                        map_action_list.append(action_rl)

                    # Clear LLM messages periodically
                    if len(llm_query.messages) > 3:
                        llm_query.delete_messages()

                    # Update action history and step environment
                    partition_state['prev_actions_rl'] = jnp.roll(partition_state['prev_actions_rl'], shift=1, axis=1)
                    partition_state['prev_actions_rl'] = partition_state['prev_actions_rl'].at[:, 0].set(action_rl)

                    wrapped_action = wrap_action_llm(action_rl, action_type)

                    # Take step with full sync
                    new_timestep = env_manager.step_with_full_global_sync(partition_idx, wrapped_action, partition_states)

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
                        print(f"    Partition {partition_idx} - reward: {reward_val:.4f}, action: {action_rl}, done: {new_timestep.done}")
                    else:
                        print(f"    Partition {partition_idx} - INVALID reward: {reward_val}, action: {action_rl}, done: {new_timestep.done}")
                    
                    # Check completion conditions
                    partition_completed = False
                
                    if env_manager.is_small_task_completed():
                        print(f"    Partition {partition_idx} COMPLETED after {partition_state['step_count']} steps!")
                        print(f"    Total reward for partition {partition_idx}: {partition_state['total_reward']:.4f}")
                        env_manager.partitions[partition_idx]['status'] = 'completed'
                        partition_state['status'] = 'completed'
                        partition_completed = True
                
                    elif partition_state['step_count'] >= max_steps_per_map:
                        print(f"    Partition {partition_idx} TIMED OUT")
                        env_manager.partitions[partition_idx]['status'] = 'failed'
                        partition_state['status'] = 'failed'
                        partition_completed = True
                
                    elif jnp.isnan(reward):
                        print(f"    Partition {partition_idx} FAILED due to NaN reward")
                        env_manager.partitions[partition_idx]['status'] = 'failed'
                        partition_state['status'] = 'failed'
                        partition_completed = True
                
                    if partition_completed:
                        partitions_to_remove.append(partition_idx)

                except Exception as e:
                    print(f"    ERROR stepping partition {partition_idx}: {e}")
                    if partition_idx < len(env_manager.partitions):
                        env_manager.partitions[partition_idx]['status'] = 'failed'
                    partition_state['status'] = 'failed'
                    partitions_to_remove.append(partition_idx)


            # Remove completed/failed partitions
            for partition_idx in partitions_to_remove:
                if partition_idx in active_partitions:
                    active_partitions.remove(partition_idx)
                    print(f"    Removed partition {partition_idx} from active list")

            print(f"    Remaining active partitions: {active_partitions}")
            map_global_step_rewards.append(current_step_reward)
            print(f"    Map {current_map_index} step {map_step} reward: {current_step_reward:.4f}")

            # Render
            if GRID_RENDERING:
                env_manager.render_all_partition_views_grid(partition_states)
            else:
                env_manager.render_global_environment_with_multiple_agents(partition_states, VISUALIZE_PARTITIONS)
            
            # After processing all partitions, check if map is complete
            map_metrics = calculate_map_completion_metrics(partition_states)
            map_done = map_metrics['done']
                        
            # Update done flag for this step
            done = jnp.array(map_done)  # Convert to JAX array for consistency
            
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
            
        print(f"\nTransitioning to map {current_map_index}...")

    # FINAL SUMMARY ACROSS ALL MAPS
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED - PROCESSED {current_map_index} MAPS")
    print(f"{'='*80}")

    # Save results
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model_name = llm_model_name.replace('/', '_')
    output_dir = os.path.join("experiments", f"{safe_model_name}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    # Save video
    if USE_RENDERING:
        video_path = os.path.join(output_dir, "gameplay_all_maps.mp4")
        save_video(all_frames, video_path)
        
        print(f"\nResults saved to: {output_dir}")

    info = {
        "episode_done_once": episode_done_once,
        "episode_length": episode_length,
        "move_cumsum": move_cumsum,
        "do_cumsum": do_cumsum,
        "areas": areas,
        "dig_tiles_per_target_map_init": dig_tiles_per_target_map_init,
        "dug_tiles_per_action_map": dug_tiles_per_action_map,
    }

        # Print intervention statistics
    if ENABLE_INTERVENTION and total_interventions > 0:  # ← USED HERE
        print(f"\n🔧 INTERVENTION STATISTICS:")
        print(f"   Total interventions: {total_interventions}")
        print(f"   Interventions per partition: {partition_interventions}")
        print(f"   Intervention rate: {total_interventions/global_step:.1%}")
        
        # Save intervention stats
        info["total_interventions"] = total_interventions
        info["partition_interventions"] = partition_interventions

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an LLM-based simulation experiment with RL agents.")
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
        default="/home/gioelemo/Documents/terra/tracked-dense.pkl",
        help="Policy to use for the experiment. Must be a valid path to a .pkl file containing the policy.",
    )


    args = parser.parse_args()
    NUM_ENVS = args.n_envs

    episode_done_once_list = []
    episode_length_list = []
    move_cumsum_list = []
    do_cumsum_list = []
    areas_list = []
    dig_tiles_per_target_map_init_list = []
    dug_tiles_per_action_map_list = []

    base_seed = args.seed

    (_, _, _, _, _, _,
     _ , _, _, _,
     _, _, 
     USE_RENDERING, USE_DISPLAY,
    _, _, _, _, _) = setup_experiment_config()



    # Track intervention statistics
    total_interventions = 0
    partition_interventions = {}

    agent_checkpoint_path = args.run_name
    model_params = None
    config = None

    print(f"Loading RL agent configuration from: {agent_checkpoint_path}")
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
    print("Initializing environment manager with all maps...")
    env_manager = EnvironmentsManager(
        seed=base_seed,
        global_env_config=global_env_config,
        small_env_config=None,
        shuffle_maps=False,
        rendering=USE_RENDERING,
        display=USE_DISPLAY
    )
    print("Environment manager initialized.")

    for i in range(NUM_ENVS):
        print(f"Running experiment {i+1}/{NUM_ENVS} with args: {args}")
        global_env_config = jax.tree_map(
            lambda x: x[0][None, ...].repeat(1, 0), log["env_config"]
        ) 
        info = run_experiment(
                    args.model_name, 
                    args.model_key, 
                    args.num_timesteps, 
                    #base_seed + i * 1000,  # Ensure different seeds
                    base_seed,
                    args.run_name,
                    i+1,
                    env_manager, global_env_config
                )
        # Collect results from this experiment
        episode_done_once = info["episode_done_once"]
        episode_length = info["episode_length"]
        move_cumsum = info["move_cumsum"]
        do_cumsum = info["do_cumsum"]
        areas = info["areas"]
        dig_tiles_per_target_map_init = info["dig_tiles_per_target_map_init"]
        dug_tiles_per_action_map = info["dug_tiles_per_action_map"]
        
        episode_done_once_list.append(episode_done_once.item())
        episode_length_list.append(episode_length.item())
        move_cumsum_list.append(move_cumsum.item())
        do_cumsum_list.append(do_cumsum.item())
        areas_list.append(areas.item())
        dig_tiles_per_target_map_init_list.append(dig_tiles_per_target_map_init.item())
        dug_tiles_per_action_map_list.append(dug_tiles_per_action_map.item())

    print("\nExperiment completed for all environments.")
    COMPUTE_BENCH_STATS = True  # Set to True to compute and print stats}
    if COMPUTE_BENCH_STATS:
        compute_stats_llm(episode_done_once_list, episode_length_list, move_cumsum_list,
                      do_cumsum_list, areas_list, dig_tiles_per_target_map_init_list,
                      dug_tiles_per_action_map_list)

