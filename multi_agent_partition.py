"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

import numpy as np
import jax
from tqdm import tqdm
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action
from terra.state import State
import matplotlib.animation as animation

# from utils.curriculum import Curriculum
from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints
from terra.config import EnvConfig
from terra.config import BatchConfig

from terra.viz.llms_utils import *
from multi_agent_utils import *
from terra.viz.llms_adk import *
from terra.viz.a_star import compute_path, simplify_path
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

import asyncio
import os
import argparse
import datetime
import time
import json
import csv
import pygame as pg

from pygame.locals import (
    K_q,
    QUIT,
)

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

FORCE_DELEGATE_TO_RL = True     # Force delegation to RL agent for testing
FORCE_DELEGATE_TO_LLM = False   # Force delegation to LLM agent for testing
LLM_CALL_FREQUENCY = 15         # Number of steps between LLM calls
USE_MANUAL_PARTITIONING = True  # Use manual partitioning for LLM (Master Agent)
NUM_PARTITIONS = 4              # Number of partitions for LLM (Master Agent)
USE_IMAGE_PROMPT = True         # Use image prompt for LLM (Master Agent)
USE_LOCAL_MAP = True            # Use local map for LLM (Excavator Agent)
USE_PATH = True                 # Use path for LLM (Excavator Agent)
APP_NAME = "ExcavatorGameApp"   # Application name for ADK
USER_ID = "user_1"              # User ID for ADK
SESSION_ID = "session_001"      # Session ID for ADK

def run_experiment(llm_model_name, llm_model_key, num_timesteps, n_envs_x, n_envs_y, seed, progressive_gif, run):
    """
    Run an LLM-based simulation experiment.

    Args:
        model_name: The name of the LLM model to use.
        model_key: The name of the LLM model key to use.
        num_timesteps: The number of timesteps to run.
        n_envs_x: The number of environments along the x-axis.
        n_envs_y: The number of environments along the y-axis.
        seed: The random seed for reproducibility.
        progressive_gif: Whether to generate a progressive GIF (1 for True, 0 for False).
        run: The path to the RL agent checkpoint file.

    Returns:
        None
    """

    agent_checkpoint_path = run
    model = None
    model_params = None
    config = None


    print(f"Loading RL agent from: {agent_checkpoint_path}")
    log = load_pkl_object(agent_checkpoint_path)
    config = log["train_config"]

    original_env_cfgs_full_map = jax.tree_map(
        lambda x: x[0][None, ...].repeat(1, 0), log["env_config"]
    ) 

    n_envs = 1
    config.num_test_rollouts = n_envs
    config.num_devices = 1

    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type

    suffle_maps = False
    print(f"Using progressive_gif = {progressive_gif}, shuffle_maps = {suffle_maps}")

    env = TerraEnvBatchWithMapOverride(
        rendering=True,
        n_envs_x_rendering=n_envs_x,
        n_envs_y_rendering=n_envs_y,
        display=True,
        progressive_gif=progressive_gif,
        shuffle_maps=suffle_maps,
    )
    config.num_embeddings_agent_min = 60  # curriculum.get_num_embeddings_agent_min()

    model = load_neural_network(config, env)
    model_params = log["model"]

    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset_initial = jax.random.split(_rng, 1)


    initial_custom_pos = None
    initial_custom_angle = None
    timestep = env.reset(original_env_cfgs_full_map, rng_reset_initial, initial_custom_pos, initial_custom_angle)

    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))
    
    # Trigger the JIT compilation
    timestep = env.step(timestep, repeat_action(action_type.do_nothing()), rng_reset_initial)
    env.terra_env.render_obs_pygame(timestep.observation, timestep.info)

    global_target_map_data = timestep.state.world.target_map.map[0].copy() 
    global_action_map_data = timestep.state.world.action_map.map[0].copy() 
    global_dumpability_mask_data = timestep.state.world.dumpability_mask.map[0].copy()
    global_dumpability_mask_init_data = timestep.state.world.dumpability_mask_init.map[0].copy()
    global_padding_mask_data = timestep.state.world.padding_mask.map[0].copy()
    global_traversability_mask_data = timestep.state.world.traversability_mask.map[0].copy()


    prev_actions_rl = jnp.zeros((1,config.num_prev_actions), dtype=jnp.int32)

    

    step = 0
    playing = True

    if USE_MANUAL_PARTITIONING:
        if NUM_PARTITIONS == 1:
            sub_tasks = [
                {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
            ]
        elif NUM_PARTITIONS == 2:
            sub_tasks = [
                {'id': 0, 'region_coords': (0, 0, 31, 63), 'start_pos': (16, 32), 'start_angle': 0, 'status': 'pending'},
                {'id': 1, 'region_coords': (32, 0, 63, 63), 'start_pos': (48, 32), 'start_angle': 0, 'status': 'pending'}
            ]
        elif NUM_PARTITIONS == 4:
            sub_tasks = [
                {'id': 0, 'region_coords': (0, 0, 31, 31), 'start_pos': (16, 16), 'start_angle': 0, 'status': 'pending'},
                {'id': 1, 'region_coords': (0, 32, 31, 63), 'start_pos': (16, 48), 'start_angle': 0, 'status': 'pending'},
                {'id': 2, 'region_coords': (32, 0, 63, 31), 'start_pos': (48, 16), 'start_angle': 0, 'status': 'pending'},
                {'id': 3, 'region_coords': (32, 32, 63, 63), 'start_pos': (48, 48), 'start_angle': 0, 'status': 'pending'}
            ]
        else:
            raise ValueError("Invalid number of partitions. Must be 1, 2 or 4.")

    current_sub_task_idx = -1

    screen = pg.display.get_surface()
    frames = []
    t_counter = 0
    reward_seq = []
    obs_seq = []
    action_list = []


    rng = jax.random.PRNGKey(seed)

    print("Starting the game loop with map partitioning...")

    while playing and step < num_timesteps:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                playing = False
        print(f"\n--- Step {step} ---")
        current_observation = timestep.observation
        obs_seq.append(current_observation)


        #current_sub_task_idx = step
        game_state_image = capture_screen(screen)
        frames.append(game_state_image)

        #if current_sub_task_idx == -1 or (sub_tasks[current_sub_task_idx]['status'] == 'completed') or step%100==0:
        if current_sub_task_idx == -1 or (sub_tasks[current_sub_task_idx]['status'] == 'completed'):

        #if current_sub_task_idx is not None:
            current_sub_task_idx += 1

            if current_sub_task_idx >= len(sub_tasks):
                print("All sub-tasks completed.")
                playing = False
                break
            
            active_task = sub_tasks[current_sub_task_idx]
            print(f"\n--- Starting Sub-Task {active_task['id']} ---")
            print(f"Region: {active_task['region_coords']}, Start Pos: {active_task['start_pos']}")

            sub_task_target_map_data = create_sub_task_target_map(
                global_target_map_data, 
                active_task['region_coords']
            )
            print(f"Sub-task target map shape: {sub_task_target_map_data.shape}")

            sub_task_action_map_data = create_sub_task_action_map(
                global_action_map_data, 
                active_task['region_coords']
            )
            print(f"Sub-task action map shape: {sub_task_action_map_data.shape}")

            sub_task_dumpability_mask_data = create_sub_task_dumpability_mask(
                global_dumpability_mask_data, 
                active_task['region_coords']
            )
            print(f"Sub-task dumbability mask shape: {sub_task_dumpability_mask_data.shape}")

            sub_task_dumpability_init_mask_data = create_sub_task_dumpability_mask(
                global_dumpability_mask_init_data, 
                active_task['region_coords']
            )
            print(f"Sub-task dumbability init mask shape: {sub_task_dumpability_init_mask_data.shape}")

            sub_task_padding_mask_data = create_sub_task_padding_mask(
                global_padding_mask_data, 
                active_task['region_coords']
            )
            print(f"Sub-task padding mask shape: {sub_task_padding_mask_data.shape}")

            sub_task_traversability_mask_data = create_sub_task_traversability_mask(
                global_traversability_mask_data, 
                active_task['region_coords']
            )
            print(f"Sub-task traversability mask shape: {sub_task_traversability_mask_data.shape}")

            rng, reset_key = jax.random.split(rng)

            
            # Convert start position and angle to the appropriate format
            custom_pos = active_task['start_pos']
            custom_angle = active_task['start_angle']

            timestep = env.reset_with_map_override(
                original_env_cfgs_full_map, 
                jax.random.split(reset_key, 1),
                custom_pos=custom_pos,
                custom_angle=custom_angle,
                target_map_override=sub_task_target_map_data,
                traversability_mask_override=sub_task_traversability_mask_data,
                padding_mask_override=sub_task_padding_mask_data,
                dumpability_mask_override=sub_task_dumpability_mask_data,
                dumpability_mask_init_override=sub_task_dumpability_init_mask_data,
                action_map_override=sub_task_action_map_data
            )

            verify_maps_override(timestep, sub_task_target_map_data, sub_task_traversability_mask_data, 
                         sub_task_padding_mask_data, sub_task_dumpability_mask_data,
                         sub_task_dumpability_init_mask_data, sub_task_action_map_data)
            
            active_task['status'] = 'active'
            # Reset metrics for the new sub-task
            prev_actions_rl = jnp.zeros((1, config.num_prev_actions), dtype=jnp.int32)


        rng, action_key, step_key = jax.random.split(rng, 3)


        active_task = sub_tasks[current_sub_task_idx]
        current_observation = timestep.observation # This obs is from the sub-task's 64x64 map
        print("CURRENT OBSERVATION OKAY", current_observation)

        obs = obs_to_model_input(current_observation, prev_actions_rl, config)
        _, logits_pi = model.apply(model_params, obs)
        pi = tfp.distributions.Categorical(logits=logits_pi)
        action_rl = pi.sample(seed=action_key)
        action_list.append(action_rl)
        print(f"Action RL: {action_rl}")

        prev_actions_rl = jnp.roll(prev_actions_rl, shift=1, axis=1)
        prev_actions_rl = prev_actions_rl.at[:, 0].set(action_rl)
        print(f"Previous actions RL: {prev_actions_rl}")

        print(wrap_action(action_rl, action_type))
        print(            jax.random.split(step_key, 1)
)
        timestep = env.step(
            timestep, 
            wrap_action(action_rl, action_type), 
            jax.random.split(step_key, 1)
        )

        reward_seq.append(timestep.reward)
        print(t_counter, timestep.reward, action_rl, timestep.done)
        print(10 * "=")
        t_counter += 1

        if timestep.info.get("task_done", jnp.array([False]))[0]:
            print(f"--- Sub-Task {active_task['id']} COMPLETED by RL agent! ---")
            active_task['status'] = 'completed'

            # Update the global dumpability mask with the changes from this sub-task
            current_region = active_task['region_coords']
            y_start, x_start, y_end, x_end = current_region
            region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    
            #Get the current dumpability mask from the environment state
            current_dumpability_mask = timestep.state.world.dumpability_mask.map[0]
    
            # Update only the relevant region in the global mask
            global_dumpability_mask_data = global_dumpability_mask_data.at[region_slice].set(
                current_dumpability_mask[region_slice]
            )
    
            # Also update other relevant global maps if needed
            global_target_map_data = global_target_map_data.at[region_slice].set(
                timestep.state.world.target_map.map[0][region_slice]
            )
            global_action_map_data = global_action_map_data.at[region_slice].set(
                timestep.state.world.action_map.map[0][region_slice]
            )
            global_traversability_mask_data = global_traversability_mask_data.at[region_slice].set(
                timestep.state.world.traversability_mask.map[0][region_slice]
            )
            global_padding_mask_data = global_padding_mask_data.at[region_slice].set(
                timestep.state.world.padding_mask.map[0][region_slice]
            )
        env.terra_env.render_obs_pygame(timestep.observation, timestep.info)
        step += 1

    print("Game loop ended.")
    print(reward_seq)
    numeric_reward_seq = [r[0] if hasattr(r, '__getitem__') and len(r) > 0 else r for r in reward_seq]
    cumulative_rewards = np.cumsum(numeric_reward_seq)

    # Generate a timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a unique output directory for the model and timestamp
    # Use a safe version of the model name for the directory
    safe_model_name = llm_model_name.replace('/', '_') # Replace slashes if any
    output_dir = os.path.join("experiments", f"{safe_model_name}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    # Save actions and cumulative rewards to a CSV file
    csv_path = os.path.join(output_dir, "actions_rewards.csv")
    save_csv(csv_path, action_list, cumulative_rewards)    
    # Save the gameplay video
    video_path = os.path.join(output_dir, "gameplay.mp4")
    save_video(frames, video_path)




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
                 "gemini-2.5-flash-preview-04-17", 
                 "claude-3-haiku-20240307", 
                 "claude-3-7-sonnet-20250219"], 
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
        "-nx",
        "--n_envs_x",
        type=int,
        default=1,
        help="Number of environments on x.",
    )
    parser.add_argument(
        "-ny",
        "--n_envs_y",
        type=int,
        default=1,
        help="Number of environments on y.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-pg",
        "--progressive_gif",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        # default="/home/gioelemo/Documents/terra/new-maps-different-order.pkl",
        # help="new-maps-different-order.pkl (12 cabin and 12 base rotations)",
        # default="/home/gioelemo/Documents/terra/gioele.pkl",
        # help="gioele.pkl (8 cabin and 4 base rotations)",
        # default="/home/gioelemo/Documents/terra/gioele_new.pkl",
        # help="gioele_new.pkl (8 cabin and 4 base rotations) Version 7 May",
        default="/home/gioelemo/Documents/terra/new-penalties.pkl",
        help="new-penalties.pkl (12 cabin and 12 base rotations) Version 7 May",
    )

    args = parser.parse_args()
    run_experiment(args.model_name, 
                   args.model_key, 
                   args.num_timesteps, 
                   args.n_envs_x, 
                   args.n_envs_y, 
                   args.seed, 
                   args.progressive_gif, 
                   args.run_name
                   )
