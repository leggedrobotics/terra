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
from multi_agent_map import *
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

FORCE_DELEGATE_TO_RL = False    # Force delegation to RL agent for testing
FORCE_DELEGATE_TO_LLM = False   # Force delegation to LLM agent for testing
LLM_CALL_FREQUENCY = 15          # Number of steps between LLM calls
USE_IMAGE_PROMPT = True         # Use image prompt for LLM (Master Agent)
USE_LOCAL_MAP = True            # Use local map for LLM (Excavator Agent)
USE_PATH = True                 # Use path for LLM (Excavator Agent)
APP_NAME = "ExcavatorGameApp"   # Application name for ADK
USER_ID = "user_1"              # User ID for ADK
SESSION_ID = "session_001"      # Session ID for ADK

def display_map_pygame(map_data: jnp.ndarray, surface: pg.Surface, target_color=(255, 0, 0), non_target_color=(255, 255, 255)):
    """
    Draws the map data onto a Pygame surface.

    Args:
        map_data: A 2D JAX array representing the map (-1 for target, 0 for non-target).
        surface: The Pygame surface to draw on. Assumes surface size is a multiple
                 of map_data shape for scaling.
        target_color: RGB color for targets (-1). Default is Red.
        non_target_color: RGB color for non-targets (0). Default is White.
    """
    # Ensure map_data is a numpy array for easier processing
    map_np = np.asarray(map_data)

    height, width = map_np.shape
    surf_width, surf_height = surface.get_size()

    # Calculate scaling factor
    scale_x = surf_width // width
    scale_y = surf_height // height

    # Clear the surface with the non-target color
    surface.fill(non_target_color)

    # Iterate through map data and draw pixels/rects
    for y in range(height):
        for x in range(width):
            # Check if the cell is a target (-1)
            if map_np[y, x] == -1:
                # Draw a rectangle for this target cell, scaled to the surface size
                pg.draw.rect(surface, target_color, (x * scale_x, y * scale_y, scale_x, scale_y))


def create_sub_task_target_map(global_target_map_data: jnp.ndarray,
                               region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 target map for an RL agent's sub-task.

    Args:
        global_target_map_data: The original full 64x64 JAX array of the target map.
        region_coords: A tuple (y_start, x_start, y_end, x_end) defining the
                       sub-region in the global map. Coordinates are inclusive.

    Returns:
        A new 64x64 JAX array where only the specified sub-region contains
        targets from the global_target_map_data, and the rest is non-target (0).
    """
    y_start, x_start, y_end, x_end = region_coords

    # Initialize a new 64x64 map filled with a non-target value (e.g., 0 for traversable ground).
    # This uses global_target_map_data to get the correct shape and dtype.
    sub_task_map = jnp.full_like(global_target_map_data, 0)

    # Extract the relevant part from the global map that corresponds to the sub-region.
    # These are the values (targets, terrain) we want to place in our new map.
    source_region_data = global_target_map_data[y_start:y_end+1, x_start:x_end+1]

    # Create a mask for where the target values (-1) are within this extracted source region.
    target_mask_in_source_region = (source_region_data == -1)

    # Get the actual target values (-1 where the mask is true, 0 otherwise)
    # This ensures we only copy -1 values, not other terrain features if we want pure 0 outside targets.
    targets_to_place = jnp.where(target_mask_in_source_region, -1, 0)
    
    # Place these targets (which are -1s, or 0s if not a target in source)
    # into the corresponding location in the sub_task_map.
    # If you wanted to copy ALL terrain features from the source_region_data into the
    # sub_task_map (not just targets), you would use:
    # sub_task_map = sub_task_map.at[y_start:y_end+1, x_start:x_end+1].set(source_region_data)
    #
    # However, the goal is: "only cells that are -1 in global_target_map AND inside region_coords are -1. All else is 0."
    # So, the sub_task_map is initialized to 0. We only need to set the -1s.

    # More direct approach based on "Final logic":
    # Start with a map of all zeros (non-targets).
    final_sub_task_map = jnp.zeros_like(global_target_map_data)

    # Iterate over the defined sub-region and copy target values (-1) only.
    # This JAX-idiomatic way avoids explicit loops for performance:
    # 1. Create a slice object for the region
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    
    # 2. Get the data from the global map for this region
    global_data_in_region = global_target_map_data[region_slice]
    
    # 3. Identify where the targets are in this specific region of the global map
    targets_in_global_region = (global_data_in_region == -1)
    
    # 4. Create the values to set: -1 where targets_in_global_region is true, 0 otherwise
    #    (though since final_sub_task_map is already 0, we only need to set -1s)
    values_to_set = jnp.where(targets_in_global_region, -1, 0) # 0 could be any non-target value
                                                              # if the map wasn't pre-filled with 0.

    # 5. Set these values in the corresponding region of our new map
    final_sub_task_map = final_sub_task_map.at[region_slice].set(values_to_set)
    
    return final_sub_task_map

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

    print("ok")

    n_envs = 1
    config.num_test_rollouts = n_envs
    config.num_devices = 1

    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type

    current_sub_task_env_cfgs = original_env_cfgs_full_map
    suffle_maps = False
    print(f"Using progressive_gif = {progressive_gif}, shuffle_maps = {suffle_maps}")

    env = TerraEnvBatch(
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

    time.sleep(10)
    
    global_target_map_data = timestep.state.world.target_map.map[0].copy() # Get the first (and only) env's map
    print(f"Captured global target map of shape: {global_target_map_data.shape}")
    global_traversability_map_data = timestep.state.world.traversability_mask.map[0].copy()


    llm_query, runner, _, system_message_master = init_llms(llm_model_key, llm_model_name, USE_PATH, 
                                                                       config, env, n_envs, 
                                                                       APP_NAME, USER_ID, SESSION_ID)
    
    prev_actions_rl = jnp.zeros((1,config.num_prev_actions), dtype=jnp.int32)

    
    print("Starting the game loop with map partitioning...")


    frames = []
    step = 0
    playing = True

    # sub_tasks = [
    #     {'id': 0, 'region_coords': (0, 0, 31, 31), 'start_pos': (16, 16), 'start_angle': 0, 'status': 'pending'},
    #     {'id': 1, 'region_coords': (0, 32, 31, 63), 'start_pos': (16, 48), 'start_angle': 0, 'status': 'pending'},
    #     {'id': 2, 'region_coords': (32, 0, 63, 31), 'start_pos': (48, 16), 'start_angle': 0, 'status': 'pending'},
    #     {'id': 3, 'region_coords': (32, 32, 63, 63), 'start_pos': (48, 48), 'start_angle': 0, 'status': 'pending'}
    # ]
    # sub_tasks = [
    #     {'id': 0, 'region_coords': (0, 0, 15, 63), 'start_pos': (7, 32), 'start_angle': 0, 'status': 'pending'},
    #     {'id': 1, 'region_coords': (16, 0, 31, 63), 'start_pos': (23, 32), 'start_angle': 0, 'status': 'pending'},
    #     {'id': 2, 'region_coords': (32, 0, 47, 63), 'start_pos': (39, 32), 'start_angle': 0, 'status': 'pending'},
    #     {'id': 3, 'region_coords': (48, 0, 63, 63), 'start_pos': (55, 32), 'start_angle': 0, 'status': 'pending'}
    # ]   

    # sub_tasks = [
    #     {'id': 0, 'region_coords': (0, 0, 31, 15), 'start_pos': (16, 8), 'start_angle': 0, 'status': 'pending'},
    #     {'id': 1, 'region_coords': (16, 16, 47, 31), 'start_pos': (32, 24), 'start_angle': 0, 'status': 'pending'},
    #     {'id': 2, 'region_coords': (32, 32, 63, 47), 'start_pos': (48, 40), 'start_angle': 0, 'status': 'pending'},
    #     {'id': 3, 'region_coords': (48, 48, 63, 63), 'start_pos': (56, 56), 'start_angle': 0, 'status': 'pending'}
    # ]

    current_sub_task_idx = -1

        # --- Pygame Visualization Setup ---
    # Determine the size for the sub-task map display (e.g., 5 times scale for a 64x64 map)
    map_display_scale = 1
    map_display_size = (global_target_map_data.shape[1] * map_display_scale, global_target_map_data.shape[0] * map_display_scale)

    sub_task_map_screen = None # Initialize screen variable
    try:
        # Check if Pygame display is already initialized by the environment
        if not pg.display.get_init():
            pg.display.init()
            print("Pygame display initialized.")
        else:
            print("Pygame display already initialized by environment.")

        # Create a new window for the sub-task map display
        sub_task_map_screen = pg.display.set_mode(map_display_size, pg.SCALED | pg.RESIZABLE)
        pg.display.set_caption("Sub-Task Target Map (Visual Check)")
        print(f"Sub-task map display window created with size: {map_display_size}")

    except pg.error as e:
        print(f"Could not create sub-task map display window: {e}")
        print("Running without visual map check.")
    # --- End Pygame Visualization Setup ---


    while playing and step < num_timesteps:
        # Handle events for both windows (main env window and sub-task map window)
        for event in pg.event.get():
            if event.type == QUIT:
                playing = False
            if event.type == pg.KEYDOWN:
                 if event.key == K_q:
                     playing = False
            # Add event handling specific to the sub-task map window if necessary,
            # but QUIT should usually handle closing all windows.
        current_sub_task_idx = step
        #if current_sub_task_idx == -1 or (sub_tasks[current_sub_task_idx]['status'] == 'completed'):
        if current_sub_task_idx is not None:
            #current_sub_task_idx += 1
            

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

            if sub_task_map_screen:
                print("Displaying generated sub-task map...")
                # Use target color (255, 0, 0) for targets, non-target color (0, 0, 0) for empty space
                display_map_pygame(sub_task_target_map_data, sub_task_map_screen, target_color=(255, 0, 0), non_target_color=(0, 0, 0))
                pg.display.flip() # Update the sub-task map display window

                # Optional: Pause to allow inspection
                print("Press any key in the map window to continue...")
                waiting_for_input = True
                while waiting_for_input:
                     for event in pg.event.get():
                         if event.type == pg.QUIT:
                             playing = False
                             waiting_for_input = False
                         if event.type == pg.KEYDOWN:
                             waiting_for_input = False
            else:
                print("Sub-task map display window not available. Skipping visual check.")

        #     single_original_env_cfg = current_sub_task_env_cfgs[0]
        #     original_target_map_config = single_original_env_cfg.target_map

        #     print(f"Original target map config generated")

        #     try:
        #         new_target_map_config_for_subtask = original_target_map_config.replace(
        #             map=sub_task_target_map_data  # Shape (H, W) for the single config
        #         )
        #     except Exception as e:
        #         print(f"CRITICAL ERROR during .replace(map=...) on TargetMapConfig:")
        #         print(f"  Original TargetMapConfig instance: {original_target_map_config}")
        #         print(f"  Type of original TargetMapConfig: {type(original_target_map_config)}")
        #         print(f"  This error suggests that the TargetMapConfig loaded from your checkpoint ")
        #         print(f"  does not have a 'map' attribute, or .replace() doesn't work as expected for it.")
        #         print(f"  The environment might load/use map data via a different attribute or mechanism.")
        #         print(f"  Underlying error: {e}")
        #         raise

        #    # Create a new (single) EnvConfig by replacing its 'target_map' attribute
        #     # with the new_target_map_config_for_subtask we just created.
        #     final_env_cfg_for_subtask = single_original_env_cfg.replace(
        #         target_map=new_target_map_config_for_subtask
        #     )
            
        #     # Re-batch this single EnvConfig so its leaves have a leading dimension of 1.
        #     # This is because env.reset() expects a batched config.
        #     batched_final_env_cfg_for_subtask = jax.tree_map(
        #         lambda x: x[None, ...], final_env_cfg_for_subtask
        #     )

        #     # 3. Reset the environment with the new map and agent position
        #     rng, _rng_reset_subtask = jax.random.split(rng)
        #     rng_reset_subtask = jax.random.split(_rng_reset_subtask, 1) # n_envs = 1

        #     timestep = env.reset(
        #         batched_final_env_cfg_for_subtask, # Use the modified batched config
        #         rng_reset_subtask,
        #         custom_pos=active_task['start_pos'],
        #         custom_angle=active_task['start_angle']
        #     )

        #     active_task['status'] = 'active'

        #     prev_actions_rl = jnp.zeros((1, config.num_prev_actions), dtype=jnp.int32)
        #     # Reset metrics for the new sub-task
        #     current_map = timestep.state.world.target_map.map[0] # Map for current sub-task
        #     initial_target_num_subtask = jnp.sum(current_map < 0)

        # active_task = sub_tasks[current_sub_task_idx]
        # current_observation = timestep.observation # This obs is from the sub-task's 64x64 map
        
        # rng, rng_act, rng_step = jax.random.split(rng, 3)


        # if step % LLM_CALL_FREQUENCY == 0 and not FORCE_DELEGATE_TO_RL:
        #         pass


        # if FORCE_DELEGATE_TO_LLM:
        #     llm_decision= "delegate_to_llm" # For testing, force delegation to LLM agent


        # action_rl = None
        # if llm_decision == "delegate_to_rl":
        #     print("Delegating to RL agent...")
        #      # Prepare the observation for the RL model
        #     obs = obs_to_model_input(current_observation, prev_actions_rl, config)
        #     _, logits_pi = model.apply(model_params, obs)
        #     pi = tfp.distributions.Categorical(logits=logits_pi)
        #     action_rl = pi.sample(seed=rng_act)

        # elif llm_decision == "delegate_to_llm":
        #     action_rl = jnp.array([0], dtype=jnp.int32)
        #     pass



        # if action_rl is not None:

            
        #     prev_actions_rl = jnp.roll(prev_actions_rl, shift=1, axis=1)
        #     prev_actions_rl = prev_actions_rl.at[:, 0].set(action_rl)

        #     _rng_step_subtask = jax.random.split(rng_step, 1) # n_envs = 1
        #     timestep = env.step(
        #         timestep, wrap_action(action_rl, env.batch_cfg.action_type), _rng_step_subtask
        #     )
            
        #     if timestep.info["task_done"][0]: # Check for the first (only) environment
        #         print(f"--- Sub-Task {active_task['id']} COMPLETED by RL agent! ---")
        #         active_task['status'] = 'completed'

        #     env.terra_env.render_obs_pygame(timestep.observation, timestep.info)
        #     if progressive_gif:
        #         game_state_image = capture_screen(pg.display.get_surface()) # Make sure capture_screen is defined
        #         frames.append(game_state_image)

        #     # if jnp.all(timestep.done).item() or t_counter == num_timesteps:
        #     #     break

        # else:
        #     print("No action generated. Skipping step.")
        #     break
        step += 1

    print("Game loop ended.")





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
        default="/home/gioelemo/Documents/terra/gioele_new.pkl",
        help="gioele_new.pkl (8 cabin and 4 base rotations) Version 7 May",
        #default="/home/gioelemo/Documents/terra/new-penalties.pkl",
        #help="new-penalties.pkl (12 cabin and 12 base rotations) Version 7 May",
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
