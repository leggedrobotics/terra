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

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

import asyncio
import os
import argparse
import datetime
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
LLM_CALL_FREQUENCY = 5          # Number of steps between LLM calls
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
    n_envs_x = n_envs_x
    n_envs_y = n_envs_y
    n_envs = n_envs_x * n_envs_y

    print(f"Loading RL agent from: {agent_checkpoint_path}")
    log = load_pkl_object(agent_checkpoint_path)
    config = log["train_config"]
    config.num_test_rollouts = n_envs
    config.num_devices = 1    

  
    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type
    env_cfgs = log["env_config"]
    env_cfgs = jax.tree_map(
        lambda x: x[0][None, ...].repeat(n_envs, 0), env_cfgs
    ) 
    print(f"Using progressive_gif = {progressive_gif}")
    suffle_maps = True
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
    rng_reset = jax.random.split(_rng, config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)


    # Initialize the LLM agent
    llm_query, runner, prev_actions, system_message_master = init_llms(llm_model_key, llm_model_name, USE_PATH, 
                                                                       config, env, n_envs, 
                                                                       APP_NAME, USER_ID, SESSION_ID)

    
    print("Starting the game loop...")

    t_counter = 0
    reward_seq = []
    obs_seq = []
    action_list = []

    last_llm_decision = "delegate_to_rl" 

    if USE_PATH:
        actions=compute_action_list(timestep, env) 

    # Define the repeat_action function
    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))
    
    # Trigger the JIT compilation
    timestep = env.step(timestep, repeat_action(action_type.do_nothing()), rng_reset)
    env.terra_env.render_obs_pygame(timestep.observation, timestep.info)

    screen = pg.display.get_surface()
    frames = []
    step = 0
    playing = True

    #previous_action = []
    current_map = timestep.state.world.target_map.map[0]  # Extract the target map
    initial_target_num = jnp.sum(current_map < 0)  # Count the initial target pixels
    print("Initial target number: ", initial_target_num)
    previous_map = current_map.copy()  # Initialize the previous map
    count_map_change = 0
    DETERMINISTIC = True

    PROMPT_FILENAME = "usr_msg8.txt"
    PROMPT_NO_PATH_FILENAME = "usr_msg7.txt" # New filename
    try:
        with open(PROMPT_FILENAME, 'r') as f:
            prompt_template_string = f.read()
        print(f"Successfully loaded prompt template from {PROMPT_FILENAME}")
        # Read the second template
        with open(PROMPT_NO_PATH_FILENAME, 'r') as f:
            prompt_template_no_path_string = f.read()
        print(f"Successfully loaded prompt template from {PROMPT_NO_PATH_FILENAME}")

    except FileNotFoundError as e:
        print(f"ERROR: Prompt template file not found: {e.filename}")
        # Handle the error appropriately, e.g., exit or use a default prompt

    while playing and step < num_timesteps:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                playing = False
        print(f"\n--- Step {step} ---")
        current_observation = timestep.observation
        obs_seq.append(current_observation)

        rng, rng_act, rng_step = jax.random.split(rng, 3)
        llm_decision = last_llm_decision # Default to last decision

        game_state_image = capture_screen(screen)
        frames.append(game_state_image)

        if step % LLM_CALL_FREQUENCY == 0:
            print("Calling LLM agent for decision...")
            try:
                obs_dict = {k: v.tolist() for k, v in current_observation.items()}
                observation_str = json.dumps(obs_dict)

            except AttributeError:
                # Handle the case where current_observation is not a dictionary
                observation_str = str(current_observation)

            if USE_IMAGE_PROMPT:
                prompt = f"Current observation: See image \n\nSystem Message: {system_message_master}\n\nDecide: Act directly (provide action details) or delegate digging to RL ('delegate_to_rl') or to LLM ('delegate_to_llm')?"
            else:
                prompt = f"Current observation: {observation_str}\n\nSystem Message: {system_message_master}\n\nDecide: Act directly (provide action details) or delegate digging ('delegate_to_rl') or to LLM ('delegate_to_llm')?"
            #print(f"Prompt: {prompt}")

            llm_decision = "act directly"  # Placeholder for the LLM decision
            #llm_decision = "delegate_to_rl" # For testing, force delegation to RL agent

            if not FORCE_DELEGATE_TO_RL:
                try:
                    if USE_IMAGE_PROMPT:
                        response = asyncio.run(call_agent_async(prompt, game_state_image, runner, USER_ID, SESSION_ID))
                    else:
                        response = asyncio.run(call_agent_async(prompt, game_state_image=None, runner=runner, USER_ID=USER_ID, SESSION_ID=SESSION_ID))
                    llm_response_text = response
                    print(f"LLM response: {llm_response_text}")
                    if "delegate_to_rl" in llm_response_text.lower():
                        llm_decision = "delegate_to_rl"
                        last_llm_decision = llm_decision # Update last decision

                except Exception as adk_err:
                    print(f"Error during ADK agent communication: {adk_err}")
                    print("Defaulting to fallback action due to ADK error.")
                    llm_decision = "fallback" # Indicate fallback needed
                    last_llm_decision = llm_decision # Update last decision
            else:
                print("Forcing delegation to RL agent for testing.")
                llm_decision = "delegate_to_rl" # For testing, force delegation to RL agent
                last_llm_decision = llm_decision # Update last decision
        if FORCE_DELEGATE_TO_LLM:
            llm_decision= "delegate_to_llm" # For testing, force delegation to LLM agent
        #if llm_decision == "delegate_to_rl" and model is not None and model_params is not None and prev_actions is not None and config is not None:
        if llm_decision == "delegate_to_rl":
            print("Delegating to RL agent...")
            try:
                # Prepare the observation for the RL model
                obs = obs_to_model_input(current_observation, prev_actions, config)
                _, logits_pi = model.apply(model_params, obs)
                pi = tfp.distributions.Categorical(logits=logits_pi)
                action = pi.sample(seed=rng_act)
                action_list.append(action)
                
                print(f"RL agent action: {action}")

            except:
                print(f"Error during RL agent action generation: ")
                print("Using fallback random action due to RL error.")
                action = jnp.array([-1], dtype=jnp.int32) # Use jnp.array
                action_list.append(action)
                llm_decision = "fallback" # Mark as fallback
                last_llm_decision = llm_decision # Update last decision
        elif llm_decision == "delegate_to_llm":
            print("Acting directly based on LLM decision...")
            current_map = timestep.state.world.target_map.map[0]  # Extract the target map
            if previous_map is None or not jnp.array_equal(previous_map, current_map):
                print("Map changed!")
                count_map_change += 1
                initial_target_num = jnp.sum(current_map < 0)  # Count the initial target pixels
                print("Current target number: ", initial_target_num)

                previous_map = current_map.copy()  # Update the previous map
                llm_query.delete_messages()  # Clear previous messages

                if USE_PATH:
                    actions=compute_action_list(timestep, env) 

            state = timestep.state
            base_orientation = extract_base_orientation(state)
            bucket_status = extract_bucket_status(state)  # Extract bucket status


            traversability_map = state.world.traversability_mask.map[0]  # Extract the traversability map

            traversability_map_np = np.array(traversability_map)  # Convert JAX array to NumPy
            traversability_map_np = (traversability_map_np * 255).astype(np.uint8)

            if USE_LOCAL_MAP:
                local_map = generate_local_map(timestep)
                local_map_image = local_map_to_image(local_map)

                start, target_positions = extract_positions(timestep.state)
                nearest_target = find_nearest_target(start, target_positions)

                percentage_digging = calculate_digging_percentage(initial_target_num, timestep)

                print(f"Current direction: {base_orientation['direction']}")
                print(f"Bucket status: {bucket_status}")
                print(f"Current position: {start} (y,x)")
                print(f"Nearest target position: {nearest_target} (y,x)")
                print(f"Previous action list: {action_list}")
                print(f"Percentage of digging left: {percentage_digging:.2f}%")
            
                if USE_PATH:

                    usr_msg8 = prompt_template_string.format(
                        base_direction=base_orientation['direction'],
                        bucket_status=bucket_status,
                        current_pos=start,
                        target_pos_list=target_positions,
                        prev_actions=action_list,
                        suggested_actions=actions,
                        dig_percentage=f"{percentage_digging:.2f}%"
                    )

                    llm_query.add_user_message(frame=game_state_image, user_msg=usr_msg8, local_map=local_map_image, traversability_map=traversability_map_np)

                else:
                    usr_msg7 = prompt_template_no_path_string.format(
                        base_direction=base_orientation['direction'],
                        bucket_status=bucket_status,
                        current_pos=start,
                        target_pos_list=target_positions,
                        prev_actions=action_list
                    )
                
                    llm_query.add_user_message(frame=game_state_image, user_msg=usr_msg7, local_map=None)

            action_output, reasoning = llm_query.generate_response("./")

            print(f"\n Action output: {action_output}, Reasoning: {reasoning}")
            llm_query.add_assistant_message()

            action = jnp.array([action_output], dtype=jnp.int32)  # Convert to JAX array
            action_list.append(action)

        else:

            if llm_decision != "delegate_to_rl" or llm_decision != "delegate_to_llm":
                print("Master Agent acts directly (or RL agent unavailable/ADK error).")
                # TODO PASS LLM response to a function that parses the action
                action = jnp.array([-1], dtype=jnp.int32) # Use jnp.array
                action_list.append(action)
            else:
                # This case means delegation was intended but RL agent wasn't loaded properly
                print("Error: Delegation requested, but RL agent is not available. Using do nothing action.")
                action = jnp.array([-1], dtype=jnp.int32) # Use jnp.array
                action_list.append(action)
                llm_decision = "fallback"
                last_llm_decision = llm_decision # Update last decision

        if action is not None:

            if prev_actions is not None:
                prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
                prev_actions = prev_actions.at[:, 0].set(action)

            rng_step = jax.random.split(rng_step, config.num_test_rollouts)
            timestep = env.step(
                timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
            )

            reward_seq.append(timestep.reward)
            print(t_counter, timestep.reward, action, timestep.done)
            print(10 * "=")
            t_counter += 1

            env.terra_env.render_obs_pygame(timestep.observation, timestep.info)

            if jnp.all(timestep.done).item() or t_counter == num_timesteps:
                break

        else:
            print("No action generated. Skipping step.")
            break
        step += 1

    print(f"Terra - Steps: {t_counter}, Return: {np.sum(reward_seq)}")
    #print(len(action_list), len(reward_seq), len(obs_seq))
    
    # for o in tqdm(obs_seq, desc="Rendering"):
    #     env.terra_env.render_obs_pygame(o, generate_gif=True)

    # Calculate cumulative rewards
    # Ensure reward_seq contains numbers before calculating cumulative sum
    numeric_reward_seq = [r[0] if hasattr(r, '__getitem__') and len(r) > 0 else r for r in reward_seq]
    cumulative_rewards = np.cumsum(numeric_reward_seq)

    #print("Individual Rewards:", reward_seq)
    #print("Cumulative Rewards:", cumulative_rewards)

    print("Actions:", action_list)

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
        default="/home/gioelemo/Documents/terra/gioele.pkl",
        help="new-maps-different-order.pkl (8 cabin and 4 base rotations)",
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
