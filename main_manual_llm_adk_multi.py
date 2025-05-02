# # import time
# # import jax
# # import jax.numpy as jnp
# # import pygame as pg
# import json
# # import os
# # from tqdm import tqdm
# # import csv
# # import numpy as np
# # import datetime
# # import pickle
# # import asyncio

# import os # Import os
# import sys # Import sys
# import datetime
# import pickle
# import asyncio


# # Imports based on visualize.py
# import numpy as np
# import jax
# from tqdm import tqdm
# from utils.models import get_model_ready
# from utils.helpers import load_pkl_object
# from terra.env import TerraEnvBatch
# import jax.numpy as jnp
# from utils.utils_ppo import obs_to_model_input, wrap_action
# from terra.state import State
# import matplotlib.animation as animation

# # from utils.curriculum import Curriculum
# from tensorflow_probability.substrates import jax as tfp
# from train import TrainConfig  # needed for unpickling checkpoints


# from terra.config import EnvConfig


# from pygame.locals import (
#     K_q,
#     QUIT,
# )

# from terra.config import BatchConfig
# from terra.config import EnvConfig
# from terra.env import TerraEnvBatch
# from terra.viz.llms_adk import *
# from terra.viz.llms_utils import *
# from terra.viz.a_star import compute_path, simplify_path



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

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

FORCE_DELEGATE_TO_RL = False
LLM_CALL_FREQUENCY = 5  # Number of steps between LLM calls

async def call_agent_async(query: str, runner, user_id, session_id):
  """Sends a query to the agent and prints the final response."""
  #print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      # You can uncomment the line below to see *all* events during execution
      # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          break # Stop processing events once the final response is found

  #print(f"<<< Agent Response: {final_response_text}")
  return final_response_text


def load_neural_network(config, env):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env)
    return model


def run_experiment(llm_model_name, llm_model_key, num_timesteps, n_envs_x, n_envs_y, out_path, seed, progressive_gif, run):
    """
    Run an LLM-based simulation experiment.

    Args:
        model_name: The name of the LLM model to use.
        model_key: The name of the LLM model key to use.
        num_timesteps: The number of timesteps to run.
        n_envs_x: The number of environments along the x-axis.
        n_envs_y: The number of environments along the y-axis.
        out_path: The output path for saving results.
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
    # Load the dictionary containing config, params, etc.
    n_envs_x = n_envs_x
    n_envs_y = n_envs_y
    n_envs = n_envs_x * n_envs_y

    log = load_pkl_object(agent_checkpoint_path)
    config = log["train_config"]
    config.num_test_rollouts = n_envs
    config.num_devices = 1    



    system_message = "You are a master agent controlling an excavator. Observe the state. Decide if you should act directly (provide action) or delegate digging tasks to a specialized RL agent (respond with 'delegate_to_rl')."



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
        display=False,
        progressive_gif=progressive_gif,
        shuffle_maps=suffle_maps,
    )
    config.num_embeddings_agent_min = 60  # curriculum.get_num_embeddings_agent_min()

    model = load_neural_network(config, env)
    model_params = log["model"]
    # replicated_params = log['network']
    # model_params = jax.tree_map(lambda x: x[0], replicated_params)


    # print("Starting the environment...")
    # start_time = time.time()
    # env_cfgs = jax.vmap(lambda x: EnvConfig.new())(jnp.arange(n_envs))
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)
    # prev_actions = jnp.zeros(
    #     (rl_config.num_test_rollouts, rl_config.num_prev_actions),
    #     dtype=jnp.int32
    # )



    # Initialize the LLM agent
    if llm_model_key == "gpt":
        llm_model_name_extended = "openai/{}".format(llm_model_name)
    elif llm_model_key == "claude":
        llm_model_name_extended = "anthropic/{}".format(llm_model_name)
    else:
        llm_model_name_extended =  llm_model_name
    
    print("Using model: ", llm_model_name_extended)

    if llm_model_key == "gemini":
        llm_agent = Agent(
            name="MasterAgent",
            model=llm_model_name_extended,
            description="You are a master agent controlling an excavator. Observe the state. Decide if you should act directly (provide action) or delegate digging tasks to a specialized RL agent (respond with 'delegate_to_rl').",
            instruction=system_message,
        )
    else:
        llm_agent = Agent(
            name="MasterAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description="You are a master agent controlling an excavator. Observe the state. Decide if you should act directly (provide action) or delegate digging tasks to a specialized RL agent (respond with 'delegate_to_rl').",
            instruction=system_message,
        )
    
    print("Agent initialized.")



    session_service = InMemorySessionService()

    APP_NAME = "ExcavatorGameApp"
    USER_ID = "user_1"
    SESSION_ID = "session_001"

    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    print("Session created. App: ", APP_NAME, " User ID: ", USER_ID, " Session ID: ", SESSION_ID)
    
    runner = Runner(
        agent=llm_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    print(f"Runner initialized for agent {runner.agent.name}.")



    prev_actions = None
    if config:
        prev_actions = jnp.zeros(
            (n_envs, config.num_prev_actions),
            dtype=jnp.int32
        )
    else:
        print("Warning: rl_config is None, prev_actions will not be initialized.")



    print("Starting the game loop...")
    max_steps = num_timesteps

    t_counter = 0
    reward_seq = []
    obs_seq = []
    action_list = []

    last_llm_decision = "delegate_to_rl" # Initial decision (or 'act_directly')


    for step in range(max_steps):
        print(f"\n--- Step {step} ---")
        current_observation = timestep.observation
        obs_seq.append(current_observation)

        rng, rng_act, rng_step = jax.random.split(rng, 3)
        llm_decision = last_llm_decision # Default to last decision

        if step % LLM_CALL_FREQUENCY == 0:
            print("Calling LLM agent for decision...")
            try:
                obs_dict = {k: v.tolist() for k, v in current_observation.items()}
                observation_str = json.dumps(obs_dict)
            except AttributeError:
                # Handle the case where current_observation is not a dictionary
                observation_str = str(current_observation)

            prompt = f"Current observation: {observation_str}\n\nSystem Message: {system_message}\n\nDecide: Act directly (provide action details) or delegate digging ('delegate_to_rl')?"
            #print(f"Prompt: {prompt}")

            llm_decision = "act directly"  # Placeholder for the LLM decision
            #llm_decision = "delegate_to_rl" # For testing, force delegation to RL agent

            #action = None

            if not FORCE_DELEGATE_TO_RL:
                try:
                    response =  asyncio.run(call_agent_async(prompt, runner, USER_ID, SESSION_ID))
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
                #action = env.action_space.sample(rng) # Fallback random action
                action = jnp.array([-1], dtype=jnp.int32) # Use jnp.array
                action_list.append(action)
                llm_decision = "fallback" # Mark as fallback
                last_llm_decision = llm_decision # Update last decision

        else:
            if llm_decision != "delegate_to_rl":
                print("Master Agent acts directly (or RL agent unavailable/ADK error).")
                 # --- Add LLM Action Parsing Logic Here ---
                # TODO PASS LLM response to a function that parses the action
                action = jnp.array([-1], dtype=jnp.int32) # Use jnp.array
                action_list.append(action)

            else:
                # This case means delegation was intended but RL agent wasn't loaded properly
                print("Error: Delegation requested, but RL agent is not available. Using do nothing action.")
                #action = env.action_space.sample(rng) # Fallback random action
                action = jnp.array([-1], dtype=jnp.int32) # Use jnp.array
                action_list.append(action)
                llm_decision = "fallback"
                last_llm_decision = llm_decision # Update last decision

        if action is not None:
            #action_formatted = jnp.array(action).reshape(1, -1)  # Reshape to match expected input

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
            # if done or t_counter == max_frames:
            #     break
            # else:
            if jnp.all(timestep.done).item() or t_counter == max_steps:
                break
            # env_state = next_env_state
            # obs = next_obs            if timestep.done.any() or timestep.truncated.any():

        else:
            print("No action generated. Skipping step.")
            break
    print(f"Terra - Steps: {t_counter}, Return: {np.sum(reward_seq)}")
    print(len(action_list), len(reward_seq), len(obs_seq))
    
    for o in tqdm(obs_seq, desc="Rendering"):
        env.terra_env.render_obs_pygame(o, generate_gif=True)

    # Calculate cumulative rewards
    # Ensure reward_seq contains numbers before calculating cumulative sum
    numeric_reward_seq = [r[0] if hasattr(r, '__getitem__') and len(r) > 0 else r for r in reward_seq]
    cumulative_rewards = np.cumsum(numeric_reward_seq)

    #print("Individual Rewards:", reward_seq)
    #print("Cumulative Rewards:", cumulative_rewards)
    #print("Actions:", action_list)


    # Generate a timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a unique output directory for the model and timestamp
    # Use a safe version of the model name for the directory
    safe_model_name = llm_model_name.replace('/', '_') # Replace slashes if any
    output_dir = os.path.join("experiments", f"{safe_model_name}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    # Save actions and cumulative rewards to a CSV file
    output_file = os.path.join(output_dir, "actions_rewards.csv") # Renamed file
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["actions", "cumulative_rewards"]) # Header updated
        # Iterate through actions and the calculated cumulative rewards
        for action, cum_reward in zip(action_list, cumulative_rewards):
            # Assuming action is array-like (e.g., JAX array) with one element
            action_value = action[0] if hasattr(action, '__getitem__') and len(action) > 0 else action
            # cum_reward from np.cumsum is already a scalar number
            reward_value = cum_reward
            writer.writerow([action_value, reward_value])

    print(f"Results saved to {output_file}")

    # Save the gameplay video
    gif_path = os.path.join(output_dir, "gameplay.gif")
    # ... (rest of the code, including gif saving) ...
    env.terra_env.rendering_engine.create_gif(gif_path)


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
        "-o",
        "--out_path",
        type=str,
        default=".",
        help="Output path.",
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
                   args.out_path, 
                   args.seed, 
                   args.progressive_gif, 
                   args.run_name)
