import time
import jax
import jax.numpy as jnp
import pygame as pg
import json
import os
from tqdm import tqdm
import csv
import numpy as np
import datetime
import pickle
import asyncio

# Imports based on visualize.py
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


from pygame.locals import (
    K_q,
    QUIT,
)

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch
from terra.viz.llms_adk import *
from terra.viz.llms_utils import *
from terra.viz.a_star import compute_path, simplify_path

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types


os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


def run_experiment(model_name, model_key, num_timesteps):
    """
    Run an LLM-based simulation experiment.

    Args:
        model_name: The name of the LLM model to use.
        model_key: The name of the LLM model key to use.
        num_timesteps: The number of timesteps to run

    Returns:
        None
    """

    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type
    n_envs_x = 1
    n_envs_y = 1
    n_envs = n_envs_x * n_envs_y
    seed = 33 #35 #33 # 5810 #24
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


    rl_agent_checkpoint_path = "good_discretized_policy.pkl"
    rl_model = None
    rl_model_params = None
    rl_config = None


    try:
        print(f"Loading RL agent from: {rl_agent_checkpoint_path}")
        # Load the dictionary containing config, params, etc.
        rl_log = load_pkl_object(rl_agent_checkpoint_path)
        if rl_log:
            # Extract the training config (needed for model init and helpers)
            rl_config = rl_log.get("train_config")
            if not isinstance(rl_config, TrainConfig):
                 # If train_config wasn't saved directly, try loading from a default path
                 # This might be necessary depending on how good_discretized_policy.pkl was saved
                 print("TrainConfig not found directly in pkl, attempting fallback load...")
                 # Example fallback: Adjust path as needed
                 # fallback_config_path = "path/to/your/original/training/config.pkl"
                 # rl_config = load_pkl_object(fallback_config_path).get("train_config")
                 # OR initialize a default TrainConfig if appropriate
                 # rl_config = TrainConfig() # Adjust with necessary defaults
                 if rl_config is None:
                     raise ValueError("Could not load or determine rl_config (TrainConfig).")


            # Extract the trained model parameters
            rl_model_params = rl_log.get("model")
            if rl_model_params is None:
                 raise ValueError("RL model parameters ('model') not found in pkl file.")

            # Initialize the neural network structure using the loaded config
            # We need a temporary RNG key here, it doesn't affect the loaded params
            rng_init_model, rng = jax.random.split(rng)
            # Ensure rl_config has necessary attributes like num_embeddings_agent_min if needed by get_model_ready
            # You might need to set defaults if they aren't in the loaded config:
            # if not hasattr(rl_config, 'num_embeddings_agent_min'):
            #    rl_config.num_embeddings_agent_min = 60 # Example default from visualize.py
            rl_model, _ = get_model_ready(rng_init_model, rl_config, env) # Pass env for potential shape inference

            print("Successfully loaded RL agent model structure and parameters.")
        else:
            print(f"Error: Failed to load data from {rl_agent_checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: RL agent checkpoint not found at {rl_agent_checkpoint_path}")
    except Exception as e:
        print(f"Error loading RL agent: {e}")
        # Ensure rl_model is None if loading fails
        rl_model = None
        rl_model_params = None
        rl_config = None


    if model_key == "gpt":
        model_name_extended = "openai/{}".format(model_name)
    elif model_key == "claude":
        model_name_extended = "anthropic/{}".format(model_name)
    else:
        model_name_extended =  model_name

    # Initialize the agent
    print("Using model: ", model_name_extended)

    llm_agent = Agent(
        name="MasterAgent",
        model=model_name_extended,
        description="You are a master agent controlling an excavator. Observe the state. Decide if you should act directly (provide action) or delegate digging tasks to a specialized RL agent (respond with 'delegate_to_rl').",
        #instruction=system_message,
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

    llm_query = LLM_query(
        model_name=model_name_extended,
        model=model_key,
        system_message=system_message,
        env=env,
        session_id=SESSION_ID,
        runner=runner,
        user_id=USER_ID,
    )

    print("LLM query initialized.")

    # Define the repeat_action function
    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))
    
    # Trigger the JIT compilation
    timestep = env.step(timestep, repeat_action(action_type.do_nothing()), _rng)
    end_time = time.time()
    print(f"Environment started. Compilation time: {end_time - start_time} seconds.")


    # if USE_PATH:
    #     # Compute the path
    #     start, target_positions = extract_positions(timestep.state)
    #     target = find_nearest_target(start, target_positions)
    #     path, path2, _ = compute_path(timestep.state, start, target)
    #     #print("Path: ", path)
    #     #print("\n Path2: ", path2)
    #     #simplified_path = simplify_path(path)
    #     #print("Simplified Path: ", simplified_path)
    #     #simplified_path2 = simplify_path(path2)
    #     #print("Simplified Path2: ", simplified_path2)

    #     initial_orientation = extract_base_orientation(timestep.state)
    #     initial_direction = initial_orientation["direction"]
    #     #print("Initial Direction: ", initial_direction)

    #     actions = path_to_actions(path, initial_direction, 6)
    #     #actions_simple = path_to_actions(simplified_path, initial_direction, 6)
    #     print("Action list", actions)
    #     #print("Simple Action list", actions_simple)

    #     if path:
    #         game = env.terra_env.rendering_engine
    #         game.path = path
    #     else:
    #         print("No path found.")

    # env.terra_env.render_obs_pygame(timestep.observation, timestep.info)

    # playing = True
    # screen = pg.display.get_surface()
    # rewards = 0
    # cumulative_rewards = []
    # action_list = []
    # steps_taken = 0
    # num_timesteps = num_timesteps
    # frames = []

    # #USE_LOCAL_MAP = True

    # #pg.image.save(screen, "screenshot.png")

    # progress_bar = tqdm(total=num_timesteps, desc="Rollout", unit="steps")

    # previous_action = []
    # current_map = timestep.state.world.target_map.map[0]  # Extract the target map
    # initial_target_num = jnp.sum(current_map < 0)  # Count the initial target pixels
    # print("Initial target number: ", initial_target_num)
    # previous_map = current_map.copy()  # Initialize the previous map
    # count_map_change = 0
    # DETERMINISTIC = True

    prev_actions = None
    if rl_config:
        prev_actions = jnp.zeros(
            (n_envs, rl_config.num_prev_actions),
            dtype=jnp.int32
        )
    else:
        print("Warning: rl_config is None, prev_actions will not be initialized.")



    print("Starting the game loop...")
    max_steps = num_timesteps

    for step in range(max_steps):
        current_observation = timestep.observation

        try:
            obs_dict = {k: v.tolist() for k, v in current_observation.items()}
            observation_str = json.dumps(obs_dict)
        except AttributeError:
            # Handle the case where current_observation is not a dictionary
            observation_str = str(current_observation)

        prompt = f"Current observation: {observation_str}\n\nSystem Message: {system_message}\n\nDecide: Act directly (provide action details) or delegate digging ('delegate_to_rl')?"
        print(f"\n--- Step {step} ---")

        llm_decision = "act directly"  # Placeholder for the LLM decision
        action = None

        try:
            response =  llm_agent.send_message(prompt)
            llm_response_text = response.text
            print(f"LLM response: {llm_response_text}")
            if "delegate_to_rl" in llm_response_text.lower():
                llm_decision = "delegate_to_rl"

        except Exception as adk_err:
            print(f"Error during ADK agent communication: {adk_err}")
            print("Defaulting to fallback action due to ADK error.")
            llm_decision = "fallback" # Indicate fallback needed

        if llm_decision == "delegate_to_rl" and rl_model is not None and rl_model_params is not None and prev_actions is not None and rl_config is not None:
            print("Delegating to RL agent...")
            try:
                # Prepare the observation for the RL model
                obs_input = obs_to_model_input(current_observation, prev_actions, rl_config)

                _, logits_pi = rl_model.apply(rl_model_params, obs_input)

                pi = tfp.distributions.Categorical(logits=logits_pi)
                action = pi.mode()
                print(f"RL agent action: {action}")

            except:
                print(f"Error during RL agent action generation: ")
                print("Using fallback random action due to RL error.")
                action = env.action_space.sample(rng) # Fallback random action
                llm_decision = "fallback" # Mark as fallback

        else:
            if llm_decision != "delegate_to_rl":
                print("Master Agent acts directly (or RL agent unavailable/ADK error).")
                 # --- Add LLM Action Parsing Logic Here ---
                # TODO PASS LLM response to a function that parses the action
                continue
                 # Example: Try to parse action from llm_response_text
                 # If parsing fails or llm_decision is "fallback":
                print("LLM direct action parsing not implemented or fallback needed. Using random action.")
                action = env.action_space.sample(rng) # Example: random action
                # --- End LLM Action Parsing ---
                print(f"LLM/Fallback action: {action}")
            else:
                # This case means delegation was intended but RL agent wasn't loaded properly
                print("Error: Delegation requested, but RL agent is not available. Using random action.")
                action = env.action_space.sample(rng) # Fallback random action
                llm_decision = "fallback"
        
        if action is not None:
            action_formatted = jnp.array(action).reshape(1, -1)  # Reshape to match expected input

            if prev_actions is not None:
                prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
                prev_actions = prev_actions.at[:, 0].set(action_formatted.squeeze())

            rng, _rng = jax.random.split(rng)
            _rng = _rng[None]

            timestep = env.step(action_formatted, _rng)

            env.render(timestep.observation)
            if timestep.done.any() or timestep.truncated.any():
                print("Episode done.")
                break
        else:
            print("No action generated. Skipping step.")
            break


    env.close()
    print("Game loop ended.")


            

    # while playing and steps_taken < num_timesteps:
    # #while playing:
    #     for event in pg.event.get():
    #         if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
    #             playing = False

    #     current_map = timestep.state.world.target_map.map[0]  # Extract the target map
    #     if previous_map is None or not jnp.array_equal(previous_map, current_map):
    #         print("Map changed!")
    #         count_map_change += 1
    #         initial_target_num = jnp.sum(current_map < 0)  # Count the initial target pixels
    #         print("Current target number: ", initial_target_num)

    #         previous_map = current_map.copy()  # Update the previous map
    #         previous_action = []  # Reset the previous action list
    #         llm_query.delete_messages()  # Clear previous messages

    #         if USE_PATH:
    #             # Compute the path
    #             start, target_positions = extract_positions(timestep.state)
    #             target = find_nearest_target(start, target_positions)
    #             path, path2, _ = compute_path(timestep.state, start, target)

    #             simplified_path = simplify_path(path)
    #             print("Simplified Path: ", simplified_path)
    #             initial_orientation = extract_base_orientation(timestep.state)
    #             initial_direction = initial_orientation["direction"]
    #             print("Initial Direction: ", initial_direction)

    #             actions = path_to_actions(path, initial_direction, 6)
    #             actions_simple = path_to_actions(simplified_path, initial_direction, 6)
    #             print("Action list", actions)
    #             print("Simple Action list", actions_simple)

    #             if path:
    #                 game = env.terra_env.rendering_engine
    #                 game.path = path
    #             else:
    #                 print("No path found.")


    #     game_state_image = capture_screen(screen)
    #     frames.append(game_state_image)

    #     state = timestep.state
    #     base_orientation = extract_base_orientation(state)
    #     bucket_status = extract_bucket_status(state)  # Extract bucket status


    #     traversability_map = state.world.traversability_mask.map[0]  # Extract the traversability map

    #     traversability_map_np = np.array(traversability_map)  # Convert JAX array to NumPy
    #     traversability_map_np = (traversability_map_np * 255).astype(np.uint8)



    #     if USE_LOCAL_MAP:
    #         local_map = generate_local_map(timestep)
    #         local_map_image = local_map_to_image(local_map)

    #         start, target_positions = extract_positions(timestep.state)
    #         nearest_target = find_nearest_target(start, target_positions)

    #         #current_target_num = jnp.sum(current_map < 0)  # Count the current target pixels
    #         #print("Current target number: ", current_target_num)
            
    #         percentage_digging = calculate_digging_percentage(initial_target_num, timestep)

    #         print(f"Current direction: {base_orientation['direction']}")
    #         print(f"Bucket status: {bucket_status}")
    #         print(f"Current position: {start} (y,x)")
    #         print(f"Nearest target position: {nearest_target} (y,x)")
    #         print(f"Previous action list: {previous_action}")
    #         print(f"Percentage of digging left: {percentage_digging:.2f}%")
            
    #         if USE_PATH:

    #             usr_msg8 = (
    #                 f"Analyze the current game frame and local map to determine the optimal next action.\n\n"
    #                 f"- Each forward/backward move advances the excavator by 6 pixels.\n"
    #                 f"- Excavator base is facing **{base_orientation['direction']}**.\n"
    #                 f"- Bucket status: **{bucket_status}**.\n"
    #                 f"- Current location: **{start}** (y, x).\n"
    #                 f"- Target digging positions: **{target_positions}** (y, x).\n"
    #                 f"- Previous actions: **{previous_action}**.\n"
    #                 f"- Suggested actions (NOT compulsory): **{actions}**.\n"
    #                 f"- Remaining area to dig: **{percentage_digging:.2f}%**.\n"
    #                 #f"- You could generate and execute Python code to help you to dig correctly (for example for mathematical operations).\n\n"

    #                 f"**DIGGING RULES**\n"
    #                 f"- Align the bucket **directly facing the long edge** of the purple trench. This may require rotation and repositioning.\n"
    #                 f"- You may **overlap the base with the purple trench** for proper alignment.\n"
    #                 f"- Digging must start from the **furthest end** of the trench (relative to the excavator's front) and proceed **backward**.\n"
    #                 f"  - Facing up → reach **topmost** trench point → dig downward\n"
    #                 f"  - Facing down → reach **bottommost** → dig upward\n"
    #                 f"  - Facing left → reach **leftmost** → dig rightward\n"
    #                 f"  - Facing right → reach **rightmost** → dig leftward\n"
    #                 f"- Ensure the **orange target area overlaps the purple trench** before digging.\n"
    #                 f"- Purple turns **green** after successful excavation.\n\n"

    #                 f"**MOVEMENT GUIDELINES**\n"
    #                 f"- Avoid repeated forward/backward moves unless repositioning.\n"
    #                 f"- Maintain **8–12 pixel distance** between the excavator and the trench for best alignment.\n"
    #                 f"- If the bucket is empty, it’s okay to reposition and try digging again next.\n"
    #                 f"- A common sequence: 6 (dig), rotate twice, 6 (deposit), rotate back, 1 (backward), 6 (dig), ...\n\n"
    #                 f"- Do not dig twice (or more) in the same position. The sequence of action [..., 6, 6, ...] is NOT allowed\n"

    #                 f"**RESTRICTIONS**\n"
    #                 f"- Do not dig in areas that are:\n"
    #                 f"  - Already dug\n"
    #                 f"  - Not marked as negative in the target map\n"
    #                 f"- Use the traversability mask (`0 = obstacle`, `1 = traversable`) to guide movement.\n"
    #                 f"- Continue acting until **0%** of the trench remains undug.\n\n"

    #                 f"Return your response in this format:\n"
    #                 f'{{"reasoning": "step-by-step analysis", "action": X}}'
    #             )


    #         else:
    #             usr_msg7 = (
    #             f"Analyze this game frame and the provided local map to select the optimal action. "
    #             f"The base of the excavator is currently facing {base_orientation['direction']}. "
    #             f"The bucket is currently {bucket_status}. "
    #             f"The excavator is currently located at {start} (y,x). "
    #             f"The target digging positions are {target_positions} (y,x). "
    #             f"The traversability mask is provided, where 0 indicates obstacles and 1 indicates traversable areas. "
    #             f"The list of the previous actions is {previous_action}. "
    #             f"Ensure that the excavator base maintains a safe minimum distance (8 to 10 pixels) from the target area to allow proper alignment of the orange area with the purple area for efficient digging. "
    #             f"Avoid moving too close to the purple area to prevent overlap with the base. "
    #             f"If the previous action was digging and the bucket is still empty, moving backward can be an appropriate action to reposition. You can then try to dig in the next action. "
    #             f"Focus on immediate gameplay elements visible in this specific frame and the spatial context from the map. "
    #             f"Follow the format: {{\"reasoning\": \"detailed step-by-step analysis\", \"action\": X}}"
    #         )
                
            
    #         llm_query.add_user_message(frame=game_state_image, user_msg=usr_msg8, local_map=local_map_image, traversability_map=traversability_map_np)
    #     else:
    #         llm_query.add_user_message(frame=game_state_image, user_msg=usr_msg7, local_map=None)

    #     action_output, reasoning = llm_query.generate_response("./")

    #     print(f"\n Action output: {action_output}, Reasoning: {reasoning}")
        
    #     llm_query.add_assistant_message()
    #     #print("agent.MESSAGES: ", agent.messages)

    #     previous_action.append(action_output)

    #     # Create the action object
    #     action = action_type.new(action_output)

    #     # Add a batch dimension to the action
    #     action = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), action)

    #     # Repeat the action for all environments
    #     batched_action = repeat_action(action)
            
    #     # Perform the action in the environment
    #     # rng, _rng = jax.random.split(rng)
    #     # _rng = _rng[None]
    #     # timestep = env.step(timestep, batched_action, _rng)
    #     if DETERMINISTIC:
    #         key = jnp.array([[count_map_change, count_map_change]], dtype=jnp.uint32)  # Convert to a JAX array

    #         timestep = env.step(
    #             timestep,
    #             repeat_action(action),
    #             key,
    #         )
    #     else:
    #         rng, _rng = jax.random.split(rng)
    #         _rng = _rng[None]

    #         timestep = env.step(
    #             timestep,
    #             repeat_action(action),
    #             _rng,
    #         )


    #     # Update rewards and actions
    #     print(f"Reward: {timestep.reward.item()}")
    #     rewards += timestep.reward.item()
    #     cumulative_rewards.append(rewards)
    #     action_list.append(action_output)

    #     # Render the environment
    #     env.terra_env.render_obs_pygame(timestep.observation, timestep.info)

    #     # if steps_taken % 5 == 1:
    #     #     agent.delete_messages()

    #     # Update progress
    #     steps_taken += 1
    #     progress_bar.update(1)
    #     progress_bar.set_postfix({"reward": rewards})
    
    # # Close progress bar
    # progress_bar.close()

    # print(f"Rollout complete. Total reward: {rewards}")


    # Generate a timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a unique output directory for the model and timestamp
    output_dir = os.path.join("experiments", f"{model_name}_{current_time}")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save actions and rewards to a CSV file
    output_file = os.path.join(output_dir, "actions_rewards.csv")
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["actions", "cumulative_rewards"])
        for action, cum_reward in zip(action_list, cumulative_rewards):
            writer.writerow([action, cum_reward])

    print(f"Results saved to {output_file}")

    # Save the gameplay video
    video_path = os.path.join(output_dir, "gameplay.mp4")
    save_video(frames, video_path)

