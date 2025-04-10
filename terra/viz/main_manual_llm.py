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

from pygame.locals import (
    K_q,
    QUIT,
)

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch
from terra.viz.llms import Agent
from terra.viz.llms_utils import *
from terra.viz.a_star import compute_path, simplify_path

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
    USE_PATH = False

    if USE_PATH:
        # Load the JSON configuration file
        with open("envs19.json", "r") as file:
            game_instructions = json.load(file)
    else:
        # Load the JSON configuration file
        with open("envs18.json", "r") as file:
            game_instructions = json.load(file)

    # Define the environment name for the Autonomous Excavator Game
    environment_name = "AutonomousExcavatorGame"

    # Retrieve the system message for the environment
    system_message = game_instructions.get(
        environment_name,
        "You are a game playing assistant. Provide the best action for the current game state."
    )

    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type
    n_envs_x = 1
    n_envs_y = 1
    n_envs = n_envs_x * n_envs_y
    seed = 5810 #24
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
    
    # Initialize the agent
    agent = Agent(model_name=model_name, model=model_key, system_message=system_message, env=env)

    # Define the repeat_action function
    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))
    
    # Trigger the JIT compilation
    timestep = env.step(timestep, repeat_action(action_type.do_nothing()), _rng)
    end_time = time.time()
    print(f"Environment started. Compilation time: {end_time - start_time} seconds.")


    if USE_PATH:
        # Compute the path
        start, target_positions = extract_positions(timestep.state)
        target = find_nearest_target(start, target_positions)
        path, path2, _ = compute_path(timestep.state, start, target)
        #print("Path: ", path)
        #print("\n Path2: ", path2)
        simplified_path = simplify_path(path)
        print("Simplified Path: ", simplified_path)
        #simplified_path2 = simplify_path(path2)
        #print("Simplified Path2: ", simplified_path2)

        initial_orientation = extract_base_orientation(timestep.state)
        initial_direction = initial_orientation["direction"]
        #print("Initial Direction: ", initial_direction)

        actions = path_to_actions(path, initial_direction, 1)
        actions_simple = path_to_actions(simplified_path, initial_direction, 1)
        #print("Action list", actions)
        print("Simple Action list", actions_simple)

        if path:
            game = env.terra_env.rendering_engine
            game.path = path
        else:
            print("No path found.")

    env.terra_env.render_obs_pygame(timestep.observation, timestep.info)

    playing = True
    screen = pg.display.get_surface()
    rewards = 0
    cumulative_rewards = []
    action_list = []
    steps_taken = 0
    num_timesteps = num_timesteps
    frames = []

    USE_LOCAL_MAP = True

    progress_bar = tqdm(total=num_timesteps, desc="Rollout", unit="steps")

    previous_action = []
    current_map = timestep.state.world.target_map.map[0]  # Extract the target map
    previous_map = current_map.copy()  # Initialize the previous map
    count_map_change = 0
    DETERMINISTIC = True

    while playing and steps_taken < num_timesteps:
    #while playing:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                playing = False

        current_map = timestep.state.world.target_map.map[0]  # Extract the target map
        if previous_map is None or not jnp.array_equal(previous_map, current_map):
            print("Map changed!")
            count_map_change += 1

            previous_map = current_map.copy()  # Update the previous map
            previous_action = []  # Reset the previous action list
            agent.delete_messages()  # Clear previous messages

            if USE_PATH:
                # Compute the path
                start, target_positions = extract_positions(timestep.state)
                target = find_nearest_target(start, target_positions)
                path, path2, _ = compute_path(timestep.state, start, target)

                simplified_path = simplify_path(path)
                print("Simplified Path: ", simplified_path)
                initial_orientation = extract_base_orientation(timestep.state)
                initial_direction = initial_orientation["direction"]
                #     #print("Initial Direction: ", initial_direction)

                actions = path_to_actions(path, initial_direction, 1)
                actions_simple = path_to_actions(simplified_path, initial_direction, 1)
                #     #print("Action list", actions)
                print("Simple Action list", actions_simple)

                if path:
                    game = env.terra_env.rendering_engine
                    game.path = path
                else:
                    print("No path found.")


        game_state_image = capture_screen(screen)
        frames.append(game_state_image)

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

            print(f"Current direction: {base_orientation['direction']}")
            print(f"Bucket status: {bucket_status}")
            print(f"Current position: {start} (y,x)")
            print(f"Nearest target position: {nearest_target} (y,x)")
            print(f"Previous action list: {previous_action}")
            
            if USE_PATH:
                usr_msg7 = (
                f"Analyze this game frame and the provided local map to select the optimal action. "
                f"The base of the excavator is currently facing {base_orientation['direction']}. "
                f"The bucket is currently {bucket_status}. "
                f"The excavator is currently located at {start} (y,x). "
                f"The target digging positions are {target_positions} (y,x). "
                f"The traversability mask is provided, where 0 indicates obstacles and 1 indicates traversable areas. "
                f"The list of the previous actions is {previous_action}. "
                f"You can use the action list, computed from the path, to help you decide the next action. The list of actions is {actions_simple}. "
                f"Ensure that the excavator base maintains a safe minimum distance (8 to 11 pixels) from the target area to allow proper alignment of the orange area with the purple area for efficient digging. "
                f"Avoid moving too close to the purple area to prevent overlap with the base. "
                f"If the previous action was digging and the bucket is still empty, moving backward can be an appropriate action to reposition. You can then try to dig (action 6) in the next action. "
                f"Focus on immediate gameplay elements visible in this specific frame and the spatial context from the map. "
                f"Follow the format: {{\"reasoning\": \"detailed step-by-step analysis\", \"action\": X}}"
            )
            else:
                usr_msg7 = (
                f"Analyze this game frame and the provided local map to select the optimal action. "
                f"The base of the excavator is currently facing {base_orientation['direction']}. "
                f"The bucket is currently {bucket_status}. "
                f"The excavator is currently located at {start} (y,x). "
                f"The target digging positions are {target_positions} (y,x). "
                f"The traversability mask is provided, where 0 indicates obstacles and 1 indicates traversable areas. "
                f"The list of the previous actions is {previous_action}. "
                f"Ensure that the excavator base maintains a safe minimum distance (8 to 10 pixels) from the target area to allow proper alignment of the orange area with the purple area for efficient digging. "
                f"Avoid moving too close to the purple area to prevent overlap with the base. "
                f"If the previous action was digging and the bucket is still empty, moving backward can be an appropriate action to reposition. You can then try to dig in the next action. "
                f"Focus on immediate gameplay elements visible in this specific frame and the spatial context from the map. "
                f"Follow the format: {{\"reasoning\": \"detailed step-by-step analysis\", \"action\": X}}"
            )
                
            
            agent.add_user_message(frame=game_state_image, user_msg=usr_msg7, local_map=local_map_image, traversability_map=traversability_map_np)
        else:
            agent.add_user_message(frame=game_state_image, user_msg=usr_msg7, local_map=None)

        action_output, reasoning = agent.generate_response("./")
        
        print(f"\n Action output: {action_output}, Reasoning: {reasoning}")
        
        agent.add_assistant_message()
        #print("agent.MESSAGES: ", agent.messages)

        previous_action.append(action_output)

        # Create the action object
        action = action_type.new(action_output)

        # Add a batch dimension to the action
        action = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), action)

        # Repeat the action for all environments
        batched_action = repeat_action(action)
            
        # Perform the action in the environment
        # rng, _rng = jax.random.split(rng)
        # _rng = _rng[None]
        # timestep = env.step(timestep, batched_action, _rng)
        if DETERMINISTIC:
            key = jnp.array([[count_map_change, count_map_change]], dtype=jnp.uint32)  # Convert to a JAX array

            timestep = env.step(
                timestep,
                repeat_action(action),
                key,
            )
        else:
            rng, _rng = jax.random.split(rng)
            _rng = _rng[None]

            timestep = env.step(
                timestep,
                repeat_action(action),
                _rng,
            )


        # Update rewards and actions
        print(f"Reward: {timestep.reward.item()}")
        rewards += timestep.reward.item()
        cumulative_rewards.append(rewards)
        action_list.append(action_output)

        # Render the environment
        env.terra_env.render_obs_pygame(timestep.observation, timestep.info)

        # if steps_taken % 10 == 0:
        #     agent.delete_messages()

        # Update progress
        steps_taken += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"reward": rewards})
    
    # Close progress bar
    progress_bar.close()

    print(f"Rollout complete. Total reward: {rewards}")


    # Generate a timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a unique output directory for the model and timestamp
    output_dir = os.path.join("experiments", f"{model_name}_{current_time}")
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

