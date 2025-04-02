import time
import jax
import jax.numpy as jnp
import pygame as pg
import json
import os
from tqdm import tqdm
import csv
import numpy as np


from pygame.locals import (
    K_q,
    QUIT,
)

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch
from terra.viz.llms import Agent
from terra.viz.llms_utils import *

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
    # Load the JSON configuration file
    with open("envs9.json", "r") as file:
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
    seed = 5810
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
    # state = timestep.state
    # base_orientation = extract_base_orientation(state)
    # print(base_orientation)

    while playing and steps_taken < num_timesteps:
    #while playing:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                playing = False
        
        game_state_image = capture_screen(screen)
        frames.append(game_state_image)

        state = timestep.state
        base_orientation = extract_base_orientation(state)
        bucket_status = extract_bucket_status(state)  # Extract bucket status


        usr_msg0 = "What action should be taken?"
        usr_msg1 ='Analyze this game frame and select the optimal action. Focus on immediate gameplay elements visible in this specific frame, and follow the format: {"reasoning": "detailed step-by-step analysis", "action": X}'
        usr_msg2 = (
            f"Analyze this game frame and the provided local map to select the optimal action. "
            f"The base of the excavator is currently facing {base_orientation['direction']}. "
            f"Focus on immediate gameplay elements visible in this specific frame and the spatial context from the map. "
            f"Follow the format: {{\"reasoning\": \"detailed step-by-step analysis\", \"action\": X}}"
        )

        usr_msg3 = (
            f"Analyze this game frame and the provided local map to select the optimal action. "
            f"The base of the excavator is currently facing {base_orientation['direction']}. "
            f"The bucket is currently {bucket_status}. "
            f"Focus on immediate gameplay elements visible in this specific frame and the spatial context from the map. "
            f"Follow the format: {{\"reasoning\": \"detailed step-by-step analysis\", \"action\": X}}"
        )

        
        if USE_LOCAL_MAP:
            local_map = generate_local_map(timestep)
            local_map_image = local_map_to_image(local_map)

            local_map_summary = summarize_local_map(local_map)
            #print(local_map_summary)

            # usr_msg3 = (
            #     f"Analyze this game frame and the provided local map to select the optimal action. "
            #     f"The base of the excavator is currently facing {base_orientation['direction']}. "
            #     f"The bucket is currently {bucket_status}. "
            #     f"{local_map_summary} "
            #     f"Focus on immediate gameplay elements visible in this specific frame and the spatial context from the map. "
            #     f"Follow the format: {{\"reasoning\": \"detailed step-by-step analysis\", \"action\": X}}"
            # )


            current, target = extract_positions(state)

            # current_position_str = f"({current_position['x']}, {current_position['y']})"


            # if target_position:
            #     target_position_str = f"at coordinates ({target_position['x']}, {target_position['y']})"
            # else:
            #     target_position_str = "not currently visible in the global map"

            print(f"Current direction: {base_orientation['direction']}")
            print(f"Bucket status: {bucket_status}")
            print(f"Current position: {current} (y,x)")
            print(f"Target position: {target} (y,x)")
            
            usr_msg4 = (
                f"Analyze this game frame and the provided local map to select the optimal action. "
                f"The base of the excavator is currently facing {base_orientation['direction']}. "
                f"The bucket is currently {bucket_status}. "
                f"The excavator is currently located at {current} (y,x). "
                f"The nearest target digging position is {target} (y,x). "
                f"Focus on immediate gameplay elements visible in this specific frame and the spatial context from the map. "
                f"Follow the format: {{\"reasoning\": \"detailed step-by-step analysis\", \"action\": X}}"
            )

            agent.add_user_message(frame=game_state_image, user_msg=usr_msg4, local_map=local_map_image)
        else:
            agent.add_user_message(frame=game_state_image, user_msg=usr_msg4, local_map=None)

        action_output, reasoning = agent.generate_response("./")
        
        print(f"Action output: {action_output}, Reasoning: {reasoning}")
        
        agent.add_assistant_message()

        # Create the action object
        action = action_type.new(action_output)

        # Add a batch dimension to the action
        action = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), action)

        # Repeat the action for all environments
        batched_action = repeat_action(action)
            
        # Perform the action in the environment
        rng, _rng = jax.random.split(rng)
        _rng = _rng[None]
        timestep = env.step(timestep, batched_action, _rng)

        # Update rewards and actions
        rewards += timestep.reward.item()
        cumulative_rewards.append(rewards)
        action_list.append(action_output)

        # Render the environment
        env.terra_env.render_obs_pygame(timestep.observation, timestep.info)

        if steps_taken % 20 == 0:
            agent.delete_messages()

        # Update progress
        steps_taken += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"reward": rewards})
    
    # Close progress bar
    progress_bar.close()

    print(f"Rollout complete. Total reward: {rewards}")

    output_dir = os.path.join("experiments", f"AutonomousExcavatorGame_{model_name}")   
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"actions_rewards.csv")
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["actions", "cumulative_rewards"])
        for action, cum_reward in zip(action_list, cumulative_rewards):
            writer.writerow([action, cum_reward])

    print(f"Results saved to {output_file}")

    video_path = os.path.join(output_dir, f"gameplay.mp4")
    save_video(frames, video_path)

