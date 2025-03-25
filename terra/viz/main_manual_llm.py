import time
import jax
import jax.numpy as jnp
import pygame as pg
import cv2
import numpy as np
from .llms import Agent
import json
import os
from tqdm import tqdm
import csv


from pygame.locals import (
    K_q,
    QUIT,
)
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch

def capture_screen(surface):
    """Captures the current screen and converts it to an image format."""
    img_array = pg.surfarray.array3d(surface)
    #img_array = np.rot90(img_array, k=3)  # Rotate if needed
    img_array = np.transpose(img_array, (1, 0, 2))  # Correct rotation

    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array

def save_video(frames, output_path, fps=1):
    """Saves a list of frames as a video."""
    if len(frames) == 0:
        print("No frames to save.")
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

def main():
    print(f"Current working directory: {os.getcwd()}")

    # Load the JSON configuration file
    with open("envs2.json", "r") as file:
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
        rendering_engine="pygame",
        n_envs_x_rendering=n_envs_x,
        n_envs_y_rendering=n_envs_y,
    )

    print("Starting the environment...")
    start_time = time.time()
    env_cfgs = jax.vmap(lambda x: EnvConfig.new())(jnp.arange(n_envs))
    rng, _rng = jax.random.split(rng)
    _rng = _rng[None]
    timestep = env.reset(env_cfgs, _rng)

    agent = Agent(model_name="gpt-4", model="gpt4", system_message=system_message, env=env)
    #agent = Agent(model_name="gemini-1.5-flash-latest", model="gemini", system_message=system_message, env=env)

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
    num_timesteps = 100
    frames = []

    progress_bar = tqdm(total=num_timesteps, desc="Rollout", unit="steps")

    while playing and steps_taken < num_timesteps:
    #while playing:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                playing = False

        game_state_image = capture_screen(screen)
        frames.append(game_state_image)

        agent.add_user_message(frame=game_state_image, user_msg="What action should be taken?")
        action_output, reasoning = agent.generate_response("./")

        print(f"Action output: {action_output}, Reasoning: {reasoning}")

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

        # Update progress
        steps_taken += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"reward": rewards})
    
    # Close progress bar
    progress_bar.close()

    print(f"Rollout complete. Total reward: {rewards}")

    output_dir = "./experiments/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{environment_name}_actions_rewards.csv")
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["actions", "cumulative_rewards"])
        for action, cum_reward in zip(action_list, cumulative_rewards):
            writer.writerow([action, cum_reward])

    print(f"Results saved to {output_file}")

    video_path = os.path.join(output_dir, f"{environment_name}_gameplay.mp4")
    save_video(frames, video_path)

if __name__ == "__main__":
    main()
