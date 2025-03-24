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
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_a,
    K_d,
    K_e,
    K_r,
    K_i,
    K_o,
    K_k,
    K_l,
    K_SPACE,
    KEYDOWN,
    K_q,
    QUIT,
)
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch

def capture_screen(surface):
    """Captures the current screen and converts it to an image format."""
    img_array = pg.surfarray.array3d(surface)
    img_array = np.rot90(img_array, k=3)  # Rotate if needed
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array


def main():
    print(f"Current working directory: {os.getcwd()}")

    # Load the JSON configuration file
    with open("envs.json", "r") as file:
        game_instructions = json.load(file)

    # Define the environment name for the Autonomous Excavator Game
    environment_name = "AutonomousExcavatorGame"

    # Retrieve the system message for the environment
    system_message = game_instructions.get(
        environment_name,
        "You are a game playing assistant. Provide the best action for the current game state."
    )
    #print(f"System message: {system_message}")
    #print(f"Type of system message: {type(system_message)}")

    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type
    n_envs_x = 1
    n_envs_y = 1
    n_envs = n_envs_x * n_envs_y
    seed = 24
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

    # Render the first frame after resetting the environment
    env.terra_env.render_obs_pygame(timestep.observation, timestep.info)

    # Capture the first frame
    screen = pg.display.get_surface()
    game_state_image = capture_screen(screen)

    #rng, _rng = jax.random.split(rng)
    #_rng = _rng[None]

    agent = Agent(model_name="gpt-4", model="gpt4", system_message=system_message, env=env)

    #def repeat_action(action, n_times=n_envs):
    #    return action_type.new(action.action[None].repeat(n_times, 0))

    # Trigger the JIT compilation
    #timestep = env.step(timestep, repeat_action(action_type.do_nothing()), _rng)
    #end_time = time.time()
    #print(f"Environment started. Compilation time: {end_time - start_time} seconds.")

    agent.add_user_message(frame=game_state_image, user_msg="What action should be taken?")

    # Generate the first action
    action_output, reasoning = agent.generate_response("./")
    print(f"Raw action output: {action_output}, Reasoning: {reasoning}")
    # if action_output is None:
    #     print("Warning: action_output is None!")

    # if action_output is not None:
    #     action = action_type.new(action_output)
    #     action = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), action)

    # else:
    #     print("Using default action: DO_NOTHING (-1)")
    #     action = action_type.new(-1)

    # print(f"Action: {action}")

    # # # Perform the first action in the environment
    # rng, _rng = jax.random.split(rng)
    # _rng = _rng[None]
    # timestep = env.step(timestep, action, _rng)


    if action_output is None:
        print("Using default action: DO_NOTHING (-1)")
        action_output = -1

    # Create the action object
    action = action_type.new(action_output)

    # Add a batch dimension to the action
    action = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), action)

    # Define the repeat_action function
    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))

    # Repeat the action for all environments
    batched_action = repeat_action(action)

    # Debugging: Print the shape of the batched action
    print(f"Batched action: {batched_action}, type: {type(batched_action)}")
    print(f"Batched action.type shape: {batched_action.type.shape}, Batched action.action shape: {batched_action.action.shape}")
    
    # Perform the action in the environment
    rng, _rng = jax.random.split(rng)
    _rng = _rng[None]
    timestep = env.step(timestep, batched_action, _rng)

    # Log the first action
    print(f"First action taken: {action_output}")
    env.terra_env.render_obs_pygame(timestep.observation, timestep.info)
    # pass

    playing = True
    screen = pg.display.get_surface()
    rewards = 0
    cumulative_rewards = []
    action_list = []
    steps_taken = 0
    num_timesteps = 50

    progress_bar = tqdm(total=num_timesteps, desc="Rollout", unit="steps")

    #while playing and steps_taken < num_timesteps:
    while playing:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                playing = False

        game_state_image = capture_screen(screen)

        agent.add_user_message(frame=game_state_image, user_msg="What action should be taken?")
        action_output, reasoning = agent.generate_response("./")

        print(f"Action output: {action_output}")

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

    # output_dir = "./experiments/"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, f"{environment_name}_actions_rewards.csv")
    # with open(output_file, "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["actions", "cumulative_rewards"])
    #     for action, cum_reward in zip(action_list, cumulative_rewards):
    #         writer.writerow([action, cum_reward])

    # print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
