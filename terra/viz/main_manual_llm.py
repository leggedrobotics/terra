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
    USE_PATH = True

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

        actions = path_to_actions(path, initial_direction, 6)
        actions_simple = path_to_actions(simplified_path, initial_direction, 6)
        print("Action list", actions)
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
    initial_target_num = jnp.sum(current_map < 0)  # Count the initial target pixels
    print("Initial target number: ", initial_target_num)
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
            initial_target_num = jnp.sum(current_map < 0)  # Count the initial target pixels
            print("Current target number: ", initial_target_num)

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
                print("Initial Direction: ", initial_direction)

                actions = path_to_actions(path, initial_direction, 6)
                actions_simple = path_to_actions(simplified_path, initial_direction, 6)
                print("Action list", actions)
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

            #current_target_num = jnp.sum(current_map < 0)  # Count the current target pixels
            #print("Current target number: ", current_target_num)
            
            percentage_digging = calculate_digging_percentage(initial_target_num, timestep)

            print(f"Current direction: {base_orientation['direction']}")
            print(f"Bucket status: {bucket_status}")
            print(f"Current position: {start} (y,x)")
            print(f"Nearest target position: {nearest_target} (y,x)")
            print(f"Previous action list: {previous_action}")
            print(f"Percentage of digging left: {percentage_digging:.2f}%")
            
            if USE_PATH:
            #     usr_msg7 = (
            #     f"Analyze this game frame and the provided local map to select the optimal action. "
            #     f"The base of the excavator is currently facing {base_orientation['direction']}. "
            #     f"The bucket is currently {bucket_status}. "
            #     f"The excavator is currently located at {start} (y,x). "
            #     f"The target digging positions are {target_positions} (y,x). "
            #     f"You MUST exactly align the excavator PARALLEL with the trench and then start digging from the furthest trench position (from the current front of the excavator) and then go backward to the closest trench position (going forward do NOT work). "
            #     f"Make sure that the front of the excavator is currently aligned with the trench. "
            #     f"The traversability mask is provided, where 0 indicates obstacles and 1 indicates traversable areas. "
            #     f"The list of the previous actions is {previous_action}. "
            #     f"You can use the action list (but NOT must), computed from the path, to help you decide the next action. The list of actions is {actions_simple}. This list is not exhaustive and you can choose other actions.  In particular, the number of steps forward (action 0) or backward (action 1) are simplfied. "
            #     f"Ensure that the excavator base maintains a safe minimum distance (7 to 13 pixels) from the target area to allow proper overlapping of the light orange area with the purple area for efficient digging. "
            #     f"The purple area becomes green when the soil is excavated. "
            #     f"Avoid repetitive actions (e.g., moving forward or backward) unless necessary. "
            #     #f"Avoid moving too close to the purple area to prevent overlap with the base."
            #     #f"If the previous action was digging and the bucket is still empty, moving backward can be an appropriate action to reposition. You can then try to dig (action 6) in the next action. "
            #     f"Focus on immediate gameplay elements visible in this specific frame and the spatial context from the map. "
            #     f"Follow the format: {{\"reasoning\": \"detailed step-by-step analysis\", \"action\": X}}"
            # )
                
                # usr_msg7 = (
                # f"Analyze this game frame and the provided local map to select the optimal action.\n\n"
                # f"- The excavator base is facing **{base_orientation['direction']}**.\n"
                # f"- The bucket is currently **{bucket_status}**.\n"
                # f"- Current excavator location: **{start}** (y, x).\n"
                # f"- Target digging positions: **{target_positions}** (y, x).\n"
                # f"- The bucket must be **directly facing and aligned with the long edge of the purple trench**, which may require the excavator to **rotate or reposition**.\n"
                # f"- You can overlap the base of the excavator with the **purple trench** to ensure proper alignment.\n"
                # f"- Once aligned, the excavator should **start digging** from the **furthest trench point** (from the front of the excavator) and proceed **backwards**. **Digging forward does not work**.\n"
                # f"- A traversability mask is provided: `0 = obstacle`, `1 = traversable`.\n"
                # f"- Previous actions: **{previous_action}**.\n"
                # f"- A suggested action list (optional) is available: **{actions_simple}**.\n"
                # f"  > This list is not exhaustive. You may choose other actions. Step counts for forward (0) and backward (1) moves are simplified.\n"
                # f"- Maintain a safe distance (**5–13 pixels**) from the target area. This ensures the **light orange** area (digging zone) overlaps properly with the **purple** area.\n"
                # f"- Purple turns **green** once soil is excavated.\n"
                # f"- Avoid unnecessary repetitions (e.g., repeated forward or backward moves).\n"
                # f"- Focus only on **this game frame** and the **local map** context.\n\n"
                # f"Return your answer in the following format:\n"
                # f'{{"reasoning": "step-by-step analysis", "action": X}}'
                # )

                usr_msg7 = (
                f"Analyze this game frame and the provided local map to select the optimal action.\n\n"
                f"A forward or backward move corresponds to 6 pixels of progress.\n"
                f"- The excavator base is facing **{base_orientation['direction']}**.\n"
                f"- The bucket is currently **{bucket_status}**.\n"
                f"- Current excavator location: **{start}** (y, x).\n"
                f"- Target digging positions: **{target_positions}** (y, x).\n"
                f"- IMPORTANT: The bucket must be **directly facing and aligned with the long edge of the purple trench**, which may require the excavator to **rotate or reposition**.\n"
                f"- You can overlap the base of the excavator with the **purple trench** to ensure proper alignment.\n"
                f"- Digging Rule Update:\n"
                f"  - Once aligned, you must **start digging from the furthest end of the trench (relative to the front of the excavator)** and proceed **backwards**. **Digging forward into the trench does not work**.\n"
                f"  - If **facing up**: move forward to reach the **topmost point** of the trench, then dig **downward**.\n"
                f"  - If **facing down**: move forward to reach the **bottommost point**, then dig **upward**.\n"
                f"  - If **facing left**: move forward to reach the **leftmost point**, then dig **rightward**.\n"
                f"  - If **facing right**: move forward to reach the **rightmost point**, then dig **leftward**.\n"
                f"  - Always make sure to **approach the trench from the far end** and **pull back** through it.\n"
                f"  - If the bucket is empty, you can move backward to reposition and then try to dig in the next action. A possible sequence of actions is: 6 (dig), rotate twice (in the best direction!), 6 (deposit), rotate back twice, 1 (backward) (IMPORTANT), 6 (dig), rotate twice, ...\n"
                f"- Double check that the light orange area (target) overlaps with the purple area before digging.\n"
                f"- A traversability mask is provided: `0 = obstacle`, `1 = traversable`.\n"
                f"- Previous actions: **{previous_action}**.\n"
                f"- Do not dig where the target map is not negative or where do you have already dig in one of the previous step\n"
                f"- As a starting point a suggested action list is available: **{actions}**.\n"
                f"- Maintain a safe distance of the excavator (**8–12 pixels**) from the target area. This ensures the **light orange** area (digging zone) overlaps properly with the **purple** area.\n"
                f"- Purple turns **green** once soil is excavated.\n"
                #f"- You can use the value of {count_map_change} to see when a new map is generated. If this value does NOT change from the previous step it is NOT useful to do the do nothing action (-1) Try instead to explore the trench area further to search for missing purple pixels.\n"
                f"- You still have to dig the following percentage of the map: **{percentage_digging:.2f}%**. Do not stop trying to excavate and move until 0% is reached\n"
                f"- Avoid unnecessary repetitions (e.g., repeated forward or backward moves).\n"
                #f"- Focus only on **this game frame** and the **local map** context.\n\n"
                f"Return your answer in the following format:\n"
                f'{{"reasoning": "step-by-step analysis", "action": X}}'
                )

                usr_msg8 = (
                    f"Analyze the current game frame and local map to determine the optimal next action.\n\n"
                    f"- Each forward/backward move advances the excavator by 6 pixels.\n"
                    f"- Excavator base is facing **{base_orientation['direction']}**.\n"
                    f"- Bucket status: **{bucket_status}**.\n"
                    f"- Current location: **{start}** (y, x).\n"
                    f"- Target digging positions: **{target_positions}** (y, x).\n"
                    f"- Previous actions: **{previous_action}**.\n"
                    f"- Suggested actions (NOT compulsory): **{actions}**.\n"
                    f"- Remaining area to dig: **{percentage_digging:.2f}%**.\n\n"

                    f"**DIGGING RULES**\n"
                    f"- Align the bucket **directly facing the long edge** of the purple trench. This may require rotation and repositioning.\n"
                    f"- You may **overlap the base with the purple trench** for proper alignment.\n"
                    f"- Digging must start from the **furthest end** of the trench (relative to the excavator's front) and proceed **backward**.\n"
                    f"  - Facing up → reach **topmost** trench point → dig downward\n"
                    f"  - Facing down → reach **bottommost** → dig upward\n"
                    f"  - Facing left → reach **leftmost** → dig rightward\n"
                    f"  - Facing right → reach **rightmost** → dig leftward\n"
                    f"- Ensure the **orange target area overlaps the purple trench** before digging.\n"
                    f"- Purple turns **green** after successful excavation.\n\n"

                    f"**MOVEMENT GUIDELINES**\n"
                    f"- Avoid repeated forward/backward moves unless repositioning.\n"
                    f"- Maintain **8–12 pixel distance** between the excavator and the trench for best alignment.\n"
                    f"- If the bucket is empty, it’s okay to reposition and try digging again next.\n"
                    f"- A common sequence: 6 (dig), rotate twice, 6 (deposit), rotate back, 1 (backward), 6 (dig), ...\n\n"

                    f"**RESTRICTIONS**\n"
                    f"- Do not dig in areas that are:\n"
                    f"  - Already dug\n"
                    f"  - Not marked as negative in the target map\n"
                    f"- Use the traversability mask (`0 = obstacle`, `1 = traversable`) to guide movement.\n"
                    f"- Continue acting until **0%** of the trench remains undug.\n\n"

                    f"Return your response in this format:\n"
                    f'{{"reasoning": "step-by-step analysis", "action": X}}'
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
                
            
            agent.add_user_message(frame=game_state_image, user_msg=usr_msg8, local_map=local_map_image, traversability_map=traversability_map_np)
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

        # if steps_taken % 5 == 1:
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

