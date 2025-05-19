import base64
import cv2
import numpy as np
import jax
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import jax.numpy as jnp
from terra.viz.llms_adk import *
from terra.viz.a_star import compute_path, simplify_path
from terra.viz.llms_utils import *
import csv

from utils.models import get_model_ready
import json
from terra.env import TerraEnvBatch
import jax.numpy as jnp

def print_stats(
    stats,
):
    episode_done_once = stats["episode_done_once"]
    episode_length = stats["episode_length"]
    path_efficiency = stats["path_efficiency"]
    workspaces_efficiency = stats["workspaces_efficiency"]
    coverage = stats["coverage"]

    completion_rate = 100 * episode_done_once.sum() / len(episode_done_once)

    print("\nStats:\n")
    print(f"Completion: {completion_rate:.2f}%")
    # print(f"First episode length average: {episode_length.mean()}")
    # print(f"First episode length min: {episode_length.min()}")
    # print(f"First episode length max: {episode_length.max()}")
    print(
        f"Path efficiency: {path_efficiency['mean']:.2f} ({path_efficiency['std']:.2f})"
    )
    print(
        f"Workspaces efficiency: {workspaces_efficiency['mean']:.2f} ({workspaces_efficiency['std']:.2f})"
    )
    print(f"Coverage: {coverage['mean']:.2f} ({coverage['std']:.2f})")
def encode_image(cv_image):
    _, buffer = cv2.imencode(".jpg", cv_image)
    return base64.b64encode(buffer).decode("utf-8")

async def call_agent_async_master(query: str, image, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    #print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    #content = types.Content(role='user', parts=[types.Part(text=query)])
    text = types.Part.from_text(text=query)
    parts = [text]
    if image is not None:
        # Convert the image to a format suitable for ADK
        image_data = encode_image(image)

        content_image = types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
        parts.append(content_image)

    user_content = types.Content(role='user', parts=parts)
    
    final_response_text = "Agent did not produce a final response." # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_content):
        # You can uncomment the line below to see *all* events during execution
        print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

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

def init_llms(llm_model_key, llm_model_name, USE_PATH, config, env, n_envs, APP_NAME, USER_ID, SESSION_ID):
    if llm_model_key == "gpt":
        llm_model_name_extended = "openai/{}".format(llm_model_name)
    elif llm_model_key == "claude":
        llm_model_name_extended = "anthropic/{}".format(llm_model_name)
    else:
        llm_model_name_extended =  llm_model_name
    
    print("Using model: ", llm_model_name_extended)

    description_master = "You are a master agent controlling an excavator. Observe the state. " \
    "Decide if you should delegate digging tasks to a " \
    "specialized RL agent (respond with 'delegate_to_rl') or to delegate the task to a" \
    "specialized LLM agent (respond with 'delegate_to_llm')."

    system_message_master = "You are a master agent controlling an excavator. Observe the state. " \
    "Decide if you should delegate digging tasks to a " \
    "specialized RL agent (respond with 'delegate_to_rl') or to delegate the task to a" \
    "specialized LLM agent (respond with 'delegate_to_llm')."

    description_excavator = "You are an excavator agent. You can control the excavator to dig and move."

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
    system_message_excavator = game_instructions.get(
        environment_name,
        "You are a game playing assistant. Provide the best action for the current game state."
    )

    if llm_model_key == "gemini":
        llm_excavator_agent = Agent(
            name="ExcavatorAgent",
            model=llm_model_name_extended,
            description=description_excavator,
            instruction=system_message_excavator,
        )

        llm_master_agent = Agent(
            name="MasterAgent",
            model=llm_model_name_extended,
            description=description_master,
            instruction=system_message_master,
            #sub_agents=[llm_excavator_agent],
        )
    else:
        llm_excavator_agent = Agent(
            name="ExcavatorAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description=description_excavator,
            instruction=system_message_excavator,
        )

        llm_master_agent = Agent(
            name="MasterAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description=description_master,
            instruction=system_message_master,
            #sub_agents=[llm_excavator_agent],
        )
    
    
    print("Master Agent initialized.")
    print("Excavator Agent initialized.")



    session_service = InMemorySessionService()



    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    print("Session created. App: ", APP_NAME, " User ID: ", USER_ID, " Session ID: ", SESSION_ID)
    
    runner = Runner(
        agent=llm_master_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    print(f"Runner initialized for agent {runner.agent.name}.")
    

    APP_NAME_2 = APP_NAME + "_excavator"
    SESSION_ID_2 = SESSION_ID + "_excavator"
    USER_ID_2 = USER_ID + "_excavator"     
        
    session_service_2 = InMemorySessionService()
    session_2 = session_service_2.create_session(
        app_name=APP_NAME_2,
        user_id=USER_ID_2,
        session_id=SESSION_ID_2,
    )
    runner_2 = Runner(
        agent=llm_excavator_agent,
        app_name=APP_NAME_2,
        session_service=session_service_2,
    )

    llm_query = LLM_query(
        model_name=llm_model_name_extended,
        model=llm_model_key,
        system_message=system_message_excavator,
        env=env,
        session_id=SESSION_ID_2,
        runner=runner_2,
        user_id=USER_ID_2,
    )

    prev_actions = None
    if config:
        prev_actions = jnp.zeros(
            (n_envs, config.num_prev_actions),
            dtype=jnp.int32
        )
    else:
        print("Warning: rl_config is None, prev_actions will not be initialized.")
    
    return llm_query, runner, prev_actions, system_message_master

def compute_action_list(timestep,env):
        # Compute the path
        start, target_positions = extract_positions(timestep.state)
        target = find_nearest_target(start, target_positions)
        path, path2, _ = compute_path(timestep.state, start, target)


        initial_orientation = extract_base_orientation(timestep.state)
        initial_direction = initial_orientation["direction"]

        actions = path_to_actions(path, initial_direction, 6)
        print("Action list", actions)

        if path:
            game = env.terra_env.rendering_engine
            game.path = path
        else:
            print("No path found.")

        return actions

def save_csv(output_file, action_list, cumulative_rewards):
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

def combine_maps_into_grid(obs):
    """
    Combine four maps from an observation dictionary into a 2x2 grid structure.
    
    Args:
        obs: Dictionary containing maps of different types
        
    Returns:
        Dictionary with the same keys but maps combined into larger 2x2 grids
    """
    result = {}
    
    # For each key in the observation dictionary
    for key, maps in obs.items():
        # Check if we have 4 maps as required for a 2x2 grid
        if len(maps) != 4:
            raise ValueError(f"Expected 4 maps for key {key}, but got {len(maps)}")
        
        #print(maps, key)
        
        if key == "agent_height" or key == "agent_state" or key == "agent_width" or key == "local_map_action_neg" or key == "local_map_action_pos" \
        or key == "local_map_dumpability" or key == "local_map_obstacles" or key == "local_map_target_neg" or key == "local_map_target_pos":
            #combined_map = np.zeros((1, maps[0].shape[0]), dtype=maps[0].dtype)
            # Make sure to preserve the array structure by using np.array() explicitly
            print(maps)
            if key == "agent_height" or key == "agent_width":
                combined_map = jnp.zeros((1,1), dtype=maps[0].dtype)
                combined_map = combined_map.at[0].set(maps[0])
            # print(key,type(combined_map))
            
            # Ensure special fields like agent_height maintain array structure with shape [9]

            result[key] = combined_map
            continue


        # Get dimensions of each map
        map_shape = maps[0].shape
        #print("Map shape: ", map_shape)
        
        # Assuming all maps have the same dimensions
        height, width = map_shape
        
        # Create a new array for the combined map
        combined_map = jnp.zeros((1,2 * height, 2 * width), dtype=maps[0].dtype)
        # print(key, type(combined_map))
        
        # Place the maps in a 2x2 grid
        # Top-left
        #ombined_map[0:height, 0:width] = maps[0]
        combined_map = combined_map.at[0,0:height, 0:width].set(maps[0])
        # Top-right
        # combined_map[0:height, width:2*width] = maps[1]
        combined_map = combined_map.at[0,0:height, width:2*width].set(maps[1])
        # Bottom-left
        # combined_map[height:2*height, 0:width] = maps[2]
        combined_map = combined_map.at[0,height:2*height, 0:width].set(maps[2])
        # Bottom-right
        # combined_map[height:2*height, width:2*width] = maps[3]
        combined_map = combined_map.at[0,height:2*height, width:2*width].set(maps[3])
        
        # Store the combined map in the result dictionary
        result[key] = combined_map
    
    return result
def extract_first_row(input_dict):
    """
    Extract only the first row/entry for each element in the dictionary.
    
    Args:
        input_dict: Dictionary containing arrays with multiple rows
        
    Returns:
        Dictionary with the same keys but only the first row of each array
    """
    result = {}
    
    for key, value in input_dict.items():

                result[key] = value[0:1]  # Keep as 2D array with just one row

    
    return result


class TerraEnvBatchWithMapOverride(TerraEnvBatch):
    def reset_with_map_override(self, env_cfgs, keys, custom_pos=None, custom_angle=None,
                               target_map_override=None, padding_mask_override=None,
                               traversability_mask_override=None, dumpability_mask_override=None, 
                               dumpability_mask_init_override=None, action_map_override=None):
        """
        Custom reset that first does a normal reset, then manually overrides the maps in the state.
        """
        # Do a normal reset first
        timestep = super().reset(env_cfgs, keys, custom_pos, custom_angle)

        # Now manually override the maps in the state
        if padding_mask_override is not None:
            state = timestep.state
            updated_world = state.world._replace(
                padding_mask=state.world.padding_mask._replace(
                    map=jnp.array([padding_mask_override])
                )
            )
            updated_state = state._replace(world=updated_world)
            timestep = timestep._replace(state=updated_state)
        
        if target_map_override is not None:
            state = timestep.state
            updated_world = state.world._replace(
                target_map=state.world.target_map._replace(
                    map=jnp.array([target_map_override])  
                )
            )
            updated_state = state._replace(world=updated_world)
            timestep = timestep._replace(state=updated_state)

        if action_map_override is not None:
            state = timestep.state
            updated_world = state.world._replace(
                action_map=state.world.action_map._replace(
                    map=jnp.array([action_map_override])
                )
            )
            updated_state = state._replace(world=updated_world)
            timestep = timestep._replace(state=updated_state)
        
        if traversability_mask_override is not None:
            state = timestep.state
            updated_world = state.world._replace(
                traversability_mask=state.world.traversability_mask._replace(
                    map=jnp.array([traversability_mask_override])
                )
            )
            updated_state = state._replace(world=updated_world)
            timestep = timestep._replace(state=updated_state)
        
        if dumpability_mask_override is not None:
            state = timestep.state
            updated_world = state.world._replace(
                dumpability_mask=state.world.dumpability_mask._replace(
                    map=jnp.array([dumpability_mask_override])
                )
            )
            updated_state = state._replace(world=updated_world)
            timestep = timestep._replace(state=updated_state)
        
        if dumpability_mask_init_override is not None:
            state = timestep.state
            updated_world = state.world._replace(
                dumpability_mask_init=state.world.dumpability_mask_init._replace(
                    map=jnp.array([dumpability_mask_init_override]) 
                )
            )
            updated_state = state._replace(world=updated_world)
            timestep = timestep._replace(state=updated_state)
        
        return timestep

def create_sub_task_target_map(global_target_map_data: jnp.ndarray,
                              region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 target map for an RL agent's sub-task.
    
    Retains both `-1` values (dig targets) and `1` values (dump targets) from 
    the specified region in the global map. Everything outside the region is set to 0 (free).
    
    Args:
        global_target_map_data: Full 64x64 target map (1: dump, 0: free, -1: dig).
        region_coords: (y_start, x_start, y_end, x_end), inclusive bounds.
    
    Returns:
        A new 64x64 map with `-1`s and `1`s from the region; everything else is 0.
    """
    y_start, x_start, y_end, x_end = region_coords
    
    # Initialize a 64x64 map with all zeros (free space)
    sub_task_map = jnp.zeros_like(global_target_map_data)
    
    # Define slice object for region
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    
    # Extract region from global map and place it directly into the sub_task map
    # This preserves both -1 (dig) and 1 (dump) values within the region
    sub_task_map = sub_task_map.at[region_slice].set(global_target_map_data[region_slice])
    
    return sub_task_map

def create_sub_task_action_map(action_map_data: jnp.ndarray,
                               region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 action map for a sub-task, preserving only actions that occurred
    inside the specified region. Outside the region, all values are reset to 0 (free).

    Args:
        action_map_data: Full 64x64 action map 
                         (-1: dug, 0: free, >0: dumped).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A new 64x64 map with only the region's actions preserved, all else is 0.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Initialize output map with zeros (free)
    sub_task_action_map = jnp.zeros_like(action_map_data)

    # Define region slice
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Extract region from input map
    region_data = action_map_data[region_slice]

    # Set region into the new map
    sub_task_action_map = sub_task_action_map.at[region_slice].set(region_data)

    return sub_task_action_map

def create_sub_task_padding_mask(padding_mask_data: jnp.ndarray,
                                 region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 padding mask for a sub-task.

    Inside the region: preserves original traversability (0 or 1).
    Outside the region: sets everything to 1 (non-traversable).

    Args:
        padding_mask_data: Full 64x64 mask (0: traversable, 1: non-traversable).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A 64x64 mask with only the region preserved; the rest is 1.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Initialize everything as non-traversable (1)
    sub_task_mask = jnp.ones_like(padding_mask_data)

    # Define slice for region
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Copy the original values only inside the region
    region_data = padding_mask_data[region_slice]
    sub_task_mask = sub_task_mask.at[region_slice].set(region_data)

    return sub_task_mask

def create_sub_task_traversability_mask(traversability_mask_data: jnp.ndarray,
                                        region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 traversability mask for a sub-task.

    Inside the region: preserves original values (-1: agent, 0: traversable, 1: non-traversable).
    Outside the region: sets everything to 1 (non-traversable).

    Args:
        traversability_mask_data: Full 64x64 mask 
                                  (-1: agent, 0: traversable, 1: non-traversable).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A 64x64 mask with only the region preserved; the rest is 1.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Start with a mask where everything is non-traversable (1)
    sub_task_mask = jnp.ones_like(traversability_mask_data)

    # Define region slice
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Copy the original values from the region (can include -1, 0, 1)
    region_data = traversability_mask_data[region_slice]
    sub_task_mask = sub_task_mask.at[region_slice].set(region_data)

    return sub_task_mask

def create_sub_task_dumpability_mask(dumpability_mask_data: jnp.ndarray,
                                     region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64×64 dumpability mask for a sub-task.

    Inside the region: preserves original values (1: can dump, 0: can't dump).
    Outside the region: sets everything to 0 (can't dump).

    Args:
        dumpability_mask_data: Full 64×64 mask (1: can dump, 0: can't).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A 64×64 mask with only the region preserved; the rest is 0.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Initialize as all 0 (can't dump)
    sub_task_mask = jnp.zeros_like(dumpability_mask_data)

    # Define region slice
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Copy over the original dumpability values inside the region
    region_data = dumpability_mask_data[region_slice]
    sub_task_mask = sub_task_mask.at[region_slice].set(region_data)

    return sub_task_mask

def verify_maps_override(timestep, sub_task_target_map_data, sub_task_traversability_mask_data, 
                         sub_task_padding_mask_data, sub_task_dumpability_mask_data,
                         sub_task_dumpability_init_mask_data, sub_task_action_map_data):
    """
    Verifies if the maps in the timestep state match the expected data.
    
    Parameters:
    - timestep: The timestep object containing state information
    - sub_task_target_map_data: Expected target map data
    - sub_task_traversability_mask_data: Expected traversability mask data
    - sub_task_padding_mask_data: Expected padding mask data
    - sub_task_dumpability_mask_data: Expected dumpability mask data
    - sub_task_dumpability_init_mask_data: Expected initial dumpability mask data
    - sub_task_action_map_data: Expected action map data
    
    Returns:
    - bool: True if all maps match, False otherwise
    """
    import numpy as np
    
    # Extract current maps from timestep
    current_target_map = timestep.state.world.target_map.map[0]
    current_traversability_mask = timestep.state.world.traversability_mask.map[0]
    current_padding_mask = timestep.state.world.padding_mask.map[0]
    current_dumpability_mask = timestep.state.world.dumpability_mask.map[0]
    current_dumpability_mask_init = timestep.state.world.dumpability_mask_init.map[0]
    current_action_map = timestep.state.world.action_map.map[0]

    # Check if they match what we passed in
    target_match = np.array_equal(current_target_map, sub_task_target_map_data)
    traversability_match = np.array_equal(current_traversability_mask, sub_task_traversability_mask_data)
    padding_match = np.array_equal(current_padding_mask, sub_task_padding_mask_data)
    dumpability_match = np.array_equal(current_dumpability_mask, sub_task_dumpability_mask_data)
    dumpability_match_init = np.array_equal(current_dumpability_mask_init, sub_task_dumpability_init_mask_data)
    action_match = np.array_equal(current_action_map, sub_task_action_map_data)

    # Print verification results
    # print(f"Target map properly overridden: {target_match}")
    # print(f"Traversability mask properly overridden: {traversability_match}")
    # print(f"Padding mask properly overridden: {padding_match}")
    # print(f"Dumpability mask properly overridden: {dumpability_match}")
    # print(f"Dumpability mask init properly overridden: {dumpability_match_init}")
    # print(f"Action map properly overridden: {action_match}")

    # Check if all maps match
    all_match = (target_match and traversability_match and padding_match and 
                dumpability_match and dumpability_match_init and action_match)
    
    if not all_match:
        print("WARNING: Maps were not properly overridden!")
    
    return all_match