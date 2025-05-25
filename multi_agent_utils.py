import base64
import cv2
import numpy as np
import jax
# from google.adk.agents import Agent
# from google.adk.models.lite_llm import LiteLlm
# from google.adk.sessions import InMemorySessionService
# from google.adk.runners import Runner
# from google.genai import types
import jax.numpy as jnp
from terra.viz.llms_adk import *
from terra.viz.a_star import compute_path, simplify_path
from terra.viz.llms_utils import *
import csv

from utils.models import get_model_ready
import json
from terra.env import TerraEnvBatch
import jax.numpy as jnp
import ast

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

    # description_master = "You are a master agent controlling an excavator. Observe the state. " \
    # "Decide if you should delegate digging tasks to a " \
    # "specialized RL agent (respond with 'delegate_to_rl') or to delegate the task to a" \
    # "specialized LLM agent (respond with 'delegate_to_llm')."

    # system_message_master = "You are a master agent controlling an excavator. Observe the state. " \
    # "Decide if you should delegate digging tasks to a " \
    # "specialized RL agent (respond with 'delegate_to_rl') or to delegate the task to a" \
    # "specialized LLM agent (respond with 'delegate_to_llm')."
    description_master = "You are a master excavation coordinator responsible for optimizing excavation operations on a site map. Your task is to analyze the given terrain and intelligently partition it into optimal regions for multiple excavator deployments.\n\n" \


    system_message_master = "You are a master excavation coordinator responsible for optimizing excavation operations on a site map. Your task is to analyze the given terrain and intelligently partition it into optimal regions for multiple excavator deployments.\n\n" \
    "IMPORTANT: All maps are 64x64 in size.\n\n" \
    "GUIDELINES FOR PARTITIONING:\n" \
    "1. Analyze the state of the map carefully, considering terrain features, obstacles, and excavation requirements\n" \
    "2. Create efficient partitions that maximize excavator productivity and minimize travel time\n" \
    "3. Ensure each partition has adequate space for the excavator to maneuver\n" \
    "4. Designate appropriate soil deposit areas within each partition or create shared deposit zones if more efficient\n" \
    "5. Position starting points strategically to minimize initial travel time\n" \
    "6. Consider terrain complexity when determining partition size - more complex areas may require smaller partitions\n\n" \
    "USE AT MOST 2 PARTITIONS TO OPTIMIZE EXCAVATION OPERATIONS. " \
    "If you see multiple trenches, you should create a partition for each trench. " \
    "RESPONSE FORMAT:\n" \
    "Respond with a JSON list of partition objects, each containing:\n" \
    "- 'id': Unique numeric identifier for each partition (starting from 0)\n" \
    "- 'region_coords': MUST BE A TUPLE with parentheses, NOT an array with brackets: (y_start, x_start, y_end, x_end)\n" \
    "- 'start_pos': MUST BE A TUPLE with parentheses, NOT an array with brackets: (y, x)\n" \
    "- 'start_angle': Always use 0 degrees for initial orientation\n" \
    "- 'status': Set to 'pending' for all new partitions\n\n" \
    "CRITICAL: You MUST use Python tuple notation with parentheses () for coordinates, NOT arrays with square brackets []. Failure to use tuple notation will result in errors.\n\n" \
    "CORRECT FORMAT (with tuples):\n" \
    "[{'id': 0, 'region_coords': (0, 0, 31, 31), 'start_pos': (16, 16), 'start_angle': 0, 'status': 'pending'}]\n\n" \
    "INCORRECT FORMAT (with arrays):\n" \
    "[{'id': 0, 'region_coords': [0, 0, 31, 31], 'start_pos': [16, 16], 'start_angle': 0, 'status': 'pending'}]\n\n" \
    "Example response for partitioning a 64x64 map into 4 equal quadrants (USING TUPLES, NOT ARRAYS):\n" \
    "[{'id': 0, 'region_coords': (0, 0, 31, 31), 'start_pos': (16, 16), 'start_angle': 0, 'status': 'pending'}, " \
    "{'id': 1, 'region_coords': (0, 32, 31, 63), 'start_pos': (16, 48), 'start_angle': 0, 'status': 'pending'}, " \
    "{'id': 2, 'region_coords': (32, 0, 63, 31), 'start_pos': (48, 16), 'start_angle': 0, 'status': 'pending'}, " \
    "{'id': 3, 'region_coords': (32, 32, 63, 63), 'start_pos': (48, 48), 'start_angle': 0, 'status': 'pending'}]\n\n" \
    "NOTE: Always return a list of partitions even if only creating a single partition." \
    "Very important. Ensure each partition has sufficient space for both excavation and soil deposit operations. " \
    "Make sure to also consider a lot of space in the partition for moving the excavator to avoid getting stuck" \
    "REMEMBER TO USE TUPLES (PARENTHESES) FOR ALL COORDINATES."

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
class TerraEnvBatchWithMapOverride(TerraEnvBatch):
    """
    Extended version of TerraEnvBatch that supports map overrides.
    This class enables working with subsets of larger maps.
    """
    def reset_with_map_override(self, env_cfgs, rngs, custom_pos=None, custom_angle=None,
                                target_map_override=None, traversability_mask_override=None,
                                padding_mask_override=None, dumpability_mask_override=None,
                                dumpability_mask_init_override=None, action_map_override=None,
                                dig_map_override=None, agent_config_override=None):
        """
        Reset the environment with custom map overrides.
        
        Args:
            env_cfgs: Environment configurations
            rngs: Random number generators
            custom_pos: Custom initial position
            custom_angle: Custom initial angle
            target_map_override: Override for target map
            traversability_mask_override: Override for traversability mask
            padding_mask_override: Override for padding mask
            dumpability_mask_override: Override for dumpability mask
            dumpability_mask_init_override: Override for initial dumpability mask
            action_map_override: Override for action map
            dig_map_override: Override for dig map
            
        Returns:
            Initial timestep
        """
        # Print the shape of the override maps for debugging
        # print("\nOverride Map Shapes:")
        # print(f"Target Map Override Shape: {target_map_override.shape if target_map_override is not None else None}")
        # print(f"Traversability Mask Override Shape: {traversability_mask_override.shape if traversability_mask_override is not None else None}")
        # print(f"Padding Mask Override Shape: {padding_mask_override.shape if padding_mask_override is not None else None}")
        # print(f"Dumpability Mask Override Shape: {dumpability_mask_override.shape if dumpability_mask_override is not None else None}")
        # print(f"Dumpability Init Mask Override Shape: {dumpability_mask_init_override.shape if dumpability_mask_init_override is not None else None}")
        # print(f"Action Map Override Shape: {action_map_override.shape if action_map_override is not None else None}")
        # print(f"Dig Map Override Shape: {dig_map_override.shape if dig_map_override is not None else None}")
        
        # Determine the new edge length based on overrides
        new_edge_length = None
        if target_map_override is not None:
            if len(target_map_override.shape) == 2:
                new_edge_length = target_map_override.shape[0]  # Use the first dimension
            else:
                new_edge_length = target_map_override.shape[1]  # Use the second dimension for batched maps
        elif action_map_override is not None:
            if len(action_map_override.shape) == 2:
                new_edge_length = action_map_override.shape[0]
            else:
                new_edge_length = action_map_override.shape[1]
    
        # If we have a new edge length, update the env_cfg
        # Update the env_cfg with new map size and agent config if provided
        if new_edge_length is not None or agent_config_override is not None:
            
            # Update maps config if new edge length is provided
            if new_edge_length is not None:
                updated_maps_config = env_cfgs.maps._replace(
                    edge_length_px=jnp.array([new_edge_length], dtype=jnp.int32)
                )
            else:
                updated_maps_config = env_cfgs.maps
            
            # Update agent config if override is provided
            if agent_config_override is not None:
                print(f"Overriding agent config: {agent_config_override}")
                updated_agent_config = env_cfgs.agent._replace(**agent_config_override)
            else:
                updated_agent_config = env_cfgs.agent
        
            # Update the env_cfgs with the new configurations
            env_cfgs = env_cfgs._replace(
                maps=updated_maps_config,
                agent=updated_agent_config
            )
        
            print(f"Updated env_cfgs - edge_length_px: {env_cfgs.maps.edge_length_px}, agent height: {env_cfgs.agent.height}, agent width: {env_cfgs.agent.width}")
        
        # First reset with possibly updated env_cfgs
        timestep = self.reset(env_cfgs, rngs, custom_pos, custom_angle)
        
        # Print the original shapes before override
        # print("\nOriginal Map Shapes:")
        # print(f"Target Map Shape: {timestep.state.world.target_map.map.shape}")
        # print(f"Action Map Shape: {timestep.state.world.action_map.map.shape}")
        # print(f"Environment Config: {timestep.state.env_cfg if hasattr(timestep.state, 'env_cfg') else 'No env_cfg in state'}")

        # Then override maps if provided - use completely new arrays
        if target_map_override is not None:
            # Add batch dimension if needed
            if len(target_map_override.shape) == 2:
                target_map_override = target_map_override[None, ...]
            timestep = timestep._replace(
                state=timestep.state._replace(
                    world=timestep.state.world._replace(
                        target_map=timestep.state.world.target_map._replace(
                            map=target_map_override
                        )
                    )
                )
            )
        
        if traversability_mask_override is not None:
            if len(traversability_mask_override.shape) == 2:
                traversability_mask_override = traversability_mask_override[None, ...]
            timestep = timestep._replace(
                state=timestep.state._replace(
                    world=timestep.state.world._replace(
                        traversability_mask=timestep.state.world.traversability_mask._replace(
                            map=traversability_mask_override
                        )
                    )
                )
            )
        
        if padding_mask_override is not None:
            if len(padding_mask_override.shape) == 2:
                padding_mask_override = padding_mask_override[None, ...]
            timestep = timestep._replace(
                state=timestep.state._replace(
                    world=timestep.state.world._replace(
                        padding_mask=timestep.state.world.padding_mask._replace(
                            map=padding_mask_override
                        )
                    )
                )
            )
        
        if dumpability_mask_override is not None:
            if len(dumpability_mask_override.shape) == 2:
                dumpability_mask_override = dumpability_mask_override[None, ...]
            timestep = timestep._replace(
                state=timestep.state._replace(
                    world=timestep.state.world._replace(
                        dumpability_mask=timestep.state.world.dumpability_mask._replace(
                            map=dumpability_mask_override
                        )
                    )
                )
            )
        
        if dumpability_mask_init_override is not None:
            if len(dumpability_mask_init_override.shape) == 2:
                dumpability_mask_init_override = dumpability_mask_init_override[None, ...]
            timestep = timestep._replace(
                state=timestep.state._replace(
                    world=timestep.state.world._replace(
                        dumpability_mask_init=timestep.state.world.dumpability_mask_init._replace(
                            map=dumpability_mask_init_override
                        )
                    )
                )
            )
        
        if action_map_override is not None:
            if len(action_map_override.shape) == 2:
                action_map_override = action_map_override[None, ...]
            timestep = timestep._replace(
                state=timestep.state._replace(
                    world=timestep.state.world._replace(
                        action_map=timestep.state.world.action_map._replace(
                            map=action_map_override
                        )
                    )
                )
            )

        if dig_map_override is not None:
            if len(dig_map_override.shape) == 2:
                dig_map_override = dig_map_override[None, ...]
            timestep = timestep._replace(
                state=timestep.state._replace(
                    world=timestep.state.world._replace(
                        dig_map=timestep.state.world.dig_map._replace(
                            map=dig_map_override
                        )
                    )
                )
            )
        
        # Print the new shapes after override
        # print("\nAfter Override Map Shapes:")
        # print(f"Target Map Shape: {timestep.state.world.target_map.map.shape}")
        # print(f"Action Map Shape: {timestep.state.world.action_map.map.shape}")
        
      
        # Update the env_cfg in the timestep state to ensure consistency
        if new_edge_length is not None or agent_config_override is not None:
            # Update the state's env_cfg if it exists
            if hasattr(timestep.state, 'env_cfg'):
                state_env_cfg = timestep.state.env_cfg
                
                if new_edge_length is not None:
                    state_env_cfg = state_env_cfg._replace(
                        maps=state_env_cfg.maps._replace(
                            edge_length_px=jnp.array([new_edge_length], dtype=jnp.int32)
                        )
                    )
                
                if agent_config_override is not None:
                    state_env_cfg = state_env_cfg._replace(
                        agent=state_env_cfg.agent._replace(**agent_config_override)
                    )
                
                timestep = timestep._replace(
                    state=timestep.state._replace(env_cfg=state_env_cfg)
                )
        
            # Update the timestep's env_cfg if it exists at the top level
            if hasattr(timestep, 'env_cfg'):
                timestep_env_cfg = timestep.env_cfg
                
                if new_edge_length is not None:
                    timestep_env_cfg = timestep_env_cfg._replace(
                        maps=timestep_env_cfg.maps._replace(
                            edge_length_px=jnp.array([new_edge_length], dtype=jnp.int32)
                        )
                    )
                
                if agent_config_override is not None:
                    timestep_env_cfg = timestep_env_cfg._replace(
                        agent=timestep_env_cfg.agent._replace(**agent_config_override)
                    )
        # We need to manually update the observation to match the new maps
        updated_obs = dict(timestep.observation)


        # Update all map-related observations
        if target_map_override is not None and 'target_map' in updated_obs:
            updated_obs['target_map'] = target_map_override
        
        if action_map_override is not None and 'action_map' in updated_obs:
            updated_obs['action_map'] = action_map_override
        
        if dumpability_mask_override is not None and 'dumpability_mask' in updated_obs:
            updated_obs['dumpability_mask'] = dumpability_mask_override
        
        if traversability_mask_override is not None and 'traversability_mask' in updated_obs:
            updated_obs['traversability_mask'] = traversability_mask_override
        
        if padding_mask_override is not None and 'padding_mask' in updated_obs:
            updated_obs['padding_mask'] = padding_mask_override
            
        if dumpability_mask_init_override is not None and 'dumpability_mask_init' in updated_obs:
            updated_obs['dumpability_mask_init'] = dumpability_mask_init_override
            
        if dig_map_override is not None and 'dig_map' in updated_obs:
            updated_obs['dig_map'] = dig_map_override
        
        # Force a reset of the environment to update the observation
        # Note: We already did a reset earlier, this is just for completeness
        #rngs_new = jax.random.split(rngs[0], 1)
        
        # Return the timestep with the updated observation
        timestep = timestep._replace(observation=updated_obs)
        
        # print(f"\nFinal timestep map shapes:")
        # print(f"Target Map Shape: {timestep.state.world.target_map.map.shape}")
        # print(f"Action Map Shape: {timestep.state.world.action_map.map.shape}")
        # if hasattr(timestep.state, 'env_cfg'):
        #     print(f"State env_cfg edge_length_px: {timestep.state.env_cfg.maps.edge_length_px}")
        # if hasattr(timestep, 'env_cfg'):
        #     print(f"Timestep env_cfg edge_length_px: {timestep.env_cfg.maps.edge_length_px}")
    
       #print(timestep.state.env_cfg)
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

def extract_sub_task_target_map(global_target_map_data: jnp.ndarray,
                               region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Extracts a target map for an RL agent's sub-task without padding.
    
    Retains both `-1` values (dig targets) and `1` values (dump targets) from 
    the specified region in the global map.
    
    Args:
        global_target_map_data: Full 64x64 target map (1: dump, 0: free, -1: dig).
        region_coords: (y_start, x_start, y_end, x_end), inclusive bounds.
    
    Returns:
        A map of shape (y_end-y_start+1, x_end-x_start+1) with `-1`s and `1`s from the region.
    """
    y_start, x_start, y_end, x_end = region_coords
    
    # Define slice object for region
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    
    # Extract region from global map
    sub_task_map = global_target_map_data[region_slice]
    
    return sub_task_map

def extract_sub_task_action_map(action_map_data: jnp.ndarray,
                                region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Extracts an action map for a sub-task without padding, preserving only actions that occurred
    inside the specified region.

    Args:
        action_map_data: Full 64x64 action map 
                         (-1: dug, 0: free, >0: dumped).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A map of shape (y_end-y_start+1, x_end-x_start+1) with the region's actions.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Define region slice
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Extract region from input map
    sub_task_action_map = action_map_data[region_slice]

    return sub_task_action_map

def extract_sub_task_padding_mask(padding_mask_data: jnp.ndarray,
                                  region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Extracts a padding mask for a sub-task without padding.

    Args:
        padding_mask_data: Full 64x64 mask (0: traversable, 1: non-traversable).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A mask of shape (y_end-y_start+1, x_end-x_start+1) with the original region values.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Define slice for region
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Extract the original values from the region
    sub_task_mask = padding_mask_data[region_slice]

    return sub_task_mask

def extract_sub_task_traversability_mask(traversability_mask_data: jnp.ndarray,
                                         region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Extracts a traversability mask for a sub-task without padding.

    Args:
        traversability_mask_data: Full 64x64 mask 
                                  (-1: agent, 0: traversable, 1: non-traversable).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A mask of shape (y_end-y_start+1, x_end-x_start+1) with the original region values.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Define region slice
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Extract the original values from the region
    sub_task_mask = traversability_mask_data[region_slice]

    return sub_task_mask

def extract_sub_task_dumpability_mask(dumpability_mask_data: jnp.ndarray,
                                      region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Extracts a dumpability mask for a sub-task without padding.

    Args:
        dumpability_mask_data: Full 64×64 mask (1: can dump, 0: can't).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A mask of shape (y_end-y_start+1, x_end-x_start+1) with the original region values.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Define region slice
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Extract the original dumpability values from the region
    sub_task_mask = dumpability_mask_data[region_slice]

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

def extract_python_format_data(llm_response_text):
    """
    Extracts Python-formatted data from LLM response, preserving tuples.
    
    Args:
        llm_response_text (str): The raw text response from the LLM
        
    Returns:
        list: The parsed Python list with tuples preserved
        
    Raises:
        ValueError: If no valid Python data could be extracted
    """
    # First, check if we have a code block and extract its content
    code_block_pattern = r'```(?:json|python)?\s*([\s\S]*?)\s*```'
    code_match = re.search(code_block_pattern, llm_response_text, re.DOTALL)
    
    if code_match:
        content = code_match.group(1)
    else:
        # If no code block, use the whole text
        content = llm_response_text
    
    # Clean up the content to ensure it's valid Python syntax
    # Replace double quotes with single quotes for keys (Python style)
    content = re.sub(r'"([^"]+)":', r"'\1':", content)
    
    # Make sure status values are properly quoted
    content = re.sub(r"'status':\s*([a-zA-Z_][a-zA-Z0-9_]*)", r"'status': '\1'", content)
    
    try:
        # Use ast.literal_eval to parse the Python literals, which preserves tuples
        return ast.literal_eval(content)
    except (SyntaxError, ValueError) as e:
        logger.warning(f"ast.literal_eval failed: {e}")
        
        # Try to extract and process each dict individually
        results = []
        dict_pattern = r'\{\s*\'id\':\s*(\d+)[\s\S]*?(?=\}\s*,|\}\s*$)'
        
        for match in re.finditer(dict_pattern, content, re.DOTALL):
            try:
                dict_str = match.group(0) + '}'
                # Make sure all string values are properly quoted
                dict_str = re.sub(r"'([^']+)':\s*([a-zA-Z_][a-zA-Z0-9_]*)", r"'\1': '\2'", dict_str)
                obj = ast.literal_eval(dict_str)
                results.append(obj)
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Failed to parse dict: {e}")
                continue
        
        if results:
            return results
    
    # If we still couldn't parse it, try a more manual approach
    try:
        # Extract data manually using regex
        result = []
        id_pattern = r"'id':\s*(\d+)"
        region_pattern = r"'region_coords':\s*\(([^)]+)\)"
        pos_pattern = r"'start_pos':\s*\(([^)]+)\)"
        angle_pattern = r"'start_angle':\s*(\d+)"
        status_pattern = r"'status':\s*'([^']+)'"
        
        # Get all IDs
        ids = re.findall(id_pattern, content)
        region_coords = re.findall(region_pattern, content)
        start_positions = re.findall(pos_pattern, content)
        start_angles = re.findall(angle_pattern, content)
        statuses = re.findall(status_pattern, content)
        
        # Ensure we have the same number of matches for each field
        min_length = min(len(ids), len(region_coords), len(start_positions), 
                         len(start_angles), len(statuses))
        
        for i in range(min_length):
            # Parse tuple values
            region_tuple = tuple(int(x.strip()) for x in region_coords[i].split(','))
            start_pos_tuple = tuple(int(x.strip()) for x in start_positions[i].split(','))
            
            obj = {
                'id': int(ids[i]),
                'region_coords': region_tuple,
                'start_pos': start_pos_tuple,
                'start_angle': int(start_angles[i]),
                'status': statuses[i]
            }
            result.append(obj)
        
        if result:
            return result
    except Exception as e:
        logger.error(f"Manual extraction failed: {e}")
    
    raise ValueError("Could not extract valid Python data with tuples from LLM response")
def is_valid_region_list(var):
    """
    Checks if the variable is a list of dictionaries with the required structure.
    The structure should be a list containing at least one dictionary with the keys:
    'id', 'region_coords', 'start_pos', 'start_angle', and 'status'.
    
    'region_coords' and 'start_pos' should be tuples.
    
    Example of valid structure:
    [{'id': 0, 'region_coords': (15, 15, 50, 50), 'start_pos': (42, 36), 'start_angle': 0, 'status': 'pending'}]
    
    Args:
        var: The variable to check
        
    Returns:
        bool: True if the variable has the valid structure, False otherwise
    """
    # Check if var is a list
    if not isinstance(var, list):
        return False
    
    # Check if list has at least one element
    if len(var) == 0:
        return False
    
    # Check each element in the list
    for item in var:
        # Check if item is a dictionary
        if not isinstance(item, dict):
            return False
        
        # Check required keys
        required_keys = {'id', 'region_coords', 'start_pos', 'start_angle', 'status'}
        if set(item.keys()) != required_keys:
            return False
        
        # Check types of specific fields
        if not isinstance(item['id'], (int, float)):
            return False
        
        if not isinstance(item['region_coords'], tuple) or len(item['region_coords']) != 4:
            return False
            
        if not isinstance(item['start_pos'], tuple) or len(item['start_pos']) != 2:
            return False
            
        if not isinstance(item['start_angle'], (int, float)):
            return False
            
        if not isinstance(item['status'], str):
            return False
    
    return True

def compute_manual_subtasks(NUM_PARTITIONS):
    if NUM_PARTITIONS == 1:
        sub_tasks_manual = [
            {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
        ]
    elif NUM_PARTITIONS == 2:
        sub_tasks_manual = [
            {'id': 0, 'region_coords': (0, 0, 31, 63), 'start_pos': (16, 32), 'start_angle': 0, 'status': 'pending'},
            {'id': 1, 'region_coords': (32, 0, 63, 63), 'start_pos': (48, 32), 'start_angle': 0, 'status': 'pending'}
        ]
    elif NUM_PARTITIONS == 4:
        sub_tasks_manual = [
            {'id': 0, 'region_coords': (0, 0, 31, 31), 'start_pos': (16, 16), 'start_angle': 0, 'status': 'pending'},
            {'id': 1, 'region_coords': (0, 32, 31, 63), 'start_pos': (16, 48), 'start_angle': 0, 'status': 'pending'},
            {'id': 2, 'region_coords': (32, 0, 63, 31), 'start_pos': (48, 16), 'start_angle': 0, 'status': 'pending'},
            {'id': 3, 'region_coords': (32, 32, 63, 63), 'start_pos': (48, 48), 'start_angle': 0, 'status': 'pending'}
        ]
    else:
        raise ValueError("Invalid number of partitions. Must be 1, 2 or 4.")
    return sub_tasks_manual