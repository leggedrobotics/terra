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
from utils.models import load_neural_network

#from utils.models import get_model_ready
import json
from terra.env import TerraEnvBatch
import jax.numpy as jnp
import ast


def encode_image(cv_image):
    _, buffer = cv2.imencode(".jpg", cv_image)
    return base64.b64encode(buffer).decode("utf-8")

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
                                agent_config_override=None):
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
                    
        # Return the timestep with the updated observation
        timestep = timestep._replace(observation=updated_obs)
        
        return timestep


def create_sub_task_target_map_64x64(global_target_map_data: jnp.ndarray,
                                     region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 target map for an RL agent's sub-task from any input size.
    
    Retains both `-1` values (dig targets) and `1` values (dump targets) from 
    the specified region in the global map. Everything outside the region is set to 0 (free).
    
    Args:
        global_target_map_data: Target map of any size (1: dump, 0: free, -1: dig).
        region_coords: (y_start, x_start, y_end, x_end), inclusive bounds.
    
    Returns:
        A new 64x64 map with `-1`s and `1`s from the region; everything else is 0.
    """
    y_start, x_start, y_end, x_end = region_coords
    
    # Initialize a 64x64 map with all zeros (free space)
    sub_task_map = jnp.zeros((64, 64), dtype=global_target_map_data.dtype)
    
    # Define slice object for region
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    
    # Extract region from global map
    region_data = global_target_map_data[region_slice]
    
    # Calculate region dimensions
    region_height, region_width = region_data.shape
    
    # Place region data into the 64x64 map, ensuring it fits
    end_y = min(64, region_height)
    end_x = min(64, region_width)
    
    sub_task_map = sub_task_map.at[:end_y, :end_x].set(region_data[:end_y, :end_x])
    
    return sub_task_map

def create_sub_task_action_map_64x64(action_map_data: jnp.ndarray,
                                    region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 action map for a sub-task from any input size, preserving only actions 
    that occurred inside the specified region. Outside the region, all values are reset to 0 (free).

    Args:
        action_map_data: Action map of any size (-1: dug, 0: free, >0: dumped).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A new 64x64 map with only the region's actions preserved, all else is 0.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Initialize output map with zeros (free)
    sub_task_action_map = jnp.zeros((64, 64), dtype=action_map_data.dtype)

    # Define region slice
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Extract region from input map
    region_data = action_map_data[region_slice]
    
    # Calculate region dimensions
    region_height, region_width = region_data.shape
    
    # Place region data into the 64x64 map, ensuring it fits
    end_y = min(64, region_height)
    end_x = min(64, region_width)
    
    sub_task_action_map = sub_task_action_map.at[:end_y, :end_x].set(region_data[:end_y, :end_x])

    return sub_task_action_map

def create_sub_task_padding_mask_64x64(padding_mask_data: jnp.ndarray,
                                      region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 padding mask for a sub-task from any input size.

    Inside the region: preserves original traversability (0 or 1).
    Outside the region: sets everything to 1 (non-traversable).

    Args:
        padding_mask_data: Mask of any size (0: traversable, 1: non-traversable).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A 64x64 mask with only the region preserved; the rest is 1.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Initialize everything as non-traversable (1)
    sub_task_mask = jnp.ones((64, 64), dtype=padding_mask_data.dtype)

    # Define slice for region
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Extract the original values from the region
    region_data = padding_mask_data[region_slice]
    
    # Calculate region dimensions
    region_height, region_width = region_data.shape
    
    # Place region data into the 64x64 map, ensuring it fits
    end_y = min(64, region_height)
    end_x = min(64, region_width)
    
    sub_task_mask = sub_task_mask.at[:end_y, :end_x].set(region_data[:end_y, :end_x])

    return sub_task_mask

def create_sub_task_traversability_mask_64x64(traversability_mask_data: jnp.ndarray,
                                             region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 traversability mask for a sub-task from any input size.

    Inside the region: preserves original values (-1: agent, 0: traversable, 1: non-traversable).
    Outside the region: sets everything to 1 (non-traversable).

    Args:
        traversability_mask_data: Mask of any size 
                                  (-1: agent, 0: traversable, 1: non-traversable).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A 64x64 mask with only the region preserved; the rest is 1.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Start with a mask where everything is non-traversable (1)
    sub_task_mask = jnp.ones((64, 64), dtype=traversability_mask_data.dtype)

    # Define region slice
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Extract the original values from the region (can include -1, 0, 1)
    region_data = traversability_mask_data[region_slice]
    
    # Calculate region dimensions
    region_height, region_width = region_data.shape
    
    # Place region data into the 64x64 map, ensuring it fits
    end_y = min(64, region_height)
    end_x = min(64, region_width)
    
    sub_task_mask = sub_task_mask.at[:end_y, :end_x].set(region_data[:end_y, :end_x])

    return sub_task_mask

def create_sub_task_dumpability_mask_64x64(dumpability_mask_data: jnp.ndarray,
                                          region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64×64 dumpability mask for a sub-task from any input size.

    Inside the region: preserves original values (1: can dump, 0: can't dump).
    Outside the region: sets everything to 0 (can't dump).

    Args:
        dumpability_mask_data: Mask of any size (1: can dump, 0: can't).
        region_coords: (y_start, x_start, y_end, x_end), inclusive.

    Returns:
        A 64×64 mask with only the region preserved; the rest is 0.
    """
    y_start, x_start, y_end, x_end = region_coords

    # Initialize as all 0 (can't dump)
    sub_task_mask = jnp.zeros((64, 64), dtype=dumpability_mask_data.dtype)

    # Define region slice
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))

    # Extract the original dumpability values from the region
    region_data = dumpability_mask_data[region_slice]
    
    # Calculate region dimensions
    region_height, region_width = region_data.shape
    
    # Place region data into the 64x64 map, ensuring it fits
    end_y = min(64, region_height)
    end_x = min(64, region_width)
    
    sub_task_mask = sub_task_mask.at[:end_y, :end_x].set(region_data[:end_y, :end_x])

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
    Each region must be maximum 64x64 in size.
    
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
    
    # Maximum allowed partition size
    MAX_PARTITION_SIZE = 64
    
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
        
        # Check partition size (region_coords format: (y_start, x_start, y_end, x_end))
        y_start, x_start, y_end, x_end = item['region_coords']
        if not all(isinstance(coord, (int, float)) for coord in item['region_coords']):
            return False
        
        # Calculate width and height from coordinates
        width = x_end - x_start
        height = y_end - y_start

        print(f"Checking partition: {item['id']} with width {width} and height {height}")
        
        if width > MAX_PARTITION_SIZE or height > MAX_PARTITION_SIZE:
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

def check_overall_completion(partition_states, env_manager):
    """
    Check if the overall task is complete based on partition completion status.
    Returns True if all partitions are completed or if sufficient progress has been made.
    """
    if not partition_states:
        return False
    
    completed_partitions = []
    failed_partitions = []
    active_partitions = []
    
    for partition_idx, partition_state in partition_states.items():
        status = partition_state.get('status', 'unknown')
        if status == 'completed':
            completed_partitions.append(partition_idx)
        elif status == 'failed':
            failed_partitions.append(partition_idx)
        elif status == 'active':
            active_partitions.append(partition_idx)
    
    total_partitions = len(partition_states)
    completion_rate = len(completed_partitions) / total_partitions if total_partitions > 0 else 0
    
    # Consider task complete if:
    # 1. All partitions are completed, OR
    # 2. At least 80% of partitions are completed and no active partitions remain, OR
    # 3. All partitions are either completed or failed (no active ones left)
    
    all_completed = len(completed_partitions) == total_partitions
    high_completion_no_active = completion_rate >= 0.8 and len(active_partitions) == 0
    no_active_remaining = len(active_partitions) == 0
    
    #is_complete = all_completed or high_completion_no_active or no_active_remaining
    is_complete = all_completed
    
    #print(f"Completion check: {completed_partitions=}, {failed_partitions=}, {active_partitions=}")
    #print(f"Completion rate: {completion_rate:.2%}, Overall complete: {is_complete}")
    
    return is_complete


def calculate_map_completion_metrics(partition_states, env_manager):
    """
    Calculate completion metrics for the current map based on partition states.
    """
    if not partition_states:
        return {
            'done': False,
            'completion_rate': 0.0,
            'total_reward': 0.0,
            'completed_partitions': 0,
            'failed_partitions': 0,
            'total_partitions': 0
        }
    
    completed_count = 0
    failed_count = 0
    total_reward = 0.0
    total_partitions = len(partition_states)
    
    for partition_idx, partition_state in partition_states.items():
        status = partition_state.get('status', 'unknown')
        partition_reward = partition_state.get('total_reward', 0.0)
        total_reward += partition_reward
        
        if status == 'completed':
            completed_count += 1
        elif status == 'failed':
            failed_count += 1
    
    completion_rate = completed_count / total_partitions if total_partitions > 0 else 0.0
    is_done = check_overall_completion(partition_states, env_manager)
    
    return {
        'done': is_done,
        'completion_rate': completion_rate,
        'total_reward': total_reward,
        'completed_partitions': completed_count,
        'failed_partitions': failed_count,
        'total_partitions': total_partitions
    }

def wrap_action2(action_rl, action_type):
    """
    Wrap RL action for the environment.
    Ensures correct shape for single environment (non-batched).
    """
    # Ensure action_rl is a single integer, not an array
    if isinstance(action_rl, jnp.ndarray):
        if action_rl.shape == (1,):
            action_val = action_rl[0]  # Extract single value
        elif action_rl.shape == ():
            action_val = action_rl  # Already scalar
        else:
            raise ValueError(f"Unexpected action shape: {action_rl.shape}")
    else:
        action_val = action_rl
    
    # Convert to proper format for single environment
    # Shape should be [1] not [1,1]
    wrapped_action = action_type(
        type=jnp.array([action_val], dtype=jnp.int8),  # Shape: [1]
        action=jnp.array([action_val], dtype=jnp.int8)  # Shape: [1]
    )

    return wrapped_action

def add_batch_dimension_to_observation(obs):
    """Add batch dimension to all observation components."""
    batched_obs = {}
    for key, value in obs.items():
        if isinstance(value, jnp.ndarray):
            batched_obs[key] = jnp.expand_dims(value, axis=0)
        else:
            batched_obs[key] = jnp.array([value])
    return batched_obs

def reset_to_next_map(map_index, seed, env_manager, global_env_config,
                       initial_custom_pos=None, initial_custom_angle=None):
    """Reset the existing environment to the next map"""
    print(f"\n{'='*60}")
    print(f"RESETTING TO MAP {map_index}")
    print(f"{'='*60}")
        
    # Create new seed for this map reset
    map_seed = seed + map_index * 1000
    map_rng = jax.random.PRNGKey(map_seed)
    map_rng, reset_rng = jax.random.split(map_rng)
    reset_keys = jax.random.split(reset_rng, 1)

    # Reset the existing environment to get a new map
    # The environment will internally cycle through its available maps
    env_manager.global_env.timestep = env_manager.global_env.reset(
        global_env_config, reset_keys, initial_custom_pos, initial_custom_angle
    )

    # Update the global maps from the new reset
    #env_manager._initialize_global_environment()

    # Extract and store the NEW global map data directly
    new_timestep = env_manager.global_env.timestep
    env_manager.global_maps['target_map'] = new_timestep.state.world.target_map.map[0].copy()
    env_manager.global_maps['action_map'] = new_timestep.state.world.action_map.map[0].copy()
    env_manager.global_maps['dumpability_mask'] = new_timestep.state.world.dumpability_mask.map[0].copy()
    env_manager.global_maps['dumpability_mask_init'] = new_timestep.state.world.dumpability_mask_init.map[0].copy()
    env_manager.global_maps['padding_mask'] = new_timestep.state.world.padding_mask.map[0].copy()
    env_manager.global_maps['traversability_mask'] = new_timestep.state.world.traversability_mask.map[0].copy()
    env_manager.global_maps['trench_axes'] = new_timestep.state.world.trench_axes.copy()
    env_manager.global_maps['trench_type'] = new_timestep.state.world.trench_type.copy()
    
    # Store the new global timestep
    env_manager.global_timestep = new_timestep
    
    # Clear any existing partitions data to ensure fresh start
    env_manager.partitions = []
    env_manager.overlap_map = {}
    env_manager.overlap_regions = {}
    
    print(f"Environment reset to map {map_index}")
    print(f"New target map has {jnp.sum(env_manager.global_maps['target_map'] < 0)} dig targets")
    
    return map_rng

def initialize_partitions_for_current_map(env_manager, config, model_params):
    """Initialize all partitions for the current map"""
    partition_states = {}
    partition_models = {}
    active_partitions = []

    num_partitions = len(env_manager.partitions)
    print(f"Number of partitions: {num_partitions}")

    # Initialize all partitions
    for partition_idx in range(num_partitions):
        try:
            print(f"Initializing partition {partition_idx}...")
                
            small_env_timestep = env_manager.initialize_small_environment(partition_idx)
            print(f"Small environment initialized for partition {partition_idx}")
                
            partition_states[partition_idx] = {
                'timestep': small_env_timestep,
                'prev_actions_rl': jnp.zeros((1, config.num_prev_actions), dtype=jnp.int32),
                'step_count': 0,
                'status': 'active',
                'rewards': [],
                'actions': [],
                'total_reward': 0.0,
            }
                
            active_partitions.append(partition_idx)
            print(f"Partition {partition_idx} is active and ready.")
            print("Loading neural network model for partition...")
                
            partition_models[partition_idx] = {
                'model': load_neural_network(config, env_manager.small_env),
                'params': model_params.copy(),
                'prev_actions': jnp.zeros((1, config.num_prev_actions), dtype=jnp.int32)
            }
                        
        except Exception as e:
            print(f"Failed to initialize partition {partition_idx}: {e}")
            if partition_idx < len(env_manager.partitions):
                env_manager.partitions[partition_idx]['status'] = 'failed'

    if not active_partitions:
        print("No partitions could be initialized!")
        return None, None, None

    print(f"Successfully initialized {len(active_partitions)} partitions: {active_partitions}")
    return partition_states, partition_models, active_partitions



class SessionManager:
    """Manages ADK sessions across multiple agents to prevent session loss."""
    
    def __init__(self):
        self.session_services = {}
        self.sessions = {}
        self.runners = {}
    
    def create_agent_session(self, agent_name, app_name, user_id, session_id):
        """Create a new session for an agent."""
        # Create unique session service for this agent
        session_service_key = f"{agent_name}_{app_name}"
        
        if session_service_key not in self.session_services:
            self.session_services[session_service_key] = InMemorySessionService()
        
        session_service = self.session_services[session_service_key]
        
        # Create session
        session = session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        
        # Store session reference
        session_key = f"{user_id}_{session_id}"
        self.sessions[session_key] = {
            'session': session,
            'service': session_service,
            'app_name': app_name,
            'user_id': user_id,
            'session_id': session_id
        }
        
        print(f"Created session: {session_key} for {agent_name}")
        return session_service
    
    def create_runner(self, agent, runner_key, app_name):
        """Create and store a runner for an agent."""
        session_service_key = f"{runner_key}_{app_name}"
        session_service = self.session_services.get(session_service_key)
        
        if not session_service:
            raise ValueError(f"No session service found for {session_service_key}")
        
        runner = Runner(
            agent=agent,
            app_name=app_name,
            session_service=session_service,
        )
        
        self.runners[runner_key] = runner
        print(f"Created runner: {runner_key}")
        return runner
    
    def get_session_info(self, user_id, session_id):
        """Get session information."""
        session_key = f"{user_id}_{session_id}"
        return self.sessions.get(session_key)
    
    def list_sessions(self):
        """List all active sessions for debugging."""
        print("\nActive Sessions:")
        for key, session_info in self.sessions.items():
            print(f"  {key}: {session_info['app_name']}")

def init_llms(llm_model_key, llm_model_name, USE_PATH, config, action_size, n_envs, 
                   APP_NAME, USER_ID, SESSION_ID, MAP_SIZE):
    """
    Initialization of LLM agents with proper session management.
    """
    # Initialize session manager
    session_manager = SessionManager()
    
    if llm_model_key == "gpt":
        llm_model_name_extended = "openai/{}".format(llm_model_name)
    elif llm_model_key == "claude":
        llm_model_name_extended = "anthropic/{}".format(llm_model_name)
    else:
        llm_model_name_extended = llm_model_name
    
    print("Using model: ", llm_model_name_extended)

    # Define system messages
    size = f"{MAP_SIZE}x{MAP_SIZE}"
    print("Map size in init llm: ", size)

    system_message_master = f"""You are a master excavation coordinator responsible for optimizing excavation operations on a site map. Your task is to analyze the given terrain and intelligently partition it into optimal regions for multiple excavator deployments.

IMPORTANT: You will receive a {size} map as input

IMPORTANT: The partitions should be of maximal size 64x64. You could also consider smaller partitions

GUIDELINES FOR PARTITIONING:
1. Analyze the state of the map carefully, considering terrain features, obstacles, and excavation requirements
2. Create efficient partitions that maximize excavator productivity and minimize travel time
3. Ensure each partition has adequate space for the excavator to maneuver
4. Designate appropriate soil deposit areas within each partition or create shared deposit zones if more efficient
5. Position starting points strategically to minimize initial travel time
6. Consider terrain complexity when determining partition size - more complex areas may require smaller partitions

USE AT MOST 2 PARTITIONS TO OPTIMIZE EXCAVATION OPERATIONS. 
IMPORTANT: If you see multiple trenches, you should create a partition for each trench. 
RESPONSE FORMAT:
Respond with a JSON list of partition objects, each containing:
- 'id': Unique numeric identifier for each partition (starting from 0)
- 'region_coords': MUST BE A TUPLE with parentheses, NOT an array with brackets: (y_start, x_start, y_end, x_end)
- 'start_pos': MUST BE A TUPLE with parentheses, NOT an array with brackets: (y, x)
- 'start_angle': Always use 0 degrees for initial orientation
- 'status': Set to 'pending' for all new partitions

CRITICAL: You MUST use Python tuple notation with parentheses () for coordinates, NOT arrays with square brackets []. Failure to use tuple notation will result in errors.

CORRECT FORMAT (with tuples):
[{{'id': 0, 'region_coords': (0, 0, 31, 31), 'start_pos': (16, 16), 'start_angle': 0, 'status': 'pending'}}]

INCORRECT FORMAT (with arrays):
[{{'id': 0, 'region_coords': [0, 0, 31, 31], 'start_pos': [16, 16], 'start_angle': 0, 'status': 'pending'}}]

Example response for partitioning a 64x64 map into 4 equal quadrants (USING TUPLES, NOT ARRAYS):
[{{'id': 0, 'region_coords': (0, 0, 31, 31), 'start_pos': (16, 16), 'start_angle': 0, 'status': 'pending'}}, {{'id': 1, 'region_coords': (0, 32, 31, 63), 'start_pos': (16, 48), 'start_angle': 0, 'status': 'pending'}}, {{'id': 2, 'region_coords': (32, 0, 63, 31), 'start_pos': (48, 16), 'start_angle': 0, 'status': 'pending'}}, {{'id': 3, 'region_coords': (32, 32, 63, 63), 'start_pos': (48, 48), 'start_angle': 0, 'status': 'pending'}}]

NOTE: Always return a list of partitions even if only creating a single partition.
Very important. Ensure each partition has sufficient space for both excavation and soil deposit operations. 
Make sure to also consider a lot of space in the partition for moving the excavator to avoid getting stuck
REMEMBER TO USE TUPLES (PARENTHESES) FOR ALL COORDINATES."""

    system_message_delegation = """You are a master agent controlling an excavator. Observe the state. Decide if you should delegate digging tasks to a specialized RL agent (respond with 'delegate_to_rl') or to delegate the task to a specialized LLM agent (respond with 'delegate_to_llm')."""

    # Load game instructions
    if USE_PATH:
        with open("envs19.json", "r") as file:
            game_instructions = json.load(file)
    else:
        with open("envs18.json", "r") as file:
            game_instructions = json.load(file)

    environment_name = "AutonomousExcavatorGame"
    system_message_excavator = game_instructions.get(
        environment_name,
        "You are a game playing assistant. Provide the best action for the current game state."
    )

    # CREATE AGENTS
    if llm_model_key == "gemini":
        llm_partitioning_agent = Agent(
            name="PartitioningAgent",
            model=llm_model_name_extended,
            description="Master excavation coordinator for partitioning",
            instruction=system_message_master,
        )
        
        llm_delegation_agent = Agent(
            name="DelegationAgent", 
            model=llm_model_name_extended,
            description="Task delegation coordinator",
            instruction=system_message_delegation,
        )

        llm_excavator_agent = Agent(
            name="ExcavatorAgent",
            model=llm_model_name_extended,
            description="Excavator control agent",
            instruction=system_message_excavator,
        )
    else:
        llm_partitioning_agent = Agent(
            name="PartitioningAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description="Master excavation coordinator for partitioning",
            instruction=system_message_master,
        )
        
        llm_delegation_agent = Agent(
            name="DelegationAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description="Task delegation coordinator",
            instruction=system_message_delegation,
        )

        llm_excavator_agent = Agent(
            name="ExcavatorAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description="Excavator control agent",
            instruction=system_message_excavator,
        )

    print("All agents initialized.")

    # CREATE SESSIONS WITH PROPER MANAGEMENT
    
    # Partitioning session
    app_name_partitioning = f"{APP_NAME}_partitioning"
    user_id_partitioning = f"{USER_ID}_partitioning"
    session_id_partitioning = f"{SESSION_ID}_partitioning"
    
    session_service_partitioning = session_manager.create_agent_session(
        "PartitioningAgent",
        app_name_partitioning,
        user_id_partitioning,
        session_id_partitioning
    )

    # Delegation session
    app_name_delegation = f"{APP_NAME}_delegation"
    user_id_delegation = f"{USER_ID}_delegation"
    session_id_delegation = f"{SESSION_ID}_delegation"
    
    session_service_delegation = session_manager.create_agent_session(
        "DelegationAgent",
        app_name_delegation,
        user_id_delegation,
        session_id_delegation
    )

    # Excavator session
    app_name_excavator = f"{APP_NAME}_excavator"
    user_id_excavator = f"{USER_ID}_excavator"
    session_id_excavator = f"{SESSION_ID}_excavator"
    
    session_service_excavator = session_manager.create_agent_session(
        "ExcavatorAgent",
        app_name_excavator,
        user_id_excavator,
        session_id_excavator
    )

    print("All sessions created successfully.")

    # CREATE RUNNERS WITH SESSION MANAGER
    runner_partitioning = session_manager.create_runner(
        llm_partitioning_agent,
        "PartitioningAgent",
        app_name_partitioning
    )
    
    runner_delegation = session_manager.create_runner(
        llm_delegation_agent,
        "DelegationAgent", 
        app_name_delegation
    )
    
    runner_excavator = session_manager.create_runner(
        llm_excavator_agent,
        "ExcavatorAgent",
        app_name_excavator
    )

    print("All runners created successfully.")

    # Create LLM query object
    from terra.viz.llms_adk import LLM_query  # Assuming this import works
    
    llm_query = LLM_query(
        model_name=llm_model_name_extended,
        model=llm_model_key,
        system_message=system_message_excavator,
        action_size=action_size,
        session_id=session_id_excavator,
        runner=runner_excavator,
        user_id=user_id_excavator,
    )

    # Initialize previous actions
    prev_actions = None
    if config:
        import jax.numpy as jnp
        prev_actions = jnp.zeros(
            (n_envs, config.num_prev_actions),
            dtype=jnp.int32
        )
    else:
        print("Warning: rl_config is None, prev_actions will not be initialized.")
    
    # Debug: List all sessions
    session_manager.list_sessions()
    
    return (llm_query, runner_partitioning, runner_delegation, prev_actions, 
            system_message_master, session_manager)

async def call_agent_async_master(query: str, image, runner, user_id, session_id, session_manager=None):
    """
    Fixed version of call_agent_async_master with better error handling and session verification.
    """
    print(f"\n>>> Calling agent with user_id: {user_id}, session_id: {session_id}")
    
    # Verify session exists if session_manager is provided
    if session_manager:
        session_info = session_manager.get_session_info(user_id, session_id)
        if not session_info:
            print(f"WARNING: Session {user_id}_{session_id} not found in session manager")
            # Try to recreate session if possible
            # This would require more context about the agent and app_name
    
    # Prepare the user's message in ADK format
    text = types.Part.from_text(text=query)
    parts = [text]
    
    if image is not None:
        # Convert the image to a format suitable for ADK
        import base64
        import cv2
        
        def encode_image(cv_image):
            _, buffer = cv2.imencode(".jpg", cv_image)
            return base64.b64encode(buffer).decode("utf-8")
        
        image_data = encode_image(image)
        content_image = types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
        parts.append(content_image)

    user_content = types.Content(role='user', parts=parts)
    
    final_response_text = "Agent did not produce a final response."  # Default

    try:
        # Execute the agent with proper error handling
        async for event in runner.run_async(
            user_id=user_id, 
            session_id=session_id, 
            new_message=user_content
        ):
            print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}")
            
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
                
    except Exception as e:
        print(f"Error during agent execution: {e}")
        final_response_text = f"Error: {str(e)}"
    
    print(f"<<< Agent Response: {final_response_text}")
    return final_response_text


def setup_partitions_and_llm(map_index, ORIGINAL_MAP_SIZE, env_manager, config, llm_model_name, llm_model_key,
                                  USE_PATH, APP_NAME, USER_ID, SESSION_ID, screen, USE_MANUAL_PARTITIONING=False,
                                  USE_IMAGE_PROMPT=False):
    """
    Setup_partitions_and_llm with proper session management.
    """
    action_size = 7
        
    # Define partitions based on map size
    if ORIGINAL_MAP_SIZE == 64:
        # single partition for 64x64 map
        # sub_tasks_manual = [
        #     {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (25, 20), 'start_angle': 0, 'status': 'pending'},
        # ]

        # sub_tasks_manual = [
        #     {'id': 0, 'region_coords': (22, 22, 41, 41), 'start_pos': (25, 20), 'start_angle': 0, 'status': 'pending'},
        # ]
        # horizontal partitioning for 64x64 map
        sub_tasks_manual = [
            {'id': 0, 'region_coords': (0, 0, 31, 63), 'start_pos': (16, 32), 'start_angle': 0, 'status': 'pending'},
            {'id': 1, 'region_coords': (32, 0, 63, 63), 'start_pos': (48, 32), 'start_angle': 0, 'status': 'pending'}
        ]
        # vertical partitioning for 64x64 map
        # sub_tasks_manual = [
        #     {'id': 0, 'region_coords': (0, 0, 63, 31), 'start_pos': (32, 16), 'start_angle': 0, 'status': 'pending'},
        #     {'id': 1, 'region_coords': (0, 32, 63, 63), 'start_pos': (32, 48), 'start_angle': 0, 'status': 'pending'}
        # ]
        # vertical partitioning with overlapping
        # sub_tasks_manual = [
        #     {'id': 0, 'region_coords': (0, 0, 63, 35), 'start_pos': (32, 18), 'start_angle': 0, 'status': 'pending'},
        #     {'id': 1, 'region_coords': (0, 28, 63, 63), 'start_pos': (32, 46), 'start_angle': 0, 'status': 'pending'}
        # ]
        # random partitioning for 64x64 map
        # sub_tasks_manual = [
        #     {'id': 0, 'region_coords': (0, 0, 32, 32), 'start_pos': (25, 20), 'start_angle': 0, 'status': 'pending'},
        #     {'id': 1, 'region_coords': (0, 33, 63, 63), 'start_pos': (40, 40), 'start_angle': 0, 'status': 'pending'},
        # ]
    elif ORIGINAL_MAP_SIZE == 128:
        sub_tasks_manual = [
            {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (25, 20), 'start_angle': 0, 'status': 'pending'},
            # Add more partitions as needed
        ]
    else:
        raise ValueError(f"Unsupported ORIGINAL_MAP_SIZE: {ORIGINAL_MAP_SIZE}")

    # Initialize LLM agent with fixed session management
    (llm_query, runner_partitioning, runner_delegation, prev_actions, 
     system_message_master, session_manager) = init_llms(
        llm_model_key, llm_model_name, USE_PATH, config, action_size, 1, 
        APP_NAME, USER_ID, f"{SESSION_ID}_map_{map_index}", ORIGINAL_MAP_SIZE
    )

    sub_tasks_llm = []
    
    # ALWAYS initialize partitions - either manual or LLM-generated
    if USE_MANUAL_PARTITIONING:
        print("Using manually defined sub-tasks.")
        env_manager.initialize_with_fixed_overlaps(sub_tasks_manual)
    else:
        print("Calling LLM agent for partitioning decision...")
        # import pygame as pg
        # from terra.viz.llms_utils import capture_screen  # Assuming this import works
        
        # screen = pg.display.get_surface()
        game_state_image = capture_screen(screen)
        #save_debug_image(game_state_image, map_index, 0, image_type="general", output_dir="debug_images")
        current_observation = env_manager.global_env.timestep.observation
            
        try:
            import json
            obs_dict = {k: v.tolist() for k, v in current_observation.items()}
            observation_str = json.dumps(obs_dict)
        except AttributeError:
            observation_str = str(current_observation)

        if USE_IMAGE_PROMPT:
            prompt = f"Current observation: See image \n\nSystem Message: {system_message_master}"
        else:
            prompt = f"Current observation: {observation_str}\n\nSystem Message: {system_message_master}"

        try:
            user_id_partitioning = f"{USER_ID}_partitioning"
            session_id_partitioning = f"{SESSION_ID}_map_{map_index}_partitioning"
            
            if USE_IMAGE_PROMPT:
                response = asyncio.run(call_agent_async_master(
                    prompt, game_state_image, runner_partitioning, 
                    user_id_partitioning, session_id_partitioning, session_manager
                ))
            else:
                response = asyncio.run(call_agent_async_master(
                    prompt, None, runner_partitioning, 
                    user_id_partitioning, session_id_partitioning, session_manager
                ))
        
            llm_response_text = response
            print(f"PARTITIONING LLM response: {llm_response_text}")

            try:
                from multi_agent_utils import extract_python_format_data  # Assuming this import works
                sub_tasks_llm = extract_python_format_data(llm_response_text)
                print("Successfully parsed LLM response with tuples preserved")
            except ValueError as e:
                print(f"Extraction failed: {e}")
                sub_tasks_llm = sub_tasks_manual

        except Exception as adk_err:
            print(f"Error during PARTITIONING ADK agent communication: {adk_err}")
            sub_tasks_llm = sub_tasks_manual

        # Use appropriate partitions - validate LLM response and fallback to manual if needed
        from multi_agent_utils import is_valid_region_list  # Assuming this import works
        partition_validation = is_valid_region_list(sub_tasks_llm)
        
        if partition_validation:
            print("Using LLM-generated sub-tasks.")
            env_manager.initialize_with_fixed_overlaps(sub_tasks_llm)
        else:
            print("LLM-generated partitions invalid, falling back to manually defined sub-tasks.")
            env_manager.initialize_with_fixed_overlaps(sub_tasks_manual)

    return llm_query, runner_partitioning, runner_delegation, system_message_master, session_manager


import datetime
def save_debug_image(image, map_index, step_count, image_type="general", output_dir="debug_images"):
    """
    Save debug images with proper naming and organization.
    
    Args:
        image: The image array to save
        map_index: Current map index
        step_count: Current step count
        image_type: Type of image (e.g., "partitioning", "delegation", "excavator")
        output_dir: Directory to save images
    
    Returns:
        str: Path to saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    
    # Create filename
    filename = f"map_{map_index:03d}_step_{step_count:04d}_{image_type}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Save image
    try:
        cv2.imwrite(filepath, image)
        print(f"Debug image saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving debug image: {e}")
        return None
    

