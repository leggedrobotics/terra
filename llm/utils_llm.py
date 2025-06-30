import base64
import cv2
import numpy as np
import jax
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
import jax.numpy as jnp
from terra.viz.llms_adk import *

from terra.viz.llms_utils import *
import csv
from utils.models import load_neural_network

import json
import jax.numpy as jnp
import ast
import io
from PIL import Image
import matplotlib.pyplot as plt

import jax.numpy as jnp
import pygame as pg

from llm.session_manager_llm import SessionManager
from llm.prompt_manager_llm import PromptManager 


def encode_image(cv_image):
    _, buffer = cv2.imencode(".jpg", cv_image)
    return base64.b64encode(buffer).decode("utf-8")

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

    
    all_completed = len(completed_partitions) == total_partitions

    is_complete = all_completed
    
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


def init_llms(llm_model_key, llm_model_name, USE_PATH, config, action_size, n_envs, 
                     APP_NAME, USER_ID, SESSION_ID, MAP_SIZE):
    """
    Simplified version of init_llms using file-based prompts.
    """
    # Initialize prompt manager
    prompts = PromptManager(prompts_dir="llm/prompts")
    
    session_manager = SessionManager()
    
    if llm_model_key == "gpt":
        llm_model_name_extended = "openai/{}".format(llm_model_name)
    elif llm_model_key == "claude":
        llm_model_name_extended = "anthropic/{}".format(llm_model_name)
    else:
        llm_model_name_extended = llm_model_name
    
    print("Using model: ", llm_model_name_extended)

    # Load system messages from files
    system_message_master = prompts.get("master_partitioning", 
                                       map_size=MAP_SIZE, 
                                       max_partitions=2)
    
    system_message_delegation = prompts.get("delegation_decision", 
                                           observation="See current state")
    try:
        if USE_PATH:
            with open("prompts/excavator_llm_advanced.txt", "r") as file:
                game_instructions = file.read()
        else:
            with open("prompts/excavator_llm_simple.txt", "r") as file:
                game_instructions = file.read()

        system_message_excavator = game_instructions
    except FileNotFoundError:
        system_message_excavator = "You are an excavator control agent. Choose actions -1-6 based on game state."

    # Create agents
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

    #print("All sessions created successfully.")

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

    #print("All runners created successfully.")

    # Create LLM query object    
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
    #session_manager.list_sessions()

    return (prompts, llm_query, runner_partitioning, runner_delegation, prev_actions, 
            system_message_master, session_manager)

def get_delegation_prompt(prompts, current_observation, context=""):
    """Get delegation prompt with current state."""
    try:
        obs_str = json.dumps({k: v.tolist() if hasattr(v, 'tolist') else str(v) 
                            for k, v in current_observation.items()}) if isinstance(current_observation, dict) else str(current_observation)
        
        prompt = prompts.get("delegation_decision", observation=obs_str)
        if context:
            prompt += f"\n\nAdditional context: {context}"
        return prompt
    except Exception as e:
        print(f"Error generating delegation prompt: {e}")
        return "Decide between 'delegate_to_rl' or 'delegate_to_llm'."


def get_excavator_prompt(prompts, direction, bucket_status, position):
    """Get excavator action prompt with current state."""
    try:
        return prompts.get("excavator_action", 
                          direction=direction, 
                          bucket_status=bucket_status, 
                          position=position)
    except Exception as e:
        print(f"Error generating excavator prompt: {e}")
        return "Choose the best action (0-6) for the current game state."
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
        # # Convert the image to a format suitable for ADK
        # import base64
        # import cv2
        
        # def encode_image(cv_image):
        #     _, buffer = cv2.imencode(".jpg", cv_image)
        #     return base64.b64encode(buffer).decode("utf-8")
        
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
                                  USE_IMAGE_PROMPT=False,MAX_NUM_PARTITIONS=4):
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

    #     sub_tasks_manual = [
    #         {
    #             "id": 0,
    #             "region_coords": (30, 0, 93, 63),
    #             "start_pos": (45, 45),
    #             "start_angle": 0,
    #             "status": "pending"
    #         },
    #         {
    #             "id": 1,
    #             "region_coords": (64, 32, 127, 95),
    #             "start_pos": (80, 50),
    #             "start_angle": 0,
    #             "status": "pending"
    #         }
    # ]
        
        sub_tasks_manual = [
                {
                    "id": 0,
                    "region_coords": (20, 15, 83, 78),
                    "start_pos": (45, 80),
                    "start_angle": 0,
                    "status": "pending"
                },
                {
                    "id": 1,
                    "region_coords": (60, 10, 123, 73),
                    "start_pos": (85, 50),
                    "start_angle": 0,
                    "status": "pending"
                }
                ]
    else:
        raise ValueError(f"Unsupported ORIGINAL_MAP_SIZE: {ORIGINAL_MAP_SIZE}")

    # Initialize LLM agent with fixed session management
    (prompts, llm_query, runner_partitioning, runner_delegation, prev_actions, 
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

        game_state_image = capture_screen(screen)
        current_observation = env_manager.global_env.timestep.observation
        
        try:
            import json
            obs_dict = {k: v.tolist() for k, v in current_observation.items()}
            observation_str = json.dumps(obs_dict)
        except AttributeError:
            observation_str = str(current_observation)
        # Use file-based prompt
        if USE_IMAGE_PROMPT:
            prompt = prompts.get("master_partitioning", 
                               map_size=ORIGINAL_MAP_SIZE, 
                               max_partitions=MAX_NUM_PARTITIONS) + "\n\nCurrent observation: See image"
        else:
            try:
                obs_dict = {k: v.tolist() for k, v in current_observation.items()}
                observation_str = json.dumps(obs_dict)
            except AttributeError:
                observation_str = str(current_observation)
            
            prompt = prompts.get("master_partitioning", 
                               map_size=ORIGINAL_MAP_SIZE, 
                               max_partitions=MAX_NUM_PARTITIONS) + f"\n\nCurrent observation: {observation_str}"

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

            try:
                sub_tasks_llm = extract_python_format_data(llm_response_text)
                print("Successfully parsed LLM response with tuples preserved")
            except ValueError as e:
                print(f"Extraction failed: {e}")
                sub_tasks_llm = sub_tasks_manual

        except Exception as adk_err:
            print(f"Error during PARTITIONING ADK agent communication: {adk_err}")
            sub_tasks_llm = sub_tasks_manual

        partition_validation = is_valid_region_list(sub_tasks_llm)
        
        if partition_validation:
            print("Using LLM-generated sub-tasks.")
            env_manager.initialize_with_fixed_overlaps(sub_tasks_llm)
        else:
            print("LLM-generated partitions invalid, falling back to manually defined sub-tasks.")
            env_manager.initialize_with_fixed_overlaps(sub_tasks_manual)

    return llm_query, runner_partitioning, runner_delegation, system_message_master, session_manager, prompts


def extract_subsurface(screen, x_start, y_start, width, height, ORIGINAL_MAP_SIZE, global_env_config, partition_idx):
    """Extract a subsurface from the screen."""
                           
    try:
        screen_width, screen_height = screen.get_size()
                                
        # Get the actual tile size from the environment
        # This should be available from your global_env_config
        env_tile_size = global_env_config.tile_size[0].item()  # From your existing code
                                
        # Calculate the rendering scale factor
        # This depends on how the environment renders to the screen
        render_scale = screen_width / (ORIGINAL_MAP_SIZE * env_tile_size)
                                
        # Convert game world coordinates to screen pixel coordinates
        screen_x_start = int(x_start * env_tile_size * render_scale)
        screen_y_start = int(y_start * env_tile_size * render_scale)
        screen_width_partition = int(width * env_tile_size * render_scale)
        screen_height_partition = int(height * env_tile_size * render_scale)
                                                            
        # Rest of the clamping and subsurface creation code remains the same...
        screen_x_start = max(0, min(screen_x_start, screen_width - 1))
        screen_y_start = max(0, min(screen_y_start, screen_height - 1))
        screen_width_partition = min(screen_width_partition, screen_width - screen_x_start)
        screen_height_partition = min(screen_height_partition, screen_height - screen_y_start)
                                
        if screen_width_partition <= 0 or screen_height_partition <= 0:
            print(f"    Warning: Invalid partition size, using fallback")
            subsurface = screen.subsurface((0, 0, min(64, screen_width), min(64, screen_height)))
        else:
            subsurface = screen.subsurface((screen_x_start, screen_y_start, screen_width_partition, screen_height_partition))
                                    
    except ValueError as e:
        print(f"Error extracting subsurface for partition {partition_idx}: {e}")
        fallback_size = min(64, screen_width, screen_height)
        subsurface = screen.subsurface((0, 0, fallback_size, fallback_size))

    return subsurface



def generate_local_map(timestep):
    """
    Generate a local map from the environment's current state.

    Args:
        env: The TerraEnvBatch environment instance.
        timestep: The current timestep of the environment.

    Returns:
        A dictionary representing the local map.
    """
    # Access the state object
    state = timestep.state

    # Access the world (GridWorld) object from the state
    world = state.world
    # Extract relevant local maps from the world
    local_map = {
        "local_map_action_neg": world.local_map_action_neg.map.tolist(),
        "local_map_action_pos": world.local_map_action_pos.map.tolist(),
        "local_map_target_neg": world.local_map_target_neg.map.tolist(),
        "local_map_target_pos": world.local_map_target_pos.map.tolist(),
        "local_map_dumpability": world.local_map_dumpability.map.tolist(),
        "local_map_obstacles": world.local_map_obstacles.map.tolist(),
    }


    return local_map

def local_map_to_image(local_map):
    """
    Convert a local map dictionary to an image.

    Args:
        local_map: A dictionary containing local map data.

    Returns:
        An image (numpy array) representing the local map.
    """
    # Example: Visualize the traversability mask
    local_map_target_pos = np.array(local_map["local_map_target_pos"])

    #local_map_target_pos = np.squeeze(local_map_target_pos)


    # Normalize the values for visualization
    normalized_map = (local_map_target_pos - local_map_target_pos.min()) / (local_map_target_pos.max() - local_map_target_pos.min() + 1e-6)

    # Create a heatmap using matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(normalized_map, cmap="viridis", interpolation="nearest")
    #plt.imshow(local_map_target_pos, cmap="viridis", interpolation="nearest")

    plt.axis("off")

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", bbox_inches="tight", pad_inches=0)
    #plt.savefig("local_map.jpg", format="jpg", bbox_inches="tight", pad_inches=0)
    buf.seek(0)

    # Convert the buffer to a PIL image and then to a numpy array
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()

    return img_array

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

def extract_bucket_status(state):
    """
    Extract the bucket status from the state.

    Args:
        state: The current State object.

    Returns:
        str: The bucket status ('loaded' or 'empty').
    """
    # Access the bucket status from the agent's state
    bucket_status = state.agent.agent_state.loaded

    # Map the status to a human-readable string
    return "loaded" if bucket_status else "empty"

def base_orientation_to_direction(angle_base):
    """
    Convert the base orientation value (0-3) to a cardinal direction.

    Args:
        angle_base (int or JAX array): The base orientation value.

    Returns:
        str: The corresponding cardinal direction ('up', 'right', 'down', 'left').
    """
    # Convert JAX array to a Python scalar if necessary
    if isinstance(angle_base, jax.Array):
        angle_base = angle_base.item()

    # Map orientation to cardinal direction
    direction_map = {
        0: "right",
        1: "up",
        2: "left",
        3: "down"
    }
    return direction_map.get(angle_base, "unknown")  # Default to 'unknown' if invalid

def extract_base_orientation(state):
    """
    Extract the excavator's base orientation from the state and convert it to a cardinal direction.

    Args:
        state: The current State object.

    Returns:
        A dictionary containing the base angle and its corresponding cardinal direction.
    """
    # Extract the base angle
    angle_base = state.agent.agent_state.angle_base

    # Convert the base angle to a cardinal direction
    direction = base_orientation_to_direction(angle_base)

    return {
        "angle_base": angle_base,
        "direction": direction,
    }    

def summarize_local_map(local_map):
    """
    Generate a textual summary of the local map.

    Args:
        local_map: A dictionary representing the local map.

    Returns:
        str: A textual summary of the local map.
    """
    num_obstacles = np.sum(np.array(local_map["local_map_obstacles"]) > 0)
    num_dumpable = np.sum(np.array(local_map["local_map_dumpability"]) > 0)
    num_target_pos = np.sum(np.array(local_map["local_map_target_pos"]) > 0)
    num_target_neg = np.sum(np.array(local_map["local_map_target_neg"]) > 0)

    return (
        f"The local map contains {num_obstacles} obstacles, "
        f"{num_dumpable} dumpable areas, "
        f"{num_target_pos} positive target areas, and "
        f"{num_target_neg} negative target areas."
    )

def extract_positions(state):
    """
    Extract the current base position and target position from the game state.

    Args:
        state: The current game state object.

    Returns:
        A tuple containing:
        - current_position: A dictionary with the current base position (x, y).
        - target_position: A dictionary with the target position (x, y), or None if not available.
        
    """

    #print(state.agent.agent_state.pos_base[0])
    # Extract th11e current base position
    current_position = {
        "x": state.agent.agent_state.pos_base[0][0],
        "y": state.agent.agent_state.pos_base[0][1]
    }

    # Extract the target position from the target_map if available
    #print(state.world.target_map.map)
    target_positions = []

    for x in range(state.world.target_map.map.shape[1]):  # Iterate over rows
        for y in range(state.world.target_map.map.shape[2]):  # Iterate over columns
            if state.world.target_map.map[0, x, y] == -1:  # Access the value at (0, x, y)
                target_positions.append((x, y))
    
    # # Convert positions to tuples
    start = (int(current_position["x"]), int(current_position["y"]))
    #target = (int(target_position["x"]), int(target_position["y"])) if target_position else None

    return start, target_positions

def path_to_actions(path, initial_orientation, step_size=1):
    """
    Convert a path into a list of action numbers based on the current base orientation.

    Args:
        path (list of tuples): The path as a list of (x, y) positions.
        initial_orientation (str): The initial base orientation ('up', 'down', 'left', 'right').
        step_size (int): The step size to consider between path points.

    Returns:
        list of int: A list of action numbers corresponding to the path.
    """
    # Define the mapping of directions to deltas
    direction_deltas = {
        "up": (-1, 0),
        "right": (0, 1),
        "down": (1, 0),
        "left": (0, -1),
    }

    # Define the order of directions for turning (90-degree increments)
    directions = ["up", "right", "down", "left"]

    # Define action numbers
    FORWARD = 0
    CLOCK = 2
    ANTICLOCK = 3

    # Helper function to determine the direction between two points
    def get_direction(from_pos, to_pos):
        delta = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        # Normalize the delta to match one of the predefined directions
        if delta[0] != 0:
            delta = (delta[0] // abs(delta[0]), 0)
        elif delta[1] != 0:
            delta = (0, delta[1] // abs(delta[1]))
        for direction, d in direction_deltas.items():
            if delta == d:
                return direction
        return None

    # Initialize the list of actions
    actions = []
    current_orientation = initial_orientation

    # Iterate through the path with the given step size
    for i in range(0, len(path) - 1, step_size):
        current_pos = path[i]
        next_pos = path[min(i + step_size, len(path) - 1)]

        # Determine the required direction to move
        required_direction = get_direction(current_pos, next_pos)
        if required_direction is None:
            actions.append(-1)
            continue

        # Determine the turns needed to face the required direction
        while current_orientation != required_direction:
            current_idx = directions.index(current_orientation)
            required_idx = directions.index(required_direction)

            # Determine if we need to turn right (CLOCK) or left (ANTICLOCK)
            if (required_idx - current_idx) % 4 == 1:  # Clockwise
                actions.append(CLOCK)
                current_orientation = directions[(current_idx + 1) % 4]
            else:  # Counter-clockwise
                actions.append(ANTICLOCK)
                current_orientation = directions[(current_idx - 1) % 4]

        # Add a forward action
        actions.append(FORWARD)

    return actions

def find_nearest_target(start, target_positions):
    """
    Find the nearest target position to the starting point.

    Args:
        start (tuple): The starting position as (x, y).
        target_positions (list of tuples): A list of target positions as (x, y).

    Returns:
        tuple: The nearest target position as (x, y), or None if the list is empty.
    """
    if not target_positions:
        return None

    # Calculate the Euclidean distance to each target and find the nearest one
    nearest_target = min(target_positions, key=lambda target: (target[0] - start[0])**2 + (target[1] - start[1])**2)
    return nearest_target

def calculate_digging_percentage(initial_target_num, timestep):
    """
    Calculate the percentage of digging left in the map.

    Args:
        initial_target_num: The initial number of dig targets (-1) in the target map.
        timestep: The current timestep object containing the state of the environment.

    Returns:
        A float representing the percentage of digging left to do.
    """
    target_map = timestep.state.world.target_map.map[0]  # -1: must dig here
    dig_map = timestep.state.world.dig_map.map[0]        # -1: dug here during the episode

    # A location was originally a dig target if target_map == -1
    # A location has been dug if dig_map == -1
    # So, if target_map == -1 and dig_map != -1 → still not dug

    still_to_dig_mask = (target_map == -1) & (dig_map != -1)
    remaining_to_dig = jnp.sum(still_to_dig_mask)

    if initial_target_num > 0:
        digging_left_percentage = (remaining_to_dig / initial_target_num) * 100
    else:
        digging_left_percentage = 0.0

    return float(digging_left_percentage)


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


def verify_sync_effectiveness(partition_states, env_manager):
    """
    Verify that synchronization is actually working by checking overlap consistency.
    """
    print("\n=== VERIFYING SYNC EFFECTIVENESS ===")
    
    for (i, j), overlap_info in env_manager.overlap_regions.items():
        if i >= j:  # Only check each pair once
            continue
            
        if (i not in partition_states or j not in partition_states or
            partition_states[i]['status'] != 'active' or 
            partition_states[j]['status'] != 'active'):
            continue
        
        # Get current states
        state_i = partition_states[i]['timestep'].state
        state_j = partition_states[j]['timestep'].state
        
        # Get slices
        slice_i = overlap_info['partition_i_slice']
        slice_j = overlap_info['partition_j_slice']
        
        # Check each map
        for map_name in ['traversability_mask', 'action_map', 'target_map', 'dumpability_mask']:
            map_i = getattr(state_i.world, map_name).map
            map_j = getattr(state_j.world, map_name).map
            
            overlap_i = map_i[slice_i]
            overlap_j = map_j[slice_j]
            
            if map_name == 'traversability_mask':
                # For traversability, compare terrain only
                terrain_i = jnp.where(overlap_i == -1, 0, overlap_i)
                terrain_j = jnp.where(overlap_j == -1, 0, overlap_j)
                
                if not jnp.array_equal(terrain_i, terrain_j):
                    print(f"  ❌ {map_name} terrain mismatch between partitions {i} and {j}")
                    print(f"     Differences: {jnp.sum(terrain_i != terrain_j)} cells")
                else:
                    print(f"  ✅ {map_name} terrain consistent between partitions {i} and {j}")
            else:
                if not jnp.array_equal(overlap_i, overlap_j):
                    print(f"  ❌ {map_name} mismatch between partitions {i} and {j}")
                    print(f"     Differences: {jnp.sum(overlap_i != overlap_j)} cells")
                else:
                    print(f"  ✅ {map_name} consistent between partitions {i} and {j}")

def simple_sync_overlapping_regions(partition_states, env_manager, source_partition_idx, target_partition_idx):
    """
    Simple synchronization: Extract overlapping region from source partition and update target partition.
    This version preserves existing obstacles in the target that might be other agents.
    """
    # Check if both partitions are active
    if (source_partition_idx not in partition_states or 
        target_partition_idx not in partition_states or
        partition_states[source_partition_idx]['status'] != 'active' or
        partition_states[target_partition_idx]['status'] != 'active'):
        return
    
    # Check if partitions overlap
    if target_partition_idx not in env_manager.overlap_map.get(source_partition_idx, set()):
        return
    
    # Get overlap region info
    overlap_key = (source_partition_idx, target_partition_idx)
    if overlap_key not in env_manager.overlap_regions:
        return
    
    overlap_info = env_manager.overlap_regions[overlap_key]
    
    # Get source and target states
    source_state = partition_states[source_partition_idx]['timestep'].state
    target_state = partition_states[target_partition_idx]['timestep'].state
    
    # Get the correct slices based on partition order
    if source_partition_idx < target_partition_idx:
        source_slice = overlap_info['partition_i_slice']
        target_slice = overlap_info['partition_j_slice']
    else:
        source_slice = overlap_info['partition_j_slice']
        target_slice = overlap_info['partition_i_slice']
    
    # Define which maps to sync
    maps_to_sync = [
        'traversability_mask',
        'action_map',
        'target_map',
        'dumpability_mask'
    ]
    
    # Process each map
    for map_name in maps_to_sync:
        # Get source map
        source_map = getattr(source_state.world, map_name).map
        
        # Get target map
        target_map = getattr(target_state.world, map_name).map.copy()
        
        # Extract overlapping regions
        source_overlap = source_map[source_slice]
        target_overlap = target_map[target_slice]
        
        if map_name == 'traversability_mask':
            # Special handling for traversability mask
            # We need to:
            # 1. Preserve target's own agent (-1)
            # 2. Preserve obstacles in target that might be other agents
            # 3. Update terrain from source
            
            target_has_agent = (target_overlap == -1)
            source_has_agent = (source_overlap == -1)
            
            # Convert source data: agent positions become free space
            source_terrain = jnp.where(source_has_agent, 0, source_overlap)
            
            # IMPORTANT: Merge obstacles instead of overwriting
            # If target has an obstacle (1) that source doesn't have, it might be another agent
            # So we take the maximum to preserve obstacles
            merged_terrain = jnp.maximum(
                source_terrain,
                jnp.where(target_has_agent, 0, target_overlap)  # Exclude target's own agent from merge
            )
            
            # Final overlap: keep target's agent, use merged terrain elsewhere
            new_overlap = jnp.where(target_has_agent, -1, merged_terrain)
            
            # Debug output
            old_obstacles = jnp.sum(target_overlap == 1)
            new_obstacles = jnp.sum(new_overlap == 1)
            if old_obstacles != new_obstacles:
                print(f"  Traversability sync {source_partition_idx}->{target_partition_idx}:")
                print(f"    Obstacles before: {old_obstacles}, after: {new_obstacles}")
                print(f"    Target agents preserved: {jnp.sum(target_has_agent)}")
        else:
            # For other maps, direct copy from source
            new_overlap = source_overlap
        
        # Update target map
        target_map = target_map.at[target_slice].set(new_overlap)
        
        # Update the world state
        updated_world = env_manager._update_world_map(target_state.world, map_name, target_map)
        target_state = target_state._replace(world=updated_world)
    
    # Update the timestep with the new state
    updated_timestep = partition_states[target_partition_idx]['timestep']._replace(state=target_state)
    partition_states[target_partition_idx]['timestep'] = updated_timestep
    
    print(f"  ✓ Synced overlapping region from partition {source_partition_idx} to {target_partition_idx}")


def debug_agent_visibility(partition_states, env_manager):
    """
    Debug function to check if agents are visible as obstacles in other partitions.
    """
    print("\n=== AGENT VISIBILITY DEBUG ===")
    
    for partition_idx, partition_state in partition_states.items():
        if partition_state['status'] != 'active':
            continue
            
        print(f"\nPartition {partition_idx}:")
        
        # Get traversability mask
        trav_mask = partition_state['timestep'].state.world.traversability_mask.map
        
        # Count different cell types
        own_agent = jnp.sum(trav_mask == -1)
        obstacles = jnp.sum(trav_mask == 1)
        free_space = jnp.sum(trav_mask == 0)
        
        print(f"  Own agent cells (-1): {own_agent}")
        print(f"  Obstacle cells (1): {obstacles}")
        print(f"  Free space cells (0): {free_space}")
        
        # Check overlap regions for other agents
        for other_idx in env_manager.overlap_map.get(partition_idx, set()):
            if other_idx not in partition_states or partition_states[other_idx]['status'] != 'active':
                continue
                
            # Get overlap info
            overlap_key = (min(partition_idx, other_idx), max(partition_idx, other_idx))
            if overlap_key in env_manager.overlap_regions:
                overlap_info = env_manager.overlap_regions[overlap_key]
                
                # Get the correct slice for this partition
                if partition_idx < other_idx:
                    my_slice = overlap_info['partition_i_slice']
                else:
                    my_slice = overlap_info['partition_j_slice']
                
                # Check overlap region
                overlap_data = trav_mask[my_slice]
                overlap_obstacles = jnp.sum(overlap_data == 1)
                
                print(f"  Overlap with partition {other_idx}: {overlap_obstacles} obstacles in overlap region")