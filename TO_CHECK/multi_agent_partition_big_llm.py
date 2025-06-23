"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

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
from terra.config import BatchConfig
from terra.env import TimeStep
from terra.env import TerraEnv

from terra.viz.llms_utils import *
from multi_agent_utils import *
from terra.viz.llms_adk import *
from terra.viz.a_star import compute_path, simplify_path
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

import asyncio
import os
import argparse
import datetime
import time
import json
import csv
import pygame as pg

from pygame.locals import (
    K_q,
    QUIT,
)
from functools import partial

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

FORCE_DELEGATE_TO_RL = True     # Force delegation to RL agent for testing
FORCE_DELEGATE_TO_LLM = False   # Force delegation to LLM agent for testing
LLM_CALL_FREQUENCY = 15         # Number of steps between LLM calls
USE_MANUAL_PARTITIONING = True  # Use manual partitioning for LLM (Master Agent)
NUM_PARTITIONS = 4              # Number of partitions for LLM (Master Agent)
VISUALIZE_PARTITIONS = True      # Visualize partitions for LLM (Master Agent)
USE_IMAGE_PROMPT = True         # Use image prompt for LLM (Master Agent)
USE_LOCAL_MAP = True            # Use local map for LLM (Excavator Agent)
USE_PATH = True                 # Use path for LLM (Excavator Agent)
APP_NAME = "ExcavatorGameApp"   # Application name for ADK
USER_ID = "user_1"              # User ID for ADK
SESSION_ID = "session_001"      # Session ID for ADK

    
class LargeMapTerraEnv(TerraEnvBatchWithMapOverride):
    """A version of TerraEnvBatch specifically for 128x128 maps"""
    
    def reset_with_map_override(self, env_cfgs, rngs, custom_pos=None, custom_angle=None,
                           target_map_override=None, traversability_mask_override=None,
                           padding_mask_override=None, dumpability_mask_override=None,
                           dumpability_mask_init_override=None, action_map_override=None,
                           dig_map_override=None, agent_config_override=None):
        """Reset with 64x64 map overrides - ensures shapes are consistent"""
    
        #print("SmallMapTerraEnv: All map overrides validated for 64x64 size")
    
        # Call the TerraEnvBatchWithMapOverride's reset_with_map_override method directly
        return TerraEnvBatchWithMapOverride.reset_with_map_override(
            self, env_cfgs, rngs, custom_pos, custom_angle,
            target_map_override, traversability_mask_override,
            padding_mask_override, dumpability_mask_override,
            dumpability_mask_init_override, action_map_override,
            dig_map_override, agent_config_override
        )

class DisjointMapEnvironments:
    """
    Manages completely separate environments for large map and small maps.
    Each environment has its own timestep, configuration, and state.
    Only map data is exchanged between environments.
    """
    # def __init__(self, seed, global_env_config, small_env_config=None, num_partitions=4, 
    #              progressive_gif=False, shuffle_maps=False):
        
    def __init__(self, seed, global_env_config, small_env_config=None, 
                 progressive_gif=False, shuffle_maps=False):
        """
        Initialize with separate configurations for large and small environments.
        
        Args:
            seed: Random seed for reproducibility
            global_env_config: Environment configuration for the large global map
            small_env_config: Environment configuration for small maps (or None to derive from global)
            num_partitions: Number of partitions for the large map
            progressive_gif: Whether to generate a progressive GIF
            shuffle_maps: Whether to shuffle maps
        """
        self.rng = jax.random.PRNGKey(seed)
        self.global_env_config = global_env_config
        # self.num_partitions = num_partitions
        self.progressive_gif = progressive_gif
        self.shuffle_maps = shuffle_maps
        # Create a custom small environment config if not provided
        if small_env_config is None:
            self.small_env_config = self._derive_small_environment_config()
        else:
            self.small_env_config = small_env_config


        # Initialize the global environment (128x128) with LargeMapTerraEnv
        print("Initializing LargeMapTerraEnv for global environment...")

        self.global_env = LargeMapTerraEnv(
            rendering=True,
            n_envs_x_rendering=1,
            n_envs_y_rendering=1,
            display=True,
            progressive_gif=progressive_gif,
            shuffle_maps=shuffle_maps,
        )
 
        # # Initialize the small environment with SmallMapTerraEnv
        # Initialize the small environment with regular TerraEnv (non-batched)
        print("Initializing TerraEnv for small environment...")
        self.small_env = TerraEnv.new(
            maps_size_px=64,
            rendering=False,
            n_envs_x=1,
            n_envs_y=1,
            display=False,
            progressive_gif=progressive_gif,
        )
        
        # Store global map data
        self.global_maps = {
            'target_map': None,
            'action_map': None,
            'dumpability_mask': None,
            'dumpability_mask_init': None,
            'padding_mask': None,
            'traversability_mask': None,
            'dig_map': None,
            'trench_axes': None,
            'trench_type': None,
        }
        
        # Define partition scheme
        self.partitions = []
        #self._define_partitions()
        
        # Initialize global environment and extract maps
        self._initialize_global_environment()
        
        # Track which environment is currently being displayed
        self.current_display_env = "global"  # or "small"
        
        # Track small environment state
        self.small_env_timestep = None
        self.current_partition_idx = None

        # Agent size configurations
        self.small_agent_config = {
            'height': jnp.array([9], dtype=jnp.int32), 
            'width': jnp.array([5], dtype=jnp.int32)
        }
        self.big_agent_config = {
            'height': jnp.array([19], dtype=jnp.int32), 
            'width': jnp.array([11], dtype=jnp.int32)
        }
        
        print(f"Agent configs - Small: {self.small_agent_config}, Big: {self.big_agent_config}")

      

    def _create_clean_env_config(self):
        """Create a clean environment config for 64x64 maps without batch dimensions"""
        # Start with a minimal config
        #from terra.config import EnvConfig
    
        # If you have a reference to the original config structure, use it
        # Otherwise, create a minimal one
        try:
            # Try to create from the global config but clean it up
            base_cfg = self.small_env_config if hasattr(self, 'small_env_config') else self.global_env_config
        
            # Remove any batch dimensions by taking the first element
            def unbatch(x):
                if hasattr(x, 'shape') and len(x.shape) > 0 and x.shape[0] == 1:
                    return x[0]
                return x
            
            clean_cfg = jax.tree_map(unbatch, base_cfg)
            return clean_cfg
        
        except Exception as e:
            print(f"Warning: Could not clean config: {e}")
            # Return the original config and hope for the best
            return self.global_env_config
    def initialize_small_environment(self, partition_idx):
        """
        Initialize the small environment with map data from a specific global map partition.
        Uses TerraEnv (non-batched) for better performance and simpler interface.
        """
        if partition_idx < 0 or partition_idx >= len(self.partitions):
            raise ValueError(f"Invalid partition index: {partition_idx}")

        partition = self.partitions[partition_idx]
        region_coords = partition['region_coords']
        custom_pos = partition['start_pos']
        custom_angle = partition['start_angle']

        # print(f"=== DIAGNOSTIC: Initializing partition {partition_idx} ===")
        # print(f"Region coords: {region_coords}")
        # print(f"Custom pos: {custom_pos}")
        # print(f"Custom angle: {custom_angle}")

        # Extract sub-maps from global maps (64x64)
        # sub_maps = {
        #     'target_map': extract_sub_task_target_map(self.global_maps['target_map'], region_coords),
        #     'action_map': extract_sub_task_action_map(self.global_maps['action_map'], region_coords),
        #     'dumpability_mask': extract_sub_task_dumpability_mask(self.global_maps['dumpability_mask'], region_coords),
        #     'dumpability_mask_init': extract_sub_task_dumpability_mask(self.global_maps['dumpability_mask_init'], region_coords),
        #     'padding_mask': extract_sub_task_padding_mask(self.global_maps['padding_mask'], region_coords),
        #     'traversability_mask': extract_sub_task_traversability_mask(self.global_maps['traversability_mask'], region_coords),
        #     'dig_map': extract_sub_task_action_map(self.global_maps['dig_map'], region_coords),
        #     #'trench_axes': self.global_maps['trench_axes'],
        #     #'trench_type': self.global_maps['trench_type'],
        # }

        sub_maps = {
            'target_map': create_sub_task_target_map_64x64(self.global_maps['target_map'], region_coords),
            'action_map': create_sub_task_action_map_64x64(self.global_maps['action_map'], region_coords),
            'dumpability_mask': create_sub_task_dumpability_mask_64x64(self.global_maps['dumpability_mask'], region_coords),
            'dumpability_mask_init': create_sub_task_dumpability_mask_64x64(self.global_maps['dumpability_mask_init'], region_coords),
            'padding_mask': create_sub_task_padding_mask_64x64(self.global_maps['padding_mask'], region_coords),
            'traversability_mask': create_sub_task_traversability_mask_64x64(self.global_maps['traversability_mask'], region_coords),
            'dig_map': create_sub_task_action_map_64x64(self.global_maps['dig_map'], region_coords),
            #'trench_axes': self.global_maps['trench_axes'],
            #'trench_type': self.global_maps['trench_type'],
        }
        #DIAGNOSTIC: Check sub-map validity
        print(f"=== SUB-MAP DIAGNOSTICS ===")
        for name, map_data in sub_maps.items():
            print(f"{name}:")
            print(f"  Shape: {map_data.shape}")
            # print(f"  Min/Max: {jnp.min(map_data):.3f} / {jnp.max(map_data):.3f}")
            # print(f"  Non-zero pixels: {jnp.sum(map_data != 0)}")
            # print(f"  NaN values: {jnp.sum(jnp.isnan(map_data))}")
            # print(f"  Inf values: {jnp.sum(jnp.isinf(map_data))}")
        
        target_areas = jnp.sum(sub_maps['target_map'] > 0)
        # print(f"Target areas to excavate: {target_areas}")
         # Check traversability at start position
        traversability = sub_maps['traversability_mask']
        pos_y, pos_x = custom_pos
        # print(f"Start position ({pos_y}, {pos_x}) traversability: {traversability[pos_y, pos_x]}")
    
        # Check if start position has valid surrounding area
        y_min, y_max = max(0, pos_y-2), min(64, pos_y+3)
        x_min, x_max = max(0, pos_x-2), min(64, pos_x+3)
        local_area = traversability[y_min:y_max, x_min:x_max]
        # print(f"Local area around start ({y_min}:{y_max}, {x_min}:{x_max}):")
        # print(f"  Traversable pixels: {jnp.sum(local_area > 0)} / {local_area.size}")
        # print(f"  Min/Max traversability: {jnp.min(local_area)} / {jnp.max(local_area)}")

        # Fix trench data shapes - remove batch dimension for single environment
        trench_axes = self.global_maps['trench_axes']
        trench_type = self.global_maps['trench_type']
    
        # Remove batch dimension if present
        if trench_axes.shape[0] == 1:
            trench_axes = trench_axes[0]  # Shape: (3, 3) instead of (1, 3, 3)
        if trench_type.shape[0] == 1:
            trench_type = trench_type[0]  # Shape: () instead of (1,)
        trench_axes = trench_axes.astype(jnp.float32)
        trench_type = trench_type.astype(jnp.int32)
        
        # print(f"Trench axes shape: {trench_axes.shape}, type: {trench_axes.dtype}")
        # print(f"Trench type shape: {trench_type.shape}, type: {trench_type.dtype}")
        # print(f"Trench axes values:\n{trench_axes}")
        # print(f"Trench type value: {trench_type}")
    

        # Reset the small environment using TerraEnv's interface (no batching)
        clean_env_cfg = self._create_clean_env_config()
        print(f"Environment config created")

        self.rng, reset_key = jax.random.split(self.rng)

        try:
            print("Resetting small environment with custom map data...")
            
            # Use TerraEnv's reset method directly - much cleaner interface
            small_timestep = self.small_env.reset(
                key=reset_key,
                target_map=sub_maps['target_map'],
                padding_mask=sub_maps['padding_mask'],
                trench_axes=trench_axes,
                trench_type=trench_type,
                dumpability_mask_init=sub_maps['dumpability_mask_init'],
                env_cfg=clean_env_cfg,
                custom_pos=custom_pos,
                custom_angle=custom_angle
            )


            # print("Small environment reset successfully.")
            # print(f"=== RESET SUCCESSFUL ===")
            # print(f"Initial reward: {small_timestep.reward}")
            # print(f"Initial done: {small_timestep.done}")
            # print(f"Agent state: {small_timestep.state.agent.agent_state}")
            # print(f"Agent position: {small_timestep.state.agent.agent_state.pos_base}")
            # print(f"Agent angle: {small_timestep.state.agent.agent_state.angle_base}")
            # print(f"Agent loaded: {small_timestep.state.agent.agent_state.loaded}")

            # Store current small environment state
            self.small_env_timestep = small_timestep
            self.current_partition_idx = partition_idx
            
            # Set partition status to active
            self.partitions[partition_idx]['status'] = 'active'
            
            # Switch display to small environment
            self.current_display_env = "small"
            #rint(self.small_env_timestep)
            return small_timestep
            
        except Exception as e:
            import traceback
            print(f"Error initializing small environment: {e}")
            print(traceback.format_exc())
            raise
    
    def _derive_small_environment_config(self):
        """
        Derive a configuration for small environments based on the global config.
        Returns a modified config with appropriate size settings.
        """
        # Create a copy of the global environment config
        small_config = jax.tree_map(lambda x: x, self.global_env_config)
        
        # Modify map size and other relevant parameters
        # This requires knowledge of the config structure
        if hasattr(small_config, 'maps') and hasattr(small_config.maps, 'edge_length_px'):
            small_config = small_config._replace(
                maps=small_config.maps._replace(
                    edge_length_px=jnp.array([64], dtype=jnp.int32)
                )
            )
        
        # If map_size is a separate attribute
        if hasattr(small_config, 'map_size'):
            small_config = small_config._replace(map_size=64)
            
        return small_config
    
    def _define_partitions(self):
        """
        Define partitions of the global map.
        """
        if self.num_partitions == 4:  # 2x2 grid for a 128x128 map
            self.partitions = [
                {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
                {'id': 1, 'region_coords': (0, 64, 63, 127), 'start_pos': (32, 96), 'start_angle': 0, 'status': 'pending'},
                {'id': 2, 'region_coords': (64, 0, 127, 63), 'start_pos': (96, 32), 'start_angle': 0, 'status': 'pending'},
                {'id': 3, 'region_coords': (64, 64, 127, 127), 'start_pos': (96, 96), 'start_angle': 0, 'status': 'pending'}
            ]
        #     self.partitions = [
        #     {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (20, 20), 'start_angle': 0, 'status': 'pending'},
        #     {'id': 1, 'region_coords': (0, 64, 63, 127), 'start_pos': (20, 44), 'start_angle': 0, 'status': 'pending'},
        #     {'id': 2, 'region_coords': (64, 0, 127, 63), 'start_pos': (44, 20), 'start_angle': 0, 'status': 'pending'},
        #     {'id': 3, 'region_coords': (64, 64, 127, 127), 'start_pos': (44, 44), 'start_angle': 0, 'status': 'pending'}
        # ]
        elif self.num_partitions == 2:  # 1x2 grid
            self.partitions = [
                {'id': 0, 'region_coords': (0, 0, 63, 127), 'start_pos': (32, 64), 'start_angle': 0, 'status': 'pending'},
                {'id': 1, 'region_coords': (64, 0, 127, 127), 'start_pos': (96, 64), 'start_angle': 0, 'status': 'pending'}
            ]
        else:
            raise ValueError("Only 2 or 4 partitions are supported currently")
        

    
    def _initialize_global_environment(self):
        """Initialize the global environment with proper batching"""
        self.rng, reset_key = jax.random.split(self.rng)
    
        # Create array of keys for batching consistency
        reset_keys = jax.random.split(reset_key, 1)  # Shape: (1, 2)
    
        print("Initializing global environment...")
        global_timestep = self.global_env.reset(self.global_env_config, reset_keys)
    
        # Extract and store global map data
        self.global_maps['target_map'] = global_timestep.state.world.target_map.map[0].copy()
        self.global_maps['action_map'] = global_timestep.state.world.action_map.map[0].copy()
        self.global_maps['dumpability_mask'] = global_timestep.state.world.dumpability_mask.map[0].copy()
        self.global_maps['dumpability_mask_init'] = global_timestep.state.world.dumpability_mask_init.map[0].copy()
        self.global_maps['padding_mask'] = global_timestep.state.world.padding_mask.map[0].copy()
        self.global_maps['traversability_mask'] = global_timestep.state.world.traversability_mask.map[0].copy()
        self.global_maps['dig_map'] = global_timestep.state.world.dig_map.map[0].copy()
        self.global_maps['trench_axes'] = global_timestep.state.world.trench_axes.copy()
        self.global_maps['trench_type'] = global_timestep.state.world.trench_type.copy()
    
        # Store global timestep
        self.global_timestep = global_timestep
    
        print("Global environment initialized successfully.")
        return self.global_timestep
    

    def map_position_small_to_global(self, small_pos, region_coords):
        """
        Map agent position from small map coordinates to global map coordinates.
        Returns position in (x, y) format for rendering.
     """
        y_start, x_start, y_end, x_end = region_coords
    
        # Extract position values
        if hasattr(small_pos, 'shape'):
            if len(small_pos.shape) == 1 and small_pos.shape[0] == 2:
                small_pos_y = float(small_pos[0])
                small_pos_x = float(small_pos[1])
            else:
                small_pos_y = float(small_pos.flatten()[0])
                small_pos_x = float(small_pos.flatten()[1])
        else:
            small_pos_y = float(small_pos[0])
            small_pos_x = float(small_pos[1])
    
        # Simple mapping: just add the region offset
        global_pos_y = small_pos_y + y_start
        global_pos_x = small_pos_x + x_start
    
        # Ensure position is within valid bounds
        global_pos_y = max(0, min(127, global_pos_y))
        global_pos_x = max(0, min(127, global_pos_x))
    
        # Return as (x, y) for rendering instead of (y, x)
        return (int(global_pos_x), int(global_pos_y))



    def is_small_task_completed(self):
        """Check if the current small environment task is completed."""
        if self.small_env_timestep is None:
            return False
        
        # Handle both scalar and array cases for done flag
        done_value = self.small_env_timestep.done
        if isinstance(done_value, jnp.ndarray):
            if done_value.shape == ():  # Scalar array
                return bool(done_value)
            elif len(done_value.shape) > 0:  # Array with dimensions
                return bool(done_value[0])
            else:
                return bool(done_value)
        else:
            return bool(done_value)
        
    def _update_global_environment_display_with_all_agents(self, partition_states):
        """
        Update the global environment display with ALL active agents.
        Fixed to handle initialization errors properly.
        """
        try:
            self.rng, reset_key = jax.random.split(self.rng)
            reset_keys = jax.random.split(reset_key, 1)

            # Collect all active agent positions and angles
            all_agent_positions = []
            all_agent_angles_base = []
            all_agent_angles_cabin = []
            all_agent_loaded = []
        
            for partition_idx, partition_state in partition_states.items():
                if partition_state['status'] == 'active' and partition_state['timestep'] is not None:
                    # Get agent state from this partition
                    small_agent_state = partition_state['timestep'].state.agent.agent_state
                    partition = self.partitions[partition_idx]
                    region_coords = partition['region_coords']

                    small_pos = small_agent_state.pos_base
                    small_angle_base = small_agent_state.angle_base
                    small_angle_cabin = small_agent_state.angle_cabin
                    small_loaded = small_agent_state.loaded
                
                    # Map position to global coordinates
                    global_pos = self.map_position_small_to_global(small_pos, region_coords)
                
                    # Handle angle extraction
                    if hasattr(small_angle_base, 'shape'):
                        if small_angle_base.shape == ():
                            angle_base_val = int(small_angle_base)
                        elif len(small_angle_base.shape) >= 1:
                            angle_base_val = int(small_angle_base.flatten()[0])
                        else:
                            angle_base_val = 0.0
                    else:
                        angle_base_val = int(small_angle_base)

                    if hasattr(small_angle_cabin, 'shape'):
                        if small_angle_cabin.shape == ():
                            angle_cabin_val = int(small_angle_cabin)
                        elif len(small_angle_cabin.shape) >= 1:
                            angle_cabin_val = int(small_angle_cabin.flatten()[0])
                        else:
                            angle_cabin_val = 0.0
                    else:
                        angle_cabin_val = int(small_angle_cabin)
                    
                    if hasattr(small_loaded, 'shape'):
                        if small_loaded.shape == ():
                            small_loaded = int(small_loaded)
                        elif len(small_loaded.shape) >= 1:
                            small_loaded = int(small_loaded.flatten()[0])
                        else:
                            small_loaded = False
                    else:
                        small_loaded = int(small_loaded)
                
                    all_agent_positions.append(global_pos)
                    all_agent_angles_base.append(angle_base_val)
                    all_agent_angles_cabin.append(angle_cabin_val)
                    all_agent_loaded.append(small_loaded)
                
                    print(f"Agent {partition_idx} at global position: {global_pos}, angle base: {angle_base_val}, angle cabin: {angle_cabin_val}, loaded: {small_loaded}")

            # Update global maps from small environments incrementally
            self.update_global_maps_from_all_small_environments(partition_states)

            # Use first agent for reset position (others will be added during rendering)
            custom_pos = all_agent_positions[0] if all_agent_positions else None
            custom_angle = all_agent_angles_base[0] if all_agent_angles_base else None

            # Reset global environment with updated maps
            self.global_timestep = self.global_env.reset_with_map_override(
                self.global_env_config,
                reset_keys,
                custom_pos=custom_pos,
                custom_angle=custom_angle,
                target_map_override=self.global_maps['target_map'],
                traversability_mask_override=self.global_maps['traversability_mask'],
                padding_mask_override=self.global_maps['padding_mask'],
                dumpability_mask_override=self.global_maps['dumpability_mask'],
                dumpability_mask_init_override=self.global_maps['dumpability_mask_init'],
                action_map_override=self.global_maps['action_map'],
                dig_map_override=self.global_maps['dig_map'],
                agent_config_override=self.small_agent_config
            )
        
            # Store all agent positions for rendering - Initialize these attributes
            if not hasattr(self.global_env, 'all_agent_positions'):
                self.global_env.all_agent_positions = []
            if not hasattr(self.global_env, 'all_agent_angles_base'):
                self.global_env.all_agent_angles_base = []
            if not hasattr(self.global_env, 'all_agent_angles_cabin'):
                self.global_env.all_agent_angles_cabin = []
            if not hasattr(self.global_env, 'all_agent_loaded'):
                self.global_env.all_agent_loaded = []
                
            self.global_env.all_agent_positions = all_agent_positions
            self.global_env.all_agent_angles_base = all_agent_angles_base
            self.global_env.all_agent_angles_cabin = all_agent_angles_cabin
            self.global_env.all_agent_loaded = all_agent_loaded

            print(f"Global environment updated with {len(all_agent_positions)} active agents.")
        
        except Exception as e:
            print(f"Warning: Could not update global environment display: {e}")
            import traceback
            traceback.print_exc()
    
    
    def update_global_maps_from_all_small_environments(self, partition_states):
        """
        Update global maps with changes from ALL active small environments.
        Fixed to handle shape mismatches by properly extracting the correct region size.
        """
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] == 'active' and partition_state['timestep'] is not None:
                partition = self.partitions[partition_idx]
                y_start, x_start, y_end, x_end = partition['region_coords']
            
                # Calculate actual region dimensions
                region_height = y_end - y_start + 1
                region_width = x_end - x_start + 1
            
                print(f"Partition {partition_idx} region: ({y_start}, {x_start}) to ({y_end}, {x_end})")
                print(f"Expected region size: {region_height} x {region_width}")
            
                region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
        
                # Get current state from small environment
                small_state = partition_state['timestep'].state
            
                # Extract the maps from small environment (these are 64x64)
                small_maps = {
                    'dumpability_mask': small_state.world.dumpability_mask.map,
                    'target_map': small_state.world.target_map.map,
                    'action_map': small_state.world.action_map.map,
                    'traversability_mask': small_state.world.traversability_mask.map,
                    'padding_mask': small_state.world.padding_mask.map,
                    'dig_map': small_state.world.dig_map.map,
                }
            
                print(f"Small environment map shapes:")
                for name, map_data in small_maps.items():
                    print(f"  {name}: {map_data.shape}")
            
                # Extract only the relevant portion from the 64x64 small maps
                # that corresponds to the actual region size
                extract_height = min(region_height, 64)
                extract_width = min(region_width, 64)
            
                for map_name, small_map in small_maps.items():
                    # Extract the portion that matches the region size
                    extracted_region = small_map[:extract_height, :extract_width]
                
                    print(f"  Extracted {map_name}: {extracted_region.shape} -> Global region: {region_height}x{region_width}")
                
                    # Update the global map with the extracted region
                    try:
                        self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(extracted_region)
                    except ValueError as e:
                        print(f"  WARNING: Shape mismatch for {map_name}: {e}")
                        # Try to handle the mismatch by padding or cropping
                        if extracted_region.shape[0] != region_height or extracted_region.shape[1] != region_width:
                            # Pad or crop to match the region size
                            if extracted_region.shape[0] < region_height or extracted_region.shape[1] < region_width:
                                # Pad with zeros
                                padded_region = jnp.zeros((region_height, region_width), dtype=extracted_region.dtype)
                                padded_region = padded_region.at[:extracted_region.shape[0], :extracted_region.shape[1]].set(extracted_region)
                                extracted_region = padded_region
                            else:
                                # Crop to fit
                                extracted_region = extracted_region[:region_height, :region_width]
                        
                            print(f"  Adjusted {map_name}: {extracted_region.shape}")
                            self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(extracted_region)


    def render_global_environment_with_multiple_agents(self, partition_states):
        """
        Update and render global environment showing ALL active excavators.
        Fixed to handle missing attributes gracefully.
        """
        # First update with all agents
        self._update_global_environment_display_with_all_agents(partition_states)

        # Then render with multiple agents
        try:
            obs = self.global_timestep.observation
            info = self.global_timestep.info
        
            # Pass additional agent positions to the rendering system
            if (hasattr(self.global_env, 'all_agent_positions') and 
                hasattr(self.global_env, 'all_agent_angles_base') and
                hasattr(self.global_env, 'all_agent_angles_cabin') and 
                hasattr(self.global_env, 'all_agent_loaded')):
                
                # Add all agent positions to the info for rendering
                info['additional_agents'] = {
                    'positions': self.global_env.all_agent_positions,
                    'angles base': self.global_env.all_agent_angles_base,
                    'angles cabin': self.global_env.all_agent_angles_cabin,
                    'loaded': self.global_env.all_agent_loaded
                }
                print(f"Passing {len(self.global_env.all_agent_positions)} agents to renderer")
            else:
                print("Warning: Agent attributes not properly initialized for rendering")
                # Initialize empty lists to prevent further errors
                info['additional_agents'] = {
                    'positions': [],
                    'angles base': [],
                    'angles cabin': [],
                    'loaded': []
                }
            # ADD THIS: Simple partition info
            if VISUALIZE_PARTITIONS:
                info['show_partitions'] = True
                info['partitions'] = self.partitions  # Just pass the whole partition list

            self.global_env.terra_env.render_obs_pygame(obs, info)
    
        except Exception as e:
            print(f"Global rendering error: {e}")
            import traceback
            traceback.print_exc()
        
 
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
    
    # print(f"Wrapped action: {wrapped_action}")
    # print(f"Action type shape: {wrapped_action.type.shape}")
    # print(f"Action value shape: {wrapped_action.action.shape}")
    
    return wrapped_action

def run_experiment_with_disjoint_environments(
    llm_model_name, llm_model_key, num_timesteps, seed, 
    progressive_gif, run, small_env_config=None):
    """
    Run an experiment with completely separate environments for global and small maps.
    Modified to display all excavators simultaneously on the global map.
    """
    agent_checkpoint_path = run
    model = None
    model_params = None
    config = None

    print(f"Loading RL agent configuration from: {agent_checkpoint_path}")
    log = load_pkl_object(agent_checkpoint_path)
    config = log["train_config"]
    model_params = log["model"]
    print(f"Loaded configuration: {config}")

    # Create the original environment configs for the full map
    global_env_config = jax.tree_map(
        lambda x: x[0][None, ...].repeat(1, 0), log["env_config"]
    ) 

    config.num_test_rollouts = 1
    config.num_devices = 1
    config.num_embeddings_agent_min = 60

    # Set the number of partitions
    #num_partitions = NUM_PARTITIONS
    
    # Initialize the environment manager
    print("Initializing disjoint environment manager...")
    env_manager = DisjointMapEnvironments(
        seed=seed,
        global_env_config=global_env_config,
        small_env_config=small_env_config,
        #num_partitions=num_partitions,
        progressive_gif=progressive_gif,
        shuffle_maps=False
    )
    print("Environment manager initialized.")

    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset_initial = jax.random.split(_rng, 1)

    initial_custom_pos = None
    initial_custom_angle = None
    env_manager.global_env.timestep = env_manager.global_env.reset(global_env_config, rng_reset_initial, initial_custom_pos, initial_custom_angle)

    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type

    def repeat_action(action, n_times=1):
        return action_type.new(action.action[None].repeat(n_times, 0))
    
    # Trigger the JIT compilation
    env_manager.global_env.timestep = env_manager.global_env.step(env_manager.global_env.timestep, repeat_action(action_type.do_nothing()), rng_reset_initial)
    env_manager.global_env.terra_env.render_obs_pygame(env_manager.global_env.timestep.observation, env_manager.global_env.timestep.info)
    
    # Initialize variables for tracking progress
    step = 0
    playing = True
    
    # For visualization and metrics
    screen = pg.display.get_surface()
    frames = []
    t_counter = 0

    reward_seq = []
    global_step_rewards = []
    obs_seq = []
    action_list = []
    
    

    # Initialize with global environment first
    partition_states = {}  # Store state for each partition
    partition_models = {}  # Store models for each partition if needed
    active_partitions = []  # List of partitions that are still active
    max_steps_per_partition = num_timesteps



    action_size = 7
    # 2x2 standard partitioning
    # sub_tasks_manual = [
    #             {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
    #             {'id': 1, 'region_coords': (0, 64, 63, 127), 'start_pos': (32, 96), 'start_angle': 0, 'status': 'pending'},
    #             {'id': 2, 'region_coords': (64, 0, 127, 63), 'start_pos': (96, 32), 'start_angle': 0, 'status': 'pending'},
    #             {'id': 3, 'region_coords': (64, 64, 127, 127), 'start_pos': (96, 96), 'start_angle': 0, 'status': 'pending'}
    #         ]
    #ideal # 1x2 partitioning
    sub_tasks_manual = [
        {'id': 0, 'region_coords': (0, 0, 52, 60), 'start_pos': (25, 20), 'start_angle': 0, 'status': 'pending'},
        {'id': 1, 'region_coords': (53, 0, 115, 60), 'start_pos': (75, 20), 'start_angle': 0, 'status': 'pending'}
    ]
    sub_tasks_llm = []
    # Initialize the LLM agent
    llm_query, runner, prev_actions, system_message_master = init_llms(llm_model_key, llm_model_name, USE_PATH, 
                                                                       config, action_size, 1, 
                                                                       APP_NAME, USER_ID, SESSION_ID)
    print("LLM agent initialized.")

    if not USE_MANUAL_PARTITIONING:
        print("Calling LLM agent for partitioning decision...")
        game_state_image = capture_screen(screen)
        current_observation = env_manager.global_env.timestep.observation
        try:
            obs_dict = {k: v.tolist() for k, v in current_observation.items()}
            observation_str = json.dumps(obs_dict)

        except AttributeError:
            # Handle the case where current_observation is not a dictionary
            observation_str = str(current_observation)

        if USE_IMAGE_PROMPT:
            prompt = f"Current observation: See image \n\nSystem Message: {system_message_master}"
        else:
            prompt = f"Current observation: {observation_str}\n\nSystem Message: {system_message_master}"

        llm_decision = "delegate_to_rl"
        try:
            if USE_IMAGE_PROMPT:
                response = asyncio.run(call_agent_async_master(prompt, game_state_image, runner, USER_ID, SESSION_ID))
            else:
                response = asyncio.run(call_agent_async_master(prompt, game_state_image=None, runner=runner, USER_ID=USER_ID, SESSION_ID=SESSION_ID))
    
            llm_response_text = response
            print(f"LLM response: {llm_response_text}")

            # Use our tuple-preserving function
            try:
                sub_tasks_llm = extract_python_format_data(llm_response_text)
                print("Successfully parsed LLM response with tuples preserved")
            except ValueError as e:
                print(f"Extraction failed: {e}")
                sub_tasks_llm = sub_tasks_manual


        except Exception as adk_err:
                print(f"Error during ADK agent communication: {adk_err}")
                print("Defaulting to fallback action due to ADK error.")
                llm_decision = "fallback"

    print(f"sub-tasks from manual definition: {sub_tasks_manual}")
    print(f"Sub-tasks from LLM: {sub_tasks_llm}")

    if is_valid_region_list(sub_tasks_llm) and USE_MANUAL_PARTITIONING == False:
        env_manager.partitions= sub_tasks_llm
        print("Using LLM-generated sub-tasks.")
    else:
        env_manager.partitions = sub_tasks_manual
        print("Using manually defined sub-tasks.")

    num_partitions = len(env_manager.partitions)
    print(f"Number of partitions: {num_partitions}")


    # Initialize all partitions
    for partition_idx in range(num_partitions):
        try:
            print(f"Initializing partition {partition_idx}...")
            
            # Initialize the small environment for this partition
            small_env_timestep = env_manager.initialize_small_environment(partition_idx)
            
            # Store partition state
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
            
            # Initialize model for each partition
            partition_models[partition_idx] = {
                'model': load_neural_network(config, env_manager.small_env),
                'params': model_params.copy(),
                'prev_actions': jnp.zeros((1, config.num_prev_actions), dtype=jnp.int32)
            }
                    
        except Exception as e:
            print(f"Failed to initialize partition {partition_idx}: {e}")
            # Mark partition as failed
            env_manager.partitions[partition_idx]['status'] = 'failed'

    if not active_partitions:
        print("No partitions could be initialized!")
        return

    print(f"Successfully initialized {len(active_partitions)} partitions: {active_partitions}")

 



    print(f"Starting the game loop with {num_partitions} map partitions and disjoint environments...")

    # MAIN GAME LOOP - MODIFIED FOR MULTI-AGENT DISPLAY
    while playing and step < num_timesteps:
        # Handle quit events
        for event in pg.event.get():
            if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                playing = False

        # Check if we have any active partitions
        if not active_partitions:
            print("No more active partitions. Ending simulation.")
            break
        
        print(f"\n--- Step {step} - Stepping ALL {len(active_partitions)} active partitions synchronously ---")

        # Capture screen state
        game_state_image = capture_screen(screen)
        frames.append(game_state_image)

        # Step all active partitions simultaneously
        partitions_to_remove = []
        current_step_reward = 0.0  # Track reward sum for this step across all partitions

    
        for partition_idx in active_partitions:
            partition_state = partition_states[partition_idx]
        
            print(f"Processing partition {partition_idx} (partition step {partition_state['step_count']})")

            try:
                # Set the small environment to the current partition's state
                env_manager.small_env_timestep = partition_state['timestep']
                env_manager.current_partition_idx = partition_idx

                # Get the current observation from the small environment
                current_observation = env_manager.small_env_timestep.observation
                obs_seq.append(current_observation)

                def add_batch_dimension_to_observation(obs):
                    """Add batch dimension to all observation components."""
                    batched_obs = {}
                    for key, value in obs.items():
                        if isinstance(value, jnp.ndarray):
                            batched_obs[key] = jnp.expand_dims(value, axis=0)
                        else:
                            batched_obs[key] = jnp.array([value])
                    return batched_obs

                # Convert single observation to batch format for the RL model
                batched_observation = add_batch_dimension_to_observation(current_observation)
                obs = obs_to_model_input(batched_observation, partition_state['prev_actions_rl'], config)

                # Get action from model
                current_model = partition_models[partition_idx]
                _, logits_pi = current_model['model'].apply(current_model['params'], obs)

                pi = tfp.distributions.Categorical(logits=logits_pi)

                # Sample an action (use partition-specific seed for reproducibility)
                rng = jax.random.PRNGKey(seed + step * num_partitions + partition_idx)
                rng, action_key, step_key = jax.random.split(rng, 3)
                action_rl = pi.sample(seed=action_key)
            
                # Store action for this partition
                partition_state['actions'].append(action_rl)
                action_list.append(action_rl)

                # Update action history for this partition
                partition_state['prev_actions_rl'] = jnp.roll(partition_state['prev_actions_rl'], shift=1, axis=1)
                partition_state['prev_actions_rl'] = partition_state['prev_actions_rl'].at[:, 0].set(action_rl)

                # Apply the action to the small environment
                wrapped_action = wrap_action2(action_rl, action_type)

                # Step the small environment
                current_state = env_manager.small_env_timestep.state
                current_env_cfg = env_manager.small_env_timestep.env_cfg

                # Extract the map data needed for step method
                current_target_map = current_state.world.target_map.map
                current_padding_mask = current_state.world.padding_mask.map
                current_dumpability_mask_init = current_state.world.dumpability_mask_init.map
                current_trench_axes = current_state.world.trench_axes
                current_trench_type = current_state.world.trench_type

                # Step the environment
                new_timestep = env_manager.small_env.step(
                    state=current_state,
                    action=wrapped_action,
                    target_map=current_target_map,
                    padding_mask=current_padding_mask,
                    trench_axes=current_trench_axes,
                    trench_type=current_trench_type,
                    dumpability_mask_init=current_dumpability_mask_init,
                    env_cfg=current_env_cfg
                )
            
                # Update partition state
                partition_state['timestep'] = new_timestep
                partition_state['step_count'] += 1
            
                # Get the reward
                reward = new_timestep.reward
                # Extract scalar reward value safely
               # Extract scalar reward value safely
                if isinstance(reward, jnp.ndarray):
                    if reward.shape == ():
                        reward_val = float(reward)
                    elif len(reward.shape) > 0:
                        reward_val = float(reward.flatten()[0])
                    else:
                        reward_val = float(reward)
                else:
                    reward_val = float(reward)
                
                # Only store valid rewards
                if not (jnp.isnan(reward_val) or jnp.isinf(reward_val)):
                    partition_state['rewards'].append(reward_val)
                    partition_state['total_reward'] += reward_val
                    reward_seq.append(reward_val)  # Only add valid rewards to global sequence
                    current_step_reward += reward_val  # Add to current step's total
                    print(f"  Partition {partition_idx} - reward: {reward_val:.4f}, action: {action_rl}, done: {new_timestep.done}")
                else:
                    print(f"  Partition {partition_idx} - INVALID reward: {reward_val}, action: {action_rl}, done: {new_timestep.done}")
                    # Don't add invalid rewards to any sequence


                print(f"  Partition {partition_idx} - reward: {reward}, action: {action_rl}, done: {new_timestep.done}")

                # Check if partition is completed or failed
                partition_completed = False
            
                # Check for completion
                if env_manager.is_small_task_completed():
                    print(f"  Partition {partition_idx} COMPLETED after {partition_state['step_count']} steps!")
                    print(f"  Total reward for partition {partition_idx}: {partition_state['total_reward']:.4f}")
                    env_manager.partitions[partition_idx]['status'] = 'completed'
                    partition_state['status'] = 'completed'
                    partition_completed = True
            
                 # Check for timeout
                elif partition_state['step_count'] >= max_steps_per_partition:
                    print(f"  Partition {partition_idx} TIMED OUT after {max_steps_per_partition} steps")
                    print(f"  Total reward for partition {partition_idx}: {partition_state['total_reward']:.4f}")

                    env_manager.partitions[partition_idx]['status'] = 'failed'
                    partition_state['status'] = 'failed'
                    partition_completed = True
            
                # Check for NaN rewards
                elif jnp.isnan(reward):
                    print(f"  Partition {partition_idx} FAILED due to NaN reward")
                    env_manager.partitions[partition_idx]['status'] = 'failed'
                    partition_state['status'] = 'failed'
                    partition_completed = True
            
                # Mark partition for removal if completed
                if partition_completed:
                    partitions_to_remove.append(partition_idx)

            except Exception as e:
                print(f"  ERROR stepping partition {partition_idx}: {e}")
                # Mark partition as failed
                env_manager.partitions[partition_idx]['status'] = 'failed'
                partition_state['status'] = 'failed'
                partitions_to_remove.append(partition_idx)

        # Remove completed/failed partitions from active list
        for partition_idx in partitions_to_remove:
            if partition_idx in active_partitions:
                active_partitions.remove(partition_idx)
                print(f"Removed partition {partition_idx} from active list")

        print(f"Remaining active partitions: {active_partitions}")
        global_step_rewards.append(current_step_reward)
        print(f"Global step {step} reward (sum across all partitions): {current_step_reward:.4f}")

        # Render ALL agents simultaneously after stepping all partitions
        env_manager.render_global_environment_with_multiple_agents(partition_states)
    
        t_counter += 1
        step += 1
    
    print(f"=== End of synchronous step {step} - {len(active_partitions)} partitions still active ===")
    print("=" * 80)
    
    total_return = np.sum(reward_seq) if len(reward_seq) > 0 else 0.0
    global_total_return = np.sum(global_step_rewards) if len(global_step_rewards) > 0 else 0.0
    valid_rewards_count = len(reward_seq)
    
    # Print per-partition statistics
    print(f"\n=== FINAL STATISTICS ===")
    for partition_idx, partition_state in partition_states.items():
        status = partition_state['status']
        total_reward = partition_state['total_reward']
        steps = partition_state['step_count']
        valid_rewards = len(partition_state['rewards'])
        print(f"Partition {partition_idx}: {status.upper()} - Total Reward: {total_reward:.4f}, Steps: {steps}, Valid Rewards: {valid_rewards}")
    
    print(f"\nOverall Statistics:")
    print(f"Terra - Steps: {t_counter}, Individual Rewards Sum: {total_return:.4f}")
    print(f"Terra - Steps: {t_counter}, Global Step-wise Return: {global_total_return:.4f}")
    print(f"Valid rewards collected: {valid_rewards_count}")
    print(f"Average reward per step (global): {global_total_return/max(1, t_counter):.4f}")
    print(f"Average reward per individual action: {total_return/max(1, valid_rewards_count):.4f}")

    # Generate a timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a unique output directory for the model and timestamp
    safe_model_name = llm_model_name.replace('/', '_')
    output_dir = os.path.join("experiments", f"{safe_model_name}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "gameplay.mp4")
    save_video(frames, video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an LLM-based simulation experiment with RL agents.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        choices=["gpt-4o", 
                 "gpt-4.1", 
                 "o4-mini", 
                 "o3", 
                 "o3-mini", 
                 "gemini-1.5-flash-latest", 
                 "gemini-2.0-flash", 
                 "gemini-2.5-pro-exp-03-25", 
                 "gemini-2.5-pro-preview-03-25", 
                 "gemini-2.5-flash-preview-04-17", 
                 "claude-3-haiku-20240307", 
                 "claude-3-7-sonnet-20250219"], 
        help="Name of the LLM model to use."
    )
    parser.add_argument(
        "--model_key", 
        type=str, 
        required=True, 
        choices=["gpt", 
                 "gemini", 
                 "claude"], 
        help="Name of the LLM model key to use."
    )
    parser.add_argument(
        "--num_timesteps", 
        type=int, 
        default=100, 
        help="Number of timesteps to run."
    )
    parser.add_argument(
        "-nx",
        "--n_envs_x",
        type=int,
        default=1,
        help="Number of environments on x.",
    )
    parser.add_argument(
        "-ny",
        "--n_envs_y",
        type=int,
        default=1,
        help="Number of environments on y.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-pg",
        "--progressive_gif",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        # default="/home/gioelemo/Documents/terra/new-maps-different-order.pkl",
        # help="new-maps-different-order.pkl (12 cabin and 12 base rotations)",
        # default="/home/gioelemo/Documents/terra/gioele.pkl",
        # help="gioele.pkl (8 cabin and 4 base rotations)",
        # default="/home/gioelemo/Documents/terra/gioele_new.pkl",
        # help="gioele_new.pkl (8 cabin and 4 base rotations) Version 7 May",
        default="/home/gioelemo/Documents/terra/new-penalties.pkl",
        help="new-penalties.pkl (12 cabin and 12 base rotations) Version 7 May",
    )

    args = parser.parse_args()
    run_experiment_with_disjoint_environments(args.model_name, 
                   args.model_key, 
                   args.num_timesteps, 
                   args.seed, 
                   args.progressive_gif, 
                   args.run_name
                   )
# run_experiment_with_disjoint_environments(
#     llm_model_name="gemini-pro",
#     llm_model_key="your-api-key",
#     num_timesteps=1000,
#     seed=42,
#     progressive_gif=True,
#     run="/path/to/checkpoint.pkl"
# )