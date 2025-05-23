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
USE_IMAGE_PROMPT = True         # Use image prompt for LLM (Master Agent)
USE_LOCAL_MAP = True            # Use local map for LLM (Excavator Agent)
USE_PATH = True                 # Use path for LLM (Excavator Agent)
APP_NAME = "ExcavatorGameApp"   # Application name for ADK
USER_ID = "user_1"              # User ID for ADK
SESSION_ID = "session_001"      # Session ID for ADK


# def reset_single_environment(env, env_cfg, rng_key, custom_pos=None, custom_angle=None):
#         """Reset environment for single environment (not batched)"""
    
#         # The issue might be that your environment expects batched inputs but you're working with single envs
#         # Let's ensure we're working with single environment configs
    
#         # If env_cfg has batch dimensions, remove them
#         if hasattr(env_cfg, 'maps') and hasattr(env_cfg.maps, 'edge_length_px'):
#             if env_cfg.maps.edge_length_px.shape == (1,):
#                 # It's already single
#                 single_env_cfg = env_cfg
#             else:
#                 # Extract single config
#                 import jax
#                 single_env_cfg = jax.tree_map(lambda x: x[0] if x.shape and x.shape[0] > 1 else x, env_cfg)
#         else:
#             single_env_cfg = env_cfg
    
#         # Create single RNG key if needed
#         if isinstance(rng_key, (list, tuple)) or (hasattr(rng_key, 'shape') and len(rng_key.shape) > 1):
#             single_rng = rng_key[0] if hasattr(rng_key, '__getitem__') else rng_key
#         else:
#             single_rng = rng_key
    
#         # Reset with single environment
#         return env.reset(single_env_cfg, single_rng, custom_pos, custom_angle)
# class SmallMapTerraEnv(TerraEnvBatch):
#     """A version of TerraEnvBatch specifically for 64x64 maps"""
    
#     # @partial(jax.jit, static_argnums=(0,))
#     # def step(self, timestep, action, keys):

#     #     """Custom step function that fixes action dimensions"""
#     #     # Fix action dimensions before processing
#     #     #fixed_action = fix_action_dimensions(action)
    
#     #     state = timestep.state
#     #     env_cfg = timestep.env_cfg if hasattr(timestep, 'env_cfg') else state.env_cfg
#     #     print("Action", action)
#     #     print("Fixed action", action)
#     #     # Perform step operations with fixed action
#     #     new_state = state._step(action)
#     #     reward = state._get_reward(new_state, action)
#     #     new_state = self.wrap_state(new_state)
#     #     obs = self._state_to_obs_dict(new_state)

#     #     done, task_done = state._is_done(
#     #         new_state.world.action_map.map,
#     #         new_state.world.target_map.map,
#     #         new_state.agent.agent_state.loaded,
#     #     )
    
#     #     # Build infos
#     #     infos = new_state._get_infos(action, task_done)
    
#     #     # Build the timestep result
#     #     result = TimeStep(
#     #         state=new_state,
#     #         observation=obs,
#     #         reward=reward,
#     #         done=done,
#     #         info=infos,
#     #         env_cfg=env_cfg
#     #     )
    
#     #     return result

#     def reset_with_map_override(self, env_cfgs, rngs, custom_pos=None, custom_angle=None,
#                            target_map_override=None, traversability_mask_override=None,
#                            padding_mask_override=None, dumpability_mask_override=None,
#                            dumpability_mask_init_override=None, action_map_override=None,
#                            dig_map_override=None):
#         """Reset with 64x64 map overrides - ensures shapes are consistent"""
    
#         print("SmallMapTerraEnv: All map overrides validated for 64x64 size")
    
#         # Call the TerraEnvBatchWithMapOverride's reset_with_map_override method directly
#         return TerraEnvBatchWithMapOverride.reset_with_map_override(
#             self, env_cfgs, rngs, custom_pos, custom_angle,
#             target_map_override, traversability_mask_override,
#             padding_mask_override, dumpability_mask_override,
#             dumpability_mask_init_override, action_map_override,
#             dig_map_override
#         )
    

    
class LargeMapTerraEnv(TerraEnvBatch):
    """A version of TerraEnvBatch specifically for 128x128 maps"""
    

    @partial(jax.jit, static_argnums=(0,))
    def step(self, timestep, action, keys):

        """Custom step function that fixes action dimensions"""
        # Fix action dimensions before processing
        #fixed_action = fix_action_dimensions(action)
    
        state = timestep.state
        env_cfg = timestep.env_cfg if hasattr(timestep, 'env_cfg') else state.env_cfg
    
        # Perform step operations with fixed action
        new_state = state._step(action)
        reward = state._get_reward(new_state, action)
        new_state = self.wrap_state(new_state)
        obs = self._state_to_obs_dict(new_state)

        done, task_done = state._is_done(
            new_state.world.action_map.map,
            new_state.world.target_map.map,
            new_state.agent.agent_state.loaded,
        )
    
        # Build infos
        infos = new_state._get_infos(action, task_done)
    
        # Build the timestep result
        result = TimeStep(
            state=new_state,
            observation=obs,
            reward=reward,
            done=done,
            info=infos,
            env_cfg=env_cfg
        )
    
        return result

    def reset_with_map_override(self, env_cfgs, rngs, custom_pos=None, custom_angle=None,
                           target_map_override=None, traversability_mask_override=None,
                           padding_mask_override=None, dumpability_mask_override=None,
                           dumpability_mask_init_override=None, action_map_override=None,
                           dig_map_override=None):
        """Reset with 64x64 map overrides - ensures shapes are consistent"""
    
        print("SmallMapTerraEnv: All map overrides validated for 64x64 size")
    
        # Call the TerraEnvBatchWithMapOverride's reset_with_map_override method directly
        return TerraEnvBatchWithMapOverride.reset_with_map_override(
            self, env_cfgs, rngs, custom_pos, custom_angle,
            target_map_override, traversability_mask_override,
            padding_mask_override, dumpability_mask_override,
            dumpability_mask_init_override, action_map_override,
            dig_map_override
        )

class DisjointMapEnvironments:
    """
    Manages completely separate environments for large map and small maps.
    Each environment has its own timestep, configuration, and state.
    Only map data is exchanged between environments.
    """
    def __init__(self, seed, global_env_config, small_env_config=None, num_partitions=4, 
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
        self.num_partitions = num_partitions
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
            rendering=True,
            n_envs_x=1,
            n_envs_y=1,
            display=True,
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
        self._define_partitions()
        
        # Initialize global environment and extract maps
        self._initialize_global_environment()
        
        # Track which environment is currently being displayed
        self.current_display_env = "global"  # or "small"
        
        # Track small environment state
        self.small_env_timestep = None
        self.current_partition_idx = None

    # def _initialize_small_environment_lazy(self):
    #     """
    #     Initialize the small environment only when needed, with the proper configuration.
    #     """
    #     if self.small_env is None:
    #         print("Lazy-initializing SmallMapTerraEnv for small environment...")
            
    #         # Create the small environment with the corrected configuration
    #         self.small_env = SmallMapTerraEnv(
    #             rendering=True,
    #             n_envs_x_rendering=1,
    #             n_envs_y_rendering=1,
    #             display=True,
    #             progressive_gif=self.progressive_gif,
    #             shuffle_maps=self.shuffle_maps,
    #         )
            
    #         print("SmallMapTerraEnv lazy-initialized successfully")
    def _create_clean_env_config(self):
        """Create a clean environment config for 64x64 maps without batch dimensions"""
        # Start with a minimal config
        from terra.config import EnvConfig
    
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

        # Extract sub-maps from global maps (64x64)
        sub_maps = {
            'target_map': extract_sub_task_target_map(self.global_maps['target_map'], region_coords),
            'action_map': extract_sub_task_action_map(self.global_maps['action_map'], region_coords),
            'dumpability_mask': extract_sub_task_dumpability_mask(self.global_maps['dumpability_mask'], region_coords),
            'dumpability_mask_init': extract_sub_task_dumpability_mask(self.global_maps['dumpability_mask_init'], region_coords),
            'padding_mask': extract_sub_task_padding_mask(self.global_maps['padding_mask'], region_coords),
            'traversability_mask': extract_sub_task_traversability_mask(self.global_maps['traversability_mask'], region_coords),
            'dig_map': extract_sub_task_action_map(self.global_maps['dig_map'], region_coords),
            'trench_axes': self.global_maps['trench_axes'],
            'trench_type': self.global_maps['trench_type'],
        }

        # Fix trench data shapes - remove batch dimension for single environment
        trench_axes = self.global_maps['trench_axes']
        trench_type = self.global_maps['trench_type']
    
        # Remove batch dimension if present
        if trench_axes.shape[0] == 1:
            trench_axes = trench_axes[0]  # Shape: (3, 3) instead of (1, 3, 3)
        if trench_type.shape[0] == 1:
            trench_type = trench_type[0]  # Shape: () instead of (1,)
        
        print(f"Trench axes shape: {trench_axes.shape}")
        print(f"Trench type shape: {trench_type.shape}")
        # Convert trench_axes to float32 to avoid type mismatch in reward calculation
        trench_axes = trench_axes.astype(jnp.float32)

        # Ensure trench_type is int32
        trench_type = trench_type.astype(jnp.int32)


        print("INITIALIZING SMALL ENVIRONMENT with TerraEnv")
        for name, map_data in sub_maps.items():
            print(f"Sub-map '{name}' shape: {map_data.shape}")

        # Reset the small environment using TerraEnv's interface (no batching)
        clean_env_cfg = self._create_clean_env_config()

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


            print("Small environment reset successfully.")

            # Store current small environment state
            self.small_env_timestep = small_timestep
            self.current_partition_idx = partition_idx
            
            # Set partition status to active
            self.partitions[partition_idx]['status'] = 'active'
            
            # Switch display to small environment
            self.current_display_env = "small"
            
            return small_timestep
            
        except Exception as e:
            import traceback
            print(f"Error initializing small environment: {e}")
            print(traceback.format_exc())
            raise
    def step_small_environment(self, action):
        """
        Step small environment using TerraEnv's simpler interface.
        No need for batch dimension handling.
        """
        if self.small_env_timestep is None:
            raise ValueError("Small environment not initialized")

        self.rng, step_key = jax.random.split(self.rng)

        print(f"Action for small env step: {action}")
        
        try:
            # Use TerraEnv's step method - much simpler than batched version
            self.small_env_timestep = self.small_env.step(
                self.small_env_timestep.state,
                action,
                step_key
            )
            print("Small environment step successful")
        except Exception as e:
            print(f"Error stepping small environment: {e}")
            raise

        return self.small_env_timestep
    def step_global_environment(self, action):
        """
        Step the global environment forward using the batched interface.
        """
        self.rng, step_key = jax.random.split(self.rng)
        step_keys = jax.random.split(step_key, 1)  # Still need batch for global env
        
        print("Stepping global environment...")
        try:
            self.global_timestep = self.global_env.step(
                self.global_timestep, 
                action, 
                step_keys
            )
            print("Global environment step successful")
        except Exception as e:
            print(f"Error stepping global environment: {e}")
            raise
        
        return self.global_timestep
    
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


    # def step_small_environment(self, action):
    #     """Step small environment without dimension fixing"""
    #     if self.small_env_timestep is None:
    #         raise ValueError("Small environment not initialized")
    
    #     self._initialize_small_environment_lazy()
    
    #     self.rng, step_key = jax.random.split(self.rng)
    #     step_keys = jax.random.split(step_key, 1)
    #     print(step_keys)
    
    #     print(f"Action for step: {action}")
         
    #     # DEBUG: Print environment context
    #     print("=== SMALL ENVIRONMENT STEP DEBUG ===")
    #     print(f"Small env timestep type: {type(self.small_env_timestep)}")
    #     print(f"Small env state type: {type(self.small_env_timestep.state)}")
    #     print(f"Agent state loaded: {self.small_env_timestep.state.agent.agent_state.loaded}")
    #     print("=== END SMALL ENV DEBUG ===")
    #     try:
    #         # Use the action directly without fixing dimensions
    #         self.small_env_timestep = self.small_env.step(
    #             self.small_env_timestep, 
    #             action, 
    #             step_keys
    #         )
    #         print("Small environment step successful")
    #     except Exception as e:
    #         print(f"Error stepping small environment: {e}")
    #         raise
    
    #     return self.small_env_timestep
    

    # def initialize_small_environment(self, partition_idx):
    #     """
    #     Initialize the small environment with map data from a specific global map partition.
    #     This is a completely separate environment that only shares map data.
    
    #     Args:
    #         partition_idx: Index of the partition to initialize
        
    #     Returns:
    #         timestep object for the initialized small environment
    #     """
    #     if partition_idx < 0 or partition_idx >= len(self.partitions):
    #         raise ValueError(f"Invalid partition index: {partition_idx}. Must be between 0 and {len(self.partitions)-1}")
    #     self._initialize_small_environment_lazy()

    #     partition = self.partitions[partition_idx]
    #     region_coords = partition['region_coords']
    #     custom_pos = partition['start_pos']
    #     custom_angle = partition['start_angle']
    
    #     # Extract sub-maps from global maps
    #     sub_maps = {
    #         'target_map': extract_sub_task_target_map(self.global_maps['target_map'], region_coords),
    #         'action_map': extract_sub_task_action_map(self.global_maps['action_map'], region_coords),
    #         'dumpability_mask': extract_sub_task_dumpability_mask(self.global_maps['dumpability_mask'], region_coords),
    #         'dumpability_mask_init': extract_sub_task_dumpability_mask(self.global_maps['dumpability_mask_init'], region_coords),
    #         'padding_mask': extract_sub_task_padding_mask(self.global_maps['padding_mask'], region_coords),
    #         'traversability_mask': extract_sub_task_traversability_mask(self.global_maps['traversability_mask'], region_coords),
    #         'dig_map': extract_sub_task_action_map(self.global_maps['dig_map'], region_coords)
    #     }
    
    #     print("INITIALIZING SMALL ENVIRONMENT")
    #     for name, map_data in sub_maps.items():
    #         print(f"Sub-map '{name}' shape: {map_data.shape}")
    
    #     # Reset the small environment with the small config and map overrides
    #     self.rng, reset_key = jax.random.split(self.rng)
    #     reset_keys = jax.random.split(reset_key, 1)
    
    #     try:
    #         print("Resetting small environment with custom map data...")
    #         small_timestep = self.small_env.reset_with_map_override(
    #             self.small_env_config,
    #             reset_keys,
    #             custom_pos=custom_pos,
    #             custom_angle=custom_angle,
    #             target_map_override=sub_maps['target_map'],
    #             traversability_mask_override=sub_maps['traversability_mask'],
    #             padding_mask_override=sub_maps['padding_mask'],
    #             dumpability_mask_override=sub_maps['dumpability_mask'],
    #             dumpability_mask_init_override=sub_maps['dumpability_mask_init'],
    #             action_map_override=sub_maps['action_map'],
    #             dig_map_override=sub_maps['dig_map']
    #         )
    #         print("Small environment reset successfully.")

    #         # Print verification of environment sizes
    #         print(f"Small environment map shape: {small_timestep.state.world.target_map.map.shape}")
        
    #         # Store current small environment state
    #         self.small_env_timestep = small_timestep
    #         self.current_partition_idx = partition_idx
        
    #         # Set partition status to active
    #         self.partitions[partition_idx]['status'] = 'active'
        
    #         # Switch display to small environment
    #         self.current_display_env = "small"
        
    #         return small_timestep
        
    #     except Exception as e:
    #         import traceback
    #         print(f"Error initializing small environment: {e}")
    #         print(traceback.format_exc())
    #         raise
    

    def _reset_small_environment_after_done(self):
        """Reset the small environment after an episode is done."""
        self.rng, reset_key = jax.random.split(self.rng)
        reset_keys = jax.random.split(reset_key, 1)
        
        partition = self.partitions[self.current_partition_idx]
        region_coords = partition['region_coords']
        custom_pos = partition['start_pos']
        custom_angle = partition['start_angle']
        
        # Extract sub-maps again (or reuse cached versions)
        sub_maps = {
            'target_map': extract_sub_task_target_map(self.global_maps['target_map'], region_coords),
            'action_map': extract_sub_task_action_map(self.global_maps['action_map'], region_coords),
            'dumpability_mask': extract_sub_task_dumpability_mask(self.global_maps['dumpability_mask'], region_coords),
            'dumpability_mask_init': extract_sub_task_dumpability_mask(self.global_maps['dumpability_mask_init'], region_coords),
            'padding_mask': extract_sub_task_padding_mask(self.global_maps['padding_mask'], region_coords),
            'traversability_mask': extract_sub_task_traversability_mask(self.global_maps['traversability_mask'], region_coords),
            'dig_map': extract_sub_task_action_map(self.global_maps['dig_map'], region_coords)
        }
        
        # Reset with the specific map data using the specialized environment
        try:
            self.small_env_timestep = self.small_env.reset_with_map_override(
                self.small_env_config,
                reset_keys,
                custom_pos=custom_pos,
                custom_angle=custom_angle,
                target_map_override=sub_maps['target_map'],
                traversability_mask_override=sub_maps['traversability_mask'],
                padding_mask_override=sub_maps['padding_mask'],
                dumpability_mask_override=sub_maps['dumpability_mask'],
                dumpability_mask_init_override=sub_maps['dumpability_mask_init'],
                action_map_override=sub_maps['action_map'],
                dig_map_override=sub_maps['dig_map']
            )
            print("Small environment reset successfully after episode done.")
        except Exception as e:
            print(f"Error resetting small environment after episode done: {e}")
            raise
    
    def step_global_environment(self, action):
        """
        Step the global environment forward, independent of small environments.
        
        Args:
            action: Action to take
            
        Returns:
            Next timestep for the global environment
        """
        self.rng, step_key = jax.random.split(self.rng)
        step_keys = jax.random.split(step_key, 1)
        # DEBUG: Print environment context
        print("=== GLOBAL ENVIRONMENT STEP DEBUG ===")
        print(f"Global env timestep type: {type(self.global_timestep)}")
        print(f"Global env state type: {type(self.global_timestep.state)}")
        print(f"Agent state loaded: {self.global_timestep.state.agent.agent_state.loaded}")
        print("=== END GLOBAL ENV DEBUG ===")
        
        # Step the global environment using our specialized class
        print("Stepping global environment...")
        try:
            self.global_timestep = self.global_env.step(self.global_timestep, action, step_keys)
            print("Global environment step successful")
        except Exception as e:
            print(f"Error stepping global environment: {e}")
            raise
        
        return self.global_timestep
    
    def switch_to_global_environment(self):
        """
        Switch display to show the global environment.
        """
        self.current_display_env = "global"
        # Additional rendering logic may be needed here
    
    def switch_to_small_environment(self):
        """
        Switch display to show the small environment.
        """
        if self.small_env_timestep is None:
            raise ValueError("Small environment not initialized")
        self.current_display_env = "small"
        # Additional rendering logic may be needed here
    
    def update_global_maps_from_small_environment(self):
        """
        Update the global maps with the changes from a completed small environment task.
        This transfers map data only, not state.
        """
        if self.small_env_timestep is None or self.current_partition_idx is None:
            raise ValueError("Small environment not initialized")
        
        partition = self.partitions[self.current_partition_idx]
        y_start, x_start, y_end, x_end = partition['region_coords']
        region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
        
        # Update the global maps with the changes from the small environment
        self.global_maps['dumpability_mask'] = self.global_maps['dumpability_mask'].at[region_slice].set(
            self.small_env_timestep.state.world.dumpability_mask.map[0]
        )
        
        self.global_maps['target_map'] = self.global_maps['target_map'].at[region_slice].set(
            self.small_env_timestep.state.world.target_map.map[0]
        )
        
        self.global_maps['action_map'] = self.global_maps['action_map'].at[region_slice].set(
            self.small_env_timestep.state.world.action_map.map[0]
        )
        
        self.global_maps['traversability_mask'] = self.global_maps['traversability_mask'].at[region_slice].set(
            self.small_env_timestep.state.world.traversability_mask.map[0]
        )
        
        self.global_maps['padding_mask'] = self.global_maps['padding_mask'].at[region_slice].set(
            self.small_env_timestep.state.world.padding_mask.map[0]
        )
        
        self.global_maps['dig_map'] = self.global_maps['dig_map'].at[region_slice].set(
            self.small_env_timestep.state.world.dig_map.map[0]
        )
        
        # Set partition status to completed
        self.partitions[self.current_partition_idx]['status'] = 'completed'
        
        print(f"Updated global maps with changes from partition {self.current_partition_idx}")
        
        # Optionally update the global environment with the new maps
        self._update_global_environment_maps()
    
    def _update_global_environment_maps(self):
        """
        Update the actual global environment instance with the latest global maps.
        This is necessary if you want to render the global environment with updates.
        """
        # This would require a reset_with_map_override equivalent for the global environment
        # or other environment-specific methods to update maps without resetting
        try:
            self.rng, reset_key = jax.random.split(self.rng)
            reset_keys = jax.random.split(reset_key, 1)
            
            self.global_timestep = self.global_env.reset_with_map_override(
                self.global_env_config,
                reset_keys,
                target_map_override=self.global_maps['target_map'],
                traversability_mask_override=self.global_maps['traversability_mask'],
                padding_mask_override=self.global_maps['padding_mask'],
                dumpability_mask_override=self.global_maps['dumpability_mask'],
                dumpability_mask_init_override=self.global_maps['dumpability_mask_init'],
                action_map_override=self.global_maps['action_map'],
                dig_map_override=self.global_maps['dig_map']
            )
            print("Global environment maps updated.")
        except Exception as e:
            print(f"Warning: Could not update global environment maps: {e}")
    
    def is_small_task_completed(self):
        """
        Check if the current small environment task is completed.
        
        Returns:
            Boolean indicating if the task is completed
        """
        if self.small_env_timestep is None:
            return False
        return self.small_env_timestep.info.get("task_done", jnp.array([False]))[0]
    
    def all_partitions_completed(self):
        """
        Check if all partitions have been completed.
        
        Returns:
            Boolean indicating if all partitions are completed
        """
        return all(partition['status'] == 'completed' for partition in self.partitions)
    
    def get_next_pending_partition_idx(self):
        """
        Get the index of the next pending partition.
        
        Returns:
            Index of the next pending partition, or None if all are completed
        """
        for i, partition in enumerate(self.partitions):
            if partition['status'] == 'pending':
                return i
        return None
    
    def get_current_observation(self):
        """
        Get the observation from the currently active environment.
        
        Returns:
            Observation from the active environment
        """
        if self.current_display_env == "global":
            return self.global_timestep.observation
        else:
            return self.small_env_timestep.observation
    # Add this debugging function to help identify the action format issue
 
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
    
    print(f"Wrapped action: {wrapped_action}")
    print(f"Action type shape: {wrapped_action.type.shape}")
    print(f"Action value shape: {wrapped_action.action.shape}")
    
    return wrapped_action

def run_experiment_with_disjoint_environments(
    llm_model_name, llm_model_key, num_timesteps, seed, 
    progressive_gif, run, small_env_config=None):
    """
    Run an experiment with completely separate environments for global and small maps.
    
    Args:
        llm_model_name: Name of the LLM model
        llm_model_key: Key for the LLM model
        num_timesteps: Number of timesteps to run
        seed: Random seed
        progressive_gif: Whether to generate a progressive GIF
        run: Path to the checkpoint file
        small_env_config: Optional custom config for small environments
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
    num_partitions = NUM_PARTITIONS
    
    # Initialize the environment manager
    print("Initializing disjoint environment manager...")
    env_manager = DisjointMapEnvironments(
        seed=seed,
        global_env_config=global_env_config,
        small_env_config=small_env_config,  # Can be None to auto-derive
        num_partitions=num_partitions,
        progressive_gif=progressive_gif,
        shuffle_maps=False
    )
    print("Environment manager initialized.")
    
    # Initialize variables for tracking progress
    step = 0
    playing = True
    prev_actions_rl = jnp.zeros((1, config.num_prev_actions), dtype=jnp.int32)
    model_initialized = False
    
    # For visualization and metrics
    screen = pg.display.get_surface()
    frames = []
    reward_seq = []
    obs_seq = []
    action_list = []
    
    print(f"Starting the game loop with {num_partitions} map partitions and disjoint environments...")

    # Initialize with global environment first
    current_partition_idx = -1

    # Main game loop
    while playing and step < num_timesteps:
        # Handle quit events
        for event in pg.event.get():
            if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                playing = False
        
        # Check if we need to move to a new partition (small environment)
        next_partition = env_manager.get_next_pending_partition_idx()
        if next_partition is None:
            print("All partitions completed!")
            playing = False
            break
        
        if current_partition_idx != next_partition:
            print(f"\n--- Moving to Partition {next_partition} ---")
            current_partition_idx = next_partition
            
            # If there was a previous small environment, update global maps with its results
            if env_manager.current_partition_idx is not None and env_manager.current_partition_idx != next_partition:
                env_manager.update_global_maps_from_small_environment()
            
            # Initialize the small environment for the new partition
            env_manager.small_env_timestep = env_manager.initialize_small_environment(current_partition_idx)
            
            # Reset RL agent's history
            prev_actions_rl = jnp.zeros((1, config.num_prev_actions), dtype=jnp.int32)
            
            
            if not model_initialized:
                print("Initializing model with the first partition's data...")
                try:
                    # Initialize the model with the small environment
                    model = load_neural_network(config, env_manager.small_env)
                    model_initialized = True
                    print("Model successfully initialized with the first partition!")
                except Exception as e:
                    print(f"Failed to initialize model: {e}")
                    playing = False
                    break
        
        print(f"\n--- Step {step} in Partition {current_partition_idx} ---")

        # Capture screen state (whichever environment is currently displayed)
        game_state_image = capture_screen(screen)
        frames.append(game_state_image)

        # Get the current observation from the small environment
        current_observation = env_manager.small_env_timestep.observation
        obs_seq.append(current_observation)
        print("CURRENT OBSERVATION OKAY")
        #print(current_observation['action_map'])
        # Process observation for the RL model


        # Add batch dimension to the observation BEFORE obs_to_model_input
        def add_batch_dimension_to_observation(obs):
            """
            Add batch dimension to all observation components.
            Transforms TerraEnv observation to TerraEnvBatch-like observation.
            """
            batched_obs = {}
            for key, value in obs.items():
                if isinstance(value, jnp.ndarray):
                    # Add batch dimension as first axis
                    batched_obs[key] = jnp.expand_dims(value, axis=0)
                else:
                    # Handle scalar values
                    batched_obs[key] = jnp.array([value])
            return batched_obs

        # Convert single observation to batch format
        batched_observation = add_batch_dimension_to_observation(current_observation)
        obs = obs_to_model_input(batched_observation, prev_actions_rl, config)
        print("OBSERVATION OKAY")
        
    
        # Get action from model
        _, logits_pi = model.apply(model_params, obs)
        print("LOGITS OKAY")
        pi = tfp.distributions.Categorical(logits=logits_pi)
        
        # Sample an action
        rng = jax.random.PRNGKey(seed + step)
        rng, action_key, step_key = jax.random.split(rng,3)
        action_rl = pi.sample(seed=action_key)
        action_list.append(action_rl)
        print("ACTION OKAY", action_rl)

        # Update action history
        prev_actions_rl = jnp.roll(prev_actions_rl, shift=1, axis=1)
        prev_actions_rl = prev_actions_rl.at[:, 0].set(action_rl)
        print("PREV ACTIONS OKAY", prev_actions_rl)
   
        # Apply the action to the small environment only
        batch_cfg = BatchConfig()
        action_type = batch_cfg.action_type

        # Step only the small environment (global environment remains unchanged)
        current_state = env_manager.small_env_timestep.state
        current_env_cfg = env_manager.small_env_timestep.env_cfg
    
        # Extract the map data from current state that's needed for step method
        current_target_map = current_state.world.target_map.map
        current_padding_mask = current_state.world.padding_mask.map
        current_dumpability_mask_init = current_state.world.dumpability_mask_init.map
    
        # Extract trench data from state
        current_trench_axes = current_state.world.trench_axes
        current_trench_type = current_state.world.trench_type

        # env_manager.small_env_timestep = env_manager.small_env.step(
        #         env_manager.small_env_timestep, 
        #         wrap_action(action_rl, action_type), 
        #         jax.random.split(step_key, 1)
        # )
        # print("Wrap action", wrap_action(action_rl, action_type))
        # print("Wrap action", wrap_action2(action_rl, action_type))
        # print("Current state", current_state)
        # print("Current target map", current_target_map.shape)
        # print("Current padding mask", current_padding_mask.shape)
        # print("Current dumpability mask init", current_dumpability_mask_init.shape)
        # print("Current trench axes", current_trench_axes.shape)
        # print("Current trench type", current_trench_type.shape)
        # print("Current env cfg", current_env_cfg)
        env_manager.small_env_timestep = env_manager.small_env.step(
            state=current_state,
            action=wrap_action2(action_rl, action_type),
            target_map=current_target_map,
            padding_mask=current_padding_mask,
            trench_axes=current_trench_axes,
            trench_type=current_trench_type,
            dumpability_mask_init=current_dumpability_mask_init,
            env_cfg=current_env_cfg
        )
        print("TIMESTEP OKAY")
        
        # Check if this small environment task is completed
        # if env_manager.is_small_task_completed():
        #     print(f"Partition {current_partition_idx} task completed!")
        #     env_manager.update_global_maps_from_small_environment()
            
        #     # Optionally switch back to global view momentarily to show progress
        #     env_manager.switch_to_global_environment()
        #     # Could add a delay or wait for user input here
            
        #     # Get next partition index (will be handled in next loop iteration)
        #     next_partition = env_manager.get_next_pending_partition_idx()
        #     if next_partition is None:
        #         print("All partitions completed!")
        #         playing = False
        #         break
        
        step += 1

    # After all partitions are complete, switch back to global view to show final result
    # if env_manager.all_partitions_completed():
    #     env_manager.switch_to_global_environment()
    #     print("Experiment completed successfully!")
        
    return frames, reward_seq, obs_seq, action_list

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