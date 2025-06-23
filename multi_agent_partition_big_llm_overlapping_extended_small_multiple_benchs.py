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
GRID_RENDERING = False
ORIGINAL_MAP_SIZE = 64

from dataclasses import dataclass, asdict

class LargeMapTerraEnv(TerraEnvBatchWithMapOverride):
    """A version of TerraEnvBatch specifically for 128x128 maps"""
    
    def reset_with_map_override(self, env_cfgs, rngs, custom_pos=None, custom_angle=None,
                           target_map_override=None, traversability_mask_override=None,
                           padding_mask_override=None, dumpability_mask_override=None,
                           dumpability_mask_init_override=None, action_map_override=None,
                           dig_map_override=None, agent_config_override=None):
        """Reset with 64x64 map overrides - ensures shapes are consistent"""
    
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

        # Overlapping partition data - will be set externally
        self.partitions = []
        self.overlap_map = {}  # Maps partition_id -> set of overlapping partition_ids
        self.overlap_regions = {}  # Cache overlap region calculations
        
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

    def _partitions_overlap_fixed(self, i: int, j: int) -> bool:
        """FIXED: Check if two partitions overlap."""
        p1_coords = self.partitions[i]['region_coords']
        p2_coords = self.partitions[j]['region_coords']
        
        y1_start, x1_start, y1_end, x1_end = p1_coords
        y2_start, x2_start, y2_end, x2_end = p2_coords
        
        # print(f"Checking overlap between partition {i}: ({y1_start}, {x1_start}, {y1_end}, {x1_end}) and partition {j}: ({y2_start}, {x2_start}, {y2_end}, {x2_end})")
        
        # Check for overlap - rectangles overlap if they overlap in BOTH dimensions
        y_overlap = (y1_start <= y2_end) and (y2_start <= y1_end)
        x_overlap = (x1_start <= x2_end) and (x2_start <= x1_end)
        
        overlap_exists = y_overlap and x_overlap
        
        # print(f"  Y overlap: {y_overlap} (y1: {y1_start}-{y1_end}, y2: {y2_start}-{y2_end})")
        # print(f"  X overlap: {x_overlap} (x1: {x1_start}-{x1_end}, x2: {x2_start}-{x2_end})")
        # print(f"  Overall overlap: {overlap_exists}")
        
        return overlap_exists


    def _calculate_overlap_region_fixed(self, partition_i: int, partition_j: int):
            """
            FIXED: Calculate the overlapping region between two partitions.
            Returns slices for global coordinates, partition i local coordinates, and partition j local coordinates.
            """
            p1_coords = self.partitions[partition_i]['region_coords']
            p2_coords = self.partitions[partition_j]['region_coords']
            
            y1_start, x1_start, y1_end, x1_end = p1_coords
            y2_start, x2_start, y2_end, x2_end = p2_coords
            
            # print(f"Calculating overlap region between partition {partition_i} and {partition_j}")
            # print(f"  Partition {partition_i}: ({y1_start}, {x1_start}) to ({y1_end}, {x1_end})")
            # print(f"  Partition {partition_j}: ({y2_start}, {x2_start}) to ({y2_end}, {x2_end})")
            
            # Find intersection in global coordinates
            overlap_y_start = max(y1_start, y2_start)
            overlap_x_start = max(x1_start, x2_start)
            overlap_y_end = min(y1_end, y2_end)
            overlap_x_end = min(x1_end, x2_end)
            
            # print(f"  Global overlap region: ({overlap_y_start}, {overlap_x_start}) to ({overlap_y_end}, {overlap_x_end})")
            
            # Check if there's actual overlap
            if overlap_y_start > overlap_y_end or overlap_x_start > overlap_x_end:
                print(f"  No actual overlap!")
                return None
            
            # Convert to local coordinates for each partition
            local_i_y_start = overlap_y_start - y1_start
            local_i_x_start = overlap_x_start - x1_start
            local_i_y_end = overlap_y_end - y1_start
            local_i_x_end = overlap_x_end - x1_start
            
            local_j_y_start = overlap_y_start - y2_start
            local_j_x_start = overlap_x_start - x2_start
            local_j_y_end = overlap_y_end - y2_start
            local_j_x_end = overlap_x_end - x2_start
            
            # print(f"  Partition {partition_i} local overlap: ({local_i_y_start}, {local_i_x_start}) to ({local_i_y_end}, {local_i_x_end})")
            # print(f"  Partition {partition_j} local overlap: ({local_j_y_start}, {local_j_x_start}) to ({local_j_y_end}, {local_j_x_end})")
            
            return {
                'global_slice': (slice(overlap_y_start, overlap_y_end + 1), 
                            slice(overlap_x_start, overlap_x_end + 1)),
                'partition_i_slice': (slice(local_i_y_start, local_i_y_end + 1), 
                                    slice(local_i_x_start, local_i_x_end + 1)),
                'partition_j_slice': (slice(local_j_y_start, local_j_y_end + 1),
                                    slice(local_j_x_start, local_j_x_end + 1)),
                'overlap_bounds': (overlap_y_start, overlap_x_start, overlap_y_end, overlap_x_end)
            }
    def debug_partition_setup(self):
        """
        Debug the partition setup to verify everything is correct.
        """
        print(f"\n=== PARTITION SETUP DEBUG ===")
        
        print(f"Number of partitions: {len(self.partitions)}")
        for i, partition in enumerate(self.partitions):
            print(f"  Partition {i}: {partition}")
        
        print(f"\nOverlap map: {dict(self.overlap_map)}")
        print(f"Overlap regions: {list(self.overlap_regions.keys())}")
        
        # Test specific pair
        if len(self.partitions) >= 2:
            print(f"\nTesting partitions 0 and 1 specifically:")
            overlap_exists = self._partitions_overlap_fixed(0, 1)
            print(f"Overlap exists: {overlap_exists}")
            
            if overlap_exists:
                overlap_info = self._calculate_overlap_region_fixed(0, 1)
                print(f"Overlap info: {overlap_info}")

    def set_partitions_fixed(self, partitions):
        """
        FIXED: Set the partitions and compute overlap relationships.
        """
        print(f"\n=== SETTING PARTITIONS ===")
        self.partitions = partitions
        
        print(f"Partitions set:")
        for i, partition in enumerate(self.partitions):
            print(f"  Partition {i}: {partition}")
        
        # Use the fixed overlap computation
        self._compute_overlap_relationships_fixed()
        
        print(f"Set {len(self.partitions)} partitions with overlaps computed.")

    def _compute_overlap_relationships_fixed(self):
        """
        FIXED: Compute which partitions overlap with each other and cache overlap regions.
        """
        print(f"\n=== COMPUTING OVERLAP RELATIONSHIPS ===")
        
        self.overlap_map = {i: set() for i in range(len(self.partitions))}
        self.overlap_regions = {}
        
        for i in range(len(self.partitions)):
            for j in range(i + 1, len(self.partitions)):
                print(f"\nChecking partitions {i} and {j}:")
                
                if self._partitions_overlap_fixed(i, j):
                    self.overlap_map[i].add(j)
                    self.overlap_map[j].add(i)
                    
                    # Cache the overlap region calculation
                    overlap_info = self._calculate_overlap_region_fixed(i, j)
                    if overlap_info is not None:
                        self.overlap_regions[(i, j)] = overlap_info
                        self.overlap_regions[(j, i)] = overlap_info  # Symmetric
                        print(f"  Stored overlap info for partitions {i} <-> {j}")
                    else:
                        print(f"  Could not calculate overlap region!")
                else:
                    print(f"  No overlap detected")
        
        # Print final overlap information
        print(f"\n=== FINAL OVERLAP RELATIONSHIPS ===")
        for i, partition in enumerate(self.partitions):
            overlaps = list(self.overlap_map[i])
            print(f"Partition {i}: region={partition['region_coords']}, overlaps with {overlaps}")
        
        print(f"Total overlap regions cached: {len(self.overlap_regions)}")
    # Fixed initialization for your DisjointMapEnvironments class
    def initialize_with_fixed_overlaps(self, partitions):
        """
        Initialize partitions with fixed overlap detection.
        Call this instead of set_partitions.
        """
        #print(f"\n=== INITIALIZING WITH FIXED OVERLAPS ===")
        
        # Set partitions using the fixed method
        self.set_partitions_fixed(partitions)
        
        # Debug the setup
        #self.debug_partition_setup()
        
        # Verify overlaps were detected
        total_overlaps = len(self.overlap_regions)
        #print(f"\nTotal overlap regions detected: {total_overlaps}")
        
        # if total_overlaps == 0:
        #     #print("WARNING: No overlaps detected! This may be incorrect.")
        # else:
        #     #print("✓ Overlaps detected successfully")


    def sync_terrain_between_overlapping_partitions(self, partition_states):
        """
        Now that overlap detection works, implement actual terrain synchronization.
        """
        #print(f"\n=== SYNCHRONIZING TERRAIN BETWEEN OVERLAPPING PARTITIONS ===")
        
        if not self.overlap_regions:
            print("No overlap regions found!")
            return
        
        # Process each unique overlap pair (avoid duplicates)
        processed_pairs = set()
        
        for (i, j), overlap_info in self.overlap_regions.items():
            # Only process each pair once
            pair_key = tuple(sorted([i, j]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Check if both partitions are active
            if (i not in partition_states or j not in partition_states or
                partition_states[i]['status'] != 'active' or 
                partition_states[j]['status'] != 'active'):
                continue
            
            #print(f"\nSyncing terrain between partitions {i} and {j}")
            
            # Get the overlap region info
            slice_i = overlap_info['partition_i_slice']
            slice_j = overlap_info['partition_j_slice']
            
            # Get current maps
            state_i = partition_states[i]['timestep'].state
            state_j = partition_states[j]['timestep'].state
            
            # Get traversability maps
            trav_i = state_i.world.traversability_mask.map
            trav_j = state_j.world.traversability_mask.map
            
            # print(f"  Partition {i} map shape: {trav_i.shape}, overlap slice: {slice_i}")
            # print(f"  Partition {j} map shape: {trav_j.shape}, overlap slice: {slice_j}")
            
            # Extract overlap data
            overlap_data_i = trav_i[slice_i]
            overlap_data_j = trav_j[slice_j]
            
            #print(f"  Overlap data shapes: {overlap_data_i.shape} vs {overlap_data_j.shape}")
            
            # Check if terrain differs (ignore agent positions -1)
            terrain_mask_i = (overlap_data_i > 0)  # Terrain obstacles
            terrain_mask_j = (overlap_data_j > 0)  # Terrain obstacles
            
            terrain_matches = jnp.array_equal(terrain_mask_i, terrain_mask_j)
            #print(f"  Terrain matches: {terrain_matches}")
            
            if not terrain_matches:
                #print(f"  Synchronizing terrain...")
                
                # Create unified terrain (union of both)
                unified_terrain = terrain_mask_i | terrain_mask_j
                
                # Update partition i overlap region
                new_overlap_i = jnp.where(
                    unified_terrain & (overlap_data_i != -1),  # Set terrain where needed, but don't touch agents
                    1,  # Terrain obstacle
                    jnp.where(
                        ~unified_terrain & (overlap_data_i > 0),  # Clear terrain where it shouldn't be
                        0,  # Free space
                        overlap_data_i  # Keep everything else (including agents)
                    )
                )
                
                # Update partition j overlap region
                new_overlap_j = jnp.where(
                    unified_terrain & (overlap_data_j != -1),  # Set terrain where needed, but don't touch agents
                    1,  # Terrain obstacle
                    jnp.where(
                        ~unified_terrain & (overlap_data_j > 0),  # Clear terrain where it shouldn't be
                        0,  # Free space
                        overlap_data_j  # Keep everything else (including agents)
                    )
                )
                
                # Update partition i
                updated_trav_i = trav_i.at[slice_i].set(new_overlap_i)
                updated_world_i = self._update_world_map(state_i.world, 'traversability_mask', updated_trav_i)
                updated_state_i = state_i._replace(world=updated_world_i)
                updated_timestep_i = partition_states[i]['timestep']._replace(state=updated_state_i)
                partition_states[i]['timestep'] = updated_timestep_i
                
                # Update partition j
                updated_trav_j = trav_j.at[slice_j].set(new_overlap_j)
                updated_world_j = self._update_world_map(state_j.world, 'traversability_mask', updated_trav_j)
                updated_state_j = state_j._replace(world=updated_world_j)
                updated_timestep_j = partition_states[j]['timestep']._replace(state=updated_state_j)
                partition_states[j]['timestep'] = updated_timestep_j
                
                #print(f"  ✓ Terrain synchronized between partitions {i} and {j}")
            
            # Also sync other environmental maps
            environmental_maps = ['action_map', 'dig_map', 'dumpability_mask', 'target_map']
            
            for map_name in environmental_maps:
                if hasattr(state_i.world, map_name) and hasattr(state_j.world, map_name):
                    map_i = getattr(state_i.world, map_name).map
                    map_j = getattr(state_j.world, map_name).map
                    
                    overlap_map_i = map_i[slice_i]
                    overlap_map_j = map_j[slice_j]
                    
                    if not jnp.array_equal(overlap_map_i, overlap_map_j):
                        #print(f"  Syncing {map_name}...")
                        
                        # For environmental maps, use the "more advanced" state
                        # (e.g., if one has been dug and the other hasn't, use the dug state)
                        if map_name == 'action_map':
                            # Use the state with more changes (more dug/dumped areas)
                            changes_i = jnp.sum(overlap_map_i != 0)
                            changes_j = jnp.sum(overlap_map_j != 0)
                            
                            if changes_i >= changes_j:
                                source_data = overlap_map_i
                                target_map = map_j
                                target_slice = slice_j
                                target_partition = j
                            else:
                                source_data = overlap_map_j
                                target_map = map_i
                                target_slice = slice_i
                                target_partition = i
                            
                            # Update the target map
                            updated_map = target_map.at[target_slice].set(source_data)
                            updated_world = self._update_world_map(
                                partition_states[target_partition]['timestep'].state.world, 
                                map_name, 
                                updated_map
                            )
                            updated_state = partition_states[target_partition]['timestep'].state._replace(world=updated_world)
                            updated_timestep = partition_states[target_partition]['timestep']._replace(state=updated_state)
                            partition_states[target_partition]['timestep'] = updated_timestep
                            
                            #print(f"    ✓ Synced {map_name} to partition {target_partition}")


    def add_agents_using_existing_representation(self, partition_states):
        """
        Alternative approach: Extract agent representation from the other partition's traversability mask.
        This preserves the exact shape and orientation as calculated by the environment.
        """
        #print(f"\nAdding agents using existing representation...")
        
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
            
            target_partition = self.partitions[target_partition_idx]
            target_region_coords = target_partition['region_coords']
            target_y_start, target_x_start, target_y_end, target_x_end = target_region_coords
            
            # Get current traversability
            current_timestep = target_partition_state['timestep']
            traversability = current_timestep.state.world.traversability_mask.map.copy()
            
            agents_added = 0
            for other_partition_idx, other_partition_state in partition_states.items():
                if (other_partition_idx == target_partition_idx or 
                    other_partition_state['status'] != 'active'):
                    continue
                
                # Get the other partition's traversability mask (which shows the agent as -1)
                other_traversability = other_partition_state['timestep'].state.world.traversability_mask.map
                
                # Find where the agent is in the other partition (value = -1)
                agent_mask = (other_traversability == -1)
                agent_positions = jnp.where(agent_mask)
                
                if len(agent_positions[0]) > 0:
                   # print(f"  Found agent {other_partition_idx} with {len(agent_positions[0])} occupied cells")
                    
                    # Convert each agent cell to global coordinates, then to target local coordinates
                    other_partition = self.partitions[other_partition_idx]
                    other_region_coords = other_partition['region_coords']
                    
                    cells_added = 0
                    for i in range(len(agent_positions[0])):
                        # Agent position in other partition's local coordinates
                        other_local_y = agent_positions[0][i]
                        other_local_x = agent_positions[1][i]
                        
                        # Convert to global coordinates
                        global_y = other_local_y + other_region_coords[0]
                        global_x = other_local_x + other_region_coords[1]
                        
                        # Check if this cell is visible in the target partition
                        if (target_y_start <= global_y <= target_y_end and 
                            target_x_start <= global_x <= target_x_end):
                            
                            # Convert to target partition's local coordinates
                            target_local_y = global_y - target_y_start
                            target_local_x = global_x - target_x_start
                            
                            # Mark as obstacle in target partition (only if it's currently free space)
                            if (0 <= target_local_y < traversability.shape[0] and 
                                0 <= target_local_x < traversability.shape[1]):
                                if traversability[target_local_y, target_local_x] == 0:
                                    traversability = traversability.at[target_local_y, target_local_x].set(1)
                                    cells_added += 1
                    
                    if cells_added > 0:
                        agents_added += 1
                        #print(f"    ✓ Added {cells_added} cells from agent {other_partition_idx}")
            
            # Update the traversability mask if we added agents
            if agents_added > 0:
                updated_world = self._update_world_map(
                    current_timestep.state.world, 
                    'traversability_mask', 
                    traversability
                )
                updated_state = current_timestep.state._replace(world=updated_world)
                updated_timestep = current_timestep._replace(state=updated_state)
                
                partition_states[target_partition_idx]['timestep'] = updated_timestep
                
               # print(f"  ✓ Added {agents_added} agents with exact representation to partition {target_partition_idx}")


    def complete_synchronization_with_full_agents(self, partition_states):
        """
        Complete synchronization with full agent representation.
        """
        #print(f"\n=== COMPLETE SYNCHRONIZATION WITH FULL AGENTS ===")
        
        # Step 1: Sync terrain and environmental maps between overlapping regions
        self.sync_terrain_between_overlapping_partitions(partition_states)
        
        # Step 2: Add other agents with their full representation

        self.add_agents_using_existing_representation(partition_states)
        
        #print("✓ Complete synchronization with full agents finished")

    # Simple step function that doesn't do any sync
    def step_simple(self, partition_idx: int, action, partition_states: dict):
        """
        Simple step function - just steps the environment without any synchronization.
        Synchronization happens separately.
        """
        partition_state = partition_states[partition_idx]
        current_state = partition_state['timestep'].state
        current_env_cfg = partition_state['timestep'].env_cfg
        
        # Extract required data for step
        current_target_map = current_state.world.target_map.map
        current_padding_mask = current_state.world.padding_mask.map
        current_dumpability_mask_init = current_state.world.dumpability_mask_init.map
        current_trench_axes = current_state.world.trench_axes
        current_trench_type = current_state.world.trench_type
        
        # Step the environment
        new_timestep = self.small_env.step(
            state=current_state,
            action=action,
            target_map=current_target_map,
            padding_mask=current_padding_mask,
            trench_axes=current_trench_axes,
            trench_type=current_trench_type,
            dumpability_mask_init=current_dumpability_mask_init,
            env_cfg=current_env_cfg
        )
        
        return new_timestep

    def _create_clean_env_config(self):
        """Create a clean environment config for 64x64 maps without batch dimensions"""

    
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
            'target_map': create_sub_task_target_map_64x64(self.global_maps['target_map'], region_coords),
            'action_map': create_sub_task_action_map_64x64(self.global_maps['action_map'], region_coords),
            'dumpability_mask': create_sub_task_dumpability_mask_64x64(self.global_maps['dumpability_mask'], region_coords),
            'dumpability_mask_init': create_sub_task_dumpability_mask_64x64(self.global_maps['dumpability_mask_init'], region_coords),
            'padding_mask': create_sub_task_padding_mask_64x64(self.global_maps['padding_mask'], region_coords),
            'traversability_mask': create_sub_task_traversability_mask_64x64(self.global_maps['traversability_mask'], region_coords),
            'dig_map': create_sub_task_action_map_64x64(self.global_maps['dig_map'], region_coords),
        }
        #DIAGNOSTIC: Check sub-map validity
        print(f"=== SUB-MAP DIAGNOSTICS ===")
        for name, map_data in sub_maps.items():
            print(f"{name}:")
            print(f"  Shape: {map_data.shape}")
        

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

    def _update_world_map(self, world_state, map_name: str, new_map):
        """
        Helper method to update a specific map in the world state.
        This creates a new world state with the updated map.
        """
        # Get the current map object
        current_map_obj = getattr(world_state, map_name)
        
        # Create new map object with updated data
        updated_map_obj = current_map_obj._replace(map=new_map)
        
        # Create new world state with updated map
        updated_world = world_state._replace(**{map_name: updated_map_obj})
        
        return updated_world

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

            #print(f"Global environment updated with {len(all_agent_positions)} active agents.")
        
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
            
                # print(f"Partition {partition_idx} region: ({y_start}, {x_start}) to ({y_end}, {x_end})")
                # print(f"Expected region size: {region_height} x {region_width}")
            
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
            
                # print(f"Small environment map shapes:")
                # for name, map_data in small_maps.items():
                #     print(f"  {name}: {map_data.shape}")
            
                # Extract only the relevant portion from the 64x64 small maps
                # that corresponds to the actual region size
                extract_height = min(region_height, 64)
                extract_width = min(region_width, 64)
            
                for map_name, small_map in small_maps.items():
                    # Extract the portion that matches the region size
                    extracted_region = small_map[:extract_height, :extract_width]
                
                    #print(f"  Extracted {map_name}: {extracted_region.shape} -> Global region: {region_height}x{region_width}")
                
                    # Update the global map with the extracted region
                    try:
                        self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(extracted_region)
                    except ValueError as e:
                        #print(f"  WARNING: Shape mismatch for {map_name}: {e}")
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
                        
                            #print(f"  Adjusted {map_name}: {extracted_region.shape}")
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
                #print(f"Passing {len(self.global_env.all_agent_positions)} agents to renderer")
            else:
                #print("Warning: Agent attributes not properly initialized for rendering")
                # Initialize empty lists to prevent further errors
                info['additional_agents'] = {
                    'positions': [],
                    'angles base': [],
                    'angles cabin': [],
                    'loaded': []
                }

            if VISUALIZE_PARTITIONS:
                info['show_partitions'] = True
                info['partitions'] = self.partitions  # Just pass the whole partition list

            self.global_env.terra_env.render_obs_pygame(obs, info)
    
        except Exception as e:
            print(f"Global rendering error: {e}")
            import traceback
            traceback.print_exc()

    def render_all_partition_views_grid(self, partition_states):
        """
        Render all active partition views in a grid layout.
        This shows what each agent sees simultaneously.
        """
        import pygame as pg
        import numpy as np
        
        active_partitions = [idx for idx, state in partition_states.items() 
                            if state['status'] == 'active']
        
        if not active_partitions:
            return
        
        # Get screen dimensions
        screen = pg.display.get_surface()
        if screen is None:
            return
        
        screen_width, screen_height = screen.get_size()
        
        # Calculate grid layout
        num_partitions = len(active_partitions)
        cols = min(2, num_partitions)  # Max 2 columns
        rows = (num_partitions + cols - 1) // cols
        
        # Calculate size for each partition view
        partition_width = screen_width // cols
        partition_height = screen_height // rows
        
        # Clear screen
        screen.fill((50, 50, 50))
        
        # Render each partition
        for i, partition_idx in enumerate(active_partitions):
            partition_state = partition_states[partition_idx]
            
            # Calculate position in grid
            col = i % cols
            row = i // cols
            x_offset = col * partition_width
            y_offset = row * partition_height
            
            # Render this partition's view
            self._render_single_partition_view(
                screen, partition_state, partition_idx,
                x_offset, y_offset, partition_width, partition_height
            )
        
        pg.display.flip()

    def _render_single_partition_view(self, screen, partition_state, partition_idx,
                                    x_offset, y_offset, width, height):
        """
        Render a single partition's view within the given screen area.
        """
        import pygame as pg
        import numpy as np
        
        # Get the maps from the partition
        current_timestep = partition_state['timestep']
        world = current_timestep.state.world
        agent_state = current_timestep.state.agent.agent_state
        
        # Extract maps
        target_map = world.target_map.map
        action_map = world.action_map.map
        traversability_mask = world.traversability_mask.map
        agent_pos = agent_state.pos_base
        
        # Map dimensions
        map_height, map_width = target_map.shape
        
        # Calculate tile size to fit in available space
        tile_width = (width - 40) // map_width  # Leave 40 pixels for margins
        tile_height = (height - 60) // map_height  # Leave 60 pixels for title and info
        tile_size = max(2, min(tile_width, tile_height))
        
        # Center the map in the available space
        map_pixel_width = map_width * tile_size
        map_pixel_height = map_height * tile_size
        map_x = x_offset + (width - map_pixel_width) // 2
        map_y = y_offset + 40  # Leave space for title
        
        # Draw title
        font = pg.font.Font(None, 32)
        title = f"Partition {partition_idx}"
        text = font.render(title, True, (255, 255, 255))
        screen.blit(text, (x_offset + 10, y_offset + 5))
        
        # Draw the map
        for y in range(map_height):
            for x in range(map_width):
                # Get cell values
                target_val = target_map[y, x]
                action_val = action_map[y, x]
                traversable = traversability_mask[y, x]
                
                # Determine color based on cell state
                if traversable == -1:  # Agent position
                    color = (255, 100, 255)  # Magenta
                elif traversable == 1:   # Obstacle (including other agents)
                    color = (255, 50, 50)   # Red
                elif action_val > 0:     # Dumped soil
                    color = (139, 69, 19)   # Brown
                elif action_val == -1:   # Dug area
                    color = (101, 67, 33)   # Dark brown
                elif target_val == -1:   # Target to dig
                    color = (255, 255, 0)   # Yellow
                elif target_val == 1:    # Target to dump
                    color = (0, 255, 0)     # Green
                else:                    # Free space
                    color = (220, 220, 220) # Light gray
                
                # Draw the tile
                rect = pg.Rect(
                    map_x + x * tile_size,
                    map_y + y * tile_size,
                    tile_size,
                    tile_size
                )
                pg.draw.rect(screen, color, rect)
        
        # Draw border around map
        border_rect = pg.Rect(map_x - 1, map_y - 1, 
                            map_pixel_width + 2, map_pixel_height + 2)
        pg.draw.rect(screen, (255, 255, 255), border_rect, 1)
        
        # Draw agent position and stats
        small_font = pg.font.Font(None, 20)
        
        # Agent position
        pos_text = f"Agent: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})"
        pos_surface = small_font.render(pos_text, True, (255, 255, 255))
        screen.blit(pos_surface, (x_offset + 10, y_offset + height - 40))
        
        # Obstacle count (red cells = other agents + terrain obstacles)
        obstacle_count = np.sum(traversability_mask == 1)
        obstacle_text = f"Red obstacles: {obstacle_count}"
        obstacle_surface = small_font.render(obstacle_text, True, (255, 100, 100))
        screen.blit(obstacle_surface, (x_offset + 10, y_offset + height - 20))
    
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
    
    print(f"Completion check: {completed_partitions=}, {failed_partitions=}, {active_partitions=}")
    print(f"Completion rate: {completion_rate:.2%}, Overall complete: {is_complete}")
    
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


def run_experiment_with_disjoint_environments(
    llm_model_name, llm_model_key, num_timesteps, seed, 
    progressive_gif, run, small_env_config=None):
    """
    Run an experiment with completely separate environments for global and small maps.
    Modified to cycle through maps in the existing environment instead of creating new ones.
    """
    agent_checkpoint_path = run
    model = None
    model_params = None
    config = None

    print(f"Loading RL agent configuration from: {agent_checkpoint_path}")
    log = load_pkl_object(agent_checkpoint_path)
    config = log["train_config"]
    model_params = log["model"]

    # Create the original environment configs for the full map
    global_env_config = jax.tree_map(
        lambda x: x[0][None, ...].repeat(1, 0), log["env_config"]
    ) 

    config.num_test_rollouts = 1
    config.num_devices = 1
    config.num_embeddings_agent_min = 60

    # Initialize the environment manager ONCE with all maps
    print("Initializing environment manager with all maps...")
    env_manager = DisjointMapEnvironments(
        seed=seed,
        global_env_config=global_env_config,
        small_env_config=small_env_config,
        progressive_gif=progressive_gif,
        shuffle_maps=False  # This will give us access to different maps
    )
    print("Environment manager initialized.")

    # Initialize once with proper batching
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset_initial = jax.random.split(_rng, 1)

    initial_custom_pos = None
    initial_custom_angle = None
    
    # Initial setup
    env_manager.global_env.timestep = env_manager.global_env.reset(
        global_env_config, rng_reset_initial, initial_custom_pos, initial_custom_angle
    )

    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type

    def repeat_action(action, n_times=1):
        return action_type.new(action.action[None].repeat(n_times, 0))
    
    # Trigger the JIT compilation
    env_manager.global_env.timestep = env_manager.global_env.step(
        env_manager.global_env.timestep, repeat_action(action_type.do_nothing()), rng_reset_initial
    )
    env_manager.global_env.terra_env.render_obs_pygame(
        env_manager.global_env.timestep.observation, env_manager.global_env.timestep.info
    )

    # Initialize variables for tracking progress across all maps
    global_step = 0
    playing = True
    current_map_index = 0
    max_maps = 10  # Set a reasonable limit for number of maps to process
    
    # For visualization and metrics across all maps
    all_frames = []
    all_reward_seq = []
    all_global_step_rewards = []
    all_obs_seq = []
    all_action_list = []
    
    def reset_to_next_map(map_index):
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
        env_manager._initialize_global_environment()
        
        print(f"Environment reset to map {map_index}")
        return map_rng

    def setup_partitions_and_llm(map_index):
        """Setup partitions and LLM for the current map"""
        action_size = 7
        
        # Define partitions based on map size
        if ORIGINAL_MAP_SIZE == 64:
            sub_tasks_manual = [
                {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (25, 20), 'start_angle': 0, 'status': 'pending'},
            ]
        elif ORIGINAL_MAP_SIZE == 128:
            sub_tasks_manual = [
                {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (25, 20), 'start_angle': 0, 'status': 'pending'},
                # Add more partitions as needed
            ]
        else:
            raise ValueError(f"Unsupported ORIGINAL_MAP_SIZE: {ORIGINAL_MAP_SIZE}")

        # Initialize LLM agent (reuse if possible, or create new session)
        llm_query, runner, prev_actions, system_message_master = init_llms(
            llm_model_key, llm_model_name, USE_PATH, config, action_size, 1, 
            APP_NAME, USER_ID, f"{SESSION_ID}_map_{map_index}"  # Unique session per map
        )

        sub_tasks_llm = []
        
        if not USE_MANUAL_PARTITIONING:
            print("Calling LLM agent for partitioning decision...")
            screen = pg.display.get_surface()
            game_state_image = capture_screen(screen)
            current_observation = env_manager.global_env.timestep.observation
            
            try:
                obs_dict = {k: v.tolist() for k, v in current_observation.items()}
                observation_str = json.dumps(obs_dict)
            except AttributeError:
                observation_str = str(current_observation)

            if USE_IMAGE_PROMPT:
                prompt = f"Current observation: See image \n\nSystem Message: {system_message_master}"
            else:
                prompt = f"Current observation: {observation_str}\n\nSystem Message: {system_message_master}"

            try:
                if USE_IMAGE_PROMPT:
                    response = asyncio.run(call_agent_async_master(prompt, game_state_image, runner, USER_ID, f"{SESSION_ID}_map_{map_index}"))
                else:
                    response = asyncio.run(call_agent_async_master(prompt, game_state_image=None, runner=runner, USER_ID=USER_ID, SESSION_ID=f"{SESSION_ID}_map_{map_index}"))
        
                llm_response_text = response
                print(f"LLM response: {llm_response_text}")

                try:
                    sub_tasks_llm = extract_python_format_data(llm_response_text)
                    print("Successfully parsed LLM response with tuples preserved")
                except ValueError as e:
                    print(f"Extraction failed: {e}")
                    sub_tasks_llm = sub_tasks_manual

            except Exception as adk_err:
                print(f"Error during ADK agent partitioning: {adk_err}")
                sub_tasks_llm = sub_tasks_manual

        # Use appropriate partitions
        partition_validation = is_valid_region_list(sub_tasks_llm)
        
        if partition_validation and USE_MANUAL_PARTITIONING == False:
            print("Using LLM-generated sub-tasks.")
            env_manager.initialize_with_fixed_overlaps(sub_tasks_llm)
        else:
            print("Using manually defined sub-tasks.")
            env_manager.initialize_with_fixed_overlaps(sub_tasks_manual)

        return llm_query, runner, system_message_master

    def initialize_partitions_for_current_map():
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

    tile_size = global_env_config.tile_size[0].item()
    move_tiles = global_env_config.agent.move_tiles[0].item()


    action_type = batch_cfg.action_type
    if action_type == TrackedAction:
        move_actions = (TrackedActionType.FORWARD, TrackedActionType.BACKWARD)
        l_actions = ()
        do_action = TrackedActionType.DO
    elif action_type == WheeledAction:
        move_actions = (WheeledActionType.FORWARD, WheeledActionType.BACKWARD)
        l_actions = (WheeledActionType.CLOCK, WheeledActionType.ANTICLOCK)
        do_action = WheeledActionType.DO
    else:
        raise (ValueError(f"{action_type=}"))

    obs = env_manager.global_env.timestep.observation
    #obs = env_manager.global_maps
    areas = (obs["target_map"] == -1).sum(
        tuple([i for i in range(len(obs["target_map"].shape))][1:])
            ) * (tile_size**2)
    print(f"Areas: {areas}")
    target_maps_init = obs["target_map"].copy()
    #target_maps_init = env_manager.global_maps['target_map'].copy()
    dig_tiles_per_target_map_init = (target_maps_init == -1).sum(
        tuple([i for i in range(len(target_maps_init.shape))][1:])
    )
    t_counter = 0
    reward_seq = []
    episode_done_once = None
    episode_length = None
    move_cumsum = None
    do_cumsum = None
    obs_seq = {}

    def _append_to_obs(o, obs_log):
        if obs_log == {}:
            return {k: v[:, None] for k, v in o.items()}
        obs_log = {
            k: jnp.concatenate((v, o[k][:, None]), axis=1) for k, v in obs_log.items()
        }
        return obs_log

    # MAIN LOOP - PROCESS MULTIPLE MAPS
    while playing and global_step < num_timesteps and current_map_index < max_maps:
        #obs_seq = _append_to_obs(obs, obs_seq)
        print(f"\n{'='*80}")
        print(f"STARTING MAP {current_map_index}")
        print(f"{'='*80}")
        
        # Reset to next map (reusing the same environment)
        try:
            if current_map_index > 0:  # Don't reset on first map since it's already initialized
                map_rng = reset_to_next_map(current_map_index)
            
            llm_query, runner, system_message_master = setup_partitions_and_llm(current_map_index)
            partition_states, partition_models, active_partitions = initialize_partitions_for_current_map()
            
            if partition_states is None:
                print(f"Failed to initialize map {current_map_index}, moving to next map")
                current_map_index += 1
                continue
                
        except Exception as e:
            print(f"Error setting up map {current_map_index}: {e}")
            current_map_index += 1
            continue

        # Track metrics for this map
        map_frames = []
        map_reward_seq = []
        map_global_step_rewards = []
        map_obs_seq = []
        map_action_list = []
        map_start_step = global_step
        
        # Calculate initial target pixels for this map
        current_map = env_manager.global_timestep.state.world.target_map.map[0]
        initial_target_num = jnp.sum(current_map < 0)
        print(f"Map {current_map_index} - Initial target number: {initial_target_num}")

        llm_decision = "delegate_to_rl"



        # MAP-SPECIFIC GAME LOOP
        map_step = 0
        #max_steps_per_map = num_timesteps // 3  # Allow reasonable time per map
        max_steps_per_map = num_timesteps 
        map_done = False  # Track map completion

        while playing and active_partitions and map_step < max_steps_per_map and global_step < num_timesteps:
            # Handle quit events
            for event in pg.event.get():
                if event.type == QUIT or (event.type == pg.KEYDOWN and event.key == K_q):
                    playing = False

            print(f"\nMap {current_map_index}, Step {map_step} (Global {global_step}) - "
                  f"Processing {len(active_partitions)} active partitions")

            # Capture screen state
            screen = pg.display.get_surface()
            game_state_image = capture_screen(screen)
            map_frames.append(game_state_image)

            # Step all active partitions simultaneously
            partitions_to_remove = []
            current_step_reward = 0.0

            for partition_idx in active_partitions:
                partition_state = partition_states[partition_idx]
            
                print(f"  Processing partition {partition_idx} (partition step {partition_state['step_count']})")

                try:
                    # Set the small environment to the current partition's state
                    env_manager.small_env_timestep = partition_state['timestep']
                    env_manager.current_partition_idx = partition_idx

                    # Get the current observation
                    current_observation = env_manager.small_env_timestep.observation
                    map_obs_seq.append(current_observation)

                    # Extract partition info and create subsurface
                    partition_info = env_manager.partitions[partition_idx]
                    region_coords = partition_info['region_coords']
                    y_start, x_start, y_end, x_end = region_coords
                    width = x_end - x_start + 1
                    height = y_end - y_start + 1

                    # Extract subsurface safely
                    try:
                        screen_width, screen_height = screen.get_size()
                        x_start = max(0, min(x_start, screen_width - 1))
                        y_start = max(0, min(y_start, screen_height - 1))
                        width = min(width, screen_width - x_start)
                        height = min(height, screen_height - y_start)
                        subsurface = screen.subsurface((x_start, y_start, width, height))
                    except ValueError as e:
                        print(f"Error extracting subsurface for partition {partition_idx}: {e}")
                        subsurface = screen.subsurface((0, 0, min(128, screen_width), min(128, screen_height)))
                    
                    game_state_image_small = capture_screen(subsurface)
                    state = env_manager.small_env_timestep.state
                    base_orientation = extract_base_orientation(state)
                    bucket_status = extract_bucket_status(state)

                    # LLM decision making (keeping the original logic but simplified for brevity)
                    if global_step % LLM_CALL_FREQUENCY == 0 and global_step > 0:
                        print("    Calling LLM agent for decision...")
                        # ... (keep the existing LLM decision logic)
                        
                    if FORCE_DELEGATE_TO_LLM:
                        llm_decision = "delegate_to_llm"
                    elif FORCE_DELEGATE_TO_RL:
                        llm_decision = "delegate_to_rl"

                    # Action selection
                    if llm_decision == "delegate_to_rl":
                        print(f"    Partition {partition_idx} - Delegating to RL agent")
                        try:
                            current_observation = env_manager.small_env_timestep.observation
                            batched_observation = add_batch_dimension_to_observation(current_observation)
                            obs = obs_to_model_input(batched_observation, partition_state['prev_actions_rl'], config)

                            current_model = partition_models[partition_idx]
                            _, logits_pi = current_model['model'].apply(current_model['params'], obs)
                            pi = tfp.distributions.Categorical(logits=logits_pi)

                            # Use map-specific random key
                            action_rng = jax.random.PRNGKey(seed + global_step * len(env_manager.partitions) + partition_idx + current_map_index * 10000)
                            action_rng, action_key, step_key = jax.random.split(action_rng, 3)
                            action_rl = pi.sample(seed=action_key)
                        
                            partition_state['actions'].append(action_rl)
                            map_action_list.append(action_rl)

                        except Exception as rl_error:
                            print(f"    ERROR getting action from RL model for partition {partition_idx}: {rl_error}")
                            action_rl = jnp.array(0)
                            partition_state['actions'].append(action_rl)
                            map_action_list.append(action_rl)

                    elif llm_decision == "delegate_to_llm":
                        print(f"    Partition {partition_idx} - Delegating to LLM agent")
                        
                        start = env_manager.small_env_timestep.state.agent.agent_state.pos_base
                        msg = (
                            f"Analyze this game frame and the provided local map to select the optimal action. "
                            f"The base of the excavator is currently facing {base_orientation['direction']}. "
                            f"The bucket is currently {bucket_status}. "
                            f"The excavator is currently located at {start} (y,x). "
                            f"Follow the format: {{\"reasoning\": \"detailed step-by-step analysis\", \"action\": X}}"
                        )

                        llm_query.add_user_message(frame=game_state_image_small, user_msg=msg, local_map=None)
                        action_output, reasoning = llm_query.generate_response("./")
                        print(f"\n    Action output: {action_output}, Reasoning: {reasoning}")
                        llm_query.add_assistant_message()

                        action_rl = jnp.array([action_output], dtype=jnp.int32)
                        map_action_list.append(action_rl)
                    
                    else:
                        print("    Master Agent stop.")
                        action_rl = jnp.array([-1], dtype=jnp.int32)
                        map_action_list.append(action_rl)

                    # Clear LLM messages periodically
                    if len(llm_query.messages) > 3:
                        llm_query.delete_messages()

                    # Update action history and step environment
                    partition_state['prev_actions_rl'] = jnp.roll(partition_state['prev_actions_rl'], shift=1, axis=1)
                    partition_state['prev_actions_rl'] = partition_state['prev_actions_rl'].at[:, 0].set(action_rl)

                    wrapped_action = wrap_action2(action_rl, action_type)
                    new_timestep = env_manager.step_simple(partition_idx, wrapped_action, partition_states)
                    partition_states[partition_idx]['timestep'] = new_timestep
                    partition_state['step_count'] += 1
                
                    # Process reward
                    reward = new_timestep.reward
                    if isinstance(reward, jnp.ndarray):
                        if reward.shape == ():
                            reward_val = float(reward)
                        elif len(reward.shape) > 0:
                            reward_val = float(reward.flatten()[0])
                        else:
                            reward_val = float(reward)
                    else:
                        reward_val = float(reward)
                    
                    if not (jnp.isnan(reward_val) or jnp.isinf(reward_val)):
                        partition_state['rewards'].append(reward_val)
                        partition_state['total_reward'] += reward_val
                        map_reward_seq.append(reward_val)
                        current_step_reward += reward_val
                        print(f"    Partition {partition_idx} - reward: {reward_val:.4f}, action: {action_rl}, done: {new_timestep.done}")
                    else:
                        print(f"    Partition {partition_idx} - INVALID reward: {reward_val}, action: {action_rl}, done: {new_timestep.done}")

                    # Check completion conditions
                    partition_completed = False
                
                    if env_manager.is_small_task_completed():
                        print(f"    Partition {partition_idx} COMPLETED after {partition_state['step_count']} steps!")
                        print(f"    Total reward for partition {partition_idx}: {partition_state['total_reward']:.4f}")
                        env_manager.partitions[partition_idx]['status'] = 'completed'
                        partition_state['status'] = 'completed'
                        partition_completed = True
                
                    elif partition_state['step_count'] >= max_steps_per_map // len(env_manager.partitions):
                        print(f"    Partition {partition_idx} TIMED OUT")
                        env_manager.partitions[partition_idx]['status'] = 'failed'
                        partition_state['status'] = 'failed'
                        partition_completed = True
                
                    elif jnp.isnan(reward):
                        print(f"    Partition {partition_idx} FAILED due to NaN reward")
                        env_manager.partitions[partition_idx]['status'] = 'failed'
                        partition_state['status'] = 'failed'
                        partition_completed = True
                
                    if partition_completed:
                        partitions_to_remove.append(partition_idx)

                except Exception as e:
                    print(f"    ERROR stepping partition {partition_idx}: {e}")
                    if partition_idx < len(env_manager.partitions):
                        env_manager.partitions[partition_idx]['status'] = 'failed'
                    partition_state['status'] = 'failed'
                    partitions_to_remove.append(partition_idx)

            # Synchronize environments
            env_manager.complete_synchronization_with_full_agents(partition_states)

            # Remove completed/failed partitions
            for partition_idx in partitions_to_remove:
                if partition_idx in active_partitions:
                    active_partitions.remove(partition_idx)
                    print(f"    Removed partition {partition_idx} from active list")

            print(f"    Remaining active partitions: {active_partitions}")
            map_global_step_rewards.append(current_step_reward)
            print(f"    Map {current_map_index} step {map_step} reward: {current_step_reward:.4f}")

            # Render
            if GRID_RENDERING:
                env_manager.render_all_partition_views_grid(partition_states)
            else:
                env_manager.render_global_environment_with_multiple_agents(partition_states)
            

            # After processing all partitions, check if map is complete
            map_metrics = calculate_map_completion_metrics(partition_states, env_manager)
            map_done = map_metrics['done']
            
            print(f"    Map {current_map_index} metrics: {map_metrics}")
            
            # Update done flag for this step
            done = jnp.array(map_done)  # Convert to JAX array for consistency
            
            # if map_done:
            #     print(f"Map {current_map_index} COMPLETED!")
            #     print(f"  Completion rate: {map_metrics['completion_rate']:.2%}")
            #     print(f"  Completed partitions: {map_metrics['completed_partitions']}")
            #     print(f"  Failed partitions: {map_metrics['failed_partitions']}")
            #     print(f"  Total reward: {map_metrics['total_reward']:.4f}")
            #     break

        
            map_step += 1
            global_step += 1
            #next_obs = env_manager.global_env.timestep.observation
            #done = env_manager.global_env.timestep.done
            reward_seq.append(current_step_reward)
            #obs = next_obs

            if episode_done_once is None:
                episode_done_once = done
            if episode_length is None:
                episode_length = jnp.zeros_like(done, dtype=jnp.int32)
            if move_cumsum is None:
                move_cumsum = jnp.zeros_like(done, dtype=jnp.int32)
            if do_cumsum is None:
                do_cumsum = jnp.zeros_like(done, dtype=jnp.int32)

            episode_done_once = episode_done_once | done
            episode_length += ~episode_done_once

            move_cumsum_tmp = jnp.zeros_like(done, dtype=jnp.int32)
            for move_action in move_actions:
                move_mask = (action_rl == move_action) * (~episode_done_once)
                move_cumsum_tmp += move_tiles * tile_size * move_mask.astype(jnp.int32)
            for l_action in l_actions:
                l_mask = (action_rl == l_action) * (~episode_done_once)
                move_cumsum_tmp += 2 * move_tiles * tile_size * l_mask.astype(jnp.int32)
            move_cumsum += move_cumsum_tmp

            do_cumsum += (action_rl == do_action) * (~episode_done_once)

            # dug_tiles_per_action_map = (obs["action_map"] == -1).sum(
            #     tuple([i for i in range(len(obs["action_map"].shape))][1:])
            # )
            dug_tiles_per_action_map = (env_manager.global_maps['action_map'] == -1).sum()
  
        # Add map data to global collections
        all_frames.extend(map_frames)
        all_reward_seq.extend(map_reward_seq)
        all_global_step_rewards.extend(map_global_step_rewards)
        all_obs_seq.extend(map_obs_seq)
        all_action_list.extend(map_action_list)
        
        # Move to next map
        current_map_index += 1
        
        # Check if we should continue
        if not playing or global_step >= num_timesteps:
            break
            
        print(f"\nTransitioning to map {current_map_index}...")

    # FINAL SUMMARY ACROSS ALL MAPS
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED - PROCESSED {current_map_index} MAPS")
    print(f"{'='*80}")

    # Save results
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model_name = llm_model_name.replace('/', '_')
    output_dir = os.path.join("experiments", f"{safe_model_name}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    # Save video
    video_path = os.path.join(output_dir, "gameplay_all_maps.mp4")
    save_video(all_frames, video_path)
        
    print(f"\nResults saved to: {output_dir}")
    print(f"Video: {video_path}")

    info = {
        "episode_done_once": episode_done_once,
        "episode_length": episode_length,
        "move_cumsum": move_cumsum,
        "do_cumsum": do_cumsum,
        "areas": areas,
        "dig_tiles_per_target_map_init": dig_tiles_per_target_map_init,
        "dug_tiles_per_action_map": dug_tiles_per_action_map,
    }

    return info


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
                 "gemini-2.5-pro-preview-05-06",
                 "gemini-2.5-flash-preview-04-17", 
                 "gemini-2.5-flash-preview-05-20",
                 "claude-3-haiku-20240307", 
                 "claude-3-7-sonnet-20250219",
                 "claude-opus-4-20250514",
                 "claude-sonnet-4-20250514",		
                 ], 
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

    NUM_ENVS = 5

    args = parser.parse_args()
    episode_done_once_list = []
    episode_length_list = []
    move_cumsum_list = []
    do_cumsum_list = []
    areas_list = []
    dig_tiles_per_target_map_init_list = []
    dug_tiles_per_action_map_list = []

    base_seed = args.seed

    for i in range(NUM_ENVS):
        print(f"Running experiment {i+1}/{NUM_ENVS} with args: {args}")
        info = run_experiment_with_disjoint_environments(args.model_name, 
                   args.model_key, 
                   args.num_timesteps, 
                   base_seed + i * 1000,  # Ensure different seeds
                   args.progressive_gif, 
                   args.run_name
                   )
        #print(info)
        # Collect results from this experiment
        episode_done_once = info["episode_done_once"]
        episode_length = info["episode_length"]
        move_cumsum = info["move_cumsum"]
        do_cumsum = info["do_cumsum"]
        areas = info["areas"]
        dig_tiles_per_target_map_init = info["dig_tiles_per_target_map_init"]
        dug_tiles_per_action_map = info["dug_tiles_per_action_map"]
        
        episode_done_once_list.append(episode_done_once.item())
        episode_length_list.append(episode_length.item())
        move_cumsum_list.append(move_cumsum.item())
        do_cumsum_list.append(do_cumsum.item())
        areas_list.append(areas.item())
        dig_tiles_per_target_map_init_list.append(dig_tiles_per_target_map_init.item())
        dug_tiles_per_action_map_list.append(dug_tiles_per_action_map.item())

    print("\nExperiment completed for all environments.")

    episode_done_once = jnp.array(episode_done_once_list)
    episode_length = jnp.array(episode_length_list)
    move_cumsum = jnp.array(move_cumsum_list)
    do_cumsum = jnp.array(do_cumsum_list)
    areas = jnp.array(areas_list)
    dig_tiles_per_target_map_init = jnp.array(dig_tiles_per_target_map_init_list)
    dug_tiles_per_action_map = jnp.array(dug_tiles_per_action_map_list)

    print("\nSummary of results across all environments:")
    print(f"Episode done once: {episode_done_once}")
    print(f"Episode length: {episode_length}")
    print(f"Move cumsum: {move_cumsum}")
    print(f"Do cumsum: {do_cumsum}")
    print(f"Areas: {areas}")
    print(f"Dig tiles per target map init: {dig_tiles_per_target_map_init}")
    print(f"Dug tiles per action map: {dug_tiles_per_action_map}")
    # print(type(episode_done_once))
    # print(type(episode_length))
    # print(type(move_cumsum))
    # print(type(do_cumsum))

    # Path efficiency -- only include finished envs
    move_cumsum *= episode_done_once
    path_efficiency = (move_cumsum / jnp.sqrt(areas))[episode_done_once]
    path_efficiency_std = path_efficiency.std()
    path_efficiency_mean = path_efficiency.mean()

    # Workspaces efficiency -- only include finished envs
    reference_workspace_area = 0.5 * np.pi * (8**2)
    n_dig_actions = do_cumsum // 2
    workspaces_efficiency = (
        reference_workspace_area
        * ((n_dig_actions * episode_done_once) / areas)[episode_done_once]
    )
    workspaces_efficiency_mean = workspaces_efficiency.mean()
    workspaces_efficiency_std = workspaces_efficiency.std()


    # Coverage scores
    # dug_tiles_per_action_map = (obs["action_map"] == -1).sum(
    #     tuple([i for i in range(len(obs["action_map"].shape))][1:])
    # )
    coverage_ratios = dug_tiles_per_action_map / dig_tiles_per_target_map_init
    coverage_scores = episode_done_once + (~episode_done_once) * coverage_ratios
    coverage_score_mean = coverage_scores.mean()
    coverage_score_std = coverage_scores.std()

    completion_rate = 100 * episode_done_once.sum() / len(episode_done_once)

    print("\nStats:\n")
    print(f"Completion: {completion_rate:.2f}%")
    # print(f"First episode length average: {episode_length.mean()}")
    # print(f"First episode length min: {episode_length.min()}")
    # print(f"First episode length max: {episode_length.max()}")
    print(
        f"Path efficiency: {path_efficiency_mean:.2f} ({path_efficiency_std:.2f})"
    )
    print(
        f"Workspaces efficiency: {workspaces_efficiency_mean:.2f} ({workspaces_efficiency_std:.2f})"
    )
    print(f"Coverage: {coverage_score_mean:.2f} ({coverage_score_std:.2f})")


# run_experiment_with_disjoint_environments(
#     llm_model_name="gemini-pro",
#     llm_model_key="your-api-key",
#     num_timesteps=1000,
#     seed=42,
#     progressive_gif=True,
#     run="/path/to/checkpoint.pkl"
# )