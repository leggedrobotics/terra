import numpy as np
import jax

import jax.numpy as jnp

from terra.env import TerraEnv

from llm.utils_llm import *
from terra.viz.llms_adk import *

import pygame as pg

from llm.env_llm import LargeMapTerraEnv

class EnvironmentsManager:
    """
    Manages completely separate environments for large map and small maps.
    Each environment has its own timestep, configuration, and state.
    Only map data is exchanged between environments.
    """
        
    def __init__(self, seed, global_env_config, small_env_config=None, shuffle_maps=False, rendering=False, display=False):
        """
        Initialize with separate configurations for large and small environments.
        
        Args:
            seed: Random seed for reproducibility
            global_env_config: Environment configuration for the large global map
            small_env_config: Environment configuration for small maps (or None to derive from global)
            num_partitions: Number of partitions for the large map
            shuffle_maps: Whether to shuffle maps
        """
        self.rng = jax.random.PRNGKey(seed)
        self.global_env_config = global_env_config
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
        self.rendering = rendering
        self.display = display
        
        # Initialize the global environment (128x128) with LargeMapTerraEnv
        print("Initializing LargeMapTerraEnv for global environment...")

        self.global_env = LargeMapTerraEnv(
            rendering=rendering,
            n_envs_x_rendering=1,
            n_envs_y_rendering=1,
            display=display,
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
        )
        
        # Store global map data
        self.global_maps = {
            'target_map': None,
            'action_map': None,
            'dumpability_mask': None,
            'dumpability_mask_init': None,
            'padding_mask': None,
            'traversability_mask': None,
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
        
        #print(f"Agent configs - Small: {self.small_agent_config}, Big: {self.big_agent_config}")

    def _partitions_overlap(self, i: int, j: int) -> bool:
        """Check if two partitions overlap."""
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


    def _calculate_overlap_region(self, partition_i: int, partition_j: int):
            """
            Calculate the overlapping region between two partitions.
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

    def set_partitions(self, partitions):
        """
        Set the partitions and compute overlap relationships.
        """
        print(f"\n=== SETTING PARTITIONS ===")
        self.partitions = partitions
        
        print(f"Partitions set:")
        for i, partition in enumerate(self.partitions):
            print(f"  Partition {i}: {partition}")
        
        # Use the fixed overlap computation
        self._compute_overlap_relationships()
        
        print(f"Set {len(self.partitions)} partitions with overlaps computed.")

    def _compute_overlap_relationships(self):
        """
        Compute which partitions overlap with each other and cache overlap regions.
        """
        print(f"\n=== COMPUTING OVERLAP RELATIONSHIPS ===")
        
        self.overlap_map = {i: set() for i in range(len(self.partitions))}
        self.overlap_regions = {}
        
        for i in range(len(self.partitions)):
            for j in range(i + 1, len(self.partitions)):
                print(f"\nChecking partitions {i} and {j}:")
                
                if self._partitions_overlap(i, j):
                    self.overlap_map[i].add(j)
                    self.overlap_map[j].add(i)
                    
                    # Cache the overlap region calculation
                    overlap_info = self._calculate_overlap_region(i, j)
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
    def initialize_with_fixed_overlaps(self, partitions):
        """
        Initialize partitions with fixed overlap detection.
        """
        
        # Set partitions using the fixed method
        self.set_partitions(partitions)
        
    def add_agents_using_existing_representation(self, partition_states):
        """
        Extract agent representation from the other partition's traversability mask.
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
                    #print(f"  Found agent {other_partition_idx} with {len(agent_positions[0])} occupied cells")
                    
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
                
                print(f"  ✓ Added {agents_added} agents with exact representation to partition {target_partition_idx}")

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
        current_action_map = current_state.world.action_map.map
        
        # Step the environment
        new_timestep = self.small_env.step(
            state=current_state,
            action=action,
            target_map=current_target_map,
            padding_mask=current_padding_mask,
            trench_axes=current_trench_axes,
            trench_type=current_trench_type,
            dumpability_mask_init=current_dumpability_mask_init,
            action_map=current_action_map,
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
        # sub_maps = {
        #     'target_map': create_sub_task_target_map_64x64(self.global_maps['target_map'], region_coords),
        #     'action_map': create_sub_task_action_map_64x64(self.global_maps['action_map'], region_coords),
        #     'dumpability_mask': create_sub_task_dumpability_mask_64x64(self.global_maps['dumpability_mask'], region_coords),
        #     'dumpability_mask_init': create_sub_task_dumpability_mask_64x64(self.global_maps['dumpability_mask_init'], region_coords),
        #     'padding_mask': create_sub_task_padding_mask_64x64(self.global_maps['padding_mask'], region_coords),
        #     'traversability_mask': create_sub_task_traversability_mask_64x64(self.global_maps['traversability_mask'], region_coords),
        # }

        sub_maps = {
            'target_map': create_sub_task_target_map_64x64(self.global_maps['target_map'], region_coords),                              #ok
            'action_map': self.global_maps['action_map'],
            'dumpability_mask': self.global_maps['dumpability_mask'],
            'dumpability_mask_init': self.global_maps['dumpability_mask_init'],
            'padding_mask': self.global_maps['padding_mask'],
            'traversability_mask': self.global_maps['traversability_mask'],                                                             #OK, keep the full traversability mask
        }


        save_mask(np.array(sub_maps['target_map']),'target', 'after_init', partition_idx, 0)
        save_mask(np.array(sub_maps['action_map']),'action', 'after_init', partition_idx, 0)
        save_mask(np.array(sub_maps['dumpability_mask']),'dumpability', 'after_init', partition_idx, 0)
        save_mask(np.array(sub_maps['dumpability_mask_init']),'dumpability_init', 'after_init', partition_idx, 0)
        save_mask(np.array(sub_maps['padding_mask']),'padding', 'after_init', partition_idx, 0)
        save_mask(np.array(sub_maps['traversability_mask']),'traversability', 'after_init', partition_idx, 0)


        #DIAGNOSTIC: Check sub-map validity
        # print(f"=== SUB-MAP DIAGNOSTICS ===")
        # for name, map_data in sub_maps.items():
        #     print(f"{name}:")
        #     print(f"  Shape: {map_data.shape}")
        

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
                action_map=sub_maps['action_map'],
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
        self.global_maps['trench_axes'] = global_timestep.state.world.trench_axes.copy()
        self.global_maps['trench_type'] = global_timestep.state.world.trench_type.copy()
    
        # Store global timestep
        self.global_timestep = global_timestep
    
        print("Global environment initialized successfully.")
        #print(f"Initial target map has {jnp.sum(self.global_maps['target_map'] < 0)} dig targets")

        return self.global_timestep
        
    def map_position_small_to_global(self, small_pos, region_coords):
        """
        Map agent position from small map coordinates to global map coordinates.
        Assumes the small map places the region at (0,0), so we need to add offsets.
        Returns position in (x, y) format for rendering.
        """
        y_start, x_start, y_end, x_end = region_coords
        
        # Extract position values - assuming agent position is [x, y]
        if hasattr(small_pos, 'shape'):
            if len(small_pos.shape) == 1 and small_pos.shape[0] == 2:
                local_x = float(small_pos[0])
                local_y = float(small_pos[1])
            else:
                local_x = float(small_pos.flatten()[0])
                local_y = float(small_pos.flatten()[1])
        else:
            local_x = float(small_pos[0])
            local_y = float(small_pos[1])
        
        # Add region offset to get global position
        # global_x = local_x + x_start
        # global_y = local_y + y_start
        global_x = local_x
        global_y = local_y 
        #print(f"Mapping small position {small_pos} to global coordinates: ({global_x}, {global_y}) with region {region_coords}")
        
        # Ensure position is within valid bounds
        #global_x = max(0, min(63, global_x))
        #global_y = max(0, min(63, global_y))
        
        # Return as (x, y) for rendering
        #return (int(global_x), int(global_y))
                # Return as (y, x) for rendering
        return (int(global_y), int(global_x))

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
                    # print("original small pos:", small_pos)
                    # print("original small angle base:", small_angle_base)
                    # print("original small angle cabin:", small_angle_cabin)
                
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

                    #print(global_pos, angle_base_val, angle_cabin_val, small_loaded)
                
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
                # region_height = y_end - y_start + 1
                # region_width = x_end - x_start + 1
            
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
                }
            
                # print(f"Small environment map shapes:")
                # for name, map_data in small_maps.items():
                #     print(f"  {name}: {map_data.shape}")
            
                # Extract only the relevant portion from the 64x64 small maps
                # that corresponds to the actual region size
                # extract_height = min(region_height, 64)
                # extract_width = min(region_width, 64)
            
                for map_name, small_map in small_maps.items():
                    # Extract the portion that matches the region size
                    #extracted_region = small_map[:extract_height, :extract_width]
                    extracted_region = small_map[region_slice]

                
                    #print(f"  Extracted {map_name}: {extracted_region.shape} -> Global region: {region_height}x{region_width}")
                    self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(extracted_region)

                    # Update the global map with the extracted region
                    # try:
                    #     self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(extracted_region)
                    # except ValueError as e:
                    #     #print(f"  WARNING: Shape mismatch for {map_name}: {e}")
                    #     # Try to handle the mismatch by padding or cropping
                    #     if extracted_region.shape[0] != region_height or extracted_region.shape[1] != region_width:
                    #         # Pad or crop to match the region size
                    #         if extracted_region.shape[0] < region_height or extracted_region.shape[1] < region_width:
                    #             # Pad with zeros
                    #             padded_region = jnp.zeros((region_height, region_width), dtype=extracted_region.dtype)
                    #             padded_region = padded_region.at[:extracted_region.shape[0], :extracted_region.shape[1]].set(extracted_region)
                    #             extracted_region = padded_region
                    #         else:
                    #             # Crop to fit
                    #             extracted_region = extracted_region[:region_height, :region_width]
                        
                    #         #print(f"  Adjusted {map_name}: {extracted_region.shape}")
                    #         self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(extracted_region)


    def render_global_environment_with_multiple_agents(self, partition_states, VISUALIZE_PARTITIONS=False):
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
                print("Warning: Agent attributes not properly initialized for rendering")
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
            if self.rendering:
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
        # Get the maps from the partition
        current_timestep = partition_state['timestep']
        world = current_timestep.state.world
        agent_state = current_timestep.state.agent.agent_state
        
        # Extract maps
        target_map = world.target_map.map
        action_map = world.action_map.map
        traversability_mask = world.traversability_mask.map
        agent_pos = agent_state.pos_base
        
        #print(current_timestep.observation)
        # target_map = current_timestep.observation['target_map']
        # action_map = current_timestep.observation['action_map']
        # traversability_mask = current_timestep.observation['traversability_mask']

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
    
    def _should_show_agent_in_partition(self, partition_idx, agent_y, agent_x):
        """
        Determine if an agent at the given position should be visible to the partition.
        
        For global maps, you might want to:
        1. Show all agents everywhere (return True)
        2. Show agents only within a certain distance of the partition's region
        3. Show agents only within the partition's assigned region
        """
        # Option 1: Show all agents everywhere (recommended for global maps)
        return True
        
        # Option 2: Show agents within partition region + buffer
        # if partition_idx < len(self.partitions):
        #     partition = self.partitions[partition_idx]
        #     y_start, x_start, y_end, x_end = partition['region_coords']
        #     
        #     # Add buffer around partition region
        #     buffer = 10
        #     return (y_start - buffer <= agent_y <= y_end + buffer and 
        #             x_start - buffer <= agent_x <= x_end + buffer)
        # 
        # return False

    def update_global_maps_from_partition_changes(self, partition_states):
        """
        Update the global maps with changes from all partitions.
        Since most maps are now shared, focus on target_map updates.
        """
        print(f"\n=== UPDATING GLOBAL MAPS FROM PARTITION CHANGES ===")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            # Get current state from partition
            current_timestep = partition_state['timestep']
            partition_maps = {
                'target_map': current_timestep.state.world.target_map.map,
                'action_map': current_timestep.state.world.action_map.map,
                'dumpability_mask': current_timestep.state.world.dumpability_mask.map,
            }
            
            # Update global maps with partition's changes
            # Since target_map is localized, we need to merge changes back to global
            partition = self.partitions[partition_idx]
            region_coords = partition['region_coords']
            self._merge_partition_target_changes_to_global(
                partition_maps['target_map'], region_coords
            )
            
            # For other maps, since they're shared, they should already be in sync
            # but we can update the global reference if needed
            self.global_maps['action_map'] = partition_maps['action_map']
            self.global_maps['dumpability_mask'] = partition_maps['dumpability_mask']

    def _merge_partition_target_changes_to_global(self, partition_target_map, region_coords):
        """
        Merge changes from a partition's target_map back to the global target_map.
        """
        y_start, x_start, y_end, x_end = region_coords
        region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
        
        # Extract the relevant region from the partition's target map
        partition_region = partition_target_map[region_slice]
        
        # Update the global target map
        self.global_maps['target_map'] = self.global_maps['target_map'].at[region_slice].set(partition_region)

    # Updated main synchronization function to replace the old one
    def add_agents_using_global_sync(self, partition_states):
        """
        Replacement for add_agents_using_existing_representation.
        Designed for the new global map approach.
        """
        self.sync_agents_in_global_environment(partition_states)

    # Simplified overlap detection for global maps
    def compute_agent_visibility_relationships(self):
        """
        Since maps are global, we don't need complex overlap detection.
        Instead, determine which agents should be visible to each partition.
        """
        print(f"\n=== COMPUTING AGENT VISIBILITY ===")
        
        # For global maps, all agents are potentially visible to all partitions
        self.agent_visibility_map = {i: set(range(len(self.partitions))) 
                                    for i in range(len(self.partitions))}
        
        # Remove self-visibility
        for i in range(len(self.partitions)):
            self.agent_visibility_map[i].discard(i)
        
        print(f"All agents will be visible to all partitions in global map mode")

    # Updated step function that uses the new sync approach
    def step_with_global_sync(self, partition_idx: int, action, partition_states: dict):
        """
        Step function adapted for global maps with proper synchronization.
        """
        # Regular step (unchanged)
        new_timestep = self.step_simple(partition_idx, action, partition_states)
        
        # Update partition state
        partition_states[partition_idx]['timestep'] = new_timestep
        
        # Sync agents across all partitions
        self.add_agents_using_global_sync(partition_states)
        
        # Update global maps if needed
        self.update_global_maps_from_partition_changes(partition_states)
        
        return new_timestep
    

    def initialize_base_traversability_masks(self, partition_states):
        """
        Store the initial clean traversability masks for each partition.
        This captures the original terrain obstacles before any agent synchronization.
        Call this ONCE after partition initialization but BEFORE any agent sync.
        """
        if not hasattr(self, 'base_traversability_masks'):
            self.base_traversability_masks = {}
        
        print("Initializing base traversability masks...")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] == 'active':
                # Get the current traversability mask
                current_mask = partition_state['timestep'].state.world.traversability_mask.map.copy()
                
                # Clean ANY agent markers to get pure terrain
                # -1 = agent position (clear to 0)
                # 1 = could be terrain or agent obstacles (assume terrain at initialization)
                # 0 = free space (keep as is)
                
                # Create a completely clean mask - only terrain obstacles, no agents
                clean_mask = jnp.where(
                    current_mask == -1,  # Remove any agent positions
                    0,  # Set to free space
                    jnp.where(
                        current_mask == 1,  # Keep terrain obstacles
                        1,
                        0  # Everything else becomes free space
                    )
                )
                
                self.base_traversability_masks[partition_idx] = clean_mask
                print(f"  Stored clean base mask for partition {partition_idx}")
                
                # Count terrain obstacles for verification
                terrain_obstacles = jnp.sum(clean_mask == 1)
                print(f"    Terrain obstacles: {terrain_obstacles}")

    def _update_partition_with_other_agents(self, target_partition_idx, target_partition_state, 
                                        all_occupied_cells, partition_states):
        """
        Update a partition's traversability mask to show other agents as obstacles.
        Now properly preserves original terrain obstacles.
        """
        current_timestep = target_partition_state['timestep']
        
        # STEP 1: Start from the clean base mask (has original terrain obstacles but no agent obstacles)
        if hasattr(self, 'base_traversability_masks') and target_partition_idx in self.base_traversability_masks:
            # Start from clean base (original terrain obstacles preserved)
            current_traversability = self.base_traversability_masks[target_partition_idx].copy()
            
            # Add back the current agent's position (-1)
            original_traversability = current_timestep.state.world.traversability_mask.map
            agent_mask = (original_traversability == -1)
            current_traversability = jnp.where(
                agent_mask,
                -1,  # Restore current agent position
                current_traversability  # Keep clean base with terrain obstacles
            )
        else:
            # Fallback: use current mask but this might not work perfectly
            print(f"Warning: No base mask for partition {target_partition_idx}, using current mask")
            current_traversability = current_timestep.state.world.traversability_mask.map.copy()
            
            # Try to clear only agent obstacles (this is less reliable)
            # Keep -1 (current agent) and assume original 1s are terrain
            # This fallback is not ideal - base masks are recommended
        
        # STEP 2: Add current positions of OTHER agents as obstacles
        agents_added = 0
        cells_added = 0
        
        for other_partition_idx, occupied_cells in all_occupied_cells.items():
            if other_partition_idx == target_partition_idx:
                continue  # Don't add self as obstacle
                
            for cell_y, cell_x in occupied_cells:
                # Check if this cell should be visible in this partition
                if self._should_show_agent_in_partition(target_partition_idx, cell_y, cell_x):
                    # Check bounds
                    if (0 <= cell_y < current_traversability.shape[0] and 
                        0 <= cell_x < current_traversability.shape[1]):
                        # Mark as obstacle (value = 1) - this represents another agent
                        # Only set if it's currently free space (0) to avoid overwriting terrain
                        if current_traversability[cell_y, cell_x] == 0:
                            current_traversability = current_traversability.at[cell_y, cell_x].set(1)
                            cells_added += 1
            
            if cells_added > 0:
                agents_added += 1
        
        # STEP 3: Update the world state
        updated_world = self._update_world_map(
            current_timestep.state.world, 
            'traversability_mask', 
            current_traversability
        )
        updated_state = current_timestep.state._replace(world=updated_world)
        updated_timestep = current_timestep._replace(state=updated_state)
        
        partition_states[target_partition_idx]['timestep'] = updated_timestep
        
        if agents_added > 0:
            print(f"  ✓ Added {agents_added} other agents ({cells_added} cells) to partition {target_partition_idx}")

    def sync_agents_in_global_environment(self, partition_states):
        """
        Updated synchronization that preserves original terrain obstacles.
        """
        print(f"\n=== SYNCING AGENTS IN GLOBAL ENVIRONMENT ===")
        
        # Initialize base masks if not done yet (IMPORTANT: do this early, before any sync operations)
        if not hasattr(self, 'base_traversability_masks'):
            print("Initializing base traversability masks...")
            self.initialize_base_traversability_masks(partition_states)
        
        # Collect all active agent positions and their occupied cells
        all_agent_positions = {}
        all_occupied_cells = {}
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            current_timestep = partition_state['timestep']
            traversability = current_timestep.state.world.traversability_mask.map
            
            # Find where this agent is (value = -1)
            agent_mask = (traversability == -1)
            agent_positions = jnp.where(agent_mask)
            
            if len(agent_positions[0]) > 0:
                # Store agent position info
                all_agent_positions[partition_idx] = {
                    'positions': agent_positions,
                    'count': len(agent_positions[0])
                }
                
                # Store occupied cells for this agent
                occupied_cells = []
                for i in range(len(agent_positions[0])):
                    cell = (int(agent_positions[0][i]), int(agent_positions[1][i]))
                    occupied_cells.append(cell)
                all_occupied_cells[partition_idx] = occupied_cells
        
        # Update each partition's traversability mask with other agents
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            self._update_partition_with_other_agents(
                target_partition_idx, target_partition_state, 
                all_occupied_cells, partition_states
            )
        
        print(f"Agent synchronization completed for {len(all_agent_positions)} active agents")


    # Add these methods to your EnvironmentsManager class

    def sync_targets_across_partitions(self, partition_states):
        """
        Synchronize targets across partitions by marking other partitions' targets as obstacles.
        This prevents agents from working on targets assigned to other partitions.
        """
        print(f"\n=== SYNCING TARGETS ACROSS PARTITIONS ===")
        
        # Collect all partition targets
        all_partition_targets = {}
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            current_timestep = partition_state['timestep']
            target_map = current_timestep.state.world.target_map.map
            
            # Store the target map for this partition
            all_partition_targets[partition_idx] = target_map
        
        # Update each partition to mark other partitions' targets as obstacles
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            self._update_partition_with_other_targets(
                target_partition_idx, target_partition_state, 
                all_partition_targets, partition_states
            )
        
        print(f"Target synchronization completed for {len(all_partition_targets)} active partitions")

    def _update_partition_with_other_targets(self, target_partition_idx, target_partition_state, 
                                        all_partition_targets, partition_states):
        """
        Update a partition's traversability mask to mark other partitions' targets as obstacles.
        """
        current_timestep = target_partition_state['timestep']
        
        # Start from the current traversability mask
        current_traversability = current_timestep.state.world.traversability_mask.map.copy()
        
        targets_blocked = 0
        cells_blocked = 0
        
        # Add other partitions' targets as obstacles
        for other_partition_idx, other_target_map in all_partition_targets.items():
            if other_partition_idx == target_partition_idx:
                continue  # Don't block own targets
            
            # Get the current partition's own target map to avoid conflicts
            own_target_map = current_timestep.state.world.target_map.map
            
            # Find other partition's targets (both dig targets -1 and dump targets 1)
            other_dig_targets = (other_target_map == -1)
            other_dump_targets = (other_target_map == 1)
            other_all_targets = other_dig_targets | other_dump_targets
            
            # Only block targets that are not also targets in current partition
            own_targets = (own_target_map == -1) | (own_target_map == 1)
            
            # Find positions to block: other partition's targets that aren't current partition's targets
            positions_to_block = other_all_targets & ~own_targets
            
            # Find valid positions to block in the traversability mask
            for y in range(current_traversability.shape[0]):
                for x in range(current_traversability.shape[1]):
                    if positions_to_block[y, x]:
                        # Only mark as obstacle if it's currently free space (0) or traversable
                        # Don't overwrite agent positions (-1) or existing obstacles (1)
                        if current_traversability[y, x] == 0:
                            current_traversability = current_traversability.at[y, x].set(1)
                            cells_blocked += 1
            
            if cells_blocked > 0:
                targets_blocked += 1
        
        # Update the world state
        updated_world = self._update_world_map(
            current_timestep.state.world, 
            'traversability_mask', 
            current_traversability
        )
        updated_state = current_timestep.state._replace(world=updated_world)
        updated_timestep = current_timestep._replace(state=updated_state)
        
        partition_states[target_partition_idx]['timestep'] = updated_timestep
        
        if targets_blocked > 0:
            print(f"  ✓ Blocked {cells_blocked} target cells from {targets_blocked} other partitions in partition {target_partition_idx}")

    def sync_targets_in_overlapping_regions_only(self, partition_states):
        """
        Alternative approach: Only block targets in overlapping regions between partitions.
        This is more conservative and only affects shared areas.
        """
        print(f"\n=== SYNCING TARGETS IN OVERLAPPING REGIONS ===")
        
        for (i, j), overlap_info in self.overlap_regions.items():
            if i >= j:  # Only process each pair once
                continue
                
            if (i not in partition_states or j not in partition_states or
                partition_states[i]['status'] != 'active' or 
                partition_states[j]['status'] != 'active'):
                continue
            
            # Sync targets in both directions for overlapping partitions
            self._sync_targets_in_overlap_region(partition_states, i, j, overlap_info)
            self._sync_targets_in_overlap_region(partition_states, j, i, overlap_info)

    def _sync_targets_in_overlap_region(self, partition_states, target_partition_idx, 
                                    source_partition_idx, overlap_info):
        """
        Block source partition's targets in the overlapping region of target partition.
        """
        target_state = partition_states[target_partition_idx]['timestep'].state
        source_state = partition_states[source_partition_idx]['timestep'].state
        
        # Get the correct slices based on partition order
        if target_partition_idx < source_partition_idx:
            target_slice = overlap_info['partition_i_slice']
            source_slice = overlap_info['partition_j_slice']
        else:
            target_slice = overlap_info['partition_j_slice']
            source_slice = overlap_info['partition_i_slice']
        
        # Get target maps
        target_traversability = target_state.world.traversability_mask.map.copy()
        source_target_map = source_state.world.target_map.map
        target_own_targets = target_state.world.target_map.map
        
        # Extract overlapping regions
        source_overlap_targets = source_target_map[source_slice]
        target_overlap_traversability = target_traversability[target_slice]
        target_overlap_own_targets = target_own_targets[target_slice]
        
        # Find source targets in overlap region (both dig and dump targets)
        source_targets_in_overlap = (source_overlap_targets == -1) | (source_overlap_targets == 1)
        
        # Don't block if target partition also has targets in the same location
        target_own_targets_in_overlap = (target_overlap_own_targets == -1) | (target_overlap_own_targets == 1)
        
        # Block only source targets that don't conflict with target's own targets
        targets_to_block = source_targets_in_overlap & ~target_own_targets_in_overlap
        
        # Update traversability mask in overlap region
        new_overlap_traversability = jnp.where(
            targets_to_block & (target_overlap_traversability == 0),  # Only block free space
            1,  # Mark as obstacle
            target_overlap_traversability  # Keep existing values
        )
        
        # Update the full traversability mask
        target_traversability = target_traversability.at[target_slice].set(new_overlap_traversability)
        
        # Count blocked cells
        blocked_cells = jnp.sum(targets_to_block & (target_overlap_traversability == 0))
        
        if blocked_cells > 0:
            # Update the world state
            updated_world = self._update_world_map(
                target_state.world, 
                'traversability_mask', 
                target_traversability
            )
            updated_state = target_state._replace(world=updated_world)
            updated_timestep = partition_states[target_partition_idx]['timestep']._replace(state=updated_state)
            
            partition_states[target_partition_idx]['timestep'] = updated_timestep
            
            print(f"  ✓ Blocked {blocked_cells} target cells from partition {source_partition_idx} in overlap region of partition {target_partition_idx}")

    # Update the main synchronization method to include target synchronization
    def add_agents_and_targets_using_global_sync(self, partition_states):
        """
        Enhanced synchronization that handles both agents and targets.
        """
        # First sync agents (existing functionality)
        self.sync_agents_in_global_environment(partition_states)
        
        # Then sync targets to prevent conflicts
        self.sync_targets_across_partitions(partition_states)
        
        # Alternative: Only sync targets in overlapping regions
        # self.sync_targets_in_overlapping_regions_only(partition_states)

    # Update the step function to use the enhanced synchronization
    def step_with_enhanced_sync(self, partition_idx: int, action, partition_states: dict):
        """
        Step function with enhanced synchronization for both agents and targets.
        """
        # Regular step (unchanged)
        new_timestep = self.step_simple(partition_idx, action, partition_states)
        
        # Update partition state
        partition_states[partition_idx]['timestep'] = new_timestep
        
        # Enhanced sync that includes both agents and targets
        self.add_agents_and_targets_using_global_sync(partition_states)
        
        # Update global maps if needed
        self.update_global_maps_from_partition_changes(partition_states)
        
        return new_timestep
    
    def step_with_full_global_sync(self, partition_idx: int, action, partition_states: dict):
        """
        UPDATED: Enhanced step function that properly handles dumped soil as obstacles.
        """
        # Step 1: Take the action in the partition
        new_timestep = self.step_simple(partition_idx, action, partition_states)
        
        # Step 2: Update the partition state
        partition_states[partition_idx]['timestep'] = new_timestep
        
        # Step 3: Extract changes from this partition and update global maps
        self._update_global_maps_from_single_partition(partition_idx, partition_states)
        
        # Step 4: Propagate global map changes to ALL partitions (EXCLUDING traversability)
        self._sync_all_partitions_from_global_maps_excluding_traversability(partition_states)
        
        # Step 5: Properly sync agent positions AND dumped soil obstacles
        self._sync_agent_positions_across_partitions(partition_states)
        
        # Step 6: Update observations to match synced states
        self._update_all_observations(partition_states)
        
        return new_timestep
    
        # Also add this comprehensive verification method for your main loop
    def comprehensive_sync_verification(self, partition_states, step_number):
        """
        Comprehensive verification that checks all aspects of synchronization.
        Use this periodically in your main loop to catch any sync issues.
        """
        print(f"\n=== COMPREHENSIVE SYNC VERIFICATION - STEP {step_number} ===")
        
        # 1. Check basic traversability correctness
        traversability_ok = self.verify_traversability_with_dumped_soil(partition_states)
        
        # 2. Check global map consistency
        global_sync_ok = self.verify_global_sync_consistency(partition_states)
        
        # 3. Check action/traversability consistency
        self.debug_action_traversability_consistency(partition_states, f"STEP_{step_number}")
        
        # 4. Check for dumped soil obstacles specifically
        print(f"\n--- DUMPED SOIL OBSTACLE CHECK ---")
        total_dumped_areas = 0
        total_dumped_obstacles = 0
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            timestep = partition_state['timestep']
            action_map = timestep.state.world.action_map.map
            traversability = timestep.state.world.traversability_mask.map
            
            # Count dumped areas
            dumped_areas = jnp.sum(action_map > 0)
            total_dumped_areas += dumped_areas
            
            # Count how many are obstacles
            dumped_positions = jnp.where(action_map > 0)
            dumped_as_obstacles = 0
            
            if len(dumped_positions[0]) > 0:
                for i in range(len(dumped_positions[0])):
                    y, x = dumped_positions[0][i], dumped_positions[1][i]
                    if traversability[y, x] == 1:
                        dumped_as_obstacles += 1
            
            total_dumped_obstacles += dumped_as_obstacles
            
            print(f"  Partition {partition_idx}: {dumped_areas} dumped areas, {dumped_as_obstacles} as obstacles")
        
        print(f"  TOTAL: {total_dumped_areas} dumped areas, {total_dumped_obstacles} marked as obstacles")
        
        if total_dumped_areas == total_dumped_obstacles:
            print("  ✅ All dumped soil correctly marked as obstacles")
            dumped_soil_ok = True
        else:
            print("  ❌ Some dumped soil not marked as obstacles")
            dumped_soil_ok = False
        
        # Overall result
        all_ok = traversability_ok and global_sync_ok and dumped_soil_ok
        
        if all_ok:
            print("🎉 ALL SYNCHRONIZATION CHECKS PASSED!")
        else:
            print("❌ SYNCHRONIZATION ISSUES DETECTED!")
        
        return all_ok

    def _update_partition_traversability_with_dumped_soil_and_dig_targets(self, target_partition_idx, target_partition_state, 
                                                                     all_agent_positions, partition_states):
        """
        FIXED: Clean approach to updating traversability that includes:
        1. Original terrain obstacles
        2. Dumped soil as obstacles  
        3. Other agents as obstacles
        4. OTHER PARTITIONS' DIG TARGETS (-1) as obstacles (FIXED - only dig targets, not dump targets)
        
        Traversability logic:
        - 0: Free space (can drive through)
        - 1: Obstacles (terrain + other agents + dumped soil + other partitions' dig targets)
        - -1: Current agent position
        """
        current_timestep = target_partition_state['timestep']
        
        # STEP 1: Start from completely clean base mask (original terrain only)
        if target_partition_idx in self.base_traversability_masks:
            clean_traversability = self.base_traversability_masks[target_partition_idx].copy()
            print(f"    Starting from clean base for partition {target_partition_idx}")
        else:
            print(f"    WARNING: No base mask for partition {target_partition_idx}, creating clean mask")
            current_mask = current_timestep.state.world.traversability_mask.map
            clean_traversability = jnp.where(
                (current_mask == -1) | (current_mask == 1),  # Remove all agent markers
                0,  # Set to free space
                current_mask  # Keep original terrain
            )
        
        # STEP 2: Add dumped soil areas as obstacles
        action_map = current_timestep.state.world.action_map.map
        dumped_areas = (action_map > 0)  # Positive values = dumped soil
        
        # Mark dumped soil areas as obstacles (1)
        clean_traversability = jnp.where(
            dumped_areas,
            1,  # Dumped soil = obstacle
            clean_traversability  # Keep existing values
        )
        
        dumped_obstacle_count = jnp.sum(dumped_areas)
        if dumped_obstacle_count > 0:
            print(f"    Added {dumped_obstacle_count} dumped soil obstacles to partition {target_partition_idx}")
        
        # STEP 3: FIXED - Add other partitions' DIG TARGETS (-1) as obstacles, but NOT dump targets (1)
        other_dig_targets_blocked = 0
        
        for other_partition_idx, other_partition_state in partition_states.items():
            if (other_partition_idx == target_partition_idx or 
                other_partition_state['status'] != 'active'):
                continue
                
            # Get the original target map for the other partition
            if hasattr(self, 'partition_target_maps') and other_partition_idx in self.partition_target_maps:
                other_target_map = self.partition_target_maps[other_partition_idx]
                
                # FIXED: Only block dig targets (-1), NOT dump targets (1)
                other_dig_targets = (other_target_map == -1)  # Only dig targets
                # Note: We don't block dump targets (other_target_map == 1) because agents can potentially traverse dump areas
                
                # Mark dig targets as obstacles in current partition's traversability
                clean_traversability = jnp.where(
                    other_dig_targets,
                    1,  # Other partitions' dig targets = obstacles
                    clean_traversability  # Keep existing values
                )
                
                dig_targets_blocked_from_this_partition = jnp.sum(other_dig_targets)
                other_dig_targets_blocked += dig_targets_blocked_from_this_partition
                
                if dig_targets_blocked_from_this_partition > 0:
                    print(f"    Blocked {dig_targets_blocked_from_this_partition} DIG TARGETS from partition {other_partition_idx}")
        
        if other_dig_targets_blocked > 0:
            print(f"    Total other dig targets blocked: {other_dig_targets_blocked}")
        
        # STEP 4: Add THIS partition's agent positions as agents (-1)
        if target_partition_idx in all_agent_positions:
            own_positions = all_agent_positions[target_partition_idx]
            for cell_y, cell_x in own_positions:
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    clean_traversability = clean_traversability.at[cell_y, cell_x].set(-1)
            
            print(f"    Added {len(own_positions)} own agent cells to partition {target_partition_idx}")
        
        # STEP 5: Add OTHER agents as OBSTACLES (1), not agents
        other_agents_added = 0
        other_cells_added = 0
        
        for other_partition_idx, other_positions in all_agent_positions.items():
            if other_partition_idx == target_partition_idx:
                continue  # Skip own agent
                
            for cell_y, cell_x in other_positions:
                # Check bounds
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    
                    # Add as OBSTACLE (1), not agent (-1)
                    # Only if it's currently free space (0) - don't overwrite own agent or existing obstacles
                    current_value = clean_traversability[cell_y, cell_x]
                    if current_value == 0:  # Free space
                        clean_traversability = clean_traversability.at[cell_y, cell_x].set(1)
                        other_cells_added += 1
                    elif current_value == -1:  # Don't overwrite own agent
                        print(f"      Conflict: Other agent at own agent position ({cell_y}, {cell_x})")
            
            if other_cells_added > 0:
                other_agents_added += 1
        
        # STEP 6: Update the world state
        updated_world = self._update_world_map(
            current_timestep.state.world, 
            'traversability_mask', 
            clean_traversability
        )
        updated_state = current_timestep.state._replace(world=updated_world)
        updated_timestep = current_timestep._replace(state=updated_state)
        
        partition_states[target_partition_idx]['timestep'] = updated_timestep
        
        # Debug output with all obstacle types
        terrain_count = jnp.sum(self.base_traversability_masks.get(target_partition_idx, jnp.zeros_like(clean_traversability)) == 1)
        dumped_soil_count = jnp.sum(dumped_areas)
        own_agent_count = jnp.sum(clean_traversability == -1)
        other_obstacle_count = other_cells_added
        total_obstacles = jnp.sum(clean_traversability == 1)
        free_space = jnp.sum(clean_traversability == 0)
        total_cells = clean_traversability.size
        
        print(f"    Partition {target_partition_idx} traversability summary:")
        print(f"      Original terrain obstacles: {terrain_count}")
        print(f"      Dumped soil obstacles: {dumped_soil_count}")
        print(f"      Other partitions' DIG TARGETS blocked: {other_dig_targets_blocked}")
        print(f"      Other agents (as obstacles): {other_obstacle_count}")
        print(f"      Total obstacles: {total_obstacles}")
        print(f"      Own agent cells: {own_agent_count}")
        print(f"      Free space: {free_space} ({free_space/total_cells:.1%})")
        
        if free_space < total_cells * 0.3:  # Less than 30% free space
            print(f"      ⚠️  Warning: Low free space percentage")
        else:
            print(f"      ✅ Good free space percentage")

    def _sync_all_partitions_from_global_maps_excluding_traversability(self, partition_states):
        """
        UPDATED: Synchronize ALL partitions with updated global maps, but preserve partition-specific targets.
        """
        print(f"  Syncing global maps to all partitions (excluding traversability and preserving partition targets)")
        
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            # Get current state
            current_timestep = target_partition_state['timestep']
            current_state = current_timestep.state
            
            # Create updated world state with global maps but preserve partition-specific targets
            updated_world = self._create_world_with_global_maps_preserve_targets(current_state.world, target_partition_idx)
            
            # Create updated state and timestep
            updated_state = current_state._replace(world=updated_world)
            updated_timestep = current_timestep._replace(state=updated_state)
            
            # Update the partition state
            partition_states[target_partition_idx]['timestep'] = updated_timestep
            
            print(f"    Synced global maps to partition {target_partition_idx} (targets preserved)")

    def _update_all_observations(self, partition_states):
        """
        Update observations for all partitions to match their synced states.
        """
        print(f"  Updating observations for all partitions")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            current_timestep = partition_state['timestep']
            
            # Create updated observation that matches the synced state
            updated_observation = self._create_observation_from_synced_state(
                current_timestep.observation, 
                current_timestep.state.world
            )
            
            # Update the timestep with the new observation
            updated_timestep = current_timestep._replace(observation=updated_observation)
            partition_states[partition_idx]['timestep'] = updated_timestep
    def debug_traversability_step_by_step(self, partition_states, step_name=""):
        """
        Debug method to track traversability changes step by step.
        Call this at different points in your sync process.
        """
        print(f"\n--- TRAVERSABILITY DEBUG: {step_name} ---")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            traversability = partition_state['timestep'].state.world.traversability_mask.map
            
            free_count = jnp.sum(traversability == 0)
            obstacle_count = jnp.sum(traversability == 1) 
            agent_count = jnp.sum(traversability == -1)
            
            print(f"  Partition {partition_idx}: Free={free_count}, Obstacles={obstacle_count}, Agents={agent_count}")
            
            # Check for unexpected values
            unique_values = jnp.unique(traversability)
            expected_values = jnp.array([-1, 0, 1])
            unexpected = jnp.setdiff1d(unique_values, expected_values)
            
            if len(unexpected) > 0:
                print(f"    ⚠️  Unexpected traversability values: {unexpected}")


    def _update_global_maps_from_single_partition(self, source_partition_idx, partition_states):
        """
        FIXED: Update global maps but handle target_map specially.
        Target maps should remain partition-specific and not be fully synchronized.
        """
        if source_partition_idx not in partition_states:
            return
            
        source_state = partition_states[source_partition_idx]['timestep'].state
        partition = self.partitions[source_partition_idx]
        region_coords = partition['region_coords']
        y_start, x_start, y_end, x_end = region_coords
        
        print(f"  Updating global maps from partition {source_partition_idx}")
        
        # Define which maps to update globally (EXCLUDE target_map)
        maps_to_update = [
            'action_map', 
            'dumpability_mask',
            'dumpability_mask_init'
        ]
        
        # Update each map in the global storage (EXCLUDING target_map)
        for map_name in maps_to_update:
            # Get the current map from the partition
            partition_map = getattr(source_state.world, map_name).map
            
            # Extract the region that corresponds to this partition
            region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
            partition_region = partition_map[region_slice]
            
            # Update the global map with this region
            self.global_maps[map_name] = self.global_maps[map_name].at[region_slice].set(partition_region)
            
            print(f"    Updated global {map_name} from partition {source_partition_idx}")
        
        # Handle target_map specially - update global but don't sync back to other partitions
        target_map = source_state.world.target_map.map
        region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
        target_region = target_map[region_slice]
        
        # Update global target map for tracking purposes, but partitions keep their own
        self.global_maps['target_map'] = self.global_maps['target_map'].at[region_slice].set(target_region)
        print(f"    Updated global target_map from partition {source_partition_idx} (for tracking only)")


    def _sync_all_partitions_from_global_maps(self, partition_states):
        """
        Synchronize ALL partitions with the updated global maps.
        This ensures all partitions see the latest changes from other partitions.
        Now also updates observations to match the synced state.
        """
        print(f"  Syncing all partitions from updated global maps")
        
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            # Get current state
            current_timestep = target_partition_state['timestep']
            current_state = current_timestep.state
            
            # Create updated world state with global map data
            updated_world = self._create_world_with_global_maps(current_state.world, target_partition_idx)
            
            # Create updated state
            updated_state = current_state._replace(world=updated_world)
            
            # Create updated observation that matches the synced state
            updated_observation = self._create_observation_from_synced_state(current_timestep.observation, updated_world)
            
            # Create the complete updated timestep
            updated_timestep = current_timestep._replace(
                state=updated_state,
                observation=updated_observation
            )
            
            # Update the partition state
            partition_states[target_partition_idx]['timestep'] = updated_timestep
            
            print(f"    Synced partition {target_partition_idx} with global maps and updated observation")

    def _create_observation_from_synced_state(self, original_observation, synced_world):
        """
        Create an updated observation dictionary that reflects the synced world state.
        """
        # Start with the original observation (copy all fields)
        updated_observation = {}
        for key, value in original_observation.items():
            updated_observation[key] = value
        
        # Update the critical fields with synced data
        updated_observation['traversability_mask'] = synced_world.traversability_mask.map
        updated_observation['action_map'] = synced_world.action_map.map
        updated_observation['target_map'] = synced_world.target_map.map
        updated_observation['dumpability_mask'] = synced_world.dumpability_mask.map
        updated_observation['padding_mask'] = synced_world.padding_mask.map
        
        return updated_observation


    def _create_world_with_global_maps(self, current_world, partition_idx):
        """
        UPDATED: Create a new world state that uses the current global maps.
        Now properly handles traversability to avoid agent trace issues.
        """
        # DON'T update traversability here - let the agent sync handle it properly
        # This prevents interference between map sync and agent sync
        
        updated_world = current_world._replace(
            target_map=current_world.target_map._replace(map=self.global_maps['target_map']),
            action_map=current_world.action_map._replace(map=self.global_maps['action_map']),
            dumpability_mask=current_world.dumpability_mask._replace(map=self.global_maps['dumpability_mask']),
            dumpability_mask_init=current_world.dumpability_mask_init._replace(map=self.global_maps['dumpability_mask_init']),
            padding_mask=current_world.padding_mask._replace(map=self.global_maps['padding_mask'])
            # NOTE: traversability_mask will be handled separately by agent sync
        )
        
        return updated_world


    def _sync_agent_positions_across_partitions(self, partition_states):
        """
        UPDATED: Properly sync agent positions with dumped soil and dig target blocking only.
        """
        print(f"  Syncing agent positions, dumped soil, and blocking other DIG TARGETS across all partitions")
        
        # Ensure base masks are initialized
        if not hasattr(self, 'base_traversability_masks'):
            print("  WARNING: Base masks not initialized, initializing now...")
            self.initialize_base_traversability_masks(partition_states)
        
        # Collect all current agent positions
        all_agent_positions = {}
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            current_timestep = partition_state['timestep']
            traversability = current_timestep.state.world.traversability_mask.map
            
            # Find this partition's agent positions (value = -1)
            agent_mask = (traversability == -1)
            agent_positions = jnp.where(agent_mask)
            
            if len(agent_positions[0]) > 0:
                occupied_cells = []
                for i in range(len(agent_positions[0])):
                    cell = (int(agent_positions[0][i]), int(agent_positions[1][i]))
                    occupied_cells.append(cell)
                all_agent_positions[partition_idx] = occupied_cells
                print(f"    Agent {partition_idx}: {len(occupied_cells)} occupied cells")
        
        # Update each partition with clean traversability including only dig target obstacles
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            self._update_partition_traversability_with_dumped_soil_and_dig_targets(
                target_partition_idx, target_partition_state, 
                all_agent_positions, partition_states
            )


    def _update_partition_traversability_clean(self, target_partition_idx, target_partition_state, 
                                         all_agent_positions, partition_states):
        """
        FIXED: Clean approach to updating traversability with proper agent representation.
        """
        current_timestep = target_partition_state['timestep']
        
        # STEP 1: Start from completely clean base mask (original terrain only)
        if target_partition_idx in self.base_traversability_masks:
            clean_traversability = self.base_traversability_masks[target_partition_idx].copy()
            print(f"    Starting from clean base for partition {target_partition_idx}")
        else:
            print(f"    WARNING: No base mask for partition {target_partition_idx}, creating clean mask")
            current_mask = current_timestep.state.world.traversability_mask.map
            clean_traversability = jnp.where(
                (current_mask == -1) | (current_mask == 1),  # Remove all agent markers
                0,  # Set to free space
                current_mask  # Keep original terrain
            )
        
        # STEP 2: Add THIS partition's agent positions as agents (-1)
        if target_partition_idx in all_agent_positions:
            own_positions = all_agent_positions[target_partition_idx]
            for cell_y, cell_x in own_positions:
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    clean_traversability = clean_traversability.at[cell_y, cell_x].set(-1)
            
            print(f"    Added {len(own_positions)} own agent cells to partition {target_partition_idx}")
        
        # STEP 3: Add OTHER agents as OBSTACLES (1), not agents
        other_agents_added = 0
        other_cells_added = 0
        
        for other_partition_idx, other_positions in all_agent_positions.items():
            if other_partition_idx == target_partition_idx:
                continue  # Skip own agent
                
            for cell_y, cell_x in other_positions:
                # Check bounds
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    
                    # Add as OBSTACLE (1), not agent (-1)
                    # Only if it's currently free space (0) - don't overwrite own agent or terrain
                    current_value = clean_traversability[cell_y, cell_x]
                    if current_value == 0:  # Free space
                        clean_traversability = clean_traversability.at[cell_y, cell_x].set(1)
                        other_cells_added += 1
                    elif current_value == -1:  # Don't overwrite own agent
                        print(f"      Conflict: Other agent at own agent position ({cell_y}, {cell_x})")
            
            if other_cells_added > 0:
                other_agents_added += 1
        
        # STEP 4: Update the world state
        updated_world = self._update_world_map(
            current_timestep.state.world, 
            'traversability_mask', 
            clean_traversability
        )
        updated_state = current_timestep.state._replace(world=updated_world)
        updated_timestep = current_timestep._replace(state=updated_state)
        
        partition_states[target_partition_idx]['timestep'] = updated_timestep
        
        # Debug output
        terrain_count = jnp.sum(clean_traversability == 1) - other_cells_added
        own_agent_count = jnp.sum(clean_traversability == -1)
        other_obstacle_count = other_cells_added
        
        print(f"    Partition {target_partition_idx} traversability:")
        print(f"      Terrain obstacles: {terrain_count}")
        print(f"      Own agent cells: {own_agent_count}")
        print(f"      Other agents (as obstacles): {other_obstacle_count}")
        
        if other_agents_added > 0:
            print(f"    ✅ Added {other_agents_added} other agents as obstacles ({other_cells_added} cells)")

    def _add_other_agents_to_partition(self, target_partition_idx, target_partition_state, 
                                    all_agent_positions, partition_states):
        """
        Add other agents as obstacles in the target partition's traversability mask.
        """
        current_timestep = target_partition_state['timestep']
        
        # Start from base traversability if available, otherwise use current
        if hasattr(self, 'base_traversability_masks') and target_partition_idx in self.base_traversability_masks:
            current_traversability = self.base_traversability_masks[target_partition_idx].copy()
            
            # Restore this partition's own agent
            original_traversability = current_timestep.state.world.traversability_mask.map
            agent_mask = (original_traversability == -1)
            current_traversability = jnp.where(agent_mask, -1, current_traversability)
        else:
            current_traversability = current_timestep.state.world.traversability_mask.map.copy()
        
        # Add other agents as obstacles
        agents_added = 0
        cells_added = 0
        
        for other_partition_idx, occupied_cells in all_agent_positions.items():
            if other_partition_idx == target_partition_idx:
                continue  # Don't add self
                
            for cell_y, cell_x in occupied_cells:
                # Check bounds and add as obstacle if it's currently free space
                if (0 <= cell_y < current_traversability.shape[0] and 
                    0 <= cell_x < current_traversability.shape[1]):
                    if current_traversability[cell_y, cell_x] == 0:
                        current_traversability = current_traversability.at[cell_y, cell_x].set(1)
                        cells_added += 1
            
            if cells_added > 0:
                agents_added += 1
        
        # Update the world state
        if agents_added > 0 or hasattr(self, 'base_traversability_masks'):
            updated_world = self._update_world_map(
                current_timestep.state.world, 
                'traversability_mask', 
                current_traversability
            )
            updated_state = current_timestep.state._replace(world=updated_world)
            updated_timestep = current_timestep._replace(state=updated_state)
            
            partition_states[target_partition_idx]['timestep'] = updated_timestep
            
            if agents_added > 0:
                print(f"    Added {agents_added} other agents ({cells_added} cells) to partition {target_partition_idx}")

    def update_global_maps_from_all_small_environments_fixed(self, partition_states):
        """
        Fixed version that properly updates global maps from all partitions.
        This should be called periodically to ensure global state consistency.
        """
        print(f"\nUpdating global maps from all {len(partition_states)} active partitions...")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active' or partition_state['timestep'] is None:
                continue
                
            self._update_global_maps_from_single_partition(partition_idx, partition_states)
        
        print(f"Global maps updated from all partitions")


    def verify_global_sync_consistency(self, partition_states):
        """
        Verify that all partitions have consistent global map data.
        Useful for debugging synchronization issues.
        """
        print(f"\n=== VERIFYING GLOBAL SYNC CONSISTENCY ===")
        
        consistent = True
        maps_to_check = ['action_map', 'dumpability_mask', 'dumpability_mask_init']
        
        # Check if we have global maps
        if not hasattr(self, 'global_maps'):
            print("❌ No global_maps attribute found!")
            return False
        
        for map_name in maps_to_check:
            if map_name not in self.global_maps:
                print(f"❌ {map_name} not found in global_maps")
                consistent = False
                continue
                
            global_map = self.global_maps[map_name]
            
            for partition_idx, partition_state in partition_states.items():
                if partition_state['status'] != 'active':
                    continue
                    
                try:
                    partition_map = getattr(partition_state['timestep'].state.world, map_name).map
                    
                    # Compare maps
                    maps_match = jnp.array_equal(global_map, partition_map)
                    
                    if not maps_match:
                        differences = jnp.sum(global_map != partition_map)
                        print(f"  ❌ {map_name} mismatch in partition {partition_idx}: {differences} different cells")
                        consistent = False
                    else:
                        print(f"  ✅ {map_name} consistent in partition {partition_idx}")
                        
                except Exception as e:
                    print(f"  ❌ Error checking {map_name} in partition {partition_idx}: {e}")
                    consistent = False
        
        # Special check for traversability (should have same terrain, different agents)
        print(f"\n--- Checking traversability consistency (terrain only) ---")
        if 'traversability_mask' in self.global_maps:
            global_traversability = self.global_maps['traversability_mask']
            
            for partition_idx, partition_state in partition_states.items():
                if partition_state['status'] != 'active':
                    continue
                    
                try:
                    partition_traversability = partition_state['timestep'].state.world.traversability_mask.map
                    
                    # Compare only terrain (ignore agent positions)
                    global_terrain = jnp.where(global_traversability == -1, 0, global_traversability)
                    partition_terrain = jnp.where(partition_traversability == -1, 0, partition_traversability)
                    
                    # Also ignore dumped soil and other dynamic obstacles for this check
                    global_base = jnp.where(global_terrain == 1, 1, 0)  # Only permanent terrain
                    partition_base = jnp.where(partition_terrain == 1, 1, 0)  # Only permanent terrain
                    
                    if hasattr(self, 'base_traversability_masks') and partition_idx in self.base_traversability_masks:
                        base_mask = self.base_traversability_masks[partition_idx]
                        base_terrain = jnp.where(base_mask == 1, 1, 0)
                        
                        terrain_match = jnp.array_equal(base_terrain, partition_base)
                        if terrain_match:
                            print(f"  ✅ Traversability base terrain consistent in partition {partition_idx}")
                        else:
                            differences = jnp.sum(base_terrain != partition_base)
                            print(f"  ⚠️  Traversability base terrain differs in partition {partition_idx}: {differences} cells")
                    else:
                        print(f"  ⚠️  No base mask for partition {partition_idx} to compare traversability")
                        
                except Exception as e:
                    print(f"  ❌ Error checking traversability in partition {partition_idx}: {e}")
                    consistent = False
        
        if consistent:
            print("🎉 All global sync checks passed!")
        else:
            print("❌ Some global sync issues detected!")
        
        return consistent
    
    def verify_traversability_correctness(self, partition_states):
        """
        Verify that traversability masks are correct:
        - No agent traces (old -1 values)
        - Other agents appear as obstacles (1), not agents (-1)
        - Each partition has exactly one agent area (-1)
        """
        print(f"\n=== VERIFYING TRAVERSABILITY CORRECTNESS ===")
        
        all_correct = True
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            traversability = partition_state['timestep'].state.world.traversability_mask.map
            
            # Count different values
            free_space = jnp.sum(traversability == 0)
            obstacles = jnp.sum(traversability == 1)
            agents = jnp.sum(traversability == -1)
            
            print(f"  Partition {partition_idx}:")
            print(f"    Free space (0): {free_space}")
            print(f"    Obstacles (1): {obstacles}")
            print(f"    Agent cells (-1): {agents}")
            
            # Check for reasonable agent size (should be small area, not traces)
            if agents > 50:  # Adjust threshold based on your agent size
                print(f"    ⚠️  Too many agent cells - possible trace issue")
                all_correct = False
            elif agents == 0:
                print(f"    ⚠️  No agent found")
                all_correct = False
            else:
                print(f"    ✅ Agent size looks correct")
        
        return all_correct
    
    def _update_partition_traversability_with_dumped_soil(self, target_partition_idx, target_partition_state, 
                                                     all_agent_positions, partition_states):
        """
        FIXED: Clean approach to updating traversability that includes dumped soil as obstacles.
        
        Traversability logic:
        - 0: Free space (can drive through)
        - 1: Obstacles (terrain + other agents + dumped soil)
        - -1: Current agent position
        """
        current_timestep = target_partition_state['timestep']
        
        # STEP 1: Start from completely clean base mask (original terrain only)
        if target_partition_idx in self.base_traversability_masks:
            clean_traversability = self.base_traversability_masks[target_partition_idx].copy()
            print(f"    Starting from clean base for partition {target_partition_idx}")
        else:
            print(f"    WARNING: No base mask for partition {target_partition_idx}, creating clean mask")
            current_mask = current_timestep.state.world.traversability_mask.map
            clean_traversability = jnp.where(
                (current_mask == -1) | (current_mask == 1),  # Remove all agent markers
                0,  # Set to free space
                current_mask  # Keep original terrain
            )
        
        # STEP 2: Add dumped soil areas as obstacles
        action_map = current_timestep.state.world.action_map.map
        dumped_areas = (action_map > 0)  # Positive values = dumped soil
        
        # Mark dumped soil areas as obstacles (1)
        clean_traversability = jnp.where(
            dumped_areas,
            1,  # Dumped soil = obstacle
            clean_traversability  # Keep existing values
        )
        
        dumped_obstacle_count = jnp.sum(dumped_areas)
        if dumped_obstacle_count > 0:
            print(f"    Added {dumped_obstacle_count} dumped soil obstacles to partition {target_partition_idx}")
        
        # STEP 3: Add THIS partition's agent positions as agents (-1)
        if target_partition_idx in all_agent_positions:
            own_positions = all_agent_positions[target_partition_idx]
            for cell_y, cell_x in own_positions:
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    clean_traversability = clean_traversability.at[cell_y, cell_x].set(-1)
            
            print(f"    Added {len(own_positions)} own agent cells to partition {target_partition_idx}")
        
        # STEP 4: Add OTHER agents as OBSTACLES (1), not agents
        other_agents_added = 0
        other_cells_added = 0
        
        for other_partition_idx, other_positions in all_agent_positions.items():
            if other_partition_idx == target_partition_idx:
                continue  # Skip own agent
                
            for cell_y, cell_x in other_positions:
                # Check bounds
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    
                    # Add as OBSTACLE (1), not agent (-1)
                    # Only if it's currently free space (0) - don't overwrite own agent or existing obstacles
                    current_value = clean_traversability[cell_y, cell_x]
                    if current_value == 0:  # Free space
                        clean_traversability = clean_traversability.at[cell_y, cell_x].set(1)
                        other_cells_added += 1
                    elif current_value == -1:  # Don't overwrite own agent
                        print(f"      Conflict: Other agent at own agent position ({cell_y}, {cell_x})")
            
            if other_cells_added > 0:
                other_agents_added += 1
        
        # STEP 5: Update the world state
        updated_world = self._update_world_map(
            current_timestep.state.world, 
            'traversability_mask', 
            clean_traversability
        )
        updated_state = current_timestep.state._replace(world=updated_world)
        updated_timestep = current_timestep._replace(state=updated_state)
        
        partition_states[target_partition_idx]['timestep'] = updated_timestep
        
        # Debug output with dumped soil info
        terrain_count = jnp.sum(self.base_traversability_masks.get(target_partition_idx, jnp.zeros_like(clean_traversability)) == 1)
        dumped_soil_count = jnp.sum(dumped_areas)
        own_agent_count = jnp.sum(clean_traversability == -1)
        other_obstacle_count = other_cells_added
        total_obstacles = jnp.sum(clean_traversability == 1)
        
        print(f"    Partition {target_partition_idx} traversability:")
        print(f"      Original terrain obstacles: {terrain_count}")
        print(f"      Dumped soil obstacles: {dumped_soil_count}")
        print(f"      Other agents (as obstacles): {other_obstacle_count}")
        print(f"      Total obstacles: {total_obstacles}")
        print(f"      Own agent cells: {own_agent_count}")
        
        if other_agents_added > 0:
            print(f"    ✅ Added {other_agents_added} other agents as obstacles ({other_cells_added} cells)")



    def verify_traversability_with_dumped_soil(self, partition_states):
        """
        UPDATED: Verify that traversability masks correctly include dumped soil as obstacles.
        """
        print(f"\n=== VERIFYING TRAVERSABILITY WITH DUMPED SOIL ===")
        
        all_correct = True
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            timestep = partition_state['timestep']
            traversability = timestep.state.world.traversability_mask.map
            action_map = timestep.state.world.action_map.map
            
            # Count different values
            free_space = jnp.sum(traversability == 0)
            obstacles = jnp.sum(traversability == 1)
            agents = jnp.sum(traversability == -1)
            
            # Count dumped soil in action map
            dumped_soil = jnp.sum(action_map > 0)
            dug_areas = jnp.sum(action_map == -1)
            
            print(f"  Partition {partition_idx}:")
            print(f"    Traversability - Free: {free_space}, Obstacles: {obstacles}, Agents: {agents}")
            print(f"    Action map - Dumped soil: {dumped_soil}, Dug areas: {dug_areas}")
            
            # Verify dumped soil is marked as obstacles in traversability
            dumped_positions = jnp.where(action_map > 0)
            dumped_soil_obstacles = 0
            
            if len(dumped_positions[0]) > 0:
                for i in range(len(dumped_positions[0])):
                    y, x = dumped_positions[0][i], dumped_positions[1][i]
                    if traversability[y, x] == 1:  # Should be obstacle
                        dumped_soil_obstacles += 1
                    elif traversability[y, x] == 0:  # Should NOT be free space
                        print(f"    ❌ Dumped soil at ({y}, {x}) is marked as free space!")
                        all_correct = False
            
            print(f"    Dumped soil correctly marked as obstacles: {dumped_soil_obstacles}/{dumped_soil}")
            
            # Check for reasonable agent size
            if agents > 50:
                print(f"    ⚠️  Too many agent cells - possible trace issue")
                all_correct = False
            elif agents == 0:
                print(f"    ⚠️  No agent found")
                all_correct = False
            else:
                print(f"    ✅ Agent size looks correct")
            
            # Check dumped soil consistency
            if dumped_soil_obstacles == dumped_soil:
                print(f"    ✅ All dumped soil correctly marked as obstacles")
            else:
                print(f"    ❌ Some dumped soil not marked as obstacles")
                all_correct = False
        
        if all_correct:
            print("🎉 All traversability checks passed (including dumped soil)!")
        else:
            print("❌ Some traversability issues detected!")
        
        return all_correct

    def debug_action_traversability_consistency(self, partition_states, step_name=""):
        """
        Debug method to check consistency between action_map and traversability_mask.
        """
        print(f"\n--- ACTION/TRAVERSABILITY DEBUG: {step_name} ---")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            timestep = partition_state['timestep']
            traversability = timestep.state.world.traversability_mask.map
            action_map = timestep.state.world.action_map.map
            
            # Check dumped soil consistency
            dumped_positions = jnp.where(action_map > 0)
            inconsistencies = 0
            
            if len(dumped_positions[0]) > 0:
                for i in range(len(dumped_positions[0])):
                    y, x = dumped_positions[0][i], dumped_positions[1][i]
                    if traversability[y, x] != 1:  # Should be obstacle
                        inconsistencies += 1
            
            print(f"  Partition {partition_idx}:")
            print(f"    Dumped soil areas: {len(dumped_positions[0]) if len(dumped_positions[0]) > 0 else 0}")
            print(f"    Inconsistencies: {inconsistencies}")
            
            if inconsistencies > 0:
                print(f"    ❌ {inconsistencies} dumped soil areas not marked as obstacles!")
            else:
                print(f"    ✅ All dumped soil correctly marked as obstacles")


    def initialize_partition_specific_target_maps(self, partition_states):
        """
        Store the original partition-specific target maps.
        Each partition should only see their own targets, never targets from other partitions.
        Call this ONCE after partition initialization.
        """
        if not hasattr(self, 'partition_target_maps'):
            self.partition_target_maps = {}
        
        print("Storing partition-specific target maps...")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] == 'active':
                # Store the original target map for this partition
                original_target_map = partition_state['timestep'].state.world.target_map.map.copy()
                self.partition_target_maps[partition_idx] = original_target_map
                
                # Count targets for verification
                dig_targets = jnp.sum(original_target_map == -1)
                dump_targets = jnp.sum(original_target_map == 1)
                
                print(f"  Partition {partition_idx}: {dig_targets} dig targets, {dump_targets} dump targets")

    def _create_world_with_global_maps_preserve_targets(self, current_world, partition_idx):
        """
        FIXED: Create a new world state that uses global maps but preserves partition-specific targets.
        """
        # Get the original partition-specific target map
        if hasattr(self, 'partition_target_maps') and partition_idx in self.partition_target_maps:
            partition_target_map = self.partition_target_maps[partition_idx]
            print(f"    Preserving original target map for partition {partition_idx}")
        else:
            # Fallback to current target map
            partition_target_map = current_world.target_map.map
            print(f"    WARNING: Using current target map for partition {partition_idx} (no stored original)")
        
        updated_world = current_world._replace(
            target_map=current_world.target_map._replace(map=partition_target_map),  # Keep partition-specific
            action_map=current_world.action_map._replace(map=self.global_maps['action_map']),
            dumpability_mask=current_world.dumpability_mask._replace(map=self.global_maps['dumpability_mask']),
            dumpability_mask_init=current_world.dumpability_mask_init._replace(map=self.global_maps['dumpability_mask_init']),
            padding_mask=current_world.padding_mask._replace(map=self.global_maps['padding_mask'])
            # NOTE: traversability_mask will be handled separately by agent sync
        )
        
        return updated_world
    
    def verify_partition_target_isolation(self, partition_states):
        """
        Verify that each partition only sees their own targets, not targets from other partitions.
        """
        print(f"\n=== VERIFYING TARGET MAP ISOLATION ===")
        
        all_correct = True
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            try:
                current_target_map = partition_state['timestep'].state.world.target_map.map
                
                if hasattr(self, 'partition_target_maps') and partition_idx in self.partition_target_maps:
                    original_target_map = self.partition_target_maps[partition_idx]
                    
                    # Check if current target map matches the original partition-specific one
                    if jnp.array_equal(current_target_map, original_target_map):
                        print(f"  ✅ Partition {partition_idx}: Target map unchanged (correct)")
                    else:
                        differences = jnp.sum(current_target_map != original_target_map)
                        print(f"  ❌ Partition {partition_idx}: Target map changed! {differences} different cells")
                        all_correct = False
                        
                        # Debug: check what changed
                        gained_dig_targets = jnp.sum((current_target_map == -1) & (original_target_map != -1))
                        gained_dump_targets = jnp.sum((current_target_map == 1) & (original_target_map != 1))
                        lost_dig_targets = jnp.sum((current_target_map != -1) & (original_target_map == -1))
                        lost_dump_targets = jnp.sum((current_target_map != 1) & (original_target_map == 1))
                        
                        if gained_dig_targets > 0 or gained_dump_targets > 0:
                            print(f"    Gained {gained_dig_targets} dig targets, {gained_dump_targets} dump targets")
                            print(f"    This suggests targets from other partitions are bleeding in!")
                        
                        if lost_dig_targets > 0 or lost_dump_targets > 0:
                            print(f"    Lost {lost_dig_targets} dig targets, {lost_dump_targets} dump targets")
                            print(f"    This suggests partition targets are being overwritten!")
                else:
                    print(f"  ⚠️  Partition {partition_idx}: No original target map stored")
                    all_correct = False
                
                # Count current targets
                current_dig_targets = jnp.sum(current_target_map == -1)
                current_dump_targets = jnp.sum(current_target_map == 1)
                current_free = jnp.sum(current_target_map == 0)
                print(f"    Current targets: {current_dig_targets} dig, {current_dump_targets} dump, {current_free} free")
                
            except Exception as e:
                print(f"  ❌ Error checking partition {partition_idx}: {e}")
                all_correct = False
        
        if all_correct:
            print("🎉 All partitions have isolated target maps!")
        else:
            print("❌ Target map isolation violated!")
        
        return all_correct

    def debug_target_map_changes(self, partition_states, step_name=""):
        """
        Debug method to track when and how target maps change.
        """
        print(f"\n--- TARGET MAP DEBUG: {step_name} ---")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            current_target_map = partition_state['timestep'].state.world.target_map.map
            
            dig_targets = jnp.sum(current_target_map == -1)
            dump_targets = jnp.sum(current_target_map == 1)
            free_space = jnp.sum(current_target_map == 0)
            
            print(f"  Partition {partition_idx}: Dig={dig_targets}, Dump={dump_targets}, Free={free_space}")
            
            # Check for unexpected values
            unique_values = jnp.unique(current_target_map)
            expected_values = jnp.array([-1, 0, 1])
            unexpected = jnp.setdiff1d(unique_values, expected_values)
            
            if len(unexpected) > 0:
                print(f"    ⚠️  Unexpected target map values: {unexpected}")
    def verify_target_blocking(self, partition_states):
        """
        Verify that other partitions' targets are properly blocked as obstacles.
        """
        print(f"\n=== VERIFYING TARGET BLOCKING ===")
        
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            traversability = target_partition_state['timestep'].state.world.traversability_mask.map
            
            targets_blocked = 0
            
            # Check each other partition
            for other_partition_idx, other_partition_state in partition_states.items():
                if (other_partition_idx == target_partition_idx or 
                    other_partition_state['status'] != 'active'):
                    continue
                    
                # Get other partition's original targets
                if hasattr(self, 'partition_target_maps') and other_partition_idx in self.partition_target_maps:
                    other_targets = self.partition_target_maps[other_partition_idx]
                    other_target_positions = jnp.where((other_targets == -1) | (other_targets == 1))
                    
                    # Check if these are blocked in current partition's traversability
                    blocked_count = 0
                    for i in range(len(other_target_positions[0])):
                        y, x = other_target_positions[0][i], other_target_positions[1][i]
                        if traversability[y, x] == 1:  # Should be obstacle
                            blocked_count += 1
                    
                    targets_blocked += blocked_count
                    print(f"  Partition {target_partition_idx}: {blocked_count} targets from partition {other_partition_idx} blocked")
            
            print(f"  Partition {target_partition_idx}: Total {targets_blocked} other targets blocked")
        
        print("✅ Target blocking verification complete")

    
    def _update_partition_traversability_debug(self, target_partition_idx, target_partition_state, 
                                            all_agent_positions, partition_states):
        """
        DEBUG VERSION: Step-by-step traversability update with detailed logging.
        This will help identify what's marking everything as obstacles.
        """
        current_timestep = target_partition_state['timestep']
        
        print(f"\n=== DEBUG TRAVERSABILITY FOR PARTITION {target_partition_idx} ===")
        
        # STEP 1: Check base mask
        if target_partition_idx in self.base_traversability_masks:
            clean_traversability = self.base_traversability_masks[target_partition_idx].copy()
            print(f"  Using stored base mask for partition {target_partition_idx}")
        else:
            print(f"  WARNING: No base mask for partition {target_partition_idx}")
            current_mask = current_timestep.state.world.traversability_mask.map
            clean_traversability = jnp.where(
                (current_mask == -1) | (current_mask == 1),
                0,
                current_mask
            )
        
        # Debug base mask
        base_free = jnp.sum(clean_traversability == 0)
        base_obstacles = jnp.sum(clean_traversability == 1)
        base_agents = jnp.sum(clean_traversability == -1)
        total_cells = clean_traversability.size
        
        print(f"  BASE MASK: Free={base_free}, Obstacles={base_obstacles}, Agents={base_agents}, Total={total_cells}")
        
        if base_free < total_cells * 0.5:  # Less than 50% free space is suspicious
            print(f"  ⚠️  WARNING: Base mask has very little free space ({base_free}/{total_cells} = {base_free/total_cells:.1%})")
        
        # STEP 2: Check dumped soil
        action_map = current_timestep.state.world.action_map.map
        dumped_areas = (action_map > 0)
        dumped_count = jnp.sum(dumped_areas)
        
        print(f"  DUMPED SOIL: {dumped_count} cells")
        
        if dumped_count > total_cells * 0.3:  # More than 30% dumped is suspicious
            print(f"  ⚠️  WARNING: Excessive dumped soil ({dumped_count}/{total_cells} = {dumped_count/total_cells:.1%})")
        
        # Apply dumped soil obstacles
        clean_traversability = jnp.where(dumped_areas, 1, clean_traversability)
        
        after_dumped_free = jnp.sum(clean_traversability == 0)
        after_dumped_obstacles = jnp.sum(clean_traversability == 1)
        
        print(f"  AFTER DUMPED: Free={after_dumped_free}, Obstacles={after_dumped_obstacles}")
        
        # STEP 3: Check target blocking
        other_targets_blocked = 0
        
        for other_partition_idx, other_partition_state in partition_states.items():
            if (other_partition_idx == target_partition_idx or 
                other_partition_state['status'] != 'active'):
                continue
                
            print(f"  Checking targets from partition {other_partition_idx}...")
            
            if hasattr(self, 'partition_target_maps') and other_partition_idx in self.partition_target_maps:
                other_target_map = self.partition_target_maps[other_partition_idx]
                
                # Debug the other target map
                other_dig = jnp.sum(other_target_map == -1)
                other_dump = jnp.sum(other_target_map == 1)
                other_free = jnp.sum(other_target_map == 0)
                
                print(f"    Other partition {other_partition_idx} targets: Dig={other_dig}, Dump={other_dump}, Free={other_free}")
                
                # Check if shapes match
                if other_target_map.shape != clean_traversability.shape:
                    print(f"    ❌ SHAPE MISMATCH: other_target_map={other_target_map.shape}, traversability={clean_traversability.shape}")
                    continue
                
                other_all_targets = (other_target_map == -1) | (other_target_map == 1)
                targets_to_block = jnp.sum(other_all_targets)
                
                print(f"    Blocking {targets_to_block} targets from partition {other_partition_idx}")
                
                if targets_to_block > total_cells * 0.5:  # More than 50% is suspicious
                    print(f"    ⚠️  WARNING: Blocking excessive targets ({targets_to_block}/{total_cells} = {targets_to_block/total_cells:.1%})")
                    
                    # Show a sample of what's being blocked
                    sample_positions = jnp.where(other_all_targets)
                    if len(sample_positions[0]) > 0:
                        sample_indices = jnp.arange(0, min(5, len(sample_positions[0])))
                        for i in sample_indices:
                            y, x = sample_positions[0][i], sample_positions[1][i]
                            print(f"      Sample blocked position: ({y}, {x}) = {other_target_map[y, x]}")
                
                # Apply target blocking
                clean_traversability = jnp.where(other_all_targets, 1, clean_traversability)
                other_targets_blocked += targets_to_block
            else:
                print(f"    No stored target map for partition {other_partition_idx}")
        
        after_targets_free = jnp.sum(clean_traversability == 0)
        after_targets_obstacles = jnp.sum(clean_traversability == 1)
        
        print(f"  AFTER TARGET BLOCKING: Free={after_targets_free}, Obstacles={after_targets_obstacles}")
        print(f"  Total targets blocked: {other_targets_blocked}")
        
        # STEP 4: Add own agent
        if target_partition_idx in all_agent_positions:
            own_positions = all_agent_positions[target_partition_idx]
            for cell_y, cell_x in own_positions:
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    clean_traversability = clean_traversability.at[cell_y, cell_x].set(-1)
            print(f"  Added {len(own_positions)} own agent cells")
        
        # STEP 5: Add other agents
        other_agents_added = 0
        for other_partition_idx, other_positions in all_agent_positions.items():
            if other_partition_idx == target_partition_idx:
                continue
                
            for cell_y, cell_x in other_positions:
                if (0 <= cell_y < clean_traversability.shape[0] and 
                    0 <= cell_x < clean_traversability.shape[1]):
                    if clean_traversability[cell_y, cell_x] == 0:
                        clean_traversability = clean_traversability.at[cell_y, cell_x].set(1)
                        other_agents_added += 1
        
        print(f"  Added {other_agents_added} other agent obstacle cells")
        
        # FINAL RESULT
        final_free = jnp.sum(clean_traversability == 0)
        final_obstacles = jnp.sum(clean_traversability == 1)
        final_agents = jnp.sum(clean_traversability == -1)
        
        print(f"  FINAL RESULT: Free={final_free}, Obstacles={final_obstacles}, Agents={final_agents}")
        print(f"  Free space percentage: {final_free/total_cells:.1%}")
        
        if final_free < total_cells * 0.1:  # Less than 10% free space
            print(f"  🚨 CRITICAL: Almost no free space! Something is wrong!")
            
            # Emergency diagnostic
            print(f"  DIAGNOSTIC BREAKDOWN:")
            print(f"    Original base free space: {base_free}")
            print(f"    Lost to dumped soil: {base_free - after_dumped_free}")
            print(f"    Lost to target blocking: {after_dumped_free - after_targets_free}")
            print(f"    Lost to other agents: {other_agents_added}")
        
        # Update the world state
        updated_world = self._update_world_map(
            current_timestep.state.world, 
            'traversability_mask', 
            clean_traversability
        )
        updated_state = current_timestep.state._replace(world=updated_world)
        updated_timestep = current_timestep._replace(state=updated_state)
        
        partition_states[target_partition_idx]['timestep'] = updated_timestep

    # QUICK FIX: Also add this method to check base mask initialization
    def debug_base_mask_initialization(self, partition_states):
        """
        Debug the base mask initialization to see if that's the problem.
        """
        print(f"\n=== DEBUGGING BASE MASK INITIALIZATION ===")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            # Check current traversability
            current_mask = partition_state['timestep'].state.world.traversability_mask.map
            current_free = jnp.sum(current_mask == 0)
            current_obstacles = jnp.sum(current_mask == 1)
            current_agents = jnp.sum(current_mask == -1)
            total = current_mask.size
            
            print(f"  Partition {partition_idx} current mask: Free={current_free}, Obstacles={current_obstacles}, Agents={current_agents}")
            print(f"    Free percentage: {current_free/total:.1%}")
            
            # Check stored base mask if exists
            if hasattr(self, 'base_traversability_masks') and partition_idx in self.base_traversability_masks:
                base_mask = self.base_traversability_masks[partition_idx]
                base_free = jnp.sum(base_mask == 0)
                base_obstacles = jnp.sum(base_mask == 1)
                base_agents = jnp.sum(base_mask == -1)
                
                print(f"  Partition {partition_idx} stored base: Free={base_free}, Obstacles={base_obstacles}, Agents={base_agents}")
                print(f"    Base free percentage: {base_free/total:.1%}")
                
                if base_free < total * 0.5:
                    print(f"    🚨 PROBLEM: Base mask has too few free cells!")
            else:
                print(f"  Partition {partition_idx}: No stored base mask")

    def verify_dig_target_blocking(self, partition_states):
        """
        Verify that other partitions' DIG TARGETS are properly blocked as obstacles.
        """
        print(f"\n=== VERIFYING DIG TARGET BLOCKING ===")
        
        for target_partition_idx, target_partition_state in partition_states.items():
            if target_partition_state['status'] != 'active':
                continue
                
            traversability = target_partition_state['timestep'].state.world.traversability_mask.map
            
            dig_targets_blocked = 0
            dump_targets_not_blocked = 0
            
            # Check each other partition
            for other_partition_idx, other_partition_state in partition_states.items():
                if (other_partition_idx == target_partition_idx or 
                    other_partition_state['status'] != 'active'):
                    continue
                    
                # Get other partition's original targets
                if hasattr(self, 'partition_target_maps') and other_partition_idx in self.partition_target_maps:
                    other_targets = self.partition_target_maps[other_partition_idx]
                    
                    # Check dig targets (should be blocked)
                    other_dig_positions = jnp.where(other_targets == -1)
                    if len(other_dig_positions[0]) > 0:
                        dig_blocked_count = 0
                        for i in range(len(other_dig_positions[0])):
                            y, x = other_dig_positions[0][i], other_dig_positions[1][i]
                            if traversability[y, x] == 1:  # Should be obstacle
                                dig_blocked_count += 1
                        dig_targets_blocked += dig_blocked_count
                        print(f"  Partition {target_partition_idx}: {dig_blocked_count}/{len(other_dig_positions[0])} dig targets from partition {other_partition_idx} blocked")
                    
                    # Check dump targets (should NOT be blocked)
                    other_dump_positions = jnp.where(other_targets == 1)
                    if len(other_dump_positions[0]) > 0:
                        dump_not_blocked_count = 0
                        for i in range(len(other_dump_positions[0])):
                            y, x = other_dump_positions[0][i], other_dump_positions[1][i]
                            if traversability[y, x] == 0:  # Should remain free space
                                dump_not_blocked_count += 1
                        dump_targets_not_blocked += dump_not_blocked_count
                        print(f"  Partition {target_partition_idx}: {dump_not_blocked_count}/{len(other_dump_positions[0])} dump targets from partition {other_partition_idx} remain free")
            
            print(f"  Partition {target_partition_idx}: Total {dig_targets_blocked} dig targets blocked, {dump_targets_not_blocked} dump targets remain free")
        
        print("✅ Dig target blocking verification complete")


    def debug_agent_positions_before_sync(self, partition_states, step_num):
        """
        Debug method to print agent positions before synchronization.
        """
        print(f"\n=== AGENT POSITIONS DEBUG - STEP {step_num} ===")
        
        all_positions = {}
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            # Get agent position from state
            agent_pos = partition_state['timestep'].state.agent.agent_state.pos_base
            
            # Get partition region coordinates
            partition = self.partitions[partition_idx]
            region_coords = partition['region_coords']
            
            # Convert to global coordinates (if using local coordinates)
            # This might be where the bug is!
            global_pos = self.map_position_small_to_global(agent_pos, region_coords)
            
            print(f"  Agent {partition_idx}:")
            print(f"    Local position: {agent_pos}")
            print(f"    Region coords: {region_coords}")
            print(f"    Global position: {global_pos}")
            
            # Get traversability mask to see where agent appears
            traversability = partition_state['timestep'].state.world.traversability_mask.map
            agent_cells = jnp.where(traversability == -1)
            
            print(f"    Agent cells in traversability mask: {len(agent_cells[0])} cells")
            if len(agent_cells[0]) > 0:
                for i in range(min(5, len(agent_cells[0]))):  # Show first 5 cells
                    y, x = agent_cells[0][i], agent_cells[1][i]
                    print(f"      Cell {i}: ({y}, {x})")
            
            all_positions[partition_idx] = {
                'local_pos': agent_pos,
                'global_pos': global_pos,
                'occupied_cells': [(int(agent_cells[0][i]), int(agent_cells[1][i])) 
                                for i in range(len(agent_cells[0]))]
            }
        
        # Check for overlaps
        print(f"\n--- OVERLAP DETECTION ---")
        partitions = list(all_positions.keys())
        for i in range(len(partitions)):
            for j in range(i + 1, len(partitions)):
                p1, p2 = partitions[i], partitions[j]
                
                cells1 = set(all_positions[p1]['occupied_cells'])
                cells2 = set(all_positions[p2]['occupied_cells'])
                
                overlap = cells1.intersection(cells2)
                if overlap:
                    print(f"  ❌ OVERLAP between agents {p1} and {p2}: {overlap}")
                else:
                    print(f"  ✅ No overlap between agents {p1} and {p2}")
        
        return all_positions

    def debug_traversability_after_sync(self, partition_states, step_num):
        """
        Debug method to verify traversability masks after synchronization.
        """
        print(f"\n=== TRAVERSABILITY DEBUG AFTER SYNC - STEP {step_num} ===")
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            traversability = partition_state['timestep'].state.world.traversability_mask.map
            
            # Count different values
            free_space = jnp.sum(traversability == 0)
            obstacles = jnp.sum(traversability == 1)
            agents = jnp.sum(traversability == -1)
            
            print(f"  Partition {partition_idx} traversability:")
            print(f"    Free space (0): {free_space}")
            print(f"    Obstacles (1): {obstacles}")  
            print(f"    Agent cells (-1): {agents}")
            
            # Check if other agents appear as obstacles
            agent_positions = jnp.where(traversability == -1)
            obstacle_positions = jnp.where(traversability == 1)
            
            print(f"    Agent cells: {[(int(agent_positions[0][i]), int(agent_positions[1][i])) for i in range(len(agent_positions[0]))]}")
            
            if len(obstacle_positions[0]) > 0:
                print(f"    First 5 obstacle cells: {[(int(obstacle_positions[0][i]), int(obstacle_positions[1][i])) for i in range(min(5, len(obstacle_positions[0])))]}")

    # Add this to your main loop for debugging:
    def debug_sync_effectiveness(self, partition_states, step_num):
        """
        Complete debugging method to call in your main loop.
        """
        print(f"\n{'='*80}")
        print(f"SYNC DEBUG - STEP {step_num}")
        print(f"{'='*80}")
        
        # Debug before sync
        positions_before = self.debug_agent_positions_before_sync(partition_states, step_num)
        
        # Debug after sync  
        self.debug_traversability_after_sync(partition_states, step_num)
        
        # Verify sync worked
        all_agent_positions = {}
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] != 'active':
                continue
                
            traversability = partition_state['timestep'].state.world.traversability_mask.map
            agent_mask = (traversability == -1)
            agent_positions = jnp.where(agent_mask)
            
            if len(agent_positions[0]) > 0:
                occupied_cells = []
                for i in range(len(agent_positions[0])):
                    cell = (int(agent_positions[0][i]), int(agent_positions[1][i]))
                    occupied_cells.append(cell)
                all_agent_positions[partition_idx] = occupied_cells
        
        # Final overlap check
        print(f"\n--- FINAL OVERLAP CHECK ---")
        partitions = list(all_agent_positions.keys())
        overlaps_found = 0
        
        for i in range(len(partitions)):
            for j in range(i + 1, len(partitions)):
                p1, p2 = partitions[i], partitions[j]
                
                if p1 in all_agent_positions and p2 in all_agent_positions:
                    cells1 = set(all_agent_positions[p1])
                    cells2 = set(all_agent_positions[p2])
                    
                    overlap = cells1.intersection(cells2)
                    if overlap:
                        print(f"  ❌ FINAL OVERLAP between agents {p1} and {p2}: {overlap}")
                        overlaps_found += 1
                    else:
                        print(f"  ✅ No final overlap between agents {p1} and {p2}")
        
        if overlaps_found == 0:
            print("🎉 NO OVERLAPS DETECTED - SYNC WORKING!")
        else:
            print(f"❌ {overlaps_found} OVERLAPS STILL PRESENT - SYNC FAILED!")
        
        return overlaps_found == 0