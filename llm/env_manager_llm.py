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
    

    # def map_position_small_to_global(self, small_pos, region_coords):
    #     """
    #     Map agent position from small map coordinates to global map coordinates.
    #     Returns position in (x, y) format for rendering.
    #  """
    #     y_start, x_start, y_end, x_end = region_coords
    
    #     # Extract position values
    #     if hasattr(small_pos, 'shape'):
    #         if len(small_pos.shape) == 1 and small_pos.shape[0] == 2:
    #             small_pos_y = float(small_pos[0])
    #             small_pos_x = float(small_pos[1])
    #         else:
    #             small_pos_y = float(small_pos.flatten()[0])
    #             small_pos_x = float(small_pos.flatten()[1])
    #     else:
    #         small_pos_y = float(small_pos[0])
    #         small_pos_x = float(small_pos[1])
    
    #     # Simple mapping: just add the region offset
    #     global_pos_y = small_pos_y + y_start
    #     global_pos_x = small_pos_x + x_start
    
    #     # Ensure position is within valid bounds
    #     global_pos_y = max(0, min(127, global_pos_y))
    #     global_pos_x = max(0, min(127, global_pos_x))
    
    #     # Return as (x, y) for rendering instead of (y, x)
    #     return (int(global_pos_x), int(global_pos_y))


    # def map_position_small_to_global(self, small_pos, region_coords):
    #     """
    #     Map agent position from small map coordinates to global map coordinates.
    #     Since create_sub_task_* functions preserve positions, the agent position
    #     in the small environment is already in global coordinates.
    #     Returns position in (x, y) format for rendering.
    #     """
    #     # Extract position values
    #     if hasattr(small_pos, 'shape'):
    #         if len(small_pos.shape) == 1 and small_pos.shape[0] == 2:
    #             # Check if position is stored as [x, y] or [y, x]
    #             # Based on the agent state structure, pos_base appears to be [x, y]
    #             pos_x = float(small_pos[0])
    #             pos_y = float(small_pos[1])
    #         else:
    #             pos_x = float(small_pos.flatten()[0])
    #             pos_y = float(small_pos.flatten()[1])
    #     else:
    #         pos_x = float(small_pos[0])
    #         pos_y = float(small_pos[1])
        
    #     # Since the small environment preserves global coordinates,
    #     # we don't need to add any offset - just ensure bounds
    #     global_pos_x = max(0, min(64, pos_x))
    #     global_pos_y = max(0, min(64, pos_y))
        
    #     # Return as (x, y) for rendering
    #     return (int(global_pos_x), int(global_pos_y))
    
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
    
    # Updated synchronization methods for the new global map approach

    # def sync_agents_in_global_environment(self, partition_states):
    #     """
    #     New synchronization approach for shared global maps.
    #     Focus on agent positioning and collision detection rather than map synchronization.
    #     """
    #     print(f"\n=== SYNCING AGENTS IN GLOBAL ENVIRONMENT ===")
        
    #     # Collect all active agent positions and their occupied cells
    #     all_agent_positions = {}
    #     all_occupied_cells = {}
        
    #     for partition_idx, partition_state in partition_states.items():
    #         if partition_state['status'] != 'active':
    #             continue
                
    #         current_timestep = partition_state['timestep']
    #         traversability = current_timestep.state.world.traversability_mask.map
            
    #         # Find where this agent is (value = -1)
    #         agent_mask = (traversability == -1)
    #         agent_positions = jnp.where(agent_mask)
            
    #         if len(agent_positions[0]) > 0:
    #             # Store agent position info
    #             all_agent_positions[partition_idx] = {
    #                 'positions': agent_positions,
    #                 'count': len(agent_positions[0])
    #             }
                
    #             # Store occupied cells for this agent
    #             occupied_cells = []
    #             for i in range(len(agent_positions[0])):
    #                 cell = (int(agent_positions[0][i]), int(agent_positions[1][i]))
    #                 occupied_cells.append(cell)
    #             all_occupied_cells[partition_idx] = occupied_cells
        
    #     # Update each partition's traversability mask with other agents
    #     for target_partition_idx, target_partition_state in partition_states.items():
    #         if target_partition_state['status'] != 'active':
    #             continue
                
    #         self._update_partition_with_other_agents(
    #             target_partition_idx, target_partition_state, 
    #             all_occupied_cells, partition_states
    #         )
        
    #     print(f"Agent synchronization completed for {len(all_agent_positions)} active agents")

    # def _update_partition_with_other_agents(self, target_partition_idx, target_partition_state, 
    #                                     all_occupied_cells, partition_states):
    #     """
    #     Update a partition's traversability mask to show other agents as obstacles.
    #     """
    #     current_timestep = target_partition_state['timestep']
    #     current_traversability = current_timestep.state.world.traversability_mask.map.copy()
        
    #     agents_added = 0
    #     cells_added = 0
        
    #     # Add other agents as obstacles
    #     for other_partition_idx, occupied_cells in all_occupied_cells.items():
    #         if other_partition_idx == target_partition_idx:
    #             continue  # Don't add self
                
    #         for cell_y, cell_x in occupied_cells:
    #             # Check if this cell is within the partition's area of interest
    #             if self._should_show_agent_in_partition(target_partition_idx, cell_y, cell_x):
    #                 # Only mark as obstacle if it's currently free space or traversable
    #                 if current_traversability[cell_y, cell_x] == 0:
    #                     current_traversability = current_traversability.at[cell_y, cell_x].set(1)
    #                     cells_added += 1
            
    #         if cells_added > 0:
    #             agents_added += 1
        
    #     # Update the world state if changes were made
    #     if agents_added > 0:
    #         updated_world = self._update_world_map(
    #             current_timestep.state.world, 
    #             'traversability_mask', 
    #             current_traversability
    #         )
    #         updated_state = current_timestep.state._replace(world=updated_world)
    #         updated_timestep = current_timestep._replace(state=updated_state)
            
    #         partition_states[target_partition_idx]['timestep'] = updated_timestep
            
    #         print(f"  ✓ Added {agents_added} agents ({cells_added} cells) to partition {target_partition_idx}")

    # def _update_partition_with_other_agents(self, target_partition_idx, target_partition_state, 
    #                                 all_occupied_cells, partition_states):
    #     """
    #     Update a partition's traversability mask to show other agents as obstacles.
    #     Fixed to prevent agent traces by clearing previous agent obstacles.
    #     """
    #     current_timestep = target_partition_state['timestep']
    #     current_traversability = current_timestep.state.world.traversability_mask.map.copy()
        
    #     # STEP 1: Clear all agent obstacles (value = 1) that were added in previous sync cycles
    #     # Keep permanent terrain obstacles and current agent position (-1)
        
    #     # Simple approach: Clear all obstacles marked as "1" (assumes these are agent obstacles)
    #     # and keep -1 (current agent) and any other permanent obstacle values
    #     current_traversability = jnp.where(
    #         current_traversability == 1,  # Agent obstacles from previous syncs
    #         0,  # Clear to free space
    #         current_traversability  # Keep current agent (-1) and permanent obstacles
    #     )
        
    #     # STEP 2: Add current positions of OTHER agents as obstacles
    #     agents_added = 0
    #     cells_added = 0
        
    #     for other_partition_idx, occupied_cells in all_occupied_cells.items():
    #         if other_partition_idx == target_partition_idx:
    #             continue  # Don't add self as obstacle
                
    #         for cell_y, cell_x in occupied_cells:
    #             # Check if this cell should be visible in this partition
    #             if self._should_show_agent_in_partition(target_partition_idx, cell_y, cell_x):
    #                 # Check bounds
    #                 if (0 <= cell_y < current_traversability.shape[0] and 
    #                     0 <= cell_x < current_traversability.shape[1]):
    #                     # Mark as obstacle (value = 1) - this represents another agent
    #                     current_traversability = current_traversability.at[cell_y, cell_x].set(1)
    #                     cells_added += 1
            
    #         if cells_added > 0:
    #             agents_added += 1
        
    #     # Always update the world state (even if no agents added, we might have cleared old obstacles)
    #     updated_world = self._update_world_map(
    #         current_timestep.state.world, 
    #         'traversability_mask', 
    #         current_traversability
    #     )
    #     updated_state = current_timestep.state._replace(world=updated_world)
    #     updated_timestep = current_timestep._replace(state=updated_state)
        
    #     partition_states[target_partition_idx]['timestep'] = updated_timestep
        
    #     if agents_added > 0:
    #         print(f"  ✓ Added {agents_added} current agents ({cells_added} cells) to partition {target_partition_idx}")
    #     else:
    #         print(f"  ✓ Cleared old agent traces from partition {target_partition_idx}")

    # def sync_agents_in_global_environment(self, partition_states):
    #     """
    #     Updated synchronization with proper cleanup of agent traces.
    #     Now properly clears old agent obstacles before adding current positions.
    #     """
    #     print(f"\n=== SYNCING AGENTS IN GLOBAL ENVIRONMENT ===")
        
    #     # Collect all active agent positions and their occupied cells
    #     all_agent_positions = {}
    #     all_occupied_cells = {}
        
    #     for partition_idx, partition_state in partition_states.items():
    #         if partition_state['status'] != 'active':
    #             continue
                
    #         current_timestep = partition_state['timestep']
    #         traversability = current_timestep.state.world.traversability_mask.map
            
    #         # Find where this agent is (value = -1)
    #         agent_mask = (traversability == -1)
    #         agent_positions = jnp.where(agent_mask)
            
    #         if len(agent_positions[0]) > 0:
    #             # Store agent position info
    #             all_agent_positions[partition_idx] = {
    #                 'positions': agent_positions,
    #                 'count': len(agent_positions[0])
    #             }
                
    #             # Store occupied cells for this agent
    #             occupied_cells = []
    #             for i in range(len(agent_positions[0])):
    #                 cell = (int(agent_positions[0][i]), int(agent_positions[1][i]))
    #                 occupied_cells.append(cell)
    #             all_occupied_cells[partition_idx] = occupied_cells
        
    #     # Update each partition's traversability mask with other agents
    #     for target_partition_idx, target_partition_state in partition_states.items():
    #         if target_partition_state['status'] != 'active':
    #             continue
                
    #         self._update_partition_with_other_agents(
    #             target_partition_idx, target_partition_state, 
    #             all_occupied_cells, partition_states
    #         )
        
    #     print(f"Agent synchronization completed for {len(all_agent_positions)} active agents")

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
        
        for partition_idx, partition_state in partition_states.items():
            if partition_state['status'] == 'active':
                # Get the current traversability mask
                current_mask = partition_state['timestep'].state.world.traversability_mask.map.copy()
                
                # Clean any agent markers but keep terrain obstacles
                # -1 = current agent position (clear to 0)
                # 1 = could be terrain obstacles or agent obstacles (we assume these are terrain at initialization)
                # 0 = free space (keep as is)
                clean_mask = jnp.where(
                    current_mask == -1,  # Current agent position
                    0,  # Clear agent position to free space
                    current_mask  # Keep everything else (terrain obstacles = 1, free space = 0)
                )
                
                self.base_traversability_masks[partition_idx] = clean_mask
                print(f"Stored base traversability mask for partition {partition_idx}")

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