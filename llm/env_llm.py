from terra.env import TerraEnvBatch
import jax.numpy as jnp

from llm.utils_llm import *

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
    

class LargeMapTerraEnv(TerraEnvBatchWithMapOverride):
    """A version of TerraEnvBatch specifically for 128x128 maps"""
    
    def reset_with_map_override(self, env_cfgs, rngs, custom_pos=None, custom_angle=None,
                           target_map_override=None, traversability_mask_override=None,
                           padding_mask_override=None, dumpability_mask_override=None,
                           dumpability_mask_init_override=None, action_map_override=None,
                           agent_config_override=None):
        """Reset with 64x64 map overrides - ensures shapes are consistent"""
    
        # Call the TerraEnvBatchWithMapOverride's reset_with_map_override method directly
        return TerraEnvBatchWithMapOverride.reset_with_map_override(
            self, env_cfgs, rngs, custom_pos, custom_angle,
            target_map_override, traversability_mask_override,
            padding_mask_override, dumpability_mask_override,
            dumpability_mask_init_override, action_map_override,
            agent_config_override
        )
    