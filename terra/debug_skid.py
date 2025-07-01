#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, '.')

try:
    from state import State
    from config import EnvConfig
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def debug_skid_step_by_step():
    print("🔍 DEBUGGING SKID STEER STEP BY STEP")
    
    # Minimal test case
    print("\n1. Creating minimal test environment...")
    images = jnp.zeros((8, 8), dtype=jnp.float32)
    actions = jnp.zeros((8, 8), dtype=jnp.int8)
    actions = actions.at[4, 4].set(1)  # Single dirt tile
    occupancy = jnp.ones((8, 8), dtype=jnp.bool_)
    dumpability = jnp.ones((8, 8), dtype=jnp.bool_)
    
    print(f"   Created action map with dirt at (4,4): {actions[4,4]}")
    
    key = jax.random.PRNGKey(42)
    env_cfg = EnvConfig()
    
    print("\n2. Creating state...")
    try:
        state = State.new(
            key=key,
            env_cfg=env_cfg,
            target_map=images,
            padding_mask=occupancy,
            trench_axes=jnp.array([0, 1]),
            trench_type=jnp.array([0]),
            dumpability_mask_init=dumpability,
            action_map=actions,
        )
        print("   ✅ State created successfully")
        print(f"   Agent types: {state.agent.agent_state.agent_type}, {state.agent.agent_state_2.agent_type}")
    except Exception as e:
        print(f"   ❌ State creation failed: {e}")
        return
        
    print("\n3. Positioning skid steer at dirt...")
    try:
        # Position skid steer exactly at dirt tile
        dirt_pos = jnp.array([4.0 * env_cfg.tile_size, 4.0 * env_cfg.tile_size])
        state = state._replace(
            agent=state.agent._replace(
                agent_state_2=state.agent.agent_state_2._replace(
                    pos_base=dirt_pos
                )
            )
        )
        print(f"   ✅ Skid steer positioned at: {state.agent.agent_state_2.pos_base}")
    except Exception as e:
        print(f"   ❌ Positioning failed: {e}")
        return
        
    print("\n4. Swapping to skid steer...")
    try:
        state_swapped = state._swap()
        print(f"   ✅ Active agent type: {state_swapped.agent.agent_state.agent_type}")
        print(f"   Initial loaded: {state_swapped.agent.agent_state.loaded[0]}")
    except Exception as e:
        print(f"   ❌ Swap failed: {e}")
        return
        
    print("\n5. Testing dig mask creation...")
    try:
        dig_mask = state_swapped._build_dig_dump_cone()
        print(f"   ✅ Dig mask created, sum: {dig_mask.sum()}")
    except Exception as e:
        print(f"   ❌ Dig mask creation failed: {e}")
        return
        
    print("\n6. Testing skid steer specific masking...")
    try:
        dig_mask_filtered = state_swapped._mask_out_wrong_dig_tiles_skidsteer(dig_mask)
        print(f"   ✅ Skid steer mask applied, sum: {dig_mask_filtered.sum()}")
        
        if dig_mask_filtered.sum() == 0:
            print("   ⚠️  WARNING: Mask filtered out ALL tiles!")
            
            # Debug why
            action_map_in_cone = state_swapped.world.action_map.map.reshape(-1) * dig_mask
            print(f"   Action map values in cone: {jnp.sum(action_map_in_cone > 0)} dirt tiles")
            
            # Check the specific masking conditions
            dig_mask_action_map = state_swapped.world.action_map.map > 0
            print(f"   Dirt tiles in map: {jnp.sum(dig_mask_action_map)}")
            print(f"   Dirt at skid steer position: {state_swapped.world.action_map.map[4,4]}")
            
            return
        else:
            print(f"   ✅ Mask allows {dig_mask_filtered.sum()} tiles")
            
    except Exception as e:
        print(f"   ❌ Skid steer masking failed: {e}")
        return
        
    print("\n7. Testing volume calculation...")
    try:
        flattened_action_map = state_swapped.world.action_map.map.reshape(-1)
        selected_tiles_sum = flattened_action_map @ dig_mask_filtered
        print(f"   ✅ Selected tiles sum: {selected_tiles_sum}")
        print(f"   Moving dumped dirt: {selected_tiles_sum > 0}")
    except Exception as e:
        print(f"   ❌ Volume calculation failed: {e}")
        return
        
    print("\n8. Testing full DO action...")
    try:
        new_state = state_swapped._handle_do()
        print("   ✅ DO action completed!")
        
        loaded_before = state_swapped.agent.agent_state.loaded[0]
        loaded_after = new_state.agent.agent_state.loaded[0]
        
        print(f"   Loaded before: {loaded_before}")
        print(f"   Loaded after: {loaded_after}")
        
        if loaded_after > loaded_before:
            print("   🎉 SUCCESS: Dirt was lifted!")
        else:
            print("   ❌ ISSUE: No dirt was loaded")
            
    except Exception as e:
        print(f"   ❌ DO action failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    debug_skid_step_by_step()
    print("\n" + "="*50) 