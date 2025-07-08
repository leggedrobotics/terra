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
        # Check initial shovel state
        shovel_before = state_swapped.agent.agent_state.shovel_lifted[0]
        print(f"   Initial shovel state: {'lifted' if shovel_before else 'lowered'}")
        
        new_state = state_swapped._handle_do()
        print("   ✅ DO action completed!")
        
        loaded_before = state_swapped.agent.agent_state.loaded[0]
        loaded_after = new_state.agent.agent_state.loaded[0]
        shovel_after = new_state.agent.agent_state.shovel_lifted[0]
        
        print(f"   Loaded before: {loaded_before}")
        print(f"   Loaded after: {loaded_after}")
        print(f"   Shovel after: {'lifted' if shovel_after else 'lowered'}")
        
        if loaded_after > loaded_before:
            print("   🎉 SUCCESS: Dirt was lifted!")
        elif shovel_after != shovel_before:
            print("   ℹ️  INFO: Shovel state changed (skid steer behavior)")
        else:
            print("   ❌ ISSUE: No dirt was loaded")
            
    except Exception as e:
        print(f"   ❌ DO action failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n9. Testing skid steer forward movement (auto-load)...")
    try:
        # Move forward to trigger auto-loading
        state_before_move = new_state
        
        # Check movement conditions
        is_skid_steer = state_before_move.agent.agent_state.agent_type[0] == 2
        is_loaded = state_before_move.agent.agent_state.loaded[0] > 0
        shovel_lifted = state_before_move.agent.agent_state.shovel_lifted[0] > 0
        
        print(f"   Pre-move conditions:")
        print(f"     Is skid steer: {is_skid_steer}")
        print(f"     Is loaded: {is_loaded}")
        print(f"     Shovel lifted: {shovel_lifted}")
        
        # Check position before move
        pos_before = state_before_move.agent.agent_state.pos_base
        print(f"     Position before: {pos_before}")
        
        state_after_move = state_before_move._handle_move_forward()
        
        # Check if movement actually happened
        pos_after = state_after_move.agent.agent_state.pos_base
        print(f"     Position after: {pos_after}")
        moved = jnp.any(pos_before != pos_after)
        print(f"     Actually moved: {moved}")
        
        loaded_before_move = state_before_move.agent.agent_state.loaded[0]
        loaded_after_move = state_after_move.agent.agent_state.loaded[0]
        
        print(f"   Loaded before move: {loaded_before_move}")
        print(f"   Loaded after move: {loaded_after_move}")
        
        if loaded_after_move > loaded_before_move:
            print("   🎉 SUCCESS: Auto-loading worked!")
        else:
            print("   ❌ ISSUE: Auto-loading failed")
            
            # Debug auto-loading conditions
            post_move_is_skid_steer = state_after_move.agent.agent_state.agent_type[0] == 2
            post_move_shovel_lowered = state_after_move.agent.agent_state.shovel_lifted[0] == 0
            post_move_not_loaded = state_after_move.agent.agent_state.loaded[0] == 0
            
            print(f"   Auto-load debug:")
            print(f"     Post-move is skid steer: {post_move_is_skid_steer}")
            print(f"     Post-move shovel lowered: {post_move_shovel_lowered}")
            print(f"     Post-move not loaded: {post_move_not_loaded}")
            
    except Exception as e:
        print(f"   ❌ Move forward failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n10. Testing correct auto-load sequence...")
    try:
        # Lower shovel first (since auto-loading only works with lowered shovel)
        state_with_lowered_shovel = state_after_move._handle_do()  # Should lower the shovel
        
        shovel_state = state_with_lowered_shovel.agent.agent_state.shovel_lifted[0]
        print(f"   Shovel after second DO: {'lifted' if shovel_state else 'lowered'}")
        
        # Now try forward movement with lowered shovel
        pos_before = state_with_lowered_shovel.agent.agent_state.pos_base
        print(f"   Position before move: {pos_before}")
        
        state_final = state_with_lowered_shovel._handle_move_forward()
        
        pos_after = state_final.agent.agent_state.pos_base
        print(f"   Position after move: {pos_after}")
        moved = jnp.any(pos_before != pos_after)
        print(f"   Actually moved: {moved}")
        
        loaded_before = state_with_lowered_shovel.agent.agent_state.loaded[0]
        loaded_after = state_final.agent.agent_state.loaded[0]
        
        print(f"   Loaded before move: {loaded_before}")
        print(f"   Loaded after move: {loaded_after}")
        
        if loaded_after > loaded_before:
            print("   🎉 SUCCESS: Auto-loading worked with lowered shovel!")
        else:
            print("   ❌ ISSUE: Auto-loading still failed")
            
    except Exception as e:
        print(f"   ❌ Second test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n11. Testing reward structure...")
    try:
        # Test the new reward structure
        old_reward_total = 0.0
        
        # Test "do nothing" (should get 0 existence penalty now)
        do_nothing_reward = state_final._get_reward(state_final, TrackedAction.do_nothing())
        print(f"   Do nothing reward: {do_nothing_reward:.6f}")
        
        # Test movement forward (should be much less negative now)
        forward_reward = state_final._get_reward(state_final, TrackedAction.forward())
        print(f"   Forward movement reward: {forward_reward:.6f}")
        
        # Test DO action (skid steer should get positive rewards)
        do_reward = state_final._get_reward(state_final._handle_do(), TrackedAction.do())
        print(f"   DO action reward: {do_reward:.6f}")
        
        print(f"   ✅ Reward structure test completed")
        print(f"   💡 Movement is no longer heavily penalized!")
        
    except Exception as e:
        print(f"   ❌ Reward structure test failed: {e}")
        
    print("\n🎉 All tests completed!")

def test_integer_soil_collapse():
    print("\n=== TEST: Integer-only Soil Collapse ===")
    # Create a simple 5x5 map with a hole in the center
    map_2d = jnp.array([
        [5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5],
        [5, 5, 0, 5, 5],
        [5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5],
    ], dtype=jnp.int32)
    # Mask: only the center tile is affected
    affected_mask = jnp.zeros((5, 5), dtype=bool).at[2, 2].set(True)

    # Dummy state for method call
    class DummyState:
        def __init__(self):
            self.env_cfg = EnvConfig()
        def _apply_local_soil_mechanics_simplified(self, action_map, affected_mask):
            # Use the real method from State
            return State._apply_local_soil_mechanics_simplified(self, action_map, affected_mask)
    state = DummyState()

    print("Before collapse:")
    print(map_2d)
    print("Affected mask:")
    print(affected_mask.astype(int))

    collapsed = state._apply_local_soil_mechanics_simplified(map_2d, affected_mask)
    print("After collapse:")
    print(collapsed)

if __name__ == "__main__":
    debug_skid_step_by_step()
    test_integer_soil_collapse()
    print("\n" + "="*50) 