import numpy as np


def binary_dilate_3x3(mask: np.ndarray) -> np.ndarray:
    """Binary dilation with a 3x3 ones kernel using only numpy (no scipy).
    Equivalent to OR of mask with its 8-neighborhood.
    """
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.zeros_like(mask, dtype=bool)
            ys = slice(max(0, dy), min(h, h + dy))
            xs = slice(max(0, dx), min(w, w + dx))
            ydst = slice(max(0, -dy), min(h, h - dy))
            xdst = slice(max(0, -dx), min(w, w - dx))
            shifted[ydst, xdst] = mask[ys, xs]
            out |= shifted
    return out


def _check_done_dump(action_map: np.ndarray, target_map: np.ndarray) -> bool:
    designated_dump_zones = target_map > 0
    expanded_dump_zones = binary_dilate_3x3(designated_dump_zones)
    dirt_outside_expanded = (action_map > 0) & (~expanded_dump_zones)
    return not np.any(dirt_outside_expanded)


def _check_done_dig(action_map: np.ndarray, target_map: np.ndarray) -> bool:
    dig_requirements = np.where(target_map < 0, target_map, 0)
    actual_digs = np.where(target_map < 0, action_map, 0)
    return np.all(actual_digs <= dig_requirements)


def _check_relocation_done(action_map: np.ndarray, target_map: np.ndarray, agent_active: np.ndarray, loaded: np.ndarray) -> bool:
    # Early exit: if any active agent is loaded, task cannot be complete
    agents_loaded = np.any(agent_active & (loaded > 0))

    if agents_loaded:
        return False

    designated_dump_zones = target_map > 0
    dump_zones_with_buffer = binary_dilate_3x3(designated_dump_zones)
    total_dirt = np.sum(action_map > 0)
    if total_dirt == 0:
        return False
    dirt_outside_buffered = (action_map > 0) & (~dump_zones_with_buffer)
    return not np.any(dirt_outside_buffered)


def is_done_task(action_map: np.ndarray,
                 target_map: np.ndarray,
                 agent_types: np.ndarray,
                 agent_active: np.ndarray,
                 loaded: np.ndarray) -> bool:
    """
    Standalone copy of the task completion logic (numpy version).

    Inputs:
      - action_map: HxW float
      - target_map: HxW float
      - agent_types: length-4 array of ints (0=excavator, 1=truck, 2=skidsteer)
      - agent_active: length-4 bool array
      - loaded: length-4 float array (carried dirt per agent)
    """
    has_dump_requirements = np.any(target_map > 0)
    has_dig_requirements = np.any(target_map < 0)

    has_skidsteer_agent = np.any(agent_active & (agent_types == 2))
    has_truck_agent = np.any(agent_active & (agent_types == 1))
    has_transport_agent = has_skidsteer_agent or has_truck_agent

    # Relocation: dump only (no dig). Does not require transport agent.
    is_relocation_task = has_dump_requirements and (not has_dig_requirements)
    # Cooperative: both dig and dump present and at least one transport agent.
    is_cooperative_task = (has_dig_requirements and has_dump_requirements and has_transport_agent)

    def traditional_logic() -> bool:
        no_requirements = (not has_dig_requirements) and (not has_dump_requirements)
        if no_requirements:
            return False
        # If there are dig requirements, they must be met; else True when no dig reqs.
        if np.all(target_map >= 0):
            return True
        return _check_done_dig(action_map, target_map)

    def relocation_logic() -> bool:
        return _check_relocation_done(action_map, target_map, agent_active, loaded)

    def cooperative_logic() -> bool:
        done_dig = _check_done_dig(action_map, target_map)
        done_dump = _check_done_dump(action_map, target_map)
        cooperative_complete = (done_dig and done_dump)
        # all active agents unloaded
        all_unloaded = np.all((~agent_active) | (loaded == 0))
        return cooperative_complete and all_unloaded

    if is_relocation_task:
        task_requirements_met = relocation_logic()
    elif is_cooperative_task:
        task_requirements_met = cooperative_logic()
    else:
        task_requirements_met = traditional_logic()

    all_unloaded_dyn = np.all((~agent_active) | (loaded == 0))
    done_task = task_requirements_met and all_unloaded_dyn
    return done_task


def demo_cases():
    H, W = 8, 8
    # Build a simple map with one dump zone (positive) and one dig zone (negative)
    target = np.zeros((H, W), dtype=np.float32)
    target[1:3, 1:3] = -1.0  # dig requirement
    target[5:7, 5:7] = 1.0   # dump zone

    # Initial dirt placed in dig area only (not yet in dump zone)
    action = np.zeros((H, W), dtype=np.float32)
    action[1:3, 1:3] = -1.0  # dug equals target initially (so dig is already satisfied)
    # But place some dirt elsewhere to simulate spoil not in dump zone
    action[2:4, 2:4] = 1.0   # dirt outside dump zone

    # Agents: excavator (0) and truck (1)
    agent_types = np.array([0, 1, 0, 0], dtype=np.int32)
    agent_active = np.array([True, True, False, False], dtype=bool)
    loaded = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    print("Cooperative expected False at reset (dirt outside dump, both unloaded):",
          is_done_task(action, target, agent_types, agent_active, loaded))

    # Move dirt into dump zone to complete dump condition
    action[:, :] = 0.0
    action[5:7, 5:7] = 1.0
    print("Cooperative expected True when dump ok, dig ok, all unloaded:",
          is_done_task(action, target, agent_types, agent_active, loaded))

    # Relocation-only case: remove dig requirements
    target2 = np.zeros_like(target)
    target2[5:7, 5:7] = 1.0
    action2 = np.zeros_like(action)
    action2[3:4, 3:4] = 1.0  # dirt outside dump -> not done
    print("Relocation-only expected False at reset:",
          is_done_task(action2, target2, agent_types, agent_active, loaded))
    action2[:, :] = 0
    action2[5:7, 5:7] = 1.0  # all in dump -> done
    print("Relocation-only expected True when all dirt in dump:",
          is_done_task(action2, target2, agent_types, agent_active, loaded))

    # No requirements: should not be done at reset
    target3 = np.zeros_like(target)
    action3 = np.zeros_like(action)
    print("No requirements expected False:",
          is_done_task(action3, target3, agent_types, agent_active, loaded))


if __name__ == "__main__":
    demo_cases()



