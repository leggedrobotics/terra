from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.config import EnvConfig
from terra.settings import IntLowDim
from terra.settings import IntMap
from terra.utils import compute_polygon_mask
from terra.utils import get_agent_corners


class AgentState(NamedTuple):
    """
    Clarifications on the agent state representation.

    angle_base:
    Orientations of the agent are an integer between 0 and 3 (included),
    where 0 means that it is aligned with the x axis, and for every positive
    increment, 90 degrees are added in the direction of the arrow going from
    the x axis to the y axes (anti-clockwise).
    
    shovel_lifted:
    For skid steer only: 0 = shovel lowered (can auto-load), 1 = shovel lifted (can dump)
    """

    pos_base: IntMap
    angle_base: IntLowDim
    angle_cabin: IntLowDim
    wheel_angle: IntLowDim
    loaded: IntLowDim
    agent_type: IntLowDim  # 0=excavator, 1=truck, 2=skidsteer
    action_type: IntLowDim  # 0=tracked, 1=wheeled (movement mechanism)
    shovel_lifted: IntLowDim  # 0=lowered, 1=lifted (for skid steer only)
    # Per-agent baseline potential cached at start of carry (0 -> >0)
    carry_baseline_potential: jnp.float32 = jnp.float32(0.0)
    # Per-agent potential immediately after lifting (post-dig/auto-load)
    carry_potential_after_lift: jnp.float32 = jnp.float32(0.0)


class Agent(NamedTuple):
    """
    Defines the state and type of the agent.
    """

    width: int
    height: int

    moving_dumped_dirt: bool

    # Variable number of agents with fixed-size storage (JAX-friendly, static max size)
    agent_states: tuple[AgentState, ...] | None = None  # Fixed-size container (e.g., 8)
    agent_active: jnp.ndarray | None = None  # shape [max_agents], 1 for active, 0 for inactive
    num_agents: int = 2  # actual number of active agents (defaults to current behavior)
    current_agent: int = 0  # index of currently acting agent (alternating training)

    @staticmethod
    def new(
        key: jax.random.PRNGKey,
        env_cfg: EnvConfig,
        max_traversable_x: int,
        max_traversable_y: int,
        padding_mask: Array,
        action_map: Array,
        dumpability_map: Array | None = None,
        target_map: Array | None = None,
        agent_types: tuple = (0, 0),  # variable length; 0=excavator, 1=truck, 2=skidsteer
        action_types: tuple = (0, 0),  # action types; 0=tracked, 1=wheeled
    ) -> tuple["Agent", jax.random.PRNGKey]:
        # Determine number of agents to initialize (clip to MAX_AGENTS)
        MAX_AGENTS = 4
        n_agents = min(len(agent_types), MAX_AGENTS)  # Use Python min for static value

        # Split keys: one per agent + one for choosing current agent
        # Use MAX_AGENTS + 1 to avoid dynamic shapes
        keys = jax.random.split(key, MAX_AGENTS + 1)
        key_current_sel = keys[0]
        per_agent_keys = list(keys[1:MAX_AGENTS + 1])

        width = env_cfg.agent.width
        height = env_cfg.agent.height
        moving_dumped_dirt = False

        # Use JAX-compatible loop for agent placement
        map_width, map_height = padding_mask.shape
        full_allowed_mask = jnp.ones_like(padding_mask, dtype=jnp.bool_)
        if dumpability_map is not None and target_map is not None:
            truck_spawn_allowed_mask = jnp.logical_or(
                dumpability_map == 0,
                target_map > 0,
            ).astype(jnp.bool_)
        else:
            truck_spawn_allowed_mask = full_allowed_mask
        truck_spawn_positions, truck_spawn_angles, truck_spawn_valid = _compute_truck_spawn_slots(dumpability_map)

        def _prepare_type_array(values, max_len, dtype=jnp.int32, default=0):
            arr = jnp.asarray(values, dtype=dtype)
            if arr.ndim == 0:
                arr = jnp.reshape(arr, (1,))
            pad_len = max_len - arr.shape[0]
            arr = jnp.pad(arr, (0, max(pad_len, 0)), constant_values=default)
            return arr[:max_len]

        agent_types_tensor = _prepare_type_array(agent_types, MAX_AGENTS, dtype=jnp.int32, default=0)
        action_types_tensor = _prepare_type_array(action_types, MAX_AGENTS, dtype=jnp.int32, default=0)

        is_truck_flags = (agent_types_tensor == 1)
        truck_slot_indices = jnp.where(
            is_truck_flags,
            jnp.cumsum(is_truck_flags.astype(IntLowDim)) - 1,
            jnp.full_like(agent_types_tensor, -1, dtype=IntLowDim),
        )
        
        def place_agent(carry, i):
            combined_mask, keys, states = carry
            # Only place agent if i < n_agents
            should_place = i < n_agents
            
            agent_type_val = agent_types[i] if i < len(agent_types) else 0
            action_type_val = action_types[i] if i < len(action_types) else 0
            allowed_mask = jax.lax.cond(
                agent_type_val == 1,
                lambda: truck_spawn_allowed_mask,
                lambda: full_allowed_mask,
            )
            pos_i, angle_i, new_key = _get_random_init_state(
                keys[i],
                env_cfg,
                max_traversable_x,
                max_traversable_y,
                combined_mask,
                action_map,
                width,
                height,
                allowed_mask,
            )

            # Create agent state
            st_i = AgentState(
                pos_base=pos_i,
                angle_base=angle_i,
                angle_cabin=jnp.full((1,), 0, dtype=jnp.int8),
                wheel_angle=jnp.full((1,), 0, dtype=jnp.int8),
                loaded=jnp.full((1,), 0, dtype=jnp.int8),
                agent_type=jnp.full((1,), agent_type_val, dtype=jnp.int8),
                action_type=jnp.full((1,), action_type_val, dtype=jnp.int8),
                shovel_lifted=jnp.full((1,), 0, dtype=jnp.int8),
                carry_baseline_potential=jnp.float32(0.0),
                carry_potential_after_lift=jnp.float32(0.0),
            )
            
            # Update mask only if we placed an agent
            agent_corners = get_agent_corners(
                pos_i, angle_i, width, height, env_cfg.agent.angles_base
            )
            agent_mask = compute_polygon_mask(agent_corners, map_width, map_height)
            new_combined_mask = jnp.where(should_place, 
                                        jnp.logical_or(combined_mask, agent_mask), 
                                        combined_mask)
            
            # Update keys array
            new_keys = keys.at[i].set(new_key)
            
            # Update states array
            new_states = states.at[i].set(st_i)
            
            return (new_combined_mask, new_keys, new_states), None
        
        # Initialize arrays
        initial_mask = (padding_mask == 1)
        initial_keys = jnp.array(per_agent_keys)
        
        # Create initial dummy state for padding
        dummy_state = AgentState(
            pos_base=jnp.array([0, 0], dtype=IntMap),
            angle_base=jnp.array([0], dtype=IntLowDim),
            angle_cabin=jnp.array([0], dtype=IntLowDim),
            wheel_angle=jnp.array([0], dtype=IntLowDim),
            loaded=jnp.array([0], dtype=IntLowDim),
            agent_type=jnp.array([0], dtype=IntLowDim),
            action_type=jnp.array([0], dtype=IntLowDim),  # Default to tracked
            shovel_lifted=jnp.array([0], dtype=IntLowDim),
        )
        
        # Initialize with dummy states that will be replaced
        initial_states = [dummy_state] * MAX_AGENTS
        
        # Use a simpler approach - just place agents one by one with regular Python loop
        # since the scan approach is too complex for NamedTuple handling
        combined_mask = initial_mask
        built_states = []
        
        for i in range(MAX_AGENTS):
            if i >= n_agents:
                # Pad with dummy state
                built_states.append(dummy_state)
                continue
                
            agent_type_val = agent_types_tensor[i]
            action_type_val = action_types_tensor[i]
            is_truck_flag = is_truck_flags[i]
            allowed_mask = jax.lax.cond(
                is_truck_flag,
                lambda _: truck_spawn_allowed_mask,
                lambda _: full_allowed_mask,
                operand=None,
            )

            slot_idx = truck_slot_indices[i]

            def _slot_lookup(array, idx):
                zero_value = jnp.zeros_like(array[0])

                def _take(valid_idx):
                    valid_idx = valid_idx.astype(jnp.int32)
                    slice_ = jax.lax.dynamic_slice_in_dim(array, valid_idx, 1, axis=0)
                    return jnp.squeeze(slice_, axis=0)

                return jax.lax.cond(
                    jnp.logical_and(idx >= 0, idx < array.shape[0]),
                    _take,
                    lambda _: zero_value,
                    idx,
                )

            def _slot_valid(idx):
                def _take(valid_idx):
                    valid_idx = valid_idx.astype(jnp.int32)
                    return jax.lax.dynamic_slice_in_dim(
                        truck_spawn_valid.astype(jnp.bool_), valid_idx, 1, axis=0
                    )[0]

                return jax.lax.cond(
                    jnp.logical_and(idx >= 0, idx < truck_spawn_valid.shape[0]),
                    _take,
                    lambda _: jnp.bool_(False),
                    idx,
                )

            slot_valid = _slot_valid(slot_idx)
            use_slot = jnp.logical_and(is_truck_flag, slot_valid)

            def _use_slot(_):
                slot_pos = _slot_lookup(truck_spawn_positions, slot_idx)
                slot_angle = _slot_lookup(truck_spawn_angles, slot_idx)
                return slot_pos.astype(IntMap), slot_angle.astype(IntLowDim), per_agent_keys[i]

            def _use_random(_):
                pos, ang, key_out = _get_random_init_state(
                    per_agent_keys[i],
                    env_cfg,
                    max_traversable_x,
                    max_traversable_y,
                    combined_mask,
                    action_map,
                    width,
                    height,
                    allowed_mask,
                )
                return pos, ang.astype(IntLowDim), key_out

            pos_i, angle_i, per_agent_keys[i] = jax.lax.cond(
                use_slot,
                _use_slot,
                _use_random,
                operand=None,
            )
            
            st_i = AgentState(
                pos_base=pos_i.astype(IntMap),
                angle_base=angle_i.astype(IntLowDim),
                angle_cabin=jnp.full((1,), 0, dtype=IntLowDim),
                wheel_angle=jnp.full((1,), 0, dtype=IntLowDim),
                loaded=jnp.full((1,), 0, dtype=IntLowDim),
                agent_type=jnp.full((1,), agent_type_val, dtype=IntLowDim),
                action_type=jnp.full((1,), action_type_val, dtype=IntLowDim),
                shovel_lifted=jnp.full((1,), 0, dtype=IntLowDim),
                carry_baseline_potential=jnp.float32(0.0),
                carry_potential_after_lift=jnp.float32(0.0),
            )
            built_states.append(st_i)

            # Update combined mask to avoid placing next agent on top of this one
            agent_corners = get_agent_corners(
                pos_i, angle_i, width, height, env_cfg.agent.angles_base
            )
            agent_mask = compute_polygon_mask(agent_corners, map_width, map_height)
            combined_mask = jnp.logical_or(combined_mask, agent_mask)

        # built_states is already MAX_AGENTS length

        agent_states_tuple = tuple(built_states)
        agent_active = jnp.concatenate([
            jnp.ones((n_agents,), dtype=jnp.int8),
            jnp.zeros((MAX_AGENTS - n_agents,), dtype=jnp.int8)
        ])

        # Randomize starting current agent among active agents
        current_agent = jax.random.randint(key_current_sel, (), 0, n_agents)

        return Agent(
            agent_states=agent_states_tuple,
            agent_active=agent_active,
            num_agents=n_agents,
            current_agent=current_agent,
            width=width,
            height=height,
            moving_dumped_dirt=moving_dumped_dirt,
        ), per_agent_keys[-1]


def _get_top_left_init_state(key: jax.random.PRNGKey, env_cfg: EnvConfig):
    max_center_coord = jnp.ceil(
        jnp.max(
            jnp.array([env_cfg.agent.width // 2 - 1, env_cfg.agent.height // 2 - 1])
        )
    ).astype(IntMap)
    pos_base = IntMap(jnp.array([max_center_coord, max_center_coord]))
    theta = jnp.full((1,), fill_value=0, dtype=IntMap)
    return pos_base, theta, key


def _get_random_init_state(
    key: jax.random.PRNGKey,
    env_cfg: EnvConfig,
    max_traversable_x: int,
    max_traversable_y: int,
    padding_mask: Array,
    action_map: Array,
    agent_width: int,
    agent_height: int,
    allowed_mask: Array,
):
    def _get_random_agent_state(carry):
        key, padding_mask, pos_base, angle_base = carry
        max_center_coord = jnp.ceil(
            jnp.max(
                jnp.array([env_cfg.agent.width / 2 - 1, env_cfg.agent.height / 2 - 1])
            )
        ).astype(IntMap)
        key, subkey_x, subkey_y, subkey_angle = jax.random.split(key, 4)

        max_w = jnp.minimum(max_traversable_x, env_cfg.maps.edge_length_px)
        max_h = jnp.minimum(max_traversable_y, env_cfg.maps.edge_length_px)

        x = jax.random.randint(
            subkey_x,
            (1,),
            minval=max_center_coord,
            maxval=max_w - max_center_coord,
        )
        y = jax.random.randint(
            subkey_y,
            (1,),
            minval=max_center_coord,
            maxval=max_h - max_center_coord,
        )
        pos_base = IntMap(jnp.concatenate((x, y)))
        angle_base = jax.random.randint(
            subkey_angle, (1,), 0, env_cfg.agent.angles_base, dtype=IntMap
        )
        return key, padding_mask, pos_base, angle_base

    def _check_agent_obstacles_intersection(carry):
        key, padding_mask, pos_base, angle_base = carry
        map_width = padding_mask.shape[0]
        map_height = padding_mask.shape[1]

        def _check_intersection():
            """
            Checks that the agent does not spawn where an obstacle is (or else it will get stuck forever).
            The check takes the four agent corners and checks that in the tiles included
            within the corners there is no obstacle-encoded tile.
            The padding mask is the map encoding obstacles (1 for obstacle and 0 for no obstacle).
            """
            agent_corners_xy = get_agent_corners(
                pos_base, angle_base, agent_width, agent_height, env_cfg.agent.angles_base
            )
            polygon_mask = compute_polygon_mask(
                agent_corners_xy, map_width, map_height
            )

            obstacle_inside = jnp.any(jnp.logical_and(polygon_mask, padding_mask == 1))
            action_illegal = jnp.any(jnp.logical_and(polygon_mask, action_map != 0))
            allowed_violation = jnp.any(
                jnp.logical_and(polygon_mask, jnp.logical_not(allowed_mask))
            )
            return obstacle_inside | action_illegal | allowed_violation

        keep_searching = jax.lax.cond(
            jnp.any(pos_base < 0) | jnp.any(angle_base < 0),
            lambda: True,
            _check_intersection,
        )
        return keep_searching

    key, padding_mask, pos_base, angle_base = jax.lax.while_loop(
        _check_agent_obstacles_intersection,
        _get_random_agent_state,
        (
            key,
            padding_mask,
            jnp.array([-1, -1], dtype=IntMap),
            jnp.full((1,), -1, dtype=IntMap),
        ),
    )

    return pos_base, angle_base, key


def _compute_truck_spawn_slots(dumpability_map: Array | None):
    if dumpability_map is None:
        zero_positions = jnp.zeros((4, 2), dtype=IntMap)
        zero_angles = jnp.zeros((4, 1), dtype=IntLowDim)
        zero_valid = jnp.zeros((4,), dtype=jnp.bool_)
        return zero_positions, zero_angles, zero_valid

    road_mask = (dumpability_map == 0)
    height, width = road_mask.shape

    row_full = jnp.all(road_mask, axis=1)
    col_full = jnp.all(road_mask, axis=0)

    top_width = _prefix_true_length(row_full)
    bottom_width = _prefix_true_length(jnp.flip(row_full, axis=0))
    left_width = _prefix_true_length(col_full)
    right_width = _prefix_true_length(jnp.flip(col_full, axis=0))

    width_half = jnp.int32(width // 2)
    height_half = jnp.int32(height // 2)

    top_center = jnp.array([width_half, top_width // 2], dtype=IntMap)
    bottom_start = height - bottom_width
    bottom_center = jnp.array(
        [width_half, bottom_start + bottom_width // 2],
        dtype=IntMap,
    )
    left_center = jnp.array([left_width // 2, height_half], dtype=IntMap)
    right_start = width - right_width
    right_center = jnp.array(
        [right_start + right_width // 2, height_half],
        dtype=IntMap,
    )

    positions = jnp.stack(
        [top_center, bottom_center, left_center, right_center],
        axis=0,
    )

    angles = jnp.stack(
        [
            jnp.array([0], dtype=IntLowDim),
            jnp.array([0], dtype=IntLowDim),
            jnp.array([1], dtype=IntLowDim),
            jnp.array([1], dtype=IntLowDim),
        ],
        axis=0,
    )

    valid = jnp.array(
        [
            top_width > 0,
            bottom_width > 0,
            left_width > 0,
            right_width > 0,
        ],
        dtype=jnp.bool_,
    )

    return positions, angles, valid


def _prefix_true_length(arr: Array) -> IntMap:
    inv = jnp.logical_not(arr)
    first_false = jnp.argmax(inv)
    has_false = jnp.any(inv)
    length = jnp.where(has_false, first_false, arr.shape[0])
    return length.astype(IntMap)
