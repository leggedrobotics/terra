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
        # For trucks: spawn on non-dumpable tiles (roads) within 10px of border
        if dumpability_map is not None:
            non_dumpable_mask = (dumpability_map == 0).astype(jnp.bool_)
            # Create border mask: within 10px of any edge
            max_border_dist = 8
            y_coords = jnp.arange(map_height)[:, None]  # (H, 1)
            x_coords = jnp.arange(map_width)[None, :]   # (1, W)
            dist_to_top = y_coords
            dist_to_bottom = map_height - 1 - y_coords
            dist_to_left = x_coords
            dist_to_right = map_width - 1 - x_coords
            min_dist_to_edge = jnp.minimum(
                jnp.minimum(dist_to_top, dist_to_bottom),
                jnp.minimum(dist_to_left, dist_to_right)
            )
            border_mask = (min_dist_to_edge <= max_border_dist)
            # Intersect: must be non-dumpable AND within 10px of border
            truck_spawn_allowed_mask = jnp.logical_and(non_dumpable_mask, border_mask)
        else:
            truck_spawn_allowed_mask = full_allowed_mask

        def _prepare_type_array(values, max_len, dtype=jnp.int32, default=0):
            arr = jnp.asarray(values, dtype=dtype)
            if arr.ndim == 0:
                arr = jnp.reshape(arr, (1,))
            pad_len = max_len - arr.shape[0]
            arr = jnp.pad(arr, (0, max(pad_len, 0)), constant_values=default)
            return arr[:max_len]

        agent_types_tensor = _prepare_type_array(agent_types, MAX_AGENTS, dtype=jnp.int32, default=0)
        action_types_tensor = _prepare_type_array(action_types, MAX_AGENTS, dtype=jnp.int32, default=0)
        
        def place_agent(carry, i):
            combined_mask, keys, states = carry
            # Only place agent if i < n_agents
            should_place = i < n_agents
            
            agent_type_val = agent_types[i] if i < len(agent_types) else 0
            action_type_val = action_types[i] if i < len(action_types) else 0
            is_truck = (agent_type_val == 1)
            allowed_mask = jax.lax.cond(
                is_truck,
                lambda: truck_spawn_allowed_mask,
                lambda: full_allowed_mask,
            )
            # For trucks: accept if ANY part is on non-dumpable tiles
            require_all_allowed = jnp.logical_not(is_truck)
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
                require_all_allowed,
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
            is_truck_flag = (agent_type_val == 1)
            
            # Trucks spawn on non-dumpable tiles (roads), others spawn anywhere
            allowed_mask = jax.lax.cond(
                is_truck_flag,
                lambda _: truck_spawn_allowed_mask,
                lambda _: full_allowed_mask,
                operand=None,
            )
            # For trucks: accept if ANY part is on non-dumpable tiles
            # For others: require ALL parts to be on allowed tiles (default)
            require_all_allowed = jnp.logical_not(is_truck_flag)
            pos_i, angle_i, per_agent_keys[i] = _get_random_init_state(
                per_agent_keys[i],
                env_cfg,
                max_traversable_x,
                max_traversable_y,
                combined_mask,
                action_map,
                width,
                height,
                allowed_mask,
                require_all_allowed,
            )
            angle_i = angle_i.astype(IntLowDim)
            
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
    require_all_allowed: Array = jnp.bool_(True),
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
            
            # Check allowed_mask: if require_all_allowed=True, reject if ANY part is not allowed
            # if require_all_allowed=False, accept if ANY part is allowed
            def _check_all_parts_allowed():
                return jnp.any(
                    jnp.logical_and(polygon_mask, jnp.logical_not(allowed_mask))
                )
            
            def _check_any_part_allowed():
                # Accept if ANY part is on allowed tiles
                allowed_satisfied = jnp.any(
                    jnp.logical_and(polygon_mask, allowed_mask)
                )
                return jnp.logical_not(allowed_satisfied)
            
            allowed_violation = jax.lax.cond(
                require_all_allowed,
                _check_all_parts_allowed,
                _check_any_part_allowed,
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




def _prefix_true_length(arr: Array) -> IntMap:
    inv = jnp.logical_not(arr)
    first_false = jnp.argmax(inv)
    has_false = jnp.any(inv)
    length = jnp.where(has_false, first_false, arr.shape[0])
    return length.astype(IntMap)
