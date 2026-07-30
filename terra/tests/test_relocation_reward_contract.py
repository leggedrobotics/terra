"""Deterministic action-path evidence for the agent-neutral reward contract.

The tests exercise fresh excavation, rehandling, transport, and handoff paths
without patching reward flags or carry credit.

The traces use Terra's real dig, dump, and transfer transition paths and the
same action-specific reward handlers selected by ``State._get_reward``.  They
intentionally avoid compiling the monolithic all-action switch and unrelated
terminal, trench, and logging branches.  Pose changes between workspaces are
direct test setup; load, soil, material flags, and carry potentials are
produced only by Terra transitions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.actions import TrackedAction
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.env import TerraEnvBatch
from terra.state import State

SEED = 20260730
SHAPE = (64, 64)
CENTER = np.array([32, 32], dtype=np.int16)


def _env_config(agent_types: tuple[int, ...] = (0,)) -> EnvConfig:
    """Build the concrete 64x64 agent-neutral reward environment."""
    batch_env = object.__new__(TerraEnvBatch)
    batch_env.batch_cfg = BatchConfig()._replace(
        maps_dims=MapsDimsConfig(maps_edge_length=SHAPE[0])
    )
    base = EnvConfig()
    batched = base._replace(
        agent=base.agent._replace(dig_depth=jnp.ones((1,), dtype=jnp.int32))
    )
    updated = batch_env.update_env_cfgs(batched)
    return base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(
            edge_length_px=int(np.asarray(updated.maps.edge_length_px)[0])
        ),
        max_steps_in_episode=450,
        agent_types=agent_types,
        action_types=tuple(0 for _ in agent_types),
        foundation_dump_min_free_fraction=0.0,
        relocation_progress_mult=1.5,
    )


def _state(
    target: np.ndarray,
    *,
    action: np.ndarray | None = None,
    distance: np.ndarray | None = None,
    env_cfg: EnvConfig | None = None,
) -> State:
    if action is None:
        action = np.zeros(SHAPE, dtype=np.int8)
    if distance is None:
        distance = np.ones(SHAPE, dtype=np.float32)
        distance[target > 0] = 0.0
    cfg = env_cfg if env_cfg is not None else _env_config()
    state = State.new(
        jax.random.PRNGKey(SEED),
        cfg,
        target,
        np.zeros(SHAPE, dtype=np.int8),
        -97.0 * np.ones((3, 3), dtype=np.float32),
        np.int32(-1),
        -97.0 * np.ones((SHAPE[0], 3), dtype=np.float32),
        np.int32(-1),
        np.ones(SHAPE, dtype=np.bool_),
        action,
        distance_map_override=distance,
    )
    state = state._replace(agent=state.agent._replace(current_agent=jnp.int32(0)))
    return _set_pose(state, 0, CENTER, cabin=0)


def _set_pose(
    state: State,
    index: int,
    position: np.ndarray,
    *,
    cabin: int | None = None,
) -> State:
    agent_state = state.agent.agent_states[index]
    replacements = {
        "pos_base": jnp.asarray(position, dtype=jnp.int16),
        "angle_base": jnp.array([0], dtype=jnp.int8),
    }
    if cabin is not None:
        replacements["angle_cabin"] = jnp.array([cabin], dtype=jnp.int8)
    return state._set_agent_state_at(index, agent_state._replace(**replacements))


def _set_current_agent(state: State, index: int) -> State:
    return state._replace(agent=state.agent._replace(current_agent=jnp.int32(index)))


def _cone(state: State, agent_index: int, cabin: int = 0) -> np.ndarray:
    posed = _set_current_agent(state, agent_index)
    posed = _set_pose(
        posed,
        agent_index,
        np.asarray(posed.agent.agent_states[agent_index].pos_base),
        cabin=cabin,
    )
    return np.asarray(posed._build_dig_dump_cone()).reshape(SHAPE)


def _mass(state: State) -> int:
    world_mass = np.asarray(state.world.action_map.map, dtype=np.int32).sum()
    carried_mass = sum(
        int(np.asarray(state.agent.agent_states[index].loaded)[0])
        for index in range(int(np.asarray(state.agent.num_agents)))
    )
    return int(world_mass + carried_mass)


def _completion(state: State) -> float:
    completion = state._get_task_completion(
        state.world.action_map.map,
        state.world.target_map.map,
    )
    return float(np.asarray(completion["absolute_completion"]))


def _potential(state: State) -> float:
    return float(
        np.asarray(state._compute_relocation_potential(state.world.action_map.map))
    )


def _step_trace(state: State, action: TrackedAction) -> tuple[State, dict[str, float]]:
    """Run one real action and expose the current contract's reward accounting."""
    acting_index = int(np.asarray(state.agent.current_agent))
    actor_before = state.agent.agent_states[acting_index]
    actor_type = int(np.asarray(actor_before.agent_type)[0])
    action_index = int(np.asarray(action.action).reshape(-1)[0])
    if action_index == 0:
        transitioned = state._handle_move_forward()
    elif action_index == 6:
        is_loaded = int(np.asarray(actor_before.loaded)[0]) > 0
        if actor_type == 0 and not is_loaded:
            transitioned = state._handle_dig()
        elif actor_type == 0:
            transferred = state._try_truck_transfer_on_excavator_dump()
            transfer_happened = int(
                np.asarray(transferred.agent.agent_states[acting_index].loaded)[0]
            ) < int(np.asarray(actor_before.loaded)[0])
            transitioned = transferred if transfer_happened else state._handle_dump()
        elif actor_type == 1 and is_loaded:
            transitioned = state._handle_dump()
        elif actor_type == 2:
            transitioned = state._handle_do()
        else:
            raise ValueError(
                f"Unsupported DO evidence path: type={actor_type}, "
                f"loaded={is_loaded}"
            )
    elif action_index == 7:
        transitioned = state._do_nothing()
    else:
        raise ValueError(f"Unsupported evidence action: {action_index}")
    next_state = transitioned._swap()._replace(env_steps=state.env_steps + 1)

    action_raw = 0.0
    if action_index == 6:
        if actor_type == 2:
            action_raw = float(
                np.asarray(
                    state._handle_rewards_skid_steer_do(
                        next_state,
                        action.action,
                    )
                )
            )
        elif int(np.asarray(actor_before.loaded)[0]) > 0:
            action_raw = float(
                np.asarray(state._handle_rewards_dump(next_state, action.action))
            )
        else:
            action_raw = float(
                np.asarray(state._handle_rewards_dig(next_state, action.action))
            )
    elif action_index == 0 and actor_type == 2:
        action_raw = float(
            np.asarray(state._get_rewards_skidsteer(next_state, action.action))
        )

    actor_after = next_state.agent.agent_states[acting_index]
    fresh_target_progress = float(
        np.asarray(
            state._get_action_map_dig_progress(
                state.world.action_map.map,
                next_state.world.action_map.map,
                state.world.target_map.map,
            )
        )
    )
    world_changed = bool(
        np.any(
            np.asarray(state.world.action_map.map)
            != np.asarray(next_state.world.action_map.map)
        )
    )
    realized_relocation_progress = 0.0
    realized_relocation_reward = 0.0
    if int(np.asarray(actor_before.loaded)[0]) > 0 and world_changed:
        realized_relocation_progress = float(
            np.asarray(state._get_relocation_progress(next_state))
        )
        realized_relocation_reward = action_raw
    # Calling the monolithic `_get_reward` here compiles termination, terminal,
    # trench, and logging branches that are unrelated to this deterministic
    # contract test.  For the nonterminal cycle below, the exact step total is
    # the action term plus the configured existence term.
    nonterminal_normalized = (
        action_raw + float(state.env_cfg.rewards.existence)
    ) / float(state.env_cfg.rewards.normalizer)

    return next_state, {
        "action_raw": action_raw,
        "nonterminal_normalized": nonterminal_normalized,
        "extraction_reward": float(fresh_target_progress > 0),
        "relocation_progress": realized_relocation_progress,
        "relocation_reward": realized_relocation_reward,
        "dig_progress": float(
            np.asarray(
                state._get_action_map_dig_progress(
                    state.world.action_map.map,
                    next_state.world.action_map.map,
                    state.world.target_map.map,
                )
            )
        ),
        "dump_progress": float(
            np.asarray(
                state._get_action_map_dump_progress(
                    state.world.action_map.map,
                    next_state.world.action_map.map,
                    state.world.target_map.map,
                )
            )
        ),
        "load_before": float(np.asarray(actor_before.loaded)[0]),
        "load_after": float(np.asarray(actor_after.loaded)[0]),
        "mass_before": float(_mass(state)),
        "mass_after": float(_mass(next_state)),
        "potential_before": _potential(state),
        "potential_after": _potential(next_state),
        "completion_before": _completion(state),
        "completion_after": _completion(next_state),
        "world_changed": float(world_changed),
    }


def _foundation_geometry() -> tuple[np.ndarray, np.ndarray]:
    probe = _state(np.zeros(SHAPE, dtype=np.int8))
    cone_dig = _cone(probe, 0, cabin=0)
    cone_stage = _cone(probe, 0, cabin=3)
    cone_dump = _cone(probe, 0, cabin=6)

    dig_candidates = np.argwhere(cone_dig & ~cone_stage & ~cone_dump)
    dump_candidates = np.argwhere(cone_dump & ~cone_stage & ~cone_dig)
    if len(dig_candidates) < 8 or len(dump_candidates) < 24:
        raise AssertionError("Expected disjoint dig, stage, and dump workspaces.")

    target = np.zeros(SHAPE, dtype=np.int8)
    target[tuple(dig_candidates[:8].T)] = -1
    target[tuple(dump_candidates[:24].T)] = 1
    distance = np.ones(SHAPE, dtype=np.float32)
    distance[target > 0] = 0.0
    return target, distance


def test_fresh_dig_and_correct_dump_use_real_do_actions():
    target, distance = _foundation_geometry()
    state = _state(target, distance=distance)
    initial_mass = _mass(state)

    after_dig, dig = _step_trace(state, TrackedAction.do())
    assert dig["dig_progress"] > 0
    assert dig["action_raw"] == pytest.approx(1.0)
    assert dig["load_before"] == 0
    assert dig["load_after"] > 0
    assert dig["mass_after"] == initial_mass
    assert dig["extraction_reward"] == 1.0
    assert float(after_dig._get_current_agent_state().carry_relocation_credit) > 0

    ready_to_dump = _set_pose(after_dig, 0, CENTER, cabin=6)
    after_dump, dump = _step_trace(ready_to_dump, TrackedAction.do())
    assert dump["dump_progress"] > 0
    assert dump["action_raw"] > 0
    assert dump["relocation_progress"] > 0
    assert dump["relocation_reward"] > 0
    assert dump["load_before"] > 0
    assert dump["load_after"] == 0
    assert dump["completion_after"] > dump["completion_before"]
    assert dump["mass_after"] == initial_mass
    assert _mass(after_dump) == initial_mass
    assert float(after_dump._get_current_agent_state().carry_relocation_credit) == 0


def test_no_progress_redig_cycle_is_zero_before_step_costs():
    target, distance = _foundation_geometry()
    state = _state(target, distance=distance)

    after_dig, _ = _step_trace(state, TrackedAction.do())
    stage_pose = _set_pose(after_dig, 0, CENTER, cabin=3)
    staged, stage_dump = _step_trace(stage_pose, TrackedAction.do())
    assert stage_dump["dump_progress"] == 0
    assert stage_dump["load_after"] == 0

    potential_before_cycle = _potential(staged)
    completion_before_cycle = _completion(staged)
    mass_before_cycle = _mass(staged)

    lifted, redig = _step_trace(staged, TrackedAction.do())
    assert redig["dig_progress"] == 0
    assert redig["extraction_reward"] == 0
    assert redig["action_raw"] == pytest.approx(0.0)
    assert float(lifted._get_current_agent_state().carry_relocation_credit) > 0

    returned, redump = _step_trace(lifted, TrackedAction.do())
    assert redump["dump_progress"] == 0
    assert redump["action_raw"] == pytest.approx(0.0)
    assert redig["action_raw"] + redump["action_raw"] == pytest.approx(0.0)
    assert redig["nonterminal_normalized"] + redump["nonterminal_normalized"] < 0
    assert _potential(returned) == pytest.approx(potential_before_cycle)
    assert _completion(returned) == pytest.approx(completion_before_cycle)
    assert _mass(returned) == mass_before_cycle


def _skid_workspace_after_forward() -> tuple[EnvConfig, np.ndarray]:
    cfg = _env_config((2,))
    empty = np.zeros(SHAPE, dtype=np.int8)
    probe = _set_pose(_state(empty, env_cfg=cfg), 0, CENTER, cabin=0)
    moved = probe._handle_move_forward()
    assert not np.array_equal(
        np.asarray(probe._get_current_agent_state().pos_base),
        np.asarray(moved._get_current_agent_state().pos_base),
    )
    map_cyl, map_local = moved._get_map_local_and_cyl_coords()
    workspace = np.asarray(
        moved._get_dig_dump_mask_skidsteer(map_cyl, map_local)
    ).reshape(SHAPE)
    coordinates = np.argwhere(workspace)
    if len(coordinates) < 52:
        raise AssertionError("Expected a usable skid-steer workspace.")
    return cfg, coordinates


def test_skid_pickup_ignores_hole_and_productive_dump_is_rewarded():
    cfg, coordinates = _skid_workspace_after_forward()
    pile_cells = coordinates[:8]
    hole_cell = coordinates[9]
    dump_cells = coordinates[20:52]

    target = np.zeros(SHAPE, dtype=np.int8)
    target[tuple(dump_cells.T)] = 1
    action = np.zeros(SHAPE, dtype=np.int8)
    action[tuple(pile_cells.T)] = 1
    action[tuple(hole_cell)] = -1

    state = _set_pose(
        _state(target, action=action, env_cfg=cfg),
        0,
        CENTER,
        cabin=0,
    )
    initial_mass = _mass(state)
    loaded, pickup = _step_trace(state, TrackedAction.forward())

    assert pickup["load_after"] == len(pile_cells)
    assert pickup["extraction_reward"] == 0
    assert pickup["mass_after"] == initial_mass
    assert np.asarray(loaded.world.action_map.map)[tuple(hole_cell)] == -1
    assert float(loaded._get_current_agent_state().carry_relocation_credit) > 0

    shovel_up, lift = _step_trace(loaded, TrackedAction.do())
    assert lift["action_raw"] == pytest.approx(0.0)
    assert int(np.asarray(shovel_up._get_current_agent_state().shovel_lifted)[0]) == 1

    dumped, dump = _step_trace(shovel_up, TrackedAction.do())
    assert dump["relocation_progress"] > 0
    assert dump["relocation_reward"] > 0
    assert dump["completion_after"] > dump["completion_before"]
    assert _mass(dumped) == initial_mass
    assert float(dumped._get_current_agent_state().carry_relocation_credit) == 0


def test_skid_pickup_and_redump_inside_accepted_zone_is_nonpositive():
    cfg, coordinates = _skid_workspace_after_forward()
    dump_cells = coordinates[:32]
    pile_cells = dump_cells[:8]

    target = np.zeros(SHAPE, dtype=np.int8)
    target[tuple(dump_cells.T)] = 1
    action = np.zeros(SHAPE, dtype=np.int8)
    action[tuple(pile_cells.T)] = 1
    state = _set_pose(
        _state(target, action=action, env_cfg=cfg),
        0,
        CENTER,
        cabin=0,
    )
    initial_mass = _mass(state)
    initial_completion = _completion(state)
    initial_potential = _potential(state)

    loaded, pickup = _step_trace(state, TrackedAction.forward())
    shovel_up, lift = _step_trace(loaded, TrackedAction.do())
    returned, redump = _step_trace(shovel_up, TrackedAction.do())

    assert pickup["extraction_reward"] == 0
    assert float(loaded._get_current_agent_state().carry_relocation_credit) == 0
    assert redump["relocation_progress"] == pytest.approx(0.0)
    assert pickup["action_raw"] + lift["action_raw"] + redump["action_raw"] <= 0
    assert _mass(returned) == initial_mass
    assert _completion(returned) == pytest.approx(initial_completion)
    assert _potential(returned) == pytest.approx(initial_potential)


def _truck_geometry() -> tuple[State, np.ndarray, np.ndarray]:
    cfg = _env_config((0, 1))
    empty = np.zeros(SHAPE, dtype=np.int8)
    probe = _state(empty, env_cfg=cfg)
    excavator_cone = _cone(probe, 0, cabin=0)
    excavator_cells = np.argwhere(excavator_cone)
    if len(excavator_cells) < 20:
        raise AssertionError("Expected a non-empty excavator workspace.")

    truck_position = excavator_cells[len(excavator_cells) // 2].astype(np.int16)
    probe = _set_pose(probe, 1, truck_position, cabin=0)
    truck_cone = _cone(probe, 1, cabin=0)

    yy, xx = np.indices(SHAPE)
    truck_clearance = (
        (yy - int(truck_position[0])) ** 2 + (xx - int(truck_position[1])) ** 2
    ) > 16
    dig_candidates = np.argwhere(excavator_cone & truck_clearance & ~truck_cone)
    dump_candidates = np.argwhere(truck_cone & ~excavator_cone)
    if len(dig_candidates) < 8 or len(dump_candidates) < 24:
        raise AssertionError("Expected disjoint excavation and truck-dump cells.")

    target = np.zeros(SHAPE, dtype=np.int8)
    target[tuple(dig_candidates[:8].T)] = -1
    target[tuple(dump_candidates[:24].T)] = 1
    distance = np.ones(SHAPE, dtype=np.float32)
    distance[target > 0] = 0.0

    state = _state(target, distance=distance, env_cfg=cfg)
    state = _set_pose(state, 0, CENTER, cabin=0)
    state = _set_pose(state, 1, truck_position, cabin=0)
    return state, target, distance


def test_truck_handoff_moves_credit_without_reward_then_dump_pays_once():
    state, _, _ = _truck_geometry()
    initial_mass = _mass(state)

    after_dig, dig = _step_trace(state, TrackedAction.do())
    assert dig["load_after"] > 0
    excavator_after_dig = after_dig.agent.agent_states[0]

    # Give the truck its ordinary no-op turn, returning control to the excavator.
    ready_to_transfer, _ = _step_trace(after_dig, TrackedAction.do_nothing())
    assert int(np.asarray(ready_to_transfer.agent.current_agent)) == 0

    after_transfer, transfer = _step_trace(
        ready_to_transfer,
        TrackedAction.do(),
    )
    excavator_after_transfer = after_transfer.agent.agent_states[0]
    truck_after_transfer = after_transfer.agent.agent_states[1]

    assert transfer["world_changed"] == 0
    assert transfer["load_before"] > 0
    assert transfer["load_after"] == 0
    assert int(np.asarray(truck_after_transfer.loaded)[0]) > 0
    assert _mass(after_transfer) == initial_mass

    assert transfer["action_raw"] == pytest.approx(0.0)
    assert transfer["relocation_reward"] == pytest.approx(0.0)
    assert float(truck_after_transfer.carry_relocation_credit) == pytest.approx(
        float(excavator_after_dig.carry_relocation_credit)
    )
    assert int(np.asarray(excavator_after_transfer.loaded)[0]) == 0
    assert float(excavator_after_transfer.carry_relocation_credit) == 0

    after_truck_dump, truck_dump = _step_trace(
        after_transfer,
        TrackedAction.do(),
    )
    assert truck_dump["world_changed"] == 1
    assert truck_dump["dump_progress"] > 0
    assert truck_dump["action_raw"] > 0
    assert truck_dump["load_after"] == 0
    assert truck_dump["completion_after"] > truck_dump["completion_before"]
    assert _mass(after_truck_dump) == initial_mass


def test_truck_handoff_is_all_or_nothing_when_capacity_is_insufficient():
    state, _, _ = _truck_geometry()
    state = state._replace(
        env_cfg=state.env_cfg._replace(truck_capacity=4),
    )
    carrying = state._handle_dig()
    excavator_before = carrying.agent.agent_states[0]
    truck_before = carrying.agent.agent_states[1]
    assert int(np.asarray(excavator_before.loaded)[0]) > state.env_cfg.truck_capacity

    attempted = carrying._try_truck_transfer_on_excavator_dump()
    excavator_after = attempted.agent.agent_states[0]
    truck_after = attempted.agent.agent_states[1]

    assert int(np.asarray(excavator_after.loaded)[0]) == int(
        np.asarray(excavator_before.loaded)[0]
    )
    assert int(np.asarray(truck_after.loaded)[0]) == int(
        np.asarray(truck_before.loaded)[0]
    )
    assert float(excavator_after.carry_relocation_credit) == pytest.approx(
        float(excavator_before.carry_relocation_credit)
    )
    assert float(truck_after.carry_relocation_credit) == pytest.approx(
        float(truck_before.carry_relocation_credit)
    )
    assert _mass(attempted) == _mass(carrying)
