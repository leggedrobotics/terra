"""Exact initial-scenario direct-service validation for Terra benchmarks."""

from __future__ import annotations

from collections import deque
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.actions import TrackedActionType
from terra.benchmark_state import validate_benchmark_initial_agent
from terra.env import TerraEnv
from terra.state import State

_MOVEMENT_ACTIONS = np.asarray(
    (
        TrackedActionType.FORWARD,
        TrackedActionType.BACKWARD,
        TrackedActionType.CLOCK,
        TrackedActionType.ANTICLOCK,
    ),
    dtype=np.int8,
)
_MOVEMENT_BATCH_SIZE = 128
_PREFILTER_BATCH_SIZE = 128
_SERVICE_BATCH_SIZE = 4


def _tracked_action(action_index: jax.Array) -> TrackedAction:
    action = jnp.reshape(jnp.asarray(action_index, dtype=jnp.int8), (1,))
    return TrackedAction(
        type=jnp.zeros((1,), dtype=jnp.int8),
        action=action,
    )


def _replace_active_agent(
    state: State,
    *,
    pos_base: jax.Array,
    angle_base: jax.Array,
) -> State:
    active = state.agent.agent_states[0]
    active = active._replace(
        pos_base=jnp.asarray(pos_base, dtype=active.pos_base.dtype),
        angle_base=jnp.reshape(
            jnp.asarray(angle_base, dtype=active.angle_base.dtype),
            active.angle_base.shape,
        ),
    )
    return state._replace(
        agent=state.agent._replace(
            agent_states=(active,) + state.agent.agent_states[1:]
        )
    )


def _state_at_base_pose(state: State, pose: jax.Array) -> State:
    return _replace_active_agent(
        state,
        pos_base=pose[:2],
        angle_base=pose[2],
    )


def _rotate_cabin_steps(state: State, steps: jax.Array) -> State:
    action = _tracked_action(jnp.int8(TrackedActionType.CABIN_ANTICLOCK))

    def rotate_once(_: int, candidate: State) -> State:
        return candidate._step(action)

    return jax.lax.fori_loop(0, jnp.asarray(steps, dtype=jnp.int32), rotate_once, state)


def _transition_state(state: State, action: TrackedAction) -> State:
    """Run the state-changing portion of ``TerraEnv.step_no_reset``."""
    new_state = state._step(action)
    is_do = action.action[0] == TrackedActionType.DO
    terrain_changed = jnp.any(
        new_state.world.action_map.map != state.world.action_map.map
    )
    update_reachability = jnp.logical_and(is_do, terrain_changed)
    return TerraEnv.wrap_state(
        new_state,
        update_reachability=update_reachability,
    )


def _movement_successors_for_pose(state: State, pose: jax.Array) -> jax.Array:
    candidate = _state_at_base_pose(state, pose)

    def take_action(action_index: jax.Array) -> jax.Array:
        moved = candidate._step(_tracked_action(action_index))
        active = moved.agent.agent_states[0]
        return jnp.concatenate(
            (
                active.pos_base.astype(jnp.int32),
                active.angle_base.astype(jnp.int32),
            )
        )

    return jax.vmap(take_action)(jnp.asarray(_MOVEMENT_ACTIONS))


@jax.jit
def _movement_successor_batch(state: State, poses: jax.Array) -> jax.Array:
    return jax.vmap(_movement_successors_for_pose, in_axes=(None, 0))(state, poses)


def _dig_prefilter_for_candidate(state: State, candidate: jax.Array) -> jax.Array:
    posed = _state_at_base_pose(state, candidate[:3])
    posed = _rotate_cabin_steps(posed, candidate[3])

    dig_mask = posed._build_dig_dump_cone()
    dig_mask = posed._mask_out_wrong_dig_tiles(dig_mask)
    dig_mask = posed._mask_out_single_tile_digs(dig_mask)
    action_map = posed.world.action_map.map.reshape(-1)
    selected_sum = action_map.astype(jnp.int32) @ dig_mask.astype(jnp.int32)
    moving_dumped_dirt = selected_sum > 0
    dig_volume = jnp.where(
        moving_dumped_dirt,
        selected_sum,
        jnp.sum(dig_mask.astype(jnp.int32)),
    )
    return jnp.logical_and(
        jnp.logical_not(posed._workspace_intersects_obstacle()),
        jnp.logical_and(
            dig_volume > 0,
            dig_volume <= jnp.iinfo(jnp.int8).max,
        ),
    )


@jax.jit
def _dig_prefilter_batch(state: State, candidates: jax.Array) -> jax.Array:
    return jax.vmap(_dig_prefilter_for_candidate, in_axes=(None, 0))(state, candidates)


def _remaining_required_volume(state: State) -> jax.Array:
    target = state.world.target_map.map.astype(jnp.int32)
    action = state.world.action_map.map.astype(jnp.int32)
    required = jnp.where(target < 0, -target, 0)
    completed = jnp.where(
        target < 0,
        jnp.minimum(jnp.maximum(-action, 0), required),
        0,
    )
    return jnp.maximum(required - completed, 0)


def _target_progress(before: State, after: State) -> jax.Array:
    target = before.world.target_map.map.astype(jnp.int32)
    required = jnp.where(target < 0, -target, 0)
    before_completed = jnp.where(
        target < 0,
        jnp.minimum(
            jnp.maximum(
                -before.world.action_map.map.astype(jnp.int32),
                0,
            ),
            required,
        ),
        0,
    )
    after_completed = jnp.where(
        target < 0,
        jnp.minimum(
            jnp.maximum(
                -after.world.action_map.map.astype(jnp.int32),
                0,
            ),
            required,
        ),
        0,
    )
    remaining = jnp.maximum(required - before_completed, 0)
    return jnp.minimum(
        jnp.maximum(after_completed - before_completed, 0),
        remaining,
    )


def _classify_complete_dump(
    before: State,
    after: State,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Classify one dump without treating an off-zone unload as legal service."""
    before_loaded = before.agent.agent_states[0].loaded[0].astype(jnp.int32)
    after_loaded = after.agent.agent_states[0].loaded[0].astype(jnp.int32)
    delta = after.world.action_map.map.astype(
        jnp.int32
    ) - before.world.action_map.map.astype(jnp.int32)
    accepted = before._accepted_dump_mask()
    complete = jnp.logical_and(
        before_loaded > 0,
        jnp.logical_and(
            after_loaded == 0,
            jnp.sum(delta) == before_loaded,
        ),
    )
    legal = jnp.logical_and(
        complete,
        jnp.logical_and(
            jnp.all(jnp.where(accepted, 0, delta) == 0),
            jnp.sum(jnp.where(accepted, delta, 0)) == before_loaded,
        ),
    )
    wrong_complete = jnp.logical_and(
        complete,
        jnp.logical_and(
            jnp.all(jnp.where(accepted, delta, 0) == 0),
            jnp.sum(jnp.where(accepted, 0, delta)) == before_loaded,
        ),
    )
    mixed_contract_violation = jnp.logical_and(
        complete,
        jnp.logical_not(jnp.logical_or(legal, wrong_complete)),
    )
    rejected = jnp.logical_not(complete)
    return legal, wrong_complete, rejected, mixed_contract_violation


def _service_for_candidate(
    state: State,
    candidate: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    posed = _state_at_base_pose(state, candidate[:3])
    posed = _rotate_cabin_steps(posed, candidate[3])
    posed = TerraEnv.wrap_state(posed, update_reachability=jnp.bool_(False))

    do_action = _tracked_action(jnp.int8(TrackedActionType.DO))
    cabin_action = _tracked_action(jnp.int8(TrackedActionType.CABIN_ANTICLOCK))
    dug = _transition_state(posed, do_action)
    progress = _target_progress(posed, dug)
    dig_valid = jnp.logical_and(
        dug.agent.agent_states[0].loaded[0] > 0,
        jnp.sum(progress) > 0,
    )

    def try_dump_heading(
        _: int,
        carry: tuple[State, jax.Array],
    ) -> tuple[State, jax.Array]:
        rotated, counts = carry
        dumped = _transition_state(rotated, do_action)
        legal, wrong, rejected, violation = _classify_complete_dump(
            rotated,
            dumped,
        )
        counts = counts + jnp.asarray(
            (legal, wrong, rejected, violation),
            dtype=jnp.int32,
        )
        return _transition_state(rotated, cabin_action), counts

    _, counts = jax.lax.fori_loop(
        0,
        state.env_cfg.agent.angles_cabin,
        try_dump_heading,
        (dug, jnp.zeros((4,), dtype=jnp.int32)),
    )
    workspace_progress = jnp.where(dig_valid, progress, jnp.zeros_like(progress))
    direct_progress = jnp.where(
        jnp.logical_and(dig_valid, counts[0] > 0),
        progress,
        jnp.zeros_like(progress),
    )
    diagnostics = jnp.concatenate(
        (
            jnp.asarray((dig_valid,), dtype=jnp.int32),
            counts,
        )
    )
    return workspace_progress, direct_progress, diagnostics


@jax.jit
def _service_batch(
    state: State,
    candidates: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return jax.vmap(_service_for_candidate, in_axes=(None, 0))(state, candidates)


def _target_dig_progress_for_candidate(
    state: State,
    candidate: jax.Array,
) -> jax.Array:
    """Replay the real dig transition used to audit prefilter false negatives."""
    posed = _state_at_base_pose(state, candidate[:3])
    posed = _rotate_cabin_steps(posed, candidate[3])
    posed = TerraEnv.wrap_state(posed, update_reachability=jnp.bool_(False))
    dug = _transition_state(
        posed,
        _tracked_action(jnp.int8(TrackedActionType.DO)),
    )
    return jnp.sum(_target_progress(posed, dug))


@jax.jit
def _target_dig_progress_batch(
    state: State,
    candidates: jax.Array,
) -> jax.Array:
    return jax.vmap(
        _target_dig_progress_for_candidate,
        in_axes=(None, 0),
    )(state, candidates)


def _pad_rows(rows: np.ndarray, batch_size: int) -> tuple[np.ndarray, int]:
    valid_count = len(rows)
    if valid_count == 0:
        raise ValueError("Cannot pad an empty row batch.")
    if valid_count > batch_size:
        raise ValueError(f"Batch has {valid_count} rows, maximum is {batch_size}.")
    padded = np.repeat(rows[:1], batch_size, axis=0)
    padded[:valid_count] = rows
    return padded, valid_count


def _reachable_base_poses(
    initial_state: State,
) -> tuple[np.ndarray, dict[str, int]]:
    active = initial_state.agent.agent_states[0]
    initial_pose = (
        int(np.asarray(active.pos_base)[0]),
        int(np.asarray(active.pos_base)[1]),
        int(np.asarray(active.angle_base).reshape(-1)[0]),
    )
    visited = {initial_pose}
    frontier: deque[tuple[int, int, int]] = deque((initial_pose,))
    source_rows_logical = 0
    source_rows_padded = 0

    while frontier:
        rows = []
        while frontier and len(rows) < _MOVEMENT_BATCH_SIZE:
            rows.append(frontier.popleft())
        source_rows = np.asarray(rows, dtype=np.int32)
        padded, valid_count = _pad_rows(source_rows, _MOVEMENT_BATCH_SIZE)
        successors = np.asarray(
            jax.device_get(
                _movement_successor_batch(
                    initial_state,
                    jnp.asarray(padded, dtype=jnp.int32),
                )
            )
        )[:valid_count]
        source_rows_logical += valid_count
        source_rows_padded += len(padded)

        for source, source_successors in zip(source_rows, successors):
            source_pose = tuple(int(value) for value in source)
            for successor in source_successors:
                successor_pose = tuple(int(value) for value in successor)
                if successor_pose == source_pose or successor_pose in visited:
                    continue
                visited.add(successor_pose)
                frontier.append(successor_pose)

    return np.asarray(sorted(visited), dtype=np.int32), {
        "source_rows_logical": source_rows_logical,
        "transition_attempts_logical": (source_rows_logical * len(_MOVEMENT_ACTIONS)),
        "source_rows_padded_executed": source_rows_padded,
        "transition_attempts_padded_executed": (
            source_rows_padded * len(_MOVEMENT_ACTIONS)
        ),
    }


def _iter_chunks(rows: np.ndarray, batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _candidate_rows(
    poses: np.ndarray,
    cabin_orientations: int,
) -> np.ndarray:
    repeated_poses = np.repeat(poses, cabin_orientations, axis=0)
    cabin_steps = np.tile(
        np.arange(cabin_orientations, dtype=np.int32),
        len(poses),
    )
    return np.column_stack((repeated_poses, cabin_steps)).astype(
        np.int32,
        copy=False,
    )


def _prefilter_candidates(
    state: State,
    candidates: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    accepted = []
    padded_rows_executed = 0
    for rows in _iter_chunks(candidates, _PREFILTER_BATCH_SIZE):
        padded, valid_count = _pad_rows(rows, _PREFILTER_BATCH_SIZE)
        padded_rows_executed += len(padded)
        mask = np.asarray(
            jax.device_get(
                _dig_prefilter_batch(
                    state,
                    jnp.asarray(padded, dtype=jnp.int32),
                )
            )
        )[:valid_count]
        accepted.extend(rows[mask])
    if not accepted:
        accepted_rows = np.empty((0, 4), dtype=np.int32)
    else:
        accepted_rows = np.asarray(accepted, dtype=np.int32)
    return accepted_rows, {
        "candidate_rows_logical": int(len(candidates)),
        "candidate_rows_padded_executed": padded_rows_executed,
    }


def compute_initial_direct_service(state: State) -> dict[str, Any]:
    """Replay exact initial dig-to-dump service from every reachable base pose."""
    if not isinstance(state, State):
        raise TypeError(f"Expected State, got {type(state).__name__}.")
    validate_benchmark_initial_agent(
        state.agent,
        env_cfg=state.env_cfg,
        padding_mask=state.world.padding_mask.map,
        action_map=state.world.action_map.map,
        dumpability_mask=state.world.dumpability_mask_init.map,
    )

    remaining = np.asarray(
        jax.device_get(_remaining_required_volume(state)),
        dtype=np.int32,
    )
    required_volume = int(remaining.sum())
    if required_volume <= 0:
        raise ValueError("Direct-service validation requires remaining dig volume.")
    if not bool(np.asarray(state._accepted_dump_mask()).any()):
        raise ValueError("Direct-service validation requires an accepted dump mask.")

    poses, movement_stats = _reachable_base_poses(state)
    cabin_orientations = int(state.env_cfg.agent.angles_cabin)
    candidates = _candidate_rows(poses, cabin_orientations)
    replay_candidates, prefilter_stats = _prefilter_candidates(state, candidates)

    workspace_union = np.zeros_like(remaining, dtype=np.int32)
    direct_union = np.zeros_like(remaining, dtype=np.int32)
    diagnostics = np.zeros((5,), dtype=np.int64)
    service_rows_padded = 0

    for rows in _iter_chunks(replay_candidates, _SERVICE_BATCH_SIZE):
        padded, valid_count = _pad_rows(rows, _SERVICE_BATCH_SIZE)
        service_rows_padded += len(padded)
        workspace, direct, batch_diagnostics = _service_batch(
            state,
            jnp.asarray(padded, dtype=jnp.int32),
        )
        workspace = np.asarray(jax.device_get(workspace))[:valid_count]
        direct = np.asarray(jax.device_get(direct))[:valid_count]
        batch_diagnostics = np.asarray(jax.device_get(batch_diagnostics))[:valid_count]
        workspace_union = np.maximum(
            workspace_union,
            workspace.max(axis=0, initial=0),
        )
        direct_union = np.maximum(
            direct_union,
            direct.max(axis=0, initial=0),
        )
        diagnostics += batch_diagnostics.sum(axis=0, dtype=np.int64)

    workspace_union = np.minimum(workspace_union, remaining)
    direct_union = np.minimum(direct_union, remaining)
    if diagnostics[4] != 0:
        raise RuntimeError(
            "A complete dump crossed the exact accepted-mask boundary; "
            "the C1a transition contract was violated."
        )

    workspace_volume = int(workspace_union.sum())
    direct_volume = int(direct_union.sum())
    successful_dig_replays = int(diagnostics[0])
    service_candidate_attempts = int(len(replay_candidates))
    dump_attempts = service_candidate_attempts * cabin_orientations
    classified_dump_attempts = int(diagnostics[1:4].sum())
    if classified_dump_attempts != dump_attempts:
        raise RuntimeError(
            "Dump-attempt accounting mismatch: "
            f"classified {classified_dump_attempts}, executed {dump_attempts}."
        )
    return {
        "required_volume": required_volume,
        "admissible_pose_count_initial": int(len(poses)),
        "base_pose_cabin_heading_candidates_initial": int(len(candidates)),
        "successful_target_dig_replays": successful_dig_replays,
        "movement_source_rows_logical": movement_stats["source_rows_logical"],
        "movement_transition_attempts_logical": movement_stats[
            "transition_attempts_logical"
        ],
        "movement_source_rows_padded_executed": movement_stats[
            "source_rows_padded_executed"
        ],
        "movement_transition_attempts_padded_executed": movement_stats[
            "transition_attempts_padded_executed"
        ],
        "dig_prefilter_candidate_rows_logical": prefilter_stats[
            "candidate_rows_logical"
        ],
        "dig_prefilter_candidate_rows_padded_executed": prefilter_stats[
            "candidate_rows_padded_executed"
        ],
        "service_dig_candidate_attempts_logical": service_candidate_attempts,
        "service_candidate_rows_padded_executed": service_rows_padded,
        "service_dig_do_transitions_padded_executed": service_rows_padded,
        "dump_do_attempts_logical": dump_attempts,
        "dump_do_transitions_padded_executed": (
            service_rows_padded * cabin_orientations
        ),
        "legal_complete_dump_attempts": int(diagnostics[1]),
        "wrong_complete_dump_attempts": int(diagnostics[2]),
        "rejected_dump_attempts": int(diagnostics[3]),
        "workspace_serviceable_volume_initial": workspace_volume,
        "direct_serviceable_volume_initial": direct_volume,
        "initial_workspace_coverage": workspace_volume / required_volume,
        "direct_service_coverage_initial": direct_volume / required_volume,
        "any_direct_transfer_pose_exists_initial": direct_volume > 0,
    }
