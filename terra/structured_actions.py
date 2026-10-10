"""Opt-in solo tracked actions with arguments and a modeled machine-time cost.

The native State/config schemas and eight legacy action IDs are unchanged. A
move requests up to 1--5 cells from its original pose (native rounded swept
translation); one-cell requests are cardinal-only because a single rounded
oblique step does not preserve its direction. Oblique requests that would
clip to a one-cell endpoint are blocked. A base turn executes 1--6 individually
checked 30 degree turns.
DO chooses an absolute cabin index relative to the chassis and performs one
native dig OR unload. Cabin IDs 4/5 remain available to the manual inspector.

This module is a functional JAX core. The caller carries ``StructuredClock``
and elapsed time separately from State, and calls ``structured_termination``
once per decision. The explicit ``material_time_v1`` reward uses undiscounted
material-potential differences, existing optional behavior costs and a time
cost. It never calls the legacy 450-step reward/termination code.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.config import (
    REWARD_V2_ALPHA, REWARD_V2_BETA, REWARD_V2_SHAPING_WEIGHT,
    REWARD_V2_SUCCESS_BONUS, REWARD_V2_HORIZON_FAILURE_PENALTY,
)
from terra.env import TerraEnv
from terra.state import State
from terra.wrappers import TraversabilityMaskWrapper


class StructuredAction(NamedTuple):
    action: Array | int
    amount: Array | int = 1
    heading: Array | int = -1


class StructuredClock(NamedTuple):
    visit_open: Array | bool = False
    moved: Array | bool = False


class StructuredTimeConfig(NamedTuple):
    # Same estimates as terra_time_reward_20261008/terra/config.py. The M445
    # 2026-10-07 run informs digging/setup; navigation/relocation/base turning
    # remain assumptions. A material unit is tile_size**3 cubic metres.
    dig_s_per_m3: float = 226.0  # includes dumping: never charge it twice
    setup_s: float = 415.0
    relocation_s: float = 15.0
    nav_speed_mps: float = 0.5
    base_turn_s_per_rad: float = 5.0
    cabin_turn_s_per_rad: float = 1.0 / 0.28
    time_cost_total: float = 3.6
    # Charge setup_s on every dig (each dig is a machine workspace) instead of
    # once per visit; dumps never pay it.
    setup_per_dig: bool = False


class StructuredTransition(NamedTuple):
    state: State
    reward: Array
    duration_s: Array
    info: dict[str, Array]


class StructuredTermination(NamedTuple):
    done: Array
    task_done: Array
    timeout: Array
    reward: Array


def _cast_like(candidate, original):
    return jax.tree_util.tree_map(
        lambda value, old: jnp.asarray(value, dtype=jnp.asarray(old).dtype),
        candidate, original,
    )


def _at_heading(state, heading):
    cur = state._get_current_agent_state()
    heading = jnp.asarray(heading, jnp.int32)
    angle = jnp.where(heading < 0, cur.angle_cabin, heading).astype(cur.angle_cabin.dtype)
    return state._set_current_agent_state(cur._replace(angle_cabin=angle))


def _move(state, direction, amount):
    # Override only this request's maximum displacement, then restore config.
    # Native code rounds each candidate from the original pose and checks its
    # swept polygon; repeatedly adding a rounded unit step is NOT equivalent.
    cfg = state.env_cfg
    request = state._replace(env_cfg=cfg._replace(
        agent=cfg.agent._replace(move_tiles=jnp.asarray(amount, jnp.int32))))
    cur = request._get_current_agent_state()
    orientation = jax.lax.cond(
        jnp.asarray(direction) == 0,
        lambda: request._base_orientation_to_one_hot_forward(cur.angle_base),
        lambda: request._base_orientation_to_one_hot_backwards(cur.angle_base),
    )
    oblique = (jnp.ravel(cur.angle_base)[0] % 3) != 0
    result = jax.lax.cond(
        (cur.loaded[0] == 0) & (~oblique | (jnp.asarray(amount) >= 2)),
        lambda: request._move_on_orientation(orientation),
        lambda: request,
    )
    result = _cast_like(result._replace(env_cfg=cfg), state)
    delta = (result._get_current_agent_state().pos_base.astype(jnp.int32)
             - cur.pos_base.astype(jnp.int32))
    # Native clipping/turn-keeping can select its one-cell candidate even for
    # a longer request. Reject that outcome as well. Oblique one-cell rounded
    # steps have length 1 or sqrt(2); every two-cell candidate is >=2 cells.
    too_short = oblique & (jnp.sum(delta * delta) < 2.25)
    return jax.lax.cond(too_short, lambda: state, lambda: result)


def _turn(state, direction, amount):
    """Stop at the first blocked intermediate orientation, never jump over it."""
    def step(_, carry):
        before, count, blocked = carry
        def execute():
            candidate = jax.lax.cond(
                jnp.asarray(direction) == 0, before._handle_clock, before._handle_anticlock)
            candidate = _cast_like(candidate, before)
            changed = jnp.any(candidate._get_current_agent_state().angle_base
                              != before._get_current_agent_state().angle_base)
            return candidate, count + changed.astype(jnp.int32), ~changed
        return jax.lax.cond(blocked, lambda: carry, execute)
    return jax.lax.fori_loop(
        0, jnp.asarray(amount, jnp.int32), step,
        (state, jnp.int32(0), jnp.bool_(False)),
    )[:2]


def _work(state, heading):
    aimed = _at_heading(state, heading)
    after = _cast_like(aimed._handle_do(), aimed)
    old_load = aimed._get_current_agent_state().loaded[0]
    new_load = after._get_current_agent_state().loaded[0]
    # This includes necessary relifting and complete off-zone staging, while
    # excluding native no-ops or failed full-load containment/storage checks.
    effective = jnp.where(old_load > 0, new_load == 0, new_load > 0)
    return aimed, after, effective


def structured_action_masks(state: State) -> dict[str, Array]:
    """Exact native effect masks, indexed by absolute cabin heading.

    A request may be clipped by native movement/turn checks; it is valid if it
    executes any progress. Masks include native full-load outcomes, not just
    candidate dump geometry or fresh-target counts. WAIT is always available
    as the all-invalid fallback. Manual cabin-only actions remain available.
    """
    cur = state._get_current_agent_state()
    def move_direction(direction):
        return jax.vmap(lambda amount: jnp.any(
            _move(state, direction, amount)._get_current_agent_state().pos_base
            != cur.pos_base))(jnp.arange(1, 6, dtype=jnp.int32))
    def turn_direction(direction):
        # A turn stops at its first blocked step, so a request of any amount
        # makes progress exactly when its first 30-degree step does.
        return jnp.broadcast_to(_turn(state, direction, 1)[1] > 0, (6,))
    move = jax.vmap(move_direction)(jnp.arange(2))
    turn = jax.vmap(turn_direction)(jnp.arange(2))
    do = jax.vmap(lambda heading: _work(state, heading)[2])(jnp.arange(12))
    action = jnp.concatenate((jnp.any(move, axis=1), jnp.any(turn, axis=1),
                              jnp.ones(2, dtype=jnp.bool_),
                              jnp.asarray([jnp.any(do), True])))
    return dict(move_mask=move, turn_mask=turn, do_mask=do, action_mask=action)


def _angle_distance(before, after, bins):
    bins = jnp.asarray(bins, jnp.float32)
    delta = jnp.ravel(after)[0].astype(jnp.float32) - jnp.ravel(before)[0].astype(jnp.float32)
    return jnp.abs((delta + bins / 2) % bins - bins / 2) * (2 * jnp.pi / bins)


def structured_transition(
    state: State, action: StructuredAction, *,
    clock: StructuredClock = StructuredClock(),
    timing: StructuredTimeConfig = StructuredTimeConfig(),
    time_budget_s: float = 14400.0,
) -> StructuredTransition:
    """One argument-bearing decision; no legacy horizon or terminal rewards.

    Invalid requests (including infeasible DO) leave the physical state intact
    and still consume one decision. The HTTP API may reject them before here.
    Only solo tracked excavators with twelve base/cabin bins are supported.
    Elapsed clock and returned visit flags must be carried/reset by the caller.
    """
    kind = jnp.asarray(action.action, jnp.int32)
    amount = jnp.asarray(action.amount, jnp.int32)
    heading = jnp.asarray(action.heading, jnp.int32)
    cur = state._get_current_agent_state()
    supported = ((state.agent.num_agents == 1) & (cur.agent_type[0] == 0)
                 & (cur.action_type[0] == 0) & (state.env_cfg.agent.angles_base == 12)
                 & (state.env_cfg.agent.angles_cabin == 12))
    valid = supported & (kind >= 0) & (kind <= 7)
    oblique = (jnp.ravel(cur.angle_base)[0] % 3) != 0
    valid &= jnp.where(kind < 2, (amount >= 1) & (amount <= 5)
                       & (~oblique | (amount >= 2)), True)
    valid &= jnp.where((kind >= 2) & (kind <= 3), (amount >= 1) & (amount <= 6), True)
    valid &= jnp.where(kind == 6, (heading >= -1) & (heading < 12), True)

    def move():
        return _move(state, kind, amount), jnp.int32(0)
    def turn():
        return _turn(state, kind - 2, amount)
    def cabin():
        after = jax.lax.cond(kind == 4, state._handle_cabin_clock, state._handle_cabin_anticlock)
        return _cast_like(after, state), jnp.int32(0)
    def work():
        _, after, effective = _work(state, heading)
        return jax.lax.cond(effective, lambda: after, lambda: state), jnp.int32(0)
    def wait():
        return state, jnp.int32(0)
    def execute():
        return jax.lax.switch(kind, (move, move, turn, turn, cabin, cabin, work, wait))
    after, turns = jax.lax.cond(valid, execute, wait)
    after = after._replace(
        env_steps=state.env_steps + jnp.asarray(1, jnp.asarray(state.env_steps).dtype),
        stall_age_steps=state._next_stall_age_steps(after),
    )
    info = TerraEnv._transition_diagnostics(state, after)
    after = TerraEnv._accumulate_productive_workspace_cycles(state, after, info)
    # Native timeout is action-count based and does not apply in this mode.
    info["timeout"] = jnp.bool_(False)
    info["productive_workspace_cycles"] = after.productive_workspace_cycles
    nxt = after._get_current_agent_state()
    tile = jnp.asarray(state.env_cfg.tile_size, jnp.float32)
    travel_m = jnp.linalg.norm(nxt.pos_base.astype(jnp.float32) - cur.pos_base.astype(jnp.float32)) * tile
    base_rad = turns.astype(jnp.float32) * (2 * jnp.pi / 12)
    cabin_rad = _angle_distance(cur.angle_cabin, nxt.angle_cabin, 12)
    loaded_m3 = jnp.maximum(nxt.loaded[0].astype(jnp.float32) - cur.loaded[0].astype(jnp.float32), 0) * tile**3
    moved = (travel_m > 0) | (base_rad > 0)
    event = after.retained_work_events[state.agent.current_agent] > state.retained_work_events[state.agent.current_agent]
    setup = (loaded_m3 > 0) if timing.setup_per_dig else event & ~jnp.asarray(clock.visit_open)
    relocation = setup & jnp.asarray(clock.moved)
    duration = (travel_m / timing.nav_speed_mps + base_rad * timing.base_turn_s_per_rad
                + cabin_rad * timing.cabin_turn_s_per_rad + loaded_m3 * timing.dig_s_per_m3
                + setup.astype(jnp.float32) * timing.setup_s
                + relocation.astype(jnp.float32) * timing.relocation_s).astype(jnp.float32)

    # Use the aimed state for orientation-dependent digging costs. The cabin
    # swing itself carries only the physical time cost, with no lost DO event.
    reward_before = jax.lax.cond((kind == 6) & valid, lambda: _at_heading(state, heading), lambda: state)
    q, _, p = state._reward_v2_progress()
    q_next, _, p_next = after._reward_v2_progress()
    material = REWARD_V2_SHAPING_WEIGHT * (REWARD_V2_ALPHA * (q_next - q) + REWARD_V2_BETA * (p_next - p))
    behavior = reward_before._reward_v2_behavior_costs(after)
    behavior_reward = sum(behavior[name] for name in (
        "reward_v2_lateral_dig", "reward_v2_base_travel", "reward_v2_base_turn",
        "reward_v2_retained_setup", "reward_v2_retained_travel", "reward_v2_retained_turn"))
    time_reward = -timing.time_cost_total * duration / jnp.asarray(time_budget_s, jnp.float32)
    reward = material + behavior_reward + time_reward
    completion = after._get_task_completion(after.world.action_map.map, after.world.target_map.map)
    info.update(behavior)
    info.update(
        request_valid=valid,
        executed_turn_steps=turns,
        executed_travel_m=travel_m,
        executed_cabin_rad=cabin_rad,
        work_loaded_m3=loaded_m3,
        time_new_setup=setup,
        time_relocation=relocation,
        time_visit_open=jnp.where(moved, False, event | jnp.asarray(clock.visit_open)),
        time_moved=jnp.where(event, False, jnp.asarray(clock.moved) | moved),
        duration_s=duration,
        reward_material=material,
        reward_behavior=behavior_reward,
        reward_time=time_reward,
        task_done=completion["absolute_completion"] >= 1.0 - 1e-6,
    )
    after = TraversabilityMaskWrapper.wrap(
        after, update_reachability=(kind == 6) & info["material_or_load_changed"])
    return StructuredTransition(after, reward.astype(jnp.float32), duration, info)


def structured_termination(
    state: State, elapsed_s, *, time_budget_s, decision_limit=450,
) -> StructuredTermination:
    """Authorize exact completion only within both budgets (boundary allowed).

    Actions execute atomically. An action that finishes after the time budget
    expires counts as a timeout, even if its final terrain completes the task.
    Add this reward once on termination; never continue rewarding a terminal
    state. A decision cap terminates zero-duration blocked/wait loops.
    """
    elapsed = jnp.asarray(elapsed_s, jnp.float32)
    budget = jnp.asarray(time_budget_s, jnp.float32)
    completed = state._get_task_completion(
        state.world.action_map.map, state.world.target_map.map)["absolute_completion"] >= 1.0 - 1e-6
    within = (elapsed <= budget) & (state.env_steps <= decision_limit)
    success = completed & within
    exhausted = (elapsed >= budget) | (state.env_steps >= decision_limit)
    timeout = exhausted & ~success
    done = success | timeout
    reward = jnp.where(success, jnp.float32(REWARD_V2_SUCCESS_BONUS),
                       jnp.where(timeout, -jnp.float32(REWARD_V2_HORIZON_FAILURE_PENALTY), jnp.float32(0)))
    return StructuredTermination(done, success, timeout, reward)
