"""Shared structured episode limits and terminal rewards for play and training."""

from typing import NamedTuple

import jax.numpy as jnp

from terra.structured_actions import (
    StructuredTermination, StructuredTimeConfig, structured_termination,
)


class StructuredEpisodeConfig(NamedTuple):
    time_budget_s: float = 14400.
    time_budget_factor: float = 0.
    time_budget_offset_s: float = 0.
    decision_limit: int = 450
    decisions_per_dig_unit: float = 0.
    time_limit: bool = True
    success_time_bonus: float = 0.
    success_decision_bonus: float = 0.
    terminate_loaded_dead_ends: bool = True
    terminate_unloaded_dead_ends: bool = False
    precision_safety: bool = False
    precision_edge_weight: float = -1.


def structured_episode_limits(
    state, *, timing=StructuredTimeConfig(), config=StructuredEpisodeConfig(),
):
    """Return this map's time reference and maximum macro decisions.

    Dig units include target depth, not only the number of target cells. The
    fixed values are lower bounds when map scaling is enabled. A disabled time
    deadline still leaves a finite reference for observations and rewards.
    ``timing`` and ``config`` are static configuration when used under JIT.
    """
    units = jnp.sum(jnp.maximum(-state.world.target_map.map.astype(jnp.float32), 0.))
    unit_s = state.env_cfg.tile_size ** 3 * timing.dig_s_per_m3
    budget = (jnp.maximum(jnp.float32(config.time_budget_s),
                          config.time_budget_offset_s + config.time_budget_factor * unit_s * units)
              if config.time_budget_factor else jnp.float32(config.time_budget_s))
    decisions = (jnp.maximum(config.decision_limit,
                             jnp.ceil(config.decisions_per_dig_unit * units).astype(jnp.int32))
                 if config.decisions_per_dig_unit else jnp.int32(config.decision_limit))
    return budget, decisions


def structured_episode_termination(
    state, elapsed_s, *, timing=StructuredTimeConfig(),
    config=StructuredEpisodeConfig(), dead_end=False, unloaded_dead_end=False,
) -> StructuredTermination:
    """Check the episode boundary and include bonuses only on valid success.

    Completion exactly at either limit succeeds; completion beyond an enabled
    limit fails. Callers must stop or reset after ``done`` and add ``reward``
    once. ``dead_end`` must come from an exact full-load unload check;
    ``unloaded_dead_end`` must come from the conservative immutable-blocker check.
    """
    budget, decisions = structured_episode_limits(state, timing=timing, config=config)
    terminal = structured_termination(
        state, elapsed_s,
        time_budget_s=budget if config.time_limit else jnp.float32(jnp.inf),
        decision_limit=decisions,
        dead_end=((jnp.asarray(dead_end) & config.terminate_loaded_dead_ends)
                  | (jnp.asarray(unloaded_dead_end) & config.terminate_unloaded_dead_ends)),
    )
    bonus = terminal.task_done.astype(jnp.float32) * (
        config.success_time_bonus * jnp.clip(1. - elapsed_s / budget, 0., 1.)
        + config.success_decision_bonus * jnp.clip(1. - state.env_steps / decisions, 0., 1.))
    return terminal._replace(reward=terminal.reward + bonus)
