"""Explicit excavator-to-truck loading exceptions to workspace separation.

The default bitset is empty. An authorized directed pair may share ONLY the
excavator workspace with the receiver's body/workspace, during a compatible
requested action phase. Chassis/chassis, receiver-workspace/excavator-body and
all unregistered pairs retain their ordinary clearance checks.

This is a 2D macro-operation contract: Terra has no truck-bed/cab geometry or
measured arm-stow posture. It does not establish physical loading clearance.
Requested actions are never rewritten here. In particular, rejecting an
incompatible request does not turn it into permission for the partner to move.
"""
from numbers import Integral

import jax.numpy as jnp
import numpy as np


MAX_AGENTS = 4
# Keep import-time constants on the host. State imports this module lazily
# while tracing its first joint action, so jnp creation here would retain a
# tracer globally and poison later compiled calls.
_SLOTS = np.arange(MAX_AGENTS, dtype=np.int32)
_PAIR_BITS = _SLOTS[:, None] * MAX_AGENTS + _SLOTS[None, :]
_ALL_BITS = (1 << (MAX_AGENTS * MAX_AGENTS)) - 1


def loading_pair_mask(agent_types, pairs) -> int:
    """Encode validated directed (excavator slot, truck slot) registrations.

    Each vehicle has at most one interaction partner. Unsupported roles,
    inactive slots, self-pairs and shared participants fail at the host boundary.
    """
    types = tuple(agent_types)
    if not 1 <= len(types) <= MAX_AGENTS or any(
        isinstance(t, (bool, np.bool_)) or not isinstance(t, Integral) or t not in (0, 1, 2)
        for t in types
    ):
        raise ValueError("agent_types must contain one to four native role IDs (0, 1, 2)")
    participants, mask = set(), 0
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each loading interaction must be an explicit (excavator_slot, truck_slot) pair")
        source, target = pair
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in pair):
            raise ValueError("Loading participant slots must be integers")
        source, target = int(source), int(target)
        if not (0 <= source < len(types) and 0 <= target < len(types)) or source == target:
            raise ValueError("Loading participants must be distinct active slots")
        if types[source] != 0 or types[target] != 1:
            raise ValueError("Only an excavator-to-truck loading interaction is defined")
        if source in participants or target in participants:
            raise ValueError("A vehicle may have only one registered interaction partner")
        participants.update((source, target))
        mask |= 1 << (source * MAX_AGENTS + target)
    return mask


def validate_loading_pair_mask(agent_types, mask) -> int:
    """Validate a persisted scalar bitset before passing config to JAX."""
    if isinstance(mask, (bool, np.bool_)) or not isinstance(mask, Integral) or not 0 <= mask <= _ALL_BITS:
        raise ValueError("workspace_loading_pairs must be a nonnegative 16-bit integer")
    pairs = [(s, t) for s in range(MAX_AGENTS) for t in range(MAX_AGENTS)
             if int(mask) & (1 << (s * MAX_AGENTS + t))]
    return loading_pair_mask(agent_types, pairs)


def configured_loading_pairs(state):
    """Active directed registrations, with invalid raw configurations denied.

    Host callers must use validate_loading_pair_mask to report invalid input;
    this JAX check additionally prevents malformed persisted masks granting it.
    """
    configured_mask = jnp.asarray(state.env_cfg.workspace_loading_pairs).reshape(())
    integer_dtype = jnp.issubdtype(configured_mask.dtype, jnp.integer)
    raw_mask = configured_mask.astype(jnp.int32)
    raw = ((raw_mask >> _PAIR_BITS) & 1).astype(jnp.bool_)
    types = jnp.stack([a.agent_type[0] for a in state.agent.agent_states])
    present = _SLOTS < len(state.env_cfg.agent_types)
    typed = ((types[:, None] == 0) & (types[None, :] == 1)
             & present[:, None] & present[None, :])
    degree = jnp.sum(raw, axis=0) + jnp.sum(raw, axis=1)
    valid = (integer_dtype & (raw_mask >= 0) & (raw_mask <= _ALL_BITS) & jnp.all(degree <= 1)
             & jnp.all(~raw | typed))
    active = state.agent.agent_active.astype(jnp.bool_)
    return raw & typed & active[:, None] & active[None, :] & valid


def _component_exceptions(pairs):
    directed = jnp.zeros((MAX_AGENTS, MAX_AGENTS, 2, 2), dtype=jnp.bool_)
    directed = directed.at[:, :, 1, 0].set(pairs)
    directed = directed.at[:, :, 1, 1].set(pairs)
    return directed | jnp.transpose(directed, (1, 0, 3, 2))


def stationary_component_exceptions(state):
    """A registered docked pair may remain holding, including after unloading.

    Use this for stationary reset/current-state audits. Action sweeps require
    candidate_component_exceptions; this function never authorizes motion.
    """
    return _component_exceptions(configured_loading_pairs(state))


def _same_fields(before, candidate, fields):
    same = jnp.ones((MAX_AGENTS,), dtype=jnp.bool_)
    for name in fields:
        old = jnp.stack([jnp.asarray(getattr(a, name)).reshape(-1) for a in before.agent.agent_states])
        new = jnp.stack([jnp.asarray(getattr(a, name)).reshape(-1) for a in candidate.agent.agent_states])
        same &= jnp.all(old == new, axis=1)
    return same


def _same_ground(before, candidate):
    return (jnp.all(before.world.action_map.map == candidate.world.action_map.map)
            & jnp.all(before.world.target_map.map == candidate.world.target_map.map)
            & jnp.all(before.world.padding_mask.map == candidate.world.padding_mask.map))


def _transfer_matrix(before, candidate):
    old_load = jnp.stack([a.loaded[0] for a in before.agent.agent_states]).astype(jnp.int32)
    new_load = jnp.stack([a.loaded[0] for a in candidate.agent.agent_states]).astype(jnp.int32)
    old_credit = jnp.stack([a.carry_relocation_credit for a in before.agent.agent_states])
    new_credit = jnp.stack([a.carry_relocation_credit for a in candidate.agent.agent_states])
    source = jnp.eye(MAX_AGENTS, dtype=jnp.int32)[:, None, :]
    target = jnp.eye(MAX_AGENTS, dtype=jnp.int32)[None, :, :]
    load_expected = old_load[None, None, :] + old_load[:, None, None] * (target - source)
    # Match native float32 credit addition directly; subtraction clears donor.
    credit_expected = jnp.where(source.astype(jnp.bool_), jnp.float32(0.), old_credit[None, None, :])
    credit_expected = credit_expected + old_credit[:, None, None] * target
    unchanged_poses = jnp.all(_same_fields(before, candidate,
        ("pos_base", "angle_base", "angle_cabin", "wheel_angle", "shovel_lifted", "agent_type", "action_type")))
    fits = old_load[:, None] + old_load[None, :] <= before.env_cfg.truck_capacity
    return (configured_loading_pairs(before) & (old_load[:, None] > 0) & fits
            & jnp.all(new_load[None, None, :] == load_expected, axis=-1)
            & jnp.all(new_credit[None, None, :] == credit_expected, axis=-1)
            & unchanged_poses & _same_ground(before, candidate)
            & (jnp.sum(old_load) == jnp.sum(new_load)))


def verified_full_transfer(before, candidate, source_slot, receiver_slot):
    """Exact authorized whole-load and carry-credit transfer, with no soil edit."""
    return _transfer_matrix(before, candidate)[source_slot, receiver_slot]


def candidate_component_exceptions(before, candidate, requested_actions, acting_slot):
    """Exceptions for one proposed native action, using original joint requests.

    * Explicit WAIT may hold its registered pair's existing overlap.
    * Truck movement/steering (0..3) requires excavator-requested WAIT.
    * Excavator cabin rotation (4/5) requires truck-requested WAIT.
    * Excavator DO (6) requires truck-requested WAIT AND verified transfer.

    WAIT's native -1 alias is accepted too. No movement inferred from a failed
    request is granted a permission. For a non-acting registered pair, holding
    permissions remain available; every third-party comparison stays strict.
    """
    actions = jnp.asarray(requested_actions, jnp.int32).reshape(-1)
    if actions.shape[0] != len(before.env_cfg.agent_types) or not 1 <= actions.shape[0] <= MAX_AGENTS:
        raise ValueError("Expected one requested action per active agent slot")
    actions = jnp.pad(actions, (0, MAX_AGENTS - actions.shape[0]), constant_values=7)
    pairs = configured_loading_pairs(before)
    same_pose = _same_fields(before, candidate,
        ("pos_base", "angle_base", "angle_cabin", "wheel_angle", "shovel_lifted", "agent_type", "action_type"))
    same_body = _same_fields(before, candidate,
        ("pos_base", "angle_base", "wheel_angle", "shovel_lifted", "agent_type", "action_type"))
    same_cabin = _same_fields(before, candidate, ("angle_cabin",))
    material_same = _same_ground(before, candidate) & jnp.all(_same_fields(
        before, candidate, ("loaded", "carry_relocation_credit")))
    no_effect = material_same & jnp.all(same_pose)
    wait = (actions == 7) | (actions == -1)
    source_acts, target_acts = _SLOTS[:, None] == acting_slot, _SLOTS[None, :] == acting_slot
    source_cabin_only = (material_same & jnp.all(same_body)
                        & jnp.all(same_cabin[None, :] | jnp.eye(MAX_AGENTS, dtype=jnp.bool_), axis=1))
    receiver_fixed = jnp.all(_same_fields(before, candidate,
        ("angle_cabin", "shovel_lifted", "agent_type", "action_type")))
    receiver_only_motion = (material_same & receiver_fixed
                           & jnp.all(same_pose[None, :] | jnp.eye(MAX_AGENTS, dtype=jnp.bool_), axis=1))
    cabin_request = (actions[:, None] == 4) | (actions[:, None] == 5)
    source_allowed = ((wait[:, None] & no_effect)
        | (wait[None, :] & ((cabin_request & source_cabin_only[:, None])
                           | ((actions[:, None] == 6) & _transfer_matrix(before, candidate)))))
    target_allowed = ((wait[None, :] & no_effect)
        | (wait[:, None] & (actions[None, :] >= 0) & (actions[None, :] <= 3)
           & receiver_only_motion[None, :]))
    permitted = pairs & jnp.where(source_acts, source_allowed,
                                 jnp.where(target_acts, target_allowed,
                                           same_pose[:, None] & same_pose[None, :]))
    return _component_exceptions(permitted)
