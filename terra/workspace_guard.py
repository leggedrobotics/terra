"""Conservative full-workspace reservations for every native machine pair.

These are simulation envelopes, not physical machine or safety geometry. Both
machines reserve their complete native work sector at EVERY pose, including WAIT,
empty travel and a raised skid bucket. There is no implicit arm-stow state.
The declared stand-off is one native tile; contact at that distance conflicts.

Each body/work component is enclosed by projection intervals on 30 axes. A
separating projection proves clearance; otherwise the candidate is rejected.
Finite axes and convex component bounds can reject some geometrically clear
motions. Continuous sweeps, full selected cell areas, native rounded corners,
and the earlier offline oracle's arc padding are conservatively covered.
No map rasterization or Shapely operation is used in the training hot path.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


_AXIS_ANGLES = np.arange(30, dtype=np.float64) * np.pi / 30
_AXES = np.stack((np.cos(_AXIS_ANGLES), np.sin(_AXIS_ANGLES)), axis=-1)
# Host constants also remain safe if this module is first imported in a trace.
AXES = _AXES.astype(np.float32)
_SIGNED_AXES = np.concatenate((_AXES, -_AXES)).astype(np.float32)
_SIGNED_ANGLES = np.concatenate((_AXIS_ANGLES, _AXIS_ANGLES + np.pi)).astype(np.float32)
_ROUND_SCALE = 1 / np.cos(np.pi / 64)
_ARC_SCALE = 1 / np.cos(np.pi / 180)
_ARC_ERROR = (1 - np.cos(np.pi / 180)) * _ROUND_SCALE
_NUMERIC_PAD_M = 2e-4


class Reservation(NamedTuple):
    # Component order is chassis, full workspace; each has 30 projection axes.
    lower: jnp.ndarray
    upper: jnp.ndarray


def _scalar(value):
    return jnp.asarray(value).reshape(-1)[0]


def _project(points):
    # Clearance must not inherit reduced-precision policy-matmul settings.
    return jnp.matmul(points, _SIGNED_AXES.T, precision=jax.lax.Precision.HIGHEST)


def _delta(before, after, count):
    first, last = _scalar(before).astype(jnp.float32), _scalar(after).astype(jnp.float32)
    return ((last - first + count / 2) % count - count / 2) * (2 * jnp.pi / count)


def _intervals(body, work):
    support = jnp.stack((body, work)) + _NUMERIC_PAD_M
    return Reservation(lower=-support[:, 30:], upper=support[:, :30])


def _arc_cosine(start, width):
    """Maximum direction cosine over the closed counterclockwise arc."""
    relative = jnp.mod(_SIGNED_ANGLES - start[..., None], 2 * jnp.pi)
    endpoints = jnp.maximum(jnp.cos(relative), jnp.cos(relative - width[..., None]))
    return jnp.where(relative <= width[..., None] + 2e-6, 1., endpoints)


def _rotated_points_support(points, delta):
    radius = jnp.linalg.norm(points, axis=-1)
    start = jnp.arctan2(points[:, 1], points[:, 0]) + jnp.minimum(delta, 0.)
    return jnp.max(radius[:, None] * _arc_cosine(start, jnp.abs(delta)), axis=0)


def _workspace_parameters(agent, cfg):
    tile = jnp.asarray(cfg.tile_size, dtype=jnp.float32)
    half_size = jnp.maximum(cfg.agent.width, cfg.agent.height) / 2
    dig_radius = cfg.agent.dig_radius_tiles
    excavator = _scalar(agent.agent_type) == 0
    outer = jnp.where(excavator, .5 + tile * (half_size + dig_radius),
                      tile * (half_size + 1.5 * dig_radius - 2))
    half_angle = (2 * jnp.pi / cfg.agent.angles_cabin) * jnp.where(excavator, 1., 1.2)
    # Enclose the circumscribed <=2-degree sector and circumscribed round
    # half-cell-diagonal buffer used by the independent offline oracle.
    radius = (outer + 1e-5) * _ARC_SCALE
    cell_buffer = (tile / jnp.sqrt(2.) + 2e-5) * _ROUND_SCALE
    return outer, radius, cell_buffer, half_angle + 1e-5


def _work_support(before, after, cfg, delta):
    tile = jnp.asarray(cfg.tile_size, dtype=jnp.float32)
    first = (jnp.asarray(before.pos_base, dtype=jnp.float32) + .5) * tile
    last = (jnp.asarray(after.pos_base, dtype=jnp.float32) + .5) * tile
    centres = jnp.maximum(_project(first), _project(last))
    outer, radius, cell_buffer, half = _workspace_parameters(before, cfg)
    bearing = (_scalar(before.angle_base) * (2 * jnp.pi / cfg.agent.angles_base)
               + _scalar(before.angle_cabin) * (2 * jnp.pi / cfg.agent.angles_cabin)
               + jnp.pi / 2)
    start = bearing + jnp.minimum(delta, 0.) - half
    width = 2 * half + jnp.abs(delta)
    support = radius * jnp.maximum(_arc_cosine(start, width), 0.) + cell_buffer
    rotates = jnp.abs(delta) > 1e-7
    # The oracle's discrete sweep has an extra arc-error buffer. Enclose that
    # too, even though the analytic support already covers the actual arc.
    support += jnp.where(rotates, jnp.sqrt(2.) * (radius + cell_buffer) * _ARC_ERROR, 0.)
    translate_and_rotate = rotates & jnp.any(first != last)
    capsule_radius = jnp.maximum(radius + cell_buffer,
                                 (outer + tile / jnp.sqrt(2.) + 2e-5) * _ROUND_SCALE)
    support = jnp.where(translate_and_rotate, capsule_radius, support)
    return centres + support


def pose_reservation(agent_state, corners, env_cfg):
    """Full stationary reservation; load and bucket posture never waive it."""
    body = jnp.max(_project(jnp.asarray(corners, dtype=jnp.float32) * env_cfg.tile_size), axis=0)
    work = _work_support(agent_state, agent_state, env_cfg, jnp.float32(0.))
    return _intervals(body, work)


def sweep_reservation(before_agent, after_agent, before_corners, after_corners, env_cfg):
    """Reserve the whole native candidate before accepting any state change.

    The caller must retain an accepted first actor's sweep until the whole joint
    round finishes. A conflict must reject the whole candidate state, including
    terrain, payload and all feedback, rather than only canceling its motion.
    """
    tile = jnp.asarray(env_cfg.tile_size, dtype=jnp.float32)
    first = jnp.asarray(before_agent.pos_base, dtype=jnp.float32) * tile
    last = jnp.asarray(after_agent.pos_base, dtype=jnp.float32) * tile
    local_first = jnp.asarray(before_corners, dtype=jnp.float32) * tile - first
    local_last = jnp.asarray(after_corners, dtype=jnp.float32) * tile - last
    delta = _delta(before_agent.angle_base, after_agent.angle_base, env_cfg.agent.angles_base)
    centres = jnp.maximum(_project(first), _project(last))
    # Cover both rounded endpoint shapes over the entire turn, matching the
    # offline oracle's common enclosing shape without constructing polygons.
    body_support = jnp.maximum(_rotated_points_support(local_first, delta),
                               _rotated_points_support(local_last, -delta))
    radius = jnp.maximum(jnp.linalg.norm(local_first, axis=1).max(),
                         jnp.linalg.norm(local_last, axis=1).max())
    rotates = jnp.abs(delta) > 1e-7
    body_support += jnp.where(rotates, jnp.sqrt(2.) * radius * _ARC_ERROR, 0.)
    body_support = jnp.where(rotates & jnp.any(first != last), radius * _ROUND_SCALE, body_support)
    body = centres + body_support
    arm_delta = delta + _delta(before_agent.angle_cabin, after_agent.angle_cabin,
                               env_cfg.agent.angles_cabin)
    work = _work_support(before_agent, after_agent, env_cfg, arm_delta)
    # Wheeled motion follows an arc, whose true heading can change even when
    # both saved headings round to the same bin. Enclose every intermediate
    # orientation with circumradius capsules and add the centre-arc sagitta.
    # Half-cell diagonal covers endpoint rounding off the analytic arc.
    curved = ((_scalar(before_agent.action_type) == 1)
              & (_scalar(before_agent.wheel_angle) != 0)
              & (jnp.any(first != last) | rotates))
    wheel = _scalar(before_agent.wheel_angle) * env_cfg.agent.wheel_step
    turn_radius = jnp.abs(env_cfg.agent.width / (jnp.tan(jnp.deg2rad(wheel)) + 1e-6))
    theta = jnp.minimum(env_cfg.agent.move_tiles / turn_radius, 2 * jnp.pi)
    arc_pad = (turn_radius * (1 - jnp.cos(theta / 2)) + 1 / jnp.sqrt(2.)) * tile
    _, work_radius, cell_pad, _ = _workspace_parameters(before_agent, env_cfg)
    body = jnp.where(curved, centres + radius * _ROUND_SCALE + arc_pad, body)
    work_centres = centres + .5 * tile * jnp.sum(_SIGNED_AXES, axis=1)
    work = jnp.where(curved, work_centres + work_radius + cell_pad + arc_pad, work)
    result = _intervals(body, work)
    # Make endpoint containment explicit. The next round constructs stationary
    # reservations again, so its envelopes must fit the already accepted sweep.
    before = pose_reservation(before_agent, before_corners, env_cfg)
    after = pose_reservation(after_agent, after_corners, env_cfg)
    return Reservation(
        lower=jnp.minimum(result.lower, jnp.minimum(before.lower, after.lower)),
        upper=jnp.maximum(result.upper, jnp.maximum(before.upper, after.upper)),
    )


def component_conflicts(left, right, env_cfg):
    """Contact or <=one-tile clearance, as [left body/work, right body/work]."""
    gap = jnp.maximum(left.lower[:, None, :] - right.upper[None, :, :],
                       right.lower[None, :, :] - left.upper[:, None, :])
    separated = jnp.any(gap > env_cfg.tile_size + 1e-5, axis=-1)
    return ~separated


def reservations_conflict(left, right, env_cfg):
    """Default rule: no body/work component pair may overlap."""
    return jnp.any(component_conflicts(left, right, env_cfg))


def state_reservations(state):
    """Stationary reservations for all fixed slots; caller masks inactive ones."""
    cfg = state.env_cfg
    reservations = [pose_reservation(agent, state._get_agent_corners(
        agent.pos_base, agent.angle_base, cfg.agent.width, cfg.agent.height), cfg)
        for agent in state.agent.agent_states]
    return Reservation(jnp.stack([r.lower for r in reservations]),
                       jnp.stack([r.upper for r in reservations]))


def state_has_conflict(state):
    """Full-envelope reset/final-pose check respecting active fixed slots."""
    from terra.workspace_interactions import stationary_component_exceptions

    reservations = state_reservations(state)
    exceptions = stationary_component_exceptions(state)
    active = state.agent.agent_active
    result = jnp.bool_(False)
    for i in range(len(state.agent.agent_states)):
        for j in range(i + 1, len(state.agent.agent_states)):
            pair_active = (active[i] != 0) & (active[j] != 0)
            a = Reservation(reservations.lower[i], reservations.upper[i])
            b = Reservation(reservations.lower[j], reservations.upper[j])
            result |= pair_active & jnp.any(
                component_conflicts(a, b, state.env_cfg) & ~exceptions[i, j])
    return result & state.env_cfg.workspace_guard_enabled


def _raise_prepared_conflict(conflict):
    if bool(conflict):
        raise ValueError("Prepared fleet reset has unauthorized workspace overlap")


def validate_prepared_workspace(state):
    """Fail at the public prepared-reset boundary, also under jit/vmap."""
    conflict = state_has_conflict(state)

    def fail(value):
        jax.debug.callback(_raise_prepared_conflict, value)
        return jnp.bool_(False)

    # Unlike a vmapped cond, this loop invokes no callback for an all-clear batch.
    jax.lax.while_loop(lambda value: value, fail, conflict)
