"""Full reservations contain sampled native curved-motion geometry."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.tests.test_relocation_reward_contract import SHAPE, _env_config, _state
from terra.utils import get_agent_corners
from terra.workspace_guard import AXES, pose_reservation, sweep_reservation


@pytest.fixture(scope="module")
def curve_state():
    cfg = _env_config((0,))._replace(workspace_guard_enabled=False,
                                   truck_road_restricted=False)
    with jax.disable_jit():
        return _state(np.zeros(SHAPE, np.int8), env_cfg=cfg)


def assert_inside(points, reservation, component):
    # These sampled points come from the native turn-radius/heading equations,
    # rather than the guard's capsule or support-function implementation.
    projected = np.asarray(points).reshape(-1, 2) @ np.asarray(AXES).T
    assert np.all(projected.min(axis=0) >= np.asarray(reservation.lower[component]) - 5e-5)
    assert np.all(projected.max(axis=0) <= np.asarray(reservation.upper[component]) + 5e-5)


@pytest.mark.parametrize("wheel_degrees", [20, 40])
def test_native_curved_sweeps_enclose_intermediate_bodies_and_full_workspaces(curve_state, wheel_degrees):
    cfg = curve_state.env_cfg
    tile = float(cfg.tile_size)
    width, height = int(cfg.agent.width), int(cfg.agent.height)
    local_corners = np.array([[-(width // 2), -(height // 2)],
                              [(width + 1) // 2, -(height // 2)],
                              [(width + 1) // 2, (height + 1) // 2],
                              [-(width // 2), (height + 1) // 2]], float)
    cell_offsets = tile * np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    with jax.disable_jit():
        for base_bin in (0, 5):
            heading = base_bin * 2 * np.pi / cfg.agent.angles_base
            for sign in (-1, 1):
                wheel_bin = sign * wheel_degrees // int(cfg.agent.wheel_step)
                for forward in (False, True):
                    before_agent = curve_state.agent.agent_states[0]._replace(
                        angle_base=jnp.array([base_bin], jnp.int8),
                        angle_cabin=jnp.zeros((1,), jnp.int8),
                        wheel_angle=jnp.array([wheel_bin], jnp.int8),
                        action_type=jnp.ones((1,), jnp.int8))
                    before = curve_state._set_agent_state_at(0, before_agent)
                    after = before._execute_curved_movement(jnp.array([heading], jnp.float32),
                                                           jnp.bool_(forward))
                    after_agent = after.agent.agent_states[0]
                    assert not np.array_equal(before_agent.pos_base, after_agent.pos_base)
                    # At 20 degrees the real heading changes about15 degrees,
                    # but native quantization records no heading change.
                    expected_delta = (0 if wheel_degrees == 20 else sign * (1 if forward else -1))
                    assert int(after_agent.angle_base[0]) == (base_bin + expected_delta) % 12

                    radius = width / (np.tan(np.deg2rad(sign * wheel_degrees)) + 1e-6)
                    theta = float(cfg.agent.move_tiles) / radius * (1 if forward else -1)
                    phase = np.linspace(0., theta, 129)
                    yaw = heading + phase
                    rotations = np.stack((np.stack((np.cos(yaw), -np.sin(yaw)), -1),
                                          np.stack((np.sin(yaw), np.cos(yaw)), -1)), -2)
                    center_of_rotation = np.asarray(before_agent.pos_base) + radius * np.array(
                        [-np.sin(heading + np.pi / 2), np.cos(heading + np.pi / 2)])
                    relative = np.asarray(before_agent.pos_base) - center_of_rotation
                    arc_rotations = np.stack((np.stack((np.cos(phase), -np.sin(phase)), -1),
                                              np.stack((np.sin(phase), np.cos(phase)), -1)), -2)
                    centers = center_of_rotation + np.einsum("tij,j->ti", arc_rotations, relative)
                    np.testing.assert_array_equal(np.rint(centers[-1]).astype(np.int16), after_agent.pos_base)
                    bodies = (centers[:, None, :] + np.einsum("tij,kj->tki", rotations, local_corners)) * tile
                    before_corners = before._get_agent_corners(before_agent.pos_base, before_agent.angle_base,
                                                               width, height)
                    after_corners = after._get_agent_corners(after_agent.pos_base, after_agent.angle_base,
                                                             width, height)
                    for role in (0, 1, 2):
                        first = before_agent._replace(agent_type=jnp.array([role], jnp.int8))
                        last = after_agent._replace(agent_type=jnp.array([role], jnp.int8))
                        reservation = sweep_reservation(first, last, before_corners, after_corners, cfg)
                        assert_inside(bodies, reservation, 0)
                        assert_inside(np.stack((before_corners, after_corners)) * tile, reservation, 0)
                        outer = (.5 + tile * (max(width, height) / 2 + cfg.agent.dig_radius_tiles)
                                 if role == 0 else tile * (max(width, height) / 2 +
                                                          1.5 * cfg.agent.dig_radius_tiles - 2))
                        half_angle = 2 * np.pi / cfg.agent.angles_cabin * (1. if role == 0 else 1.2)
                        angles = yaw[:, None] + np.pi / 2 + np.linspace(-half_angle, half_angle, 97)
                        # Reserve complete sectors including the centers, with
                        # full-cell padding. Workspace centers have the native
                        # +half-cell offset even while the chassis follows an arc.
                        work_centers = (centers + .5) * tile
                        rim = work_centers[:, None, :] + outer * np.stack((np.cos(angles), np.sin(angles)), -1)
                        sector = np.concatenate((work_centers[:, None, :], rim), axis=1)
                        assert_inside(sector[:, :, None, :] + cell_offsets, reservation, 1)


def test_batched_tracked_sweeps_contain_endpoints_independent_of_model_dot_precision(curve_state):
    # Includes the far-corner WAIT that missed its own chassis by18.8mm on
    # the GPU with default model matmul precision. Keep this test CPU-compatible;
    # running the same file in a CUDA runtime exercises that GPU regression.
    cfg = curve_state.env_cfg
    cases = list(itertools.product(((8, 53), (53, 53)), (0, 2), range(12), (0, 5, 11), range(4)))
    positions, roles, bases, cabins, changes = map(np.asarray, zip(*cases))
    prototype = curve_state.agent.agent_states[0]

    def one(position, role, base, cabin, change):
        first = prototype._replace(pos_base=position.astype(jnp.int16),
            agent_type=role.astype(jnp.int8).reshape(1), angle_base=base.astype(jnp.int8).reshape(1),
            angle_cabin=cabin.astype(jnp.int8).reshape(1), action_type=jnp.zeros((1,), jnp.int8),
            wheel_angle=jnp.zeros((1,), jnp.int8))
        last = first._replace(
            pos_base=first.pos_base + jnp.array([5, 0], jnp.int16) * (change == 1),
            angle_base=(first.angle_base + (change == 2)) % 12,
            angle_cabin=(first.angle_cabin + (change == 3)) % 12)
        before_corners = get_agent_corners(first.pos_base, first.angle_base, 7, 11, 12)
        after_corners = get_agent_corners(last.pos_base, last.angle_base, 7, 11, 12)
        return (pose_reservation(first, before_corners, cfg),
                pose_reservation(last, after_corners, cfg),
                sweep_reservation(first, last, before_corners, after_corners, cfg),
                before_corners, after_corners)

    # Policies may intentionally use reduced dot precision. Geometric checks
    # must carry their own precision setting instead of inheriting that choice.
    with jax.default_matmul_precision("bfloat16"):
        before, after, sweep, before_corners, after_corners = jax.block_until_ready(
            jax.jit(jax.vmap(one))(*(jnp.asarray(x) for x in (positions, roles, bases, cabins, changes))))
    for endpoint, corners in ((before, before_corners), (after, after_corners)):
        assert np.all(np.asarray(sweep.lower) <= np.asarray(endpoint.lower))
        assert np.all(np.asarray(sweep.upper) >= np.asarray(endpoint.upper))
        # Independent host projection also catches inaccurate stationary body
        # bounds; endpoint union alone would hide that second precision error.
        actual = np.asarray(corners, np.float64) * float(cfg.tile_size) @ np.asarray(AXES, np.float64).T
        assert np.all(np.asarray(endpoint.lower[:, 0]) <= actual.min(axis=1))
        assert np.all(np.asarray(endpoint.upper[:, 0]) >= actual.max(axis=1))
