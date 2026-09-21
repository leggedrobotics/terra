"""Metric geometry must survive storage and include its closed boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.maps_buffer import MapsBuffer
from terra.state import State
from terra.tests.test_foundation_behavior import _pose
from terra.tests.test_reward_v2_contract import _env_config


@pytest.fixture(scope="module")
def geometry_buffer():
    target = np.zeros((1, 1, 64, 64), dtype=np.int8)
    target[0, 0, 31:33, 12:52] = -1
    axes = -97 * np.ones((1, 1, 4, 8), dtype=np.float32)
    axes[0, 0, 0] = [0, -39.282, 1237.383, 31.5, 11.859, 31.5, 51.141, 1.1375]
    borders = -97 * np.ones((1, 1, 64, 3), dtype=np.float32)
    borders[0, 0, 0] = axes[0, 0, 0, :3]
    buffer = MapsBuffer.new(
        maps=jnp.asarray(target), padding_mask=jnp.zeros_like(target),
        trench_axes=jnp.asarray(axes), trench_types=jnp.ones((1, 1), dtype=jnp.int32),
        foundation_border_axes=jnp.asarray(borders),
        foundation_border_types=jnp.ones((1, 1), dtype=jnp.int32),
        dumpability_masks_init=jnp.ones_like(target, dtype=jnp.bool_),
        action_maps=jnp.zeros_like(target),
        distance_maps=jnp.ones_like(target, dtype=jnp.float32),
    )
    return buffer, axes, borders


def test_geometry_storage_preserves_line_endpoints_width_and_sentinels(geometry_buffer):
    buffer, axes, borders = geometry_buffer
    assert buffer.trench_axes.dtype == jnp.float32
    assert buffer.foundation_border_axes.dtype == jnp.float32
    np.testing.assert_array_equal(buffer.trench_axes, axes)
    np.testing.assert_array_equal(buffer.foundation_border_axes, borders)


@jax.jit
def _admitted_fresh_count(state):
    _, fresh_trench, valid = state._fresh_trench_pose_valid_cells()
    return jnp.sum(fresh_trench & valid)


def test_mirrored_and_rotated_poses_share_closed_metric_limit(geometry_buffer):
    buffer, _, _ = geometry_buffer
    for transpose in (False, True):
        target = np.asarray(buffer.maps[0, 0])
        axes = np.asarray(buffer.trench_axes[0, 0]).copy()
        positions = np.array([[28, 57], [35, 57], [27, 57], [36, 57]])
        if transpose:
            target = target.T
            axes[0, [0, 1]] = axes[0, [1, 0]]
            axes[0, [3, 4, 5, 6]] = axes[0, [4, 3, 6, 5]]
            positions = positions[:, ::-1].copy()
        cfg = _env_config()._replace(
            enforce_trench_dig_alignment=True, trench_dig_max_offset_m=2.0,
        )
        state = State.new(
            jax.random.PRNGKey(7), cfg, target, np.zeros_like(target),
            axes, np.int32(1), buffer.foundation_border_axes[0, 0],
            np.int32(-1), np.ones_like(target, dtype=np.bool_),
            np.zeros_like(target), distance_map_override=np.ones_like(target, dtype=np.float32),
        )
        state = _pose(state, base=3 if transpose else 0, loaded=0)
        scalar = [_admitted_fresh_count(_pose(state, position=p)) for p in positions]
        batched = jax.jit(jax.vmap(
            lambda p: _admitted_fresh_count(_pose(state, position=p)),
        ))(jnp.asarray(positions))
        np.testing.assert_array_equal(scalar, [80, 80, 0, 0])
        np.testing.assert_array_equal(batched, scalar)
        # A real millimetre beyond the permitted offset must remain rejected.
        narrower = state._replace(env_cfg=cfg._replace(trench_dig_max_offset_m=1.999))
        for p in positions[:2]:
            assert int(_admitted_fresh_count(_pose(narrower, position=p))) == 0
