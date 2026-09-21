import sys
import numpy as np
import jax.numpy as jnp
import jax
import pytest

from terra.tests.test_foundation_behavior import foundation, _map, _pose

# Ensure project imports work when running directly
try:
    from terra.utils import compute_polygon_mask, get_agent_corners
except Exception:
    print("Import error: run from project root so 'terra' is importable.")
    raise


def mask_to_numpy(mask: jnp.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.int32)


def build_expected_axis_aligned_mask(
    map_w: int, map_h: int, x0: int, y0: int, x1: int, y1: int
) -> np.ndarray:
    """
    Build an expected mask for an axis-aligned rectangle covering cells
    x in [x0, x1-1], y in [y0, y1-1]. Terra stores x as rows and y as
    columns, so the mask shape is (map_w, map_h).
    """
    m = np.zeros((map_w, map_h), dtype=np.int32)
    xs0 = max(0, min(map_w, x0))
    xs1 = max(0, min(map_w, x1))
    ys0 = max(0, min(map_h, y0))
    ys1 = max(0, min(map_h, y1))
    m[xs0:xs1, ys0:ys1] = 1
    return m


def assert_mask_equal(name: str, got: np.ndarray, expected: np.ndarray):
    if got.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch {got.shape} != {expected.shape}")
    if not np.array_equal(got, expected):
        # Show a small diff summary
        diff = np.where(got != expected)
        samples = list(zip(diff[0][:5], diff[1][:5]))
        raise AssertionError(
            f"{name}: values differ at {len(diff[0])} cells, examples={samples}"
        )


def log_dtype_shape(label: str, arr: jnp.ndarray):
    print(f"{label}: dtype={arr.dtype}, shape={arr.shape}")


def test_polygon_mask_axes_and_edges():
    map_w = 8
    map_h = 8

    # 1) Center 2x2 square: (2,2)-(4,4)
    corners = jnp.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=jnp.int32)
    log_dtype_shape("corners_center", corners)
    mask_j = compute_polygon_mask(corners, map_w, map_h)
    log_dtype_shape("mask_center", mask_j)
    mask = mask_to_numpy(mask_j)
    expected = build_expected_axis_aligned_mask(map_w, map_h, 2, 2, 4, 4)
    assert_mask_equal("center_square", mask, expected)

    # 2) Touching right edge: (7,2)-(8,4) should only fill x=7
    corners = jnp.array([[7, 2], [8, 2], [8, 4], [7, 4]], dtype=jnp.int32)
    log_dtype_shape("corners_right", corners)
    mask_j = compute_polygon_mask(corners, map_w, map_h)
    log_dtype_shape("mask_right", mask_j)
    mask = mask_to_numpy(mask_j)
    expected = build_expected_axis_aligned_mask(map_w, map_h, 7, 2, 8, 4)
    assert_mask_equal("right_edge_square", mask, expected)

    # 3) Touching bottom edge: (2,7)-(4,8) should only fill y=7
    corners = jnp.array([[2, 7], [4, 7], [4, 8], [2, 8]], dtype=jnp.int32)
    log_dtype_shape("corners_bottom", corners)
    mask_j = compute_polygon_mask(corners, map_w, map_h)
    log_dtype_shape("mask_bottom", mask_j)
    mask = mask_to_numpy(mask_j)
    expected = build_expected_axis_aligned_mask(map_w, map_h, 2, 7, 4, 8)
    assert_mask_equal("bottom_edge_square", mask, expected)

    # 4) Top-left corner: (0,0)-(1,1)
    corners = jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=jnp.int32)
    log_dtype_shape("corners_tl", corners)
    mask_j = compute_polygon_mask(corners, map_w, map_h)
    log_dtype_shape("mask_tl", mask_j)
    mask = mask_to_numpy(mask_j)
    expected = build_expected_axis_aligned_mask(map_w, map_h, 0, 0, 1, 1)
    assert_mask_equal("top_left_pixel", mask, expected)

    print("OK: compute_polygon_mask axis order, dtype, and edge handling look correct.")


def test_actual_odd_agent_footprint_preserves_width_and_height():
    map_w = 20
    map_h = 16
    agent_w = 5
    agent_h = 3
    pos_base = jnp.array([8, 8], dtype=jnp.int32)

    corners = get_agent_corners(
        pos_base=pos_base,
        base_orientation=jnp.array(0, dtype=jnp.int32),
        agent_width=jnp.array(agent_w, dtype=jnp.int32),
        agent_height=jnp.array(agent_h, dtype=jnp.int32),
        angles_base=jnp.array(8, dtype=jnp.int32),
    )
    mask = mask_to_numpy(compute_polygon_mask(corners, map_w, map_h))

    expected = build_expected_axis_aligned_mask(
        map_w,
        map_h,
        x0=6,
        y0=7,
        x1=11,
        y1=10,
    )
    assert_mask_equal("actual_odd_agent_footprint", mask, expected)
    assert int(mask.sum()) == agent_w * agent_h


def test_bounds_check(foundation):
    valid = jax.jit(lambda state, corners: state._is_valid_move(corners))
    # Each real 7x11 footprint touches exactly one continuous map edge.
    for position, outward in (
        ((3, 32), (-1, 0)), ((60, 32), (1, 0)),
        ((32, 5), (0, -1)), ((32, 58), (0, 1)),
    ):
        state = _pose(foundation, position=position, base=0)
        corners = state._get_agent_corners(jnp.asarray(position), jnp.array([0]), 7, 11)
        footprint = np.asarray(state._current_base_footprint_mask())
        assert int(footprint.sum()) == 77
        assert bool(valid(state, corners))
        assert not bool(valid(state, corners + jnp.asarray(outward)))
        # Touching the edge permits no exception for the occupied boundary cells.
        cells = np.argwhere(footprint)
        edge_cell = cells[np.argmax(cells @ np.asarray(outward))]
        for field, height in (("action_map", 1), ("action_map", -1), ("static_traversability_base", 1)):
            blocked = np.zeros((64, 64), dtype=np.int8)
            blocked[tuple(edge_cell)] = height
            assert not bool(valid(_map(state, field, blocked), corners))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
