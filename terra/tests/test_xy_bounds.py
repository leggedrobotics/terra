import sys
import numpy as np
import jax.numpy as jnp

# Ensure project imports work when running directly
try:
    from terra.utils import compute_polygon_mask
except Exception as e:
    print("Import error: run from project root so 'terra' is importable.")
    raise


def mask_to_numpy(mask: jnp.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.int32)


def build_expected_axis_aligned_mask(map_w: int, map_h: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """
    Build an expected mask for an axis-aligned rectangle covering cells
    x in [x0, x1-1], y in [y0, y1-1]. Mask shape is (map_h, map_w).
    """
    m = np.zeros((map_h, map_w), dtype=np.int32)
    xs0 = max(0, min(map_w, x0))
    xs1 = max(0, min(map_w, x1))
    ys0 = max(0, min(map_h, y0))
    ys1 = max(0, min(map_h, y1))
    m[ys0:ys1, xs0:xs1] = 1
    return m


def assert_mask_equal(name: str, got: np.ndarray, expected: np.ndarray):
    if got.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch {got.shape} != {expected.shape}")
    if not np.array_equal(got, expected):
        # Show a small diff summary
        diff = np.where(got != expected)
        samples = list(zip(diff[0][:5], diff[1][:5]))
        raise AssertionError(f"{name}: values differ at {len(diff[0])} cells, examples={samples}")


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


def test_bounds_check():
    map_w = 8
    map_h = 8

    def valid_bounds(corners: np.ndarray) -> bool:
        corners_j = jnp.array(corners, dtype=jnp.int32)
        log_dtype_shape("corners_bounds", corners_j)
        vb = jnp.all(
            jnp.logical_and(
                corners_j >= jnp.array([0, 0], dtype=jnp.int32),
                corners_j < jnp.array([map_w, map_h], dtype=jnp.int32),
            )
        )
        return bool(vb)

    # Inside
    assert valid_bounds(np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.int32))

    # Exactly on right edge (x==8) -> invalid
    assert not valid_bounds(np.array([[7, 1], [8, 1], [8, 2], [7, 2]], dtype=np.int32))

    # Exactly on bottom edge (y==8) -> invalid
    assert not valid_bounds(np.array([[1, 7], [2, 7], [2, 8], [1, 8]], dtype=np.int32))

    # Negative -> invalid
    assert not valid_bounds(np.array([[-1, 1], [0, 1], [0, 2], [-1, 2]], dtype=np.int32))

    print("OK: bounds check logic dtype/shape and edge behavior correct.")


if __name__ == "__main__":
    try:
        test_polygon_mask_axes_and_edges()
        test_bounds_check()
    except AssertionError as e:
        print("FAILED:", e)
        sys.exit(1)
    print("All tests passed.") 