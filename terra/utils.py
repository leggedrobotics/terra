import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from terra.settings import IntLowDim
from terra.settings import Float


def increase_angle_circular(angle: IntLowDim, max_angle: IntLowDim) -> IntLowDim:
    """
    Increases the angle by 1 until max angle. In case of max angle, 0 is returned.

    Args:
        angle: int >= 0
        max_angle: int > 0
    """
    return (angle + 1) % max_angle


def decrease_angle_circular(angle: IntLowDim, max_angle: IntLowDim) -> IntLowDim:
    """
    Decreases the angle by 1 until 0. In case of a negative value, max_angle - 1 is returned.

    Args:
        angle: int >= 0
        max_angle: int > 0
    """
    return (angle + max_angle - 1) % max_angle


def apply_rot_transl(anchor_state: Array, global_coords: Array) -> Array:
    """
    Applies the following transform to every element of global_coords:
    local_coords = [R|t]^-1 global_coords
    where R and t are extracted from anchor state.

    Args:
        - anchor_state: (3, ) Array containing [x, y, theta (rad)]
            Note: this is intended as the local frame expressed in the global frame.
        - global_coords: (2, N) Array containing [x, y]
    Returns:
        - local_coords: (2, N) Array containing local [x, y]
    """
    theta = anchor_state[2]
    costheta, sintheta = jnp.cos(theta), jnp.sin(theta)
    R_t = jnp.array([[costheta, sintheta], [-sintheta, costheta]])

    t = anchor_state[:2]
    neg_Rt_t = -R_t @ t

    # Build the inverse transformation matrix
    T = jnp.block([[R_t, neg_Rt_t[:, None]], [jnp.array([0.0, 0.0, 1.0])]])

    local_coords = jnp.einsum(
        "ij,jk->ik",
        T,
        jnp.vstack([global_coords, jnp.ones((1, global_coords.shape[1]))]),
    )
    return local_coords[:2]


def apply_local_cartesian_to_cyl(local_coords: Array) -> Array:
    """
    Transforms the input array from local cartesian coordinates to
    cyilindrical coordinates.

    Note: this function takes also care of the fact that we use an
    unconventional reference frame (x vertical axis to the bottom,
    y horizontal axis to the right). You can see this in the computation of theta.

    Args:
        - local_coords: (2, N) Array with [x, y] rows
    Returns:
        - cyl_coords: (2, N) Array with [r, theta] rows,
            Note: theta belongs to [-pi, pi]
    """
    x, y = local_coords[0], local_coords[1]
    r = jnp.sqrt(x * x + y * y)
    theta = jnp.arctan2(-x, y)
    return jnp.vstack([r, theta])


def wrap_angle_rad(angle: Float) -> Float:
    """
    Wraps an angle in rad to the interval [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_idx_to_rad(angle: IntLowDim, idx_tot: IntLowDim) -> Float:
    """
    Converts an angle idx (e.g. 4) to an angle in rad, given the max
    angle idx possible (e.g. 8).
    """
    angle = 2.0 * np.pi * angle / idx_tot
    return wrap_angle_rad(angle)


def get_arm_angle_int(
    angle_base, angle_cabin, n_angles_base, n_angles_cabin
) -> IntLowDim:
    """
    Returns the equivalent int angle of the arm, expressed in
    the range of numbers allowed by the cabin angles.
    """
    angles_cabin_base_ratio = round(n_angles_cabin / n_angles_base)
    return (angles_cabin_base_ratio * angle_base + angle_cabin) % n_angles_cabin


def get_distance_point_to_line(p, abc):
    """
    p = Array[x, y]
    abc = Array[A, B, C]
    """
    # Note: this is swapped because in digmap we are computing the axes coefficients
    #   with the opposite convention.
    p_x = p[1]
    p_y = p[0]

    numerator = jnp.abs(abc[0] * p_x + abc[1] * p_y + abc[2])
    denominator = jnp.sqrt(abc[0] ** 2 + abc[1] ** 2)
    distance = numerator / denominator
    return jnp.array([distance])


def get_min_distance_point_to_lines(p, lines, trench_type):
    """
    p = Array[x, y]
    lines = Array[Array[A, B, C]]
    trench_type = int, number of axis of a trench (-1 if not a trench)
    """
    p = p.astype(Float)
    lines = lines.astype(Float)

    def _for_body_it(i, d_min):
        d = get_distance_point_to_line(p, lines[i])
        return jnp.min(
            jnp.concatenate((d, d_min)),
            axis=0,
            keepdims=True,
        ).astype(Float)

    d_min = jax.lax.fori_loop(
        0,
        trench_type,
        _for_body_it,
        jnp.full((1,), 9999.0, dtype=Float),
    )
    return d_min[0]


def get_agent_corners(
    pos_base: Array,
    base_orientation: IntLowDim,
    agent_width: IntLowDim,
    agent_height: IntLowDim,
    angles_base: IntLowDim,
):
    """
    Gets the coordinates of the 4 corners of the agent.
    The function uses a biased rounding strategy to avoid rectangle shrinkage.
    """
    # Determine half dimensions using floor/ceil to properly handle odd dimensions.
    half_width_left = jnp.floor(agent_width / 2.0)
    half_width_right = jnp.ceil(agent_width / 2.0)
    half_height_bottom = jnp.floor(agent_height / 2.0)
    half_height_top = jnp.ceil(agent_height / 2.0)

    # Define corners in local coordinates relative to the center.
    local_corners = jnp.array([
        [-half_width_left, -half_height_bottom],
        [ half_width_right, -half_height_bottom],
        [ half_width_right,  half_height_top],
        [-half_width_left,  half_height_top]
    ])

    # Convert degrees to radians using JAX.
    angle_rad = (base_orientation.astype(jnp.float32) / jnp.array(angles_base, dtype=jnp.float32)) * (2 * jnp.pi)
    cos_a = jnp.cos(angle_rad)
    sin_a = jnp.sin(angle_rad)
    # Build the rotation matrix.
    R = jnp.array([[cos_a, -sin_a],
                [sin_a,  cos_a]])
    R = R.squeeze()

    # Rotate local corners and translate by the center position.
    global_corners_float = (R @ local_corners.T).T + jnp.array(pos_base, dtype=IntLowDim)

    # Bias the rounding: use floor if below the center, ceil otherwise.
    center_arr = jnp.array(pos_base, dtype=IntLowDim)
    biased_corners = jnp.where(
        global_corners_float < center_arr,
        jnp.floor(global_corners_float),
        jnp.ceil(global_corners_float)
    ).astype(IntLowDim)

    return biased_corners


def compute_polygon_mask(corners: Array, map_width: int, map_height: int) -> Array:
    """
    Compute a mask (map_width x map_height) indicating the cells covered
    by the polygon defined by its corners.
    """
    # Create a grid of points.
    xs = jnp.arange(map_height)
    ys = jnp.arange(map_width)
    X, Y = jnp.meshgrid(xs, ys, indexing='xy')
    pts = jnp.stack([Y, X], axis=-1).reshape((-1, 2))  # (N,2) as [y,x]
    edges = jnp.roll(corners, -1, axis=0) - corners  # (4,2)
    diff = pts[None, :, :] - corners[:, None, :]  # (4, N, 2)
    edges_exp = edges[:, None, :]  # (4, 1, 2)
    cross = edges_exp[..., 0] * diff[..., 1] - edges_exp[..., 1] * diff[..., 0]  # (4, N)
    inside = jnp.logical_or(jnp.all(cross > 0, axis=0), jnp.all(cross < 0, axis=0))
    mask = inside.reshape((map_height, map_width))
    return mask
