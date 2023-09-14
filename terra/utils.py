import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# from terra.config import EnvConfig

IntMap = jnp.int16
INTMAP_MAX = jnp.iinfo(IntMap).max

IntLowDim = jnp.int8
INTLOWDIM_MAX = jnp.iinfo(IntLowDim).max

Float = jnp.float32


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
    T = jnp.block([[R_t, neg_Rt_t[:, None]], [jnp.array([0., 0., 1.])]])

    local_coords = jnp.einsum('ij,jk->ik', T, jnp.vstack([global_coords, jnp.ones((1, global_coords.shape[1]))]))
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
    r = jnp.sqrt(x*x + y*y)
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
    # TODO change the map convention such that this is swapped...
    p_x = p[1]
    p_y = p[0]
    # p_x = p[0]
    # p_y = p[1]

    # jax.debug.print("\nabc = {x}\n", x=abc)

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
        # jax.debug.print("p = {x}", x=p)
        # jax.debug.print("i={x}, d={y}", x=i, y=d)
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

    # jax.debug.print("d_min = {x}", x=d_min)
    return d_min[0]
