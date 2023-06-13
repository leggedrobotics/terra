import os

import jax.numpy as jnp
import numpy as np
from jax import Array
from tqdm import tqdm

IntMap = jnp.int8
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
    costheta = jnp.cos(anchor_state @ jnp.array([0.0, 0.0, 1.0]))
    sintheta = jnp.sin(anchor_state @ jnp.array([0.0, 0.0, 1.0]))

    R = jnp.array([[costheta, -sintheta], [sintheta, costheta]])
    t = anchor_state[:2]

    # Build the inverse transformation matrix
    R_t = R.T
    T_left = jnp.vstack([R_t, jnp.array([[0.0, 0.0]])])
    T_right = jnp.vstack([(-R_t @ t)[:, None], jnp.array([[1.0]])])
    T = jnp.hstack([T_left, T_right])

    local_coords = T @ jnp.vstack(
        [global_coords, jnp.ones((1, global_coords.shape[1]))]
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
    r = jnp.sqrt(jnp.sum(local_coords**2, axis=0, keepdims=True))
    # theta = jnp.arctan2(local_coords[1], local_coords[0])
    theta = jnp.arctan2(-local_coords[0], local_coords[1])
    # theta = wrap_angle_rad(theta)
    return jnp.vstack([r, theta[None]])


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


def load_maps_from_disk(folder_path: str) -> Array:
    dataset_size = int(os.getenv("DATASET_SIZE", -1))
    maps = []
    for i in tqdm(range(dataset_size), desc="Data Loader"):
        map = np.load(f"{folder_path}/img_{i}.npy")
        maps.append(map)
    print(f"Loaded {dataset_size} maps from disk.")
    return jnp.array(maps, dtype=IntMap)
