import jax.numpy as jnp
from jax import Array
import numpy as np

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
    return (angle + max_angle -1) % max_angle

def apply_rot_transl(anchor_state: Array, map_global_coords: Array) -> Array:
    pass

def apply_local_cartesian_to_curv(local_coords: Array) -> Array:
    pass

def angle_idx_to_rad(angle: IntLowDim, idx_tot: IntLowDim) -> Float:
    return 2. * np.pi * angle / (idx_tot - 1)

def wrap_angle_rad(angle: Float) -> Float:
    return angle % (2 * np.pi)
