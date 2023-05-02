""""
Original source: https://github.com/RoboEden/jux
"""
from enum import IntEnum
from typing import NamedTuple

from jax import Array
from jax import lax
from jax import numpy as jnp

from terra.noise.simplexnoise import SimplexNoise


class SymmetryType(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    ROTATIONAL = 2
    ANTI_DIAG = 3
    DIAG = 4

    @classmethod
    def from_lux(cls, lux_symmetry: str) -> "SymmetryType":
        idx = ["horizontal", "vertical", "rotational", "/", "\\"].index(lux_symmetry)
        return cls(idx)

    def to_lux(self) -> str:
        return ["horizontal", "vertical", "rotational", "/", "\\"][self]


def symmetrize(x: Array, symmetry: SymmetryType = SymmetryType.VERTICAL):
    # In place operation to average along the symmetry.
    def _horizontal(x):
        x = x + jnp.flipud(x)
        return x

    def _vertical(x):
        x = x + jnp.fliplr(x)
        return x

    def _rotational(x):
        x = x + jnp.rot90(jnp.rot90(x))
        return x

    def _anti_diag(x):
        x = x + jnp.flip(x).T
        return x

    def _diag(x):
        x = x + x.T
        return x

    x = lax.switch(
        symmetry, [_horizontal, _vertical, _rotational, _anti_diag, _diag], x
    )
    return lax.cond(x.dtype.kind == "i", lambda x: jnp.floor(x / 2), lambda x: x / 2, x)


class SymmetryNoise(NamedTuple):
    seed: jnp.int32
    octaves: jnp.int32
    symmetry: SymmetryType

    def noise(self, x: Array, y: Array, frequency: jnp.float32 = 1):
        """
        x: int [1, N]
        y: int [1, N]
        """
        x, y = jnp.meshgrid(x, y)
        total = SimplexNoise.dispatch_noise2(x, y, octaves=self.octaves)
        total = symmetrize(total, self.symmetry)
        # Normalize between [0, 1]
        total = total - jnp.amin(total)
        total = lax.cond(
            jnp.allclose(jnp.amax(total), 0.0),
            lambda total: total,
            lambda total: total / jnp.amax(total),
            total,
        )
        return total
