from enum import IntEnum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from src.utils import IntMap


class MapType(IntEnum):
    SQUARE_SINGLE_TRENCH = 0
    RECTANGULAR_SINGLE_TRENCH = 1
    SINGLE_RAMP = 2
    # SQUARE_TRENCHES = 3
    # RECTANGULAR_TRENCHES = 4


class GridMap(NamedTuple):
    """
    Clarifications on the map representation.

    The x axis corresponds to the first dimension of the map matrix.
    The y axis to the second.
    The origin is on the top left corner of the map matrix.

    The term "width" is associated with the x direction.
    The term "height" is associated with the y direction.
    """

    map: IntMap

    @property
    def width(self) -> int:
        return self.map.shape[0]

    @property
    def height(self) -> int:
        return self.map.shape[1]

    @staticmethod
    def new(map: Array) -> "GridMap":
        assert len(map.shape) == 2

        return GridMap(map=map)

    @staticmethod
    def dummy_map() -> "GridMap":
        return GridMap.new(jnp.full((1, 1), fill_value=0, dtype=jnp.bool_))

    @staticmethod
    def random_map(
        seed: jnp.int32, map_type: MapType, width: IntMap, height: IntMap
    ) -> "GridMap":
        map = jax.lax.switch(
            map_type,
            [
                partial(single_square_trench, width, height),
                partial(single_rectangular_trench, width, height),
                partial(single_ramp, width, height),
            ],
            seed,
        )
        return GridMap(map)


def single_square_trench(width: IntMap, height: IntMap, seed: jnp.int32) -> Array:
    trench_size = 8  # TODO config

    map = jnp.zeros((width, height), dtype=IntMap)
    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    x_top_left = jax.random.randint(
        subkey, (1,), minval=0, maxval=width - trench_size - 1
    )
    x = (jnp.arange(trench_size) + x_top_left)[None].repeat(trench_size, 0).reshape(-1)

    key, subkey = jax.random.split(key)
    y_top_left = jax.random.randint(
        subkey, (1,), minval=0, maxval=height - trench_size - 1
    )
    y = (
        (jnp.arange(trench_size) + y_top_left)[:, None]
        .repeat(trench_size, 1)
        .reshape(-1)
    )

    map = map.at[x, y].set(-1)
    return map


def single_rectangular_trench(width: IntMap, height: IntMap, seed: jnp.int32) -> Array:
    # TODO implement
    return jnp.zeros((width, height), dtype=IntMap)


def single_ramp(width: IntMap, height: IntMap, seed: jnp.int32) -> Array:
    # TODO implement
    return jnp.zeros((width, height), dtype=IntMap)
