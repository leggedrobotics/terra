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


class MapParams(NamedTuple):
    pass


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
        seed: jnp.int32, width: IntMap, height: IntMap, map_params: MapParams
    ) -> "GridMap":
        map = jax.lax.switch(
            map_params.type,
            [
                partial(single_square_trench, width, height),
                partial(single_rectangular_trench, width, height),
                partial(single_ramp, width, height),
            ],
            seed,
            map_params,
        )
        return GridMap(map)


def _get_generic_rectangular_target_map(
    width: IntMap,
    height: IntMap,
    x_top_left: IntMap,
    y_top_left: IntMap,
    size_x: IntMap,
    size_y: IntMap,
    depth: IntMap,
):
    map = jnp.zeros((width, height), dtype=IntMap)
    map = jnp.where(
        jnp.logical_or(
            (
                jnp.logical_or(
                    jnp.arange(width) < x_top_left,
                    jnp.arange(width) > x_top_left + size_x,
                )
            )[:, None].repeat(height, -1),
            (
                jnp.logical_or(
                    jnp.arange(height) < y_top_left,
                    jnp.arange(height) > y_top_left + size_y,
                )
            )[None].repeat(width, 0),
        ),
        map,
        depth,
    )
    return map


def single_square_trench(
    width: IntMap, height: IntMap, seed: jnp.int32, map_params: MapParams
) -> Array:
    trench_size_edge_min = map_params.trench_size_edge_min
    trench_size_edge_max = map_params.trench_size_edge_max
    trench_depth = map_params.trench_depth

    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    trench_size_edge = jax.random.randint(
        subkey, (1,), minval=trench_size_edge_min, maxval=trench_size_edge_max + 1
    )

    key, subkey = jax.random.split(key)
    x_top_left = jax.random.randint(
        subkey, (1,), minval=0, maxval=width - trench_size_edge - 1
    )
    key, subkey = jax.random.split(key)
    y_top_left = jax.random.randint(
        subkey, (1,), minval=0, maxval=height - trench_size_edge - 1
    )
    return _get_generic_rectangular_target_map(
        width,
        height,
        x_top_left,
        y_top_left,
        trench_size_edge,
        trench_size_edge,
        trench_depth,
    )


def single_rectangular_trench(
    width: IntMap, height: IntMap, seed: jnp.int32, map_params: MapParams
) -> Array:
    trench_size_edge_min = map_params.trench_size_edge_min
    trench_size_edge_max = map_params.trench_size_edge_max
    trench_depth = map_params.trench_depth

    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    trench_size_edge = jax.random.randint(
        subkey, (2,), minval=trench_size_edge_min, maxval=trench_size_edge_max + 1
    )

    key, subkey = jax.random.split(key)
    x_top_left = jax.random.randint(
        subkey, (1,), minval=0, maxval=width - trench_size_edge[0] - 1
    )
    key, subkey = jax.random.split(key)
    y_top_left = jax.random.randint(
        subkey, (1,), minval=0, maxval=height - trench_size_edge[1] - 1
    )
    return _get_generic_rectangular_target_map(
        width,
        height,
        x_top_left,
        y_top_left,
        trench_size_edge[0],
        trench_size_edge[1],
        trench_depth,
    )


def single_ramp(
    width: IntMap, height: IntMap, seed: jnp.int32, map_params: MapParams
) -> Array:
    # TODO implement
    return jnp.zeros((width, height), dtype=IntMap)
