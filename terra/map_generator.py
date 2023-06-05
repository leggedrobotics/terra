from enum import IntEnum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from terra.map_utils.terrain_generation import generate_polygonal_bitmap
from terra.utils import IntMap


class MapType(IntEnum):
    SINGLE_TILE = 0
    SQUARE_SINGLE_TRENCH = 1
    RECTANGULAR_SINGLE_TRENCH = 2
    SQUARE_SINGLE_RAMP = 3
    SQUARE_SINGLE_TRENCH_RIGHT_SIDE = 4
    SINGLE_TILE_SAME_POSITION = 5
    SINGLE_TILE_EASY_POSITION = 6
    MULTIPLE_SINGLE_TILES = 7
    MULTIPLE_SINGLE_TILES_WITH_DUMP_TILES = 8
    TWO_SQUARE_TRENCHES_TWO_DUMP_AREAS = 9
    RANDOM_MULTISHAPE = 10


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
        key: jax.random.KeyArray, width: IntMap, height: IntMap, map_params: MapParams
    ) -> "GridMap":
        # jax.debug.print("map_params.type={x}", x=map_params.type)
        map, key = jax.lax.switch(
            map_params.type,
            [
                partial(single_tile, width, height),
                partial(single_square_trench, width, height),
                partial(single_rectangular_trench, width, height),
                partial(single_square_ramp, width, height),
                partial(single_square_trench_right_side, width, height),
                partial(single_tile_same_position, width, height),
                partial(single_tile_easy_position, width, height),
                partial(multiple_single_tiles, width, height),
                partial(multiple_single_tiles_with_dump_tiles, width, height),
                partial(two_square_trenches_two_dump_areas, width, height),
                partial(random_multishape, width, height),
            ],
            key,
            map_params,
        )
        return GridMap(map), key


def _get_generic_rectangular_trench(
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
    return map.astype(IntMap)


def _get_generic_rectangular_ramp(
    width: IntMap,
    height: IntMap,
    x_top_left: IntMap,
    y_top_left: IntMap,
    size_x: IntMap,
    size_y: IntMap,
    orientation: IntMap,
):
    ramp_element = jax.lax.cond(
        (orientation[0] == 0) | (orientation[0] == 2),
        lambda: (jnp.arange(width, dtype=IntMap) - x_top_left - size_x)[:, None].repeat(
            height, -1
        ),
        lambda: (jnp.arange(height, dtype=IntMap) - y_top_left - size_y)[None].repeat(
            width, 0
        ),
    )

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
        ramp_element,
    )

    map = jax.lax.switch(
        orientation[0],
        [lambda: map, lambda: map, lambda: jnp.flipud(map), lambda: jnp.fliplr(map)],
    )

    return map.astype(IntMap)


def single_tile(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
):
    map = jnp.zeros((width, height), dtype=IntMap)
    key, subkey = jax.random.split(key)
    x = jax.random.randint(subkey, (1,), minval=0, maxval=width)
    key, subkey = jax.random.split(key)
    y = jax.random.randint(subkey, (1,), minval=0, maxval=height)
    map = map.at[x, y].set(map_params.depth)
    return map, key


def single_tile_same_position(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
):
    map = jnp.zeros((width, height), dtype=IntMap)
    x = jnp.full((1,), 1)
    # y = jnp.full((1,), 5)
    y = jnp.full((1,), 7)
    map = map.at[x, y].set(map_params.depth)
    return map, key


def single_tile_easy_position(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
):
    map = jnp.zeros((width, height), dtype=IntMap)
    key, subkey = jax.random.split(key)
    x = jax.random.randint(subkey, (1,), minval=0, maxval=width // 2)
    key, subkey = jax.random.split(key)
    y = jax.random.randint(subkey, (1,), minval=height // 2, maxval=height)
    map = map.at[x, y].set(map_params.depth)
    return map, key


def multiple_single_tiles(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
):
    n_tiles = 4  # TODO config
    map = jnp.zeros((width, height), dtype=IntMap)
    key, subkey = jax.random.split(key)
    x = jax.random.randint(subkey, (n_tiles,), minval=0, maxval=width)
    key, subkey = jax.random.split(key)
    y = jax.random.randint(subkey, (n_tiles,), minval=0, maxval=height)
    map = map.at[x, y].set(map_params.depth)
    return map, key


def multiple_single_tiles_with_dump_tiles(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
):
    n_tiles = 4  # TODO config
    map = jnp.zeros((width, height), dtype=IntMap)

    # Dig
    key, subkey = jax.random.split(key)
    x_dig = jax.random.randint(subkey, (n_tiles,), minval=0, maxval=width)
    key, subkey = jax.random.split(key)
    y_dig = jax.random.randint(subkey, (n_tiles,), minval=0, maxval=height)
    map = map.at[x_dig, y_dig].set(map_params.depth)

    # Dump
    key, subkey = jax.random.split(key)
    x_dump = jax.random.randint(subkey, (n_tiles,), minval=0, maxval=width)
    key, subkey = jax.random.split(key)
    y_dump = jax.random.randint(subkey, (n_tiles,), minval=0, maxval=height)
    map = map.at[x_dump, y_dump].set(-map_params.depth)
    return map, key


def single_square_trench(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
) -> Array:
    trench_size_edge_min = map_params.edge_min
    trench_size_edge_max = map_params.edge_max
    trench_depth = map_params.depth

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
    map = _get_generic_rectangular_trench(
        width,
        height,
        x_top_left,
        y_top_left,
        trench_size_edge,
        trench_size_edge,
        trench_depth,
    )
    return map, key


def two_square_trenches_two_dump_areas(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
) -> Array:
    trench_size_edge_min = map_params.edge_min
    trench_size_edge_max = map_params.edge_max
    trench_depth = map_params.depth

    key, subkey = jax.random.split(key)
    trench_size_edge = jax.random.randint(
        subkey, (1,), minval=trench_size_edge_min, maxval=trench_size_edge_max + 1
    )

    maps = []
    # First trench
    for i in range(2):
        key, subkey = jax.random.split(key)
        x_top_left = jax.random.randint(
            subkey, (1,), minval=0, maxval=width - trench_size_edge - 1
        )
        key, subkey = jax.random.split(key)
        y_top_left = jax.random.randint(
            subkey, (1,), minval=0, maxval=height - trench_size_edge - 1
        )
        map = _get_generic_rectangular_trench(
            width,
            height,
            x_top_left,
            y_top_left,
            trench_size_edge,
            trench_size_edge,
            trench_depth,
        )
        maps.append(map)

    # Dump zone 1
    for i in range(2):
        key, subkey = jax.random.split(key)
        x_top_left = jax.random.randint(
            subkey, (1,), minval=0, maxval=width - trench_size_edge - 1
        )
        key, subkey = jax.random.split(key)
        y_top_left = jax.random.randint(
            subkey, (1,), minval=0, maxval=height - trench_size_edge - 1
        )
        map = _get_generic_rectangular_trench(
            width,
            height,
            x_top_left,
            y_top_left,
            trench_size_edge,
            trench_size_edge,
            1,
        )
        maps.append(map)

    map = jnp.zeros_like(maps[0])
    for m in maps:
        map += m
    # Avoid some tiles with more depth or with higher than 1 dump
    map = jnp.where(map < 0, trench_depth, map)
    map = jnp.where(map > 0, 1, map)

    return map, key


def single_square_trench_right_side(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
) -> Array:
    trench_size_edge_min = map_params.edge_min
    trench_size_edge_max = map_params.edge_max
    trench_depth = map_params.depth

    key, subkey = jax.random.split(key)
    trench_size_edge = jax.random.randint(
        subkey, (1,), minval=trench_size_edge_min, maxval=trench_size_edge_max + 1
    )

    key, subkey = jax.random.split(key)
    x_top_left = jax.random.randint(
        subkey, (1,), minval=width // 2, maxval=width - trench_size_edge - 1
    )
    key, subkey = jax.random.split(key)
    y_top_left = jax.random.randint(
        subkey, (1,), minval=0, maxval=height - trench_size_edge - 1
    )
    map = _get_generic_rectangular_trench(
        width,
        height,
        x_top_left,
        y_top_left,
        trench_size_edge,
        trench_size_edge,
        trench_depth,
    )
    return map, key


def single_rectangular_trench(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
) -> Array:
    trench_size_edge_min = map_params.edge_min
    trench_size_edge_max = map_params.edge_max
    trench_depth = map_params.depth

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
    map = _get_generic_rectangular_trench(
        width,
        height,
        x_top_left,
        y_top_left,
        trench_size_edge[0],
        trench_size_edge[1],
        trench_depth,
    )
    return map, key


def single_square_ramp(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
) -> Array:
    edge_min = map_params.edge_min
    edge_max = map_params.edge_max

    key, subkey = jax.random.split(key)
    trench_size_edge = jax.random.randint(
        subkey, (1,), minval=edge_min, maxval=edge_max + 1
    )

    key, subkey = jax.random.split(key)
    x_top_left = jax.random.randint(
        subkey, (1,), minval=0, maxval=width - trench_size_edge - 1
    )
    key, subkey = jax.random.split(key)
    y_top_left = jax.random.randint(
        subkey, (1,), minval=0, maxval=height - trench_size_edge - 1
    )

    key, subkey = jax.random.split(key)
    orientation = jax.random.randint(subkey, (1,), minval=0, maxval=5)
    map = _get_generic_rectangular_ramp(
        width,
        height,
        x_top_left,
        y_top_left,
        trench_size_edge,
        trench_size_edge,
        orientation,
    )
    return map, key


def random_multishape(
    width: IntMap, height: IntMap, key: jax.random.KeyArray, map_params: MapParams
) -> Array:
    # TODO move to config
    map_dict = {
        "shapes": {
            "triangle": 0,
            "trapezoid": 0,
            "rectangle": 1,
            "pentagon": 1,
            "hexagon": 1,
            "L": 1,
            "Z": 1,
            # 'regular_polygon': 6,
        },
        "dimensions": (1000, 1000),
        "max_edge": 400,
        "min_edge": 50,
        "radius": 300,
        "regular_num_sides": [3, 4, 5],
        "scale_factor": 0.7,
        "area_threshold": 1000,
    }

    final_edge_size = width
    # TODO convert following to JIT-compilable code
    image = jax.pure_callback(
        generate_polygonal_bitmap,
        np.zeros((width, width), dtype=np.int8),
        map_dict,
        final_edge_size,
    )
    return jnp.array(image, dtype=IntMap), key
