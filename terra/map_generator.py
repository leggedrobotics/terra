from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# from terra.utils import IntMap
IntMap = jnp.int16  # TODO import


class MapParams(NamedTuple):
    edge_min: IntMap
    edge_max: IntMap
    depth: int = -1  # TODO to config


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

    # @partial(jax.jit, static_argnums=(3, 4))
    @staticmethod
    def procedural_map(
        key: jax.random.KeyArray,
        min_width: IntMap,
        min_height: IntMap,
        max_width: IntMap,
        max_height: IntMap,
        element_edge_min: IntMap,
        element_edge_max: IntMap,
        map_type: int,
    ) -> "GridMap":
        """
        Procedurally generate a map.
        """
        params = MapParams(
            edge_min=element_edge_min,
            edge_max=element_edge_max,
        )
        map, padding_mask, key = jax.lax.switch(
            map_type,
            [
                partial(single_tile, min_width, min_height, max_width, max_height),
                partial(
                    single_square_trench, min_width, min_height, max_width, max_height
                ),
                partial(
                    single_rectangular_trench,
                    min_width,
                    min_height,
                    max_width,
                    max_height,
                ),
                partial(
                    single_square_ramp, min_width, min_height, max_width, max_height
                ),
                partial(
                    single_square_trench_right_side,
                    min_width,
                    min_height,
                    max_width,
                    max_height,
                ),
                partial(
                    single_tile_same_position,
                    min_width,
                    min_height,
                    max_width,
                    max_height,
                ),
                partial(
                    single_tile_easy_position,
                    min_width,
                    min_height,
                    max_width,
                    max_height,
                ),
                partial(
                    multiple_single_tiles, min_width, min_height, max_width, max_height
                ),
                partial(
                    multiple_single_tiles_with_dump_tiles,
                    min_width,
                    min_height,
                    max_width,
                    max_height,
                ),
                partial(
                    two_square_trenches_two_dump_areas,
                    min_width,
                    min_height,
                    max_width,
                    max_height,
                ),
            ],
            key,
            params,
        )
        return map, padding_mask, key


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


def _sample_width_height(
    key: jax.random.KeyArray,
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
):
    key, *subkeys = jax.random.split(key, 3)
    width = jax.random.randint(subkeys[0], (), min_width, max_width, IntMap)
    height = jax.random.randint(subkeys[1], (), min_height, max_height, IntMap)
    # padding_mask = jnp.zeros((max_width, max_height), dtype=IntMap)
    padding_mask = jnp.zeros((40, 40), dtype=IntMap)  # TODO change
    padding_mask = jnp.where(
        jnp.logical_or(
            (jnp.arange(max_width) >= width)[:, None].repeat(max_height, -1),
            (jnp.arange(max_height) >= height)[None].repeat(max_width, 0),
        ),
        1,
        padding_mask,
    )
    padding_mask = padding_mask.astype(IntMap)
    return width, height, padding_mask, key


def single_tile(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
):
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
    map = jnp.zeros((max_width, max_height), dtype=IntMap)
    key, subkey = jax.random.split(key)
    x = jax.random.randint(subkey, (1,), minval=0, maxval=width)
    key, subkey = jax.random.split(key)
    y = jax.random.randint(subkey, (1,), minval=0, maxval=height)
    map = map.at[x, y].set(map_params.depth)
    return map, padding_mask, key


def single_tile_same_position(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
):
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
    map = jnp.zeros((max_width, max_height), dtype=IntMap)
    x = jnp.full((1,), 1)
    # y = jnp.full((1,), 5)
    y = jnp.full((1,), 7)
    map = map.at[x, y].set(map_params.depth)
    return map, padding_mask, key


def single_tile_easy_position(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
):
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
    map = jnp.zeros((max_width, max_height), dtype=IntMap)
    key, subkey = jax.random.split(key)
    x = jax.random.randint(subkey, (1,), minval=0, maxval=width // 2)
    key, subkey = jax.random.split(key)
    y = jax.random.randint(subkey, (1,), minval=height // 2, maxval=height)
    map = map.at[x, y].set(map_params.depth)
    return map, padding_mask, key


def multiple_single_tiles(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
):
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
    n_tiles = 4  # TODO config
    map = jnp.zeros((max_width, max_height), dtype=IntMap)
    key, subkey = jax.random.split(key)
    x = jax.random.randint(subkey, (n_tiles,), minval=0, maxval=width)
    key, subkey = jax.random.split(key)
    y = jax.random.randint(subkey, (n_tiles,), minval=0, maxval=height)
    map = map.at[x, y].set(map_params.depth)
    return map, padding_mask, key


def multiple_single_tiles_with_dump_tiles(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
):
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
    n_tiles = 4  # TODO config
    map = jnp.zeros((max_width, max_height), dtype=IntMap)

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
    return map, padding_mask, key


def single_square_trench(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
) -> Array:
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
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
        max_width,
        max_height,
        x_top_left,
        y_top_left,
        trench_size_edge,
        trench_size_edge,
        trench_depth,
    )
    return map, padding_mask, key


def two_square_trenches_two_dump_areas(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
) -> Array:
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
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
            max_width,
            max_height,
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
            max_width,
            max_height,
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

    return map, padding_mask, key


def single_square_trench_right_side(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
) -> Array:
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
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
        max_width,
        max_height,
        x_top_left,
        y_top_left,
        trench_size_edge,
        trench_size_edge,
        trench_depth,
    )
    return map, padding_mask, key


def single_rectangular_trench(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
) -> Array:
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
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
        max_width,
        max_height,
        x_top_left,
        y_top_left,
        trench_size_edge[0],
        trench_size_edge[1],
        trench_depth,
    )
    return map, padding_mask, key


def single_square_ramp(
    min_width: IntMap,
    min_height: IntMap,
    max_width: IntMap,
    max_height: IntMap,
    key: jax.random.KeyArray,
    map_params: MapParams,
) -> Array:
    width, height, padding_mask, key = _sample_width_height(
        key, min_width, min_height, max_width, max_height
    )
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
        max_width,
        max_height,
        x_top_left,
        y_top_left,
        trench_size_edge,
        trench_size_edge,
        orientation,
    )
    return map, padding_mask, key
