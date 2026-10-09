"""Finite-grid working room for one DO; no bucket trajectory or kinematics.

A metric rectangle must fit wholly in this DO's eligible fresh cells plus
already-dug cells. The returned fresh mask is the union of complete valid
rectangles, so any fresh soil used as room is also included in this DO.
"""

import jax
import jax.numpy as jnp


# Seven cells span the intended 1 m by 1.3 m strips at Terra's 0.571 m tiles.
# Larger settings must fit this stencil; unsupported dimensions fail closed.
_KERNEL_RADIUS = 3


def working_strip_mask(fresh_mask, dug_mask, directions_rc, tile_size,
                      min_length_m, width_m):
    """Return eligible fresh cells covered by a fully supported working strip.

    ``directions_rc`` is [N,2] in row/column coordinates. Rectangles are
    undirected; callers can supply the cabin direction and cone limits. Four
    center phases (0 or half a tile per axis) allow a 1 m cardinal strip to
    occupy two cells longitudinally and three across a 1.3 m bucket width.

    Every cell whose interior intersects the rectangle must be fresh or dug.
    Obstacles, protected ground, chassis and heading/reach constraints must
    already be excluded by the caller. Outside-map support is always false.
    """
    fresh = jnp.asarray(fresh_mask, jnp.bool_)
    support = fresh | jnp.asarray(dug_mask, jnp.bool_)
    directions = jnp.asarray(directions_rc, jnp.float32)
    tile = jnp.asarray(tile_size, jnp.float32)
    length = jnp.asarray(min_length_m, jnp.float32)
    width = jnp.asarray(width_m, jnp.float32)
    norm = jnp.linalg.norm(directions, axis=-1)
    unit = directions / jnp.where(norm[:, None] > 0, norm[:, None], jnp.float32(1.0))
    normal = jnp.stack((-unit[:, 1], unit[:, 0]), axis=-1)
    extents = .5 * (length * jnp.abs(unit) + width * jnp.abs(normal))
    valid = (jnp.isfinite(tile) & (tile > 0) & jnp.isfinite(length) & (length > 0)
             & jnp.isfinite(width) & (width > 0))
    valid_direction = (jnp.all(jnp.isfinite(directions), axis=-1) & (norm > 0)
                       & jnp.all(extents <= _KERNEL_RADIUS * tile, axis=-1))

    indices = jnp.arange(-_KERNEL_RADIUS, _KERNEL_RADIUS + 1, dtype=jnp.float32)
    offsets = jnp.stack(jnp.meshgrid(indices, indices, indexing='ij'), axis=-1)
    phases = jnp.asarray(((0., 0.), (0., .5), (.5, 0.), (.5, .5)), jnp.float32)

    def templates_for_direction(direction, across, extent, direction_ok):
        def template(phase):
            relative = (offsets - phase) * tile
            # Separating-axis test between each tile square and the oriented
            # rectangle. Strict overlap avoids requiring merely touching cells.
            tolerance = tile * jnp.float32(1e-5)
            axes_overlap = jnp.all(jnp.abs(relative) < extent + tile / 2 - tolerance, axis=-1)
            along = jnp.abs(jnp.sum(relative * direction, axis=-1))
            lateral = jnp.abs(jnp.sum(relative * across, axis=-1))
            cell_along = tile / 2 * jnp.sum(jnp.abs(direction))
            cell_across = tile / 2 * jnp.sum(jnp.abs(across))
            return (axes_overlap & (along < length / 2 + cell_along - tolerance)
                    & (lateral < width / 2 + cell_across - tolerance)
                    & valid & direction_ok)
        return jax.vmap(template)(phases)

    templates = jax.vmap(templates_for_direction)(unit, normal, extents, valid_direction)
    templates = templates.reshape((-1, 2 * _KERNEL_RADIUS + 1, 2 * _KERNEL_RADIUS + 1))
    source = support.astype(jnp.float32)[None, :, :, None]

    def convolution(values, kernel):
        return jax.lax.conv_general_dilated(
            values, kernel[:, :, None, None], window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

    def covered_by_template(template):
        kernel = template.astype(jnp.float32)
        size = kernel.sum()
        count = convolution(source, kernel)
        fits = (count >= size - .5) & (size > 0)
        # Reflect the footprint for dilation: cover the cells used by each fit.
        covered = convolution(fits.astype(jnp.float32), kernel[::-1, ::-1])
        return covered[0, :, :, 0] > 0

    return fresh & jnp.any(jax.vmap(covered_by_template)(templates), axis=0)
