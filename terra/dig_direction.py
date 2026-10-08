"""Finite excavation boundaries and cell-to-base pull directions.

Offline records use integer cell centres in ``[row, col]`` coordinates. Each
record is ``[A, B, C, row0, col0, row1, col1]`` with ``A*col+B*row+C=0``.
The runtime compares the pull with a finite boundary's tangent, modulo pi.
"""

import jax
import jax.numpy as jnp
import numpy as np


BOUNDARY_RECORD_SIZE = 7
MAX_BOUNDARY_SEGMENTS = 256
BOUNDARY_NEAREST_TIE_TILES = 0.25


def boundary_records_from_mask(
    target_or_mask, max_segments: int = MAX_BOUNDARY_SEGMENTS, simplification_tiles: float = 0.5
) -> tuple[np.ndarray, int]:
    """Extract every external boundary and hole, without truncating geometry.

    Boolean input is a dig mask; numeric input is a Terra target map (negative
    cells are excavation). Marching squares puts a straight raster boundary
    half a cell from its cell centres. Simplification removes sub-cell stairs;
    its tolerance is reduced until target centres are strictly inside and
    protected centres strictly outside. It does not recover design geometry
    lost in rasterization.
    """
    from shapely import contains_xy, intersects_xy
    from shapely.geometry import MultiPolygon, Polygon
    from skimage.measure import find_contours

    source = np.asarray(target_or_mask)
    if source.ndim != 2 or not np.all(np.isfinite(source)):
        raise ValueError("Excavation target must be a finite 2D array.")
    if not isinstance(max_segments, (int, np.integer)) or max_segments <= 0:
        raise ValueError("max_segments must be a positive integer.")
    if not np.isfinite(simplification_tiles) or simplification_tiles < 0:
        raise ValueError("simplification_tiles must be finite and nonnegative.")
    dig = source if source.dtype == np.bool_ else source < 0
    records = np.full((max_segments, BOUNDARY_RECORD_SIZE), -97.0, np.float32)
    if not np.any(dig):
        return records, 0

    # Padding closes components that touch the map boundary. High-valued
    # diagonal contacts stay separate, so a tiny island is never discarded.
    contours = find_contours(
        np.pad(dig.astype(np.float32), 1),
        0.5,
        fully_connected="low",
        positive_orientation="low",
    )
    rings = [Polygon((contour - 1.0)[:, ::-1]) for contour in contours]
    shells = [ring for ring in rings if ring.exterior.is_ccw]
    holes_by_shell = [[] for _ in shells]
    for hole in (ring for ring in rings if not ring.exterior.is_ccw):
        owners = [i for i, shell in enumerate(shells) if shell.contains(hole)]
        if not owners:
            raise ValueError("Excavation hole has no containing outer boundary.")
        owner = min(owners, key=lambda i: shells[i].area)
        holes_by_shell[owner].append(hole.exterior.coords)

    geometry = MultiPolygon([
        Polygon(shell.exterior.coords, holes)
        for shell, holes in zip(shells, holes_by_shell)
    ])
    if not geometry.is_valid:
        raise ValueError("Excavation contours do not form valid polygon geometry.")
    # Topology preservation keeps rings/components but can still move an edge
    # across a target or protected cell centre. Such a target would become
    # unreachable by a finite-interval gate, even though its raster says dig.
    rows, cols = np.indices(dig.shape)
    tolerance = float(simplification_tiles)
    while True:
        candidate = geometry.simplify(tolerance, preserve_topology=True)
        if (
            np.array_equal(contains_xy(candidate, cols, rows), dig)
            and np.array_equal(intersects_xy(candidate, cols, rows), dig)
        ):
            geometry = candidate
            break
        if tolerance == 0.0:
            raise ValueError("Excavation contours change raster cell classification.")
        tolerance = tolerance / 2.0 if tolerance > 1e-6 else 0.0
    segments = []
    polygons = (geometry,) if geometry.geom_type == "Polygon" else geometry.geoms
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            points = np.asarray(ring.coords, dtype=np.float64)[:, ::-1]
            for start, end in zip(points[:-1], points[1:]):
                dr, dc = end - start
                length = float(np.hypot(dr, dc))
                if length <= 1e-9:
                    raise ValueError("Excavation boundary contains a zero-length segment.")
                a, b = dr / length, -dc / length
                c = -(a * start[1] + b * start[0])
                segments.append([a, b, c, *start, *end])

    count = len(segments)
    if count > max_segments:
        raise ValueError(
            f"Excavation boundary needs {count} finite segments, exceeding "
            f"max_segments={max_segments}; refusing to truncate geometry."
        )
    if count == 0:
        raise ValueError("Nonempty excavation target has no boundary segments.")
    records[:count] = np.asarray(segments, dtype=np.float32)
    return records, count


def boundary_pull_details(
    target_map, records, count, base_rc, tile_size, width_m, tolerance_rad
):
    """Return ``(edge_mask, allowed, angular_error)`` as 2D JAX arrays.

    Only dig cells within the metric band are constrained. Among their nearest
    finite segments, ties within a quarter-cell accept either tangent at a
    corner. Errors are radians in [0, pi/2], zero outside the edge band. Missing
    or malformed metadata fails closed on all dig cells; empty targets are
    neutral. Two scans avoid a persistent segments-by-height-by-width tensor.
    """
    source = jnp.asarray(target_map)
    if source.ndim != 2:
        raise ValueError("Excavation target must be a 2D array.")
    target = source if source.dtype == jnp.bool_ else source < 0
    records = jnp.asarray(records, dtype=jnp.float32)
    zero = jnp.zeros(target.shape, dtype=jnp.float32)
    right_angle = jnp.float32(jnp.pi / 2)

    def fail_closed():
        return target, ~target, jnp.where(target, right_angle, zero)

    # Legacy tables can be traced by the unused arm of a mode selector. Never
    # reinterpret their infinite lines as finite new-mode geometry.
    if records.ndim != 2 or records.shape[1] != BOUNDARY_RECORD_SIZE or not records.shape[0]:
        return fail_closed()

    count_raw = jnp.asarray(count).reshape(())
    count = count_raw.astype(jnp.int32)
    # A vmapped cond can evaluate measure even for an invalid lane. Bound its
    # work independently of validation, which still rejects the raw count.
    scan_count = jnp.clip(count, 0, records.shape[0])
    base = jnp.asarray(base_rc, dtype=jnp.float32).reshape((2,))
    tile_size = jnp.asarray(tile_size, dtype=jnp.float32)
    width_m = jnp.asarray(width_m, dtype=jnp.float32)
    tolerance_rad = jnp.asarray(tolerance_rad, dtype=jnp.float32)
    starts, ends = records[:, 3:5], records[:, 5:7]
    vectors = ends - starts
    lengths_sq = jnp.sum(vectors * vectors, axis=1)
    normal_lengths = jnp.linalg.norm(records[:, :2], axis=1)
    start_residual = jnp.abs(
        records[:, 0] * starts[:, 1] + records[:, 1] * starts[:, 0] + records[:, 2]
    ) / jnp.maximum(normal_lengths, 1e-6)
    end_residual = jnp.abs(
        records[:, 0] * ends[:, 1] + records[:, 1] * ends[:, 0] + records[:, 2]
    ) / jnp.maximum(normal_lengths, 1e-6)
    valid_rows = (
        jnp.all(jnp.isfinite(records), axis=1)
        & (lengths_sq > 1e-12)
        & (normal_lengths > 1e-6)
        & (start_residual <= 1e-3)
        & (end_residual <= 1e-3)
    )
    declared = jnp.arange(records.shape[0]) < count
    valid_metadata = (
        (count > 0)
        & (count <= records.shape[0])
        & (count_raw == count)
        & jnp.all(~declared | valid_rows)
        & jnp.all(jnp.isfinite(base))
        & jnp.isfinite(tile_size)
        & (tile_size > 0)
        & jnp.isfinite(width_m)
        & (width_m > 0)
        & jnp.isfinite(tolerance_rad)
        & (tolerance_rad >= 0)
        & (tolerance_rad < right_angle)
    )

    def measure():
        rows, cols = jnp.indices(target.shape, dtype=jnp.float32)
        points = jnp.stack([rows, cols], axis=-1)
        pull = base - points
        pull_lengths = jnp.linalg.norm(pull, axis=-1)

        def segment_distance(index):
            delta = points - starts[index]
            projection = jnp.clip(
                jnp.sum(delta * vectors[index], axis=-1)
                / jnp.maximum(lengths_sq[index], 1e-12), 0.0, 1.0
            )
            return jnp.linalg.norm(
                delta - projection[..., None] * vectors[index], axis=-1
            )

        def nearest_segment(index, nearest):
            distance = segment_distance(index)
            return jnp.minimum(nearest, jnp.where(index < count, distance, jnp.inf))

        nearest = jax.lax.fori_loop(
            0, scan_count, nearest_segment, jnp.full(target.shape, jnp.inf)
        )

        def nearest_tangent_error(index, error):
            distance = segment_distance(index)
            owns = (index < count) & (
                distance <= nearest + BOUNDARY_NEAREST_TIE_TILES + 1e-5
            )
            dot = jnp.sum(pull * vectors[index], axis=-1)
            cross = pull[..., 0] * vectors[index, 1] - pull[..., 1] * vectors[index, 0]
            # atan2 retains exactly parallel fp32 pulls at zero tolerance;
            # acos(normalized_dot) can manufacture a small positive error.
            angle = jnp.arctan2(jnp.abs(cross), jnp.abs(dot))
            return jnp.minimum(error, jnp.where(owns, angle, right_angle))

        error = jax.lax.fori_loop(
            0, scan_count, nearest_tangent_error,
            jnp.full(target.shape, right_angle),
        )
        edge = target & (nearest * tile_size <= width_m + 1e-5)
        aligned = (error <= tolerance_rad + 1e-6) & (pull_lengths > 1e-6)
        return edge, ~edge | aligned, jnp.where(edge, error, zero)

    def nonempty():
        return jax.lax.cond(valid_metadata, measure, fail_closed)

    return jax.lax.cond(
        jnp.any(target), nonempty,
        lambda: (jnp.zeros_like(target), jnp.ones_like(target), zero),
    )


def pull_stroke_details(target_map, records, count, base_rc, tile_size,
                        min_radius_m, max_radius_m, min_length_m):
    """Return permission and available continuous stroke length, in metres.

    Intersect the line from each target cell to the base with its containing
    target component, then with the legal radial reach. Holes and gaps end a
    stroke; already excavated cells remain useful stroke space because records
    describe the immutable target. This is a bulk-cut allowance, not a model
    of the remaining ramp height or the bucket's three-dimensional motion.
    Records must come from boundary_records_from_mask: strict cell-center
    classification, half-grid contour vertices, and integer base/cell poses
    exclude collinear boundary chains. Unsupported collinear rays fail closed.
    """
    source = jnp.asarray(target_map)
    target = source if source.dtype == jnp.bool_ else source < 0
    records = jnp.asarray(records, jnp.float32)
    zero = jnp.zeros(target.shape, jnp.float32)
    if records.ndim != 2 or records.shape[1] != BOUNDARY_RECORD_SIZE or not records.shape[0]:
        return ~target, zero
    raw_count = jnp.asarray(count).reshape(())
    count = raw_count.astype(jnp.int32)
    declared = jnp.arange(records.shape[0]) < count
    starts, ends = records[:, 3:5], records[:, 5:7]
    vectors = ends - starts
    # Rings have no duplicated vertices. Find the preceding segment once,
    # outside the per-cell scan, to distinguish crossing a vertex from merely
    # touching it. Counting tangent vertices as exits would shorten valid cuts.
    joins = jnp.all(jnp.abs(starts[:, None] - ends[None, :]) < 1e-6, axis=-1)
    joins &= declared[:, None] & declared[None, :]
    previous = starts[jnp.argmax(joins, axis=1)]
    normal_lengths = jnp.linalg.norm(records[:, :2], axis=-1)
    residual0 = jnp.abs(records[:, 0] * starts[:, 1] + records[:, 1] * starts[:, 0] + records[:, 2])
    residual1 = jnp.abs(records[:, 0] * ends[:, 1] + records[:, 1] * ends[:, 0] + records[:, 2])
    valid_rows = (jnp.all(jnp.isfinite(records), axis=-1)
                  & (jnp.linalg.norm(vectors, axis=-1) > 1e-6)
                  & (normal_lengths > 1e-6)
                  & (residual0 <= normal_lengths * 1e-3)
                  & (residual1 <= normal_lengths * 1e-3)
                  & (jnp.sum(joins, axis=1) == 1))
    base = jnp.asarray(base_rc, jnp.float32).reshape((2,))
    tile = jnp.asarray(tile_size, jnp.float32)
    min_radius = jnp.asarray(min_radius_m, jnp.float32)
    max_radius = jnp.asarray(max_radius_m, jnp.float32)
    min_length = jnp.asarray(min_length_m, jnp.float32)
    valid = ((raw_count == count) & (count > 0) & (count <= records.shape[0])
             & jnp.all(~declared | valid_rows) & jnp.all(jnp.isfinite(base))
             & jnp.isfinite(tile) & (tile > 0)
             & jnp.isfinite(min_radius) & jnp.isfinite(max_radius)
             & (min_radius >= 0) & (max_radius > min_radius)
             & jnp.isfinite(min_length) & (min_length > 0))

    def measure():
        points = jnp.stack(jnp.indices(target.shape, dtype=jnp.float32), axis=-1)
        pull = base - points
        radius_tiles = jnp.linalg.norm(pull, axis=-1)
        unit = pull / jnp.maximum(radius_tiles[..., None], 1e-6)

        def cross(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        def intersect(i, carry):
            lower, upper, forward_count, collinear = carry
            offset = starts[i] - points
            denominator = cross(unit, vectors[i])
            safe_denominator = jnp.where(jnp.abs(denominator) > 1e-7, denominator, 1.0)
            distance = cross(offset, vectors[i]) / safe_denominator
            fraction = cross(offset, unit) / safe_denominator
            interior = (fraction > 1e-6) & (fraction < 1.0 - 1e-6)
            previous_side = cross(unit, previous[i] - points)
            next_side = cross(unit, ends[i] - points)
            vertex_crossing = (jnp.abs(fraction) <= 1e-6) & (previous_side * next_side < -1e-10)
            hit = (jnp.abs(denominator) > 1e-7) & (interior | vertex_crossing)
            lower = jnp.maximum(lower, jnp.where(hit & (distance < -1e-6), distance, -jnp.inf))
            upper = jnp.minimum(upper, jnp.where(hit & (distance > 1e-6), distance, jnp.inf))
            forward_count += (hit & (distance > 1e-6)).astype(jnp.int32)
            collinear |= (jnp.abs(denominator) <= 1e-7) & (jnp.abs(cross(offset, unit)) <= 1e-6)
            return lower, upper, forward_count, collinear

        lower, upper, forward_count, collinear = jax.lax.fori_loop(
            0, jnp.clip(count, 0, records.shape[0]), intersect,
            (jnp.full(target.shape, -jnp.inf), jnp.full(target.shape, jnp.inf),
             jnp.zeros(target.shape, jnp.int32), jnp.zeros(target.shape, jnp.bool_)),
        )
        radius = radius_tiles * tile
        lower = jnp.maximum(lower * tile, radius - max_radius)
        upper = jnp.minimum(upper * tile, radius - min_radius)
        inside = (forward_count % 2 == 1) & (radius_tiles > 1e-6) & ~collinear
        in_reach = (radius >= min_radius - 1e-5) & (radius <= max_radius + 1e-5)
        length = jnp.where(target & inside & in_reach, jnp.maximum(upper - lower, 0.0), 0.0)
        return ~target | (length >= min_length - 1e-5), length

    return jax.lax.cond(valid, measure, lambda: (~target, zero))
