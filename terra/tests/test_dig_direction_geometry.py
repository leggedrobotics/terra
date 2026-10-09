import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.dig_direction import (
    MAX_BOUNDARY_SEGMENTS,
    boundary_pull_details,
    boundary_records_from_mask,
)


details = jax.jit(boundary_pull_details)


def finite_records(segments, capacity=64):
    records = np.full((capacity, 7), -97, dtype=np.float32)
    for i, (start, end) in enumerate(segments):
        dr, dc = np.subtract(end, start)
        a, b = dr, -dc
        records[i] = [a, b, -(a * start[1] + b * start[0]), *start, *end]
    return records, len(segments)


class DigDirectionGeometryTests(unittest.TestCase):
    def test_rectangle_keeps_straight_edges_and_chamfer_corners(self):
        # Manual slot 17411: half-cell simplification joined alternating
        # chamfer vertices, tilting all four sides and cutting pull room.
        target = np.zeros((64, 64), dtype=bool)
        target[21:45, 20:40] = True
        for mask in (target, target.T, target[::-1], target[:, ::-1]):
            records, count = boundary_records_from_mask(mask)
            exact, exact_count = boundary_records_from_mask(mask, simplification_tiles=0)
            self.assertEqual(count, 8)
            self.assertEqual(count, exact_count)
            np.testing.assert_array_equal(records, exact)
            vectors = records[:count, 5:7] - records[:count, 3:5]
            long = np.linalg.norm(vectors, axis=1) > 1
            self.assertEqual(int(long.sum()), 4)
            self.assertTrue(np.all(np.any(vectors[long] == 0, axis=1)))

    def test_straight_rectangle_edge_keeps_available_pull_room(self):
        from terra.dig_direction import pull_cone_details

        target = np.zeros((64, 64), dtype=bool)
        target[21:45, 20:40] = True
        records, count = boundary_records_from_mask(target)
        allowed, length, _ = jax.jit(pull_cone_details)(
            target, records, count, jnp.array([22, 31]), 4 / 7,
            4.0, 6.5, 2.5, np.deg2rad(30), True, 0.6,
            np.deg2rad(25), perpendicular_ok=True,
        )
        for cell in ((21, 22), (21, 23)):
            self.assertTrue(bool(allowed[cell]))
            self.assertGreaterEqual(float(length[cell]), 2.5)

    def test_simplification_preserves_target_and_protected_cell_centres(self):
        from shapely.geometry import MultiPolygon, Point, Polygon

        # A fixed half-cell simplification cuts target [4, 5] out of this
        # bent strip. Its complement also checks protected-cell preservation.
        mask = np.array([
            [char == "#" for char in row]
            for row in (
                "##.##.#.", ".###...#", ".##.###.", "...#....",
                "#..###.#", "....###.", "#.#.####", ".##....#",
            )
        ])
        for target in (mask, ~mask, np.rot90(mask), mask[:, ::-1]):
            with self.subTest(target=target.tolist()):
                records, count = boundary_records_from_mask(target, simplification_tiles=0.5)
                rings, start = [], 0
                for i in range(count):
                    if np.array_equal(records[i, 5:7], records[start, 3:5]):
                        rings.append(Polygon(records[start:i + 1, 3:5][:, ::-1]))
                        start = i + 1
                self.assertEqual(start, count)
                shells = [ring for ring in rings if ring.exterior.is_ccw]
                holes = [[] for _ in shells]
                for hole in (ring for ring in rings if not ring.exterior.is_ccw):
                    owners = [i for i, shell in enumerate(shells) if shell.contains(hole)]
                    owner = min(owners, key=lambda i: shells[i].area)
                    holes[owner].append(hole.exterior.coords)
                geometry = MultiPolygon([
                    Polygon(shell.exterior.coords, inner)
                    for shell, inner in zip(shells, holes)
                ])
                for predicate in (geometry.contains, geometry.covers):
                    actual = np.array([
                        predicate(Point(col, row))
                        for row, col in np.ndindex(target.shape)
                    ]).reshape(target.shape)
                    np.testing.assert_array_equal(actual, target)

    def test_all_components_holes_and_single_cell_survive(self):
        target = np.zeros((28, 30), dtype=bool)
        target[3:17, 4:18] = True
        target[8:12, 9:13] = False
        target[20:25, 3:8] = True
        target[22, 25] = True
        records, count = boundary_records_from_mask(target)
        self.assertEqual(records.dtype, np.float32)
        self.assertEqual(records.shape, (MAX_BOUNDARY_SEGMENTS, 7))
        self.assertTrue(np.all(records[count:] == -97))
        # Each stored ring ends where it started, including the tiny island.
        rings, start = [], 0
        for i in range(count):
            if np.array_equal(records[i, 5:7], records[start, 3:5]):
                rings.append(records[start:i + 1])
                start = i + 1
        self.assertEqual(len(rings), 4)
        self.assertEqual(start, count)
        self.assertTrue(any(np.all(np.abs(ring[:, 3:5] - [22, 25]) <= 0.5) for ring in rings))
        edge, _, _ = details(target, records, count, [7, -8], 0.5714286, 0.6, 0.4363323)
        self.assertTrue(edge[3, 10])
        self.assertTrue(edge[7, 10])  # Boundary of the interior hole.
        self.assertTrue(edge[22, 25])
        self.assertFalse(edge[5, 6])

    def test_oblique_sides_allow_parallel_pulls_and_reject_crosswise_pulls(self):
        rows, cols = np.indices((64, 64))
        centre = np.array([31.5, 31.5])
        offsets = np.stack([rows, cols], axis=-1) - centre
        for degrees in (0, 15, 30, 45):
            angle = np.deg2rad(degrees)
            tangent = np.array([np.sin(angle), np.cos(angle)])
            normal = np.array([tangent[1], -tangent[0]])
            along, across = offsets @ tangent, offsets @ normal
            target = (np.abs(along) <= 18) & (np.abs(across) <= 8)
            side = target & (across >= 7) & (np.abs(along) <= 8)
            parallel_base = centre + 8 * normal - 30 * tangent
            crosswise_base = centre + 30 * normal
            for mirrored in (False, True):
                with self.subTest(degrees=degrees, mirrored=mirrored):
                    mask = target[:, ::-1] if mirrored else target
                    side_mask = side[:, ::-1] if mirrored else side
                    parallel = parallel_base.copy()
                    crosswise = crosswise_base.copy()
                    if mirrored:
                        parallel[1] = 63 - parallel[1]
                        crosswise[1] = 63 - crosswise[1]
                    records, count = boundary_records_from_mask(mask)
                    edge, allowed, error = details(mask, records, count, parallel, 0.5714286, 0.6, np.deg2rad(25))
                    checked = side_mask & np.asarray(edge)
                    self.assertGreater(checked.sum(), 8)
                    self.assertTrue(np.all(np.asarray(allowed)[checked]))
                    self.assertTrue(np.all(np.asarray(error)[~np.asarray(edge)] == 0))
                    _, cross_allowed, _ = details(mask, records, count, crosswise, 0.5714286, 0.6, np.deg2rad(25))
                    self.assertFalse(np.any(np.asarray(cross_allowed)[checked]))

    def test_metric_band_corner_or_and_bulk(self):
        target = np.zeros((20, 20), dtype=bool)
        target[5:15, 5:15] = True
        corners = [(4.5, 4.5), (4.5, 14.5), (14.5, 14.5), (14.5, 4.5)]
        records, count = finite_records(list(zip(corners, corners[1:] + corners[:1])))
        for base in ([5, -5], [-5, 5]):
            edge, allowed, _ = details(target, records, count, base, 0.6, 0.6, np.deg2rad(25))
            self.assertTrue(edge[5, 5])
            self.assertTrue(allowed[5, 5])
            self.assertFalse(edge[6, 6])
            self.assertTrue(allowed[10, 10])
        _, allowed, _ = details(target, records, count, [-5, -5], 0.6, 0.6, np.deg2rad(25))
        self.assertFalse(allowed[5, 5])
        narrow, _, _ = details(target, records, count, [5, -5], 0.6, 0.3, 0.0)
        wider, _, _ = details(target, records, count, [5, -5], 0.6, 0.9, 0.0)
        self.assertTrue(narrow[5, 10])  # Exactly 0.3 m to the boundary.
        self.assertFalse(narrow[6, 10])
        self.assertTrue(wider[6, 10])  # Exactly 0.9 m to the boundary.

    def test_infinite_line_extensions_do_not_authorize_unrelated_edges(self):
        target = np.zeros((24, 24), dtype=bool)
        target[15, 10] = True
        records, count = finite_records([
            ((14.5, 1), (14.5, 3)),  # Same infinite line, far finite segment.
            ((12, 10.5), (18, 10.5)),
        ])
        edge, allowed, error = details(target, records, count, [15, 20], 0.6, 0.6, np.deg2rad(25))
        self.assertTrue(edge[15, 10])
        self.assertFalse(allowed[15, 10])
        self.assertAlmostEqual(float(error[15, 10]), np.pi / 2, places=6)

    def test_angle_boundary_reversed_pull_and_zero_length_pull(self):
        target = np.zeros((12, 12), dtype=bool)
        target[5, 5] = True
        records, count = finite_records([((4.5, 0), (4.5, 10))])
        for sign in (-1, 1):
            for degrees, expected in ((15, True), (15.02, False)):
                angle = np.deg2rad(degrees)
                base = np.array([5, 5]) + sign * 6 * np.array([np.sin(angle), np.cos(angle)])
                _, allowed, _ = details(target, records, count, base, 0.6, 0.6, np.deg2rad(15))
                self.assertEqual(bool(allowed[5, 5]), expected)
        _, allowed, _ = details(target, records, count, [5, 5], 0.6, 0.6, np.pi / 2 - 0.001)
        self.assertFalse(allowed[5, 5])
        oblique, n = finite_records([((4, 1), (5, 7))])
        _, allowed, error = details(target, oblique, n, [6, 11], 0.6, 0.6, 0.0)
        self.assertTrue(allowed[5, 5])
        self.assertEqual(float(error[5, 5]), 0.0)

    def test_bad_metadata_fails_closed_empty_targets_and_overflow(self):
        target = np.zeros((12, 12), dtype=np.int8)
        target[4:8, 4:8] = -1
        records, count = boundary_records_from_mask(target)
        corrupt = records.copy()
        corrupt[0, 5:7] = corrupt[0, 3:5]
        inconsistent = records.copy()
        inconsistent[0, 2] += 1
        nonfinite = records.copy()
        nonfinite[0, 0] = np.nan
        for table, n in ((records[:, :3], count), (records, 0), (records, MAX_BOUNDARY_SEGMENTS + 1), (corrupt, count), (inconsistent, count), (nonfinite, count)):
            with self.subTest(columns=table.shape[1], count=n):
                edge, allowed, error = details(target, table, n, [5, -5], 0.6, 0.6, 0.4)
                np.testing.assert_array_equal(edge, target < 0)
                np.testing.assert_array_equal(allowed, target >= 0)
                self.assertTrue(np.all(np.isfinite(error)))
        for width, tolerance in ((0.0, 0.4), (0.6, np.pi / 2)):
            _, allowed, _ = details(target, records, count, [5, -5], 0.6, width, tolerance)
            np.testing.assert_array_equal(allowed, target >= 0)
        empty = np.zeros((12, 12), dtype=np.int8)
        table, n = boundary_records_from_mask(empty)
        self.assertEqual(n, 0)
        edge, allowed, error = details(empty, table, n, [0, 0], 0.6, 0.6, 0.4)
        self.assertFalse(np.any(edge))
        self.assertTrue(np.all(allowed))
        self.assertFalse(np.any(error))
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            boundary_records_from_mask(target, max_segments=2)

        # A bad count in one vmapped lane must remain bounded and fail closed
        # even though batching can evaluate both branches of its predicate.
        _, allowed, _ = jax.jit(jax.vmap(lambda n: boundary_pull_details(
            target, records, n, jnp.array([5, -5]), 0.6, 0.6, 0.4
        )))(jnp.asarray([count, 1_000_000, -5]))
        np.testing.assert_array_equal(allowed[1], target >= 0)
        np.testing.assert_array_equal(allowed[2], target >= 0)


if __name__ == "__main__":
    unittest.main()
