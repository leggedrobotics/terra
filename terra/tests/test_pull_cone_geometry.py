"""Pull-direction cone: any pull within +-half-angle of the cell-to-base line."""
import unittest

import jax
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from terra.dig_direction import (
    PULL_CONE_DIRECTIONS,
    boundary_pull_details,
    boundary_records_from_mask,
    pull_cone_details,
    pull_stroke_details,
)

cone = jax.jit(pull_cone_details)
full = jax.jit(lambda *args: pull_cone_details(*args, window=None))
stroke = jax.jit(pull_stroke_details)
edges = jax.jit(boundary_pull_details)
THIRTY = np.pi / 6
TOL25 = np.deg2rad(25.0)


def trench(shape=(48, 48), rows=(23, 25), cols=(4, 44)):
    target = np.zeros(shape, bool)
    target[rows[0]:rows[1], cols[0]:cols[1]] = True
    return target


class PullConeTests(unittest.TestCase):
    def test_zero_half_angle_reproduces_the_radial_rule(self):
        rows, cols = np.indices((48, 48))
        target = (np.abs(rows - 24) <= 6) & (np.abs(cols - 22) <= 14)
        target |= (np.abs(rows - cols) <= 1) & (rows < 20)
        records, count = boundary_records_from_mask(target)
        for base in ([9, 9], [24, 3], [40, 33], [24, 22]):
            for precision in (False, True):
                with self.subTest(base=base, precision=precision):
                    allowed, length, error = cone(target, records, count, base, .5714286, 4., 6.5, 2.5,
                                                  0., precision, .6, TOL25)
                    expected, expected_length = stroke(target, records, count, base, .5714286, 4., 6.5, 2.5)
                    edge, aligned, radial_error = edges(target, records, count, base, .5714286, .6, TOL25)
                    if precision:
                        expected = np.asarray(expected) & np.asarray(aligned)
                    np.testing.assert_array_equal(np.asarray(allowed), np.asarray(expected))
                    np.testing.assert_allclose(np.asarray(length), np.asarray(expected_length), atol=2e-4)
                    if precision:
                        band = np.asarray(edge)
                        room = np.asarray(expected_length) >= 2.5 - 1e-5
                        np.testing.assert_allclose(np.asarray(error)[band & room],
                                                   np.asarray(radial_error)[band & room], atol=1e-5)

    def test_cone_admits_oblique_trench_pull_but_not_perpendicular(self):
        target = trench()
        records, count = boundary_records_from_mask(target)
        tile = .5714286
        cell = np.array([24, 20])
        # Base 5.2 m from the cell. A 2-cell trench keeps 2.5 m of room for
        # pulls up to about 15 degrees off its axis, so the cone accepts bases
        # up to about 45 degrees off the axis and still rejects cross pulls.
        for degrees, radial_ok, cone_ok in ((25, False, True), (40, False, True),
                                            (50, False, False), (90, False, False)):
            angle = np.deg2rad(degrees)
            base = np.round(cell + 5.2 / tile * np.array([np.sin(angle), -np.cos(angle)])).astype(int)
            with self.subTest(degrees=degrees, base=base.tolist()):
                radial, _ = stroke(target, records, count, base, tile, 4., 6.5, 2.5)
                allowed, length, _ = cone(target, records, count, base, tile, 4., 6.5, 2.5,
                                          THIRTY, False, .6, TOL25)
                self.assertEqual(bool(radial[tuple(cell)]), radial_ok)
                self.assertEqual(bool(allowed[tuple(cell)]), cone_ok)
                if cone_ok:
                    self.assertGreaterEqual(float(length[tuple(cell)]), 2.5)

    def test_lengths_match_shapely_lines_through_the_annulus(self):
        rows, cols = np.indices((56, 56))
        target = ((rows - 28) ** 2 / 15 ** 2 + (cols - 26) ** 2 / 9 ** 2 <= 1) & ~((rows > 30) & (cols > 30))
        records, count = boundary_records_from_mask(target)
        polygon = Polygon(records[:count, 3:5][:, ::-1]).buffer(0)
        tile, near, far, half = 1.0, 4.0, 14.0, np.deg2rad(20.0)
        angles = half * np.linspace(-1, 1, PULL_CONE_DIRECTIONS)
        annulus = Point(0, 0).buffer(far, 512).difference(Point(0, 0).buffer(near, 512))
        checked = 0
        for base in (np.array([10., 3.]), np.array([28., 47.]), np.array([46., 20.])):
            _, best, _ = full(target, records, count, base, tile, near, far, 2.5, half, False, .6, TOL25)
            ring = Polygon([(c + base[1], r + base[0]) for c, r in np.asarray(annulus.exterior.coords)],
                           [[(c + base[1], r + base[0]) for c, r in np.asarray(hole.coords)]
                            for hole in annulus.interiors])
            for cell in np.argwhere(target)[::3]:
                radius = np.hypot(*(base - cell))
                if not near + .05 < radius < far - .05:
                    continue
                radial = (base - cell) / radius

                def room(angle):
                    c, s = np.cos(angle), np.sin(angle)
                    unit = np.array([radial[0] * c - radial[1] * s, radial[0] * s + radial[1] * c])
                    line = LineString([tuple((cell - 80 * unit)[::-1]), tuple((cell + 80 * unit)[::-1])])
                    cut = polygon.intersection(ring).intersection(line)
                    parts = [cut] if cut.geom_type == 'LineString' else list(getattr(cut, 'geoms', []))
                    own = [part for part in parts if part.distance(Point(cell[1], cell[0])) < 1e-6]
                    return own[0].length if own else 0.0

                # The sampled directions are a lower bound; the exact edge
                # candidates lie inside the cone, so a fine scan bounds above.
                sampled = max(room(angle) for angle in angles)
                scanned = max(room(angle) for angle in np.linspace(-half, half, 161))
                value = float(best[tuple(cell)])
                self.assertGreaterEqual(value, sampled - 2e-2, msg=f'{base} {cell}')
                self.assertLessEqual(value, scanned + 2e-2, msg=f'{base} {cell}')
                checked += 1
        self.assertGreater(checked, 20)

    def test_precision_edge_needs_a_tangent_direction_inside_the_cone(self):
        target = np.zeros((48, 48), bool)
        target[14:34, 10:40] = True
        records, count = boundary_records_from_mask(target)
        tile = .5714286
        cell = np.array([14, 22])  # top edge; tangent runs along columns
        # Base 4.3 m away, outside the edge: the cone turns a 50 degree pull to
        # 20 degrees (within 25); at 55 and 60 degrees it can reach only 25+.
        for degrees, cone_ok in ((40, True), (50, True), (55, False), (60, False)):
            angle = np.deg2rad(degrees)  # pull angle from the edge tangent, base outside
            base = np.round(cell + 4.3 / tile * np.array([-np.sin(angle), -np.cos(angle)])).astype(int)
            with self.subTest(degrees=degrees, base=base.tolist()):
                _, aligned, _ = edges(target, records, count, base, tile, .6, TOL25)
                allowed, _, error = cone(target, records, count, base, tile, 4., 6.5, 2.5,
                                         THIRTY, True, .6, TOL25)
                self.assertFalse(bool(aligned[tuple(cell)]))
                self.assertEqual(bool(allowed[tuple(cell)]), cone_ok)
                if cone_ok:
                    self.assertLessEqual(float(error[tuple(cell)]), TOL25 + 1e-6)

    def test_perpendicular_option_admits_normal_pulls(self):
        target = np.zeros((48, 48), bool)
        target[14:34, 10:40] = True
        records, count = boundary_records_from_mask(target)
        tile = .5714286
        cell = (14, 22)  # top edge; tangent runs along columns, normal along rows

        def base_at(degrees, metres):  # approach angle from the edge tangent, base outside
            angle = np.deg2rad(degrees)
            return np.round(np.array(cell) + metres / tile * np.array([-np.sin(angle), -np.cos(angle)])).astype(int)

        # Radial pull: only the exact approach direction counts.
        for base, perpendicular, expected in ((base_at(90, 4.0), False, False), (base_at(90, 4.0), True, True),
                                              (base_at(45, 4.3), False, False), (base_at(45, 4.3), True, False)):
            with self.subTest(rule='radial', base=base.tolist(), perpendicular=perpendicular):
                _, aligned, error = edges(target, records, count, base, tile, .6, TOL25, perpendicular_ok=perpendicular)
                self.assertEqual(bool(aligned[cell]), expected)
                if perpendicular:
                    self.assertLessEqual(float(error[cell]), np.pi / 4 + 1e-6)
        # +-30 degree cone: tangent reachable from approaches up to 55 degrees,
        # normal (with the option) from 35 degrees, so no approach angle blocks.
        for degrees in (45, 70, 90):
            for perpendicular in (False, True):
                base = base_at(degrees, 4.1)
                with self.subTest(rule='cone', degrees=degrees, perpendicular=perpendicular):
                    allowed, _, error = cone(target, records, count, base, tile, 4., 6.5, 2.5,
                                             THIRTY, True, .6, TOL25, perpendicular_ok=perpendicular)
                    self.assertEqual(bool(allowed[cell]), degrees <= 55 or perpendicular)
                    if bool(allowed[cell]):
                        self.assertLessEqual(float(error[cell]), TOL25 + 1e-6)

    def test_edge_cell_between_samples_uses_its_exact_edge_direction(self):
        # Manual game, slot 17411: from base (24, 45) the edge cell (21, 36)
        # has 2.5 m of room only for pulls within about 2 degrees of parallel
        # to the top edge, between the +10 and +20 degree samples.
        target = np.zeros((64, 64), bool)
        target[21:45, 20:40] = True
        records, count = boundary_records_from_mask(target)
        allowed, best, error = cone(target, records, count, [24, 45], .5714286, 4., 6.5, 2.5,
                                    THIRTY, True, .6, TOL25)
        self.assertTrue(bool(allowed[21, 36]))
        self.assertGreaterEqual(float(best[21, 36]), 2.5)
        self.assertLessEqual(float(error[21, 36]), np.deg2rad(5.0))

    @staticmethod
    def _saved_simplified_rectangle():
        # Preserve the actual step-114 continuous geometry independently of
        # changes to offline contour extraction/simplification defaults.
        target = np.zeros((64, 64), bool)
        target[21:45, 20:40] = True
        vertices = np.array([[44.5, 39.], [44., 19.5], [20.5, 20.], [21., 39.5]], np.float64)
        records = np.full((256, 7), -97., np.float32)
        for index, start in enumerate(vertices):
            end = vertices[(index + 1) % len(vertices)]
            dr, dc = end - start
            a, b = dr / np.hypot(dr, dc), -dc / np.hypot(dr, dc)
            records[index] = [a, b, -(a * start[1] + b * start[0]), *start, *end]
        return target, records, 4

    def test_reach_boundary_intersection_captures_narrow_valid_direction(self):
        # Saved step114 cell(21,23): seven angular samples miss a valid pull
        # peaking at26.181 degrees, where the top boundary meets inner reach.
        target, records, count = self._saved_simplified_rectangle()
        allowed, best, error = cone(target, records, count, [22, 31], .5714286,
                                    4., 6.5, 2.5, THIRTY, True, .6, TOL25,
                                    perpendicular_ok=True)
        self.assertTrue(bool(allowed[21, 23]))
        self.assertAlmostEqual(float(best[21, 23]), 2.522885, delta=2e-4)
        self.assertLessEqual(float(error[21, 23]), TOL25 + 1e-6)
        # The neighboring cell truly has too little room in this saved polygon.
        self.assertFalse(bool(allowed[21, 22]))
        self.assertLess(float(best[21, 22]), 2.5)
        # Raising the required length leaves an interval much narrower than a
        # one-degree grid; the geometric intersection remains a valid witness.
        narrow, _, _ = cone(target, records, count, [22, 31], .5714286,
                             4., 6.5, 2.522, THIRTY, True, .6, TOL25,
                             perpendicular_ok=True)
        self.assertTrue(bool(narrow[21, 23]))

    def test_known_narrow_interval_stays_admissible_when_cone_widens(self):
        target, records, count = self._saved_simplified_rectangle()
        previous_length = 0.
        for degrees in (25., 26.181, 27., 30., 35.):
            allowed, best, _ = cone(target, records, count, [22, 31], .5714286,
                                    4., 6.5, 2.5, np.deg2rad(degrees), True, .6, TOL25,
                                    perpendicular_ok=True)
            self.assertTrue(bool(allowed[21, 23]), degrees)
            self.assertGreaterEqual(float(best[21, 23]) + 1e-5, previous_length)
            previous_length = float(best[21, 23])

    def test_stroke_window_extension_only_widens_the_stroke(self):
        # Manual game, slot 17411: from (42, 46), right-edge cells cannot
        # fit an edge-aligned 2.5 m stroke inside the ordinary reach. Wider
        # stroke room admits them; cells are still only dug in 4.0-6.5 m.
        target = np.zeros((64, 64), bool)
        target[21:45, 20:40] = True
        records, count = boundary_records_from_mask(target)
        args = (target, records, count, [42, 46], .5714286, 4., 6.5, 2.5, THIRTY, True, .6, TOL25)
        plain = cone(*args, perpendicular_ok=True)
        zero = cone(*args, perpendicular_ok=True, stroke_inner_extension_m=0., stroke_outer_extension_m=0.)
        for a, b in zip(plain, zero):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        wide = cone(*args, perpendicular_ok=True, stroke_inner_extension_m=.5, stroke_outer_extension_m=1.)
        for cell in ((36, 39), (37, 39), (38, 39)):
            self.assertFalse(bool(plain[0][cell]), cell)
            self.assertTrue(bool(wide[0][cell]), cell)
        self.assertTrue(np.all(np.asarray(wide[0])[np.asarray(plain[0])]))
        rows, cols = np.indices(target.shape)
        radius = np.hypot(rows - 42, cols - 46) * .5714286
        beyond = target & ((radius < 4.0 - 1e-3) | (radius > 6.5 + 1e-3))
        self.assertFalse(np.any(np.asarray(wide[0])[beyond]))

    def test_window_matches_the_whole_map_and_refuses_reach_beyond_it(self):
        rows, cols = np.indices((64, 64))
        target = ((rows - 30) ** 2 / 20 ** 2 + (cols - 34) ** 2 / 12 ** 2 <= 1) & ~((rows > 33) & (cols < 30))
        records, count = boundary_records_from_mask(target)
        for base in ([2, 3], [30, 34], [61, 60], [20, 52], [0, 63]):
            for precision in (False, True):
                with self.subTest(base=base, precision=precision):
                    args = (target, records, count, base, .5714286, 4., 6.5, 2.5, THIRTY, precision, .6, TOL25)
                    windowed, whole = cone(*args), full(*args)
                    np.testing.assert_array_equal(np.asarray(windowed[0]), np.asarray(whole[0]))
                    np.testing.assert_allclose(np.asarray(windowed[1]), np.asarray(whole[1]), atol=1e-6)
                    band = np.asarray(whole[2]) > 0
                    reach = np.hypot(rows - base[0], cols - base[1]) * .5714286 <= 6.6
                    np.testing.assert_allclose(np.asarray(windowed[2])[reach], np.asarray(whole[2])[reach], atol=1e-6)
                    self.assertTrue(np.all(np.asarray(windowed[2])[band] > 0))
        too_far = cone(target, records, count, [30, 34], .5714286, 4., 8., 2.5, THIRTY, False, .6, TOL25)
        self.assertFalse(np.any(np.asarray(too_far[0]) & target))

    def test_invalid_geometry_or_angle_fails_closed(self):
        target = trench()
        records, count = boundary_records_from_mask(target)
        for table, n, half in ((records, 0, THIRTY), (records[:, :3], count, THIRTY),
                               (records, count, np.nan), (records, count, np.pi / 2)):
            allowed, length, _ = cone(target, table, n, [24, 4], .6, 4., 6.5, 2.5, half, False, .6, TOL25)
            self.assertFalse(np.any(np.asarray(allowed) & target))
            self.assertEqual(float(np.max(np.asarray(length))), 0.0)


if __name__ == '__main__':
    unittest.main()
