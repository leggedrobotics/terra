"""Physical-space claims of the coarse bulk stroke allowance."""
import unittest
import jax
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge
from terra.dig_direction import boundary_records_from_mask, pull_stroke_details

stroke = jax.jit(pull_stroke_details)


def polygon_records(vertices, capacity=128):
    records = np.full((capacity, 7), -97, np.float32)
    for i, (a, b) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
        dr, dc = np.subtract(b, a)
        records[i] = [dr, -dc, -(dr*a[1]-dc*a[0]), *a, *b]
    return records, len(vertices)


class PullStrokeTests(unittest.TestCase):
    def test_straight_trench_requires_longitudinal_room_and_respects_reach(self):
        target = np.zeros((40, 40), bool)
        target[19:21, 4:36] = True
        records, count = boundary_records_from_mask(target)
        along, length = stroke(target, records, count, [19, 8], .6, 4, 6.5, 2.5)
        self.assertTrue(along[19, 16])
        self.assertAlmostEqual(float(length[19, 16]), 2.5, places=5)
        across, length = stroke(target, records, count, [11, 20], .6, 4, 6.5, 2.5)
        self.assertFalse(across[19, 20])
        self.assertLess(float(length[19, 20]), 1.3)
        insufficient, _ = stroke(target, records, count, [19, 8], .6, 4, 6.4, 2.5)
        self.assertFalse(np.any(np.asarray(insufficient) & target))

    def test_junction_cross_cut_is_allowed_but_other_branch_is_not(self):
        target = np.zeros((40, 40), bool)
        target[19:21, 3:37] = True
        target[7:33, 20:22] = True
        records, count = boundary_records_from_mask(target)
        allowed, _ = stroke(target, records, count, [19, 12], .6, 4, 6.5, 2.5)
        self.assertTrue(allowed[19, 21])  # Both branches share this cutting room.
        self.assertFalse(allowed[24, 21])  # Does not borrow the junction's chord.

    def test_gap_and_hole_cannot_be_bridged(self):
        target = np.zeros((40, 40), bool)
        target[18:23, 8:33] = True
        target[18:23, 19] = False
        records, count = boundary_records_from_mask(target)
        allowed, length = stroke(target, records, count, [20, 10], .6, 4, 6.5, 2.5)
        self.assertFalse(allowed[20, 18])
        self.assertLess(float(length[20, 18]), 1.2)
        target[18, 19] = target[22, 19] = True  # Turn the slit into an inner hole.
        records, count = boundary_records_from_mask(target)
        allowed, length = stroke(target, records, count, [20, 10], .6, 4, 6.5, 2.5)
        self.assertFalse(allowed[20, 18])
        self.assertLess(float(length[20, 18]), 1.2)

    def test_tangent_vertex_does_not_truncate_and_matches_independent_geometry(self):
        # The inward V touches row10 atcol15. A line alongrow10 continues
        # through the containing polygon instead of stopping at that touch.
        vertices = [(5, 5), (5, 12), (10, 15), (5, 18), (5, 30), (20, 30), (20, 5)]
        records, count = polygon_records(vertices)
        polygon = Polygon([(c, r) for r, c in vertices])
        rr, cc = np.indices((32, 36))
        target = np.array([polygon.contains(Point(c, r)) for r, c in zip(rr.flat, cc.flat)]).reshape(rr.shape)
        allowed, length = stroke(target, records, count, [10, 0], 1., 0., 100., 20.)
        self.assertTrue(allowed[10, 20])
        self.assertAlmostEqual(float(length[10, 20]), 25., places=5)
        # Compare arbitrary oblique rays with Shapely's connected intersection.
        base = np.array([2., 1.])
        _, lengths = stroke(target, records, count, base, 1., 0., 100., 2.5)
        for p in ([12, 9], [11, 20], [18, 25], [7, 8]):
            direction = (base - p) / np.linalg.norm(base-p)
            endpoints = [np.array(p) - 100*direction, np.array(p) + 100*direction]
            cut = polygon.intersection(LineString([(q[1],q[0]) for q in endpoints]))
            parts = [cut] if cut.geom_type == 'LineString' else list(cut.geoms)
            component = [part for part in parts if part.distance(Point(p[1],p[0])) < 1e-7][0]
            expected = component.length  # All geometry lies beyond this external base.
            self.assertAlmostEqual(float(lengths[tuple(p)]), expected, places=4)

    def test_missing_geometry_and_invalid_lengths_fail_closed(self):
        target = np.zeros((16,16), bool); target[3:13,3:13] = True
        records,count = boundary_records_from_mask(target)
        for table,n,length in ((records,0,2.5),(records,10**6,2.5),(records[:,:3],count,2.5),(records,count,np.nan)):
            allowed,_ = stroke(target, table,n,[5,0],1.,0.,10.,length)
            self.assertFalse(np.any(np.asarray(allowed)&target))

    def test_custom_collinear_boundary_chain_fails_closed(self):
        # This custom polygon is outside the builder's strict half-grid
        # contract: row 5 overlaps a boundary chain. Ignoring that chain used
        # to return 19 m from the left and 0 m from the right, although its
        # actual connected intersection is only 7 m. Reject such rays.
        vertices = [(1, 1), (1, 6), (5, 6), (5, 8), (8, 8), (8, 1)]
        records, count = polygon_records(vertices)
        polygon = Polygon([(col, row) for row, col in vertices])
        target = np.array([
            polygon.contains(Point(col, row))
            for row, col in np.ndindex((12, 12))
        ]).reshape(12, 12)
        for base in ([5, 0], [5, 11]):
            with self.subTest(base=base):
                allowed, lengths = stroke(target, records, count, base, 1., 0., 20., 8.)
                self.assertFalse(allowed[5, 3])
                self.assertEqual(float(lengths[5, 3]), 0.)

    def test_rotated_raster_intervals_match_geometry_at_discrete_base_poses(self):
        rows, cols = np.indices((48, 48))
        offset = np.stack([rows - 24, cols - 24], axis=-1)
        tile, near, far = .6, 4., 6.5
        checked = 0
        for degrees in (0, 17, 33, 61):
            theta = np.deg2rad(degrees)
            tangent = np.array([np.sin(theta), np.cos(theta)])
            normal = np.array([tangent[1], -tangent[0]])
            target = (np.abs(offset @ tangent) <= 15) & (np.abs(offset @ normal) <= 3)
            records, count = boundary_records_from_mask(target)
            polygon = Polygon(records[:count, 3:5][:, ::-1])
            for base in (np.array([8, 9]), np.array([23, 7]), np.array([41, 40])):
                with self.subTest(degrees=degrees, base=base.tolist()):
                    allowed, lengths = stroke(target, records, count, base, tile, near, far, 2.5)
                    radii = np.hypot(rows-base[0], cols-base[1]) * tile
                    candidates = np.argwhere(target & (radii >= near) & (radii <= far))
                    if len(candidates) > 12:
                        candidates = candidates[np.linspace(0, len(candidates)-1, 12, dtype=int)]
                    for p in candidates:
                        direction = (p-base) / np.linalg.norm(p-base)
                        endpoints = [base + radius/tile * direction for radius in (near, far)]
                        cut = polygon.intersection(LineString([q[::-1] for q in endpoints]))
                        if cut.geom_type == "MultiLineString":
                            cut = linemerge(cut)
                        parts = [cut] if cut.geom_type == "LineString" else list(cut.geoms)
                        point = Point(p[1], p[0])
                        containing = [part for part in parts if part.distance(point) < 1e-7]
                        self.assertEqual(len(containing), 1)
                        expected = containing[0].length * tile
                        self.assertAlmostEqual(float(lengths[tuple(p)]), expected, places=4)
                        self.assertEqual(bool(allowed[tuple(p)]), expected >= 2.5-1e-5)
                        checked += 1
        self.assertGreater(checked, 60)

if __name__ == '__main__':
    unittest.main()
