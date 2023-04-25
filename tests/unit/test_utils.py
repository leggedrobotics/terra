import unittest

import numpy as np

from src.utils import apply_local_cartesian_to_cyl
from src.utils import apply_rot_transl
from src.utils import decrease_angle_circular
from src.utils import increase_angle_circular
from src.utils import wrap_angle_rad


class TestUtils(unittest.TestCase):
    def test_increase_angle_circular(self):
        self.assertEqual(increase_angle_circular(0, 3), 1)
        self.assertEqual(increase_angle_circular(1, 2), 0)

    def test_decrease_angle_circular(self):
        self.assertEqual(decrease_angle_circular(0, 3), 2)
        self.assertEqual(decrease_angle_circular(1, 3), 0)
        self.assertEqual(decrease_angle_circular(2, 3), 1)
        self.assertEqual(decrease_angle_circular(1, 55), 0)
        self.assertEqual(decrease_angle_circular(0, 55), 54)
        self.assertEqual(decrease_angle_circular(0, 8), 7)

    def test_wrap_angle_rad(self):
        self.assertTrue(np.allclose(wrap_angle_rad(3 * np.pi), -np.pi))
        print(wrap_angle_rad(np.pi))
        self.assertTrue(np.allclose(wrap_angle_rad(np.pi), -np.pi))
        self.assertTrue(np.allclose(wrap_angle_rad(0), 0))
        self.assertTrue(np.allclose(wrap_angle_rad(2 * np.pi + 1), +1))

    def test_apply_rot_transl(self):
        anchor = np.array([0, 1, np.pi])
        global_coords = np.array([[4, 6], [0, 2]])
        local_coords = apply_rot_transl(anchor, global_coords)
        local_coords_gt = np.array([[-4, -6], [1, -1]])
        self.assertTrue(np.allclose(local_coords, local_coords_gt))

    def test_apply_local_cartesian_to_cyl(self):
        local = np.array([[1, 1], [0, 2]])
        cyl = apply_local_cartesian_to_cyl(local)
        cyl_gt = np.array([[1, np.sqrt(5)], [np.arctan2(-1, 0), np.arctan2(-1, 2)]])
        self.assertTrue(np.allclose(cyl, cyl_gt))


if __name__ == "__main__":
    unittest.main()
