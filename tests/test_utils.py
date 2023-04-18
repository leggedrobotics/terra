import unittest
import numpy as np
from src.utils import (
    increase_angle_circular,
    decrease_angle_circular,
    wrap_angle_rad
    )

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
        self.assertTrue(np.allclose(wrap_angle_rad(3 * np.pi), np.pi))
        self.assertTrue(np.allclose(wrap_angle_rad(np.pi), np.pi))
        self.assertTrue(np.allclose(wrap_angle_rad(0), 0))
        self.assertTrue(np.allclose(wrap_angle_rad(2 * np.pi + 1), 1.))


if __name__ == "__main__":
    unittest.main()
