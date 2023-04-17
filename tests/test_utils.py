import unittest
from src.utils import increase_angle_circular, decrease_angle_circular

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


if __name__ == "__main__":
    unittest.main()
