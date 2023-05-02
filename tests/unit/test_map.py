import unittest

from terra.config import EnvConfig
from terra.map import GridWorld


class TestMap(unittest.TestCase):
    def test_create_grid_world(self):
        seed = 5
        GridWorld.new(seed, EnvConfig())


if __name__ == "__main__":
    unittest.main()
