import unittest

from src.config import EnvConfig
from src.map import GridWorld


class TestMap(unittest.TestCase):
    def test_create_grid_world(self):
        seed = 5
        GridWorld.new(seed, EnvConfig())


if __name__ == "__main__":
    unittest.main()
