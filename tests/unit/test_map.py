import unittest

import jax

from terra.config import EnvConfig
from terra.map import GridWorld


class TestMap(unittest.TestCase):
    def test_create_grid_world(self):
        seed = 5
        key = jax.random.PRNGKey(seed)
        GridWorld.new(key, EnvConfig())


if __name__ == "__main__":
    unittest.main()
