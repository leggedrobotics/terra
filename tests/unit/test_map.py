import unittest

import jax.numpy as jnp

from src.config import EnvConfig
from src.map import GridMap
from src.map import GridWorld


class TestMap(unittest.TestCase):
    def test_random_map_one_dig(self):
        seed = 3
        width = 3
        height = 4
        map1 = GridMap.random_map_one_dig(seed, width, height)
        map2 = GridMap.random_map_one_dig(seed, width, height)

        self.assertTrue(jnp.equal(map1.map, map2.map).all())

        map3 = GridMap.random_map_one_dig(seed + 1, width, height)

        self.assertFalse(jnp.equal(map1.map, map3.map).all())

    def test_create_grid_world(self):
        seed = 5
        GridWorld.new(seed, EnvConfig())


if __name__ == "__main__":
    unittest.main()
