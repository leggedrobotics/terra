import unittest
import jax.numpy as jnp
from src.map import GridMap


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

        # print(map1)
        # print(map2)
        # print(map3)


if __name__ == "__main__":
    unittest.main()
