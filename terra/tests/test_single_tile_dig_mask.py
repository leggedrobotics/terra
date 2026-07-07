import unittest

import jax.numpy as jnp
import numpy as np

from terra.state import State


class SingleTileDigMaskTest(unittest.TestCase):
    def test_single_tile_dig_mask_is_rejected(self):
        dig_mask = jnp.array([False, True, False])

        filtered = State._mask_out_single_tile_digs(dig_mask)

        np.testing.assert_array_equal(
            np.asarray(filtered),
            np.array([False, False, False]),
        )

    def test_multi_tile_dig_mask_is_preserved(self):
        dig_mask = jnp.array([False, True, True, False])

        filtered = State._mask_out_single_tile_digs(dig_mask)

        np.testing.assert_array_equal(np.asarray(filtered), np.asarray(dig_mask))


if __name__ == "__main__":
    unittest.main()
