from types import SimpleNamespace
import unittest

import jax.numpy as jnp
import numpy as np

from terra.settings import IntMap
from terra.wrappers import LocalMapWrapper


class LocalMapDtypeTest(unittest.TestCase):
    def test_local_map_workspace_sums_do_not_wrap_int8(self):
        state = SimpleNamespace(world=SimpleNamespace(height=12, width=12))
        map_to_wrap = jnp.ones((12, 12), dtype=jnp.int16)
        full_workspace = jnp.ones((1, 12 * 12), dtype=jnp.bool_)

        local = LocalMapWrapper._wrap_with_masks(
            state,
            map_to_wrap,
            full_workspace,
            jnp.array(0, dtype=jnp.int32),
        )

        self.assertEqual(local.dtype, IntMap)
        np.testing.assert_array_equal(
            np.asarray(local), np.array([144], dtype=np.int16)
        )


if __name__ == "__main__":
    unittest.main()
