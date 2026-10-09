"""Four working-room claims, using metric cell footprints rather than paths."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.working_strip import working_strip_mask


fit = jax.jit(working_strip_mask)
TILE = 4. / 7.


class WorkingStripTest(unittest.TestCase):
    def room(self, fresh, dug=None, directions=((1., 0.),)):
        if dug is None:
            dug = np.zeros_like(fresh)
        return np.asarray(fit(fresh, dug, jnp.asarray(directions), TILE, 1., 1.3))

    def test_lone_inner_cell_can_use_connected_dug_outer_room(self):
        fresh = np.zeros((15, 15), bool)
        fresh[7, 7] = True
        dug = np.zeros_like(fresh)
        dug[6:8, 6:9] = True  # 2 cells long, 3 cells across
        dug[fresh] = False
        np.testing.assert_array_equal(self.room(fresh, dug), fresh)

    def test_isolated_cell_and_one_cell_wide_lane_have_no_room(self):
        fresh = np.zeros((15, 15), bool)
        fresh[7, 7] = True
        self.assertFalse(self.room(fresh).any())
        fresh[2:13, 7] = True
        self.assertFalse(self.room(fresh).any())

    def test_all_borrowed_fresh_support_is_admitted_and_a_gap_breaks_the_strip(self):
        fresh = np.zeros((15, 15), bool)
        fresh[6:8, 6:9] = True
        admitted = self.room(fresh)
        np.testing.assert_array_equal(admitted, fresh)
        # The top row cannot borrow from a bottom row excluded by the caller.
        fresh[7, 7] = False
        self.assertFalse(self.room(fresh).any())
        # A separated dug patch cannot act as connected working room.
        isolated = np.zeros_like(fresh)
        isolated[7, 7] = True
        dug = np.zeros_like(fresh)
        dug[4:6, 6:9] = True
        self.assertFalse(self.room(isolated, dug).any())

    def test_rotation_and_batched_execution_preserve_the_same_room(self):
        fresh = np.zeros((15, 15), bool)
        fresh[6:8, 6:9] = True
        vertical = self.room(fresh)
        horizontal = self.room(fresh.T, directions=((0., 1.),))
        np.testing.assert_array_equal(horizontal, vertical.T)
        direction = jnp.asarray([[1., 0.], [np.cos(np.pi / 6), np.sin(np.pi / 6)],
                                 [np.cos(np.pi / 6), -np.sin(np.pi / 6)]], jnp.float32)
        batched = jax.jit(jax.vmap(lambda f: working_strip_mask(
            f, jnp.zeros_like(f), direction, TILE, 1., 1.3)))(jnp.asarray([fresh, fresh]))
        np.testing.assert_array_equal(batched[0], batched[1])
        np.testing.assert_array_equal(batched[0], fresh)
        # Two-cell width is only 1.143m; neither tilted nor straight strip fits.
        narrow = np.zeros_like(fresh)
        narrow[2:13, 6:8] = True
        self.assertFalse(self.room(narrow, directions=direction).any())


if __name__ == '__main__':
    unittest.main()
