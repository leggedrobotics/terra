"""Weighted excavation potential retains telescoping and bucket independence."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.structured_actions import structured_excavation_potential, structured_progress
from terra.tests import test_pull_direction_alignment as native_fixtures


class PrecisionRewardTest(unittest.TestCase):
    def test_fixed_edges_each_get_half_and_bulk_keeps_volume_weighting(self):
        target = np.zeros((64, 64), np.int8)
        target[20:40, 30:50] = -1
        state = native_fixtures.PullDirectionAlignmentTest().state(target, position=(30, 20))
        state = state._replace(env_cfg=state.env_cfg._replace(enforce_foundation_border_alignment=True))
        edge = np.asarray(state._get_foundation_border_mask()) & (target < 0)
        inner = (target < 0) & ~edge
        self.assertTrue(edge.any() and inner.any())
        def changed(mask, precision=True):
            return state._replace(
                env_cfg=state.env_cfg._replace(enforce_foundation_border_alignment=precision),
                world=state.world._replace(action_map=state.world.action_map._replace(
                    map=jnp.asarray(np.where(mask, target, 0), state.world.action_map.map.dtype))))
        potential = jax.jit(lambda s: structured_excavation_potential(s, .5))
        empty, edges, interior, full = (changed(m) for m in
                                       (np.zeros_like(edge), edge, inner, target < 0))
        values = [float(potential(s)) for s in (empty, edges, interior, full)]
        np.testing.assert_allclose(values, [0., .5, .5, 1.])
        # Intermediate state and an already partly completed reset telescope.
        self.assertAlmostEqual((values[1] - values[0]) + (values[3] - values[1]), 1.)
        self.assertAlmostEqual(values[3] - values[1], .5)
        self.assertAlmostEqual(float(potential(changed(edge, False))), edge.sum() / (target < 0).sum())
        cur = full._get_current_agent_state()
        loaded = full._set_current_agent_state(cur._replace(loaded=jnp.ones_like(cur.loaded)))
        self.assertEqual(float(potential(loaded)), 1.)
        self.assertEqual(float(structured_progress(loaded)["total"]), 1.)

    def test_missing_edge_or_interior_group_uses_total_without_nan(self):
        # This small fixture isolates the fixed-group mathematics; native
        # geometry and full-map completion are covered above and elsewhere.
        from collections import namedtuple
        Map = namedtuple("Map", "map")
        World = namedtuple("World", "target_map action_map")
        Config = namedtuple("Config", "enforce_foundation_border_alignment")
        class State(namedtuple("State", "world env_cfg edge")):
            def _get_foundation_border_mask(self):
                return self.edge
        for edge in (False, True):
            state = State(World(Map(jnp.array([[-1, -1]])), Map(jnp.array([[-1, 0]]))),
                          Config(True), jnp.full((1, 2), edge))
            self.assertEqual(float(structured_excavation_potential(state, .5)), .5)


if __name__ == "__main__":
    unittest.main()
