"""Workspace geometry must agree across GPU batching and matmul precision."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.state import State
from terra.utils import apply_rot_transl
from terra.tests import test_trench_dig_alignment as trench_alignment


class WorkspaceGeometryPrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trench_alignment.FreshTrenchDigAlignmentTest.setUpClass()

    @staticmethod
    def _state(pile=True):
        target = np.zeros((64, 64), np.int8)
        action = np.zeros_like(target)
        for cell in ((35,39), (35,40), (36,39), (36,40),
                     (36,41), (37,39), (37,40), (38,40)):
            target[cell] = -1
        if pile:
            # A positive cell exactly on the +30-degree sector edge takes
            # precedence over fresh excavation. The old GPU vmap missed it.
            target[40,43] = action[40,43] = 1
            target[10,10] = action[10,10] = -1
        return trench_alignment.FreshTrenchDigAlignmentTest._state(
            target, -97*np.ones((4,8), np.float32), action=action,
            base_angle=8, cabin_angle=0, position=(29,43),
        )

    def test_batched_transform_matches_double_precision_reference(self):
        points = State._map_to_flattened_global_coords(64, 64, jnp.float32(4/7))
        anchors = jnp.asarray([[16.857143,25.142858,a] for a in
                               np.linspace(-np.pi,np.pi,24,endpoint=False)], jnp.float32)
        run = jax.jit(jax.vmap(apply_rot_transl, in_axes=(0,None)))
        for precision in ('bfloat16', 'float32'):
            with self.subTest(precision=precision), jax.default_matmul_precision(precision):
                actual = np.asarray(run(anchors,points))
                expected = []
                for x,y,theta in np.asarray(anchors,dtype=np.float64):
                    delta = np.asarray(points,dtype=np.float64)-np.array([[x],[y]])
                    c,s = np.cos(theta), np.sin(theta)
                    expected.append(np.array([[c,s],[-s,c]]) @ delta)
                np.testing.assert_allclose(actual,expected,atol=7e-6,rtol=2e-7)

    def test_position_gather_keeps_first_and_last_cell_centres(self):
        points = State._map_to_flattened_global_coords(64,64,jnp.float32(4/7))
        indices = jnp.asarray([[0],[29*64+43],[4095]],jnp.int32)
        batched = jax.jit(jax.vmap(State._get_current_pos_from_flattened_map,
                                  in_axes=(None,0)))
        np.testing.assert_array_equal(batched(points,indices),
                                      np.asarray(points)[:,np.asarray(indices[:,0])].T)

    def test_closed_angular_and_radial_edges_without_macroscopic_expansion(self):
        state = self._state(False)
        lo,hi = map(float,state._dig_cone_radius_bounds())
        angle = 2*np.pi/12
        mid = (lo+hi)/2
        coords = jnp.asarray([[mid,mid,lo,hi,mid,mid,lo-1e-3,hi+1e-3],
                              [-angle,angle,0,0,-angle-1e-3,angle+1e-3,0,0]])
        actual = jax.jit(lambda s,c: s._get_dig_dump_mask_cyl(c))(state,coords)
        np.testing.assert_array_equal(actual,[True]*4+[False]*4)

    def test_scalar_batched_and_nested_masks_agree_all_headings(self):
        state = self._state()
        poses = jnp.asarray([[r,c,b,a] for r,c in ((12,12),(29,43),(50,50))
                             for b in range(12) for a in range(12)],jnp.int32)
        def cone(p):
            cur = state._get_current_agent_state()._replace(
                pos_base=p[:2].astype(jnp.int16), angle_base=p[2:3].astype(jnp.int8),
                angle_cabin=p[3:4].astype(jnp.int8))
            return state._set_current_agent_state(cur)._build_dig_dump_cone()
        scalar = jax.jit(cone)
        expected = np.stack([np.asarray(scalar(p)) for p in poses])
        actual = jax.jit(jax.vmap(cone))(poses)
        nested = jax.jit(jax.vmap(jax.vmap(cone)))(poses.reshape(3,144,4))
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(nested.reshape(expected.shape),expected)

    def test_boundary_relift_observation_matches_executed_do(self):
        def inspect(s):
            counts = s._executable_fresh_dig_counts()
            result = s._handle_do()
            return counts, result.world.action_map.map, result._get_current_agent_state().loaded
        scalar = jax.jit(inspect)
        batched = jax.jit(jax.vmap(inspect))
        for pile in (False, True):
            with self.subTest(pile=pile):
                state = self._state(pile)
                expected = scalar(state)
                states = jax.tree_util.tree_map(lambda x: jnp.stack([x]*8), state)
                actual = batched(states)
                for got,want in zip(actual,expected):
                    np.testing.assert_array_equal(got,np.broadcast_to(want,got.shape))
                counts,after,load = map(np.asarray,expected)
                before = np.asarray(state.world.action_map.map)
                fresh = int(np.count_nonzero((after<0)&(before==0)))
                self.assertEqual(int(counts[0]),fresh)
                self.assertEqual(fresh,0 if pile else 8)
                self.assertEqual(int(load[0]),1 if pile else 8)


if __name__ == '__main__':
    unittest.main()
