"""Native DO and observations under the +-30 degree pull cone."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.tests import test_pull_direction_alignment as radial_tests
from terra.wrappers import LocalMapWrapper

THIRTY = float(np.pi / 6)


class PullConeNativeTest(unittest.TestCase):
    SHAPE = (64, 64)
    fixture = radial_tests.PullDirectionAlignmentTest()

    def state(self, target, half_angle, **kwargs):
        state = self.fixture.state(target, **kwargs)
        return state._replace(env_cfg=state.env_cfg._replace(pull_half_angle_rad=half_angle))

    @staticmethod
    def trench():
        target = np.zeros((64, 64), np.int8)
        target[24:26, 10:58] = -1
        return target

    def test_oblique_base_digs_trench_only_with_the_cone(self):
        # Base 30 degrees off the trench axis: the radial rule finds a short
        # crossing chord, the cone pulls along the trench.
        target = self.trench()
        radial = self.state(target, 0.0, position=(29, 25), cabin=0)
        cone = self.state(target, THIRTY, position=(29, 25), cabin=0)
        radial_ok = np.asarray(radial._fresh_trench_pose_valid_cells()[2]) & (target < 0)
        cone_ok = np.asarray(cone._fresh_trench_pose_valid_cells()[2]) & (target < 0)
        self.assertTrue(np.all(cone_ok[radial_ok]))
        self.assertGreater(int(cone_ok.sum()), int(radial_ok.sum()))
        for cabin in range(12):
            s = self.state(target, THIRTY, position=(29, 25), cabin=cabin)
            after = s._handle_do()
            removed = (np.asarray(after.world.action_map.map) < 0) & (target < 0)
            self.assertTrue(np.all(cone_ok[removed]), cabin)
            counts = np.asarray(s._executable_fresh_dig_counts())
            self.assertEqual(int(counts[0]), int(removed.sum()), cabin)
            np.testing.assert_array_equal(after.world.target_map.map, target)

    def test_cone_never_admits_less_than_radial(self):
        rows, cols = np.indices(self.SHAPE)
        target = np.where((np.abs(rows - 30) <= 8) & (np.abs(cols - 32) <= 12), -1, 0).astype(np.int8)
        for precision in (False, True):
            for position in ((12, 20), (30, 8), (47, 47)):
                radial = self.state(target, 0.0, position=position, precision=precision)
                cone = self.state(target, THIRTY, position=position, precision=precision)
                a = np.asarray(radial._get_pull_dig_permission())
                b = np.asarray(cone._get_pull_dig_permission())
                self.assertTrue(np.all(b[a]), (precision, position))

    def test_static_switch_none_follows_config_and_false_compiles_radial(self):
        from terra.env import TerraEnv
        from terra.state import static_rules

        self.assertIsNone(TerraEnv().pull_cone)
        self.assertIsNone(TerraEnv().tracked_move_keeps_turn)
        target = self.trench()
        cone = self.state(target, THIRTY, position=(29, 25), cabin=0)
        radial = np.asarray(self.state(target, 0.0, position=(29, 25), cabin=0)._get_pull_dig_permission())
        with static_rules(pull_cone=None):
            follows = np.asarray(cone._get_pull_dig_permission())
        with static_rules(pull_cone=False):
            compiled_out = np.asarray(cone._get_pull_dig_permission())
        self.assertGreater(int(follows.sum()), int(radial.sum()))
        np.testing.assert_array_equal(compiled_out, radial)

    def test_jit_vmap_and_observation_share_cone_admission(self):
        target = self.trench()
        state = self.state(target, THIRTY, position=(29, 25), cabin=0)
        execute = jax.jit(lambda s: (s._handle_do(), s._executable_fresh_dig_counts()))
        after, counts = execute(state)
        removed = int(np.sum((np.asarray(after.world.action_map.map) < 0) & (target < 0)))
        self.assertGreater(removed, 0)
        self.assertEqual(int(counts[0]), removed)
        wrapped = LocalMapWrapper.wrap(state, executable_dig_observation=True)
        np.testing.assert_array_equal(wrapped.world.local_map_admissible_dig.map, counts)
        pair = jax.tree_util.tree_map(lambda x: jnp.stack([jnp.asarray(x)] * 2), state)
        cfg = pair.env_cfg._replace(pull_half_angle_rad=jnp.asarray([0.0, THIRTY], jnp.float32))
        both = jax.jit(jax.vmap(lambda s: s._handle_do().world.action_map.map))(pair._replace(env_cfg=cfg))
        radial_after = self.state(target, 0.0, position=(29, 25), cabin=0)._handle_do()
        np.testing.assert_array_equal(both[0], radial_after.world.action_map.map)
        np.testing.assert_array_equal(both[1], after.world.action_map.map)


if __name__ == '__main__':
    unittest.main()
