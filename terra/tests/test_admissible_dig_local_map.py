"""``local_map_admissible_dig``: fresh digs a DO would be admitted per cabin angle.

Parity claim: for every cabin angle k, the entry equals the number of fresh
target cells the prospective-DO gate would admit on that angle's workspace
cone (the same cyl cone the other local maps are summed over), from the
current base pose. Neutral maps reduce to the fresh-target count.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.env import TerraEnv
from terra.env import TerraEnvBatch
from terra.settings import IntMap
from terra.state import State
from terra.wrappers import LocalMapWrapper


class AdmissibleDigLocalMapTest(unittest.TestCase):
    SHAPE = (64, 64)

    @classmethod
    def setUpClass(cls):
        batch_env = object.__new__(TerraEnvBatch)
        batch_env.batch_cfg = BatchConfig()._replace(
            maps_dims=MapsDimsConfig(maps_edge_length=cls.SHAPE[0])
        )
        base = EnvConfig()
        batched = base._replace(
            agent=base.agent._replace(
                dig_depth=jnp.ones((1,), dtype=jnp.int32)
            )
        )
        updated = batch_env.update_env_cfgs(batched)
        cls.cfg = base._replace(
            tile_size=float(np.asarray(updated.tile_size)[0]),
            agent=base.agent._replace(
                width=int(np.asarray(updated.agent.width)[0]),
                height=int(np.asarray(updated.agent.height)[0]),
            ),
            maps=base.maps._replace(edge_length_px=cls.SHAPE[0]),
            agent_types=(0,),
            action_types=(0,),
            enforce_trench_dig_alignment=True,
        )

    @staticmethod
    def _axes(*records):
        axes = -97.0 * np.ones((4, 8), dtype=np.float32)
        for index, record in enumerate(records):
            axes[index] = np.asarray(record, dtype=np.float32)
        return axes

    @classmethod
    def _state(cls, target, axes, *, base_angle, cabin_angle, position, loaded=0):
        state = State.new(
            jax.random.PRNGKey(7),
            cls.cfg,
            target,
            np.zeros(cls.SHAPE, dtype=np.int8),
            axes,
            np.int32(np.count_nonzero(axes[:, 0] > -96.0)),
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            np.ones(cls.SHAPE, dtype=np.bool_),
            np.zeros(cls.SHAPE, dtype=np.int8),
            distance_map_override=np.ones(cls.SHAPE, dtype=np.float32),
        )
        current = state._get_current_agent_state()._replace(
            pos_base=jnp.array(position, dtype=jnp.int16),
            angle_base=jnp.array([base_angle], dtype=jnp.int8),
            angle_cabin=jnp.array([cabin_angle], dtype=jnp.int8),
            loaded=jnp.array([loaded], dtype=jnp.int8),
        )
        return state._set_current_agent_state(current)

    @staticmethod
    def _strip_target():
        # Horizontal strip along row 24, with no junction cells.
        target = np.zeros((64, 64), dtype=np.int8)
        target[24, 20:50] = -1
        return target

    @classmethod
    def _junction(cls):
        axes = cls._axes(
            [0, 1, -24, 24, 20, 24, 50, 1],
            [1, 0, -40, 16, 40, 42, 40, 1],
        )
        shared = (24, 40)
        horizontal_only = (24, 42)
        vertical_only = (26, 40)
        target = np.zeros(cls.SHAPE, dtype=np.int8)
        for cell in (shared, horizontal_only, vertical_only):
            target[cell] = -1
        return target, axes, shared, horizontal_only, vertical_only

    def _expected_per_angle(self, state):
        """Gate verdict per cyl cone, evaluated by the prospective-DO gate."""
        masks, arm_angle = LocalMapWrapper._build_local_cartesian_masks(
            state, state.agent.current_agent
        )
        target = np.asarray(state.world.target_map.map)
        action = np.asarray(state.world.action_map.map)
        fresh = np.logical_and(target < 0, action == 0).reshape(-1)
        expected = np.zeros((masks.shape[0],), dtype=np.int32)
        for angle in range(masks.shape[0]):
            cone = np.asarray(masks[angle]).reshape(-1)
            valid, _, _, admitted = state._get_fresh_trench_dig_alignment_details(
                jnp.asarray(cone)
            )
            admitted = np.asarray(admitted).reshape(-1)
            expected[angle] = int(np.logical_and(admitted, fresh).sum())
            if not bool(valid):
                self.assertEqual(expected[angle], 0)
        return np.roll(expected, -int(arm_angle)), int(arm_angle)

    def _observed(self, state):
        wrapped = TerraEnv.wrap_state(state)
        obs = TerraEnv.new(64)._state_to_obs_dict(wrapped)
        return obs, np.asarray(obs["local_map_admissible_dig"])

    def test_aligned_pose_matches_gate_per_cabin_angle(self):
        axes = self._axes([0, 1, -24, 24, 20, 24, 50, 1])
        state = self._state(
            self._strip_target(), axes, base_angle=0, cabin_angle=1, position=(26, 32)
        )
        expected, _ = self._expected_per_angle(state)
        obs, observed = self._observed(state)

        self.assertEqual(observed.dtype, IntMap)
        self.assertEqual(observed.shape, (12,))
        np.testing.assert_array_equal(observed, expected)
        # The aligned pose admits real digs on the cone ahead (index 0 is the
        # current arm angle) and they are a subset of the fresh target sum.
        self.assertGreater(int(observed[0]), 0)
        target_neg = -np.asarray(obs["local_map_target_neg"])
        self.assertTrue(np.all(observed <= target_neg))
        # The distance map rides along verbatim.
        np.testing.assert_array_equal(
            np.asarray(obs["relocation_distance_map"]),
            np.asarray(state.world.relocation_distance_map),
        )
        self.assertEqual(obs["relocation_distance_map"].shape, self.SHAPE)

    def test_misaligned_pose_sees_targets_but_no_admissible_dig(self):
        axes = self._axes([0, 1, -24, 24, 20, 24, 50, 1])
        state = self._state(
            self._strip_target(), axes, base_angle=3, cabin_angle=10, position=(32, 32)
        )
        expected, _ = self._expected_per_angle(state)
        obs, observed = self._observed(state)
        np.testing.assert_array_equal(observed, expected)
        target_neg = -np.asarray(obs["local_map_target_neg"])
        # Targets are in reach of several cabin angles, none is admissible.
        self.assertGreater(int(target_neg.sum()), 0)
        self.assertEqual(int(observed.sum()), 0)

    def test_junction_counts_match_do_from_either_owning_axis(self):
        target, axes, shared, horizontal_only, vertical_only = self._junction()
        do = jax.jit(lambda state: state._handle_do())
        approaches = (
            (0, (24, 32), horizontal_only, vertical_only),
            (3, (34, 40), vertical_only, horizontal_only),
        )
        for base_angle, position, aligned_only, unaligned_only in approaches:
            with self.subTest(base_angle=base_angle):
                state = self._state(
                    target, axes, base_angle=base_angle,
                    cabin_angle=0, position=position,
                )
                membership = np.asarray(state.world.trench_axis_membership)
                self.assertEqual(int(membership[shared]), 0b11)
                self.assertEqual(int(membership[horizontal_only]), 0b01)
                self.assertEqual(int(membership[vertical_only]), 0b10)

                expected, _ = self._expected_per_angle(state)
                obs, observed = self._observed(state)
                np.testing.assert_array_equal(observed, expected)
                self.assertEqual(int(observed[0]), 2)
                self.assertEqual(float(obs["fresh_trench_dig_alignment_valid"]), 1.0)

                # All three targets lie beyond the inner cone-cleaning band,
                # so each local count also equals the actual fresh DO volume.
                # Restart from the same fresh state for every cabin heading.
                for cabin_angle in range(12):
                    with self.subTest(cabin_angle=cabin_angle):
                        candidate = state._set_current_agent_state(
                            state._get_current_agent_state()._replace(
                                angle_cabin=jnp.array([cabin_angle], dtype=jnp.int8),
                            )
                        )
                        result = do(candidate)
                        dug = np.asarray(result.world.action_map.map) < 0
                        self.assertEqual(int(dug.sum()), int(observed[cabin_angle]))
                        self.assertFalse(dug[unaligned_only])
                        self.assertEqual(
                            int(result._get_current_agent_state().loaded[0]),
                            int(dug.sum()),
                        )
                        if cabin_angle == 0:
                            # The same shared cell is diggable from BOTH axes;
                            # another aligned section cannot admit an unowned cell.
                            self.assertTrue(dug[shared])
                            self.assertTrue(dug[aligned_only])

    def test_junction_rejects_pose_aligned_with_neither_owning_axis(self):
        target, axes, _, _, _ = self._junction()
        # Rotate the chassis 30 degrees while keeping the cabin cone pointed
        # along the horizontal section. Neither section accepts that yaw.
        state = self._state(
            target, axes, base_angle=1, cabin_angle=11, position=(24, 32)
        )
        expected, _ = self._expected_per_angle(state)
        obs, observed = self._observed(state)
        np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(observed, np.zeros((12,), dtype=IntMap))
        self.assertGreater(int(-np.asarray(obs["local_map_target_neg"])[0]), 0)
        self.assertEqual(float(obs["fresh_trench_dig_alignment_valid"]), 0.0)
        np.testing.assert_array_equal(
            np.asarray(state._handle_do().world.action_map.map),
            np.asarray(state.world.action_map.map),
        )

    def test_neutral_cases_reduce_to_fresh_target_count(self):
        axes = self._axes([0, 1, -24, 24, 20, 24, 50, 1])
        strip = self._strip_target()
        # (a) loaded excavator: gate neutral -> plain fresh count per cone.
        loaded = self._state(strip, axes, base_angle=3, cabin_angle=10, position=(32, 32), loaded=5)
        expected, _ = self._expected_per_angle(loaded)
        obs, observed = self._observed(loaded)
        np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(observed, -np.asarray(obs["local_map_target_neg"]))
        # (b) non-trench map: no axes at all.
        no_axes = -97.0 * np.ones((4, 8), dtype=np.float32)
        foundation = self._state(strip, no_axes, base_angle=3, cabin_angle=10, position=(32, 32))
        obs, observed = self._observed(foundation)
        np.testing.assert_array_equal(observed, -np.asarray(obs["local_map_target_neg"]))

    def test_jit_matches_eager(self):
        axes = self._axes([0, 1, -24, 24, 20, 24, 50, 1])
        state = self._state(
            self._strip_target(), axes, base_angle=0, cabin_angle=1, position=(26, 32)
        )
        eager = np.asarray(TerraEnv.wrap_state(state).world.local_map_admissible_dig.map)
        jitted = np.asarray(
            jax.jit(lambda s: TerraEnv.wrap_state(s).world.local_map_admissible_dig.map)(state)
        )
        np.testing.assert_array_equal(eager, jitted)


if __name__ == "__main__":
    unittest.main()
