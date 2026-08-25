import math
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.env import TerraEnvBatch
from terra.maps_buffer import _trench_records_from_metadata
from terra.state import State


class FreshTrenchDigAlignmentTest(unittest.TestCase):
    SHAPE = (64, 64)
    BASE_POSITION = (32, 32)

    @classmethod
    def setUpClass(cls):
        batch_env = object.__new__(TerraEnvBatch)
        batch_env.batch_cfg = BatchConfig()._replace(
            maps_dims=MapsDimsConfig(maps_edge_length=cls.SHAPE[0])
        )
        base = EnvConfig()
        updated = batch_env.update_env_cfgs(
            base._replace(
                agent=base.agent._replace(
                    dig_depth=jnp.ones((1,), dtype=jnp.int32)
                )
            )
        )
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
        axes = -97.0 * np.ones((4, 3), dtype=np.float32)
        for index, record in enumerate(records):
            axes[index] = np.asarray(record, dtype=np.float32)
        return axes

    @classmethod
    def _state(
        cls,
        target,
        axes,
        owners,
        *,
        trench_type=None,
        action=None,
        base_angle=0,
        cabin_angle=1,
        loaded=0,
        position=None,
    ):
        if action is None:
            action = np.zeros(cls.SHAPE, dtype=np.int8)
        if trench_type is None:
            trench_type = int(np.count_nonzero(axes[:, 0] > -96.0))
        state = State.new(
            jax.random.PRNGKey(7),
            cls.cfg,
            target,
            np.zeros(cls.SHAPE, dtype=np.int8),
            axes,
            np.int32(trench_type),
            owners,
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            np.ones(cls.SHAPE, dtype=np.bool_),
            action,
            distance_map_override=np.ones(cls.SHAPE, dtype=np.float32),
        )
        current = state._get_current_agent_state()._replace(
            pos_base=jnp.asarray(
                cls.BASE_POSITION if position is None else position,
                dtype=jnp.int16,
            ),
            angle_base=jnp.asarray([base_angle], dtype=jnp.int8),
            angle_cabin=jnp.asarray([cabin_angle], dtype=jnp.int8),
            loaded=jnp.asarray([loaded], dtype=jnp.int8),
        )
        return state._set_current_agent_state(current)

    @classmethod
    def _mask(cls, *cells):
        mask = np.zeros(cls.SHAPE, dtype=np.bool_)
        for cell in cells:
            mask[cell] = True
        return jnp.asarray(mask.reshape(-1))

    def test_arbitrary_half_bin_axis_is_feasible_and_boundary_is_inclusive(self):
        cell = (24, 37)
        angle = math.radians(15.0)
        a = math.sin(angle)
        b = math.cos(angle)
        axis = [a, b, -(a * cell[1] + b * cell[0])]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[cell] = -1
        owners = np.zeros(self.SHAPE, dtype=np.uint8)
        owners[cell] = 1
        mask = self._mask(cell)

        left = self._state(target, self._axes(axis), owners, base_angle=0)
        right = self._state(target, self._axes(axis), owners, base_angle=1)
        wrong = self._state(target, self._axes(axis), owners, base_angle=2)
        left_result = left._get_fresh_trench_dig_alignment_details(mask)
        right_result = right._get_fresh_trench_dig_alignment_details(mask)
        wrong_result = wrong._get_fresh_trench_dig_alignment_details(mask)
        jitted = jax.jit(
            lambda state: state._get_fresh_trench_dig_alignment_details(mask)
        )(left)

        self.assertTrue(bool(left_result[0]))
        self.assertTrue(bool(right_result[0]))
        self.assertFalse(bool(wrong_result[0]))
        self.assertAlmostEqual(float(left_result[1]), 15.0 / 90.0, places=5)
        self.assertEqual(bool(jitted[0]), bool(left_result[0]))
        self.assertEqual(int(left._handle_do().world.action_map.map[cell]), -1)

    def test_intersection_owners_are_multi_axis_and_do_remains_atomic(self):
        horizontal = (24, 37)
        vertical = (26, 40)
        junction = (24, 40)
        target = np.zeros(self.SHAPE, dtype=np.int8)
        owners = np.zeros(self.SHAPE, dtype=np.uint8)
        for cell, owner in ((horizontal, 1), (vertical, 2), (junction, 3)):
            target[cell] = -1
            owners[cell] = owner
        axes = self._axes([0.0, 1.0, -24.0], [1.0, 0.0, -40.0])

        horizontal_pose = self._state(target, axes, owners, base_angle=0)
        vertical_pose = self._state(target, axes, owners, base_angle=3)
        self.assertFalse(
            bool(
                horizontal_pose._get_fresh_trench_dig_alignment_details(
                    self._mask(horizontal, vertical)
                )[0]
            )
        )
        self.assertTrue(
            bool(
                horizontal_pose._get_fresh_trench_dig_alignment_details(
                    self._mask(horizontal, junction)
                )[0]
            )
        )
        self.assertTrue(
            bool(
                vertical_pose._get_fresh_trench_dig_alignment_details(
                    self._mask(vertical, junction)
                )[0]
            )
        )
        rejected = horizontal_pose._handle_do()
        self.assertFalse(np.any(np.asarray(rejected.world.action_map.map)))

    def test_ownerless_trench_fails_closed_at_state_and_batch_boundaries(self):
        cell = (24, 37)
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[cell] = -1
        axes = self._axes([0.0, 1.0, -24.0])
        owners = np.zeros(self.SHAPE, dtype=np.uint8)
        state = self._state(target, axes, owners)
        self.assertFalse(bool(state._get_fresh_trench_dig_alignment()[0]))
        self.assertEqual(int(state._handle_do().world.action_map.map[cell]), 0)

        records, count = _trench_records_from_metadata(
            {"axes_ABC": [{"A": 0.0, "B": 1.0, "C": -24.0}]},
            4,
        )
        self.assertEqual(count, 1)
        self.assertEqual(records[0], [0.0, 1.0, -24.0])

        batch = object.__new__(TerraEnvBatch)
        batch.maps_buffer = SimpleNamespace(
            maps=target[None, None, ...],
            trench_axes=axes[None, None, ...],
            trench_types=np.asarray([[1]], dtype=np.int32),
            trench_axis_owners=owners[None, None, ...],
        )
        with self.assertRaisesRegex(RuntimeError, "must have an owning axis"):
            batch._validate_trench_alignment_metadata_requirements(self.cfg)

    def test_foundations_and_relifts_ignore_the_fresh_trench_gate(self):
        cell = (24, 37)
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[cell] = -1
        no_owners = np.zeros(self.SHAPE, dtype=np.uint8)
        foundation = self._state(
            target,
            self._axes(),
            no_owners,
            trench_type=-1,
            base_angle=3,
            cabin_angle=10,
        )
        self.assertEqual(int(foundation._handle_do().world.action_map.map[cell]), -1)

        staged = np.zeros(self.SHAPE, dtype=np.int8)
        staged[cell] = 7
        trench = self._state(
            target,
            self._axes([0.0, 1.0, -24.0]),
            np.where(target < 0, 1, 0).astype(np.uint8),
            action=staged,
            base_angle=3,
            cabin_angle=10,
        )
        lifted = trench._handle_do()
        self.assertEqual(int(lifted.world.action_map.map[cell]), 0)
        self.assertEqual(int(lifted._get_current_agent_state().loaded[0]), 7)


if __name__ == "__main__":
    unittest.main()
