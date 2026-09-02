import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.env import TerraEnv
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
        batched = base._replace(
            agent=base.agent._replace(
                dig_depth=jnp.ones((1,), dtype=jnp.int32)
            )
        )
        updated = batch_env.update_env_cfgs(batched)
        # ``cls.cfg`` is the v2 contract: yaw-parallel only, no standoff band.
        # ``cls.cfg_v1`` restores the retired lateral band so the C0/T1 pilot
        # stays replayable. See TRENCH_GATE_STANDOFF_SEMANTICS_BUG_20260901.md.
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
        cls.cfg_v1 = cls.cfg._replace(trench_dig_standoff_enforced=True)
        assert not cls.cfg.trench_dig_standoff_enforced
        cls.tile = float(cls.cfg.tile_size)
        # The cone's outer reach, the v2 standoff normalizer.
        cls.cone_r_max = (
            0.5
            + cls.tile * max(cls.cfg.agent.width / 2, cls.cfg.agent.height / 2)
            + cls.cfg.agent.dig_radius_tiles * cls.tile
        )

    @staticmethod
    def _axes(*records):
        axes = -97.0 * np.ones((4, 8), dtype=np.float32)
        for index, record in enumerate(records):
            axes[index] = np.asarray(record, dtype=np.float32)
        return axes

    @classmethod
    def _state(
        cls,
        target,
        axes,
        *,
        action=None,
        base_angle=0,
        cabin_angle=0,
        loaded=0,
        position=None,
        cfg=None,
    ):
        if action is None:
            action = np.zeros(cls.SHAPE, dtype=np.int8)
        state = State.new(
            jax.random.PRNGKey(7),
            cls.cfg if cfg is None else cfg,
            target,
            np.zeros(cls.SHAPE, dtype=np.int8),
            axes,
            np.int32(np.count_nonzero(axes[:, 0] > -96.0)),
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            np.ones(cls.SHAPE, dtype=np.bool_),
            action,
            distance_map_override=np.ones(cls.SHAPE, dtype=np.float32),
        )
        current = state._get_current_agent_state()._replace(
            pos_base=jnp.array(
                cls.BASE_POSITION if position is None else position,
                dtype=jnp.int16,
            ),
            angle_base=jnp.array([base_angle], dtype=jnp.int8),
            angle_cabin=jnp.array([cabin_angle], dtype=jnp.int8),
            loaded=jnp.array([loaded], dtype=jnp.int8),
        )
        return state._set_current_agent_state(current)

    @classmethod
    def _horizontal_axis(cls):
        # row=24, with the base eight cells away at row=32.
        return cls._axes([0, 1, -24, 24, 20, 24, 50, 1])

    def _expected_v2_standoff(self, base_row, axis_row=24.0):
        """Signed perpendicular offset / cone reach, the v2 obs semantics."""
        return (base_row - axis_row) * self.tile / self.cone_r_max

    def test_aligned_fresh_dig_executes_and_observation_explains_it(self):
        # v2 "on the line": the base sits 2 cells (1.14 m) off the axis, inside
        # the 2.0 m on-line clause, chassis parallel, cone ahead along the strip.
        target = self._on_axis_strip_target()
        state = self._state(
            target,
            self._horizontal_axis(),
            base_angle=0,
            cabin_angle=1,
            position=(26, 32),
        )

        valid, yaw_error, standoff_error = (
            state._get_fresh_trench_dig_alignment()
        )
        dug = state._handle_do()
        jitted_valid, jitted_yaw, jitted_standoff = jax.jit(
            lambda candidate: candidate._get_fresh_trench_dig_alignment()
        )(state)
        jitted_dug = jax.jit(lambda candidate: candidate._handle_do())(state)

        self.assertTrue(bool(valid))
        self.assertEqual(float(yaw_error), 0.0)
        # v2: the diagnostic reports the signed offset / cone reach (1.14 / 6.50)
        # instead of a flat in-band 0.
        self.assertAlmostEqual(
            float(standoff_error), self._expected_v2_standoff(26.0), places=6
        )
        self.assertEqual(bool(jitted_valid), bool(valid))
        self.assertEqual(float(jitted_yaw), float(yaw_error))
        # The v2 diagnostic is a float32 division, so XLA's fused form and the
        # eager form can disagree in the last ULP (~6e-8 here).  v1's in-band
        # error was an exact 0.0 and had no such freedom.  The gate verdict is
        # a comparison on the yaw only and is bit-stable either way.
        self.assertAlmostEqual(
            float(jitted_standoff), float(standoff_error), places=6
        )
        self.assertEqual(int((np.asarray(dug.world.action_map.map) < 0).sum()), 5)
        self.assertEqual(int(dug._get_current_agent_state().loaded[0]), 5)
        np.testing.assert_array_equal(
            jitted_dug.world.action_map.map,
            dug.world.action_map.map,
        )

        wrapped = TerraEnv.wrap_state(state)
        observation = TerraEnv.new(64)._state_to_obs_dict(wrapped)
        self.assertEqual(
            float(observation["fresh_trench_dig_alignment_valid"]), 1.0
        )
        self.assertEqual(
            float(observation["fresh_trench_dig_yaw_error"]), 0.0
        )
        self.assertAlmostEqual(
            float(observation["fresh_trench_dig_standoff_error"]),
            self._expected_v2_standoff(26.0),
            places=6,
        )

        # The old sideways-lane pose (8 cells = 4.57 m off) under v1: admitted,
        # it is inside the retired band, and the exported standoff collapses to
        # the band-relative 0.0.
        v1_state = self._state(
            target,
            self._horizontal_axis(),
            base_angle=0,
            cabin_angle=1,
            cfg=self.cfg_v1,
        )
        v1_valid, v1_yaw, v1_standoff = (
            v1_state._get_fresh_trench_dig_alignment()
        )
        self.assertTrue(bool(v1_valid))
        self.assertEqual(float(v1_yaw), 0.0)
        self.assertEqual(float(v1_standoff), 0.0)

    def test_misaligned_fresh_dig_is_rejected_with_nonzero_reason(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[24, 37] = -1
        for label, cfg in (("v2", self.cfg), ("v1", self.cfg_v1)):
            with self.subTest(gate=label):
                state = self._state(
                    target,
                    self._horizontal_axis(),
                    base_angle=3,
                    cabin_angle=10,
                    cfg=cfg,
                )
                old_action = np.asarray(state.world.action_map.map)

                valid, yaw_error, standoff_error = (
                    state._get_fresh_trench_dig_alignment()
                )
                rejected = state._handle_do()
                ungated = state._replace(
                    env_cfg=state.env_cfg._replace(
                        enforce_trench_dig_alignment=False
                    )
                )._handle_do()

                self.assertFalse(bool(valid))
                self.assertAlmostEqual(float(yaw_error), 1.0, places=6)
                if label == "v1":
                    self.assertEqual(float(standoff_error), 0.0)
                else:
                    self.assertAlmostEqual(
                        float(standoff_error),
                        self._expected_v2_standoff(32.0),
                        places=6,
                    )
                np.testing.assert_array_equal(
                    rejected.world.action_map.map, old_action
                )
                self.assertEqual(
                    int(rejected._get_current_agent_state().loaded[0]), 0
                )
                self.assertEqual(
                    int(ungated.world.action_map.map[24, 37]), -1
                )

    def test_far_non_trench_target_on_mixed_map_is_unchanged(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        # This target is in the current workspace but far outside the finite
        # horizontal section's generated width plus raster-fringe tolerance.
        target[40, 37] = -1
        state = self._state(
            target,
            self._horizontal_axis(),
            base_angle=0,
            cabin_angle=1,
            position=(48, 32),
        )

        valid, yaw_error, standoff_error = (
            state._get_fresh_trench_dig_alignment()
        )
        dug = state._handle_do()

        self.assertTrue(bool(valid))
        self.assertEqual(float(yaw_error), 0.0)
        self.assertEqual(float(standoff_error), 0.0)
        self.assertEqual(int(dug.world.action_map.map[40, 37]), -1)

    def test_intersection_requires_every_fresh_cell_to_have_a_valid_axis(self):
        axes = self._axes(
            [0, 1, -24, 24, 20, 24, 50, 1],
            [1, 0, -40, 16, 40, 42, 40, 1],
        )
        # On the horizontal line at (24, 32), cone ahead: the junction cell
        # (24, 40) is shared, (26, 40) is exclusive to the perpendicular axis.
        target = np.zeros(self.SHAPE, dtype=np.int8)
        horizontal_cell = (24, 40)
        vertical_cell = (26, 40)
        target[horizontal_cell] = -1
        target[vertical_cell] = -1

        mixed = self._state(
            target,
            axes,
            base_angle=0,
            cabin_angle=0,
            position=(24, 32),
        )
        rejected = mixed._handle_do()
        valid, yaw_error, _ = mixed._get_fresh_trench_dig_alignment()
        self.assertFalse(bool(valid))
        self.assertAlmostEqual(float(yaw_error), 1.0, places=6)
        self.assertFalse(np.any(np.asarray(rejected.world.action_map.map)))

        vertical_done = np.zeros(self.SHAPE, dtype=np.int8)
        vertical_done[vertical_cell] = -1
        along_horizontal = self._state(
            target,
            axes,
            action=vertical_done,
            base_angle=0,
            cabin_angle=0,
            position=(24, 32),
        )._handle_do()
        self.assertEqual(
            int(along_horizontal.world.action_map.map[horizontal_cell]), -1
        )

        horizontal_done = np.zeros(self.SHAPE, dtype=np.int8)
        horizontal_done[horizontal_cell] = -1
        # From the vertical line at (34, 40), chassis parallel to it.
        along_vertical = self._state(
            target,
            axes,
            action=horizontal_done,
            base_angle=3,
            cabin_angle=0,
            position=(34, 40),
        )._handle_do()
        self.assertEqual(
            int(along_vertical.world.action_map.map[vertical_cell]), -1
        )

    def test_relift_and_dump_are_unaffected(self):
        axes = self._horizontal_axis()
        relift_target = np.zeros(self.SHAPE, dtype=np.int8)
        relift_target[24, 37] = -1
        staged = np.zeros(self.SHAPE, dtype=np.int8)
        staged[24, 37] = 7
        relift = self._state(
            relift_target,
            axes,
            action=staged,
            base_angle=3,
            cabin_angle=10,
        )
        lifted = relift._handle_do()
        self.assertEqual(int(lifted.world.action_map.map[24, 37]), 0)
        self.assertEqual(int(lifted._get_current_agent_state().loaded[0]), 7)

        dump_target = np.zeros(self.SHAPE, dtype=np.int8)
        dump_target[40, 27] = 1
        loaded = self._state(
            dump_target,
            axes,
            base_angle=3,
            cabin_angle=4,
            loaded=5,
        )
        dumped = loaded._handle_do()
        self.assertEqual(int(dumped._get_current_agent_state().loaded[0]), 0)
        self.assertEqual(int(np.asarray(dumped.world.action_map.map).sum()), 5)

    def test_complete_short_trench_and_backward_progression_are_feasible(self):
        """On the line: dig the five cells ahead, swing to the zone beside the
        machine, dump, then retreat one move along the axis."""
        target = np.zeros(self.SHAPE, dtype=np.int8)
        dig_cells = [(24, c) for c in (39, 40, 41, 42, 43)]
        dump_cells = [(r, c) for r in (31, 32, 33) for c in (31, 32, 33)]
        for cell in dig_cells:
            target[cell] = -1
        for cell in dump_cells:
            target[cell] = 1

        state = self._state(
            target,
            self._horizontal_axis(),
            base_angle=0,
            cabin_angle=0,
            position=(24, 32),
        )
        dug = state._handle_do()
        self.assertEqual(int(dug._get_current_agent_state().loaded[0]), 5)
        self.assertTrue(
            all(
                int(dug.world.action_map.map[cell]) == -1
                for cell in dig_cells
            )
        )

        dump_agent = dug._get_current_agent_state()._replace(
            angle_cabin=jnp.array([8], dtype=jnp.int8)
        )
        dumped = dug._set_current_agent_state(dump_agent)._handle_do()
        completion = dumped._get_task_completion(
            dumped.world.action_map.map,
            dumped.world.target_map.map,
        )
        self.assertEqual(int(dumped._get_current_agent_state().loaded[0]), 0)
        self.assertEqual(float(completion["absolute_completion"]), 1.0)

        retreated = TerraEnv.wrap_state(dumped)._handle_move_backward()
        # Backward along the axis: same row, five cells back.
        np.testing.assert_array_equal(
            retreated._get_current_agent_state().pos_base,
            np.array([24, 27], dtype=np.int16),
        )

    def test_v1_standoff_band_reports_signed_close_and_far_errors(self):
        """The retired v1 band, kept selectable for pilot replay."""
        close_target = np.zeros(self.SHAPE, dtype=np.int8)
        close_target[24, 37] = -1
        close = self._state(
            close_target,
            self._horizontal_axis(),
            base_angle=0,
            cabin_angle=1,
            position=(29, 32),
            cfg=self.cfg_v1,
        )
        close_valid, close_yaw, close_error = (
            close._get_fresh_trench_dig_alignment()
        )
        self.assertFalse(bool(close_valid))
        self.assertEqual(float(close_yaw), 0.0)
        self.assertLess(float(close_error), 0.0)

        wide_axis = self._axes([0, 1, -24, 24, 20, 24, 50, 2])
        far_target = np.zeros(self.SHAPE, dtype=np.int8)
        far_target[26, 32] = -1
        far = self._state(
            far_target,
            wide_axis,
            base_angle=0,
            cabin_angle=3,
            position=(37, 32),
            cfg=self.cfg_v1,
        )
        far_valid, far_yaw, far_error = far._get_fresh_trench_dig_alignment()
        self.assertFalse(bool(far_valid))
        self.assertEqual(float(far_yaw), 0.0)
        self.assertGreater(float(far_error), 0.0)

    def test_v2_on_the_line_clause_admits_near_line_and_refuses_the_sideways_lane(self):
        """v2 = yaw-parallel AND perpendicular offset <= 2.0 m.

        Yaw alone still admitted the old sideways lane (parallel at 3-6 m to
        the side, cabin swung in), which is not "on top of the trench".  The
        on-line clause closes that: near-line poses dig, the lane is refused,
        and the exported standoff is the signed offset / cone reach either way.
        """
        strip = self._on_axis_strip_target()

        near = self._state(
            strip, self._horizontal_axis(), base_angle=0, cabin_angle=0,
            position=(26, 32),
        )
        near_valid, near_yaw, near_error = near._get_fresh_trench_dig_alignment()
        self.assertTrue(bool(near_valid))
        self.assertEqual(float(near_yaw), 0.0)
        self.assertAlmostEqual(
            float(near_error), self._expected_v2_standoff(26.0), places=6
        )
        self.assertEqual(
            int((np.asarray(near._handle_do().world.action_map.map) < 0).sum()), 5
        )

        # 5 cells = 2.86 m off: v1's "too close", v2's "off the line".
        lane = self._state(
            strip, self._horizontal_axis(), base_angle=0, cabin_angle=1,
            position=(29, 32),
        )
        lane_valid, lane_yaw, lane_error = lane._get_fresh_trench_dig_alignment()
        self.assertFalse(bool(lane_valid))
        self.assertEqual(float(lane_yaw), 0.0)
        self.assertAlmostEqual(
            float(lane_error), self._expected_v2_standoff(29.0), places=6
        )
        self.assertEqual(
            int((np.asarray(lane._handle_do().world.action_map.map) < 0).sum()), 0
        )

        # 13 cells = 7.43 m off, past the cone's reach: refused, offset saturates.
        wide_axis = self._axes([0, 1, -24, 24, 20, 24, 50, 2])
        far_target = np.zeros(self.SHAPE, dtype=np.int8)
        far_target[26, 32] = -1
        far = self._state(
            far_target, wide_axis, base_angle=0, cabin_angle=3, position=(37, 32)
        )
        far_valid, far_yaw, far_error = far._get_fresh_trench_dig_alignment()
        self.assertFalse(bool(far_valid))
        self.assertEqual(float(far_yaw), 0.0)
        self.assertEqual(float(far_error), 1.0)
        self.assertEqual(
            int((np.asarray(far._handle_do().world.action_map.map) < 0).sum()), 0
        )

    # ---------------------------------------------------------------- #
    # v2 semantics: the on-axis dig-ahead pattern v1 wrongly refused    #
    # ---------------------------------------------------------------- #
    def _on_axis_strip_target(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[24, 20:51] = -1
        return target

    def test_v1_refuses_the_on_axis_dig_ahead_pose_that_v2_admits(self):
        """The design error, reproduced.

        The machine sits ON the trench axis (row 24), chassis parallel to it,
        empty, cabin pointing straight ahead along the trench.  Terra's own dig
        cone selects five fresh trench cells at 4.00-6.29 m RADIAL distance,
        i.e. inside the 3.64-6.50 m reach annulus the cone enforces.  Nothing
        physical objects.  v1 refuses the whole macro action because the
        PERPENDICULAR base-to-axis distance is 0.00 m, below its 3.5 m floor.
        """
        target = self._on_axis_strip_target()
        axes = self._horizontal_axis()

        v1 = self._state(
            target, axes, base_angle=0, cabin_angle=0,
            position=(24, 32), cfg=self.cfg_v1,
        )
        v1_valid, v1_yaw, v1_standoff = v1._get_fresh_trench_dig_alignment()
        v1_dug = v1._handle_do()

        # The pose is aligned and the cone is loaded with reachable fresh soil.
        self.assertEqual(float(v1_yaw), 0.0)
        selected = np.asarray(
            v1._mask_out_wrong_dig_tiles(v1._build_dig_dump_cone())
        ).reshape(self.SHAPE)
        self.assertEqual(int(selected.sum()), 5)
        np.testing.assert_array_equal(
            np.nonzero(selected[24])[0], np.array([39, 40, 41, 42, 43])
        )
        # ... and v1 refuses it, purely on the lateral band.
        self.assertFalse(bool(v1_valid))
        self.assertEqual(float(v1_standoff), -1.0)
        self.assertEqual(int((np.asarray(v1_dug.world.action_map.map) < 0).sum()), 0)
        self.assertEqual(int(v1_dug._get_current_agent_state().loaded[0]), 0)

        v2 = self._state(
            target, axes, base_angle=0, cabin_angle=0, position=(24, 32)
        )
        v2_valid, v2_yaw, v2_standoff = v2._get_fresh_trench_dig_alignment()
        v2_dug = v2._handle_do()

        self.assertTrue(bool(v2_valid))
        self.assertEqual(float(v2_yaw), 0.0)
        # Exactly on the line: the v2 diagnostic is 0.0 here, where v1's was the
        # saturated -1.0.  Same pose, opposite reading.
        self.assertEqual(float(v2_standoff), 0.0)
        dug_cols = np.nonzero(np.asarray(v2_dug.world.action_map.map)[24] < 0)[0]
        np.testing.assert_array_equal(dug_cols, np.array([39, 40, 41, 42, 43]))
        self.assertEqual(int(v2_dug._get_current_agent_state().loaded[0]), 5)

    def test_v2_dig_ahead_then_retreat_backward_clears_a_strip(self):
        """Lorenzo's pattern: sit on the trench, dig ahead, back up, repeat.

        The machine never drives over its own hole (the cells it removes are
        always ahead of it), so this is exactly the manoeuvre the lane band was
        supposedly protecting against, executed without a lane.
        """
        state = TerraEnv.wrap_state(
            self._state(
                self._on_axis_strip_target(),
                self._horizontal_axis(),
                base_angle=0,
                cabin_angle=0,
                position=(24, 32),
            )
        )
        for _ in range(4):
            valid, yaw, standoff = state._get_fresh_trench_dig_alignment()
            self.assertTrue(bool(valid))
            self.assertEqual(float(yaw), 0.0)
            self.assertEqual(float(standoff), 0.0)
            dug = state._handle_do()
            self.assertEqual(int(dug._get_current_agent_state().loaded[0]), 5)
            # Unload in place: this test is about the dig/retreat geometry, not
            # about the dump contract, which has its own suite.
            emptied = dug._set_current_agent_state(
                dug._get_current_agent_state()._replace(
                    loaded=jnp.array([0], dtype=jnp.int8)
                )
            )
            state = emptied._handle_move_backward()
            # The chassis stays on the axis; only its column changes.
            self.assertEqual(int(state._get_current_agent_state().pos_base[0]), 24)

        dug_cols = np.nonzero(np.asarray(state.world.action_map.map)[24] < 0)[0]
        np.testing.assert_array_equal(dug_cols, np.arange(24, 44))

    def test_v2_junction_veto_still_fires_from_an_on_axis_pose(self):
        """The all-or-nothing junction clause is untouched by v2."""
        axes = self._axes(
            [0, 1, -24, 24, 20, 24, 50, 1],
            [1, 0, -40, 16, 40, 42, 40, 1],
        )
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[24, 20:51] = -1
        target[16:43, 40] = -1

        # On axis 0, aligned to it, digging ahead into the T junction: the cone
        # reaches cells owned EXCLUSIVELY by the perpendicular axis 1, which is
        # 90 deg off, so the complete DO is refused.
        into_junction = self._state(
            target, axes, base_angle=0, cabin_angle=0, position=(24, 32)
        )
        valid, _, _ = into_junction._get_fresh_trench_dig_alignment()
        self.assertFalse(bool(valid))
        self.assertEqual(
            int((np.asarray(into_junction._handle_do().world.action_map.map) < 0).sum()),
            0,
        )

        # Same pose, cabin turned around to dig away from the junction: only
        # axis-0 cells are selected and the dig is admitted.
        away = self._state(
            target, axes, base_angle=0, cabin_angle=6, position=(24, 32)
        )
        away_valid, _, _ = away._get_fresh_trench_dig_alignment()
        self.assertTrue(bool(away_valid))
        dug_cols = np.nonzero(np.asarray(away._handle_do().world.action_map.map)[24] < 0)[0]
        np.testing.assert_array_equal(dug_cols, np.array([21, 22, 23, 24, 25]))

    def test_v2_still_refuses_a_misaligned_pose_on_the_axis(self):
        """Dropping the band does not weaken the yaw clause.

        Chassis at 90 deg to the trench, standing on the axis.  The cabin is
        swung so the cone points back along the trench (cabin 0 would point it
        perpendicular, where there is no trench soil and the gate is simply
        inapplicable).  Five fresh cells are selected; yaw refuses them.
        """
        target = self._on_axis_strip_target()
        state = self._state(
            target,
            self._horizontal_axis(),
            base_angle=3,
            cabin_angle=9,
            position=(24, 32),
        )
        selected = np.asarray(
            state._mask_out_wrong_dig_tiles(state._build_dig_dump_cone())
        ).reshape(self.SHAPE)
        self.assertEqual(int(selected.sum()), 5)
        valid, yaw, standoff = state._get_fresh_trench_dig_alignment()
        self.assertFalse(bool(valid))
        self.assertAlmostEqual(float(yaw), 1.0, places=6)
        self.assertEqual(float(standoff), 0.0)
        self.assertEqual(
            int((np.asarray(state._handle_do().world.action_map.map) < 0).sum()), 0
        )

    def test_non_trench_fresh_dig_is_unchanged_when_gate_is_enabled(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[24, 37] = -1
        state = self._state(
            target,
            self._axes(),
            base_angle=0,
            cabin_angle=1,
        )
        dug = state._handle_do()
        self.assertEqual(int(dug.world.action_map.map[24, 37]), -1)
        self.assertEqual(int(dug._get_current_agent_state().loaded[0]), 1)

    def test_generated_segment_metadata_is_required_and_validated(self):
        metadata = {
            "trench_axes_count": 1,
            "axes_ABC": [{"A": 0.0, "B": 1.0, "C": -24.0}],
            "trench_segments_yx": [[[24.0, 20.0], [24.0, 50.0]]],
            "trench_half_width_tiles": 1.0,
        }
        records, count = _trench_records_from_metadata(
            metadata,
            4,
            require_finite_segments=True,
        )
        self.assertEqual(count, 1)
        self.assertEqual(records[0], [0.0, 1.0, -24.0, 24.0, 20.0, 24.0, 50.0, 1.0])

        missing_segments = {
            key: value
            for key, value in metadata.items()
            if key != "trench_segments_yx"
        }
        with self.assertRaisesRegex(RuntimeError, "finite generated segment"):
            _trench_records_from_metadata(
                missing_segments,
                4,
                require_finite_segments=True,
            )

        stale = dict(metadata)
        stale["trench_segments_yx"] = [[[25.0, 20.0], [25.0, 50.0]]]
        with self.assertRaisesRegex(RuntimeError, "does not lie on its paired axis"):
            _trench_records_from_metadata(
                stale,
                4,
                require_finite_segments=True,
            )

    def test_global_gate_activation_validates_loaded_finite_metadata(self):
        batch = object.__new__(TerraEnvBatch)
        batch.maps_buffer = SimpleNamespace(
            trench_axes=self._horizontal_axis()[None, None, ...],
            trench_types=np.array([[1]], dtype=np.int32),
            family_names=("trn-straight",),
            family_ids=np.array([[0]], dtype=np.int32),
        )
        batch._validate_trench_alignment_metadata_requirements(self.cfg)

        missing = -97.0 * np.ones((1, 1, 4, 8), dtype=np.float32)
        batch.maps_buffer = SimpleNamespace(
            trench_axes=missing,
            trench_types=np.array([[1]], dtype=np.int32),
            family_names=("trn-straight",),
            family_ids=np.array([[0]], dtype=np.int32),
        )
        with self.assertRaisesRegex(RuntimeError, "finite section metadata"):
            batch._validate_trench_alignment_metadata_requirements(self.cfg)

        batch._validate_trench_alignment_metadata_requirements(
            self.cfg._replace(enforce_trench_dig_alignment=False)
        )

    def test_lower_level_state_fails_closed_on_incomplete_trench_metadata(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[24, 37] = -1
        incomplete = self._axes(
            [0, 1, -24, -97, -97, -97, -97, -97]
        )
        state = self._state(
            target,
            incomplete,
            base_angle=0,
            cabin_angle=1,
        )

        valid, _, _ = state._get_fresh_trench_dig_alignment()
        rejected = state._handle_do()

        self.assertFalse(bool(valid))
        self.assertEqual(int(rejected.world.action_map.map[24, 37]), 0)
        self.assertEqual(int(rejected._get_current_agent_state().loaded[0]), 0)


if __name__ == "__main__":
    unittest.main()
