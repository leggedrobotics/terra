"""Native transition tests for radial edge/trench admission."""

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig, EnvConfig, MapsDimsConfig
from terra.dig_direction import MAX_BOUNDARY_SEGMENTS, boundary_records_from_mask
from terra.env import TerraEnv, TerraEnvBatch
from terra.maps_buffer import MapsBuffer
from terra.state import State
from terra.wrappers import LocalMapWrapper


class PullDirectionAlignmentTest(unittest.TestCase):
    SHAPE = (64, 64)

    @staticmethod
    def axes(*records):
        result = np.full((8, 8), -97, np.float32)
        for i, record in enumerate(records):
            result[i] = record
        return result

    def state(self, target, *, axes=None, position=(24, 30), base=0, cabin=0,
              action=None, pull=True, agent_types=(0,), precision=False):
        cfg = EnvConfig()._replace(
            tile_size=40 / 70,
            agent=EnvConfig().agent._replace(width=7, height=11),
            maps=EnvConfig().maps._replace(edge_length_px=64),
            agent_types=agent_types, action_types=tuple(0 for _ in agent_types),
            pull_direction_alignment=pull,
            enforce_foundation_border_alignment=precision,
            dig_pull_min_length_m=2.5,
        )
        if axes is None:
            axes = self.axes()
        boundaries, count = boundary_records_from_mask(target < 0)
        state = State.new(
            jax.random.PRNGKey(7), cfg, target, np.zeros(self.SHAPE, np.int8),
            axes, np.int32(np.count_nonzero(axes[:, 7] > 0)), boundaries,
            np.int32(count), np.ones(self.SHAPE, bool),
            np.zeros(self.SHAPE, np.int8) if action is None else action,
            distance_map_override=np.ones(self.SHAPE, np.float32),
        )
        cur = state._get_current_agent_state()._replace(
            pos_base=jnp.asarray(position, jnp.int16),
            angle_base=jnp.asarray([base], jnp.int8),
            angle_cabin=jnp.asarray([cabin], jnp.int8),
            loaded=jnp.asarray([0], jnp.int8),
        )
        return state._set_current_agent_state(cur)

    def strip(self):
        target = np.zeros(self.SHAPE, np.int8)
        target[24, 32:55] = -1
        return target, self.axes([0, 1, -24, 24, 32, 24, 54, 0.5])

    def test_chassis_yaw_is_not_pull_direction(self):
        target, axes = self.strip()
        forward = self.state(target, axes=axes)
        sideways_chassis = self.state(target, axes=axes, base=3, cabin=9)
        masks = []
        for state in (forward, sideways_chassis):
            mask, _, _, admitted = state._dig_eligibility(state._build_dig_dump_cone())
            self.assertTrue(bool(admitted))
            masks.append(np.asarray(mask))
        np.testing.assert_array_equal(*masks)
        legacy = sideways_chassis._replace(env_cfg=sideways_chassis.env_cfg._replace(
            pull_direction_alignment=False, enforce_trench_dig_alignment=True))
        self.assertFalse(bool(legacy._dig_eligibility(legacy._build_dig_dump_cone())[3]))

    def test_narrow_trench_has_no_bucket_width_gate_even_in_old_saved_config(self):
        target, axes = self.strip()
        state = self.state(target, axes=axes)
        state = state._replace(env_cfg=state.env_cfg._replace(
            dig_pull_min_length_m=1.0, pull_half_angle_rad=float(np.pi / 6)))
        legacy = state._replace(env_cfg=state.env_cfg._replace(
            dig_working_strip_width_m=1.3))
        pair = jax.tree.map(lambda a, b: jnp.stack([a, b]), state, legacy)

        def execute(s):
            after = s._handle_do()
            return (after.world.action_map.map, after._get_current_agent_state().loaded,
                    s._executable_fresh_dig_counts(), after.world.target_map.map,
                    after.world.padding_mask.map)

        action, loaded, counts, targets, obstacles = jax.jit(jax.vmap(execute))(pair)
        self.assertGreater(int(loaded[0, 0]), 0)
        for value in (action, loaded, counts):
            np.testing.assert_array_equal(value[0], value[1])
        self.assertEqual(int(loaded[0, 0]), int(counts[0, 0]))
        self.assertEqual(int(action[0].sum()) + int(loaded[0, 0]), 0)
        for i in range(2):
            np.testing.assert_array_equal(targets[i], target)
            np.testing.assert_array_equal(obstacles[i], state.world.padding_mask.map)

    def test_narrow_crosswise_cut_fails_but_junction_uses_connected_room(self):
        target = np.zeros(self.SHAPE, np.int8)
        target[24:26, 15:55] = -1
        state = self.state(target, position=(16, 40))
        self.assertFalse(bool(state._fresh_trench_pose_valid_cells()[2][24, 40]))
        target[12:50, 40:42] = -1
        state = self.state(target, position=(16, 40))
        self.assertTrue(bool(state._fresh_trench_pose_valid_cells()[2][24, 40]))
        self.assertFalse(bool(state._fresh_trench_pose_valid_cells()[2][24, 46]))

    def test_last_fresh_cell_can_use_previously_excavated_stroke_space(self):
        target = np.zeros(self.SHAPE, np.int8)
        target[24:26, 15:55] = -1
        action = target.copy()
        action[24, 39] = 0
        state = self.state(target, position=(24, 30), action=action)
        after = state._handle_do()
        self.assertEqual(int(after.world.action_map.map[24, 39]), -1)
        self.assertEqual(int(after._get_current_agent_state().loaded[0]), 1)

    def test_bulk_is_free_but_edge_pull_is_parallel(self):
        target = np.zeros(self.SHAPE, np.int8)
        target[20:40, 38:53] = -1
        state = self.state(target, position=(30, 28), precision=True)
        edge, allowed, _ = state._get_pull_boundary_details()
        self.assertFalse(bool(edge[30, 42]))
        self.assertTrue(bool(allowed[30, 42]))
        self.assertTrue(bool(edge[30, 38]))
        self.assertFalse(bool(allowed[30, 38]))  # pulling across the left edge
        along = self.state(target, position=(10, 38), precision=True)
        self.assertTrue(bool(along._get_pull_boundary_details()[1][25, 38]))

    def test_relift_is_unchanged_and_target_is_immutable(self):
        target, axes = self.strip()
        action = np.zeros(self.SHAPE, np.int8)
        action[24, 40] = 3
        new = self.state(target, axes=axes, position=(27, 30), action=action)
        old = new._replace(env_cfg=new.env_cfg._replace(pull_direction_alignment=False))
        for state in (new, old):
            result = state._handle_do()
            np.testing.assert_array_equal(result.world.target_map.map, target)
            self.assertEqual(int(result._get_current_agent_state().loaded[0]), 3)
        np.testing.assert_array_equal(new._handle_do().world.action_map.map,
                                      old._handle_do().world.action_map.map)

    def test_native_do_counts_and_jit_vmap_share_admission(self):
        target, axes = self.strip()
        state = self.state(target, axes=axes, position=(24, 30))
        execute = jax.jit(lambda s: (s._handle_do(), s._executable_fresh_dig_counts()))
        after, counts = execute(state)
        removed = np.sum(np.maximum(-np.asarray(after.world.action_map.map), 0))
        self.assertGreater(removed, 0)
        self.assertEqual(int(counts[0]), int(removed))
        np.testing.assert_array_equal(after.world.target_map.map, target)
        wrapped = LocalMapWrapper.wrap(state, executable_dig_observation=True)
        np.testing.assert_array_equal(wrapped.world.local_map_admissible_dig.map, counts)
        pair = jax.tree_util.tree_map(lambda x: jnp.stack([jnp.asarray(x)] * 2), state)
        actual = jax.jit(jax.vmap(lambda s: s._handle_do().world.action_map.map))(pair)
        np.testing.assert_array_equal(actual[0], after.world.action_map.map)
        np.testing.assert_array_equal(actual[1], after.world.action_map.map)

    def test_mode_requires_prepared_geometry_and_valid_tolerances(self):
        batch = object.__new__(TerraEnvBatch)
        batch.pull_direction_alignment = False
        with self.assertRaisesRegex(RuntimeError, "must match"):
            batch._validate_foundation_border_metadata_requirements(
                EnvConfig(pull_direction_alignment=True))
        batch.pull_direction_alignment = True
        for fields in ({"edge_band_width_m": 0}, {"edge_band_width_m": np.nan},
                       {"trench_pull_tolerance_rad": np.pi / 2},
                       {"dig_pull_min_length_m": 0}, {"dig_pull_min_length_m": np.nan}):
            with self.assertRaises(RuntimeError):
                batch._validate_foundation_border_metadata_requirements(
                    EnvConfig(pull_direction_alignment=True, **fields))
        batch._validate_foundation_border_metadata_requirements(
            EnvConfig(pull_direction_alignment=True))

    def test_preparation_preserves_legacy_foundation_dump_applicability(self):
        target, axes = self.strip()
        maps = MapsBuffer.new(
            maps=jnp.asarray(target[None, None]),
            padding_mask=jnp.zeros((1, 1, *self.SHAPE), jnp.int8),
            trench_axes=jnp.asarray(axes[None, None]),
            trench_types=jnp.ones((1, 1), jnp.int32),
            foundation_border_axes=jnp.full((1, 1, 64, 3), -97, jnp.float32),
            foundation_border_types=-jnp.ones((1, 1), jnp.int32),
            dumpability_masks_init=jnp.ones((1, 1, *self.SHAPE), bool),
            action_maps=jnp.zeros((1, 1, *self.SHAPE), jnp.int8),
            distance_maps=jnp.ones((1, 1, *self.SHAPE), jnp.float32),
        )
        cfg = BatchConfig()._replace(maps_dims=MapsDimsConfig(maps_edge_length=64))
        with patch("terra.env.init_maps_buffer", return_value=(maps, cfg)):
            env = TerraEnvBatch(batch_cfg=cfg, pull_direction_alignment=True)
        self.assertEqual(env.maps_buffer.foundation_border_axes.shape, (1, 1, MAX_BOUNDARY_SEGMENTS, 7))
        np.testing.assert_array_equal(env.maps_buffer.foundation_border_types, [[-1]])

    def test_trench_metadata_is_not_needed_for_new_geometric_rule(self):
        target, axes = self.strip()
        state = self.state(target, axes=axes)
        expected = np.asarray(state._fresh_trench_pose_valid_cells()[2])
        axes[:] = -97
        without = self.state(target, axes=axes)
        np.testing.assert_array_equal(without._fresh_trench_pose_valid_cells()[2], expected)
        batch = object.__new__(TerraEnvBatch)
        batch._validate_trench_alignment_metadata_requirements(
            EnvConfig(pull_direction_alignment=True, enforce_trench_dig_alignment=True))

    def test_precision_is_optional_and_independent_per_environment(self):
        target = np.zeros(self.SHAPE, np.int8)
        target[20:45, 30:53] = -1
        bulk = self.state(target, position=(30, 41))
        precise = bulk._replace(env_cfg=bulk.env_cfg._replace(enforce_foundation_border_alignment=True))
        # Pull across the right edge; there is sufficient room for a bulk cut.
        self.assertTrue(bool(bulk._get_pull_dig_permission()[30, 52]))
        self.assertFalse(bool(precise._get_pull_dig_permission()[30, 52]))
        self.assertFalse(np.any(bulk._get_foundation_border_mask()))
        self.assertTrue(np.any(precise._get_foundation_border_mask()))
        pair = jax.tree_util.tree_map(lambda a,b: jnp.stack([jnp.asarray(a),jnp.asarray(b)]), bulk, precise)
        masks = jax.jit(jax.vmap(lambda s: s._get_pull_dig_permission()))(pair)
        self.assertTrue(bool(masks[0, 30, 52]))
        self.assertFalse(bool(masks[1, 30, 52]))

    def test_precision_band_observation_matches_rule_and_is_immutable(self):
        target = np.zeros(self.SHAPE, np.int8)
        target[20:45, 30:53] = -1
        precise = self.state(target, position=(30, 41), precision=True)
        expected = np.asarray(precise._get_pull_boundary_details()[0])
        actual = TerraEnv._state_to_obs_dict(precise)["precision_required_band"]
        self.assertEqual(actual.dtype, jnp.bool_)
        self.assertEqual(actual.shape, self.SHAPE)
        np.testing.assert_array_equal(actual, expected)
        moved = precise._set_current_agent_state(
            precise._get_current_agent_state()._replace(pos_base=jnp.asarray([10, 10], jnp.int16))
        )
        moved = moved._replace(world=moved.world._replace(
            action_map=moved.world.action_map._replace(map=jnp.asarray(target))
        ))
        np.testing.assert_array_equal(moved._get_precision_required_band(), expected)
        bulk = precise._replace(env_cfg=precise.env_cfg._replace(enforce_foundation_border_alignment=False))
        legacy = precise._replace(env_cfg=precise.env_cfg._replace(pull_direction_alignment=False))
        states = jax.tree_util.tree_map(
            lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), precise, bulk, legacy
        )
        masks = jax.jit(jax.vmap(lambda s: s._get_precision_required_band()))(states)
        np.testing.assert_array_equal(masks[0], expected)
        self.assertFalse(np.any(masks[1:]))


if __name__ == "__main__":
    unittest.main()
