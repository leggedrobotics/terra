"""Per-heading native dump observation agrees with what a loaded DO does."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import EnvConfig
from terra.dig_direction import boundary_records_from_mask
from terra.env import TerraEnvBatch
from terra.state import State
from terra.wrappers import LocalMapWrapper

SHAPE = (64, 64)
BASE = (32, 32)


def machine_state(target, native, loaded=3, cabin=0, last_dig=None):
    cfg = EnvConfig()._replace(
        tile_size=40 / 70,
        agent=EnvConfig().agent._replace(
            width=7, height=11, dig_min_radius_m=4.0, dump_max_radius_m=6.0,
            dug_clearance_m=0.57, centre_chassis_on_base=True),
        maps=EnvConfig().maps._replace(edge_length_px=64),
        agent_types=(0,), action_types=(0,),
        pull_direction_alignment=True,
        native_dump_observation=native,
    )
    boundaries, count = boundary_records_from_mask(target < 0)
    state = State.new(
        jax.random.PRNGKey(5), cfg, target, np.zeros(SHAPE, np.int8),
        np.full((8, 8), -97, np.float32), np.int32(0), boundaries, np.int32(count),
        np.ones(SHAPE, bool), np.zeros(SHAPE, np.int8),
        distance_map_override=np.ones(SHAPE, np.float32),
    )
    if last_dig is not None:
        state = state._replace(world=state.world._replace(
            last_dig_mask=state.world.last_dig_mask._replace(map=jnp.asarray(last_dig))))
    cur = state._get_current_agent_state()._replace(
        pos_base=jnp.asarray(BASE, jnp.int16), angle_base=jnp.asarray([0], jnp.int8),
        angle_cabin=jnp.asarray([cabin], jnp.int8), loaded=jnp.asarray([loaded], jnp.int8))
    return state._set_current_agent_state(cur)


def cone(state, cabin):
    cur = state._get_current_agent_state()._replace(angle_cabin=jnp.asarray([cabin], jnp.int8))
    return np.asarray(state._set_current_agent_state(cur)._build_dig_dump_cone()).reshape(SHAPE).astype(bool)


def task():
    """Accepted ground only beyond 6.0 m at heading 0, within 4.5-5.5 m at heading 6."""
    target = np.zeros(SHAPE, np.int8)
    target[2:6, 2:9] = -1
    probe = machine_state(target, False)
    rows, cols = np.indices(SHAPE)
    radius = np.hypot(rows - BASE[0], cols - BASE[1]) * 40 / 70
    target[cone(probe, 0) & (radius > 6.05)] = 1
    target[cone(probe, 6) & (radius > 4.5) & (radius < 5.5)] = 1
    return target


def do_outcome(state):
    before = np.asarray(state.world.action_map.map, np.int32)
    after = state._handle_dump()
    delta = np.asarray(after.world.action_map.map, np.int32) - before
    if int(after._get_current_agent_state().loaded[0]) != 0:
        return 0
    accepted = np.asarray(state._accepted_dump_mask())
    return 1 if np.all(accepted[delta > 0]) else -1


class NativeDumpObservationTest(unittest.TestCase):
    def test_sign_matches_loaded_do_at_every_heading(self):
        target = task()
        state = machine_state(target, True)
        counts = np.asarray(state._native_dump_counts())
        for offset in range(12):
            rotated = machine_state(target, True, cabin=offset)
            self.assertEqual(int(np.sign(counts[offset])), do_outcome(rotated), offset)
        self.assertGreater(int(counts[6]), 0)

    def test_dig_cone_count_overstates_reach_and_last_dig_exclusion_is_seen(self):
        target = task()
        old = LocalMapWrapper.wrap(machine_state(target, False), native_dump_observation=False)
        new = LocalMapWrapper.wrap(machine_state(target, True), native_dump_observation=True)
        old_counts = np.asarray(old.world.local_map_dumpability.map)
        new_counts = np.asarray(new.world.local_map_dumpability.map)
        self.assertGreater(int(old_counts[0]), 0)
        self.assertLessEqual(int(new_counts[0]), 0)  # accepted ground lies beyond 6.0 m
        self.assertEqual(do_outcome(machine_state(target, True, cabin=0)) > 0, False)
        # Soil just lifted inside heading 6's cone excludes that whole workspace.
        last_dig = cone(machine_state(target, True), 6)
        action = np.zeros(SHAPE, np.int8)
        spot = np.argwhere(last_dig & (target == 0))[0]
        action[tuple(spot)] = 1
        excluded = machine_state(target, True, last_dig=last_dig)
        excluded = excluded._replace(world=excluded.world._replace(
            action_map=excluded.world.action_map._replace(map=jnp.asarray(action))))
        self.assertLessEqual(int(np.asarray(excluded._native_dump_counts())[6]), 0)
        self.assertEqual(do_outcome(excluded._set_current_agent_state(
            excluded._get_current_agent_state()._replace(angle_cabin=jnp.asarray([6], jnp.int8)))) > 0, False)

    def test_default_keeps_the_dig_cone_count_and_empty_agent_is_evaluated_loaded(self):
        target = task()
        state = machine_state(target, False, loaded=0)
        dynamic = LocalMapWrapper.wrap(state).world.local_map_dumpability.map
        static = LocalMapWrapper.wrap(state, native_dump_observation=False).world.local_map_dumpability.map
        np.testing.assert_array_equal(dynamic, static)
        empty = machine_state(target, True, loaded=0)
        np.testing.assert_array_equal(np.asarray(empty._native_dump_counts()),
                                      np.asarray(machine_state(target, True, loaded=1)._native_dump_counts()))

    def test_batch_option_must_match_config(self):
        batch = object.__new__(TerraEnvBatch)
        batch.native_dump_observation = False
        batch.pull_direction_alignment = False
        cfg = EnvConfig()._replace(native_dump_observation=True)
        with self.assertRaisesRegex(RuntimeError, "native_dump_observation must match"):
            batch._validate_foundation_border_metadata_requirements(cfg)


if __name__ == '__main__':
    unittest.main()
