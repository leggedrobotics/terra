"""Bounded interactive reset checks and exact seeded parity with Agent.new."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.agent import Agent
from terra.config import EnvConfig
from terra.map import compute_dynamic_dumpability
from terra.viewer3d.spawn import validate_spawn


def fixture(agent_types=(0,), *, partial=False):
    cfg = EnvConfig()
    cfg = cfg._replace(
        agent_types=agent_types,
        action_types=tuple(0 for _ in agent_types),
        tile_size=cfg.maps.edge_length_m / 32,
        maps=cfg.maps._replace(edge_length_px=32),
        agent=cfg.agent._replace(width=3, height=5),
    )
    target = np.zeros((32, 32), dtype=np.int16)
    target[14:18, 22:25] = -1
    target[20:27, 5:9] = 1
    padding = np.zeros_like(target)
    padding[4:9, 19:24] = 1
    dumpability = np.ones_like(target, dtype=bool)
    dumpability[25:29, 20:28] = False
    action = np.zeros_like(target)
    if partial:
        action[15, 24] = -1
        action[21, 6] = 1
    maps = (
        jnp.asarray(target),
        jnp.asarray(padding),
        jnp.full((3, 3), -97.0),
        jnp.int32(-1),
        jnp.full((64, 3), -97.0),
        jnp.int32(-1),
        jnp.asarray(dumpability),
        jnp.asarray(action),
        jnp.zeros((32, 32)),
    )
    return cfg, maps


class SpawnPreflightTest(unittest.TestCase):
    def test_seeded_multiagent_poses_match_real_agent_sampler(self):
        cfg, maps = fixture((0, 1, 2), partial=True)
        padding, action = maps[1], maps[7]
        dynamic_dumpability = compute_dynamic_dumpability(maps[6], action)

        @jax.jit
        def actual_reset(key):
            return Agent.new(
                key,
                cfg,
                (padding[:, 0] == 0).sum(),
                (padding[0] == 0).sum(),
                padding,
                action,
                dynamic_dumpability,
                maps[0],
                agent_types=cfg.agent_types,
                action_types=cfg.action_types,
            )[0]

        for seed in (0, 7, 23):
            with self.subTest(seed=seed):
                poses = validate_spawn(seed, cfg, maps)
                actual = actual_reset(jax.random.PRNGKey(seed))
                self.assertEqual(len(poses), 3)
                for pose, state in zip(poses, actual.agent_states):
                    np.testing.assert_array_equal(pose["position"], state.pos_base)
                    self.assertEqual(
                        pose["angle_base"], int(np.asarray(state.angle_base)[0])
                    )
                    self.assertGreaterEqual(pose["attempts"], 1)

    def test_completely_blocked_map_fails_with_actionable_message(self):
        cfg, maps = fixture()
        maps = (*maps[:1], jnp.ones((32, 32), dtype=jnp.int16), *maps[2:])
        with self.assertRaisesRegex(ValueError, "4096.*Try another --seed"):
            validate_spawn(0, cfg, maps)

    def test_first_column_obstacles_preserve_runtime_clamped_sampling_domain(self):
        cfg, maps = fixture()
        padding = maps[1].at[:, 0].set(1)
        maps = (*maps[:1], padding, *maps[2:])
        # Most of the map is free, but Terra only samples row=2 here and the
        # excavator's minimum eight-cell border distance can never be met.
        with self.assertRaisesRegex(ValueError, "first row/column"):
            validate_spawn(0, cfg, maps)

    def test_dynamic_hole_clearance_is_part_of_spawning(self):
        cfg, maps = fixture()
        action = jnp.zeros((32, 32), dtype=jnp.int16).at[1::4, 1::4].set(-1)
        maps = (*maps[:6], jnp.ones((32, 32), dtype=bool), action, maps[8])
        self.assertFalse(np.asarray(compute_dynamic_dumpability(maps[6], action)).any())
        with self.assertRaisesRegex(ValueError, "flat, dumpable space"):
            validate_spawn(7, cfg, maps)

    def test_later_agent_cannot_share_the_only_spawn_footprint(self):
        cfg, maps = fixture((1, 2))
        padding = jnp.zeros((32, 32), dtype=jnp.int16)
        # One 3 x 5 footprint fits this 4 x 6 window (Terra rasterizes whole
        # cells since f3eeca6a); two non-overlapping footprints cannot.
        dumpability = jnp.zeros((32, 32), dtype=bool).at[14:18, 13:19].set(True)
        maps = (
            maps[0],
            padding,
            *maps[2:6],
            dumpability,
            jnp.zeros_like(padding),
            maps[8],
        )
        single_cfg = cfg._replace(agent_types=(1,), action_types=(0,))
        poses = validate_spawn(0, single_cfg, maps)
        self.assertEqual(len(poses), 1)
        with self.assertRaisesRegex(ValueError, "agent 2 of 2"):
            validate_spawn(0, cfg, maps)

    def test_unsupported_road_restrictions_and_invalid_bound_are_explicit(self):
        cfg, maps = fixture()
        with self.assertRaisesRegex(ValueError, "truck_road_restricted=False"):
            validate_spawn(0, cfg._replace(truck_road_restricted=True), maps)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_spawn(0, cfg, maps, max_attempts=0)


if __name__ == "__main__":
    unittest.main()
