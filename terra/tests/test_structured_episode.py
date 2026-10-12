"""Episode boundaries and bonuses use the same contract in play and PPO."""

from collections import namedtuple
from functools import partial
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.structured_actions import StructuredTimeConfig, structured_termination
from terra.structured_episode import (
    StructuredEpisodeConfig, structured_episode_limits, structured_episode_termination,
)
from terra.tests import test_pull_direction_alignment as native_fixtures


@partial(jax.jit, static_argnames=("config",))
def terminate(state, elapsed, config):
    return structured_episode_termination(state, elapsed, config=config)


class StructuredEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target = np.zeros((64, 64), np.int8)
        target[30, 40] = -2
        target[30, 50] = 1
        cls.unfinished = native_fixtures.PullDirectionAlignmentTest().state(target)
        action = np.zeros_like(target)
        action[30, 40] = -2
        action[30, 50] = 2
        cls.finished = cls.unfinished._replace(world=cls.unfinished.world._replace(
            action_map=cls.unfinished.world.action_map._replace(map=jnp.asarray(action))))
        cls.config = StructuredEpisodeConfig(time_budget_s=100., decision_limit=10,
                                            success_time_bonus=2., success_decision_bonus=1.)

    def test_completion_exactly_on_each_limit_succeeds_beyond_either_fails(self):
        for elapsed, steps, success in (
                (99., 9, True), (100., 9, True), (99., 10, True), (100., 10, True),
                (100.01, 9, False), (99., 11, False), (100.01, 11, False)):
            with self.subTest(elapsed=elapsed, steps=steps):
                terminal = terminate(self.finished._replace(env_steps=jnp.int32(steps)),
                                     jnp.float32(elapsed), self.config)
                self.assertTrue(bool(terminal.done))
                self.assertEqual(bool(terminal.task_done), success)
                self.assertEqual(bool(terminal.timeout), not success)
                expected = (6. + 2. * max(1. - elapsed / 100., 0.)
                            + max(1. - steps / 10., 0.)) if success else -1.
                self.assertAlmostEqual(float(terminal.reward), expected, places=5)

    def test_unfinished_states_get_no_success_bonus_and_fail_at_the_limit(self):
        for elapsed, steps, failed in ((25., 4, False), (100., 4, True), (25., 10, True)):
            terminal = terminate(self.unfinished._replace(env_steps=jnp.int32(steps)),
                                 jnp.float32(elapsed), self.config)
            self.assertFalse(bool(terminal.task_done))
            self.assertEqual(bool(terminal.done), failed)
            self.assertEqual(float(terminal.reward), -1. if failed else 0.)
        terminal = terminate(self.finished._replace(env_steps=jnp.int32(4)),
                             jnp.float32(25.), self.config)
        self.assertAlmostEqual(float(terminal.reward), 8.1, places=5)

    def test_disabled_time_deadline_keeps_reference_and_decision_cap(self):
        config = self.config._replace(time_limit=False)
        budget, decisions = structured_episode_limits(self.unfinished, config=config)
        self.assertEqual(float(budget), 100.)
        self.assertEqual(int(decisions), 10)
        for elapsed in (100., 1000.):
            pending = terminate(self.unfinished._replace(env_steps=jnp.int32(4)),
                                jnp.float32(elapsed), config)
            self.assertFalse(bool(pending.done))
            success = terminate(self.finished._replace(env_steps=jnp.int32(4)),
                                jnp.float32(elapsed), config)
            self.assertTrue(bool(success.task_done))
            self.assertAlmostEqual(float(success.reward), 6.6, places=5)
            failed = terminate(self.unfinished._replace(env_steps=jnp.int32(10)),
                               jnp.float32(elapsed), config)
            self.assertTrue(bool(failed.timeout))
            self.assertEqual(float(failed.reward), -1.)

    def test_native_loaded_agent_cannot_finish_even_with_completed_terrain(self):
        loaded = self.finished._get_current_agent_state()._replace(loaded=jnp.array([1], jnp.int8))
        state = self.finished._set_current_agent_state(loaded)
        terminal = terminate(state, jnp.float32(25.), self.config)
        self.assertFalse(bool(terminal.task_done))
        self.assertFalse(bool(terminal.done))
        self.assertEqual(float(terminal.reward), 0.)

    def test_exact_dead_end_failure_respects_protocol_flag_and_never_gets_bonus(self):
        for enabled in (True, False):
            terminal = structured_episode_termination(
                self.unfinished, 25., config=self.config._replace(terminate_loaded_dead_ends=enabled),
                dead_end=True)
            self.assertEqual(bool(terminal.dead_end), enabled)
            self.assertEqual(bool(terminal.done), enabled)
            self.assertFalse(bool(terminal.timeout))
            self.assertFalse(bool(terminal.task_done))
            self.assertEqual(float(terminal.reward), -1. if enabled else 0.)

    def test_default_protocol_matches_original_fixed_budget_termination(self):
        config = StructuredEpisodeConfig()
        for state in (self.unfinished, self.finished):
            for elapsed, steps in ((0., 0), (14400., 450), (14401., 450), (0., 451)):
                state = state._replace(env_steps=jnp.int32(steps))
                original = structured_termination(state, elapsed, time_budget_s=14400., decision_limit=450)
                shared = structured_episode_termination(state, elapsed, config=config)
                for actual, expected in zip(shared, original):
                    np.testing.assert_array_equal(actual, expected)

    def test_map_scaled_limits_use_depth_and_maximum_not_addition(self):
        Map = namedtuple("Map", "map")
        World = namedtuple("World", "target_map")
        Config = namedtuple("Config", "tile_size")
        State = namedtuple("State", "env_cfg world")
        config = StructuredEpisodeConfig(time_budget_s=100., time_budget_factor=3.,
            time_budget_offset_s=50., decision_limit=300, decisions_per_dig_unit=1.2)
        timing = StructuredTimeConfig(dig_s_per_m3=120.)
        limits = jax.jit(lambda state: structured_episode_limits(state, config=config, timing=timing))
        for target, budget, cap in (([[-1, 1, 0]], 100., 300),
                                    ([[-200, -200, -101]], 50. + 3. * 15. * 501., 602)):
            state = State(Config(jnp.float32(.5)), World(Map(jnp.asarray(target, jnp.int16))))
            actual_budget, actual_cap = limits(state)
            self.assertEqual(float(actual_budget), budget)
            self.assertEqual(int(actual_cap), cap)
        # A stale offset alone does not enable map scaling.
        fixed = config._replace(time_budget_factor=0., decisions_per_dig_unit=0.)
        actual_budget, actual_cap = structured_episode_limits(state, timing=timing, config=fixed)
        self.assertEqual(float(actual_budget), 100.)
        self.assertEqual(int(actual_cap), 300)


if __name__ == "__main__":
    unittest.main()
