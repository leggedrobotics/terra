"""Structured requests preserve native geometry and count modeled time once."""

from functools import partial
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.structured_actions import (
    StructuredAction, StructuredClock, StructuredTimeConfig,
    structured_action_masks, structured_termination, structured_transition, _move,
)
from terra.tests import test_pull_direction_alignment as native_fixtures


@partial(jax.jit, static_argnums=(3,))
def advance(state, action, clock, cfg):
    return structured_transition(state._replace(env_cfg=cfg), action, clock=clock,
                                 time_budget_s=14400.)


@partial(jax.jit, static_argnums=(1,))
def masks(state, cfg):
    return structured_action_masks(state._replace(env_cfg=cfg))


@partial(jax.jit, static_argnums=(3,))
def native(state, action, distance, cfg):
    old = state._replace(env_cfg=cfg._replace(agent=cfg.agent._replace(move_tiles=distance)))
    return old._step(TrackedAction.new(jnp.asarray([action], jnp.int8)), turn=False)


@partial(jax.jit, static_argnums=(1,))
def footprint(state, cfg):
    return state._replace(env_cfg=cfg)._current_base_footprint_mask()


@partial(jax.jit, static_argnums=(3,))
def translation_probe(state, direction, amount, cfg):
    state = state._replace(env_cfg=cfg)
    cur = state._get_current_agent_state()
    orientation = jax.lax.cond(
        direction == 0,
        lambda: state._base_orientation_to_one_hot_forward(cur.angle_base),
        lambda: state._base_orientation_to_one_hot_backwards(cur.angle_base))
    legacy = state._replace(env_cfg=cfg._replace(agent=cfg.agent._replace(move_tiles=amount)))
    expected = legacy._move_on_orientation(orientation)
    actual = _move(state, direction, amount)
    return (expected._get_current_agent_state().pos_base,
            actual._get_current_agent_state().pos_base)


@partial(jax.jit, static_argnums=(1,))
def translation_masks(state, cfg):
    return structured_action_masks(state._replace(env_cfg=cfg))["move_mask"]


class StructuredActionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target = np.zeros((64, 64), np.int8)
        target[20:40, 32:53] = -1
        cls.state = native_fixtures.PullDirectionAlignmentTest().state(target, position=(30, 25))
        cls.cfg = cls.state.env_cfg._replace(
            max_steps_in_episode=900, pull_half_angle_rad=float(np.pi / 6),
            tracked_move_keeps_turn=True)
        cls.state = cls.state._replace(env_cfg=cls.cfg)
        cls.timing = StructuredTimeConfig()

    def step(self, state, action, amount=1, heading=-1, clock=StructuredClock()):
        return advance(state._replace(env_cfg=None),
                       StructuredAction(jnp.int32(action), jnp.int32(amount), jnp.int32(heading)),
                       clock, self.cfg)

    def mask(self, state):
        return masks(state._replace(env_cfg=None), self.cfg)

    def native(self, state, action, amount=5):
        return native(state._replace(env_cfg=None), jnp.int32(action), amount, self.cfg)

    @staticmethod
    def pose(state, base=None, cabin=None, loaded=None):
        cur = state._get_current_agent_state()
        values = {}
        if base is not None:
            values['angle_base'] = jnp.asarray([base], cur.angle_base.dtype)
        if cabin is not None:
            values['angle_cabin'] = jnp.asarray([cabin], cur.angle_cabin.dtype)
        if loaded is not None:
            values['loaded'] = jnp.asarray([loaded], cur.loaded.dtype)
        return state._set_current_agent_state(cur._replace(**values))

    def test_move_distance_keeps_original_rounding_and_native_rules(self):
        for direction, amount, heading in ((0, 1, 0), (0, 5, 1), (1, 3, 2)):
            before = self.pose(self.state, base=heading)
            actual = self.step(before, direction, amount)
            expected = self.native(before, direction, amount)
            np.testing.assert_array_equal(actual.state._get_current_agent_state().pos_base,
                                          expected._get_current_agent_state().pos_base)
            self.assertEqual(int(actual.state.env_steps), 1)
            self.assertEqual(int(actual.state.env_cfg.agent.move_tiles), 5)
            distance = np.linalg.norm(np.asarray(actual.state._get_current_agent_state().pos_base)
                                      - np.asarray(before._get_current_agent_state().pos_base)) * self.cfg.tile_size
            self.assertAlmostEqual(float(actual.duration_s), distance / self.timing.nav_speed_mps, places=5)
            self.assertEqual(float(actual.info['transition_mass_residual']), 0.)

    def test_one_cell_moves_only_at_cardinal_headings(self):
        for heading in range(12):
            state = self.pose(self.state, base=heading)
            before = state._get_current_agent_state().pos_base
            for direction in (0, 1):
                legacy, actual = translation_probe(state._replace(env_cfg=None),
                                                   jnp.int32(direction), jnp.int32(1), self.cfg)
                expected = legacy if heading % 3 == 0 else before
                np.testing.assert_array_equal(actual, expected,
                                              err_msg=f"heading={heading}, direction={direction}")
                legacy_two, actual_two = translation_probe(state._replace(env_cfg=None),
                                                           jnp.int32(direction), jnp.int32(2), self.cfg)
                np.testing.assert_array_equal(actual_two, legacy_two)
        for heading, allowed in ((0, True), (5, False)):
            state = self.pose(self.state, base=heading)
            move_mask = translation_masks(state._replace(env_cfg=None), self.cfg)
            np.testing.assert_array_equal(move_mask[:, 0], [allowed, allowed])

    def test_oblique_long_request_cannot_clip_to_one_cell(self):
        from terra.state import compute_swept_polygon_mask
        state = self.pose(self.state, base=5)
        cur = state._get_current_agent_state()
        endpoint, _ = translation_probe(state._replace(env_cfg=None), jnp.int32(0), jnp.int32(1), self.cfg)
        corners = state._get_agent_corners(cur.pos_base, cur.angle_base,
                                           self.cfg.agent.width, self.cfg.agent.height)
        swept = compute_swept_polygon_mask(corners, endpoint - cur.pos_base,
                                            state.world.width, state.world.height)
        state = state._replace(world=state.world._replace(
            static_traversability_base=state.world.static_traversability_base._replace(
                map=(~swept).astype(jnp.int8))))
        for amount in range(2, 6):
            legacy, actual = translation_probe(state._replace(env_cfg=None),
                                               jnp.int32(0), jnp.int32(amount), self.cfg)
            np.testing.assert_array_equal(legacy, endpoint)
            np.testing.assert_array_equal(actual, cur.pos_base)
        move_mask = translation_masks(state._replace(env_cfg=None), self.cfg)
        self.assertFalse(np.any(move_mask[0]))

    def test_loaded_motion_is_masked_and_wait_remains_available(self):
        state = self.pose(self.state, loaded=3)
        mask = self.mask(state)
        self.assertFalse(np.any(mask['move_mask']))
        self.assertFalse(np.any(mask['turn_mask']))
        self.assertTrue(bool(mask['action_mask'][7]))
        for action in (0, 1, 2, 3):
            result = self.step(state, action, 3)
            self.assertFalse(bool(result.info['action_had_effect']))
            self.assertEqual(float(result.duration_s), 0.)
            self.assertEqual(int(result.state.env_steps), 1)

    def test_turn_stops_at_blocked_intermediate_pose(self):
        # A 180-degree endpoint is clear but orientation 10 intersects a cell
        # that neither the original orientation nor first turn (11) covers.
        shape = lambda angle: np.asarray(footprint(
            self.pose(self.state, base=angle)._replace(env_cfg=None), self.cfg))
        possible = shape(10) & ~shape(0) & ~shape(11) & ~shape(6)
        self.assertTrue(possible.any())
        cell = tuple(np.argwhere(possible)[0])
        blocked = self.state.world.static_traversability_base.map.at[cell].set(1)
        state = self.state._replace(world=self.state.world._replace(
            static_traversability_base=self.state.world.static_traversability_base._replace(map=blocked)))
        first = self.native(state, 2)
        self.assertEqual(int(first._get_current_agent_state().angle_base[0]), 11)
        self.assertEqual(int(self.native(first, 2)._get_current_agent_state().angle_base[0]), 11)
        result = self.step(state, 2, 6)
        self.assertEqual(int(result.state._get_current_agent_state().angle_base[0]), 11)
        self.assertEqual(int(result.info['executed_turn_steps']), 1)
        self.assertAlmostEqual(float(result.duration_s), np.pi / 6 * self.timing.base_turn_s_per_rad, places=5)

    def test_heading_do_matches_native_and_uses_shortest_cabin_swing(self):
        state = self.pose(self.state, cabin=11)
        result = self.step(state, 6, heading=0)
        expected = self.native(self.pose(state, cabin=0), 6)
        self.assertTrue(bool(result.info['material_or_load_changed']))
        np.testing.assert_array_equal(result.state.world.action_map.map, expected.world.action_map.map)
        self.assertEqual(int(result.state._get_current_agent_state().loaded[0]),
                         int(expected._get_current_agent_state().loaded[0]))
        self.assertAlmostEqual(float(result.info['executed_cabin_rad']), np.pi / 6, places=5)
        self.assertEqual(int(result.state.env_steps), 1)
        self.assertEqual(int(result.state.retained_work_events[0]), 1)
        self.assertEqual(int(result.state.productive_workspace_cycles), 1)
        self.assertTrue(np.isfinite(float(result.reward)))  # cfg horizon=900 is intentionally nonlegacy

    def test_mask_matches_native_do_for_all_headings_and_relift(self):
        state = self.state
        actual = self.mask(state)['do_mask']
        for heading in range(12):
            expected = self.native(self.pose(state, cabin=heading), 6)
            self.assertEqual(bool(actual[heading]), int(expected._get_current_agent_state().loaded[0]) > 0)
        # No target is in the selected sector; a loose-soil pickup is still valid.
        far = np.zeros((64, 64), np.int8)
        far[2:8, 2:8] = -1
        cone = np.asarray(state._build_dig_dump_cone()).reshape((64, 64))
        outside = ~np.asarray(footprint(state._replace(env_cfg=None), self.cfg))
        cell = tuple(np.argwhere(cone & outside)[0])
        soil = jnp.zeros((64, 64), jnp.int8).at[cell].set(1)
        relift = state._replace(world=state.world._replace(
            target_map=state.world.target_map._replace(map=jnp.asarray(far)),
            action_map=state.world.action_map._replace(map=soil)))
        self.assertTrue(bool(self.mask(relift)['do_mask'][0]))
        lifted = self.step(relift, 6, heading=0)
        self.assertGreater(int(lifted.state._get_current_agent_state().loaded[0]), 0)
        self.assertEqual(float(lifted.info['reward_v2_fresh_dig_volume']), 0.)

    def test_dump_mask_uses_complete_native_outcome_and_no_heading_fallback(self):
        loaded = self.pose(self.state, loaded=3)
        actual = self.mask(loaded)['do_mask']
        for heading in range(12):
            expected = self.native(self.pose(loaded, cabin=heading), 6)
            self.assertEqual(bool(actual[heading]), int(expected._get_current_agent_state().loaded[0]) == 0)
        # Exclude all placement cells. A request cannot swing to another heading
        # or lose its load when the selected native unload is impossible.
        blocked = loaded._replace(world=loaded.world._replace(
            dumpability_mask=loaded.world.dumpability_mask._replace(map=jnp.zeros((64, 64), jnp.bool_))))
        self.assertFalse(np.any(self.mask(blocked)['do_mask']))
        result = self.step(blocked, 6, heading=5)
        self.assertFalse(bool(result.info['action_had_effect']))
        self.assertEqual(int(result.state._get_current_agent_state().angle_cabin[0]), 0)
        self.assertEqual(int(result.state._get_current_agent_state().loaded[0]), 3)

    def test_dump_does_not_double_count_material_and_visit_context_survives_navigation(self):
        dig = self.step(self.state, 6, heading=0)
        clock = StructuredClock(dig.info['time_visit_open'], dig.info['time_moved'])
        options = np.flatnonzero(np.asarray(self.mask(dig.state)['do_mask']))
        self.assertGreater(len(options), 0)
        dump = self.step(dig.state, 6, heading=int(options[0]), clock=clock)
        self.assertEqual(float(dump.info['work_loaded_m3']), 0.)
        self.assertFalse(bool(dump.info['time_new_setup']))
        self.assertAlmostEqual(float(dump.duration_s), float(dump.info['executed_cabin_rad'])
                               * self.timing.cabin_turn_s_per_rad, places=4)
        # An earlier move closes a visit even if the next work returns to its
        # original pose. The explicit clock preserves this beyond pose equality.
        next_visit = self.step(self.state, 6, heading=0, clock=StructuredClock(False, True))
        self.assertTrue(bool(next_visit.info['time_new_setup']))
        self.assertTrue(bool(next_visit.info['time_relocation']))
        self.assertAlmostEqual(float(next_visit.duration_s - dig.duration_s), 15., places=3)

    def test_time_budget_and_decision_cap_replace_native_450_termination(self):
        before = self.state._replace(env_steps=jnp.int32(449))
        waiting = self.step(before, 7)
        self.assertFalse(bool(waiting.info['timeout']))
        terminate = jax.jit(lambda s, elapsed: structured_termination(
            s._replace(env_cfg=self.cfg), elapsed, time_budget_s=100., decision_limit=500))
        not_done = terminate(waiting.state, jnp.float32(99.))
        self.assertFalse(bool(not_done.done))
        self.assertTrue(bool(terminate(waiting.state, jnp.float32(100.)).timeout))
        at_cap = waiting.state._replace(env_steps=jnp.int32(500))
        self.assertTrue(bool(terminate(at_cap, jnp.float32(0.)).timeout))
        complete = self.state._replace(world=self.state.world._replace(
            action_map=self.state.world.action_map._replace(map=self.state.world.target_map.map)))
        on_boundary = terminate(complete, jnp.float32(100.))
        after_boundary = terminate(complete, jnp.float32(100.01))
        self.assertTrue(bool(on_boundary.task_done))
        self.assertEqual(float(on_boundary.reward), 6.)
        self.assertFalse(bool(after_boundary.task_done))
        self.assertEqual(float(after_boundary.reward), -1.)


if __name__ == '__main__':
    unittest.main()
