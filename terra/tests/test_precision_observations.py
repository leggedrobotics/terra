"""Remaining precision edges and structured-heading observation contracts."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import EnvConfig
from terra.dig_direction import boundary_records_from_mask
from terra.state import State
from terra.wrappers import LocalMapWrapper


LOCAL_FIELDS = (
    "local_map_target_pos", "local_map_target_neg",
    "local_map_action_pos", "local_map_action_neg",
    "local_map_dumpability", "local_map_obstacles",
    "local_map_border_workspace", "local_map_edge_alignment_error",
    "local_map_border_diggable", "local_map_admissible_dig",
)


def observe(state, v2=True):
    wrapped = LocalMapWrapper.wrap(
        state, executable_dig_observation=True, native_dump_observation=True,
        precision_observation_v2=v2,
    )
    return {name: getattr(wrapped.world, name).map for name in LOCAL_FIELDS}


class PrecisionObservationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shape = (24, 24)
        target = np.zeros(shape, np.int8)
        target[8:17, 10:19] = -1
        target[2:6, 5:10] = 1
        cfg = EnvConfig()._replace(
            tile_size=1.0,
            agent=EnvConfig().agent._replace(width=3, height=5, dig_radius_tiles=6),
            maps=EnvConfig().maps._replace(edge_length_px=shape[0]),
            agent_types=(0,), action_types=(0,),
            pull_direction_alignment=True, enforce_foundation_border_alignment=True,
            dig_pull_min_length_m=1.0,
        )
        boundary, count = boundary_records_from_mask(target < 0)
        state = State.new(
            jax.random.PRNGKey(7), cfg, target, np.zeros(shape, np.int8),
            np.full((4, 8), -97, np.float32), np.int32(0), boundary, np.int32(count),
            np.ones(shape, bool), np.zeros(shape, np.int8),
            distance_map_override=np.ones(shape, np.float32),
        )
        current = state._get_current_agent_state()._replace(
            pos_base=jnp.asarray([10, 4], jnp.int16),
            angle_base=jnp.asarray([2], jnp.int8),
            angle_cabin=jnp.asarray([3], jnp.int8), loaded=jnp.asarray([0], jnp.int8),
        )
        cls.state = state._set_current_agent_state(current)
        cls.observe_v2 = staticmethod(jax.jit(observe))
        cls.observe_legacy = staticmethod(jax.jit(lambda s: observe(s, False)))

    @staticmethod
    def with_action(state, action):
        return state._replace(world=state.world._replace(
            action_map=state.world.action_map._replace(map=jnp.asarray(action, jnp.int8))))

    def test_chassis_frame_and_historical_opt_out(self):
        state = self.state
        zero_cabin = state._set_current_agent_state(
            state._get_current_agent_state()._replace(angle_cabin=jnp.asarray([0], jnp.int8)))
        current, zero = self.observe_v2(state), self.observe_v2(zero_cabin)
        for name in LOCAL_FIELDS:
            np.testing.assert_allclose(current[name], zero[name], atol=2e-5, err_msg=name)
        legacy = self.observe_legacy(state)
        for name in LOCAL_FIELDS[:6] + ("local_map_admissible_dig",):
            np.testing.assert_array_equal(current[name], np.roll(legacy[name], 3), err_msg=name)
        self.assertEqual(current["local_map_edge_alignment_error"].dtype, jnp.float32)
        self.assertEqual(legacy["local_map_edge_alignment_error"].dtype, jnp.int16)
        errors = np.asarray(current["local_map_edge_alignment_error"])
        self.assertTrue(np.any(np.abs(errors - np.trunc(errors)) > 1e-3))
        # The default remains the old frame and integer representation.
        default = LocalMapWrapper.wrap(
            state, executable_dig_observation=True, native_dump_observation=True)
        for name in LOCAL_FIELDS:
            np.testing.assert_array_equal(getattr(default.world, name).map, legacy[name])

    def test_remaining_band_never_creates_interior_edges(self):
        state = self.state
        band = np.asarray(state._get_foundation_border_mask())
        target = np.asarray(state.world.target_map.map)
        action = np.where(band, -1, 0).astype(np.int8)
        # An interior hole does not add its perimeter to the precision band.
        action[12, 14] = -1
        partial = self.with_action(state, action)
        observed = self.observe_v2(partial)
        for name in ("local_map_border_workspace", "local_map_border_diggable",
                     "local_map_edge_alignment_error"):
            self.assertTrue(np.all(np.asarray(observed[name]) == 0), name)
        self.assertGreater(int(np.sum(self.observe_legacy(partial)["local_map_border_workspace"])), 0)
        np.testing.assert_array_equal(partial._get_precision_required_band(), band)
        np.testing.assert_array_equal(partial.world.target_map.map, target)

    def test_edge_and_volume_counts_match_native_digs_and_relift_precedence(self):
        state = self.state
        observed = self.observe_v2(state)
        edge_counts = np.asarray(observed["local_map_border_diggable"])
        self.assertGreater(int(edge_counts.max()), 0)
        heading = int(np.argmax(edge_counts))
        current = state._get_current_agent_state()
        aimed = state._set_current_agent_state(current._replace(
            angle_cabin=jnp.asarray([heading], jnp.int8)))
        cone = np.asarray(aimed._build_dig_dump_cone()).reshape(24, 24).astype(bool)
        neutral = cone & (np.asarray(state.world.target_map.map) == 0)
        self.assertTrue(neutral.any())
        action = np.zeros((24, 24), np.int8)
        action[tuple(np.argwhere(neutral)[0])] = 1
        spoiled = self.with_action(state, action)
        spoiled_obs = self.observe_v2(spoiled)
        self.assertEqual(int(spoiled_obs["local_map_border_diggable"][heading]), 0)
        self.assertEqual(int(spoiled_obs["local_map_admissible_dig"][heading]), 0)

        def dig_outcomes(s):
            def dig(h):
                aimed = s._set_current_agent_state(s._get_current_agent_state()._replace(
                    angle_cabin=jnp.asarray([h], jnp.int8)))
                return aimed._handle_dig().world.action_map.map
            return jax.vmap(dig)(jnp.arange(12, dtype=jnp.int32))

        pair = jax.tree.map(lambda a, b: jnp.stack((jnp.asarray(a), jnp.asarray(b))), state, spoiled)
        outcomes = np.asarray(jax.jit(jax.vmap(dig_outcomes))(pair))
        band = np.asarray(state._get_foundation_border_mask())
        for index, obs in enumerate((observed, spoiled_obs)):
            removed = outcomes[index] < 0
            np.testing.assert_array_equal(obs["local_map_border_diggable"], (removed & band).sum((1, 2)))
            np.testing.assert_array_equal(obs["local_map_admissible_dig"], removed.sum((1, 2)))


if __name__ == "__main__":
    unittest.main()
