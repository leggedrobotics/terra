import unittest

import jax
import jax.numpy as jnp

from terra.actions import TrackedAction
from terra.config import EnvConfig
from terra.state import State
from terra.settings import IntMap


class TestAgent(unittest.TestCase):
    def test_create_state(self):
        seed = 33
        key = jax.random.PRNGKey(seed)
        State.new(key, env_cfg=EnvConfig())

    def test_call_step_forward(self):
        seed = 25
        key = jax.random.PRNGKey(seed)
        state = State.new(key, env_cfg=EnvConfig())

        action = TrackedAction.forward()
        state = state._step(action)

    def test_get_agent_corners(self):
        seed = 25
        key = jax.random.PRNGKey(seed)
        state = State.new(key, env_cfg=EnvConfig())

        state._get_agent_corners(
            pos_base=state.agent.agent_state.pos_base,
            base_orientation=state.agent.agent_state.angle_base,
            agent_width=state.env_cfg.agent.width,
            agent_height=state.env_cfg.agent.height,
        )

    def test_arm_extension(self):
        seed = 25
        key = jax.random.PRNGKey(seed)
        state = State.new(key, env_cfg=EnvConfig())

        state = state._handle_extend_arm()
        self.assertEqual(state.agent.agent_state.arm_extension, 1)
        state = state._handle_extend_arm()
        self.assertEqual(state.agent.agent_state.arm_extension, 1)
        state = state._handle_retract_arm()
        self.assertEqual(state.agent.agent_state.arm_extension, 0)
        state = state._handle_retract_arm()
        self.assertEqual(state.agent.agent_state.arm_extension, 0)

    def test_get_current_pos_vector_idx(self):
        pos_base = IntMap(jnp.array([3, 5]))
        map_h = 7

        idx = State._get_current_pos_vector_idx(pos_base, map_h)
        self.assertEqual(idx, 26)

    def test_map_to_flattened_global_coords(self):
        map_h = 2
        map_w = 3
        tile_size = 10.0
        flat_map = State._map_to_flattened_global_coords(map_w, map_h, tile_size)
        flat_map_gt = jnp.array(
            [[5.0, 5.0, 15.0, 15.0, 25.0, 25.0], [5.0, 15.0, 5.0, 15.0, 5.0, 15.0]]
        )
        self.assertTrue(jnp.allclose(flat_map, flat_map_gt))

    def test_get_current_pos_from_flattened_map(self):
        map_h = 2
        map_w = 3
        tile_size = 10.0

        idx = jnp.full((1,), fill_value=2)
        flat_map = State._map_to_flattened_global_coords(map_w, map_h, tile_size)
        current_pos = State._get_current_pos_from_flattened_map(flat_map, idx)

        self.assertTrue(jnp.allclose(current_pos, jnp.array([15.0, 5.0])))

    def test_handle_dig(self):
        seed = 25
        key = jax.random.PRNGKey(seed)
        state = State.new(key, env_cfg=EnvConfig())

        action = TrackedAction.do()
        state = state._step(action)

    def test_is_done(self):
        seed = 25
        key = jax.random.PRNGKey(seed)
        state = State.new(key, env_cfg=EnvConfig())

        loaded = jnp.full((1,), 0, dtype=jnp.int16)
        x = jnp.arange(3)[None].repeat(3, 0)
        target_map = x - jnp.max(x)

        action_map1 = jnp.ones((3, 3))
        self.assertFalse(state._is_done(action_map1, target_map, loaded))

        action_map2 = target_map.copy()
        self.assertTrue(state._is_done(action_map2, target_map, loaded))
        self.assertFalse(state._is_done(action_map2, target_map, loaded + 3))

        action_map3 = target_map.copy()
        action_map3 = action_map3.at[:, -1].set(action_map3[:, -1] + 20)
        self.assertTrue(state._is_done(action_map3, target_map, loaded))

        state = state._replace(env_steps=state.env_cfg.max_steps_in_episode + 1)
        self.assertTrue(state._is_done(action_map1, target_map, loaded))


if __name__ == "__main__":
    unittest.main()
