import unittest
import jax.numpy as jnp
from src.state import State
from src.config import EnvConfig
from src.actions import TrackedActionType
from src.utils import IntMap


class TestAgent(unittest.TestCase):

    def test_create_state(self):
        seed = 33
        state = State.new(seed, env_cfg=EnvConfig())

        # print(state)
    
    def test_call_step_forward(self):
        seed = 25
        state = State.new(seed, env_cfg=EnvConfig())

        # print(state.agent.agent_state)

        action = TrackedActionType.FORWARD
        state = state._step(action)

        # print(state.agent.agent_state)


    def test_get_agent_corners(self):
        seed = 25
        state = State.new(seed, env_cfg=EnvConfig())
        
        corners = state._get_agent_corners(
            pos_base=state.agent.agent_state.pos_base,
            base_orientation=state.agent.agent_state.angle_base,
            agent_width=state.env_cfg.agent.width,
            agent_height=state.env_cfg.agent.height
        )

    def test_arm_extension(self):
        seed = 25
        state = State.new(seed, env_cfg=EnvConfig())

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
        # map_w = 4
        # map_flattened = jnp.arange(map_w * map_h)
        # map = map_flattened.reshape((map_w, map_h))
        # print(f"{map=}")
        # print(f"{map_flattened=}")

        idx = State._get_current_pos_vector_idx(pos_base, map_h)
        self.assertEqual(idx, 26)

    def test_map_to_flattened_global_coords(self):
        map_h = 2
        map_w = 3
        tile_size = 10.
        flat_map = State._map_to_flattened_global_coords(map_w, map_h, tile_size)
        # print(f"{flat_map=}")
        # print(f"{flat_map[0].reshape(map_w, map_h)=}")
        # print(f"{flat_map[1].reshape(map_w, map_h)=}")
        flat_map_gt = jnp.array([[5., 5., 15., 15., 25., 25.],
                                 [5., 15.,  5., 15.,  5., 15.]])
        self.assertTrue(jnp.allclose(flat_map, flat_map_gt))

    def test_get_current_pos_from_flattened_map(self):
        map_h = 2
        map_w = 3
        tile_size = 10.

        idx = jnp.full((1,), fill_value=2)
        flat_map = State._map_to_flattened_global_coords(map_w, map_h, tile_size)
        current_pos = State._get_current_pos_from_flattened_map(flat_map, idx)

        self.assertTrue(jnp.allclose(current_pos, jnp.array([15.,  5.])))

        # print(f"{flat_map=}")
        # print(f"{flat_map[0].reshape(map_w, map_h)=}")
        # print(f"{flat_map[1].reshape(map_w, map_h)=}")
        # print(f"{current_pos=}")

    def test_handle_dig(self):
        seed = 25
        state = State.new(seed, env_cfg=EnvConfig())

        # print(state.agent.agent_state)
        # print(f"{state=}")
        action = TrackedActionType.DO
        state = state._step(action)
        # print(f"{state=}")


if __name__ == "__main__":
    unittest.main()
