import unittest
import jax.numpy as jnp
from src.state import State
from src.config import EnvConfig
from src.actions import TrackedActionType


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
        # state = state._handle_retract_arm()
        # self.assertEqual(state.agent.agent_state.arm_extension, 0)
        # state = state._handle_retract_arm()
        # self.assertEqual(state.agent.agent_state.arm_extension, 0)


if __name__ == "__main__":
    unittest.main()
