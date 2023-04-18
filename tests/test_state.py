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
        
        print(f"{state.agent.agent_state.pos_base=}")

        corners = state._get_agent_corners(
            pos_base=state.agent.agent_state.pos_base,
            base_orientation=state.agent.agent_state.angle_base,
            agent_width=state.env_cfg.agent.width,
            agent_height=state.env_cfg.agent.height
        )

        print(f"{corners=}")


if __name__ == "__main__":
    unittest.main()
