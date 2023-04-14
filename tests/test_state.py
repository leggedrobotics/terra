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
    
    def test_step_forward(self):
        seed = 25
        state = State.new(seed, env_cfg=EnvConfig())

        # print(state)

        action = TrackedActionType.FORWARD
        state = state._step(action)

        # print(state)


if __name__ == "__main__":
    unittest.main()
