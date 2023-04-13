import unittest
import jax.numpy as jnp
from src.agent import Agent
from src.config import EnvConfig


class TestAgent(unittest.TestCase):

    def test_create_agent(self):
        agent = Agent.new(EnvConfig())

        # print(agent)


if __name__ == "__main__":
    unittest.main()
