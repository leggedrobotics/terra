import unittest

import jax

from terra.agent import Agent
from terra.config import EnvConfig


class TestAgent(unittest.TestCase):
    def test_create_agent(self):
        seed = 3
        key = jax.random.PRNGKey(seed)
        Agent.new(key, EnvConfig())


if __name__ == "__main__":
    unittest.main()
