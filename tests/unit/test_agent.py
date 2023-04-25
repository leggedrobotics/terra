import unittest

from src.agent import Agent
from src.config import EnvConfig


class TestAgent(unittest.TestCase):
    def test_create_agent(self):
        Agent.new(EnvConfig())


if __name__ == "__main__":
    unittest.main()
