import unittest
from src.env import TerraEnv
from src.config import EnvConfig
from src.actions import TrackedActionType


class TestEnv(unittest.TestCase):

    def test_create_env(self):
        seed = 27
        env = TerraEnv(env_cfg=EnvConfig())
        state = env.reset(seed)

        # print(state)

    def test_step_env(self):
        seed = 29
        env = TerraEnv(env_cfg=EnvConfig())
        state = env.reset(seed)

        # print(state)
        action = TrackedActionType.FORWARD
        _, (state1, reward, dones, infos) = env.step(state, action)

        # print(state)

        # self.assertFalse(state == state1)  # TODO implement __eq__

if __name__ == "__main__":
    unittest.main()
