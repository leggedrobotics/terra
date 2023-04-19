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

    def _test_step_action(self, action):
        seed = 29
        env = TerraEnv(env_cfg=EnvConfig())
        state = env.reset(seed)

        # print(state)
        _, (state1, reward, dones, infos) = env.step(state, action)

        # print(state)

        # self.assertFalse(state == state1)  # TODO implement __eq__

        return state1

    def test_step_fwd(self):
        state = self._test_step_action(TrackedActionType.FORWARD)
    
    def test_step_bkwd(self):
        state = self._test_step_action(TrackedActionType.BACKWARD)
    
    def test_step_clock(self):
        state = self._test_step_action(TrackedActionType.CLOCK)
    
    def test_step_anticlock(self):
        state = self._test_step_action(TrackedActionType.ANTICLOCK)
    
    def test_step_cabin_clock(self):
        state = self._test_step_action(TrackedActionType.CABIN_CLOCK)
    
    def test_step_cabin_anticlock(self):
        state = self._test_step_action(TrackedActionType.CABIN_ANTICLOCK)

    def test_step_extend_arm(self):
        state = self._test_step_action(TrackedActionType.EXTEND_ARM)

    def test_step_retract_arm(self):
        state = self._test_step_action(TrackedActionType.RETRACT_ARM)

    def test_step_do(self):
        action = TrackedActionType.DO
        seed = 29
        env = TerraEnv(env_cfg=EnvConfig())
        state = env.reset(seed)

        # Dig
        _, (state1, reward, dones, infos) = env.step(state, action)
        # Dump
        _, (state2, reward, dones, infos) = env.step(state1, action)


if __name__ == "__main__":
    unittest.main()
