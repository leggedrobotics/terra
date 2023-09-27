import unittest

from terra.actions import TrackedAction
from terra.config import EnvConfig
from terra.env import TerraEnv


class TestEnv(unittest.TestCase):
    def test_create_env(self):
        seed = 27
        env = TerraEnv(env_cfg=EnvConfig())
        env.reset(seed)

        # print(state)

    def _test_step_action(self, action):
        seed = 29
        env = TerraEnv(env_cfg=EnvConfig())
        state, obs = env.reset(seed)
        _, (state1, reward, dones, infos) = env.step(state, action)

        return state1

    def test_step_fwd(self):
        self._test_step_action(TrackedAction.forward())

    def test_step_bkwd(self):
        self._test_step_action(TrackedAction.backward())

    def test_step_clock(self):
        self._test_step_action(TrackedAction.clock())

    def test_step_anticlock(self):
        self._test_step_action(TrackedAction.anticlock())

    def test_step_cabin_clock(self):
        self._test_step_action(TrackedAction.cabin_clock())

    def test_step_cabin_anticlock(self):
        self._test_step_action(TrackedAction.cabin_anticlock())

    def test_step_extend_arm(self):
        self._test_step_action(TrackedAction.extend_arm())

    def test_step_retract_arm(self):
        self._test_step_action(TrackedAction.retract_arm())

    def test_step_do(self):
        action = TrackedAction.do()
        seed = 29
        env = TerraEnv(env_cfg=EnvConfig())
        state, obs = env.reset(seed)

        # Dig
        state1, (obs, reward, dones, infos) = env.step(state, action)
        # Dump
        state2, (obs, reward, dones, infos) = env.step(state1, action)


if __name__ == "__main__":
    unittest.main()
