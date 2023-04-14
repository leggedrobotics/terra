import unittest
import jax.numpy as jnp
from src.env import TerraEnv, TerraEnvBatch
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


class TestEnvBatch(unittest.TestCase):

    def test_create_env_batch(self):
        batch_size = 2
        seeds = jnp.arange(batch_size)
        env_batch = TerraEnvBatch(env_cfg=EnvConfig())
        states = env_batch.reset(seeds)

        # print(states)

    def test_step_env_batch(self):
        batch_size = 2
        seeds = jnp.arange(batch_size)
        env_batch = TerraEnvBatch(env_cfg=EnvConfig())
        states = env_batch.reset(seeds)

        actions = jnp.array([TrackedActionType.FORWARD])[None].repeat(batch_size)

        print(f"{actions=}")

        # print(states)

        _, (states1, reward, dones, infos) = env_batch.step(states, actions)

        print(states1)

        # self.assertFalse(states == states1)  # TODO implement __eq__

if __name__ == "__main__":
    unittest.main()
