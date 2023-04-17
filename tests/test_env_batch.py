import unittest
import jax.numpy as jnp
from src.env import TerraEnvBatch
from src.config import EnvConfig
from src.utils import IntLowDim


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

        print(f"{states=}")

        print("###########################")

        # actions = jnp.array([TrackedActionType.FORWARD])[None].repeat(repeats=batch_size, axis=0)
        actions = jnp.array([0, 0], dtype=IntLowDim)

        _, (states1, reward, dones, infos) = env_batch.step(states, actions)

        print(states1)

        # self.assertFalse(states == states1)  # TODO implement __eq__

if __name__ == "__main__":
    unittest.main()
