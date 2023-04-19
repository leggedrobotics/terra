import unittest
import jax.numpy as jnp
import numpy as np
from src.env import TerraEnvBatch
from src.config import EnvConfig
from src.utils import IntLowDim
from src.actions import TrackedActionType


class TestEnvBatch(unittest.TestCase):

    def test_create_env_batch(self):
        batch_size = 2
        seeds = jnp.arange(batch_size)
        env_batch = TerraEnvBatch(env_cfg=EnvConfig())
        states = env_batch.reset(seeds)

        # print(states)

    def test_step_env_batch(self):
        batch_size = 10
        seeds = jnp.arange(batch_size)
        env_batch = TerraEnvBatch(env_cfg=EnvConfig())
        states = env_batch.reset(seeds)

        # print(f"{states=}")

        # print("###########################")

        # actions = jnp.array([TrackedActionType.FORWARD])[None].repeat(repeats=batch_size, axis=0)
        # actions = jnp.zeros((batch_size), dtype=IntLowDim)
        actions = np.random.randint(TrackedActionType.FORWARD, TrackedActionType.DO, (batch_size))

        _, (states1, reward, dones, infos) = env_batch.step(states, actions)
        _, (states2, reward, dones, infos) = env_batch.step(states1, actions)

        # print(states1)

        # self.assertFalse(states == states1)  # TODO implement __eq__

if __name__ == "__main__":
    print(f"Device = {jnp.ones(1).device_buffer.device()}\n")
    unittest.main()
