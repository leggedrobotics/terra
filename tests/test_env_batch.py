import unittest
import time
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
        batch_size = int(1 * 1e4)
        episode_length = 100
        print(f"{batch_size=}")
        print(f"{episode_length=}")

        seeds = jnp.arange(batch_size)
        env_batch = TerraEnvBatch(env_cfg=EnvConfig())
        states = env_batch.reset(seeds)

        # print(f"{states=}")

        # print("###########################")

        # actions = jnp.array([TrackedActionType.FORWARD])[None].repeat(repeats=batch_size, axis=0)
        # actions = jnp.zeros((batch_size), dtype=IntLowDim)
        s = time.time()
        for i in range(episode_length):
            # print(f"{i=}")
            actions = np.random.randint(TrackedActionType.FORWARD, TrackedActionType.DO + 1, (batch_size))
            # print(f"{actions=}")
            _, (states, reward, dones, infos) = env_batch.step(states, actions)
        e = time.time()

        print(f"Duration = {e-s}")
        print(f"Average step duration per batch = {(e-s) / episode_length}")
        print(f"Average step duration per environment = {(e-s) / (episode_length * batch_size)}")
        # self.assertFalse(states == states1)  # TODO implement __eq__


if __name__ == "__main__":
    print(f"Device = {jnp.ones(1).device_buffer.device()}\n")
    unittest.main()
