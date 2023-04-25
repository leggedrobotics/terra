import unittest
import time
import jax
import jax.numpy as jnp
import numpy as np
from src.env import TerraEnvBatch
from src.config import EnvConfig, BatchConfig
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
        batch_size = 20
        episode_length = 10
        seeds = jnp.arange(batch_size)
        env_cfg = EnvConfig()
        batch_cfg = BatchConfig()

        env_batch = TerraEnvBatch(env_cfg=env_cfg)
        states = env_batch.reset(seeds)

        # TODO make this more elegant
        dummy = jnp.ones((batch_size))
        action_type_batch = jax.vmap(lambda actions: batch_cfg.action_type)(dummy)

        for i in range(episode_length):
            actions = np.random.randint(TrackedActionType.FORWARD, TrackedActionType.DO + 1, (batch_size))
            _, (states, reward, dones, infos) = env_batch.step(states, actions, action_type_batch)


if __name__ == "__main__":
    print(f"Device = {jnp.ones(1).device_buffer.device()}\n")
    unittest.main()
