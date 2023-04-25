import unittest
import jax
import jax.numpy as jnp
from src.env import TerraEnvBatch
from src.config import EnvConfig, BatchConfig


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

        keys = jax.vmap(jax.random.PRNGKey)(seeds)
        ks = jax.vmap(jax.random.split)(keys)
        keys = ks[..., 0]
        subkeys = ks[..., 1]

        for i in range(episode_length):
            actions = jax.vmap(batch_cfg.action_type.random)(subkeys)

            # jax.debug.print("actions {i} = {x}", x=actions, i=i)

            _, (states, reward, dones, infos) = env_batch.step(states, actions)
            ks = jax.vmap(jax.random.split)(keys)
            keys = ks[..., 0]
            subkeys = ks[..., 1]


if __name__ == "__main__":
    print(f"Device = {jnp.ones(1).device_buffer.device()}\n")
    unittest.main()
