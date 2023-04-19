import time
import jax.numpy as jnp
import numpy as np
from src.env import TerraEnvBatch
from src.config import EnvConfig
from src.actions import TrackedActionType
from time import gmtime, strftime
import pandas as pd

if __name__ == "__main__":
    policy = "random"

    device = jnp.ones(1).device_buffer.device()
    print(f"Device = {device}\n")

    now = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    benchmark_path = f"/media/benchmarks/terra/Benchmark_{policy}_{device}_{now}.csv"
    config_path = f"/media/benchmarks/terra/Config_{policy}_{device}_{now}.csv"

    batch_sizes = [1e3, 1e4, 1e5, 1e6]
    episode_length = 100

    benchmark_dict = {
        "time": [],
        "avg_step_batch": [],
        "avg_step_env": []
    }
    for batch_size in range(batch_sizes):
        print("\n")
        print(f"{batch_size=}")
        print(f"{episode_length=}")

        seeds = np.random.randint(0, 1000000, (batch_size))
        env_batch = TerraEnvBatch(env_cfg=EnvConfig())
        states = env_batch.reset(seeds)

        duration = 0
        for i in range(episode_length):
            actions = np.random.randint(TrackedActionType.FORWARD, TrackedActionType.DO + 1, (batch_size))
            s = time.time()
            _, (states, reward, dones, infos) = env_batch.step(states, actions)
            e = time.time()
            duration += e-s

        print(f"Duration = {e-s}")
        print(f"Average step duration per batch = {(duration) / episode_length}")
        print(f"Average step duration per environment = {(duration) / (episode_length * batch_size)}")
