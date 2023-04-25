import time
from time import gmtime
from time import strftime

import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.actions import TrackedActionType
from src.config import EnvConfig
from src.env import TerraEnvBatch

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
        "batch_size": [],
        "time": [],
        "avg_step_batch": [],
        "avg_step_env": [],
    }
    for batch_size in batch_sizes:
        batch_size = int(batch_size)
        print("\n")
        print(f"{batch_size=}")
        print(f"{episode_length=}")

        seeds = np.random.randint(0, 1000000, (batch_size))
        env_batch = TerraEnvBatch(env_cfg=EnvConfig())
        states = env_batch.reset(seeds)

        duration = 0
        for i in range(episode_length):
            actions = np.random.randint(
                TrackedActionType.FORWARD, TrackedActionType.DO + 1, (batch_size)
            )
            s = time.time()
            _, (states, reward, dones, infos) = env_batch.step(states, actions)
            e = time.time()
            duration += e - s

        benchmark_dict["batch_size"].append(batch_size)
        benchmark_dict["time"].append(duration)
        benchmark_dict["avg_step_batch"].append((duration) / episode_length)
        benchmark_dict["avg_step_env"].append(
            (duration) / (episode_length * batch_size)
        )

        print(f"Duration = {benchmark_dict['time'][-1]}")
        print(
            f"Average step duration per batch = {benchmark_dict['avg_step_batch'][-1]}"
        )
        print(
            f"Average step duration per environment = {benchmark_dict['avg_step_env'][-1]}"
        )

    benchmark_df = pd.DataFrame(benchmark_dict)
    benchmark_df.to_csv(benchmark_path, index=False)
