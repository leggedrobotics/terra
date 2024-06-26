import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    policy = "random"
    n_steps_per_env = 100
    gpu_path = "/home/antonio/thesis/benchmarks/terra/Benchmark_random_gpu:0_2023_04_20_14_50_44.csv"
    cpu_path = "/home/antonio/thesis/benchmarks/terra/Benchmark_random_TFRT_CPU_0_2023_04_20_14_51_15.csv"

    # config_path = f"/media/benchmarks/terra/Config_{policy}_{device}_{now}.csv"

    gpu_df = pd.read_csv(gpu_path).to_dict()
    cpu_df = pd.read_csv(cpu_path).to_dict()

    batch_sizes = gpu_df["batch_size"].values()
    gpu_times = gpu_df["time"]
    cpu_times = cpu_df["time"]

    gpu_step_env = gpu_df["avg_step_env"]
    cpu_step_env = cpu_df["avg_step_env"]

    # Times
    fig = plt.figure(0)
    plt.scatter(batch_sizes, gpu_times.values(), c="g", label="gpu")
    plt.plot(batch_sizes, gpu_times.values(), c="g", label="gpu")
    plt.scatter(batch_sizes, cpu_times.values(), c="r", label="cpu")
    plt.plot(batch_sizes, cpu_times.values(), c="r", label="cpu")
    plt.xlabel("number of environments")
    plt.ylabel("duration (s)")
    plt.title(f"Terra - {n_steps_per_env} steps per environment - {policy} policy")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")

    fig = plt.figure(1)
    plt.scatter(batch_sizes, gpu_step_env.values(), c="g", label="gpu")
    plt.plot(batch_sizes, gpu_step_env.values(), c="g", label="gpu")
    plt.scatter(batch_sizes, cpu_step_env.values(), c="r", label="cpu")
    plt.plot(batch_sizes, cpu_step_env.values(), c="r", label="cpu")
    plt.xlabel("number of environments")
    plt.ylabel("avg step duration (s)")
    plt.title(f"Terra - {n_steps_per_env} steps per environment - {policy} policy")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")

    plt.show()
