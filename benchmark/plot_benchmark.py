import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    gpu_path = f"/home/antonio/thesis/benchmarks/terra/Benchmark_random_gpu:0_2023_04_20_14_50_44.csv"
    cpu_path = f"/home/antonio/thesis/benchmarks/terra/Benchmark_random_TFRT_CPU_0_2023_04_20_14_51_15.csv"
    
    # config_path = f"/media/benchmarks/terra/Config_{policy}_{device}_{now}.csv"

    gpu_df = pd.read_csv(gpu_path).to_dict()
    cpu_df = pd.read_csv(cpu_path).to_dict()

    batch_sizes = gpu_df["batch_size"].values()
    gpu_times = gpu_df["time"]
    cpu_times = cpu_df["time"]

    fig = plt.figure(0)
    plt.scatter(batch_sizes, gpu_times.values(), c="g", label="gpu")
    plt.plot(batch_sizes, gpu_times.values(), c="g", label="gpu")
    plt.scatter(batch_sizes, cpu_times.values(), c="r", label="cpu")
    plt.plot(batch_sizes, cpu_times.values(), c="r", label="cpu")
    plt.xlabel("number of environments")
    plt.ylabel("duration (s)")
    plt.title("GPU vs CPU - 100 steps per environment")
    plt.legend()
    plt.xscale("log")
    plt.show()
