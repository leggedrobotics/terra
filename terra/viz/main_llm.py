import argparse
from .main_manual_llm import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an LLM-based simulation experiment.")
    parser.add_argument("--modelname", type=str, required=True, help="Name of the LLM model to use.")
    parser.add_argument("--model", type=str, required=True, help="Name of the LLM model to use.")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of timesteps to run.")

    args = parser.parse_args()
    print(args.modelname)
    run_experiment(args.modelname, args.model, args.num_timesteps)