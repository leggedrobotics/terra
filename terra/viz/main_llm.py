import argparse
from .main_manual_llm import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an LLM-based simulation experiment.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the LLM model to use.")
    parser.add_argument("--model_key", type=str, required=True, help="Name of the LLM model key to use.")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of timesteps to run.")

    args = parser.parse_args()
    run_experiment(args.model_name, args.model_key, args.num_timesteps)