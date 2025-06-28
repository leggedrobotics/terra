#!/usr/bin/env python3
"""
Modified version of your experiment script with parallel execution support.
Save this as a new file (e.g., parallel_experiment.py) and run it.
"""

import multiprocessing as mp
from functools import partial
import numpy as np
import os
import sys

# Import your original module
# Assuming your original file is named 'experiment.py'
# Change this to match your actual filename
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all the necessary components from your original file
from llm.main_llm import *

# If compute_stats_llm is in a different module
from llm.eval_llm import compute_stats_llm


def run_single_experiment_wrapper(experiment_id, args, base_seed):
    """
    Wrapper function to run a single experiment with proper process isolation.
    This handles JAX device assignment and process-specific configuration.
    """
    import os
    import jax
    
    # Set process-specific environment variables
    # This helps with JAX device assignment
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    # Optionally, limit each process to specific CPU cores
    # This can help with performance on NUMA systems
    if hasattr(os, 'sched_setaffinity'):
        try:
            # Assign each process to a specific CPU core
            cpu_id = experiment_id % mp.cpu_count()
            os.sched_setaffinity(0, {cpu_id})
        except:
            pass  # Not critical if this fails
    
    # Each process gets its own seed
    seed = base_seed + experiment_id * 1000
    
    print(f"Process {mp.current_process().name}: Running experiment {experiment_id+1}/{args.n_envs} with seed {seed}")
    
    try:
        # Run the experiment
        info = run_experiment(
            args.model_name, 
            args.model_key, 
            args.num_timesteps, 
            seed,
            args.run_name,
            args.small_env_config if hasattr(args, 'small_env_config') else None
        )
        
        # Extract results
        result = {
            'experiment_id': experiment_id,
            'episode_done_once': info["episode_done_once"].item() if hasattr(info["episode_done_once"], 'item') else info["episode_done_once"],
            'episode_length': info["episode_length"].item() if hasattr(info["episode_length"], 'item') else info["episode_length"],
            'move_cumsum': info["move_cumsum"].item() if hasattr(info["move_cumsum"], 'item') else info["move_cumsum"],
            'do_cumsum': info["do_cumsum"].item() if hasattr(info["do_cumsum"], 'item') else info["do_cumsum"],
            'areas': info["areas"].item() if hasattr(info["areas"], 'item') else info["areas"],
            'dig_tiles_per_target_map_init': info["dig_tiles_per_target_map_init"].item() if hasattr(info["dig_tiles_per_target_map_init"], 'item') else info["dig_tiles_per_target_map_init"],
            'dug_tiles_per_action_map': info["dug_tiles_per_action_map"].item() if hasattr(info["dug_tiles_per_action_map"], 'item') else info["dug_tiles_per_action_map"],
        }
        
        print(f"Process {mp.current_process().name}: Completed experiment {experiment_id+1}")
        return result
        
    except Exception as e:
        print(f"Error in experiment {experiment_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function that sets up argument parsing and runs parallel experiments.
    """
    import argparse
    
    # Set multiprocessing start method to 'spawn' for better compatibility
    # This is important for JAX and CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Run an LLM-based simulation experiment with RL agents in parallel.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        choices=["gpt-4o", 
                 "gpt-4.1", 
                 "o4-mini", 
                 "o3", 
                 "o3-mini", 
                 "gemini-1.5-flash-latest", 
                 "gemini-2.0-flash", 
                 "gemini-2.5-pro-exp-03-25", 
                 "gemini-2.5-pro-preview-03-25",
                 "gemini-2.5-pro-preview-05-06",
                 "gemini-2.5-flash-preview-04-17", 
                 "gemini-2.5-flash-preview-05-20",
                 "claude-3-haiku-20240307", 
                 "claude-3-7-sonnet-20250219",
                 "claude-opus-4-20250514",
                 "claude-sonnet-4-20250514",
                 ], 
        help="Name of the LLM model to use."
    )
    parser.add_argument(
        "--model_key", 
        type=str, 
        required=True, 
        choices=["gpt", "gemini", "claude"], 
        help="Name of the LLM model key to use."
    )
    parser.add_argument(
        "--num_timesteps", 
        type=int, 
        default=100, 
        help="Number of timesteps to run."
    )
    parser.add_argument(
        "-n",
        "--n_envs",
        type=int,
        default=1,
        help="Number of environments",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        default="/home/gioelemo/Documents/terra/no-action-map.pkl",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of processes to use for parallel execution. Default: number of CPUs - 1",
    )
    parser.add_argument(
        "--disable_rendering",
        action="store_true",
        help="Disable rendering for parallel execution (recommended)",
    )
    
    args = parser.parse_args()
    
    # For parallel execution, it's recommended to disable rendering
    if args.n_envs > 1 and not args.disable_rendering:
        print("WARNING: Running multiple environments with rendering enabled may cause issues.")
        print("Consider using --disable_rendering flag for parallel execution.")
    
    NUM_ENVS = args.n_envs
    base_seed = args.seed
    
    # Determine number of processes to use
    if args.num_processes is None:
        # Use all available CPUs minus 1 to leave some resources for the system
        num_processes = min(mp.cpu_count() - 1, NUM_ENVS)
    else:
        num_processes = args.num_processes
    
    num_processes = max(1, num_processes)  # Ensure at least 1 process
    
    print(f"Running {NUM_ENVS} experiments using {num_processes} processes")
    print(f"Available CPUs: {mp.cpu_count()}")
    
    # Create a partial function with fixed arguments
    run_experiment_partial = partial(run_single_experiment_wrapper, args=args, base_seed=base_seed)
    
    # Run experiments in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Map experiment IDs to the worker function
        results = pool.map(run_experiment_partial, range(NUM_ENVS))
    
    # Filter out None results (failed experiments)
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("All experiments failed!")
        return
    
    # Sort results by experiment ID to maintain order
    valid_results.sort(key=lambda x: x['experiment_id'])
    
    # Extract results into lists
    episode_done_once_list = [r['episode_done_once'] for r in valid_results]
    episode_length_list = [r['episode_length'] for r in valid_results]
    move_cumsum_list = [r['move_cumsum'] for r in valid_results]
    do_cumsum_list = [r['do_cumsum'] for r in valid_results]
    areas_list = [r['areas'] for r in valid_results]
    dig_tiles_per_target_map_init_list = [r['dig_tiles_per_target_map_init'] for r in valid_results]
    dug_tiles_per_action_map_list = [r['dug_tiles_per_action_map'] for r in valid_results]
    
    print(f"\nSuccessfully completed {len(valid_results)}/{NUM_ENVS} experiments")
    
    # Compute statistics if enabled
    if COMPUTE_BENCH_STATS:
        compute_stats_llm(
            episode_done_once_list, 
            episode_length_list, 
            move_cumsum_list,
            do_cumsum_list, 
            areas_list, 
            dig_tiles_per_target_map_init_list,
            dug_tiles_per_action_map_list
        )


if __name__ == "__main__":
    main()