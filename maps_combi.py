"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

import numpy as np
import jax
from tqdm import tqdm
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action
from terra.state import State
import matplotlib.animation as animation

# from utils.curriculum import Curriculum
from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints
from terra.config import EnvConfig
from terra.config import BatchConfig

from terra.viz.llms_utils import *
from multi_agent_utils import *
from multi_agent_map import *
from terra.viz.llms_adk import *
from terra.viz.a_star import compute_path, simplify_path
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

import asyncio
import os
import argparse
import datetime
import json
import csv
import pygame as pg

from pygame.locals import (
    K_q,
    QUIT,
)

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

FORCE_DELEGATE_TO_RL = True    # Force delegation to RL agent for testing
FORCE_DELEGATE_TO_LLM = False   # Force delegation to LLM agent for testing
LLM_CALL_FREQUENCY = 15          # Number of steps between LLM calls
USE_IMAGE_PROMPT = True         # Use image prompt for LLM (Master Agent)
USE_LOCAL_MAP = True            # Use local map for LLM (Excavator Agent)
USE_PATH = True                 # Use path for LLM (Excavator Agent)
APP_NAME = "ExcavatorGameApp"   # Application name for ADK
USER_ID = "user_1"              # User ID for ADK
SESSION_ID = "session_001"      # Session ID for ADK

def create_composite_map(env, n_maps_x=2, n_maps_y=2, overlap=8):
    """
    Create a composite map by combining multiple 64x64 maps with optional overlap.
    
    Args:
        env: TerraEnvBatch environment object
        n_maps_x: Number of maps to combine horizontally
        n_maps_y: Number of maps to combine vertically
        overlap: Number of pixels to overlap between adjacent maps
        
    Returns:
        composite_maps: Dictionary containing composite map components
    """
    # Extract dimensions from the original map
    orig_height, orig_width = 64, 64
    
    # Calculate new dimensions considering overlap
    new_height = orig_height * n_maps_y - overlap * (n_maps_y - 1)
    new_width = orig_width * n_maps_x - overlap * (n_maps_x - 1)
    print(f"New dimensions: {new_height} x {new_width}")

    # We need to get a map from the environment first
    # Let's reset the environment and use its initial state to extract maps
    rng_key = jax.random.PRNGKey(0)
    
    # Reset the environment to get the initial TimeStep which contains maps
    timestep = env.reset(env.batch_cfg, rng_key)
    
    # Extract maps from the state observation
    target_map = timestep.observation["target_map"]
    padding_mask = timestep.observation["padding_mask"]
    dumpability_mask = timestep.observation["dumpability_mask"]
    


def run_experiment(llm_model_name, llm_model_key, num_timesteps, n_envs_x, n_envs_y, seed, progressive_gif, run):
    """
    Run an LLM-based simulation experiment.

    Args:
        model_name: The name of the LLM model to use.
        model_key: The name of the LLM model key to use.
        num_timesteps: The number of timesteps to run.
        n_envs_x: The number of environments along the x-axis.
        n_envs_y: The number of environments along the y-axis.
        seed: The random seed for reproducibility.
        progressive_gif: Whether to generate a progressive GIF (1 for True, 0 for False).
        run: The path to the RL agent checkpoint file.

    Returns:
        None
    """

    agent_checkpoint_path = run
    model = None
    model_params = None
    config = None
    n_envs_x = n_envs_x
    n_envs_y = n_envs_y
    n_envs = n_envs_x * n_envs_y

    print(f"Loading RL agent from: {agent_checkpoint_path}")
    log = load_pkl_object(agent_checkpoint_path)
    config = log["train_config"]
    config.num_test_rollouts = n_envs
    config.num_devices = 1    

  
    batch_cfg = BatchConfig()
    action_type = batch_cfg.action_type
    env_cfgs = log["env_config"]
    env_cfgs = jax.tree_map(
        lambda x: x[0][None, ...].repeat(n_envs, 0), env_cfgs
    ) 
    print(f"Using progressive_gif = {progressive_gif}")

    n_partitions_x = 2
    n_partitions_y = 2
   
    suffle_maps = True
    env = TerraEnvBatch(
        rendering=True,
        n_envs_x_rendering=n_envs_x,
        n_envs_y_rendering=n_envs_y,
        display=True,
        progressive_gif=progressive_gif,
        shuffle_maps=suffle_maps,
    )

    composite_env = create_composite_map(env, n_maps_x=n_partitions_x, n_maps_y=n_partitions_y)


    config.num_embeddings_agent_min = 60  # curriculum.get_num_embeddings_agent_min()
    model = load_neural_network(config, env)
    model_params = log["model"]
    
    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, config.num_test_rollouts)
    
    # Reset the environment with the composite map
    timestep = env.reset(composite_env, rng_reset)
    # Game loop setup
    print("Starting the game loop...")
    t_counter = 0
    reward_seq = []
    obs_seq = []
    action_list = []
    
    # Define the repeat_action function
    def repeat_action(action, n_times=n_envs_x * n_envs_y):
        return action_type.new(action.action[None].repeat(n_times, 0))
    
    timestep = env.step(timestep, repeat_action(action_type.do_nothing()), rng_reset)
    env.terra_env.render_obs_pygame(timestep.observation, timestep.info)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an LLM-based simulation experiment with RL agents.")
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
                 "gemini-2.5-flash-preview-04-17", 
                 "claude-3-haiku-20240307", 
                 "claude-3-7-sonnet-20250219"], 
        help="Name of the LLM model to use."
    )
    parser.add_argument(
        "--model_key", 
        type=str, 
        required=True, 
        choices=["gpt", 
                 "gemini", 
                 "claude"], 
        help="Name of the LLM model key to use."
    )
    parser.add_argument(
        "--num_timesteps", 
        type=int, 
        default=100, 
        help="Number of timesteps to run."
    )
    parser.add_argument(
        "-nx",
        "--n_envs_x",
        type=int,
        default=1,
        help="Number of environments on x.",
    )
    parser.add_argument(
        "-ny",
        "--n_envs_y",
        type=int,
        default=1,
        help="Number of environments on y.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-pg",
        "--progressive_gif",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        # default="/home/gioelemo/Documents/terra/new-maps-different-order.pkl",
        # help="new-maps-different-order.pkl (12 cabin and 12 base rotations)",
        # default="/home/gioelemo/Documents/terra/gioele.pkl",
        # help="gioele.pkl (8 cabin and 4 base rotations)",
        default="/home/gioelemo/Documents/terra/gioele_new.pkl",
        help="gioele_new.pkl (8 cabin and 4 base rotations) Version 7 May",
        #default="/home/gioelemo/Documents/terra/new-penalties.pkl",
        #help="new-penalties.pkl (12 cabin and 12 base rotations) Version 7 May",
    )

    args = parser.parse_args()
    run_experiment(args.model_name, 
                   args.model_key, 
                   args.num_timesteps, 
                   args.n_envs_x, 
                   args.n_envs_y, 
                   args.seed, 
                   args.progressive_gif, 
                   args.run_name
                   )
