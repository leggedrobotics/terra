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


# rl_agent_checkpoint_path = "good_discretized_policy.pkl"
# with open(rl_agent_checkpoint_path, "rb") as f:
#     rl_agent = pickle.load(f)
# print("RL agent loaded.")


rl_agent_checkpoint_path = "good_discretized_policy.pkl"
rl_model = None
rl_model_params = None
rl_config = None


try:
        print(f"Loading RL agent from: {rl_agent_checkpoint_path}")
        # Load the dictionary containing config, params, etc.
        rl_log = load_pkl_object(rl_agent_checkpoint_path)
        if rl_log:
            # Extract the training config (needed for model init and helpers)
            rl_config = rl_log.get("train_config")
            if not isinstance(rl_config, TrainConfig):
                 # If train_config wasn't saved directly, try loading from a default path
                 # This might be necessary depending on how good_discretized_policy.pkl was saved
                 print("TrainConfig not found directly in pkl, attempting fallback load...")
                 # Example fallback: Adjust path as needed
                 # fallback_config_path = "path/to/your/original/training/config.pkl"
                 # rl_config = load_pkl_object(fallback_config_path).get("train_config")
                 # OR initialize a default TrainConfig if appropriate
                 # rl_config = TrainConfig() # Adjust with necessary defaults
                 if rl_config is None:
                     raise ValueError("Could not load or determine rl_config (TrainConfig).")


            # Extract the trained model parameters
            rl_model_params = rl_log.get("model")
            if rl_model_params is None:
                 raise ValueError("RL model parameters ('model') not found in pkl file.")

            # Initialize the neural network structure using the loaded config
            # We need a temporary RNG key here, it doesn't affect the loaded params
            rng_init_model, rng = jax.random.split(rng)
            # Ensure rl_config has necessary attributes like num_embeddings_agent_min if needed by get_model_ready
            # You might need to set defaults if they aren't in the loaded config:
            # if not hasattr(rl_config, 'num_embeddings_agent_min'):
            #    rl_config.num_embeddings_agent_min = 60 # Example default from visualize.py
            rl_model, _ = get_model_ready(rng_init_model, rl_config, env) # Pass env for potential shape inference

            print("Successfully loaded RL agent model structure and parameters.")
        else:
            print(f"Error: Failed to load data from {rl_agent_checkpoint_path}")
except FileNotFoundError:
        print(f"Error: RL agent checkpoint not found at {rl_agent_checkpoint_path}")
except Exception as e:
        print(f"Error loading RL agent: {e}")
        # Ensure rl_model is None if loading fails
        rl_model = None
        rl_model_params = None
        rl_config = None

