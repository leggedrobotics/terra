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
from terra.env import TimeStep
from terra.env import TerraEnv

from terra.viz.llms_utils import *
from multi_agent_utils import *
from terra.viz.llms_adk import *
from terra.viz.a_star import compute_path, simplify_path
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)


class LargeMapTerraEnv(TerraEnvBatchWithMapOverride):
    """A version of TerraEnvBatch specifically for 128x128 maps"""
    
    def reset_with_map_override(self, env_cfgs, rngs, custom_pos=None, custom_angle=None,
                           target_map_override=None, traversability_mask_override=None,
                           padding_mask_override=None, dumpability_mask_override=None,
                           dumpability_mask_init_override=None, action_map_override=None,
                           dig_map_override=None, agent_config_override=None):
        """Reset with 64x64 map overrides - ensures shapes are consistent"""
    
        # Call the TerraEnvBatchWithMapOverride's reset_with_map_override method directly
        return TerraEnvBatchWithMapOverride.reset_with_map_override(
            self, env_cfgs, rngs, custom_pos, custom_angle,
            target_map_override, traversability_mask_override,
            padding_mask_override, dumpability_mask_override,
            dumpability_mask_init_override, action_map_override,
            dig_map_override, agent_config_override
        )