import torch
from src.env import TerraEnv
from typing import Tuple, Generator, Dict

def load_replay(replay: str) -> Tuple[TerraEnv, Generator[Dict, None, None]]:
    """
    Loads a replay from json format.

    Args:
        replay (str) path to the replay json file

    Returns:
        terra environment (TerraEnv)
        actions generator (Generator[Dict, None, None]), where one action is a dictionary
    """
    pass
