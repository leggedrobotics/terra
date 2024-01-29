from typing import Dict, NamedTuple, Any
import jax

from terra.state import State, Infos

class Transition(NamedTuple):
    next_state: State
    obs: Any  # Replace with a more specific type as needed
    reward: float
    done: bool
    infos: Infos

