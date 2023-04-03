import torch
from enum import IntEnum

class Actions(IntEnum):
    """
    Base class for actions.
    """


class WheeledActions(Actions):
    """
    Wheeled robot specific actions.
    """
    FORWARD = 0
    BACKWARD = 1
    CLOCK_FORWARD = 2
    CLOCK_BACKWARD = 3
    ANTICLOCK_FORWARD = 4
    ANTICLOCK_BACKWARD = 5
    CABIN_CLOCK = 6
    CABIN_ANTICLOCK = 7
    EXTEND_ARM = 8
    RETRACT_ARM = 9
    DO = 10


class TrackedActions(Actions):
    """
    Tracked robot specific actions.
    """
    FORWARD = 0
    BACKWARD = 1
    CLOCK = 2
    ANTICLOCK = 3
    CABIN_CLOCK = 4
    CABIN_ANTICLOCK = 5
    EXTEND_ARM = 6
    RETRACT_ARM = 7
    DO = 8


def action_from_frontend(action: Actions) -> torch.Tensor:
    """
    Encodes the action coming from the agent into tensor encoding.
    """
    pass

def action_to_frontend(action: torch.Tensor) -> Actions:
    """
    Encodes the action coming from the backend into frontend interface.
    """
    pass
