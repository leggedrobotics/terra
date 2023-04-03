import torch
from enum import IntEnum
from typing import NamedTuple, List, Tuple
from frontend import WheeledActionFrontend, TrackedActionFrontend, ActionFrontend

class ActionType(IntEnum):
    """
    Base class for actions.
    """


class WheeledActionType(ActionType):
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


class TrackedActionType(ActionType):
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


class Action(NamedTuple):
    pass
    

class WheeledAction(Action):
    action: torch.Tensor

    @classmethod
    def new(cls, action_type: WheeledActionType) -> "WheeledAction":
        pass

    @classmethod
    def forward(cls):
        return cls.new(WheeledActionType.FORWARD)
    
    @classmethod
    def backward(cls):
        return cls.new(WheeledActionType.BACKWARD)
    
    @classmethod
    def clock_forward(cls):
        return cls.new(WheeledActionType.CLOCK_FORWARD)
    
    @classmethod
    def clock_backward(cls):
        return cls.new(WheeledActionType.CLOCK_BACKWARD)
    
    @classmethod
    def anticlock_forward(cls):
        return cls.new(WheeledActionType.ANTICLOCK_FORWARD)
    
    @classmethod
    def anticlock_backward(cls):
        return cls.new(WheeledActionType.ANTICLOCK_BACKWARD)
    
    @classmethod
    def cabin_clock(cls):
        return cls.new(WheeledActionType.CABIN_CLOCK)
    
    @classmethod
    def cabin_anticlock(cls):
        return cls.new(WheeledActionType.CABIN_ANTICLOCK)
    
    @classmethod
    def extend_arm(cls):
        return cls.new(WheeledActionType.EXTEND_ARM)
    
    @classmethod
    def retract_arm(cls):
        return cls.new(WheeledActionType.RETRACT_ARM)
    
    @classmethod
    def do(cls):
        return cls.new(WheeledActionType.DO)

    @classmethod
    def from_frontend(cls, action_frontend: WheeledActionFrontend) -> "WheeledAction":
        pass

    def to_frontend(self) -> WheeledActionFrontend:
        pass


class TrackedAction(Action):
    action: torch.Tensor

    @classmethod
    def new(cls, action_type: TrackedActionType) -> "TrackedAction":
        pass

    @classmethod
    def forward(cls):
        return cls.new(TrackedActionType.FORWARD)
    
    @classmethod
    def backward(cls):
        return cls.new(TrackedActionType.BACKWARD)
    
    @classmethod
    def clock(cls):
        return cls.new(TrackedActionType.CLOCK)
    
    @classmethod
    def anticlock(cls):
        return cls.new(TrackedActionType.ANTICLOCK)
        
    @classmethod
    def cabin_clock(cls):
        return cls.new(TrackedActionType.CABIN_CLOCK)
    
    @classmethod
    def cabin_anticlock(cls):
        return cls.new(TrackedActionType.CABIN_ANTICLOCK)
    
    @classmethod
    def extend_arm(cls):
        return cls.new(TrackedActionType.EXTEND_ARM)
    
    @classmethod
    def retract_arm(cls):
        return cls.new(TrackedActionType.RETRACT_ARM)
    
    @classmethod
    def do(cls):
        return cls.new(TrackedActionType.DO)

    @classmethod
    def from_frontend(cls, action_frontend: TrackedActionFrontend) -> "WheeledAction":
        pass

    def to_frontend(self) -> TrackedActionFrontend:
        pass


class ActionQueue(NamedTuple):
    """
    Defines a FIFO queue of actions in an attribute-first fashion.
    """
    data: Action
    front: torch.Tensor
    rear: torch.Tensor
    count: torch.Tensor

    @staticmethod
    def empty(capacity: int) -> "ActionQueue":
        pass

    @classmethod
    def from_frontend(cls, actions: List[ActionFrontend], max_queue_size: int) -> "ActionQueue":
        pass

    def to_frontend(self) -> List[ActionFrontend]:
        pass

    def push_back(self, action: Action) -> "ActionQueue":
        """
        Push an action to the back of the queue.
        """
        pass

    def push_front(self, action:Action) -> "ActionQueue":
        """
        Push an action to the front of the queue.
        """
        pass

    def pop(self) -> Tuple[Action, "ActionQueue"]:
        """
        Remove an element from the front of the queue and return a tuple
        containing the removed action and the updated queue.
        """
        pass

    def peek(self) -> Action:
        """
        Return the front of the queue.
        """
        pass

    def clear(self) -> "ActionQueue":
        pass

    def is_full(self) -> bool:
        pass

    def is_empty(self) -> bool:
        pass

    def __eq__(self, other: "ActionQueue") -> bool:
        pass


def wheeled_action_from_frontend(action: ActionType) -> torch.Tensor:
    """
    # TODO not sure of the interfaces
    Encodes the action coming from the agent into tensor encoding.
    """
    pass


def tracked_action_to_frontend(action: torch.Tensor) -> ActionType:
    """
    # TODO not sure of the interfaces
    Encodes the action coming from the backend into frontend interface.
    """
    pass
