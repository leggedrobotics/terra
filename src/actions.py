import jax
from enum import IntEnum
from typing import NamedTuple
from src.utils import IntLowDim

ActionType = IntEnum

class TrackedActionType(ActionType):
    """
    Tracked robot specific actions.
    """
    DO_NOTHING = -1
    FORWARD = 0
    BACKWARD = 1
    CLOCK = 2
    ANTICLOCK = 3
    CABIN_CLOCK = 4
    CABIN_ANTICLOCK = 5
    EXTEND_ARM = 6
    RETRACT_ARM = 7
    DO = 8

Action = NamedTuple

class TrackedAction(Action):
    action: IntLowDim = IntLowDim(TrackedActionType.DO_NOTHING)

    @classmethod
    def new(cls, action: TrackedActionType) -> "TrackedAction":
        return TrackedAction(
            action=IntLowDim(action)
        )

    @classmethod
    def do_nothing(cls):
        return cls.new(TrackedActionType.DO_NOTHING)

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


# class WheeledActionType(ActionType):
#     """
#     Wheeled robot specific actions.
#     """
#     DO_NOTHING = -1
#     FORWARD = 0
#     BACKWARD = 1
#     CLOCK_FORWARD = 2
#     CLOCK_BACKWARD = 3
#     ANTICLOCK_FORWARD = 4
#     ANTICLOCK_BACKWARD = 5
#     CABIN_CLOCK = 6
#     CABIN_ANTICLOCK = 7
#     EXTEND_ARM = 8
#     RETRACT_ARM = 9
#     DO = 10


# class WheeledAction(Action):
#     action: torch.Tensor = torch.tensor([WheeledActionType.DO_NOTHING], dtype=torch.int)

#     @classmethod
#     def new(cls, action: WheeledActionType) -> "WheeledAction":
#         return WheeledAction(
#             action=torch.tensor([action], dtype=torch.int)
#         )

#     @classmethod
#     def do_nothing(cls):
#         return cls.new(WheeledActionType.DO_NOTHING)

#     @classmethod
#     def forward(cls):
#         return cls.new(WheeledActionType.FORWARD)
    
#     @classmethod
#     def backward(cls):
#         return cls.new(WheeledActionType.BACKWARD)
    
#     @classmethod
#     def clock_forward(cls):
#         return cls.new(WheeledActionType.CLOCK_FORWARD)
    
#     @classmethod
#     def clock_backward(cls):
#         return cls.new(WheeledActionType.CLOCK_BACKWARD)
    
#     @classmethod
#     def anticlock_forward(cls):
#         return cls.new(WheeledActionType.ANTICLOCK_FORWARD)
    
#     @classmethod
#     def anticlock_backward(cls):
#         return cls.new(WheeledActionType.ANTICLOCK_BACKWARD)
    
#     @classmethod
#     def cabin_clock(cls):
#         return cls.new(WheeledActionType.CABIN_CLOCK)
    
#     @classmethod
#     def cabin_anticlock(cls):
#         return cls.new(WheeledActionType.CABIN_ANTICLOCK)
    
#     @classmethod
#     def extend_arm(cls):
#         return cls.new(WheeledActionType.EXTEND_ARM)
    
#     @classmethod
#     def retract_arm(cls):
#         return cls.new(WheeledActionType.RETRACT_ARM)
    
#     @classmethod
#     def do(cls):
#         return cls.new(WheeledActionType.DO)
    
#     def __eq__(self, other: "WheeledAction") -> bool:
#         return torch.equal(self.action, other.action)

#     @classmethod
#     def from_frontend(cls, action_frontend: WheeledActionFrontend) -> "WheeledAction":
#         pass

#     def to_frontend(self) -> WheeledActionFrontend:
#         pass


