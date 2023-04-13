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


# class ActionBatch(NamedTuple):
#     """
#     Add the batch dimension to the actions.
#     """
#     data: Action  # Action[capacity, 1]

#     @staticmethod
#     def empty(capacity: int) -> "ActionBatch":
#         # 1. Generate a TrackedAction where action=-1 (do nothing)
#         # 2. unsqueeze and expand to capacity every member of TrackedAction (action)
#         # 3. store this in data
#         data = jax.tree_map(lambda x: x[None].repeat(capacity), TrackedAction.do_nothing())

#         return ActionBatch(data=data)


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


# class ActionQueue(NamedTuple):
#     """
#     Defines a FIFO queue of actions in an attribute-first fashion.
#     """
#     data: Action  # TODO not sure of the data type
#     front: torch.Tensor = torch.zeros((1,), dtype=torch.int)
#     rear: torch.Tensor = torch.zeros((1,), dtype=torch.int)
#     count: torch.Tensor = torch.zeros((1,), dtype=torch.int)

#     @staticmethod
#     def empty(capacity: int, agent_type: str) -> "ActionQueue":

#         # TODO make this more elegant
        
#         # TODO check that the following is actually equivalent to:
#         """
#         data = jax.tree_map(lambda x: x[None].repeat(capacity), UnitAction.do_nothing())
#         """

#         # TODO data type is wrong here: should be Action, it is Tensor
#         if agent_type == "WHEELED":
#             data = tree_map_repeat(torch.tensor(WheeledAction.do_nothing()), capacity)
#         elif agent_type == "TRACKED":
#             data = tree_map_repeat(torch.tensor(TrackedAction.do_nothing()), capacity)

#         print(f"{type(data)=}")

#         return ActionQueue(data=data)
    
#     def __eq__(self, other: "ActionQueue") -> bool:
        
#         print("HERE")

#         if not isinstance(other, ActionQueue):
#             return False

#         print(f"{self.data=}")

#         return (self.data == other.data and
#                 torch.equal(self.front, other.front) and
#                 torch.equal(self.rear, other.rear) and
#                 torch.equal(self.count, other.count))

#     @classmethod
#     def from_frontend(cls, actions: List[ActionFrontend], max_queue_size: int) -> "ActionQueue":
#         pass

#     def to_frontend(self) -> List[ActionFrontend]:
#         pass

#     def push_back(self, action: Action) -> "ActionQueue":
#         """
#         Push an action to the back of the queue.
#         """
#         pass

#     def push_front(self, action:Action) -> "ActionQueue":
#         """
#         Push an action to the front of the queue.
#         """
#         pass

#     def pop(self) -> Tuple[Action, "ActionQueue"]:
#         """
#         Remove an element from the front of the queue and return a tuple
#         containing the removed action and the updated queue.
#         """
#         pass

#     def peek(self) -> Action:
#         """
#         Return the front of the queue.
#         """
#         pass

#     def clear(self) -> "ActionQueue":
#         pass

#     def is_full(self) -> bool:
#         pass

#     def is_empty(self) -> bool:
#         pass

#     def __eq__(self, other: "ActionQueue") -> bool:
#         pass
