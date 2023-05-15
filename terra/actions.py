from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.utils import IntLowDim

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
    type: Array = jnp.full((1,), fill_value=0, dtype=IntLowDim)
    action: Array = jnp.full(
        (1,), fill_value=TrackedActionType.DO_NOTHING, dtype=IntLowDim
    )

    @classmethod
    def new(cls, action: TrackedActionType) -> "TrackedAction":
        return TrackedAction(
            action=IntLowDim(action), type=jnp.zeros_like(action, dtype=IntLowDim)
        )

    @classmethod
    def do_nothing(cls):
        return cls.new(jnp.full((1,), TrackedActionType.DO_NOTHING, dtype=IntLowDim))

    @classmethod
    def forward(cls):
        return cls.new(jnp.full((1,), TrackedActionType.FORWARD, dtype=IntLowDim))

    @classmethod
    def backward(cls):
        return cls.new(jnp.full((1,), TrackedActionType.BACKWARD, dtype=IntLowDim))

    @classmethod
    def clock(cls):
        return cls.new(jnp.full((1,), TrackedActionType.CLOCK, dtype=IntLowDim))

    @classmethod
    def anticlock(cls):
        return cls.new(jnp.full((1,), TrackedActionType.ANTICLOCK, dtype=IntLowDim))

    @classmethod
    def cabin_clock(cls):
        return cls.new(jnp.full((1,), TrackedActionType.CABIN_CLOCK, dtype=IntLowDim))

    @classmethod
    def cabin_anticlock(cls):
        return cls.new(
            jnp.full((1,), TrackedActionType.CABIN_ANTICLOCK, dtype=IntLowDim)
        )

    @classmethod
    def extend_arm(cls):
        return cls.new(jnp.full((1,), TrackedActionType.EXTEND_ARM, dtype=IntLowDim))

    @classmethod
    def retract_arm(cls):
        return cls.new(jnp.full((1,), TrackedActionType.RETRACT_ARM, dtype=IntLowDim))

    @classmethod
    def do(cls):
        return cls.new(jnp.full((1,), TrackedActionType.DO, dtype=IntLowDim))

    @classmethod
    def random(cls, key: jnp.int32):
        return cls.new(
            jax.random.choice(
                key,
                jnp.arange(TrackedActionType.FORWARD, TrackedActionType.DO + 1),
                (1,),
            )
        )

    @staticmethod
    def get_num_actions():
        return 9

    # def __hash__(self) -> int:
    #     return hash((self.type[0],))

    # def __eq__(self, __o: "TrackedAction") -> bool:
    #     return (jnp.all(self.type == __o.type)) & (jnp.all(self.action == __o.action))


class WheeledActionType(ActionType):
    """
    Wheeled robot specific actions.
    """

    DO_NOTHING = -1
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


class WheeledAction(Action):
    type: Array = jnp.full((1,), fill_value=1, dtype=IntLowDim)
    action: Array = jnp.full(
        (1,), fill_value=WheeledActionType.DO_NOTHING, dtype=IntLowDim
    )

    @classmethod
    def new(cls, action: WheeledActionType) -> "TrackedAction":
        return TrackedAction(
            action=IntLowDim(action), type=jnp.ones_like(action, dtype=IntLowDim)
        )

    @classmethod
    def do_nothing(cls):
        return cls.new(jnp.full((1,), WheeledActionType.DO_NOTHING, dtype=IntLowDim))

    @classmethod
    def forward(cls):
        return cls.new(jnp.full((1,), WheeledActionType.FORWARD, dtype=IntLowDim))

    @classmethod
    def backward(cls):
        return cls.new(jnp.full((1,), WheeledActionType.BACKWARD, dtype=IntLowDim))

    @classmethod
    def clock_forward(cls):
        return cls.new(jnp.full((1,), WheeledActionType.CLOCK_FORWARD, dtype=IntLowDim))

    @classmethod
    def clock_backward(cls):
        return cls.new(
            jnp.full((1,), WheeledActionType.CLOCK_BACKWARD, dtype=IntLowDim)
        )

    @classmethod
    def anticlock_forward(cls):
        return cls.new(
            jnp.full((1,), WheeledActionType.ANTICLOCK_FORWARD, dtype=IntLowDim)
        )

    @classmethod
    def anticlock_backward(cls):
        return cls.new(
            jnp.full((1,), WheeledActionType.ANTICLOCK_BACKWARD, dtype=IntLowDim)
        )

    @classmethod
    def cabin_clock(cls):
        return cls.new(jnp.full((1,), WheeledActionType.CABIN_CLOCK, dtype=IntLowDim))

    @classmethod
    def cabin_anticlock(cls):
        return cls.new(
            jnp.full((1,), WheeledActionType.CABIN_ANTICLOCK, dtype=IntLowDim)
        )

    @classmethod
    def extend_arm(cls):
        return cls.new(jnp.full((1,), WheeledActionType.EXTEND_ARM, dtype=IntLowDim))

    @classmethod
    def retract_arm(cls):
        return cls.new(jnp.full((1,), WheeledActionType.RETRACT_ARM, dtype=IntLowDim))

    @classmethod
    def do(cls):
        return cls.new(jnp.full((1,), WheeledActionType.DO, dtype=IntLowDim))

    @classmethod
    def random(cls, key: jnp.int32):
        return cls.new(
            jax.random.choice(
                key,
                jnp.arange(WheeledActionType.FORWARD, WheeledActionType.DO + 1),
                (1,),
            )
        )

    @staticmethod
    def get_num_actions():
        return 11

    # def __hash__(self) -> int:
    #     return hash((self.type[0],))

    # def __eq__(self, __o: "WheeledAction") -> bool:
    #     return (jnp.all(self.type == __o.type)) & (jnp.all(self.action == __o.action))
