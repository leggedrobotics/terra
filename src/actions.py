from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

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
    # action: IntLowDim = IntLowDim(TrackedActionType.DO_NOTHING)
    action: Array = jnp.full(
        (1,), fill_value=TrackedActionType.DO_NOTHING, dtype=IntLowDim
    )

    @classmethod
    def new(cls, action: TrackedActionType) -> "TrackedAction":
        return TrackedAction(action=IntLowDim(action))

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

    @classmethod
    def random(cls, key: jnp.int32):
        return cls.new(
            jax.random.choice(
                key, jnp.arange(TrackedActionType.FORWARD, TrackedActionType.DO + 1)
            )
        )


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
    action: Array = jnp.full(
        (1,), fill_value=WheeledActionType.DO_NOTHING, dtype=IntLowDim
    )

    @classmethod
    def new(cls, action: WheeledActionType) -> "TrackedAction":
        return TrackedAction(action=IntLowDim(action))

    @classmethod
    def do_nothing(cls):
        return cls.new(WheeledActionType.DO_NOTHING)

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
    def random(cls, key: jnp.int32):
        return cls.new(
            jax.random.choice(
                key, jnp.arange(WheeledActionType.FORWARD, WheeledActionType.DO + 1)
            )
        )
