from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from terra.settings import IntLowDim
from terra.settings import IntMap

ActionType = IntEnum


class TrackedActionType(ActionType):
    """
    Tracked robot specific actions.
    """

    NONE = -1
    FORWARD = 0
    BACKWARD = 1
    CLOCK = 2
    ANTICLOCK = 3
    CABIN_CLOCK = 4
    CABIN_ANTICLOCK = 5
    DO = 6
    DO_NOTHING = 7
    MOVE_TO_POSE = 8
    SET_CABIN_ANGLE = 9
    SET_BASE_ANGLE = 10


Action = NamedTuple


class TrackedAction(Action):
    type: Array = jnp.full((1,), fill_value=0, dtype=IntLowDim)
    action: Array = jnp.full(
        (1,), fill_value=TrackedActionType.NONE, dtype=IntLowDim
    )
    # Used by macro actions: [x, y, base_angle, cabin_angle].
    target_pose: Array = jnp.zeros((4,), dtype=IntMap)

    @classmethod
    def new(cls, action: TrackedActionType, target_pose: Array | None = None) -> "TrackedAction":
        if target_pose is None:
            target_pose = jnp.zeros((4,), dtype=IntMap)
        target_pose = jnp.asarray(target_pose, dtype=IntMap)
        if target_pose.shape[-1] == 3:
            target_pose = jnp.concatenate(
                [target_pose, jnp.zeros((*target_pose.shape[:-1], 1), dtype=IntMap)],
                axis=-1,
            )
        return TrackedAction(
            action=IntLowDim(action),
            type=jnp.zeros_like(action, dtype=IntLowDim),
            target_pose=target_pose,
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
    def do(cls):
        return cls.new(jnp.full((1,), TrackedActionType.DO, dtype=IntLowDim))

    @classmethod
    def move_to_pose(cls, x: int, y: int, base_angle: int):
        return cls.new(
            jnp.full((1,), TrackedActionType.MOVE_TO_POSE, dtype=IntLowDim),
            jnp.array([x, y, base_angle, 0], dtype=IntMap),
        )

    @classmethod
    def set_cabin_angle(cls, cabin_angle: int):
        return cls.new(
            jnp.full((1,), TrackedActionType.SET_CABIN_ANGLE, dtype=IntLowDim),
            jnp.array([0, 0, 0, cabin_angle], dtype=IntMap),
        )

    @classmethod
    def set_base_angle(cls, base_angle: int):
        return cls.new(
            jnp.full((1,), TrackedActionType.SET_BASE_ANGLE, dtype=IntLowDim),
            jnp.array([0, 0, base_angle, 0], dtype=IntMap),
        )

    @classmethod
    def random(cls, key: jnp.int32):
        # Include DO_NOTHING (index 7) as a selectable action
        choices = jnp.array([
            TrackedActionType.FORWARD,
            TrackedActionType.BACKWARD,
            TrackedActionType.CLOCK,
            TrackedActionType.ANTICLOCK,
            TrackedActionType.CABIN_CLOCK,
            TrackedActionType.CABIN_ANTICLOCK,
            TrackedActionType.DO,
            TrackedActionType.DO_NOTHING,
            TrackedActionType.MOVE_TO_POSE,
            TrackedActionType.SET_CABIN_ANGLE,
            TrackedActionType.SET_BASE_ANGLE,
        ], dtype=IntLowDim)
        return cls.new(jax.random.choice(key, choices, (1,)))

    @staticmethod
    def get_num_actions():
        # 11 actions including DO_NOTHING and macro pose/base/cabin actions.
        return 11


class WheeledActionType(ActionType):
    """
    Wheeled robot specific actions.
    """

    NONE = -1
    FORWARD = 0
    BACKWARD = 1
    WHEELS_LEFT = 2
    WHEELS_RIGHT = 3
    CABIN_CLOCK = 4
    CABIN_ANTICLOCK = 5
    DO = 6
    DO_NOTHING = 7
    MOVE_TO_POSE = 8
    SET_CABIN_ANGLE = 9
    SET_BASE_ANGLE = 10


class WheeledAction(Action):
    type: Array = jnp.full((1,), fill_value=1, dtype=IntLowDim)
    action: Array = jnp.full(
        (1,), fill_value=WheeledActionType.NONE, dtype=IntLowDim
    )
    # Used by macro actions: [x, y, base_angle, cabin_angle].
    target_pose: Array = jnp.zeros((4,), dtype=IntMap)

    @classmethod
    def new(cls, action: WheeledActionType, target_pose: Array | None = None) -> "WheeledAction":
        if target_pose is None:
            target_pose = jnp.zeros((4,), dtype=IntMap)
        target_pose = jnp.asarray(target_pose, dtype=IntMap)
        if target_pose.shape[-1] == 3:
            target_pose = jnp.concatenate(
                [target_pose, jnp.zeros((*target_pose.shape[:-1], 1), dtype=IntMap)],
                axis=-1,
            )
        return WheeledAction(
            action=IntLowDim(action),
            type=jnp.ones_like(action, dtype=IntLowDim),
            target_pose=target_pose,
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
    def wheels_left(cls):
        return cls.new(jnp.full((1,), WheeledActionType.WHEELS_LEFT, dtype=IntLowDim))

    @classmethod
    def wheels_right(cls):
        return cls.new(
            jnp.full((1,), WheeledActionType.WHEELS_RIGHT, dtype=IntLowDim)
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
    def do(cls):
        return cls.new(jnp.full((1,), WheeledActionType.DO, dtype=IntLowDim))

    @classmethod
    def move_to_pose(cls, x: int, y: int, base_angle: int):
        return cls.new(
            jnp.full((1,), WheeledActionType.MOVE_TO_POSE, dtype=IntLowDim),
            jnp.array([x, y, base_angle, 0], dtype=IntMap),
        )

    @classmethod
    def set_cabin_angle(cls, cabin_angle: int):
        return cls.new(
            jnp.full((1,), WheeledActionType.SET_CABIN_ANGLE, dtype=IntLowDim),
            jnp.array([0, 0, 0, cabin_angle], dtype=IntMap),
        )

    @classmethod
    def set_base_angle(cls, base_angle: int):
        return cls.new(
            jnp.full((1,), WheeledActionType.SET_BASE_ANGLE, dtype=IntLowDim),
            jnp.array([0, 0, base_angle, 0], dtype=IntMap),
        )

    @classmethod
    def random(cls, key: jnp.int32):
        # Include DO_NOTHING (index 7) as a selectable action
        choices = jnp.array([
            WheeledActionType.FORWARD,
            WheeledActionType.BACKWARD,
            WheeledActionType.WHEELS_LEFT,
            WheeledActionType.WHEELS_RIGHT,
            WheeledActionType.CABIN_CLOCK,
            WheeledActionType.CABIN_ANTICLOCK,
            WheeledActionType.DO,
            WheeledActionType.DO_NOTHING,
            WheeledActionType.MOVE_TO_POSE,
            WheeledActionType.SET_CABIN_ANGLE,
            WheeledActionType.SET_BASE_ANGLE,
        ], dtype=IntLowDim)
        return cls.new(jax.random.choice(key, choices, (1,)))

    @staticmethod
    def get_num_actions():
        # 11 actions including DO_NOTHING and macro pose/base/cabin actions.
        return 11
