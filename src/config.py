from typing import NamedTuple

from src.actions import Action
from src.actions import TrackedAction  # noqa: F401
from src.actions import WheeledAction  # noqa: F401
from src.utils import Float


class MapDims(NamedTuple):
    tile_size: Float = 1.5  # in meters
    width_m: Float = 40.0  # in meters
    height_m: Float = 40.0  # in meters


class MapConfig(NamedTuple):
    width: int = round(MapDims().width_m / MapDims().tile_size)
    height: int = round(MapDims().height_m / MapDims().tile_size)


class TargetMapConfig(MapConfig):
    pass


class ActionMapConfig(MapConfig):
    pass


class AgentConfig(NamedTuple):
    angles_base: int = 4
    angles_cabin: int = 8

    move_tiles: int = 2  # number of tiles of progress for every move action
    #  Note: move_tiles is also used as radius of excavation
    #       (we dig as much as move_tiles in the radial distance)

    max_arm_extension: int = 1  # numbering starts from 0 (0 is the closest level)

    dig_depth: int = 1  # how much every dig action digs
    # max_dig: int = -3  # soft max after which the agent pays a cost  # TODO implement
    # max_dump: int = 3  # soft max after which the agent pays a cost  # TODO implement

    height: int = (
        round(6.08 / MapDims().tile_size)
        if (round(6.08 / MapDims().tile_size)) % 2 != 0
        else round(6.08 / MapDims().tile_size) + 1
    )
    width: int = (
        round(3.5 / MapDims().tile_size)
        if (round(3.5 / MapDims().tile_size)) % 2 != 0
        else round(3.5 / MapDims().tile_size) + 1
    )


class Rewards(NamedTuple):
    existence: Float = -0.05

    collision_move: Float = -1.0
    move_while_loaded: Float = -0.2
    move: Float = -0.1

    collision_turn: Float = -1.0
    base_turn: Float = -0.2

    cabin_turn: Float = -0.05

    dig_wrong: Float = (
        -2.0
    )  # given both if loaded stayed the same, or if new map is not closer than old to target
    dump_wrong: Float = -2.0  # given if loaded stayed the same

    dig_correct: Float = 2.0  # given if the new map is closer to target map than before
    dump_correct: Float = 0.2  # implemented as dump where not digged

    terminal: Float = 10.0  # given if the action map is the same as the target map where it matters (digged tiles)


class EnvConfig(NamedTuple):
    tile_size: Float = MapDims().tile_size

    agent: AgentConfig = AgentConfig()

    target_map: MapConfig = TargetMapConfig()
    action_map: MapConfig = ActionMapConfig()

    rewards = Rewards()

    rewards_level: int = 0  # 0 to N, the level of the rewards to assign in curriculum learning (the higher, the more sparse)

    max_episode_duration: int = 100  # in number of steps


class BatchConfig(NamedTuple):
    action_type: Action = TrackedAction
