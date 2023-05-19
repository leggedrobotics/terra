from typing import NamedTuple

from terra.actions import Action
from terra.actions import TrackedAction  # noqa: F401
from terra.actions import WheeledAction  # noqa: F401
from terra.map_generator import MapParams
from terra.map_generator import MapType
from terra.utils import Float


class MapDims(NamedTuple):
    tile_size: Float = 1.5  # in meters
    width_m: Float = 15.0  # in meters
    height_m: Float = 15.0  # in meters


class MapConfig(NamedTuple):
    width: int = round(MapDims().width_m / MapDims().tile_size)
    height: int = round(MapDims().height_m / MapDims().tile_size)

    # Bounds on the volume per tile  # TODO implement in code
    min_height: int = -10
    max_height: int = 10


# Map params #####
class MapParamsSquareSingleTile(MapParams):
    type: MapType = MapType.SINGLE_TILE
    depth: int = -1

    edge_min: int = 2  # ignore
    edge_max: int = 2  # ignore


class MapParamsSquareSingleTrench(MapParams):
    type: MapType = MapType.SQUARE_SINGLE_TRENCH
    edge_min: int = 2
    edge_max: int = 2
    depth: int = -1


class MapParamsRectangularSingleTrench(MapParams):
    type: MapType = MapType.RECTANGULAR_SINGLE_TRENCH
    edge_min: int = 2
    edge_max: int = 2
    depth: int = -1


class MapParamsSquareSingleRamp(MapParams):
    type: MapType = MapType.SQUARE_SINGLE_RAMP
    edge_min: int = 4
    edge_max: int = 4
    depth: int = -97  # ignore


class MapParamsSquareSingleTrenchRightSide(MapParams):
    type: MapType = MapType.SQUARE_SINGLE_TRENCH_RIGHT_SIDE
    edge_min: int = 1
    edge_max: int = 1
    depth: int = -1


# end Map params #####


class TargetMapConfig(MapConfig):
    params: MapParams = MapParamsSquareSingleTile()


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

    max_loaded: int = 100  # TODO implement

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
    existence: Float = -0.01

    collision_move: Float = -0.02
    move_while_loaded: Float = 0.0
    move: Float = -0.01

    collision_turn: Float = -0.02
    base_turn: Float = -0.01

    cabin_turn: Float = -0.01

    dig_wrong: Float = (
        -0.05
    )  # given both if loaded stayed the same, or if new map is not closer than old to target
    dump_wrong: Float = -0.01  # given if loaded stayed the same

    dig_correct: Float = 0.2  # given if the new map is closer to target map than before
    dump_correct: Float = 0.1  # implemented as dump where not digged

    terminal: Float = 10.0  # given if the action map is the same as the target map where it matters (digged tiles)


class EnvConfig(NamedTuple):
    tile_size: Float = MapDims().tile_size

    agent: AgentConfig = AgentConfig()

    target_map: MapConfig = TargetMapConfig()
    action_map: MapConfig = ActionMapConfig()

    rewards = Rewards()

    rewards_level: int = 0  # 0 to N, the level of the rewards to assign in curriculum learning (the higher, the more sparse)
    max_steps_in_episode: int = 10


class BatchConfig(NamedTuple):
    action_type: Action = TrackedAction
