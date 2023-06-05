from typing import NamedTuple

from terra.actions import Action
from terra.actions import TrackedAction  # noqa: F401
from terra.actions import WheeledAction  # noqa: F401
from terra.map_generator import MapParams
from terra.map_generator import MapType
from terra.utils import Float


class MapDims(NamedTuple):
    width_m: Float = 30.0  # in meters
    height_m: Float = 30.0  # in meters
    tile_size: Float = 1.5  # in meters


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


class MapParamsSquareSingleTileSamePosition(MapParams):
    type: MapType = MapType.SINGLE_TILE_SAME_POSITION
    edge_min: int = 1
    edge_max: int = 1
    depth: int = -1


class MapParamsSquareSingleTileEasyPosition(MapParams):
    type: MapType = MapType.SINGLE_TILE_EASY_POSITION
    edge_min: int = 1
    edge_max: int = 1
    depth: int = -1


class MapParamsMultipleSingleTiles(MapParams):
    type: MapType = MapType.MULTIPLE_SINGLE_TILES
    edge_min: int = 1
    edge_max: int = 1
    depth: int = -1


class MapParamsMultipleSingleTilesWithDumpTiles(MapParams):
    type: MapType = MapType.MULTIPLE_SINGLE_TILES_WITH_DUMP_TILES
    edge_min: int = 1
    edge_max: int = 1
    depth: int = -1


class MapParamsTwoSquareTrenchesTwoDumpAreas(MapParams):
    type: MapType = MapType.TWO_SQUARE_TRENCHES_TWO_DUMP_AREAS
    edge_min: int = 1
    edge_max: int = 1
    depth: int = -1


class MapParamsRandomMultishape(MapParams):
    type: MapType = MapType.RANDOM_MULTISHAPE
    edge_min: int = 1
    edge_max: int = 1
    depth: int = -1


# end Map params #####


class TargetMapConfig(NamedTuple):
    params: MapParams = MapParamsRandomMultishape()

    width: int = round(MapDims().width_m / MapDims().tile_size)
    height: int = round(MapDims().height_m / MapDims().tile_size)

    # Bounds on the volume per tile  # TODO implement in code
    min_height: int = -10
    max_height: int = 10

    @staticmethod
    def from_map_dims(map_dims: MapDims) -> "TargetMapConfig":
        return TargetMapConfig(
            width=round(map_dims.width_m / map_dims.tile_size),
            height=round(map_dims.height_m / map_dims.tile_size),
        )


class ActionMapConfig(NamedTuple):
    width: int = round(MapDims().width_m / MapDims().tile_size)
    height: int = round(MapDims().height_m / MapDims().tile_size)

    # Bounds on the volume per tile  # TODO implement in code
    min_height: int = -10
    max_height: int = 10

    @staticmethod
    def from_map_dims(map_dims: MapDims) -> "ActionMapConfig":
        return ActionMapConfig(
            width=round(map_dims.width_m / map_dims.tile_size),
            height=round(map_dims.height_m / map_dims.tile_size),
        )


class AgentConfig(NamedTuple):
    random_init_pos: bool = True
    random_init_base_angle: bool = True

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

    @staticmethod
    def from_map_dims(map_dims: MapDims) -> "AgentConfig":
        return AgentConfig(
            height=(
                round(6.08 / map_dims.tile_size)
                if (round(6.08 / map_dims.tile_size)) % 2 != 0
                else round(6.08 / map_dims.tile_size) + 1
            ),
            width=(
                round(3.5 / map_dims.tile_size)
                if (round(3.5 / map_dims.tile_size)) % 2 != 0
                else round(3.5 / map_dims.tile_size) + 1
            ),
        )


class Rewards(NamedTuple):
    existence: Float = -0.01

    collision_move: Float = -0.2
    move_while_loaded: Float = 0.0
    move: Float = -0.05

    collision_turn: Float = -0.2
    base_turn: Float = -0.1

    cabin_turn: Float = -0.01

    dig_wrong: Float = (
        -0.2
    )  # dig where the target map is not negative (exclude case of positive action map -> moving dumped terrain)
    dump_wrong: Float = -0.2  # given if loaded stayed the same
    dump_no_dump_area: Float = -0.02  # given if dumps in an area that is not the dump area

    dig_correct: Float = (
        2.0  # dig where the target map is negative, and not more than required
    )
    dump_correct: Float = 2.0  # dump where the target map is positive

    terminal: Float = 5.0  # given if the action map is the same as the target map where it matters (digged tiles)


class EnvConfig(NamedTuple):
    tile_size: Float = MapDims().tile_size

    agent: AgentConfig = AgentConfig()

    target_map: TargetMapConfig = TargetMapConfig()
    action_map: ActionMapConfig = ActionMapConfig()

    rewards = Rewards()

    rewards_level: int = 0  # 0 to N, the level of the rewards to assign in curriculum learning (the higher, the more sparse)
    max_steps_in_episode: int = 10

    @staticmethod
    def parametrized(
        map_dims: MapDims,
        max_steps_in_episode: int,
    ) -> "EnvConfig":
        return EnvConfig(
            tile_size=map_dims.tile_size,
            max_steps_in_episode=max_steps_in_episode,
            agent=AgentConfig.from_map_dims(map_dims),
            target_map=TargetMapConfig.from_map_dims(map_dims),
            action_map=ActionMapConfig.from_map_dims(map_dims),
        )


class BatchConfig(NamedTuple):
    action_type: Action = TrackedAction
