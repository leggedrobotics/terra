from enum import IntEnum
from typing import NamedTuple

from terra.actions import Action
from terra.actions import TrackedAction  # noqa: F401
from terra.actions import WheeledAction  # noqa: F401


class MapType(IntEnum):
    SINGLE_TILE = 0
    SQUARE_SINGLE_TRENCH = 1
    RECTANGULAR_SINGLE_TRENCH = 2
    SQUARE_SINGLE_RAMP = 3
    SQUARE_SINGLE_TRENCH_RIGHT_SIDE = 4
    SINGLE_TILE_SAME_POSITION = 5
    SINGLE_TILE_EASY_POSITION = 6
    MULTIPLE_SINGLE_TILES = 7
    MULTIPLE_SINGLE_TILES_WITH_DUMP_TILES = 8
    TWO_SQUARE_TRENCHES_TWO_DUMP_AREAS = 9

    # Loaded from disk
    OPENSTREET_2_DIG_DUMP = 10
    OPENSTREET_3_DIG_DIG_DUMP = 11
    TRENCHES = 12
    FOUNDATIONS = 13


class ImmutableMapsConfig(NamedTuple):
    """
    Define the max size of the map.
    Used for padding in case it's needed.
    """

    min_width: int = 16  # number of tiles
    min_height: int = 16  # number of tiles

    max_width: int = 40  # number of tiles
    max_height: int = 40  # number of tiles


class MapDims(NamedTuple):
    width_m: float = 60.0  # in meters
    height_m: float = 60.0  # in meters
    tile_size: float = 1.0  # in meters  # TODO changing tile_size to smtg not 1.0 can make stuff not work as intended


class TargetMapConfig(NamedTuple):
    type: int = MapType.TWO_SQUARE_TRENCHES_TWO_DUMP_AREAS
    map_dof: int = 0  # for curriculum

    # Used only for procedural maps with elements bigger than 1 tile
    element_edge_min: int = 2
    element_edge_max: int = 6

    # width: int = round(MapDims().width_m / MapDims().tile_size)
    # height: int = round(MapDims().height_m / MapDims().tile_size)

    # For clusters type of map
    # n_clusters: int = 5
    # n_tiles_per_cluster: int = 10
    # kernel_size_initial_sampling: tuple[int] = 10

    # Bounds on the volume per tile  # TODO implement in code
    # min_height: int = -10
    # max_height: int = 10

    @staticmethod
    def parametrized(map_dof: int, map_type: int) -> "TargetMapConfig":
        return TargetMapConfig(
            map_dof=map_dof,
            type=map_type,
        )


class ActionMapConfig(NamedTuple):
    # width: int = round(MapDims().width_m / MapDims().tile_size)
    # height: int = round(MapDims().height_m / MapDims().tile_size)

    # Bounds on the volume per tile  # TODO implement in code
    # min_height: int = -10
    # max_height: int = 10

    # @staticmethod
    # def from_map_dims(map_dims: MapDims) -> "ActionMapConfig":
    #     return ActionMapConfig(
    #         width=round(map_dims.width_m / map_dims.tile_size),
    #         height=round(map_dims.height_m / map_dims.tile_size),
    #     )
    pass


class ImmutableAgentConfig(NamedTuple):
    """
    The part of the AgentConfig that won't change based on curriculum.
    """

    angles_base: int = 4
    angles_cabin: int = 8
    max_arm_extension: int = 1  # numbering starts from 0 (0 is the closest level)


class AgentConfig(NamedTuple):
    random_init_pos: bool = True
    random_init_base_angle: bool = True

    angles_base: int = ImmutableAgentConfig().angles_base
    angles_cabin: int = ImmutableAgentConfig().angles_cabin
    max_arm_extension: int = ImmutableAgentConfig().max_arm_extension

    move_tiles: int = 2  # number of tiles of progress for every move action
    #  Note: move_tiles is also used as radius of excavation
    #       (we dig as much as move_tiles in the radial distance)

    dig_depth: int = 1  # how much every dig action digs
    # max_dig: int = -3  # soft max after which the agent pays a cost  # TODO implement
    # max_dump: int = 3  # soft max after which the agent pays a cost  # TODO implement

    # max_loaded: int = 100  # TODO implement

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
    existence: float = -0.01

    collision_move: float = -0.2
    move_while_loaded: float = 0.0
    move: float = -0.05

    collision_turn: float = -0.2
    base_turn: float = -0.1

    cabin_turn: float = -0.01

    dig_wrong: float = (
        -0.2
    )  # dig where the target map is not negative (exclude case of positive action map -> moving dumped terrain)
    dump_wrong: float = -0.2  # given if loaded stayed the same
    dump_no_dump_area: float = (
        -0.02
    )  # given if dumps in an area that is not the dump area

    dig_correct: float = (
        2.0  # dig where the target map is negative, and not more than required
    )
    dump_correct: float = 2.0  # dump where the target map is positive

    terminal: float = 5.0  # given if the action map is the same as the target map where it matters (digged tiles)


class EnvConfig(NamedTuple):
    tile_size: float = MapDims().tile_size

    agent: AgentConfig = AgentConfig()

    target_map: TargetMapConfig = TargetMapConfig()
    action_map: ActionMapConfig = ActionMapConfig()

    maps: ImmutableMapsConfig = ImmutableMapsConfig()

    rewards = Rewards()

    # rewards_level: int = 0  # 0 to N, the level of the rewards to assign in curriculum learning (the higher, the more sparse)
    max_steps_in_episode: int = 1

    @staticmethod
    def parametrized(
        width_m: int,
        height_m: int,
        max_steps_in_episode: int,
        map_dof: int,
        map_type: int,
    ) -> "EnvConfig":
        map_dims = MapDims(width_m, height_m)
        return EnvConfig(
            tile_size=map_dims.tile_size,
            max_steps_in_episode=max_steps_in_episode,
            agent=AgentConfig.from_map_dims(map_dims),
            target_map=TargetMapConfig.parametrized(map_dof, map_type),
            # action_map=ActionMapConfig.from_map_dims(map_dims),
        )

    @classmethod
    def new(cls):
        return EnvConfig()


class BatchConfig(NamedTuple):
    action_type: Action = TrackedAction
    load_maps_from_disk: bool = True

    # Config to get data for batched env initialization
    agent: ImmutableAgentConfig = ImmutableAgentConfig()
    maps: ImmutableMapsConfig = ImmutableMapsConfig()

    # Maps folders (the order matters -> DOF)
    maps_paths = [
        # "2_buildings/20x20/",
        # "2_buildings/40x40/",
        # "2_buildings/60x60/",
        "trenches/easy/images",
        "trenches/medium/images",
        "trenches/hard/images",
        "foundations/easy/images",
        "foundations/medium/images",
        "foundations/hard/images",
    ]
