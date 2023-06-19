from typing import NamedTuple

from terra.actions import Action
from terra.actions import TrackedAction  # noqa: F401
from terra.actions import WheeledAction  # noqa: F401
from terra.map_generator import MapType
from terra.utils import Float


class ImmutableMapsConfig(NamedTuple):
    """
    Define the max size of the map.
    Used for padding in case it's needed.
    """

    max_width: int = 60  # number of tiles
    max_height: int = 60  # number of tiles


class MapDims(NamedTuple):
    width_m: Float = 60.0  # in meters
    height_m: Float = 60.0  # in meters
    tile_size: Float = 1.0  # in meters


class TargetMapConfig(NamedTuple):
    type: int = MapType.OPENSTREET_2_DIG_DUMP
    map_dof: int = 0  # for curriculum

    width: int = round(MapDims().width_m / MapDims().tile_size)
    height: int = round(MapDims().height_m / MapDims().tile_size)

    # For clusters type of map
    n_clusters: int = 5
    n_tiles_per_cluster: int = 10
    kernel_size_initial_sampling: tuple[int] = 10

    # Bounds on the volume per tile  # TODO implement in code
    min_height: int = -10
    max_height: int = 10

    @staticmethod
    def parametrized(
        map_dims: MapDims,
        # n_clusters: int,
        # n_tiles_per_cluster: int,
        # kernel_size_initial_sampling: tuple[int],
    ) -> "TargetMapConfig":
        return TargetMapConfig(
            width=round(map_dims.width_m / map_dims.tile_size),
            height=round(map_dims.height_m / map_dims.tile_size),
            # n_clusters=n_clusters,
            # n_tiles_per_cluster=n_tiles_per_cluster,
            # kernel_size_initial_sampling=kernel_size_initial_sampling,
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
    dump_no_dump_area: Float = (
        -0.02
    )  # given if dumps in an area that is not the dump area

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

    # rewards_level: int = 0  # 0 to N, the level of the rewards to assign in curriculum learning (the higher, the more sparse)
    max_steps_in_episode: int = 2

    @staticmethod
    def parametrized(
        width_m: int,
        height_m: int,
        max_steps_in_episode: int,
        # n_clusters: int,
        # n_tiles_per_cluster: int,
        # kernel_size_initial_sampling: tuple[int],
    ) -> "EnvConfig":
        map_dims = MapDims(width_m, height_m)
        return EnvConfig(
            tile_size=map_dims.tile_size,
            max_steps_in_episode=max_steps_in_episode,
            agent=AgentConfig.from_map_dims(map_dims),
            target_map=TargetMapConfig.parametrized(
                map_dims,  # n_clusters, n_tiles_per_cluster, kernel_size_initial_sampling
            ),
            action_map=ActionMapConfig.from_map_dims(map_dims),
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

    # Maps folders (select here the data paths you want to load)
    maps_paths = [
        "2_buildings/20x20/",
        "2_buildings/60x60/",
    ]
