from enum import IntEnum
from typing import NamedTuple

import jax

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
    ROTATED_RECTANGLE_MAX_WIDTH = 10

    # Loaded from disk
    OPENSTREET_2_DIG_DUMP = 11
    OPENSTREET_3_DIG_DIG_DUMP = 12
    TRENCHES = 13
    FOUNDATIONS = 14
    RECTANGLES = 15


class RewardsType(IntEnum):
    DENSE = 0
    SPARSE = 1
    TERMINAL_ONLY = 2
    MIXED = 3


class ImmutableMapsConfig(NamedTuple):
    """
    Define the max size of the map.
    Used for padding in case it's needed.
    """

    min_width: int = 60  # number of tiles
    min_height: int = 60  # number of tiles

    max_width: int = 60  # number of tiles
    max_height: int = 60  # number of tiles


class MapDims(NamedTuple):
    width_m: float = 60.0  # in meters
    height_m: float = 60.0  # in meters
    tile_size: float = 0.67  # in meters  # TODO changing tile_size to smtg not 1.0 can make stuff not work as intended


class TargetMapConfig(NamedTuple):
    type: int = MapType.RECTANGLES
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

    move_tiles: int = 6  # number of tiles of progress for every move action
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
    existence: float

    collision_move: float
    move_while_loaded: float
    move: float

    collision_turn: float
    base_turn: float

    cabin_turn: float

    dig_wrong: float  # dig where the target map is not negative (exclude case of positive action map -> moving dumped terrain)
    dump_wrong: float  # given if loaded stayed the same
    dump_no_dump_area: float  # given if dumps in an area that is not the dump area
    # dig_dump_area: float  # dig dumped terrain on the dump area (prevents loops)

    dig_correct: float  # dig where the target map is negative, and not more than required
    dump_correct: float  # dump where the target map is positive, only if digged and not moved soil around

    terminal: float  # given if the action map is the same as the target map where it matters (digged tiles)

    force_reset: float  # given if the training algorithm calls a force reset on the environment
    
    # TODO quickfix for compatibility -> remove the '=0.0'  
    terminal_completed_tiles: float = 0.0  # gets linearly scaled by ratio of completed tiles

    @staticmethod
    def dense():
        return Rewards(
            existence=-0.1,
            collision_move=-0.1,
            move_while_loaded=0.0,
            move=-0.05,
            collision_turn=-0.1,
            base_turn=-0.1,
            cabin_turn=-0.02,
            dig_wrong=-0.3,
            dump_wrong=-0.3,
            dump_no_dump_area=0.0,
            # dig_dump_area=-0.3,
            dig_correct=3.0,
            dump_correct=3.0,
            terminal_completed_tiles=0.0,
            terminal=200.0,
            force_reset=0.0,
        )
    
    @staticmethod
    def mixed():
        return Rewards(
            existence=-0.1,
            collision_move=-0.1,
            move_while_loaded=0.0,
            move=-0.05,
            collision_turn=-0.1,
            base_turn=-0.1,
            cabin_turn=-0.02,
            dig_wrong=-0.3,
            dump_wrong=-0.3,
            dump_no_dump_area=0.0,
            # dig_dump_area=-0.3,
            dig_correct=0.0,
            dump_correct=0.0,
            terminal_completed_tiles=200.0,  # gets linearly scaled by ratio of completed tiles
            terminal=0.0,
            force_reset=0.0,
        )

    @staticmethod
    def sparse():
        return Rewards(
            existence=-0.1,
            collision_move=-0.1,
            move_while_loaded=0.0,
            move=-0.05,
            collision_turn=-0.1,
            base_turn=-0.1,
            cabin_turn=-0.02,
            dig_wrong=-0.3,
            dump_wrong=-0.3,
            dump_no_dump_area=0.0,
            # dig_dump_area=0.0,
            dig_correct=0.0,
            dump_correct=0.0,
            terminal_completed_tiles=0.0,
            terminal=200.0,
            force_reset=0.0,
        )

    @staticmethod
    def terminal_only():
        return Rewards(
            existence=-0.01,
            collision_move=0.0,
            move_while_loaded=0.0,
            move=0.0,
            collision_turn=0.0,
            base_turn=0.0,
            cabin_turn=0.0,
            dig_wrong=0.0,
            dump_wrong=0.0,
            dump_no_dump_area=0.0,
            # dig_dump_area=0.0,
            dig_correct=0.0,
            dump_correct=0.0,
            terminal_completed_tiles=0.0,
            terminal=1.0,
            force_reset=0.0,
        )


class TrenchRewards(NamedTuple):
    distance_coefficient: float = (
        -0.4
    )  # distance_coefficient * distance, if distance > agent_width / 2


class EnvConfig(NamedTuple):
    tile_size: float = MapDims().tile_size

    agent: AgentConfig = AgentConfig()

    target_map: TargetMapConfig = TargetMapConfig()
    action_map: ActionMapConfig = ActionMapConfig()

    maps: ImmutableMapsConfig = ImmutableMapsConfig()

    rewards: Rewards = Rewards.dense()

    apply_trench_rewards: bool = True
    trench_rewards: TrenchRewards = TrenchRewards()

    max_steps_in_episode: int = 1000

    @staticmethod
    def parametrized(
        width_m: int,
        height_m: int,
        max_steps_in_episode: int,
        map_dof: int,
        map_type: int,
        rewards_type: int,
        apply_trench_rewards: bool,
    ) -> "EnvConfig":
        map_dims = MapDims(width_m, height_m)

        # if rewards_type == RewardsType.DENSE:
        #     rewards = DenseRewards()
        # elif rewards_type == RewardsType.SPARSE:
        #     rewards = SparseRewards()
        # else:
        #     raise ValueError(f"{rewards_type=} doesn't exist.")

        rewards_list = [Rewards.dense, Rewards.sparse, Rewards.terminal_only, Rewards.mixed]

        rewards = jax.lax.switch(rewards_type, rewards_list)

        return EnvConfig(
            tile_size=map_dims.tile_size,
            max_steps_in_episode=max_steps_in_episode,
            agent=AgentConfig.from_map_dims(map_dims),
            target_map=TargetMapConfig.parametrized(map_dof, map_type),
            # action_map=ActionMapConfig.from_map_dims(map_dims),
            rewards=rewards,
            apply_trench_rewards=apply_trench_rewards,
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
        # "trenches/easy",
        # "trenches/medium",
        # "trenches/hard",
        # "foundations/easy",
        # "foundations/medium",
        # "foundations/hard",
        # "rectangles_1",
        # "rectangles_60_1",
        # "rectangles_60_2",
        # "trenches_60_1/easy",
        # "trenches_60_2/easy",
        # "trenches_60_1/medium",
        # "trenches_60_2/medium",
        # "trenches_60_1/hard",
        # "trenches_60_2/hard",

        # "trenches_metadata_60_1/easy",
        # "trenches_metadata_60_2/easy",
        # "trenches_metadata_60_1/medium",
        # "trenches_metadata_60_2/medium",
        # "trenches_metadata_60_1/hard",
        # "trenches_metadata_60_2/hard",

        # "onetile",
        # "dumping_constraints/squares_2/terra",
        # "dumping_constraints/squares_5/terra",
        # "dumping_constraints/squares_6/terra",
        # "dumping_constraints/lev1-T-trenches-contour",
        # "dumping_constraints/squares_7/terra",
        # "dumping_constraints/squares_8/terra",
        # "dumping_constraints/lev2-T-trenches-contour",
        # "dumping_constraints/squares_9/terra",
        # "dumping_constraints/lev3-T-trenches-contour",
        # "dumping_constraints/squares_10/terra",
        # "dumping_constraints/lev4-T-trenches-contour",
        # "dumping_constraints/trenches_metadata_60/easy",
        # "dumping_constraints/trenches_metadata_60/medium",
        # "dumping_constraints/trenches_metadata_60/hard",

        "trenches_occ_dmp_met_v2/all",

        # "dumping_constraints/trenches_metadata_60/all",

        # "small-rectangles_1",
        # "small-rectangles_2",
        # "rectangles_60_1",

        # "onetile"
    ]


class TestbenchConfig(BatchConfig):
    # Maps folders (the order matters -> DOF)
    maps_paths = ["onetile"]
