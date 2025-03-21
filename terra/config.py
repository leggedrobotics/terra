from enum import IntEnum
from typing import NamedTuple

from terra.actions import Action
from terra.actions import TrackedAction  # noqa: F401
from terra.actions import WheeledAction  # noqa: F401


class ExcavatorDims(NamedTuple):
    WIDTH: float = 6.08  # longer side
    HEIGHT: float = 3.5  # shorter side


class RewardsType(IntEnum):
    DENSE = 0
    SPARSE = 1


class ImmutableMapsConfig(NamedTuple):
    """
    Define the max size of the map in meters.
    This defines the proportion between the map and the agent.
    """

    edge_length_m: float = 44.0  # map edge length in meters
    edge_length_px: int = 0  # updated in the code


class TargetMapConfig(NamedTuple):
    pass


class ActionMapConfig(NamedTuple):
    pass


class ImmutableAgentConfig(NamedTuple):
    """
    The part of the AgentConfig that won't change based on curriculum.
    """

    dimensions: ExcavatorDims = ExcavatorDims()
    angles_base: int = 4
    angles_cabin: int = 8
    max_arm_extension: int = 1  # numbering starts from 0 (0 is the closest level)


class AgentConfig(NamedTuple):
    random_init_state: bool = True

    angles_base: int = ImmutableAgentConfig().angles_base
    angles_cabin: int = ImmutableAgentConfig().angles_cabin
    max_arm_extension: int = ImmutableAgentConfig().max_arm_extension

    move_tiles: int = 6  # number of tiles of progress for every move action
    #  Note: move_tiles is also used as radius of excavation
    #       (we dig as much as move_tiles in the radial distance)

    dig_depth: int = 1  # how much every dig action digs

    height: int = 0  # updated in the code
    width: int = 0  # updated in the code


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

    dig_correct: (
        float  # dig where the target map is negative, and not more than required
    )
    dump_correct: float  # dump where the target map is positive, only if digged and not moved soil around

    terminal: float  # given if the action map is the same as the target map where it matters (digged tiles)

    terminal_completed_tiles: float  # gets linearly scaled by ratio of completed tiles

    normalizer: float  # constant scaling factor for all rewards

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
            dig_correct=3.0,
            dump_correct=3.0,
            terminal_completed_tiles=0.0,
            terminal=100.0,
            normalizer=100.0,
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
            dig_correct=0.0,
            dump_correct=0.0,
            terminal_completed_tiles=0.0,
            terminal=100.0,
            normalizer=100.0,
        )


class TrenchRewards(NamedTuple):
    distance_coefficient: float = (
        -0.4
    )  # distance_coefficient * distance, if distance > agent_width / 2


class CurriculumConfig(NamedTuple):
    """State of the curriculum. This config should not be changed."""

    level: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class EnvConfig(NamedTuple):
    agent: AgentConfig = AgentConfig()

    target_map: TargetMapConfig = TargetMapConfig()
    action_map: ActionMapConfig = ActionMapConfig()

    maps: ImmutableMapsConfig = ImmutableMapsConfig()

    rewards: Rewards = Rewards.dense()

    apply_trench_rewards: bool = False
    trench_rewards: TrenchRewards = TrenchRewards()

    curriculum: CurriculumConfig = CurriculumConfig()

    max_steps_in_episode: int = 0  # changed by CurriculumManager
    tile_size: float = 0  # updated in the code

    @classmethod
    def new(cls):
        return EnvConfig()


class MapsDimsConfig(NamedTuple):
    maps_edge_length: int = 0  # updated in the code


class CurriculumGlobalConfig(NamedTuple):
    increase_level_threshold: int = 5
    decrease_level_threshold: int = 50
    last_level_type = "random"  # ["random", "none"]

    # NOTE: all maps need to have the same size
    # levels = [
    #     {
    #         "maps_path": "terra/foundations",
    #         "max_steps_in_episode": 300,
    #         "rewards_type": RewardsType.DENSE,
    #         "apply_trench_rewards": False,
    #     },
    #     {
    #         "maps_path": "terra/trenches/easy",
    #         "max_steps_in_episode": 300,
    #         "rewards_type": RewardsType.DENSE,
    #         "apply_trench_rewards": True,
    #     },
    #     {
    #         "maps_path": "terra/foundations",
    #         "max_steps_in_episode": 300,
    #         "rewards_type": RewardsType.DENSE,
    #         "apply_trench_rewards": False,
    #     },
    #     {
    #         "maps_path": "terra/trenches/medium",
    #         "max_steps_in_episode": 300,
    #         "rewards_type": RewardsType.DENSE,
    #         "apply_trench_rewards": True,
    #     },
    #     {
    #         "maps_path": "terra/foundations",
    #         "max_steps_in_episode": 300,
    #         "rewards_type": RewardsType.DENSE,
    #         "apply_trench_rewards": False,
    #     },
    #     {
    #         "maps_path": "terra/trenches/hard",
    #         "max_steps_in_episode": 300,
    #         "rewards_type": RewardsType.DENSE,
    #         "apply_trench_rewards": True,
    #     },
    # ]

    levels = [
        {
            "maps_path": "squares/64x64/2",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "squares/64x64/3",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "trenches/easy_size_small",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": True,
        },
        {
            "maps_path": "squares/64x64/5",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "squares/64x64/8",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "trenches/easy_size_medium",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": True,
        },
        {
            "maps_path": "squares/64x64/13",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "foundations",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "trenches/easy_size_large",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": True,
        },
        {
            "maps_path": "foundations",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "trenches/medium_size_large",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": True,
        },
        {
            "maps_path": "foundations",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "trenches/hard_size_large",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.SPARSE,
            "apply_trench_rewards": True,
        },
    ]


class BatchConfig(NamedTuple):
    action_type: Action = TrackedAction  # [WheeledAction, TrackedAction]

    # Config to get data for batched env initialization
    agent: ImmutableAgentConfig = ImmutableAgentConfig()
    maps: ImmutableMapsConfig = ImmutableMapsConfig()
    maps_dims: MapsDimsConfig = MapsDimsConfig()

    curriculum_global: CurriculumGlobalConfig = CurriculumGlobalConfig()
