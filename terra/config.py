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
    Define the max size of the map.
    Used for padding in case it's needed.
    """
    max_width: int = 64  # number of tiles
    max_height: int = 64  # number of tiles


class MapDims(NamedTuple):
    tile_size: float = 0.67  # in meters


class TargetMapConfig(NamedTuple):
    pass


class ActionMapConfig(NamedTuple):
    pass


class ImmutableAgentConfig(NamedTuple):
    """
    The part of the AgentConfig that won't change based on curriculum.
    """

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

    height: int = (
        round(ExcavatorDims().WIDTH / MapDims().tile_size)
        if (round(ExcavatorDims().WIDTH / MapDims().tile_size)) % 2 != 0
        else round(ExcavatorDims().WIDTH / MapDims().tile_size) + 1
    )
    width: int = (
        round(ExcavatorDims().HEIGHT / MapDims().tile_size)
        if (round(ExcavatorDims().HEIGHT / MapDims().tile_size)) % 2 != 0
        else round(ExcavatorDims().HEIGHT / MapDims().tile_size) + 1
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

    dig_correct: float  # dig where the target map is negative, and not more than required
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
    tile_size: float = MapDims().tile_size

    agent: AgentConfig = AgentConfig()

    target_map: TargetMapConfig = TargetMapConfig()
    action_map: ActionMapConfig = ActionMapConfig()

    maps: ImmutableMapsConfig = ImmutableMapsConfig()

    rewards: Rewards = Rewards.dense()

    apply_trench_rewards: bool = False
    trench_rewards: TrenchRewards = TrenchRewards()

    curriculum: CurriculumConfig = CurriculumConfig()

    max_steps_in_episode: int = 0  # changed by CurriculumManager

    @classmethod
    def new(cls):
        return EnvConfig()

class CurriculumGlobalConfig(NamedTuple):
    increase_level_threshold: int = 3
    decrease_level_threshold: int = 10
    last_level_type = "random"  # ["random", "none"]
    
    levels = [
        {
            "maps_path": "terra/foundations",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "terra/trenches/easy",
            "max_steps_in_episode": 200,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": True,
        },
        {
            "maps_path": "terra/foundations",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "terra/trenches/medium",
            "max_steps_in_episode": 200,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": True,
        },
        {
            "maps_path": "terra/foundations",
            "max_steps_in_episode": 300,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "terra/trenches/hard",
            "max_steps_in_episode": 200,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": True,
        },
    ]

class BatchConfig(NamedTuple):
    action_type: Action = TrackedAction

    # Config to get data for batched env initialization
    agent: ImmutableAgentConfig = ImmutableAgentConfig()
    maps: ImmutableMapsConfig = ImmutableMapsConfig()

    curriculum_global: CurriculumGlobalConfig = CurriculumGlobalConfig()
