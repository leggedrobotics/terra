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
    angles_base: int = 12
    angles_cabin: int = 12
    max_wheel_angle: int = 2
    wheel_step: float = 20.0  # difference between next angles in discretization (in degrees)
    num_state_obs: int = 8  # number of state observations: [pos_x, pos_y, angle_base, angle_cabin, wheel_angle, loaded, agent_type, shovel_lifted]


class AgentConfig(NamedTuple):
    random_init_state: bool = True

    angles_base: int = ImmutableAgentConfig().angles_base
    angles_cabin: int = ImmutableAgentConfig().angles_cabin
    max_wheel_angle: int = ImmutableAgentConfig().max_wheel_angle
    wheel_step: float = ImmutableAgentConfig().wheel_step

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
    move_with_turned_wheels: float

    collision_turn: float
    base_turn: float

    cabin_turn: float
    wheel_turn: float

    dig_wrong: float  # dig where the target map is not negative (exclude case of positive action map -> moving dumped terrain)
    dump_wrong: float  # given if loaded stayed the same or tried to dump in non-dumpable tile

    dig_correct: float  # dig where the target map is negative, and not more than required
    dump_correct: float  # dump where the target map is positive

    # Skid steer specific rewards
    skid_lift_correct: float  # reward for successfully lifting dirt (auto-loading or manual)
    skid_dump_correct: float  # reward for successfully dumping in correct areas
    skid_dump_wrong: float  # penalty for failed dump attempts
    skid_shovel_control: float  # small reward for effective shovel control (lift/lower)
    skid_auto_load: float  # reward for efficient auto-loading while moving

    terminal: float  # given if the action map is the same as the target map where it matters (digged tiles)

    normalizer: float  # constant scaling factor for all rewards

    @staticmethod
    def dense():
        return Rewards(
            existence=-0.02,  # Small time pressure penalty (was -0.1)
            collision_move=-0.2,
            move_while_loaded=-0.005,  # Reduced penalty  was -0.01
            move=-0.02,  # Heavily reduced movement penalty was -0.1
            move_with_turned_wheels=-0.02,  # Reduced penalty was -0.1
            collision_turn=-0.1,
            base_turn=-0.02,  # Reduced turning penalty was -0.1
            cabin_turn=-0.01,  # Much reduced cabin penalty was -0.05
            wheel_turn=-0.05,  # Reduced turning penalty was -0.05
            dig_wrong=-0.25,
            dump_wrong=-1.0,
            dig_correct=1.0,  # Much higher positive rewards was 0.2
            dump_correct=1.5,  # Even higher reward for dumping correctly was 0.15
            # Skid steer specific rewards
            skid_lift_correct=1.2,  # Much higher reward for successful dirt lifting was 0.3
            skid_dump_correct=1.5,  # Much higher reward for correct dumping was 0.25
            skid_dump_wrong=-0.5,  # Moderate penalty for failed dumps was -0.5
            skid_shovel_control=0.05,  # Higher reward for shovel control was 0.05
            skid_auto_load=0.8,  # Much higher reward for efficient auto-loading was 0.1
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
            move_with_turned_wheels=-0.15,
            collision_turn=-0.1,
            base_turn=-0.1,
            cabin_turn=-0.01,
            wheel_turn=-0.005,
            dig_wrong=-0.3,
            dump_wrong=-0.3,
            dig_correct=0.0,
            dump_correct=0.0,
            # Skid steer specific rewards (more sparse)
            skid_lift_correct=0.1,  # Lower reward in sparse mode
            skid_dump_correct=0.05,  # Lower reward in sparse mode
            skid_dump_wrong=-0.2,  # Lower penalty in sparse mode
            skid_shovel_control=0.01,  # Very small reward in sparse mode
            skid_auto_load=0.02,  # Very small reward in sparse mode
            terminal=100.0,
            normalizer=100.0,
        )


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
    alignment_coefficient: float = -0.0
    distance_coefficient: float = -0.0

    curriculum: CurriculumConfig = CurriculumConfig()

    max_steps_in_episode: int = 0  # changed by CurriculumManager
    tile_size: float = 0  # updated in the code

    @classmethod
    def new(cls):
        return EnvConfig()


class MapsDimsConfig(NamedTuple):
    maps_edge_length: int = 0  # updated in the code



class CurriculumGlobalConfig(NamedTuple):
    increase_level_threshold: int = 15  # Reduced from 20 for faster progression
    decrease_level_threshold: int = 30  # Reduced from 50 for quicker regression
    last_level_type = "random"  # ["random", "none"]

    # NOTE: all maps need to have the same size
    # Mixed Agent Training Curriculum: Skid Steer (Agent 0) + Excavator (Agent 2)
    levels = [
        {
            "maps_path": "relocations",
            "max_steps_in_episode": 300,  # Shorter episodes for faster learning
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": False,
        },
        {
            "maps_path": "foundations_dumpzones", 
            "max_steps_in_episode": 350,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": False,
        },
        # Stage 1: Basic Skills - Learn individual capabilities
        
        
        # Stage 2: Foundation Coordination - Learn to work together on mixed tasks
        # {
        #     "maps_path": "foundations", 
        #     "max_steps_in_episode": 350,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        
        # # Stage 3: Basic Excavation - Excavator leads, skid steer supports
        # {
        #     "maps_path": "trenches/single",
        #     "max_steps_in_episode": 400,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": True,  # Enable trench-specific rewards
        # },
        
        # # Stage 4: Mixed Practice - Alternate between specializations
        # {
        #     "maps_path": "relocations",
        #     "max_steps_in_episode": 350,
        #     "rewards_type": RewardsType.DENSE, 
        #     "apply_trench_rewards": False,
        # },
        
        # # Stage 5: Advanced Coordination - Complex mixed scenarios
        # {
        #     "maps_path": "foundations",
        #     "max_steps_in_episode": 400,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        
        # # Stage 6: Complex Excavation - Advanced trench work
        # {
        #     "maps_path": "trenches/double", 
        #     "max_steps_in_episode": 500,  # Longer for complex tasks
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": True,
        # },
        
        # # Stage 7: Mastery Testing - Return to basics with higher expectations
        # {
        #     "maps_path": "relocations",
        #     "max_steps_in_episode": 250,  # Shorter time pressure
        #     "rewards_type": RewardsType.SPARSE,  # More challenging rewards
        #     "apply_trench_rewards": False,
        # },
        
        # # Stage 8: Final Integration - All skills combined
        # {
        #     "maps_path": "foundations",
        #     "max_steps_in_episode": 400, 
        #     "rewards_type": RewardsType.SPARSE,  # Sparse rewards for mastery
        #     "apply_trench_rewards": False,
        # },
    ]

'''
class CurriculumGlobalConfig(NamedTuple):
    increase_level_threshold: int = 20
    decrease_level_threshold: int = 50
    last_level_type = "random"  # ["random", "none"]

    # NOTE: all maps need to have the same size
    levels = [
        {
            "maps_path": "trenches/easy",
            "max_steps_in_episode": 600,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": False,
        },
        # {
        #     "maps_path": "trenches/easy",
        #     "max_steps_in_episode": 600,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        # {
        #     "maps_path": "trenches/medium",
        #     "max_steps_in_episode": 600,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        # {
        #     "maps_path": "trenches/hard",
        #     "max_steps_in_episode": 600,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        # {
        #     "maps_path": "foundations",
        #     "max_steps_in_episode": 600,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        # {
        #     "maps_path": "trenches/hard",
        #     "max_steps_in_episode": 600,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": True,
        # },
        
    ]
'''


class BatchConfig(NamedTuple):
    action_type: Action = TrackedAction  # [WheeledAction, TrackedAction]

    # Config to get data for batched env initialization
    agent: ImmutableAgentConfig = ImmutableAgentConfig()
    maps: ImmutableMapsConfig = ImmutableMapsConfig()
    maps_dims: MapsDimsConfig = MapsDimsConfig()

    curriculum_global: CurriculumGlobalConfig = CurriculumGlobalConfig()
