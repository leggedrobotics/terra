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

    edge_length_m: float = 44.0  # map edge length in meters 44  , 66 for 96x96 maps
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

    move_tiles: int = 6  #6 number of tiles of progress for every move action
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
    skid_move: float  # reward for skidsteer movement
    skid_dump_correct: float  # reward for successfully dumping in correct areas
    skid_dump_wrong: float  # penalty for failed dump attempts
    skid_shovel_control: float  # small reward for effective shovel control (lift/lower)
    skid_auto_load: float  # reward for efficient auto-loading while moving
    holding_dirt: float  # penalty per step for being loaded (holding dirt)
    skid_lift_shovel_with_dirt: float  # reward for lifting the shovel while loaded
    skid_move_loaded_shovel_up: float  # reward for moving while loaded and shovel is up
    skid_auto_load_from_dumpzone_penalty: float  # penalty for loading dirt from a dump zone

    move_to_dump_zone: float  # reward/penalty for distance to dump zone

    terminal: float  # given if the action map is the same as the target map where it matters (digged tiles)

    normalizer: float  # constant scaling factor for all rewards
    

    @staticmethod
    def dense():
        return Rewards(
            existence=-0.25,   # -0.25      -0.1 for 96x96 maps
            collision_move=-0.2,  #-0.2,
            move_while_loaded=-0.0,  # Reduced penalty  was -0.01
            move=-0.1,  # Heavily reduced movement penalty was -0.1
            move_with_turned_wheels=-0.1,  # Reduced penalty was -0.1
            collision_turn=-0.1, #-0.1,
            base_turn=-0.1,  #-0.1  
            cabin_turn=-0.05,
            wheel_turn=-0.05,  
            dig_wrong=-0.25,
            dump_wrong=-1.0,
            dig_correct=0.6,  #1.5    0.2      0.35
            dump_correct=1.0,  #0.15   0.0

            # Skid steer specific rewards
            skid_move=-0.05,                 # Remove positive movement reward to discourage random movement
            terminal=200.0, #250.0
            normalizer=70.0,

            #UNUSED REWARDS
            skid_dump_correct=50.0,         # 5  20  # Large reward for correct dumping (0.05 after normalization)
            skid_dump_wrong=-0.6,          #-0.5  # Moderate penalty for failed dumps (-0.005 after normalization)
            skid_shovel_control=0.0,       # Small reward for shovel control (0.0001 after normalization)
            skid_auto_load=0.7,              # Core reward: agents need to load dirt to dump it (0.005 after normalization)
            holding_dirt=-0.0,        #-0.1      # Per-step penalty for holding dirt (-0.001 after normalization)
            skid_lift_shovel_with_dirt=0.0,  # 0.05 New: reward for lifting shovel with dirt (0.0005 after normalization)
            skid_move_loaded_shovel_up=0.0,  # 0.02 New: reward for moving while loaded and shovel is up (0.0002 after normalization)
            move_to_dump_zone=0.0,  #0.3 worked        # Moderate reward for being close to dump zones (0.003 after normalization)
            skid_auto_load_from_dumpzone_penalty=-0.0,  #-1.5# Large penalty for loading from dump zone
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
            skid_move=0.0,  # Remove positive movement reward to discourage random movement
            skid_dump_correct=1.0,  # Lower reward in sparse mode (0.01 after normalization)
            skid_dump_wrong=-0.25,  # Moderate penalty in sparse mode (-0.0025 after normalization)
            skid_shovel_control=0.01,  # Very small reward in sparse mode (0.0001 after normalization)
            skid_auto_load=0.2,  # Core reward: agents need to load dirt to dump it (0.002 after normalization)
            holding_dirt=-0.05,  # Per-step penalty for holding dirt (-0.0005 after normalization)
            skid_lift_shovel_with_dirt=0.05,  # New: reward for lifting shovel with dirt (0.0005 after normalization)
            skid_move_loaded_shovel_up=0.02,  # New: reward for moving while loaded and shovel is up (0.0002 after normalization)
            move_to_dump_zone=0.1,          # Small reward for being close to dump zones (0.001 after normalization)
            terminal=100.0,
            normalizer=100.0,
            skid_auto_load_from_dumpzone_penalty=-2.0,  # Penalty for loading from dump zone
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
    alignment_coefficient: float = -0.16 #-0.08
    distance_coefficient: float = -0.08 #-0.04

    curriculum: CurriculumConfig = CurriculumConfig()

    max_steps_in_episode: int = 0  # changed by CurriculumManager
    tile_size: float = 0  # updated in the code
    
    # Agent types configuration: (agent1_type, agent2_type)
    # 0=tracked, 1=wheeled, 2=skidsteer
    agent_types: tuple = (0, 3)  # Default: tracked + skidsteer, override with --agent_types in training script

    @classmethod
    def new(cls):
        return EnvConfig()


class MapsDimsConfig(NamedTuple):
    maps_edge_length: int = 0  # updated in the code



class CurriculumGlobalConfig(NamedTuple):
    increase_level_threshold: int = 20  
    decrease_level_threshold: int = 80  
    last_level_type = "random"  # ["random", "none"]

    # NOTE: all maps need to have the same size
    levels = [

        # {
        #     "maps_path": "foundations", 
        #     "max_steps_in_episode": 700,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        # {
        #     "maps_path": "trenches/single",
        #     "max_steps_in_episode": 750,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": True,
        # },
        # {
        #     "maps_path": "foundations_dumpzones_harder_nodump",
        #     "max_steps_in_episode": 750,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },

        # {
        #     "maps_path": "experimental_96x96",
        #     "max_steps_in_episode":1600,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        
        # {
        #     "maps_path": "foundations_hybrid_dumpzones",
        #     "max_steps_in_episode":1200,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        # 

        {
            "maps_path": "foundations_dumpzones_v3", 
            "max_steps_in_episode": 800,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": False,
        },
    
        # {
        #     "maps_path": "foundations_dumpzones_v3_separated", 
        #     "max_steps_in_episode": 800,
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },



        # {
        #     "maps_path": "trenches/single_dumpzone_v2",
        #     "max_steps_in_episode":800,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": True,
        # },

        # {
        #     "maps_path": "trenches/separated_v2",
        #     "max_steps_in_episode":800,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },






        # {
        #     "maps_path": "foundations_dumpzones_1.5",
        #     "max_steps_in_episode":800,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        # {
        #     "maps_path": "relocations_harder",
        #     "max_steps_in_episode":800,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        # {
        #     "maps_path": "trenches/single_dumpzone",
        #     "max_steps_in_episode":800,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },


        


        


        # {
        #     "maps_path": "relocations",
        #     "max_steps_in_episode": 550,  # Balanced: increased from 400 but reduced from 600
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        
        # Stage 1: Basic Skills - Learn individual capabilities
        
        
        # Stage 2: Foundation Coordination - Learn to work together on mixed tasks
        
        
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
