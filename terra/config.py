from enum import IntEnum
import math
from typing import NamedTuple

from terra.actions import Action
from terra.actions import TrackedAction  # noqa: F401
from terra.actions import WheeledAction  # noqa: F401


class ExcavatorDims(NamedTuple):
    WIDTH: float = 6.08  # longer side in meters
    HEIGHT: float = 3.5  # shorter side in meters


class RewardsType(IntEnum):
    DENSE = 0
    SPARSE = 1


class RewardStage(IntEnum):
    """Global reward mode for the compact excavation experiments."""

    DENSE_SKILL = 0
    TERMINAL_OBJECTIVE = 1
    ANNEALED_OBJECTIVE = 2
    REWARD_V2 = 3


DENSE_REWARD_PROTOCOL_ID = "dense_skill_legacy_relocation_v1"
REWARD_V2_PROTOCOL_ID = "material_potential_v2"
REWARD_V2_SUCCESS_BONUS = 6.0
REWARD_V2_HORIZON_FAILURE_PENALTY = 1.0
REWARD_V2_STEP_COST_TOTAL = 1.0
REWARD_V2_ALPHA = 1.0
REWARD_V2_BETA = 1.5
REWARD_V2_POTENTIAL_GAMMA = 0.9984
REWARD_V2_SHAPING_WEIGHT = 1.0
REWARD_V2_DISTANCE_REF_M = 16.0
REWARD_V2_DISTANCE_BOUND = 2.5

# Reward-v2.1 timing (adopted 2026-08-12). Discounted shaping of a potential
# with a large additive constant charges implicit rent for standing still:
# w*(1-gamma)*Phi, measured at 0.0060-0.0072/step, i.e. 76.5% of all time
# pressure, and it is an untuned by-product of beta*D_bound rather than a
# chosen pace. Shaping undiscounted removes the rent (and with it the
# procrastination gap: the constant cancels exactly at gamma=1, so no
# re-centering is needed), and the step cost then carries the whole, explicit,
# Phi-independent pace pressure at the same total magnitude.
#   0 BASELINE  shaping = w*(gamma*Phi_next - Phi), step cost 1.0/450
#               (frozen reward_v2, bit for bit; kept for replay only)
#   1 V21       shaping = w*(Phi_next - Phi),       step cost 3.6/450 = 0.0080
# The two changes are one treatment and move together under a single selector.
REWARD_V2_TIMING_BASELINE = 0
REWARD_V2_TIMING_V21 = 1
REWARD_V2_TIMING_V21_ID = "gamma1_stepcost_3.6"
REWARD_V2_V21_SHAPING_GAMMA = 1.0
REWARD_V2_V21_STEP_COST_TOTAL = 3.6

# Backplay-inspired reset tiers. Tier zero always selects the canonical full
# reset; tiers 1-3 substitute only the generated action map for the canonical
# source slot.
PARTIAL_RESET_FRACTIONS = (0.90, 0.75, 0.50)

# The tracked base has 12 headings, so half a heading bin is 15 degrees.
# This single value is shared by runtime gating and offline admission.
TRENCH_DIG_YAW_TOLERANCE_RAD = math.pi / 12.0


class ImmutableMapsConfig(NamedTuple):
    """
    Define the max size of the map in meters.
    This defines the proportion between the map and the agent.
    """

    edge_length_m: float = 36.5714285714  # 64 tiles at 0.5714285714 m/tile
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
    # [pos_x, pos_y, angle_base, angle_cabin, wheel_angle, loaded,
    #  agent_type, shovel_lifted, normalized carry work]
    num_state_obs: int = 9


class AgentConfig(NamedTuple):
    random_init_state: bool = True

    angles_base: int = ImmutableAgentConfig().angles_base
    angles_cabin: int = ImmutableAgentConfig().angles_cabin
    max_wheel_angle: int = ImmutableAgentConfig().max_wheel_angle
    wheel_step: float = ImmutableAgentConfig().wheel_step

    move_tiles: int = 5  # number of tiles of progress for every move action
    dig_radius_tiles: int = 5  # radial excavation/workspace reach in tiles, 6.5 m at 64x64 default

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
    dig_edge_bonus: float  # bonus per correctly dug border tile in foundation targets
    dump_correct: float  # dump where the target map is positive

    # Skid steer specific rewards
    skid_move: float  # reward for skidsteer movement
    skid_dump_wrong: float  # penalty for failed dump attempts

    terminal: float  # given if the action map is the same as the target map where it matters (digged tiles)

    normalizer: float  # constant scaling factor for all rewards
    

    @staticmethod
    def dense():
        return Rewards(
            existence=-0.25,  #-0.1 for 96x96 maps 
            collision_move=-0.2,  
            move_while_loaded=-0.0,  
            move=-0.1,  
            move_with_turned_wheels=-0.1,  
            collision_turn=-0.05, #-0.1
            base_turn=-0.05, #-0.1
            cabin_turn=-0.02, #-0.05
            wheel_turn=-0.02, #-0.05
            dig_wrong=-0.12, #-0.25
            dump_wrong=-1.0,
            dig_correct=0.6,  
            dig_edge_bonus=1.3,
            dump_correct=1.0,

            # Skid steer specific rewards
            skid_move=-0.05,             
            skid_dump_wrong=-0.6, 


            terminal=200.0, #250.0
            normalizer=70.0,

        )

    @staticmethod
    def sparse():
        return Rewards(
            existence=-0.1,
            collision_move=-0.1,
            move_while_loaded=0.0,
            move=-0.05,
            move_with_turned_wheels=-0.05,
            collision_turn=-0.1,
            base_turn=-0.1,
            cabin_turn=-0.01,
            wheel_turn=-0.005,
            dig_wrong=-0.3,
            dump_wrong=-0.3,
            dig_correct=0.0,
            dig_edge_bonus=0.0,
            dump_correct=0.0,
            # Skid steer specific rewards (more sparse)
            skid_move=0.0,  # Remove positive movement reward to discourage random movement
            skid_dump_wrong=-0.25,  # Moderate penalty in sparse mode (-0.0025 after normalization)
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
    alignment_coefficient: float = -0.16  # stronger trench-axis alignment
    distance_coefficient: float = -0.14   #-0.04
    # Cabin alignment shaping disabled for now (handled indirectly via stronger base alignment).
    cabin_alignment_coefficient: float = 0.0

    curriculum: CurriculumConfig = CurriculumConfig()

    max_steps_in_episode: int = 0  # changed by CurriculumManager
    tile_size: float = 0  # updated in the code
    
    # Agent types configuration: (agent1_type, agent2_type)
    # 0=excavator, 1=truck, 2=skidsteer
    agent_types: tuple = (0,)  # Default: excavator + skidsteer, override with --agent_types in training script
    
    # Action types configuration: (action1_type, action2_type) - optional override
    # 0=tracked, 1=wheeled
    action_types: tuple = (0,)  # Default: (0,0) (uses tracked for all), override with --action_types in training script

    # Agent capacities
    # Truck maximum load capacity (units of dirt), set to the workspace capacity of the excavator to have the same abstraction level
    truck_capacity: int = 52 
    # Skid steer maximum load capacity (units of dirt), set to the workspace capacity of the excavator to have the same abstraction level
    skidsteer_capacity: int = 52
    
    # Truck road restrictions
    # If True, trucks can only move on roads (non-dumpable tiles) OR dump zones (original behavior)
    # If False, trucks can move everywhere (no restrictions)
    truck_road_restricted: bool = False
    
    # Agent-neutral relocation reward settings.
    relocation_progress_mult: float = 1.5

    # Foundation border digging alignment constraints (env-enforced)
    enforce_foundation_border_alignment: bool = False # set in configs
    foundation_border_width_tiles: int = 2
    foundation_border_proximity_tiles: float = 3.5
    foundation_border_hv_tolerance_rad: float = 0.436  # ~25deg
    foundation_border_diag_tolerance_rad: float = 0.436  # ~25deg
    foundation_corner_relaxation_tiles: float = 2.5
    # 0.0 disables the gate. Values in (0, 1] block excavator dumps unless
    # more than this fraction of raw cone tiles remain dumpable after filter (higher = more restrictive)
    foundation_dump_min_free_fraction: float = 0.0
    debug_foundation_border_checks: bool = False
    enable_reachability_obs: bool = False
    reachability_inflation_tiles: int = 3 # not used if downsample factor != 1
    reachability_downsample_factor: int = 2

    # Appended for positional compatibility with pre-reward-v2 checkpoints.
    reward_stage: int = RewardStage.DENSE_SKILL
    # Used only by ANNEALED_OBJECTIVE: 0 is exactly dense and 1 is exactly
    # terminal-only. The trainer updates this scalar between PPO rollouts.
    terminal_reward_mix: float = 0.0
    # Reward-v2 timing selector (REWARD_V2_TIMING_*). Appended last so earlier
    # checkpoints stay positionally compatible; 0 is the frozen reward_v2
    # reward, bit for bit, and 1 is the adopted v2.1 timing.
    reward_v2_timing_variant: int = REWARD_V2_TIMING_BASELINE
    # Desired tier for the next reset: 0=full, 1=90%, 2=75%, 3=50% complete.
    # State.reset_tier records the tier of the episode already in progress.
    reset_tier: int = 0
    # Global opt-in treatment: block only empty-excavator DO actions that would
    # remove fresh trench target soil from a physically misaligned base pose.
    # This is intentionally not coupled to the legacy reward curriculum.
    enforce_trench_dig_alignment: bool = False
    trench_dig_yaw_tolerance_rad: float = TRENCH_DIG_YAW_TOLERANCE_RAD
    trench_dig_standoff_min_m: float = 3.5
    trench_dig_standoff_max_m: float = 7.0

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

        {
            "maps_path": "foundations_rectangles_real_ring", 
            "max_steps_in_episode": 200,
            "rewards_type": RewardsType.DENSE,
            "apply_trench_rewards": False,
        },
        
        # {
        #     "maps_path": "test_map2", 
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
        #     "maps_path": "foundations_dumpzones_roads", 
        #     "max_steps_in_episode": 800,
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
        #     "max_steps_in_episode":900,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        
        # {
        #     "maps_path": "foundations_hybrid_dumpzones",
        #     "max_steps_in_episode":800,  # 600 Balanced: increased from 300 but reduced from 500
        #     "rewards_type": RewardsType.DENSE,
        #     "apply_trench_rewards": False,
        # },
        #
    
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




class BatchConfig(NamedTuple):
    action_type: Action = TrackedAction  # [WheeledAction, TrackedAction]

    # Config to get data for batched env initialization
    agent: ImmutableAgentConfig = ImmutableAgentConfig()
    maps: ImmutableMapsConfig = ImmutableMapsConfig()
    maps_dims: MapsDimsConfig = MapsDimsConfig()

    curriculum_global: CurriculumGlobalConfig = CurriculumGlobalConfig()
