from typing import NamedTuple
from src.utils import Float
from src.actions import Action, TrackedAction

MapConfig = NamedTuple

class TargetMapConfig(MapConfig):
    width: int = 20
    height: int = 20


class ActionMapConfig(MapConfig):
    width: int = 20
    height: int = 20


class AgentConfig(NamedTuple):
    # Have the following numbers odd for the COG to be the center cell of the robot
    # NOT SURE IT WILL WORK NICELY IF EVEN
    width: int = 1
    height: int = 3
    
    angles_base: int = 4
    angles_cabin: int = 8

    move_tiles: int = 2  # number of tiles of progress for every move action
    #  Note: move_tiles is also used as radius of excavation
    #       (we dig as much as move_tiles in the radial distance)

    max_arm_extension: int = 1  # numbering starts from 0 (0 is the closest level)

    dig_depth: int = 1  # how much every dig action digs
    # max_dig: int = -3  # soft max after which the agent pays a cost  # TODO implement
    # max_dump: int = 3  # soft max after which the agent pays a cost  # TODO implement
    

class Rewards(NamedTuple):
    existence: Float = -0.05

    collision_move: Float = -1.
    move_while_loaded: Float = -0.2
    move: Float = -0.1

    collision_turn: Float = -1.
    base_turn: Float = -0.05

    cabin_turn: Float = -0.05

    dig_wrong: Float = -2.  # given both if loaded stayed the same, or if new map is not closer than old to target
    dump_wrong: Float = -2.  # given if loaded stayed the same

    dig_correct: Float = 2.  # given if the new map is closer to target map than before
    dump_correct: Float = 0.2  # implemented as dump where not digged

    terminal: Float = 10.  # given if the action map is the same as the target map where it matters (digged tiles)


class EnvConfig(NamedTuple):
    agent: AgentConfig = AgentConfig()

    target_map: MapConfig = TargetMapConfig()
    action_map: MapConfig = ActionMapConfig()

    rewards = Rewards()

    tile_size: float = 10.

    rewards_level: int = 0  # 0 to N, the level of the rewards to assign in curriculum learning (the higher, the more sparse)

    max_episode_duration: int = 100  # in number of steps

class BatchConfig(NamedTuple):
    action_type: Action = TrackedAction()
