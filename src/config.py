from typing import NamedTuple
from src.utils import Float

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
    collision: Float = -1.
    move_while_loaded: Float = -0.2
    move: Float = -0.1
    base_turn: Float = -0.05
    cabin_turn: Float = -0.05
    dig_wrong: Float = -2.
    dump_wrong: Float = -2.
    existence: Float = -0.05

    dig_correct: Float = 2.
    dump_correct: Float = 2.
    terminal: Float = 100.


class EnvConfig(NamedTuple):
    agent: AgentConfig = AgentConfig()

    target_map: MapConfig = TargetMapConfig()
    action_map: MapConfig = ActionMapConfig()

    rewards = Rewards()

    tile_size: float = 10.

    rewards_level: int = 0  # 0 to N, the level of the rewards to assign in curriculum learning (the higher, the more sparse)

    max_episode_duration: int = 100  # in number of steps
