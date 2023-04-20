from typing import NamedTuple

MapConfig = NamedTuple

class TargetMapConfig(MapConfig):
    width: int = 12
    height: int = 12


class ActionMapConfig(MapConfig):
    width: int = 12
    height: int = 12


class AgentConfig(NamedTuple):
    # Have the following numbers odd for the COG to be the center cell of the robot
    # NOT SURE IT WILL WORK NICELY IF EVEN
    width: int = 1
    height: int = 3
    
    angles_base: int = 4
    angles_cabin: int = 8

    move_tiles: int = 3  # number of tiles of progress for every move action
    #  Note: move_tiles is also used as radius of excavation
    #       (we dig as much as move_tiles in the radial distance)

    max_arm_extension: int = 1  # numbering starts from 0 (0 is the closest level)

    dig_depth: int = 1  # how much every dig action digs


class EnvConfig(NamedTuple):
    agent: AgentConfig = AgentConfig()

    target_map: MapConfig = TargetMapConfig()
    action_map: MapConfig = ActionMapConfig()

    tile_size: float = 10.
