from typing import NamedTuple

MapConfig = NamedTuple

class TargetMapConfig(MapConfig):
    width: int = 10
    height: int = 12


class ActionMapConfig(MapConfig):
    width: int = 10
    height: int = 12


class AgentConfig(NamedTuple):
    # Have the following numbers odd for the COG to be the center cell of the robot
    width: int = 3
    height: int = 1
    angles_base: int = 4
    angles_cabin: int = 8

    move_tiles: int = 2  # number of tiles of progress for every move action
    #  (to be made congruent with dig space dimensions and tile dimensions)


class EnvConfig(NamedTuple):
    agent: AgentConfig = AgentConfig()

    target_map: MapConfig = TargetMapConfig()
    action_map: MapConfig = ActionMapConfig()


# class BufferConfig(NamedTuple):
#     pass
