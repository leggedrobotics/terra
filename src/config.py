from typing import NamedTuple

MapConfig = NamedTuple

class TargetMapConfig(MapConfig):
    width: int = 5
    height: int = 6


class ActionMapConfig(MapConfig):
    width: int = 5
    height: int = 6


class AgentConfig(NamedTuple):
    width: int = 3
    height: int = 2


class EnvConfig(NamedTuple):
    agent: AgentConfig = AgentConfig()

    target_map: MapConfig = TargetMapConfig()
    action_map: MapConfig = ActionMapConfig()


# class BufferConfig(NamedTuple):
#     pass
