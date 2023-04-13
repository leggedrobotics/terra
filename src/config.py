from typing import NamedTuple

MapConfig = NamedTuple

class TargetMapConfig(MapConfig):
    width: int = 3
    height: int = 4


class ActionMapConfig(MapConfig):
    width: int = 3
    height: int = 4


class EnvConfig(NamedTuple):
    target_map: MapConfig = TargetMapConfig()
    action_map: MapConfig = ActionMapConfig()

    action_batch_capacity: int = 10


class BufferConfig(NamedTuple):
    pass
