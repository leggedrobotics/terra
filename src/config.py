from typing import NamedTuple, Tuple

MapConfig = NamedTuple


class EnvConfig(NamedTuple):
    target_map: MapConfig
    action_map: MapConfig


class BufferConfig(NamedTuple):
    pass
