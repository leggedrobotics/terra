from typing import NamedTuple, Tuple

MapConfig = NamedTuple

class HeightMapSingleHoleConfig(MapConfig):
    dims: Tuple[int]
    type: str = "HeightMapSingleHole"

class HeightMapConfig(MapConfig):
    dims: Tuple[int]
    low: int
    high: int
    type: str = "HeightMap"

class ZeroHeightMapConfig(MapConfig):
    dims: Tuple[int]
    type: str = "ZeroHeightMap"

class FreeTraversabilityMaskMapConfig(MapConfig):
    dims: Tuple[int]
    type: str = "FreeTraversabilityMaskMap"


class EnvConfig(NamedTuple):
    target_map: MapConfig
    action_map: MapConfig
    traversability_mask_map: MapConfig

    agent_type: str
    action_queue_capacity: int


class BufferConfig(NamedTuple):
    pass
