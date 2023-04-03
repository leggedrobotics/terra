import torch
from typing import NamedTuple, Type
from frontend import GridMapFrontend, GridWorldFrontend, FrontendConfig, AgentFrontend
from config import BufferConfig, EnvConfig


class GridMap(NamedTuple):
    dims: torch.Tensor

    @property
    def width(self) -> int:
        return self.dims[0]
    
    @property
    def height(self) -> int:
        return self.dims[1]
    
    @staticmethod
    def new(dims: torch.Tensor) -> "GridMap":
        pass

    @staticmethod
    def random_map(seed: torch.Tensor) -> "GridMap":
        pass

    @classmethod
    def from_frontend(cls: Type['GridMap'], lux_map: GridMapFrontend) -> "GridMap":
        pass

    def to_frontend(self) -> GridMapFrontend:
        pass


class HeightMap(GridMap):
    pass


class MaskMap(GridMap):
    pass


class GridWorld(NamedTuple):
    seed: torch.Tensor

    target_map: HeightMap
    action_map: HeightMap
    traversability_mask_map: MaskMap

    @property
    def width(self) -> int:
        return self.target_map.width

    @property
    def height(self) -> int:
        return self.target_map.height
    
    @classmethod
    def new(cls, seed: int, env_cfg: EnvConfig, buf_cfg: BufferConfig) -> "GridWorld":
        pass

    @classmethod
    def from_frontend(cls: Type["GridWorld"], grid_world: GridWorldFrontend, buf_cfg: BufferConfig) -> "GridWorld":
        pass

    def to_frontend(self, frontend_cfg: FrontendConfig, agent_frontend: AgentFrontend) -> GridWorldFrontend:
        pass
    