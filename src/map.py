import torch
from typing import NamedTuple, Type
from src.frontend import GridMapFrontend, GridWorldFrontend, FrontendConfig, AgentFrontend
from src.config import BufferConfig, EnvConfig


class GridMap(NamedTuple):
    map: torch.Tensor

    @property
    def width(self) -> int:
        pass
    
    @property
    def height(self) -> int:
        pass
    
    @staticmethod
    def new(map: torch.Tensor) -> "GridMap":
        assert len(map.shape) == 2
        
        return GridMap(
            GridMap.__annotations__["map"](map)
        )

    @staticmethod
    def random_map(dims: torch.Tensor, low: int, high: int) -> "GridMap":
        assert len(dims.shape) == 1
        assert dims.shape[0] == 2
        dims = dims.to(dtype=torch.int)

        return GridMap.new(torch.randint(low=low, high=high, size=(dims[0], dims[1])))

    @classmethod
    def from_frontend(cls: Type['GridMap'], lux_map: GridMapFrontend) -> "GridMap":
        pass

    def to_frontend(self) -> GridMapFrontend:
        pass


class HeightMap(GridMap):
    pass


class MaskMap(GridMap):
    pass


class HeightMapSingleHole(HeightMap):
    """
    This map has a single cell with value -1, and all the rest 0.
    """
    @staticmethod
    def create(dims: torch.Tensor) -> "HeightMapSingleHole":
        assert len(dims.shape) == 1
        assert dims.shape[0] == 2
        dims = dims.to(dtype=torch.int)

        map = torch.zeros((dims[0].item(), dims[1].item()), dtype=torch.int)
        map[torch.randint(high=dims[0], size=(1,)), torch.randint(high=dims[1], size=(1,))] = -1
        return HeightMapSingleHole.new(map)


class ZeroHeightMap(HeightMap):
    """
    This map has all the cells with value 0.
    """
    @staticmethod
    def create(dims: torch.Tensor) -> "ZeroHeightMap":
        assert len(dims.shape) == 1
        assert dims.shape[0] == 2
        dims = dims.to(dtype=torch.int)

        return ZeroHeightMap.new(torch.zeros((dims[0].item(), dims[1].item()), dtype=torch.int))


class FreeTraversabilityMaskMap(MaskMap):
    """
    This traversability map has all the cells = 0 (meaning no obstacles).
    """
    @staticmethod
    def create(dims: torch.Tensor) -> "FreeTraversabilityMaskMap":
        assert len(dims.shape) == 1
        assert dims.shape[0] == 2
        dims = dims.to(dtype=torch.int)

        return FreeTraversabilityMaskMap.new(torch.zeros((dims[0].item(), dims[1].item()), dtype=torch.int))


class GridWorld(NamedTuple):
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
    def new(cls, env_cfg: EnvConfig, buf_cfg: BufferConfig) -> "GridWorld":
        assert env_cfg.target_map.dims == env_cfg.action_map.dims
        assert env_cfg.traversability_mask_map.dims == env_cfg.action_map.dims

        # TODO make the following more elegant (remove if statements)

        # Target map
        if env_cfg.target_map.type == "HeightMapSingleHole":
            target_map = HeightMapSingleHole.create(torch.tensor(env_cfg.target_map.dims))
        elif env_cfg.target_map.type == "HeightMap":
            target_map = HeightMap.random_map(
                torch.tensor(env_cfg.target_map.dims),
                env_cfg.target_map.low,
                env_cfg.target_map.high
                )
        else:
            raise ValueError(f"Map type {env_cfg.target_map.type} is not supported.")

        # Action map
        if env_cfg.action_map.type == "ZeroHeightMap":
            action_map = ZeroHeightMap.create(torch.tensor(env_cfg.action_map.dims))
        else:
            raise ValueError(f"Map type {env_cfg.action_map.type} is not supported.")
        
        # Traversability mask map
        if env_cfg.traversability_mask_map.type == "FreeTraversabilityMaskMap":
            traversability_mask_map = FreeTraversabilityMaskMap.create(
                torch.tensor(env_cfg.traversability_mask_map.dims)
                )

        return GridWorld(
            GridWorld.__annotations__["target_map"](target_map),
            GridWorld.__annotations__["action_map"](action_map),
            GridWorld.__annotations__["traversability_mask_map"](traversability_mask_map)
        )


    @classmethod
    def from_frontend(cls: Type["GridWorld"], grid_world: GridWorldFrontend, buf_cfg: BufferConfig) -> "GridWorld":
        pass

    def to_frontend(self, frontend_cfg: FrontendConfig, agent_frontend: AgentFrontend) -> GridWorldFrontend:
        pass
    