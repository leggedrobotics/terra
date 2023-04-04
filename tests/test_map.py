import torch
import unittest
from src.map import GridMap, HeightMapSingleHole, GridWorld
from src.config import (
    EnvConfig,
    BufferConfig,
    HeightMapSingleHoleConfig,
    ZeroHeightMapConfig,
    FreeTraversabilityMaskMapConfig
    )

class GridMapTest(unittest.TestCase):

    def test_map_init(self):
        map = torch.randn(32, 16)
        gw = GridMap.new(map)
        self.assertTrue(torch.equal(gw.map, map))

    def test_random_map(self):
        gm = GridMap.random_map(torch.tensor([3, 4]), -3, 3)
        self.assertTrue(gm.map.shape[0]== 3 and gm.map.shape[1] == 4)

    def test_single_hole(self):
        gm = HeightMapSingleHole.create(torch.Tensor([3, 4]))
        self.assertEqual(gm.map.sum().item(), -1)


class GridWorldTest(unittest.TestCase):

    def test_world_setup(self):
        tm_config = HeightMapSingleHoleConfig(dims=(3, 2))
        am_config = ZeroHeightMapConfig(dims=(3, 2))
        travmask_config = FreeTraversabilityMaskMapConfig(dims=(3, 2))
        env_cfg = EnvConfig(
            target_map=tm_config,
            action_map=am_config,
            traversability_mask_map=travmask_config
        )
        buf_cfg = BufferConfig()

        gw = GridWorld.new(env_cfg, buf_cfg)

        print(gw.target_map.map)
        print(gw.action_map.map)
        print(gw.traversability_mask_map.map)

if __name__ == "__main__":
    unittest.main()
