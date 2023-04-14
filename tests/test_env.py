import unittest
from src.env import TerraEnv
from src.config import (
    HeightMapSingleHoleConfig,
    ZeroHeightMapConfig,
    FreeTraversabilityMaskMapConfig,
    EnvConfig,
    BufferConfig
)

class EnvTest(unittest.TestCase):

    def test_env_init(self):
        tm_config = HeightMapSingleHoleConfig(dims=(3, 2))
        am_config = ZeroHeightMapConfig(dims=(3, 2))
        travmask_config = FreeTraversabilityMaskMapConfig(dims=(3, 2))
        env_cfg = EnvConfig(
            target_map=tm_config,
            action_map=am_config,
            traversability_mask_map=travmask_config,
            agent_type="TRACKED",
            action_queue_capacity=10
        )
        buf_cfg = BufferConfig()
        env = TerraEnv(env_cfg, buf_cfg)

        s1 = env.reset()
        s2 = env.reset()

        self.assertEqual(s1, s2)

    def test_env_step(self):
        pass

    def test_env_close(self):
        pass


if __name__ == "__main__":
    unittest.main()
