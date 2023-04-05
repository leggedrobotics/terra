import unittest
from src.state import State
from src.config import (
    HeightMapSingleHoleConfig,
    ZeroHeightMapConfig,
    FreeTraversabilityMaskMapConfig,
    EnvConfig,
    BufferConfig
)

class StateTest(unittest.TestCase):

    def test_state_init(self):
        tm_config = HeightMapSingleHoleConfig(dims=(3, 2))
        am_config = ZeroHeightMapConfig(dims=(3, 2))
        travmask_config = FreeTraversabilityMaskMapConfig(dims=(3, 2))
        env_cfg = EnvConfig(
            target_map=tm_config,
            action_map=am_config,
            traversability_mask_map=travmask_config,
            agent_type="WHEELED",
            action_queue_capacity=10
        )
        buf_cfg = BufferConfig()
        state = State.new(env_cfg, buf_cfg)

        self.assertEqual(state.env_steps, 0)


if __name__ == "__main__":
    unittest.main()
