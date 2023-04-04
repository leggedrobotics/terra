import unittest
from src.agent import Agent
from src.config import (
    HeightMapSingleHoleConfig,
    ZeroHeightMapConfig,
    FreeTraversabilityMaskMapConfig,
    EnvConfig,
)

class AgentTest(unittest.TestCase):

    def test_wheeled_agent_init(self):
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
        agent = Agent.new(env_cfg)

        self.assertEqual(agent.action_queue.data.shape[0], 10)
        self.assertEqual(len(agent.agent_type.shape), 1)

    def test_tracked_agent_init(self):
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
        agent = Agent.new(env_cfg)

        self.assertEqual(agent.action_queue.data.shape[0], 10)
        self.assertEqual(len(agent.agent_type.shape), 1)


if __name__ == "__main__":
    unittest.main()
