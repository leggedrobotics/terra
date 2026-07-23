import unittest

import jax
import jax.numpy as jnp

from terra.config import EnvConfig
from terra.curriculum import CurriculumManager
from terra.env import TimeStep


class CurriculumTransitionsTest(unittest.TestCase):
    def setUp(self):
        self.manager = CurriculumManager(
            max_level=2,
            increase_level_threshold=3,
            decrease_level_threshold=3,
            max_steps_in_episode_per_level=jnp.array(
                [450, 450, 450], dtype=jnp.int32
            ),
            apply_trench_rewards_per_level=jnp.array(
                [False, False, False], dtype=jnp.bool_
            ),
            reward_type_per_level=jnp.array(
                [0, 0, 0], dtype=jnp.int32
            ),
            last_level_type="none",
        )

    @staticmethod
    def terminal(env_cfg, success):
        return TimeStep(
            state=jnp.int32(0),
            observation={},
            reward=jnp.float32(0),
            done=jnp.bool_(True),
            info={"task_done": jnp.bool_(success)},
            env_cfg=env_cfg,
        )

    def apply_outcomes(self, level, outcomes):
        env_cfg = EnvConfig()._replace(
            curriculum=EnvConfig().curriculum._replace(level=level)
        )
        for index, success in enumerate(outcomes):
            timestep = self.terminal(env_cfg, success)
            timestep = self.manager._update_single_cfg(
                timestep, jax.random.PRNGKey(index)
            )
            env_cfg = timestep.env_cfg
        return env_cfg

    def test_three_successes_promote_and_three_failures_demote(self):
        promoted = self.apply_outcomes(0, [True, True, True])
        self.assertEqual(int(promoted.curriculum.level), 1)
        self.assertEqual(
            int(promoted.curriculum.consecutive_successes), 0
        )

        demoted = self.apply_outcomes(1, [False, False, False])
        self.assertEqual(int(demoted.curriculum.level), 0)
        self.assertEqual(
            int(demoted.curriculum.consecutive_failures), 0
        )

    def test_last_level_none_never_randomizes(self):
        terminal = self.apply_outcomes(2, [True] * 9)
        self.assertEqual(int(terminal.curriculum.level), 2)


if __name__ == "__main__":
    unittest.main()
