from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from terra.config import Rewards

class CurriculumManager(NamedTuple):
    """
    This class defines the logic to change the environment configuration given the performance of the agent.
    This class is not stateful, the state of the curriculum is fully contained in the EnvConfig object.
    """
    max_level: int
    increase_level_threshold: int
    decrease_level_threshold: int
    max_steps_in_episode_per_level: Array
    apply_trench_rewards_per_level: Array
    reward_type_per_level: Array

    def _update_single_cfg(self, timestep):
        if self.max_level == 0:
            return timestep
        
        env_cfg = timestep.env_cfg
        done = jnp.all(timestep.done)
        completed = jnp.all(timestep.info["task_done"])

        failure = done & ~completed
        success = done & completed

        def handle_update():
            consecutive_failures = jax.lax.cond(
                failure,
                lambda: env_cfg.curriculum.consecutive_failures + 1,
                lambda: 0,
            )
            consecutive_successes = jax.lax.cond(
                success,
                lambda: env_cfg.curriculum.consecutive_successes + 1,
                lambda: 0,
            )
            return consecutive_failures, consecutive_successes

        consecutive_failures, consecutive_successes = jax.lax.cond(
            done,
            handle_update,
            lambda: (env_cfg.curriculum.consecutive_failures, env_cfg.curriculum.consecutive_successes),
        )

        do_increase = consecutive_successes >= self.increase_level_threshold
        do_decrease = consecutive_failures >= self.decrease_level_threshold

        level = jax.lax.cond(
            do_increase,
            lambda: jnp.minimum(env_cfg.curriculum.level + 1, self.max_level),
            lambda: jax.lax.cond(
                do_decrease,
                lambda: jnp.maximum(env_cfg.curriculum.level - 1, 0),
                lambda: env_cfg.curriculum.level,
            ),
        )

        max_steps_in_episode = self.max_steps_in_episode_per_level[level]
        apply_trench_rewards = self.apply_trench_rewards_per_level[level]

        rewards_list = [Rewards.dense, Rewards.sparse]
        reward_type = self.reward_type_per_level[level]
        rewards = jax.lax.switch(reward_type, rewards_list)
        
        env_cfg = env_cfg._replace(
            rewards=rewards,
            apply_trench_rewards=apply_trench_rewards,
            max_steps_in_episode=max_steps_in_episode,
            curriculum=env_cfg.curriculum._replace(
                level=level,
                consecutive_failures=consecutive_failures,
                consecutive_successes=consecutive_successes,
            )
        )
        timestep = timestep._replace(env_cfg=env_cfg)
        return timestep

    def update_cfgs(self, timesteps):
        return jax.vmap(self._update_single_cfg)(timesteps)
