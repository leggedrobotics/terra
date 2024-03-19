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
    last_level_type: str

    def _update_single_cfg(self, timestep, rng):
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
            lambda: jax.lax.cond(
                env_cfg.curriculum.level < self.max_level,
                lambda: env_cfg.curriculum.level + 1,
                lambda: jax.lax.cond(
                    self.last_level_type == "none",
                    lambda: env_cfg.curriculum.level,
                    lambda: jax.lax.cond(
                        self.last_level_type == "random",
                        lambda: jax.random.randint(rng, (), 0, self.max_level + 1),
                        lambda: 97,  # Error case
                    ),
                ),
            ),
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
    
    def _reset_single_cfg(self, env_cfg):
        max_steps_in_episode = self.max_steps_in_episode_per_level[0]
        apply_trench_rewards = self.apply_trench_rewards_per_level[0]
        
        rewards_list = [Rewards.dense, Rewards.sparse]
        reward_type = self.reward_type_per_level[0]
        rewards = jax.lax.switch(reward_type, rewards_list)

        env_cfg = env_cfg._replace(
            rewards=rewards,
            apply_trench_rewards=apply_trench_rewards,
            max_steps_in_episode=max_steps_in_episode,
        )
        return env_cfg

    def update_cfgs(self, timesteps, rng):
        return jax.vmap(self._update_single_cfg)(timesteps, rng)
    
    def reset_cfgs(self, env_cfgs):
        return jax.vmap(self._reset_single_cfg)(env_cfgs)
