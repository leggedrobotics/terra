"""
Benchmark:

batch 100
CPU 0.6
GPU 1.1

batch 10k
CPU 0.7
GPU 1.1

batch 100k
CPU 1.0
GPU 1.1

batch 1M
CPU 5.4
GPU 1.1

batch 10M
CPU 84
GPU 1.5
"""

import os
import jax
import jax.numpy as jnp
from jax import random
from jax import Array
from typing import NamedTuple
import time


class Line(NamedTuple):
    state: jnp.float32
    reward_l: jnp.float32
    dones: jnp.bool_ = False
    rewards: jnp.float32 = 0.0

    @classmethod
    def new(cls, seed: int, reward_l: int = 5) -> "Line":
        key = random.PRNGKey(seed)
        state = random.randint(key, (1,), minval=-2, maxval=2)
        reward_l = jnp.ones_like(state) * reward_l

        return Line(state, reward_l)

    def step(self, u: jnp.float32) -> "Line":
        # TODO assert

        state = self.state + u
        dones = self.state >= self.reward_l
        rewards = self.state - self.reward_l
        return Line(
            state=state,
            reward_l=self.reward_l,
            dones=dones,
            rewards=rewards
        )


class LineBatch:
    def reset(self, seeds: Array):
        return jax.vmap(Line.new)(seeds)
    
    def step(self, states: Line, u: Array):
        # TODO if vmap is used here, everty single u is applied to every state
        return states.step(u)


class Policy:
    pass

def get_random_u(key: jnp.float32) -> jnp.float32:
    return random.normal(key, (1,))

def vmap_get_random_u(keys: Array) -> Array:
    return jax.vmap(get_random_u)(keys)


if __name__ == "__main__":
    batch_size = 10000000
    epochs = 10
    max_rollout = 50

    print(f"Using {jnp.ones(3).device_buffer.device()}")
    print(f"{batch_size=}")
    print(f"{epochs=}")
    print(f"{max_rollout=}")

    seed = 30
    key = random.PRNGKey(seed)

    seeds_env = jnp.arange(batch_size)

    # policy = Policy()

    env = LineBatch()

    s = time.time()

    for i in range(epochs):
        states = env.reset(seeds_env)
        for r_i in range(max_rollout):
            key, sub_key = random.split(key)
            # u = vmap_get_random_u(jnp.array(sub_keys))

            # print(f"{states=}")

            u = random.normal(sub_key, (batch_size, 1))

            # print(f"{u=}")

            states = env.step(states, u)

        #     if r_i > 1:
        #         break

        # break

    e = time.time()
    print(f"Duration = {e - s} seconds.")
