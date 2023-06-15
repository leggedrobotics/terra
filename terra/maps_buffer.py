from functools import partial
from typing import NamedTuple

import jax
from jax import Array


class MapsBuffer(NamedTuple):
    maps: Array
    key: jax.random.KeyArray

    @classmethod
    def new(
        self,
        maps: Array,
        key: jax.random.KeyArray,
    ) -> "MapsBuffer":
        return MapsBuffer(
            maps=maps,
            key=key,
        )

    @jax.jit
    def shuffle(self) -> "MapsBuffer":
        key, subkey = jax.random.split(self.key)
        maps = jax.random.permutation(subkey, self.maps, 0)
        return MapsBuffer.new(
            maps=maps,
            key=key,
        )

    @partial(jax.jit, static_argnums=(1,))
    def sample(self, n_envs: int) -> Array:
        maps = self.maps[:n_envs]
        return maps

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, __o: "MapsBuffer") -> bool:
        return self.key == __o.key
