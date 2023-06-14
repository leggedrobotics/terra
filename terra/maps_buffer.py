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
    def _shuffle(self) -> "MapsBuffer":
        key, subkey = jax.random.split(self.key)
        print(f"{self.maps.shape=}")
        maps = jax.random.permutation(subkey, self.maps, 0)
        return MapsBuffer.new(
            maps=maps,
            key=key,
        )

    @partial(jax.jit, static_argnums=(1,))
    def sample(self, n_envs: int) -> tuple["MapsBuffer", Array]:
        maps_buffer = self._shuffle()
        maps = maps_buffer.maps[:n_envs]
        return maps_buffer, maps
