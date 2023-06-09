import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

IntMap = jnp.int16  # TODO import


def generate_clustered_bitmap(
    key: int,
    width: int,
    height: int,
    n_clusters: int,
    n_tiles_per_cluster: int,
    kernel_size_aggregation: tuple = (3, 3),
    kernel_size_initial_sampling: tuple = (5, 5),
):
    kernel_init = jnp.ones(kernel_size_initial_sampling)

    def _loop_init(i, carry):
        """
        Init the map by sampling spaced tiles based on the position
        of the previously sampled ones
        (low chance to sample neighbouring tiles).
        """
        key, map, set_value = carry

        mask = (map != 0).astype(IntMap)
        mask_convolved = convolve2d(mask, kernel_init, mode="same", boundary="fill")
        mask_convolved_opposite = (mask_convolved <= mask_convolved.min()).astype(
            IntMap
        )
        p_sampling = mask_convolved_opposite / mask_convolved_opposite.sum()

        key, subkey = jax.random.split(key)
        idx = jax.random.choice(
            subkey, jnp.arange(start=0, stop=width * height), p=p_sampling.reshape(-1)
        )
        map = map.reshape(-1).at[idx].set(set_value).reshape(width, height)

        set_value *= -1
        carry = key, map, set_value
        return carry

    carry = key, jnp.zeros((width, height), dtype=IntMap), -1
    carry = jax.lax.fori_loop(
        lower=0, upper=n_clusters, body_fun=_loop_init, init_val=carry
    )
    key, map, _ = carry

    def _loop(i, carry):
        key, map = carry
        mask_dig = (map < 0).astype(IntMap)
        mask_dump = (map > 0).astype(IntMap)

        mask_dump_probs = convolve2d(mask_dump, kernel, mode="same", boundary="fill")
        mask_dump_probs = mask_dump_probs * (~(mask_dump).astype(jnp.bool_))
        mask_dump_probs = mask_dump_probs / mask_dump_probs.sum()

        mask_dig_probs = convolve2d(mask_dig, kernel, mode="same", boundary="fill")
        mask_dig_probs = mask_dig_probs * (~(mask_dig).astype(jnp.bool_))
        mask_dig_probs = mask_dig_probs / mask_dig_probs.sum()

        key, *subkeys = jax.random.split(key, 3)
        next_dump_tile_idx = jax.random.choice(
            subkeys[0], jnp.arange(0, width * height), p=mask_dump_probs.reshape(-1)
        )
        map = map.reshape(-1).at[next_dump_tile_idx].set(1.0).reshape(width, height)
        next_dig_tile_idx = jax.random.choice(
            subkeys[1], jnp.arange(0, width * height), p=mask_dig_probs.reshape(-1)
        )
        map = map.reshape(-1).at[next_dig_tile_idx].set(-1.0).reshape(width, height)
        carry = key, map
        return carry

    kernel = jnp.ones(kernel_size_aggregation)

    carry = key, map
    carry = jax.lax.fori_loop(0, n_tiles_per_cluster, _loop, carry)
    key, map = carry
    return map, key


if __name__ == "__main__":
    key = jax.random.PRNGKey(131)
    map, key = generate_clustered_bitmap(
        key,
        10,
        10,
        n_clusters=4,
        n_tiles_per_cluster=5,
    )
    import numpy as np

    # import cv2
    map = np.array(map)
    print(f"{map=}")
