import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

IntMap = jnp.int16  # TODO import


def generate_clustered_bitmap(
    width: int,
    height: int,
    n_clusters: int,
    n_tiles_per_cluster: int,
    kernel_size_aggregation: int,
    kernel_size_initial_sampling: int,
    key: int,
    placeholder=None,
):
    kernel_init = jnp.ones((kernel_size_initial_sampling, kernel_size_initial_sampling))

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
        key, map, set_value = carry

        mask = jax.lax.cond(
            set_value < 0,
            lambda: (map < 0).astype(IntMap),
            lambda: (map > 0).astype(IntMap),
        )

        mask_probs = convolve2d(mask, kernel, mode="same", boundary="fill")
        mask_probs = mask_probs * (~(mask).astype(jnp.bool_))
        mask_probs = mask_probs / mask_probs.sum()

        key, subkey = jax.random.split(key)
        # 75% random sample, 25% argmax
        do_random_sample = jax.random.randint(subkey, (), 0, 4).astype(jnp.bool_)

        def _random_sample(key, map):
            key, *subkeys = jax.random.split(key, 3)
            next_tile_idx = jax.random.choice(
                subkeys[1], jnp.arange(0, width * height), p=mask_probs.reshape(-1)
            )
            map = (
                map.reshape(-1).at[next_tile_idx].set(set_value).reshape(width, height)
            )
            return key, map

        def _argmax(key, map):
            next_tile_idx = jnp.argmax(mask_probs.reshape(-1))
            map = (
                map.reshape(-1).at[next_tile_idx].set(set_value).reshape(width, height)
            )
            return key, map

        key, map = jax.lax.cond(do_random_sample, _random_sample, _argmax, key, map)

        set_value *= -1
        carry = key, map, set_value
        return carry

    kernel = jnp.ones((kernel_size_aggregation, kernel_size_aggregation))

    carry = key, map, -1
    carry = jax.lax.fori_loop(0, n_tiles_per_cluster * n_clusters, _loop, carry)
    key, map, _ = carry
    return map, key


if __name__ == "__main__":
    key = jax.random.PRNGKey(131)
    map, key = generate_clustered_bitmap(
        10,
        10,
        4,
        3,
        5,
        key,
    )
    import numpy as np

    # import cv2
    map = np.array(map)
    print(f"{map=}")
