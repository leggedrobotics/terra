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
    kernel_size: tuple = (3, 3),
):
    n_dump_areas = n_clusters // 2
    n_dig_areas = n_clusters - n_dump_areas

    key, *subkeys = jax.random.split(key, 5)
    map = jnp.zeros((width, height), dtype=IntMap)
    x_dump = jax.random.randint(subkeys[0], (n_dump_areas,), minval=0, maxval=width)
    y_dump = jax.random.randint(subkeys[1], (n_dump_areas,), minval=0, maxval=height)
    x_dig = jax.random.randint(subkeys[2], (n_dig_areas,), minval=0, maxval=width)
    y_dig = jax.random.randint(subkeys[3], (n_dig_areas,), minval=0, maxval=height)

    # TODO guarantee they do not cancel each other
    map = map.at[(x_dump, y_dump)].set(1.0)
    map = map.at[(x_dig, y_dig)].set(-1.0)

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

    kernel = jnp.ones(kernel_size)

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
