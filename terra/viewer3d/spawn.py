"""Bounded preflight of Terra's seeded reset sampler for interactive sessions.

This checks the same proposal stream and footprints as Agent.new. It does not
create or replace Terra states, and does not alter the training reset path.
"""

import jax
import jax.numpy as jnp
import numpy as np

from terra.map import compute_dynamic_dumpability
from terra.settings import IntMap
from terra.utils import compute_polygon_mask, get_agent_corners


@jax.jit
def _bounded_pose(
    key,
    width,
    height,
    angles_base,
    edge_length_px,
    max_traversable_x,
    max_traversable_y,
    obstacles,
    action_map,
    allowed_mask,
    min_border_distance,
    max_attempts,
):
    """Mirror agent._get_random_init_state, adding only a proposal-count limit."""
    margin = jnp.ceil(jnp.max(jnp.array([width / 2 - 1, height / 2 - 1]))).astype(
        IntMap
    )
    # Preserve the runtime sampler's clamped domain, including its narrow domain
    # when obstacles occupy the first row or column. A full-grid fallback here
    # could approve a seed that the actual unbounded reset can never sample.
    minimum_extent = 2 * margin + 1
    max_x = jnp.maximum(
        jnp.minimum(max_traversable_x, edge_length_px), minimum_extent
    ).astype(IntMap)
    max_y = jnp.maximum(
        jnp.minimum(max_traversable_y, edge_length_px), minimum_extent
    ).astype(IntMap)
    rows, cols = obstacles.shape

    def condition(carry):
        _, _, _, attempts, found = carry
        return (attempts < max_attempts) & ~found

    def propose(carry):
        key, _, _, attempts, _ = carry
        key, x_key, y_key, angle_key = jax.random.split(key, 4)
        # The runtime samples coordinates with randint's default int32 dtype,
        # then casts, while it samples the angle directly as IntMap.
        x = jax.random.randint(x_key, (1,), margin, max_x - margin)
        y = jax.random.randint(y_key, (1,), margin, max_y - margin)
        position = IntMap(jnp.concatenate((x, y)))
        angle = jax.random.randint(angle_key, (1,), 0, angles_base, dtype=IntMap)
        corners = get_agent_corners(position, angle, width, height, angles_base)
        footprint = compute_polygon_mask(corners, rows, cols)
        collision = jnp.any(footprint & (obstacles == 1))
        nonflat = jnp.any(footprint & (action_map != 0))
        prohibited = jnp.any(footprint & ~allowed_mask)
        border_distance = jnp.minimum(
            jnp.minimum(position[0], rows - 1 - position[0]),
            jnp.minimum(position[1], cols - 1 - position[1]),
        )
        border_violation = (min_border_distance >= 0) & (
            border_distance < min_border_distance
        )
        found = ~(collision | nonflat | prohibited | border_violation)
        return key, position, angle, attempts + 1, found

    return jax.lax.while_loop(
        condition,
        propose,
        (
            key,
            jnp.array([-1, -1], dtype=IntMap),
            jnp.full((1,), -1, dtype=IntMap),
            jnp.int32(0),
            jnp.bool_(False),
        ),
    )


def validate_spawn(seed, config, maps, max_attempts=4096):
    """Validate the exact configured reset and return its accepted pose records.

    ``maps`` uses ManualSession's target, padding, trench axes/type, foundation
    axes/type, static dumpability, action, distance order. The result contains
    ``id``, ``position`` ([row, column]), ``angle_base`` (discrete index), and
    ``attempts`` for each original active slot. Terra reset still chooses these
    poses itself. A failed bounded search raises before starting its unbounded
    sampler, with a suggestion to try another seed or map.
    """
    if bool(np.asarray(config.truck_road_restricted)):
        raise ValueError("Manual spawn preflight requires truck_road_restricted=False.")
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise ValueError("max_attempts must be a positive integer.")
    agent_types = tuple(config.agent_types)
    if not 1 <= len(agent_types) <= 4 or any(
        int(kind) not in (0, 1, 2) for kind in agent_types
    ):
        raise ValueError("Spawn preflight requires one to four valid agent types.")
    padding = jnp.asarray(maps[1])
    action = jnp.asarray(maps[7])
    if (
        padding.ndim != 2
        or padding.shape[0] != padding.shape[1]
        or action.shape != padding.shape
    ):
        raise ValueError("Spawn preflight requires aligned square, unbatched maps.")
    dynamic_dumpability = compute_dynamic_dumpability(maps[6], action)
    combined_obstacles = padding == 1
    max_x = (padding[:, 0] == 0).sum()
    max_y = (padding[0] == 0).sum()
    # State.new passes the original reset key straight into Agent.new, which
    # reserves split 0 and uses splits 1..4 for the four original agent slots.
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    poses = []
    for index, kind in enumerate(agent_types):
        _, position, angle, attempts, found = _bounded_pose(
            keys[index + 1],
            config.agent.width,
            config.agent.height,
            config.agent.angles_base,
            config.maps.edge_length_px,
            max_x,
            max_y,
            combined_obstacles,
            action,
            dynamic_dumpability,
            jnp.int32(8 if int(kind) == 0 else -1),
            jnp.int32(max_attempts),
        )
        if not bool(np.asarray(found)):
            raise ValueError(
                f"Cannot place agent {index + 1} of {len(agent_types)} within "
                f"{max_attempts} reset proposals for seed {seed}. Terra requires "
                "flat, dumpable space for each footprint in its sampling domain. "
                "Try another --seed, fewer --agents, or a map with more free space; "
                "obstacles along the first row/column can also restrict spawning."
            )
        poses.append(
            {
                "id": index,
                "position": np.asarray(position).tolist(),
                "angle_base": int(np.asarray(angle)[0]),
                "attempts": int(np.asarray(attempts)),
            }
        )
        # Earlier accepted agents affect all later proposals. Use precisely the
        # runtime geometry; a free-space count alone cannot prove this seed fits.
        corners = get_agent_corners(
            position,
            angle,
            config.agent.width,
            config.agent.height,
            config.agent.angles_base,
        )
        combined_obstacles = combined_obstacles | compute_polygon_mask(
            corners, *padding.shape
        )
    return poses
