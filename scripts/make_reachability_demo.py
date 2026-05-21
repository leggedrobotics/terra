from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from terra.wrappers import TraversabilityMaskWrapper


DATASET_MAP_INDEX = 10
DATASET_MAP_DIR = Path("data/terra/train/foundations")


def downsample_or_factor(mask: np.ndarray, factor: int) -> np.ndarray:
    h, w = mask.shape
    h2 = h // factor
    w2 = w // factor
    trimmed = mask[: factor * h2, : factor * w2]
    return trimmed.reshape(h2, factor, w2, factor).any(axis=(1, 3))


def upsample_nearest_factor(mask: np.ndarray, factor: int, out_h: int, out_w: int) -> np.ndarray:
    up = np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
    out = np.zeros((out_h, out_w), dtype=up.dtype)
    copy_h = min(out_h, up.shape[0])
    copy_w = min(out_w, up.shape[1])
    out[:copy_h, :copy_w] = up[:copy_h, :copy_w]
    return out


def _find_agent_footprint(blocked: np.ndarray, height: int = 5, width: int = 6) -> np.ndarray:
    h, w = blocked.shape
    # Prefer a lower-left-ish starting pose, similar to entering the site from one side.
    for y in range(h - height - 2, 2, -1):
        for x in range(2, w - width - 2):
            if not blocked[y : y + height, x : x + width].any():
                agent_mask = np.zeros_like(blocked, dtype=bool)
                agent_mask[y : y + height, x : x + width] = True
                return agent_mask
    raise RuntimeError("Could not place demo agent footprint on a passable patch")


def _stamp_spoil_pile(action_map: np.ndarray, blocked: np.ndarray, cy: int, cx: int, radius: int) -> None:
    h, w = action_map.shape
    yy, xx = np.ogrid[:h, :w]
    pile = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    pile &= ~blocked
    action_map[pile] = 1


def load_real_half_dug_foundation(index: int = DATASET_MAP_INDEX):
    target_map = np.load(DATASET_MAP_DIR / "images" / f"img_{index}.npy")
    static_obstacles = np.load(DATASET_MAP_DIR / "occupancy" / f"img_{index}.npy").astype(bool)
    target_dig = target_map < 0

    ys, xs = np.where(target_dig)
    split_x = np.median(xs)
    already_dug = target_dig & (np.indices(target_map.shape)[1] <= split_x)

    action_map = np.zeros_like(target_map, dtype=np.int8)
    action_map[already_dug] = -1

    dug_y, dug_x = np.where(already_dug)
    centroid_y = int(np.round(dug_y.mean()))
    centroid_x = int(np.round(dug_x.mean()))
    blocked_for_piles = static_obstacles | (action_map != 0) | target_dig
    for dy, dx, radius in [(-10, -12, 3), (-6, 14, 4), (10, -10, 3), (13, 10, 3)]:
        _stamp_spoil_pile(
            action_map,
            blocked_for_piles,
            np.clip(centroid_y + dy, 3, action_map.shape[0] - 4),
            np.clip(centroid_x + dx, 3, action_map.shape[1] - 4),
            radius,
        )
        blocked_for_piles = static_obstacles | (action_map != 0) | target_dig

    blocked = static_obstacles | (action_map != 0)
    agent_mask = _find_agent_footprint(blocked)
    return target_map, action_map, static_obstacles, agent_mask


def make_demo_map(size: int = 64, scenario: str = "corridor") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_map = np.zeros((size, size), dtype=np.int8)
    static_obstacles = np.zeros_like(action_map, dtype=bool)

    if scenario == "foundation_half_dug":
        # Foundation-like trench footprint. The north/west/center runs are dug,
        # while the south/east runs are still open ground, as in a half-finished job.
        action_map[13:17, 13:51] = -1
        action_map[17:42, 13:17] = -1
        action_map[27:31, 20:44] = -1
        action_map[34:38, 18:33] = -1

        # Small over-excavated and spoil areas from the first half of the work.
        action_map[17:22, 19:25] = -1
        action_map[9:15, 52:60] = 1
        action_map[45:53, 7:19] = 1
        action_map[49:57, 41:55] = 1

        # Site constraints: perimeter fence/no-go zones, a machine pad, and a
        # narrow entrance. These produce realistic disconnected pockets.
        static_obstacles[4:8, 4:56] = True
        static_obstacles[8:30, 4:8] = True
        static_obstacles[42:60, 4:8] = True
        static_obstacles[56:60, 4:56] = True
        static_obstacles[8:20, 56:60] = True
        static_obstacles[36:56, 56:60] = True
        static_obstacles[22:34, 54:58] = True
        static_obstacles[38:43, 24:38] = True
        static_obstacles[20:25, 35:40] = True

        # A small sealed utility pit inside the site: empty-looking but unreachable.
        static_obstacles[41:51, 28:31] = True
        static_obstacles[41:51, 39:42] = True
        static_obstacles[41:44, 28:42] = True
        static_obstacles[48:51, 28:42] = True

        agent_mask = np.zeros_like(action_map, dtype=bool)
        agent_mask[30:35, 8:14] = True
        return action_map, static_obstacles, agent_mask

    if scenario == "enclosed_rooms":
        action_map[8:16, 42:54] = -1
        action_map[45:52, 8:20] = 1
        action_map[28:34, 35:45] = -1

        static_obstacles[6:58, 6:10] = True
        static_obstacles[6:58, 54:58] = True
        static_obstacles[6:10, 6:58] = True
        static_obstacles[54:58, 6:58] = True
        static_obstacles[28:32, 10:54] = True
        static_obstacles[10:28, 30:34] = True
        static_obstacles[32:54, 38:42] = True

        # Two sealed islands on the far side of walls.
        static_obstacles[13:25, 42:46] = True
        static_obstacles[41:50, 25:29] = True

        agent_mask = np.zeros_like(action_map, dtype=bool)
        agent_mask[14:19, 14:20] = True
        return action_map, static_obstacles, agent_mask

    if scenario == "sealed_pocket":
        action_map[12:20, 12:24] = -1
        action_map[39:48, 44:56] = 1
        action_map[25:31, 44:52] = -1

        static_obstacles[4:60, 30:34] = True
        static_obstacles[8:12, 30:34] = False
        static_obstacles[48:52, 30:34] = False

        # Closed square obstacle ring. The inside is visibly empty but unreachable.
        static_obstacles[22:46, 8:12] = True
        static_obstacles[22:46, 24:28] = True
        static_obstacles[22:26, 8:28] = True
        static_obstacles[42:46, 8:28] = True

        # Extra small barriers make the right side partly accessible but with pockets.
        static_obstacles[18:35, 48:52] = True
        static_obstacles[18:22, 40:52] = True
        static_obstacles[31:35, 40:52] = True

        agent_mask = np.zeros_like(action_map, dtype=bool)
        agent_mask[52:57, 40:46] = True
        return action_map, static_obstacles, agent_mask

    # Dug foundation/trench-like regions and dumped piles. Terra treats any
    # nonzero action_map tile as blocked for traversability.
    action_map[14:34, 14:19] = -1
    action_map[29:34, 14:42] = -1
    action_map[41:48, 40:52] = 1
    action_map[8:14, 45:55] = 1
    action_map[23:27, 46:50] = -1

    # Static obstacles create a narrow doorway and a disconnected pocket.
    static_obstacles[4:60, 30:34] = True
    static_obstacles[27:37, 30:34] = False
    static_obstacles[38:42, 30:34] = False
    static_obstacles[50:55, 8:24] = True
    static_obstacles[6:16, 6:10] = True
    static_obstacles[18:22, 52:59] = True

    # Current agent footprint, matching the mask used as the flood-fill seed.
    agent_mask = np.zeros_like(action_map, dtype=bool)
    agent_mask[54:59, 42:48] = True

    return action_map, static_obstacles, agent_mask


def compute_terra_ds2_reachability(
    action_map: np.ndarray,
    static_obstacles: np.ndarray,
    agent_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traversability = action_map != 0
    tm_blocked = static_obstacles | traversability

    # Use Terra's exact static helpers for the ds2 reachability path:
    # _downsample_or_factor2 -> _build_reachability_mask -> _upsample_nearest_factor2.
    blocked_j = jnp.asarray(tm_blocked)
    start_j = jnp.asarray(agent_mask)
    blocked_ds = TraversabilityMaskWrapper._downsample_or_factor2(blocked_j)
    start_ds = TraversabilityMaskWrapper._downsample_or_factor2(start_j)
    passable_ds = jnp.logical_not(blocked_ds)
    reach_ds = TraversabilityMaskWrapper._build_reachability_mask(
        passable_ds,
        start_ds,
    ).astype(jnp.bool_)
    reach_up = TraversabilityMaskWrapper._upsample_nearest_factor2(
        reach_ds,
        action_map.shape[0],
        action_map.shape[1],
    )

    def _reach_for_factor(factor: int):
        blocked_ds = downsample_or_factor(tm_blocked, factor)
        start_ds = downsample_or_factor(agent_mask, factor)
        passable_ds = jnp.logical_not(jnp.asarray(blocked_ds))
        reach_ds = TraversabilityMaskWrapper._build_reachability_mask(
            passable_ds,
            jnp.asarray(start_ds),
        ).astype(jnp.bool_)
        reach_up = upsample_nearest_factor(
            np.asarray(reach_ds, dtype=bool),
            factor,
            action_map.shape[0],
            action_map.shape[1],
        )
        return np.asarray(reach_ds), reach_up

    reach_ds3, reach_up3 = _reach_for_factor(3)
    reach_ds4, reach_up4 = _reach_for_factor(4)

    return (
        tm_blocked,
        np.asarray(reach_ds),
        np.asarray(reach_up, dtype=bool),
        reach_ds3,
        reach_up3,
        np.asarray(reach_ds4),
        reach_up4,
    )


def draw_case(fig, axes, scenario: str, title: str) -> None:
    if scenario == "real_foundation":
        target_map, action_map, static_obstacles, agent_mask = load_real_half_dug_foundation()
    else:
        action_map, static_obstacles, agent_mask = make_demo_map(scenario=scenario)
        target_map = np.zeros_like(action_map)

    blocked, reach_ds2, reach_up2, reach_ds3, reach_up3, reach_ds4, reach_up4 = compute_terra_ds2_reachability(
        action_map,
        static_obstacles,
        agent_mask,
    )

    map_classes = np.zeros_like(action_map, dtype=np.int8)
    remaining_target = (target_map < 0) & (action_map == 0)
    map_classes[remaining_target] = 1
    map_classes[action_map < 0] = 2
    map_classes[action_map > 0] = 3
    map_classes[static_obstacles] = 4
    map_classes[agent_mask] = 5

    reach_classes = np.zeros_like(action_map, dtype=np.int8)
    reach_classes[reach_up2] = 1
    reach_classes[blocked] = 2
    reach_classes[agent_mask] = 3

    reach4_classes = np.zeros_like(action_map, dtype=np.int8)
    reach4_classes[reach_up4] = 1
    reach4_classes[blocked] = 2
    reach4_classes[agent_mask] = 3

    reach3_classes = np.zeros_like(action_map, dtype=np.int8)
    reach3_classes[reach_up3] = 1
    reach3_classes[blocked] = 2
    reach3_classes[agent_mask] = 3

    axes[0].imshow(
        map_classes,
        interpolation="nearest",
        cmap=ListedColormap(["#f7f3e8", "#b9dcf1", "#357fb8", "#b98250", "#20242a", "#40b87a"]),
        vmin=0,
        vmax=5,
    )
    axes[0].set_title(f"{title}\nmap + obstacles + agent", fontsize=10)

    axes[1].imshow(
        reach_classes,
        interpolation="nearest",
        cmap=ListedColormap(["#d9d9d9", "#58b36b", "#20242a", "#2f6eea"]),
        vmin=0,
        vmax=3,
    )
    axes[1].set_title(
        f"Terra ds2 reachability\nreachable cells: {int(reach_ds2.sum())}/1024",
        fontsize=10,
    )

    axes[2].imshow(
        reach3_classes,
        interpolation="nearest",
        cmap=ListedColormap(["#d9d9d9", "#58b36b", "#20242a", "#2f6eea"]),
        vmin=0,
        vmax=3,
    )
    axes[2].set_title(
        f"ds3 reachability trial\nreachable cells: {int(reach_ds3.sum())}/{reach_ds3.size}",
        fontsize=10,
    )

    axes[3].imshow(
        reach4_classes,
        interpolation="nearest",
        cmap=ListedColormap(["#d9d9d9", "#58b36b", "#20242a", "#2f6eea"]),
        vmin=0,
        vmax=3,
    )
    axes[3].set_title(
        f"ds4 reachability trial\nreachable cells: {int(reach_ds4.sum())}/256",
        fontsize=10,
    )

    for ax in axes:
        ax.set_xticks(np.arange(-0.5, 64, 8), minor=True)
        ax.set_yticks(np.arange(-0.5, 64, 8), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.55)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 63.5)
        ax.set_ylim(63.5, -0.5)


def render(output_path: Path) -> None:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5.8),
        dpi=180,
        constrained_layout=True,
    )

    draw_case(
        fig,
        axes,
        "real_foundation",
        f"real foundation img_{DATASET_MAP_INDEX} half dug",
    )

    fig.legend(
        handles=[
            Patch(facecolor="#f7f3e8", label="empty/passable"),
            Patch(facecolor="#b9dcf1", label="remaining target"),
            Patch(facecolor="#357fb8", label="already dug < 0"),
            Patch(facecolor="#b98250", label="dumped > 0"),
            Patch(facecolor="#20242a", label="blocked obstacle/nonzero action"),
            Patch(facecolor="#40b87a", label="agent footprint"),
            Patch(facecolor="#58b36b", label="reachable after ds2 fill"),
            Patch(facecolor="#d9d9d9", label="unreachable"),
        ],
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Reachability examples: map vs Terra ds2 vs ds3/ds4 trials "
        "(OR-downsample -> 4-neighbor flood fill from agent footprint -> nearest upsample)",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render(Path("outputs/reachability_demo_ds2.png"))
