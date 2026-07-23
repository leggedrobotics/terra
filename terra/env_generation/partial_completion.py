"""Generate physically plausible partial-completion reset datasets for Terra."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_PILE_MODES = ("in_zone", "near_zone", "mixed")
ACTION_MAP_MIN = -1
ACTION_MAP_MAX = int(np.iinfo(np.int8).max)
FORMAT_VERSION = 1
MAP_SIZE = 64
MAP_EDGE_M = 36.5714285714
TILE_SIZE_M = MAP_EDGE_M / MAP_SIZE
EXCAVATOR_LONG_SIDE_M = 6.08
EXCAVATOR_SHORT_SIDE_M = 3.5
DIG_RADIUS_TILES = 5


def _odd_tile_span(length_m: float) -> int:
    rounded = round(length_m / TILE_SIZE_M)
    return rounded if rounded % 2 else rounded + 1


AGENT_WIDTH_TILES = _odd_tile_span(EXCAVATOR_SHORT_SIDE_M)
AGENT_HEIGHT_TILES = _odd_tile_span(EXCAVATOR_LONG_SIDE_M)
FOOTPRINT_RADIUS_TILES = math.ceil(
    math.hypot(AGENT_WIDTH_TILES / 2, AGENT_HEIGHT_TILES / 2)
)
SPAWN_BORDER_TILES = 8
SPAWN_MAX_CENTER_COORD = math.ceil(
    max(AGENT_WIDTH_TILES / 2 - 1, AGENT_HEIGHT_TILES / 2 - 1)
)
_WORKSPACE_AGENT_RADIUS_TILES = max(
    AGENT_WIDTH_TILES / 2,
    AGENT_HEIGHT_TILES / 2,
)
WORKSPACE_MIN_RADIUS_TILES = math.floor(
    0.5 / TILE_SIZE_M + _WORKSPACE_AGENT_RADIUS_TILES
)
WORKSPACE_MAX_RADIUS_TILES = math.ceil(
    0.5 / TILE_SIZE_M + _WORKSPACE_AGENT_RADIUS_TILES + DIG_RADIUS_TILES
)
WORKSPACE_HALF_ANGLE_RAD = math.pi / 6.0


class PartialCompletionError(RuntimeError):
    """Raised when a source map cannot produce a valid partial reset."""


@dataclass(frozen=True)
class PartialCompletionConfig:
    completion_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)
    variants_per_fraction: int = 1
    mode_weights: tuple[tuple[str, float], ...] = (("in_zone", 1.0),)
    min_piles: int = 1
    max_piles: int = 3
    min_center_separation: int = 4
    near_distance_min: int = 2
    near_distance_max: int = 8
    mixed_in_zone_fraction_min: float = 0.60
    mixed_in_zone_fraction_max: float = 0.90
    max_pile_height: int = 32
    max_workspace_load: int = ACTION_MAP_MAX
    min_spawn_centers: int = 16
    max_attempts_per_variant: int = 100
    include_full: bool = False
    limit: int | None = None
    seed: int = 0

    def validate(self) -> None:
        if not self.completion_fractions:
            raise ValueError("completion_fractions must not be empty.")
        if any(not 0.0 < value < 1.0 for value in self.completion_fractions):
            raise ValueError("Every completion fraction must lie strictly in (0, 1).")
        if self.variants_per_fraction <= 0:
            raise ValueError("variants_per_fraction must be positive.")

        names = tuple(name for name, _ in self.mode_weights)
        weights = tuple(float(weight) for _, weight in self.mode_weights)
        if not names or any(name not in SUPPORTED_PILE_MODES for name in names):
            raise ValueError(
                f"Mode names must be selected from {SUPPORTED_PILE_MODES}; got {names}."
            )
        if len(set(names)) != len(names):
            raise ValueError(f"Mode names must be unique; got {names}.")
        if any(weight < 0.0 for weight in weights) or not math.isclose(
            sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"Mode weights must be nonnegative and sum to one; got {weights}."
            )

        if self.min_piles <= 0 or self.max_piles < self.min_piles:
            raise ValueError("Pile-count bounds are invalid.")
        if dict(self.mode_weights).get("mixed", 0.0) > 0.0 and self.max_piles < 2:
            raise ValueError("Mixed mode requires max_piles to be at least two.")
        if self.min_center_separation < 0:
            raise ValueError("min_center_separation must be nonnegative.")
        if (
            self.near_distance_min < 0
            or self.near_distance_max < self.near_distance_min
        ):
            raise ValueError("Near-zone distance bounds are invalid.")
        if not (
            0.0
            < self.mixed_in_zone_fraction_min
            <= self.mixed_in_zone_fraction_max
            < 1.0
        ):
            raise ValueError("Mixed-mode in-zone fraction bounds are invalid.")
        if not 1 <= self.max_pile_height <= ACTION_MAP_MAX:
            raise ValueError(f"max_pile_height must lie in [1, {ACTION_MAP_MAX}].")
        if not 1 <= self.max_workspace_load <= ACTION_MAP_MAX:
            raise ValueError(f"max_workspace_load must lie in [1, {ACTION_MAP_MAX}].")
        if self.min_spawn_centers <= 0:
            raise ValueError("min_spawn_centers must be positive.")
        if self.max_attempts_per_variant <= 0:
            raise ValueError("max_attempts_per_variant must be positive.")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided.")


@dataclass(frozen=True)
class PartialCompletionResult:
    action_map: np.ndarray
    manifest: dict[str, Any]


def _ensure_2d(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 2:
        return value
    if value.ndim == 3 and 1 in (value.shape[0], value.shape[-1]):
        squeezed = np.squeeze(value)
        if squeezed.ndim == 2:
            return squeezed
    raise PartialCompletionError(
        f"{name} must be a two-dimensional grid; got {value.shape}."
    )


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    result = np.zeros_like(mask)
    for dx in range(-radius, radius + 1):
        source_x0 = max(0, dx)
        source_x1 = min(height, height + dx)
        dest_x0 = max(0, -dx)
        dest_x1 = min(height, height - dx)
        for dy in range(-radius, radius + 1):
            source_y0 = max(0, dy)
            source_y1 = min(width, width + dy)
            dest_y0 = max(0, -dy)
            dest_y1 = min(width, width - dy)
            result[dest_x0:dest_x1, dest_y0:dest_y1] |= mask[
                source_x0:source_x1,
                source_y0:source_y1,
            ]
    return result


def _binary_erode_square(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    result = np.ones_like(mask)
    for dx in range(-radius, radius + 1):
        shifted = np.zeros_like(mask)
        source_x0 = max(0, dx)
        source_x1 = min(height, height + dx)
        dest_x0 = max(0, -dx)
        dest_x1 = min(height, height - dx)
        for dy in range(-radius, radius + 1):
            shifted.fill(False)
            source_y0 = max(0, dy)
            source_y1 = min(width, width + dy)
            dest_y0 = max(0, -dy)
            dest_y1 = min(width, width - dy)
            shifted[dest_x0:dest_x1, dest_y0:dest_y1] = mask[
                source_x0:source_x1,
                source_y0:source_y1,
            ]
            result &= shifted
    return result


def _manhattan_distance_to(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise PartialCompletionError("The target map has no designated dump zone.")
    height, width = mask.shape
    distance = np.full((height, width), height + width + 1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for x, y in np.argwhere(mask):
        distance[x, y] = 0
        queue.append((int(x), int(y)))
    while queue:
        x, y = queue.popleft()
        next_distance = distance[x, y] + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (
                0 <= nx < height
                and 0 <= ny < width
                and next_distance < distance[nx, ny]
            ):
                distance[nx, ny] = next_distance
                queue.append((nx, ny))
    return distance


def _component_masks(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask)
    components: list[np.ndarray] = []
    if connectivity == 4:
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        neighbors = tuple(
            (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx != 0 or dy != 0
        )
    for start_x, start_y in np.argwhere(mask):
        start = (int(start_x), int(start_y))
        if seen[start]:
            continue
        component = np.zeros_like(mask)
        queue = deque([start])
        seen[start] = True
        component[start] = True
        while queue:
            x, y = queue.popleft()
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < height
                    and 0 <= ny < width
                    and mask[nx, ny]
                    and not seen[nx, ny]
                ):
                    seen[nx, ny] = True
                    component[nx, ny] = True
                    queue.append((nx, ny))
        components.append(component)
    return components


def _has_no_singleton_components(mask: np.ndarray) -> bool:
    return all(
        int(np.count_nonzero(component)) >= 2
        for component in _component_masks(mask, connectivity=8)
    )


def _repair_singleton_residuals(
    selected: np.ndarray,
    dig_target: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Return completed tiles to singleton residuals, then regrow exactly."""
    repaired = selected.copy()

    # Returning a neighboring completed tile grows or joins every singleton
    # residual. Restore the requested count only after all singletons are gone,
    # so each regrowth choice can preserve that invariant.
    for _ in range(int(np.count_nonzero(dig_target))):
        remaining = dig_target & ~repaired
        singletons = [
            component
            for component in _component_masks(remaining, connectivity=8)
            if int(np.count_nonzero(component)) == 1
        ]
        if not singletons:
            break
        singleton = singletons[int(rng.integers(0, len(singletons)))]
        neighbors = (
            _binary_dilate(singleton, radius=1)
            & repaired
            & np.asarray(dig_target, dtype=bool)
        )
        candidates = np.argwhere(neighbors)
        if not len(candidates):
            return None
        candidate = candidates[int(rng.integers(0, len(candidates)))]
        repaired[int(candidate[0]), int(candidate[1])] = False
    else:
        return None

    while int(np.count_nonzero(repaired)) < count:
        remaining = dig_target & ~repaired
        frontier = remaining & _binary_dilate(repaired, radius=1)
        candidates = np.argwhere(frontier)
        if not len(candidates):
            candidates = np.argwhere(remaining)
        accepted = False
        for candidate_index in rng.permutation(len(candidates)):
            candidate = candidates[int(candidate_index)]
            x, y = int(candidate[0]), int(candidate[1])
            repaired[x, y] = True
            if _has_no_singleton_components(dig_target & ~repaired):
                accepted = True
                break
            repaired[x, y] = False
        if not accepted:
            return None

    if int(np.count_nonzero(repaired)) != count:
        return None
    if not _has_no_singleton_components(dig_target & ~repaired):
        return None
    return repaired


@lru_cache(maxsize=1)
def _conservative_workspace_offsets() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Oversized 12-heading cones used only for the int8 load bound."""
    result: list[tuple[tuple[int, int], ...]] = []
    radius = math.ceil(WORKSPACE_MAX_RADIUS_TILES)
    for heading_index in range(12):
        heading = 2.0 * math.pi * heading_index / 12
        cos_heading = math.cos(heading)
        sin_heading = math.sin(heading)
        offsets: list[tuple[int, int]] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                local_x = cos_heading * dx + sin_heading * dy
                local_y = -sin_heading * dx + cos_heading * dy
                distance = math.hypot(local_x, local_y)
                angle = math.atan2(-local_x, local_y)
                if (
                    WORKSPACE_MIN_RADIUS_TILES - 1e-6
                    <= distance
                    <= WORKSPACE_MAX_RADIUS_TILES + 1e-6
                    and abs(angle) <= WORKSPACE_HALF_ANGLE_RAD + 1e-6
                ):
                    offsets.append((dx, dy))
        result.append(tuple(offsets))
    return tuple(result)


def compute_dynamic_dumpability_numpy(
    static_dumpability: np.ndarray,
    action_map: np.ndarray,
) -> np.ndarray:
    static_dumpability = _ensure_2d(static_dumpability, "static_dumpability").astype(
        bool
    )
    action_map = _ensure_2d(action_map, "action_map")
    return static_dumpability & ~_binary_dilate(action_map < 0, radius=2)


def _runtime_sampling_domain(
    occupancy: np.ndarray,
) -> np.ndarray:
    occupancy = np.asarray(occupancy, dtype=bool)
    height, width = occupancy.shape
    max_traversable_x = int(np.count_nonzero(~occupancy[:, 0]))
    max_traversable_y = int(np.count_nonzero(~occupancy[0, :]))
    minimum_sampling_range = 2 * SPAWN_MAX_CENTER_COORD + 1
    max_w = max(min(max_traversable_x, height), minimum_sampling_range)
    max_h = max(min(max_traversable_y, width), minimum_sampling_range)

    domain = np.zeros_like(occupancy)
    x_min = SPAWN_MAX_CENTER_COORD
    x_max_exclusive = max_w - SPAWN_MAX_CENTER_COORD
    y_min = SPAWN_MAX_CENTER_COORD
    y_max_exclusive = max_h - SPAWN_MAX_CENTER_COORD
    if x_max_exclusive <= x_min or y_max_exclusive <= y_min:
        return domain
    domain[
        x_min : min(x_max_exclusive, height),
        y_min : min(y_max_exclusive, width),
    ] = True

    coordinates_x = np.arange(height)[:, None]
    coordinates_y = np.arange(width)[None, :]
    border_distance = np.minimum.reduce(
        (
            np.broadcast_to(coordinates_x, (height, width)),
            np.broadcast_to(height - 1 - coordinates_x, (height, width)),
            np.broadcast_to(coordinates_y, (height, width)),
            np.broadcast_to(width - 1 - coordinates_y, (height, width)),
        )
    )
    domain &= border_distance >= SPAWN_BORDER_TILES
    return domain


def _spawn_center_mask(
    occupancy: np.ndarray,
    dynamic_dumpability: np.ndarray,
    action_map: np.ndarray,
) -> np.ndarray:
    footprint_clear = (
        ~np.asarray(occupancy, dtype=bool)
        & np.asarray(dynamic_dumpability, dtype=bool)
        & (np.asarray(action_map) == 0)
    )
    conservative_centers = _binary_erode_square(
        footprint_clear,
        FOOTPRINT_RADIUS_TILES,
    )
    return conservative_centers & _runtime_sampling_domain(occupancy)


def _shifted_sum(
    values: np.ndarray, offsets: tuple[tuple[int, int], ...]
) -> np.ndarray:
    values = np.asarray(values)
    height, width = values.shape
    result = np.zeros((height, width), dtype=np.int32)
    for dx, dy in offsets:
        center_x0 = max(0, -dx)
        center_x1 = min(height, height - dx)
        center_y0 = max(0, -dy)
        center_y1 = min(width, width - dy)
        if center_x0 >= center_x1 or center_y0 >= center_y1:
            continue
        result[center_x0:center_x1, center_y0:center_y1] += values[
            center_x0 + dx : center_x1 + dx,
            center_y0 + dy : center_y1 + dy,
        ].astype(np.int32)
    return result


def _maximum_workspace_load(
    positive_heights: np.ndarray,
    offsets_collection: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[int, tuple[int, int] | None]:
    maximum = 0
    maximum_position: tuple[int, int] | None = None
    for offsets in offsets_collection:
        volume = _shifted_sum(positive_heights, offsets)
        local_maximum = int(volume.max(initial=0))
        if local_maximum > maximum:
            maximum = local_maximum
            position = np.unravel_index(int(np.argmax(volume)), volume.shape)
            maximum_position = (int(position[0]), int(position[1]))
    return maximum, maximum_position


def _select_completed_mask(
    dig_target: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coordinates = np.argwhere(dig_target)
    if count <= 0 or count >= len(coordinates):
        raise PartialCompletionError(
            "Completed excavation count is outside the partial range."
        )
    components = _component_masks(dig_target, connectivity=8)
    for _ in range(32):
        distance = np.full(dig_target.shape, np.iinfo(np.int32).max, dtype=np.int32)
        queue: deque[tuple[int, int]] = deque()
        for component in components:
            component_boundary = np.argwhere(
                component & ~_binary_erode_square(component, radius=1)
            )
            if not len(component_boundary):
                component_boundary = np.argwhere(component)
            x, y = component_boundary[int(rng.integers(0, len(component_boundary)))]
            position = (int(x), int(y))
            distance[position] = 0
            queue.append(position)
        while queue:
            x, y = queue.popleft()
            next_distance = distance[x, y] + 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if (
                        (dx != 0 or dy != 0)
                        and 0 <= nx < dig_target.shape[0]
                        and 0 <= ny < dig_target.shape[1]
                        and dig_target[nx, ny]
                        and next_distance < distance[nx, ny]
                    ):
                        distance[nx, ny] = next_distance
                        queue.append((nx, ny))

        target_distances = distance[dig_target]
        order = np.lexsort((rng.random(len(coordinates)), target_distances))
        selected = np.zeros_like(dig_target, dtype=bool)
        chosen = coordinates[order[:count]]
        selected[chosen[:, 0], chosen[:, 1]] = True

        repaired = _repair_singleton_residuals(
            selected,
            dig_target,
            count,
            rng,
        )
        if repaired is not None:
            return repaired
    raise PartialCompletionError(
        "Could not grow coherent completed patches without a singleton residual component."
    )


def _choose_centers(
    center_mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
    minimum_separation: int,
) -> list[tuple[int, int]]:
    candidates = np.argwhere(center_mask)
    if len(candidates) < count:
        raise PartialCompletionError(
            f"Need {count} pile centers but only {len(candidates)} candidates are available."
        )
    for _ in range(64):
        order = rng.permutation(len(candidates))
        centers: list[tuple[int, int]] = []
        for index in order:
            candidate = tuple(int(value) for value in candidates[int(index)])
            if all(
                abs(candidate[0] - existing[0]) + abs(candidate[1] - existing[1])
                >= minimum_separation
                for existing in centers
            ):
                centers.append(candidate)
                if len(centers) == count:
                    return centers
    raise PartialCompletionError(
        "Could not choose sufficiently separated pile centers."
    )


def _integer_split(
    total: int,
    parts: int,
    rng: np.random.Generator,
) -> list[int]:
    if parts <= 0 or total < parts:
        raise PartialCompletionError(
            f"Cannot split volume {total} across {parts} piles."
        )
    allocation = np.ones(parts, dtype=np.int32)
    remaining = total - parts
    if remaining:
        allocation += rng.multinomial(remaining, np.full(parts, 1.0 / parts))
    return [int(value) for value in allocation]


def _allowed_component(
    allowed: np.ndarray,
    center: tuple[int, int],
) -> np.ndarray:
    for component in _component_masks(allowed, connectivity=4):
        if component[center]:
            return component
    raise PartialCompletionError(f"Pile center {center} is outside valid support.")


def _legal_height_increment_mask(
    heights: np.ndarray,
    allowed: np.ndarray,
    max_pile_height: int,
) -> np.ndarray:
    """Return cells whose increment preserves a stable one-step slope."""
    padded = np.pad(
        heights,
        pad_width=1,
        mode="constant",
        constant_values=0,
    )
    minimum_neighbor = np.minimum.reduce(
        (
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
        )
    )
    return (
        np.asarray(allowed, dtype=bool)
        & (heights < max_pile_height)
        & (heights <= minimum_neighbor)
    )


def _deposit_mode_volume(
    volume: int,
    allowed: np.ndarray,
    center_mask: np.ndarray,
    rng: np.random.Generator,
    config: PartialCompletionConfig,
    *,
    pile_count: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    if volume <= 0:
        raise PartialCompletionError("Pile volume must be positive.")
    if pile_count is None:
        maximum_piles = min(config.max_piles, volume)
        minimum_piles = min(config.min_piles, maximum_piles)
        pile_count = int(rng.integers(minimum_piles, maximum_piles + 1))
    elif not 1 <= pile_count <= volume:
        raise PartialCompletionError(
            f"Cannot allocate {volume} units across {pile_count} piles."
        )
    centers = _choose_centers(
        center_mask,
        pile_count,
        rng,
        config.min_center_separation,
    )
    volumes = _integer_split(volume, pile_count, rng)
    heights = np.zeros_like(allowed, dtype=np.int32)
    for center, pile_volume in zip(centers, volumes):
        component = _allowed_component(allowed, center)
        coordinates_x = np.arange(heights.shape[0])[:, None]
        coordinates_y = np.arange(heights.shape[1])[None, :]
        center_cost = (coordinates_x - center[0]) ** 2 + (
            coordinates_y - center[1]
        ) ** 2
        for _ in range(pile_volume):
            legal = _legal_height_increment_mask(
                heights,
                component,
                config.max_pile_height,
            )
            candidates = np.argwhere(legal)
            if not len(candidates):
                raise PartialCompletionError(
                    f"Pile centered at {center} cannot fit {pile_volume} soil units "
                    "under its support, height, and slope constraints."
                )
            candidate_costs = center_cost[legal]
            minimum_cost = int(candidate_costs.min())
            nearest = candidates[candidate_costs == minimum_cost]
            chosen = nearest[int(rng.integers(0, len(nearest)))]
            heights[int(chosen[0]), int(chosen[1])] += 1
    if int(heights.sum()) != volume:
        raise PartialCompletionError("Pile construction did not conserve volume.")
    return heights, centers, volumes


def _construct_piles(
    target_map: np.ndarray,
    occupancy: np.ndarray,
    dynamic_dumpability: np.ndarray,
    volume: int,
    mode: str,
    rng: np.random.Generator,
    config: PartialCompletionConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dump_zone = target_map > 0
    dump_buffer = _binary_dilate(dump_zone, radius=1)
    distance = _manhattan_distance_to(dump_zone)
    physically_valid = (
        dynamic_dumpability.astype(bool) & ~occupancy.astype(bool) & ~(target_map < 0)
    )

    in_allowed = physically_valid & dump_buffer
    in_centers = (
        physically_valid & dump_zone & _binary_erode_square(in_allowed, radius=1)
    )
    near_band = (
        (distance >= config.near_distance_min)
        & (distance <= config.near_distance_max)
        & ~dump_buffer
        & (target_map == 0)
    )
    near_allowed = physically_valid & near_band
    near_centers = near_allowed & _binary_erode_square(near_allowed, radius=1)
    pile_records: list[dict[str, Any]] = []

    if mode == "in_zone":
        heights, centers, volumes = _deposit_mode_volume(
            volume, in_allowed, in_centers, rng, config
        )
        pile_records.extend(
            {"mode": "in_zone", "center": list(center), "volume": pile_volume}
            for center, pile_volume in zip(centers, volumes)
        )
        return heights, pile_records

    if mode == "near_zone":
        heights, centers, volumes = _deposit_mode_volume(
            volume, near_allowed, near_centers, rng, config
        )
        pile_records.extend(
            {"mode": "near_zone", "center": list(center), "volume": pile_volume}
            for center, pile_volume in zip(centers, volumes)
        )
        return heights, pile_records

    if mode != "mixed":
        raise PartialCompletionError(f"Unsupported pile mode {mode!r}.")
    if volume < 2:
        raise PartialCompletionError("Mixed mode requires at least two soil units.")
    minimum_in = max(1, math.ceil(volume * config.mixed_in_zone_fraction_min))
    maximum_in = min(volume - 1, math.floor(volume * config.mixed_in_zone_fraction_max))
    if minimum_in > maximum_in:
        raise PartialCompletionError(
            "Mixed-mode volume cannot satisfy both partitions."
        )
    in_volume = int(rng.integers(minimum_in, maximum_in + 1))
    near_volume = volume - in_volume
    minimum_total_piles = max(2, config.min_piles)
    maximum_total_piles = min(config.max_piles, volume)
    if minimum_total_piles > maximum_total_piles:
        raise PartialCompletionError(
            "Mixed mode cannot satisfy the configured total pile-count bounds."
        )
    total_pile_count = int(rng.integers(minimum_total_piles, maximum_total_piles + 1))
    minimum_in_piles = max(1, total_pile_count - near_volume)
    maximum_in_piles = min(in_volume, total_pile_count - 1)
    if minimum_in_piles > maximum_in_piles:
        raise PartialCompletionError(
            "Mixed-mode partition volumes cannot support the selected pile count."
        )
    in_pile_count = int(rng.integers(minimum_in_piles, maximum_in_piles + 1))
    near_pile_count = total_pile_count - in_pile_count
    in_heights, in_centers_values, in_volumes = _deposit_mode_volume(
        in_volume,
        in_allowed,
        in_centers,
        rng,
        config,
        pile_count=in_pile_count,
    )
    near_heights, near_centers_values, near_volumes = _deposit_mode_volume(
        near_volume,
        near_allowed,
        near_centers,
        rng,
        config,
        pile_count=near_pile_count,
    )
    if np.any((in_heights > 0) & (near_heights > 0)):
        raise PartialCompletionError("Mixed-mode pile partitions overlapped.")
    pile_records.extend(
        {"mode": "in_zone", "center": list(center), "volume": pile_volume}
        for center, pile_volume in zip(in_centers_values, in_volumes)
    )
    pile_records.extend(
        {"mode": "near_zone", "center": list(center), "volume": pile_volume}
        for center, pile_volume in zip(near_centers_values, near_volumes)
    )
    return in_heights + near_heights, pile_records


def _validate_slopes(
    pile_heights: np.ndarray,
    physically_valid: np.ndarray,
) -> None:
    for axis in (0, 1):
        left_slice = [slice(None), slice(None)]
        right_slice = [slice(None), slice(None)]
        left_slice[axis] = slice(0, -1)
        right_slice[axis] = slice(1, None)
        left_slice_tuple = tuple(left_slice)
        right_slice_tuple = tuple(right_slice)
        valid_pair = (
            physically_valid[left_slice_tuple] & physically_valid[right_slice_tuple]
        )
        differences = np.abs(
            pile_heights[left_slice_tuple].astype(np.int32)
            - pile_heights[right_slice_tuple].astype(np.int32)
        )
        if np.any(valid_pair & (differences > 1)):
            raise PartialCompletionError(
                "Pile field contains a four-neighbor height difference greater than one."
            )


def _validate_access(
    target_map: np.ndarray,
    occupancy: np.ndarray,
    action_map: np.ndarray,
    spawn_centers: np.ndarray,
) -> dict[str, Any]:
    center_free = _binary_erode_square(
        ~occupancy.astype(bool) & (action_map == 0),
        FOOTPRINT_RADIUS_TILES,
    )
    base_components = _component_masks(center_free, connectivity=4)
    spawn_components = [
        component for component in base_components if np.any(component & spawn_centers)
    ]
    if not spawn_components:
        raise PartialCompletionError(
            "No footprint-aware base component contains a spawn center."
        )

    positive = action_map > 0
    remaining = (target_map < 0) & (action_map >= 0)
    dump_buffer = _binary_dilate(target_map > 0, radius=1)
    staged = positive & ~dump_buffer

    work_regions: list[tuple[str, np.ndarray]] = []
    relevant_regions = [
        (f"dig_{index}", component)
        for index, component in enumerate(_component_masks(remaining, connectivity=8))
    ]
    staged_components = _component_masks(staged, connectivity=8)
    for index, component in enumerate(staged_components):
        if int(np.count_nonzero(component)) < 2:
            raise PartialCompletionError(
                f"Staged pile component {index} has fewer than two support tiles."
            )
        relevant_regions.append((f"lift_{index}", component))
    relevant_regions.append(("dump", target_map > 0))

    for name, relevant in relevant_regions:
        interaction_region = (
            center_free
            & _binary_dilate(relevant, radius=int(WORKSPACE_MAX_RADIUS_TILES))
            & ~_binary_dilate(
                relevant,
                radius=max(0, int(WORKSPACE_MIN_RADIUS_TILES) - 1),
            )
        )
        if not np.any(interaction_region):
            raise PartialCompletionError(
                f"{name} has no footprint-clear nearby interaction region."
            )
        work_regions.append((name, interaction_region))

    for component_index, component in enumerate(spawn_components):
        for region_name, region_mask in work_regions:
            if not np.any(component & region_mask):
                raise PartialCompletionError(
                    f"Spawn-bearing component {component_index} cannot access {region_name}."
                )

    return {
        "spawn_component_count": len(spawn_components),
        "spawn_component_sizes": [
            int(np.count_nonzero(component)) for component in spawn_components
        ],
        "interaction_region_counts": {
            name: int(np.count_nonzero(region)) for name, region in work_regions
        },
    }


def validate_partial_state(
    target_map: np.ndarray,
    occupancy: np.ndarray,
    static_dumpability: np.ndarray,
    action_map: np.ndarray,
    *,
    config: PartialCompletionConfig,
    expected_mode: str,
) -> dict[str, Any]:
    """Validate one generated partial state and return auditable diagnostics."""
    config.validate()
    if expected_mode not in SUPPORTED_PILE_MODES:
        raise PartialCompletionError(f"Unknown expected mode {expected_mode!r}.")

    target_map = _ensure_2d(target_map, "target_map")
    occupancy = _ensure_2d(occupancy, "occupancy")
    static_dumpability = _ensure_2d(static_dumpability, "static_dumpability")
    action_map = _ensure_2d(action_map, "action_map")
    if not (
        target_map.shape
        == occupancy.shape
        == static_dumpability.shape
        == action_map.shape
    ):
        raise PartialCompletionError(
            "All partial-state layers must have identical shapes."
        )
    if target_map.shape != (MAP_SIZE, MAP_SIZE):
        raise PartialCompletionError(
            f"Partial resets support only {(MAP_SIZE, MAP_SIZE)} maps; "
            f"got {target_map.shape}."
        )
    if not np.all(np.isin(target_map, (-1, 0, 1))):
        raise PartialCompletionError("Target map must remain categorical -1/0/+1.")
    if not np.all(np.isin(occupancy, (0, 1))):
        raise PartialCompletionError("Occupancy must remain binary.")
    if not np.all(np.isin(static_dumpability, (0, 1))):
        raise PartialCompletionError("Static dumpability must remain binary.")
    if not np.issubdtype(action_map.dtype, np.integer):
        raise PartialCompletionError(
            f"Action map must be integer, got {action_map.dtype}."
        )
    minimum = int(action_map.min())
    maximum = int(action_map.max())
    if minimum < ACTION_MAP_MIN or maximum > ACTION_MAP_MAX:
        raise PartialCompletionError(
            f"Action-map range [{minimum}, {maximum}] exceeds [{ACTION_MAP_MIN}, {ACTION_MAP_MAX}]."
        )

    negative = action_map < 0
    positive = action_map > 0
    if not np.any(negative) or not np.any(positive):
        raise PartialCompletionError(
            "A partial state must contain completed holes and soil."
        )
    if np.any(negative & ~(target_map < 0)):
        raise PartialCompletionError(
            "Negative action tiles must be target excavation tiles."
        )
    if np.any(positive & (target_map < 0)):
        raise PartialCompletionError("Positive soil may not overlap target excavation.")
    if np.any(positive & occupancy.astype(bool)):
        raise PartialCompletionError("Positive soil may not overlap static obstacles.")

    dynamic_dumpability = compute_dynamic_dumpability_numpy(
        static_dumpability,
        action_map,
    )
    if np.any(positive & ~dynamic_dumpability):
        raise PartialCompletionError(
            "Positive soil lies outside initial dynamic dumpability."
        )

    negative_volume = -int(action_map[negative].astype(np.int64).sum())
    positive_volume = int(action_map[positive].astype(np.int64).sum())
    if positive_volume != negative_volume:
        raise PartialCompletionError(
            f"Mass is not conserved: removed={negative_volume}, piled={positive_volume}."
        )

    remaining = (target_map < 0) & (action_map >= 0)
    remaining_components = _component_masks(remaining, connectivity=8)
    if not remaining_components:
        raise PartialCompletionError("Generated reset is already fully excavated.")
    remaining_component_sizes = [
        int(np.count_nonzero(component)) for component in remaining_components
    ]
    if any(size < 2 for size in remaining_component_sizes):
        raise PartialCompletionError(
            f"Remaining excavation has singleton components: {remaining_component_sizes}."
        )

    dump_zone = target_map > 0
    dump_buffer = _binary_dilate(dump_zone, radius=1)
    distance = _manhattan_distance_to(dump_zone)
    in_zone_positive = positive & dump_buffer
    near_positive = positive & ~dump_buffer
    if expected_mode == "in_zone" and np.any(near_positive):
        raise PartialCompletionError(
            "in_zone mode placed soil outside the terminal buffer."
        )
    if expected_mode == "near_zone" and np.any(in_zone_positive):
        raise PartialCompletionError(
            "near_zone mode placed soil inside the terminal buffer."
        )
    if expected_mode == "mixed" and (
        not np.any(in_zone_positive) or not np.any(near_positive)
    ):
        raise PartialCompletionError("mixed mode must retain both soil partitions.")
    if expected_mode in ("near_zone", "mixed") and np.any(
        near_positive
        & (
            (distance < config.near_distance_min)
            | (distance > config.near_distance_max)
        )
    ):
        raise PartialCompletionError(
            "Near-zone support left its Manhattan-distance band."
        )

    physically_valid = dynamic_dumpability & ~occupancy.astype(bool) & ~(target_map < 0)
    pile_heights = np.where(positive, action_map, 0).astype(np.int32)
    _validate_slopes(pile_heights, physically_valid)
    if int(pile_heights.max(initial=0)) > config.max_pile_height:
        raise PartialCompletionError(
            "Generated pile exceeds configured maximum height."
        )

    # Soil already in the final zone is complete and need not be lifted again.
    # Bound only staged soil that the policy still has to relocate.
    staged_pile_heights = np.where(near_positive, action_map, 0).astype(np.int32)
    maximum_staged_workspace_load, maximum_staged_workspace_position = (
        _maximum_workspace_load(
            staged_pile_heights,
            _conservative_workspace_offsets(),
        )
    )
    if maximum_staged_workspace_load > config.max_workspace_load:
        raise PartialCompletionError(
            "A possible excavator workspace over staged soil contains "
            f"{maximum_staged_workspace_load} soil units, exceeding "
            f"{config.max_workspace_load}."
        )

    spawn_centers = _spawn_center_mask(
        occupancy,
        dynamic_dumpability,
        action_map,
    )
    valid_spawn_count = int(np.count_nonzero(spawn_centers))
    if valid_spawn_count < config.min_spawn_centers:
        raise PartialCompletionError(
            f"Only {valid_spawn_count} conservative runtime-sampleable spawn centers remain; "
            f"need {config.min_spawn_centers}."
        )

    access = _validate_access(
        target_map,
        occupancy.astype(bool),
        action_map,
        spawn_centers,
    )
    positive_distances = distance[positive]
    return {
        "negative_volume": negative_volume,
        "positive_volume": positive_volume,
        "remaining_component_sizes": remaining_component_sizes,
        "maximum_pile_height": int(pile_heights.max(initial=0)),
        "positive_support_area": int(np.count_nonzero(positive)),
        "minimum_positive_dump_distance": int(positive_distances.min()),
        "maximum_positive_dump_distance": int(positive_distances.max()),
        "valid_spawn_center_count": valid_spawn_count,
        "maximum_staged_workspace_load": int(maximum_staged_workspace_load),
        "maximum_staged_workspace_position": (
            list(maximum_staged_workspace_position)
            if maximum_staged_workspace_position is not None
            else None
        ),
        **access,
    }


def _choose_mode(
    rng: np.random.Generator,
    mode_weights: tuple[tuple[str, float], ...],
) -> str:
    names = [name for name, _ in mode_weights]
    probabilities = np.asarray([weight for _, weight in mode_weights], dtype=np.float64)
    return str(rng.choice(names, p=probabilities))


def generate_partial_action_map(
    target_map: np.ndarray,
    occupancy: np.ndarray,
    static_dumpability: np.ndarray,
    *,
    rng: np.random.Generator,
    config: PartialCompletionConfig,
) -> PartialCompletionResult:
    """Generate and validate one partial action map."""
    config.validate()
    target_map = _ensure_2d(target_map, "target_map")
    occupancy = _ensure_2d(occupancy, "occupancy")
    static_dumpability = _ensure_2d(static_dumpability, "static_dumpability")
    if not target_map.shape == occupancy.shape == static_dumpability.shape:
        raise PartialCompletionError("Source map layers must have identical shapes.")
    if target_map.shape != (MAP_SIZE, MAP_SIZE):
        raise PartialCompletionError(
            f"Partial resets support only {(MAP_SIZE, MAP_SIZE)} maps; "
            f"got {target_map.shape}."
        )
    dig_target = target_map < 0
    original_dig_count = int(np.count_nonzero(dig_target))
    if original_dig_count < 3:
        raise PartialCompletionError(
            "Partial generation requires at least three dig tiles."
        )
    completion_fraction = float(config.completion_fractions[0])
    completed_count = int(round(completion_fraction * original_dig_count))
    completed_count = min(max(completed_count, 1), original_dig_count - 2)
    mode = _choose_mode(rng, config.mode_weights)

    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts_per_variant + 1):
        try:
            completed = _select_completed_mask(
                dig_target,
                completed_count,
                rng,
            )
            action_map = np.zeros_like(target_map, dtype=np.int32)
            action_map[completed] = -1
            dynamic_dumpability = compute_dynamic_dumpability_numpy(
                static_dumpability,
                action_map,
            )
            pile_heights, piles = _construct_piles(
                target_map,
                occupancy.astype(bool),
                dynamic_dumpability,
                completed_count,
                mode,
                rng,
                config,
            )
            action_map += pile_heights
            action_map_int8 = action_map.astype(np.int8)
            diagnostics = validate_partial_state(
                target_map,
                occupancy,
                static_dumpability,
                action_map_int8,
                config=config,
                expected_mode=mode,
            )
            manifest = {
                "format_version": FORMAT_VERSION,
                "requested_completion_fraction": completion_fraction,
                "achieved_completion_fraction": completed_count / original_dig_count,
                "original_dig_tile_count": original_dig_count,
                "completed_dig_tile_count": completed_count,
                "remaining_dig_tile_count": original_dig_count - completed_count,
                "pile_mode": mode,
                "pile_count": len(piles),
                "piles": piles,
                "generation_attempt": attempt,
                "rejected_candidate_count": attempt - 1,
                **diagnostics,
            }
            return PartialCompletionResult(
                action_map=action_map_int8, manifest=manifest
            )
        except (PartialCompletionError, ValueError) as error:
            last_error = error
    raise PartialCompletionError(
        f"Failed to generate a valid {mode} partial state after "
        f"{config.max_attempts_per_variant} attempts: {last_error}"
    )


def _source_indices(input_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in (input_dir / "images").glob("img_*.npy"):
        try:
            indices.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(indices))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_source_layers(
    input_dir: Path,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        "target": input_dir / "images" / f"img_{index}.npy",
        "occupancy": input_dir / "occupancy" / f"img_{index}.npy",
        "dumpability": input_dir / "dumpability" / f"img_{index}.npy",
        "distance": input_dir / "distance" / f"img_{index}.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise PartialCompletionError(f"Source map {index} is missing files: {missing}.")
    target_map = _ensure_2d(np.load(paths["target"]), "target_map")
    occupancy = _ensure_2d(np.load(paths["occupancy"]), "occupancy")
    dumpability = _ensure_2d(np.load(paths["dumpability"]), "dumpability")
    distance = _ensure_2d(np.load(paths["distance"]), "distance")
    if not target_map.shape == occupancy.shape == dumpability.shape == distance.shape:
        raise PartialCompletionError(f"Source map {index} has mismatched layer shapes.")

    actions_path = input_dir / "actions" / f"img_{index}.npy"
    if actions_path.exists():
        source_actions = _ensure_2d(np.load(actions_path), "source_actions")
        if np.any(source_actions != 0):
            raise PartialCompletionError(
                f"Source map {index} already has a nonzero action map."
            )
    return target_map, occupancy, dumpability


def _copy_layer(
    input_dir: Path,
    temporary_output: Path,
    folder: str,
    source_index: int,
    output_index: int,
) -> None:
    source = input_dir / folder / f"img_{source_index}.npy"
    destination = temporary_output / folder / f"img_{output_index}.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_metadata(
    input_dir: Path,
    temporary_output: Path,
    source_index: int,
    output_index: int,
) -> None:
    source = input_dir / "metadata" / f"trench_{source_index}.json"
    if not source.exists():
        return
    destination = temporary_output / "metadata" / f"trench_{output_index}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def generate_partial_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    config: PartialCompletionConfig,
) -> None:
    """Generate an atomic Terra-format partial-reset dataset."""
    config.validate()
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise PartialCompletionError(f"Input dataset does not exist: {input_dir}.")
    if output_dir.exists():
        raise PartialCompletionError(f"Output path already exists: {output_dir}.")
    indices = _source_indices(input_dir)
    if config.limit is not None:
        indices = indices[: config.limit]
    if not indices:
        raise PartialCompletionError(
            f"No source maps found under {input_dir / 'images'}."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    manifest_records: list[dict[str, Any]] = []
    output_index = 0
    rejected_total = 0
    generation_summary: dict[str, dict[str, Any]] = {}
    try:
        for source_index in indices:
            target_map, occupancy, dumpability = _load_source_layers(
                input_dir,
                source_index,
            )
            if config.include_full:
                output_index += 1
                for folder in ("images", "occupancy", "dumpability", "distance"):
                    _copy_layer(
                        input_dir,
                        temporary_output,
                        folder,
                        source_index,
                        output_index,
                    )
                _copy_metadata(input_dir, temporary_output, source_index, output_index)
                actions_dir = temporary_output / "actions"
                actions_dir.mkdir(parents=True, exist_ok=True)
                full_actions = np.zeros_like(target_map, dtype=np.int8)
                np.save(actions_dir / f"img_{output_index}.npy", full_actions)
                manifest_records.append(
                    {
                        "format_version": FORMAT_VERSION,
                        "output_index": output_index,
                        "source_index": source_index,
                        "variant_index": -1,
                        "global_seed": config.seed,
                        "variant_seed": None,
                        "pile_mode": "full",
                        "requested_completion_fraction": 0.0,
                        "achieved_completion_fraction": 0.0,
                    }
                )

            for fraction_index, fraction in enumerate(config.completion_fractions):
                for variant_index in range(config.variants_per_fraction):
                    seed_sequence = np.random.SeedSequence(
                        [config.seed, source_index, fraction_index, variant_index]
                    )
                    variant_seed = int(
                        seed_sequence.generate_state(1, dtype=np.uint64)[0]
                    )
                    rng = np.random.default_rng(variant_seed)
                    variant_config = PartialCompletionConfig(
                        **{
                            **asdict(config),
                            "completion_fractions": (float(fraction),),
                            "mode_weights": tuple(config.mode_weights),
                        }
                    )
                    try:
                        result = generate_partial_action_map(
                            target_map,
                            occupancy,
                            dumpability,
                            rng=rng,
                            config=variant_config,
                        )
                    except PartialCompletionError as error:
                        raise PartialCompletionError(
                            f"Source map {source_index}, completion fraction "
                            f"{fraction}, variant {variant_index}: {error}"
                        ) from error
                    rejected_total += int(result.manifest["rejected_candidate_count"])
                    summary_key = f"fraction={float(fraction):g},mode={result.manifest['pile_mode']}"
                    summary_entry = generation_summary.setdefault(
                        summary_key,
                        {
                            "completion_fraction": float(fraction),
                            "pile_mode": result.manifest["pile_mode"],
                            "generated_count": 0,
                            "rejected_candidate_count": 0,
                        },
                    )
                    summary_entry["generated_count"] += 1
                    summary_entry["rejected_candidate_count"] += int(
                        result.manifest["rejected_candidate_count"]
                    )
                    output_index += 1
                    for folder in ("images", "occupancy", "dumpability", "distance"):
                        _copy_layer(
                            input_dir,
                            temporary_output,
                            folder,
                            source_index,
                            output_index,
                        )
                    _copy_metadata(
                        input_dir, temporary_output, source_index, output_index
                    )
                    actions_dir = temporary_output / "actions"
                    actions_dir.mkdir(parents=True, exist_ok=True)
                    np.save(actions_dir / f"img_{output_index}.npy", result.action_map)
                    manifest_records.append(
                        {
                            "output_index": output_index,
                            "source_index": source_index,
                            "variant_index": variant_index,
                            "completion_fraction_index": fraction_index,
                            "global_seed": config.seed,
                            "variant_seed": variant_seed,
                            **result.manifest,
                        }
                    )

        config_payload = {
            "format_version": FORMAT_VERSION,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "source_map_count": len(indices),
            "output_map_count": output_index,
            "rejected_candidate_count": rejected_total,
            "generation_summary": generation_summary,
            "config": _jsonable(asdict(config)),
        }
        with (temporary_output / "partial_completion_config.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(config_payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        with (temporary_output / "partial_completion_manifest.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for record in manifest_records:
                stream.write(json.dumps(_jsonable(record), sort_keys=True))
                stream.write("\n")

        temporary_output.rename(output_dir)
    except Exception:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        raise

    print(
        "Generated "
        f"{output_index} partial-reset dataset entries from {len(indices)} source maps "
        f"at {output_dir}; rejected {rejected_total} intermediate candidates."
    )
    for key in sorted(generation_summary):
        entry = generation_summary[key]
        print(
            "  "
            f"{key}: generated={entry['generated_count']}, "
            f"rejected_candidates={entry['rejected_candidate_count']}"
        )
