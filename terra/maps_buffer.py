import hashlib
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from tqdm import tqdm
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import ImmutableMapsConfig
from terra.config import PARTIAL_RESET_FRACTIONS
from terra.config import REWARD_V2_DISTANCE_BOUND
from terra.config import REWARD_V2_DISTANCE_REF_M
from terra.env_generation.distance import REWARD_V2_DISTANCE_METRIC
from terra.env_generation.distance import REWARD_V2_DISTANCE_NORMALIZATION
from terra.env_generation.distance import REWARD_V2_DISTANCE_PROTOCOL_ID
from terra.env_generation.distance import compute_reward_v2_distance_map
from terra.map import compute_dynamic_dumpability
from terra.settings import IntMap
from terra.settings import IntLowDim


class MapsBuffer(NamedTuple):
    """
    Handles the retrieval of maps saved on disk,
    and the generation of the procedurally-generated maps.
    """

    maps: Array  # [map_type, n_maps, W, H]
    padding_mask: Array  # [map_type, n_maps, W, H]
    dumpability_masks_init: Array  # [map_type, n_maps, W, H]
    # [map_type, n_maps, n_axes_per_map, 3]: A,B,C line coefficients.
    trench_axes: Array
    trench_types: Array  # [map_type, n_maps], number of trench axes, or -1 if unavailable
    # [map_type, n_maps, W, H]: generator-emitted owner bits for trench cells.
    trench_axis_owners: Array
    foundation_border_axes: Array  # [map_type, n_maps, n_border_axes_per_map, 3]
    foundation_border_types: Array  # [map_type, n_maps], number of border axes, or -1
    action_maps: Array  # [map_type, n_maps, W, H]
    slot_indices: Array  # [map_type, n_maps], zero-based manifest slot
    family_ids: Array  # [map_type, n_maps], index into family_names
    primary_cell_ids: Array  # [map_type, n_maps], index into primary_cell_names
    n_maps: int  # number of maps for each map type
    distance_maps: Array  # [map_type, n_maps, W, H] normalized float32
    # [partial_tier - 1, map_type, n_maps, W, H]. The canonical target,
    # obstacles, dumpability, and distance arrays remain shared with full resets.
    partial_action_maps: Array
    partial_action_available: Array  # [partial_tier - 1, map_type, n_maps]
    partial_reset_supported_levels: Array  # [reset_tier, map_type]
    partial_reset_bank_sha256: str

    immutable_maps_cfg: ImmutableMapsConfig = ImmutableMapsConfig()
    family_names: tuple[str, ...] = ("unknown",)
    primary_cell_names: tuple[str, ...] = ("unknown",)

    def __hash__(self) -> int:
        return hash((len(self.maps),))

    def __eq__(self, __o: "MapsBuffer") -> bool:
        return len(self.maps) == len(__o.maps)

    @classmethod
    def new(
        cls,
        maps: Array,
        padding_mask: Array,
        trench_axes: Array,
        trench_types: Array,
        trench_axis_owners: Array,
        foundation_border_axes: Array,
        foundation_border_types: Array,
        dumpability_masks_init: Array,
        action_maps: Array,
        distance_maps: Array,
        slot_indices: Array | None = None,
        family_ids: Array | None = None,
        primary_cell_ids: Array | None = None,
        partial_action_maps: Array | None = None,
        partial_action_available: Array | None = None,
        partial_reset_supported_levels: Array | None = None,
        partial_reset_bank_sha256: str = "",
        family_names: tuple[str, ...] = ("unknown",),
        primary_cell_names: tuple[str, ...] = ("unknown",),
    ) -> "MapsBuffer":
        # PATCH: Set all action_map values of 1 to 5 at load time
        #action_maps = jnp.where(action_maps == 1, 5, action_maps)   #DELETE IF NOT NEEDED ANYMORE
        provenance_shape = maps.shape[:2]
        if slot_indices is None:
            slot_indices = jnp.broadcast_to(
                jnp.arange(maps.shape[1], dtype=jnp.int32),
                provenance_shape,
            )
        if family_ids is None:
            family_ids = jnp.zeros(provenance_shape, dtype=jnp.int32)
        if primary_cell_ids is None:
            primary_cell_ids = jnp.zeros(
                provenance_shape,
                dtype=jnp.int32,
            )
        if partial_action_maps is None:
            partial_action_maps = jnp.zeros(
                (0, *maps.shape),
                dtype=IntLowDim,
            )
        if partial_action_available is None:
            partial_action_available = jnp.zeros(
                (0, *maps.shape[:2]),
                dtype=jnp.bool_,
            )
        if partial_reset_supported_levels is None:
            partial_reset_supported_levels = jnp.concatenate(
                (
                    jnp.ones((1, maps.shape[0]), dtype=jnp.bool_),
                    jnp.zeros(
                        (len(PARTIAL_RESET_FRACTIONS), maps.shape[0]),
                        dtype=jnp.bool_,
                    ),
                ),
                axis=0,
            )
        return MapsBuffer(
            maps=maps.astype(IntLowDim),
            padding_mask=padding_mask.astype(IntLowDim),
            dumpability_masks_init=dumpability_masks_init.astype(jnp.bool_),
            trench_axes=trench_axes.astype(jnp.float32),
            trench_types=trench_types,
            trench_axis_owners=trench_axis_owners.astype(jnp.uint8),
            foundation_border_axes=foundation_border_axes.astype(jnp.float16),
            foundation_border_types=foundation_border_types,
            n_maps=maps.shape[1],
            action_maps=action_maps.astype(IntLowDim),
            slot_indices=jnp.asarray(slot_indices, dtype=jnp.int32),
            family_ids=jnp.asarray(family_ids, dtype=jnp.int32),
            primary_cell_ids=jnp.asarray(
                primary_cell_ids,
                dtype=jnp.int32,
            ),
            distance_maps=distance_maps.astype(jnp.float32),
            partial_action_maps=partial_action_maps.astype(IntLowDim),
            partial_action_available=partial_action_available.astype(jnp.bool_),
            partial_reset_supported_levels=jnp.asarray(
                partial_reset_supported_levels,
                dtype=jnp.bool_,
            ),
            partial_reset_bank_sha256=str(partial_reset_bank_sha256),
            family_names=tuple(family_names),
            primary_cell_names=tuple(primary_cell_names),
        )

    def _select_index(
        self,
        key: jax.random.PRNGKey,
        env_cfg: EnvConfig,
    ) -> tuple[Array, Array, Array]:
        curriculum_level = env_cfg.curriculum.level
        key, subkey = jax.random.split(key)
        full_idx = jax.random.randint(subkey, (), 0, self.n_maps)
        reset_tier = jnp.asarray(env_cfg.reset_tier, dtype=jnp.int32)
        # TerraEnvBatch.validate_reset_tiers owns fail-loud validation before
        # tracing. Effectful checks here are not valid under pmap(vmap(cond)).
        # The exact benchmark materializer intentionally calls this method on
        # an index-only namespace; keep that full-reset path byte-identical.
        partial_action_maps = getattr(self, "partial_action_maps", None)
        if (
            partial_action_maps is not None
            and partial_action_maps.shape[0] == len(PARTIAL_RESET_FRACTIONS)
        ):
            safe_tier_index = jnp.clip(
                reset_tier - 1,
                0,
                len(PARTIAL_RESET_FRACTIONS) - 1,
            )
            available = self.partial_action_available[
                safe_tier_index,
                curriculum_level,
            ]

            def _select_partial(_: None) -> Array:
                return jax.random.categorical(
                    subkey,
                    jnp.where(available, 0.0, -jnp.inf),
                ).astype(jnp.int32)

            idx = jax.lax.cond(
                reset_tier == 0,
                lambda _: full_idx,
                _select_partial,
                operand=None,
            )
        else:
            idx = full_idx
        return curriculum_level, idx, key

    def _select_map(self, key: jax.random.PRNGKey, env_cfg: EnvConfig) -> Array:
        curriculum_level, idx, key = self._select_index(key, env_cfg)
        map = self.maps[curriculum_level, idx]
        padding_mask = self.padding_mask[curriculum_level, idx]
        trench_axes = self.trench_axes[curriculum_level, idx]
        trench_type = self.trench_types[curriculum_level, idx]
        trench_axis_owners = self.trench_axis_owners[curriculum_level, idx]
        foundation_border_axes = self.foundation_border_axes[curriculum_level, idx]
        foundation_border_type = self.foundation_border_types[curriculum_level, idx]
        # make sure is int 32
        trench_type = trench_type.astype(jnp.int32)
        foundation_border_type = foundation_border_type.astype(jnp.int32)
        dumpability_mask_init = self.dumpability_masks_init[curriculum_level, idx]
        action_map = self.action_maps[curriculum_level, idx]
        reset_tier = jnp.asarray(env_cfg.reset_tier, dtype=jnp.int32)
        if self.partial_action_maps.shape[0] == len(PARTIAL_RESET_FRACTIONS):
            partial_action_map = self.partial_action_maps[
                jnp.clip(reset_tier - 1, 0, len(PARTIAL_RESET_FRACTIONS) - 1),
                curriculum_level,
                idx,
            ]
            action_map = jnp.where(
                reset_tier == 0,
                action_map,
                partial_action_map,
            )
        distance_map = self.distance_maps[curriculum_level, idx]
        return map, padding_mask, trench_axes, trench_type, trench_axis_owners, foundation_border_axes, foundation_border_type, dumpability_mask_init, action_map, distance_map, key

    @partial(jax.jit, static_argnums=(0,))
    def get_map_provenance(
        self,
        key: jax.random.PRNGKey,
        env_cfg: EnvConfig,
    ) -> tuple[Array, Array, Array, Array]:
        """Return the provenance selected by the same key path as get_map."""
        curriculum_level, idx, key = self._select_index(key, env_cfg)
        return (
            self.slot_indices[curriculum_level, idx],
            self.family_ids[curriculum_level, idx],
            self.primary_cell_ids[curriculum_level, idx],
            key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_map_from_disk(self, key: jax.random.PRNGKey, env_cfg: EnvConfig) -> Array:
        return self._select_map(key, env_cfg)

    @partial(jax.jit, static_argnums=(0,))
    def get_map(self, key: jax.random.PRNGKey, env_cfg) -> Array:
        (
            map,
            padding_mask,
            trench_axes,
            trench_type,
            trench_axis_owners,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_map,
            distance_map,
            key,
        ) = self._get_map_from_disk(key, env_cfg)
        # Ensure consistent dtypes for all return values
        trench_type = trench_type.astype(jnp.int32)
        foundation_border_type = foundation_border_type.astype(jnp.int32)
        return map, padding_mask, trench_axes, trench_type, trench_axis_owners, foundation_border_axes, foundation_border_type, dumpability_mask_init, action_map, distance_map, key

    def sample_map(self, key: jax.random.PRNGKey, env_cfg) -> Array:
        (
            map,
            padding_mask,
            trench_axes,
            trench_type,
            trench_axis_owners,
            foundation_border_axes,
            foundation_border_type,
            dumpability_mask_init,
            action_map,
            distance_map,
            key,
        ) = self._select_map(key, env_cfg)
        trench_type = trench_type.astype(jnp.int32)
        foundation_border_type = foundation_border_type.astype(jnp.int32)
        return map, padding_mask, trench_axes, trench_type, trench_axis_owners, foundation_border_axes, foundation_border_type, dumpability_mask_init, action_map, distance_map, key

    @partial(jax.jit, static_argnums=(0,))
    def get_map_init(self, key: int, env_cfg):
        return self.get_map(key, env_cfg)

    def sample_map_init(self, key: int, env_cfg):
        return self.sample_map(key, env_cfg)


def map_sanity_check(map: Array) -> None:
    valid = np.all((map == 0) | (map == 1) | (map == -1))
    if not valid:
        raise RuntimeError("Loaded target map is not valid.")


def occupancy_sanity_check(map: Array) -> None:
    valid = np.all((map == 0) | (map == 1))
    if not valid:
        raise RuntimeError("Loaded occupancy is not valid.")


def dumpability_sanity_check(map: Array) -> None:
    valid = np.all((map == 0) | (map == 1))
    if not valid:
        raise RuntimeError("Loaded dumpability mask is not valid.")


def actions_sanity_check(map: Array) -> None:
    array = np.asarray(map)
    dtype = array.dtype
    if array.size == 0:
        raise RuntimeError(
            f"Loaded actions map must not be empty; got dtype={dtype}, shape={array.shape}."
        )
    if not np.issubdtype(dtype, np.integer):
        raise RuntimeError(
            "Loaded actions map must use an integer dtype; "
            f"got dtype={dtype}, min={array.min()}, max={array.max()}."
        )

    minimum = int(array.min())
    maximum = int(array.max())
    int_low_dim_max = int(np.iinfo(np.int8).max)
    if minimum < -1 or maximum > int_low_dim_max:
        raise RuntimeError(
            "Loaded actions map values must fit Terra's signed action-map "
            f"range [-1, {int_low_dim_max}]; got dtype={dtype}, "
            f"min={minimum}, max={maximum}."
        )


PARTIAL_COMPLETION_CONFIG = "partial_completion_config.json"
PARTIAL_COMPLETION_MANIFEST = "partial_completion_manifest.jsonl"
PARTIAL_COMPLETION_REJECTIONS = "partial_completion_rejections.jsonl"
PARTIAL_RESET_BANK_INDEX = "partial_reset_bank.json"
PARTIAL_RESET_BANK_SCHEMA = "terra_sparse_partial_reset_bank_v1"
PARTIAL_RESET_LEAF_SCHEMA = "terra_sparse_partial_reset_leaf_v1"
PARTIAL_RESET_TRIPLET_CONTRACT = "strict_nested_source_triplet_v1"
PARTIAL_RESET_PILE_MODES = (
    "relay_corridor",
    "in_zone",
    "near_zone",
    "mixed",
)


def _partial_reset_pile_mode_policy(
    payload: dict[str, Any], source: Path
) -> tuple[str, ...]:
    """Read a new ordered policy while preserving relay-only v1 banks."""
    raw_policy = payload.get("pile_mode_policy")
    if raw_policy is None:
        raw_policy = [payload.get("pile_mode", "relay_corridor")]
    if (
        not isinstance(raw_policy, list)
        or not raw_policy
        or any(not isinstance(mode, str) for mode in raw_policy)
        or len(raw_policy) != len(set(raw_policy))
        or any(mode not in PARTIAL_RESET_PILE_MODES for mode in raw_policy)
    ):
        raise RuntimeError(
            f"{source} has invalid pile_mode_policy {raw_policy!r}; expected a "
            f"unique ordered subset of {PARTIAL_RESET_PILE_MODES}."
        )
    legacy_mode = payload.get("pile_mode")
    if legacy_mode is not None and (
        len(raw_policy) != 1 or raw_policy[0] != legacy_mode
    ):
        raise RuntimeError(
            f"{source} has inconsistent pile_mode and pile_mode_policy."
        )
    return tuple(raw_policy)


def partial_reset_action_sanity_check(
    target_map: Array,
    occupancy_map: Array,
    dumpability_map: Array,
    action_map: Array,
    *,
    expected_fraction: float,
) -> None:
    """Validate the load-bearing partial-reset invariants at the runtime boundary."""
    target = np.asarray(target_map)
    occupancy = np.asarray(occupancy_map, dtype=np.bool_)
    dumpability = np.asarray(dumpability_map, dtype=np.bool_)
    action = np.asarray(action_map)
    actions_sanity_check(action)
    if not target.shape == occupancy.shape == dumpability.shape == action.shape:
        raise RuntimeError(
            "Partial target, occupancy, dumpability, and action maps must have "
            "the same shape."
        )

    completed = action < 0
    positive = action > 0
    dig_target = target < 0
    if np.any(completed & ~dig_target):
        raise RuntimeError("Partial reset excavates outside the target dig region.")
    if np.any(positive & dig_target):
        raise RuntimeError("Partial reset places positive soil on unfinished dig target.")
    if np.any(positive & occupancy):
        raise RuntimeError("Partial reset places positive soil on an obstacle.")

    dynamic_dumpability = np.asarray(
        compute_dynamic_dumpability(dumpability, action),
        dtype=np.bool_,
    )
    if np.any(positive & ~dynamic_dumpability):
        raise RuntimeError(
            "Partial reset places positive soil outside initial dynamic dumpability."
        )

    required_dig_tiles = int(np.count_nonzero(dig_target))
    expected_completed = int(round(expected_fraction * required_dig_tiles))
    completed_volume = -int(action[completed].astype(np.int64).sum())
    positive_volume = int(action[positive].astype(np.int64).sum())
    if completed_volume != expected_completed:
        raise RuntimeError(
            "Partial reset completion does not match its tier: "
            f"expected {expected_completed}, got {completed_volume}."
        )
    if positive_volume != completed_volume:
        raise RuntimeError(
            "Partial reset violates mass conservation: "
            f"positive volume {positive_volume}, completed volume {completed_volume}."
        )
    if not np.any(dig_target & ~completed):
        raise RuntimeError("Partial reset must leave unfinished excavation work.")


def partial_reset_triplet_sanity_check(
    action_90: Array,
    action_75: Array,
    action_50: Array,
) -> None:
    """Require one strictly nested Backplay prefix for a canonical source."""
    completed_90 = np.asarray(action_90) < 0
    completed_75 = np.asarray(action_75) < 0
    completed_50 = np.asarray(action_50) < 0
    if not completed_90.shape == completed_75.shape == completed_50.shape:
        raise RuntimeError("Partial-reset source triplet shapes do not match.")
    if not (
        np.all(~completed_50 | completed_75)
        and np.all(~completed_75 | completed_90)
    ):
        raise RuntimeError(
            "Partial-reset source triplet is not nested as 50% within 75% "
            "within 90%."
        )
    counts = tuple(
        int(np.count_nonzero(mask))
        for mask in (completed_50, completed_75, completed_90)
    )
    if not counts[0] < counts[1] < counts[2]:
        raise RuntimeError(
            "Partial-reset source triplet must have strictly increasing "
            f"completed prefixes; got 50/75/90 counts {counts}."
        )


def partial_reset_bank_sha256(partial_reset_root: str | Path) -> str:
    """Hash the sparse sidecar bytes without hashing the self-naming root index."""
    root = Path(partial_reset_root)
    selected: list[Path] = []
    for name in (
        PARTIAL_COMPLETION_CONFIG,
        PARTIAL_COMPLETION_MANIFEST,
        PARTIAL_COMPLETION_REJECTIONS,
    ):
        selected.extend(root.rglob(name))
    selected.extend(root.rglob("actions/img_*.npy"))
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def load_partial_reset_action_sidecars(
    partial_reset_root: str | Path,
    maps_paths: list[str],
    maps: list[Array],
    occupancies: list[Array],
    dumpability_masks: list[Array],
    canonical_manifest_rows: list[list[dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """Load sparse, source-bound partial action maps over canonical map slots."""
    root = Path(partial_reset_root)
    if not root.is_dir():
        raise RuntimeError(f"Partial-reset root does not exist: {root}")
    index_path = root / PARTIAL_RESET_BANK_INDEX
    if not index_path.is_file():
        raise RuntimeError(f"Missing partial-reset bank index: {index_path}")
    try:
        bank_index = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {index_path}: {exc}") from exc
    if bank_index.get("schema") != PARTIAL_RESET_BANK_SCHEMA:
        raise RuntimeError(
            f"{index_path} must use schema {PARTIAL_RESET_BANK_SCHEMA!r}."
        )
    if (
        bank_index.get("source_triplet_contract")
        != PARTIAL_RESET_TRIPLET_CONTRACT
    ):
        raise RuntimeError(
            f"{index_path} must use source_triplet_contract="
            f"{PARTIAL_RESET_TRIPLET_CONTRACT!r}."
        )
    bank_pile_mode_policy = _partial_reset_pile_mode_policy(
        bank_index,
        index_path,
    )
    actual_bank_sha256 = partial_reset_bank_sha256(root)
    if bank_index.get("bank_sha256") != actual_bank_sha256:
        raise RuntimeError(
            "Partial-reset bank digest mismatch: "
            f"index declares {bank_index.get('bank_sha256')}, observed "
            f"{actual_bank_sha256}."
        )
    if not (
        len(maps_paths)
        == len(maps)
        == len(occupancies)
        == len(dumpability_masks)
        == len(canonical_manifest_rows)
    ):
        raise RuntimeError("Partial-reset level inputs are inconsistent.")

    level_count = len(maps_paths)
    source_count = int(maps[0].shape[0])
    partial_actions = np.zeros(
        (
            len(PARTIAL_RESET_FRACTIONS),
            level_count,
            source_count,
            *maps[0].shape[1:],
        ),
        dtype=np.int8,
    )
    available = np.zeros(
        (len(PARTIAL_RESET_FRACTIONS), level_count, source_count),
        dtype=np.bool_,
    )
    supported_levels = np.zeros(
        (len(PARTIAL_RESET_FRACTIONS) + 1, level_count),
        dtype=np.bool_,
    )
    supported_levels[0] = True
    indexed_paths = bank_index.get("supported_maps_paths")
    if not isinstance(indexed_paths, list) or any(
        not isinstance(value, str) for value in indexed_paths
    ):
        raise RuntimeError(f"{index_path} has invalid supported_maps_paths.")
    indexed_paths_set = set(indexed_paths)

    for level_index, maps_path in enumerate(maps_paths):
        if int(maps[level_index].shape[0]) != source_count:
            raise RuntimeError("All partial-reset curriculum levels need equal slot counts.")
        directory = root / maps_path
        config_path = directory / PARTIAL_COMPLETION_CONFIG
        manifest_path = directory / PARTIAL_COMPLETION_MANIFEST
        declared_supported = maps_path in indexed_paths_set
        if not declared_supported:
            if config_path.exists() or manifest_path.exists():
                raise RuntimeError(
                    f"Undeclared partial-reset leaf exists under {directory}."
                )
            continue
        if not config_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                "Missing declared partial-reset leaf under "
                f"{directory}; expected {PARTIAL_COMPLETION_CONFIG} and "
                f"{PARTIAL_COMPLETION_MANIFEST}."
            )
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {config_path}: {exc}") from exc

        if config.get("schema") != PARTIAL_RESET_LEAF_SCHEMA:
            raise RuntimeError(
                f"{config_path} must use schema {PARTIAL_RESET_LEAF_SCHEMA!r}."
            )
        if config.get("maps_path") != maps_path:
            raise RuntimeError(f"{config_path} is bound to the wrong maps_path.")
        configured_fractions = config.get("completion_fractions")
        if configured_fractions != list(PARTIAL_RESET_FRACTIONS):
            raise RuntimeError(
                f"{config_path} must contain exactly the reset fractions "
                f"{PARTIAL_RESET_FRACTIONS}."
            )
        leaf_pile_mode_policy = _partial_reset_pile_mode_policy(
            config,
            config_path,
        )
        if leaf_pile_mode_policy != bank_pile_mode_policy:
            raise RuntimeError(
                f"{config_path} pile mode policy {leaf_pile_mode_policy} does "
                f"not match bank policy {bank_pile_mode_policy}."
            )
        if (
            config.get("source_triplet_contract")
            != PARTIAL_RESET_TRIPLET_CONTRACT
        ):
            raise RuntimeError(
                f"{config_path} must use source_triplet_contract="
                f"{PARTIAL_RESET_TRIPLET_CONTRACT!r}."
            )
        if config.get("canonical_slot_count") != source_count:
            raise RuntimeError(
                f"{config_path} canonical_slot_count="
                f"{config.get('canonical_slot_count')} "
                f"does not match the canonical bank count {source_count}."
            )

        rows = _load_json_lines(manifest_path)
        if config.get("successful_variant_count") != len(rows):
            raise RuntimeError(
                f"{config_path} successful_variant_count does not match "
                f"{manifest_path}."
            )
        observed_sidecar_indices = [row.get("sidecar_index") for row in rows]
        expected_sidecar_indices = list(range(1, len(rows) + 1))
        if observed_sidecar_indices != expected_sidecar_indices:
            raise RuntimeError(
                f"{manifest_path} must enumerate ordered sidecar indices "
                f"1..{len(rows)}."
            )
        action_indices = _indexed_sidecars(directory / "actions", "img_", ".npy")
        if action_indices != expected_sidecar_indices:
            raise RuntimeError(
                f"Partial action sidecars in {directory / 'actions'} must "
                f"enumerate exactly 1..{len(rows)}."
            )
        seen: set[tuple[int, int]] = set()
        source_variant_seeds: dict[int, set[int]] = {}
        source_pile_modes: dict[int, set[str]] = {}
        for row in rows:
            source_index = row.get("source_index")
            reset_tier = row.get("reset_tier")
            requested_fraction = row.get("requested_completion_fraction")
            variant_seed = row.get("variant_seed")
            pile_mode = row.get("pile_mode")
            if (
                not isinstance(source_index, int)
                or not 1 <= source_index <= source_count
                or not isinstance(reset_tier, int)
                or not 1 <= reset_tier <= len(PARTIAL_RESET_FRACTIONS)
                or not isinstance(requested_fraction, (int, float))
                or not isinstance(variant_seed, int)
                or pile_mode not in leaf_pile_mode_policy
            ):
                raise RuntimeError(f"Invalid partial-reset manifest row: {row}")
            tier_index = reset_tier - 1
            if not np.isclose(
                float(requested_fraction),
                PARTIAL_RESET_FRACTIONS[tier_index],
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"Reset tier {reset_tier} has wrong fraction "
                    f"{requested_fraction}."
                )
            key = (tier_index, source_index - 1)
            if key in seen:
                raise RuntimeError(
                    "Partial-reset manifest contains duplicate source/fraction "
                    f"entry {key}."
                )

            canonical_row = canonical_manifest_rows[level_index][source_index - 1]
            if (
                row.get("source_map_id") != canonical_row.get("map_id")
                or row.get("source_scenario_id") != canonical_row.get("scenario_id")
                or not _is_sha256(row.get("source_scenario_id"))
            ):
                raise RuntimeError(
                    "Partial-reset source identity does not match canonical slot "
                    f"{maps_path}:{source_index}."
                )

            action_path = directory / "actions" / f"img_{row['sidecar_index']}.npy"
            if _sha256_file(action_path) != row.get("action_sha256"):
                raise RuntimeError(f"Partial action hash mismatch: {action_path}")
            action = _ensure_spatial_2d(np.load(action_path), str(action_path))
            partial_reset_action_sanity_check(
                maps[level_index][source_index - 1],
                occupancies[level_index][source_index - 1],
                dumpability_masks[level_index][source_index - 1],
                action,
                expected_fraction=PARTIAL_RESET_FRACTIONS[tier_index],
            )
            contained_dump_capacity_sanity_check(
                maps[level_index][source_index - 1],
                occupancies[level_index][source_index - 1],
                dumpability_masks[level_index][source_index - 1],
                action,
            )
            achieved_fraction = row.get("achieved_completion_fraction")
            actual_fraction = (
                int(np.count_nonzero(action < 0))
                / int(np.count_nonzero(maps[level_index][source_index - 1] < 0))
            )
            if not isinstance(achieved_fraction, (int, float)) or not np.isclose(
                float(achieved_fraction),
                actual_fraction,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Partial-reset manifest/action mismatch for {action_path}."
                )
            partial_actions[tier_index, level_index, source_index - 1] = action.astype(
                np.int8
            )
            available[tier_index, level_index, source_index - 1] = True
            source_variant_seeds.setdefault(source_index - 1, set()).add(variant_seed)
            source_pile_modes.setdefault(source_index - 1, set()).add(pile_mode)
            seen.add(key)

        tier_counts = available[:, level_index].sum(axis=1)
        tier_availability = available[:, level_index]
        if not (
            np.array_equal(tier_availability[0], tier_availability[1])
            and np.array_equal(tier_availability[1], tier_availability[2])
        ):
            raise RuntimeError(
                f"Declared supported level {maps_path} must contain complete "
                "source triplets with identical availability in every tier."
            )
        if not np.all(tier_counts > 0):
            raise RuntimeError(
                f"Declared supported level {maps_path} must have at least one "
                f"sidecar in every tier; got {tier_counts.tolist()}."
            )
        for source_slot in np.flatnonzero(tier_availability[0]):
            source_slot = int(source_slot)
            if source_variant_seeds.get(source_slot) is None or len(
                source_variant_seeds[source_slot]
            ) != 1:
                raise RuntimeError(
                    f"Partial-reset source triplet {maps_path}:{source_slot + 1} "
                    "must share one deterministic variant_seed."
                )
            if source_pile_modes.get(source_slot) is None or len(
                source_pile_modes[source_slot]
            ) != 1:
                raise RuntimeError(
                    f"Partial-reset source triplet {maps_path}:{source_slot + 1} "
                    "must use one pile mode across every tier."
                )
            partial_reset_triplet_sanity_check(
                partial_actions[0, level_index, source_slot],
                partial_actions[1, level_index, source_slot],
                partial_actions[2, level_index, source_slot],
            )
        supported_levels[1:, level_index] = True

    unknown_indexed_paths = indexed_paths_set - set(maps_paths)
    if unknown_indexed_paths:
        raise RuntimeError(
            "Partial-reset bank contains unknown curriculum paths: "
            f"{sorted(unknown_indexed_paths)}."
        )
    if not np.all(np.any(supported_levels[1:], axis=1)):
        raise RuntimeError(
            "Partial-reset bank must support at least one curriculum level for "
            "every scheduled tier."
        )
    return partial_actions, available, actual_bank_sha256, supported_levels


def contained_dump_capacity_sanity_check(
    target_map: Array,
    occupancy_map: Array,
    dumpability_map: Array,
    action_map: Array,
    *,
    minimum_single_layer_ratio: float | None = None,
    maximum_bucket_load: int = int(np.iinfo(np.int8).max),
) -> dict[str, float | int | bool]:
    """Validate exact-mask storage for the contained tracked-excavator contract."""
    target = np.asarray(target_map)
    occupancy = np.asarray(occupancy_map, dtype=np.bool_)
    dumpability = np.asarray(dumpability_map, dtype=np.bool_)
    action = np.asarray(action_map)
    if not (
        target.shape
        == occupancy.shape
        == dumpability.shape
        == action.shape
    ):
        raise RuntimeError(
            "Target, occupancy, dumpability, and action maps must have the "
            "same shape for dump-capacity validation."
        )

    declared_dump = target > 0
    if np.any(declared_dump & occupancy):
        raise RuntimeError("Accepted dump target overlaps an obstacle.")
    accepted_dump = declared_dump & ~occupancy
    if np.any(accepted_dump & ~dumpability):
        raise RuntimeError("Accepted dump target contains non-dumpable cells.")

    accepted_cells = int(accepted_dump.sum())
    required_dig_volume = int(
        np.clip(-target.astype(np.int64), a_min=0, a_max=None).sum()
    )
    if not np.any(declared_dump):
        return {
            "has_dump_requirement": False,
            "accepted_dump_cells": 0,
            "required_dig_volume": required_dig_volume,
            "single_layer_capacity_ratio": 0.0,
            "representable_remaining_volume": 0,
            "required_remaining_volume": 0,
            "maximum_bucket_load": 0,
            "maximum_single_cell_headroom": 0,
        }

    positive_soil = np.clip(
        action.astype(np.int64),
        a_min=0,
        a_max=None,
    )
    accepted_positive_volume = int(positive_soil[accepted_dump].sum())
    total_positive_volume = int(positive_soil.sum())
    eventual_dump_volume = max(required_dig_volume, total_positive_volume)

    if accepted_cells == 0:
        raise RuntimeError(
            "Map declares a dump requirement but has no accepted dump cells."
        )

    if minimum_single_layer_ratio is not None and required_dig_volume <= 0:
        raise RuntimeError(
            "A minimum single-layer dump-capacity ratio requires a dig task."
        )
    single_layer_ratio = (
        accepted_cells / required_dig_volume
        if required_dig_volume > 0
        else float("inf")
    )
    if (
        minimum_single_layer_ratio is not None
        and single_layer_ratio + 1e-12 < minimum_single_layer_ratio
    ):
        raise RuntimeError(
            "Exact accepted dump mask fails the single-layer capacity ratio: "
            f"{single_layer_ratio:.6f} < {minimum_single_layer_ratio:.6f}."
        )

    int8_max = int(np.iinfo(np.int8).max)
    accepted_heights = positive_soil[accepted_dump]
    headroom = int8_max - accepted_heights
    if np.any(headroom < 0):
        raise RuntimeError(
            "Accepted dump target contains soil outside the int8 height range."
        )
    representable_remaining_volume = int(headroom.sum())
    required_remaining_volume = max(
        eventual_dump_volume - accepted_positive_volume,
        0,
    )
    if representable_remaining_volume < required_remaining_volume:
        raise RuntimeError(
            "Exact accepted dump mask cannot represent all remaining soil: "
            f"{representable_remaining_volume} < {required_remaining_volume}."
        )

    valid_maximum_bucket = min(maximum_bucket_load, eventual_dump_volume)
    maximum_single_cell_headroom = int(headroom.max())
    if maximum_single_cell_headroom < valid_maximum_bucket:
        raise RuntimeError(
            "Exact accepted dump mask cannot represent the largest valid "
            "contained bucket on one reachable cell: "
            f"{maximum_single_cell_headroom} < {valid_maximum_bucket}."
        )

    return {
        "has_dump_requirement": True,
        "accepted_dump_cells": accepted_cells,
        "required_dig_volume": required_dig_volume,
        "single_layer_capacity_ratio": float(single_layer_ratio),
        "representable_remaining_volume": representable_remaining_volume,
        "required_remaining_volume": required_remaining_volume,
        "maximum_bucket_load": valid_maximum_bucket,
        "maximum_single_cell_headroom": maximum_single_cell_headroom,
    }


def _ensure_spatial_2d(array: Array, source: str) -> Array:
    """Normalize singleton-channel maps and reject true non-2D spatial data."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and 1 in (array.shape[0], array.shape[-1]):
        squeezed = np.squeeze(array)
        if squeezed.ndim == 2:
            return squeezed
    raise RuntimeError(
        f"Loaded map array from {source} has shape {array.shape}; expected a 2D "
        "grid or a singleton-channel 3D grid."
    )


def metadata_sanity_check(metadata: dict[str, Any]) -> None:
    valid = True
    k = metadata.keys()
    valid &= "A" in k
    valid &= "B" in k
    valid &= "C" in k
    valid &= isinstance(metadata["A"], float)
    valid &= isinstance(metadata["B"], float)
    valid &= isinstance(metadata["C"], float)
    if not valid:
        raise RuntimeError("Loaded metadata is not valid.")


TRENCH_AXIS_RECORD_SIZE = 3
TRENCH_AXIS_CONTRACT = "generator_owner_bits_v1"


def trench_axis_owners_sanity_check(
    target_map: Array,
    owners_map: Array,
    trench_type: int,
    max_trench_type: int,
) -> np.ndarray:
    """Validate the one supported trench contract at the loader boundary."""

    target = np.asarray(target_map)
    owners = np.asarray(owners_map)
    if owners.shape != target.shape:
        raise RuntimeError(
            "Trench owner map shape does not match the target map: "
            f"{owners.shape} != {target.shape}."
        )
    if owners.dtype != np.uint8:
        raise RuntimeError(
            f"Trench owner maps must use uint8, got {owners.dtype}."
        )
    if trench_type > max_trench_type:
        raise RuntimeError(
            f"Trench declares {trench_type} axes, maximum is {max_trench_type}."
        )
    if trench_type <= 0:
        if np.any(owners):
            raise RuntimeError("Non-trench maps must have an all-zero owner map.")
        return owners

    valid_bits = (1 << trench_type) - 1
    if np.any(np.bitwise_and(owners, np.uint8(~valid_bits & 0xFF))):
        raise RuntimeError(
            "Trench owner map references an undeclared axis bit."
        )
    dig_target = target < 0
    if np.any(dig_target & (owners == 0)):
        raise RuntimeError("Every trench target cell must have an owning axis.")
    if np.any((~dig_target) & (owners != 0)):
        raise RuntimeError("Trench owner bits may appear only on dig target cells.")
    return owners


def trench_axis_contract_sanity_check(
    metadata: Mapping[str, Any],
    owners_map: Array,
    trench_type: int,
) -> None:
    """Bind a trench's exact owner sidecar to its axis metadata."""

    if trench_type <= 0:
        return
    if metadata.get("trench_axis_contract") != TRENCH_AXIS_CONTRACT:
        raise RuntimeError(
            f"Trench metadata must declare {TRENCH_AXIS_CONTRACT!r}."
        )
    expected = metadata.get("trench_axis_owners_sha256")
    actual = hashlib.sha256(
        np.ascontiguousarray(owners_map, dtype=np.uint8).tobytes()
    ).hexdigest()
    if expected != actual:
        raise RuntimeError(
            "Trench owner sidecar hash disagrees with its metadata: "
            f"expected {expected!r}, got {actual}."
        )


def _trench_records_from_metadata(
    metadata: dict[str, Any],
    max_trench_type: int,
) -> tuple[list[list[float]], int]:
    """Load generator-owned line equations into a fixed-width table."""

    raw_axes = list(metadata.get("axes_ABC", []) or [])
    declared_count = metadata.get("trench_axes_count")
    if declared_count not in (None, -1) and int(declared_count) != len(raw_axes):
        raise RuntimeError(
            "Trench metadata count disagrees with axes_ABC: "
            f"declared {declared_count}, found {len(raw_axes)}."
        )
    if len(raw_axes) > max_trench_type:
        raise RuntimeError(
            f"Trench metadata has {len(raw_axes)} axes but Terra supports "
            f"at most {max_trench_type}."
        )

    records: list[list[float]] = []
    for axis in raw_axes:
        metadata_sanity_check(axis)
        normal = np.hypot(float(axis["A"]), float(axis["B"]))
        if not np.isfinite(normal) or normal <= 1e-6:
            raise RuntimeError("Trench axis has a non-finite or zero normal.")
        records.append(
            [float(axis["A"]), float(axis["B"]), float(axis["C"])]
        )

    trench_type = len(records) if records else -1
    while len(records) < max_trench_type:
        records.append([-97.0] * TRENCH_AXIS_RECORD_SIZE)
    return records, trench_type


EXACT_DATASET_SCHEMA = "terra_exact_map_dataset_v1"
EXACT_DATASET_MANIFEST = "manifest.jsonl"
EXACT_DATASET_METADATA = "dataset.json"
RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT = "terra_reset_arrays_sha256_v1"
LEGACY_SCENARIO_IDENTITY_CONTRACT = "terra_legacy_map_id_v0"
LEGACY_DISTANCE_PROTOCOL_ID = "legacy_dataset_distance"
RESET_ARRAY_FOLDERS = (
    "images",
    "occupancy",
    "dumpability",
    "actions",
    "distance",
)
EXACT_DATASET_REQUIRED_ROW_FIELDS = (
    "slot_index",
    "map_id",
    "source_id",
    "split",
    "family",
    "stratum",
    "primary_cell",
    "slot_weight",
    "identity_slot_multiplicity",
)


def _reward_v2_distance_contract(
    directory: Path,
    expected_shape: tuple[int, int],
) -> dict[str, float] | None:
    """Read the one supported R2 distance contract, or identify legacy data."""
    metadata_path = directory / EXACT_DATASET_METADATA
    metadata = json.loads(metadata_path.read_text())
    protocol_id = metadata.get("distance_protocol_id")
    if protocol_id is None:
        return None
    if protocol_id != REWARD_V2_DISTANCE_PROTOCOL_ID:
        raise RuntimeError(
            f"Unsupported distance_protocol_id {protocol_id!r} in {metadata_path}."
        )
    if metadata.get("distance_metric") != REWARD_V2_DISTANCE_METRIC:
        raise RuntimeError(
            f"{metadata_path} must declare distance_metric "
            f"{REWARD_V2_DISTANCE_METRIC!r} for R2."
        )
    if metadata.get("distance_normalization") != REWARD_V2_DISTANCE_NORMALIZATION:
        raise RuntimeError(
            f"{metadata_path} must declare distance_normalization "
            f"{REWARD_V2_DISTANCE_NORMALIZATION!r} for R2."
        )
    if expected_shape[0] != expected_shape[1]:
        raise RuntimeError("The R2 physical-distance contract requires square maps.")

    expected_tile_size_m = (
        ImmutableMapsConfig().edge_length_m / expected_shape[0]
    )
    values: dict[str, float] = {}
    for field in ("tile_size_m", "distance_ref_m", "distance_bound"):
        value = metadata.get(field)
        if (
            not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise RuntimeError(f"{metadata_path} has invalid R2 field {field}.")
        values[field] = float(value)
    if not np.isclose(
        values["tile_size_m"],
        expected_tile_size_m,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{metadata_path} tile_size_m={values['tile_size_m']} does not match "
            f"Terra geometry {expected_tile_size_m}."
        )
    for field, expected in (
        ("distance_ref_m", REWARD_V2_DISTANCE_REF_M),
        ("distance_bound", REWARD_V2_DISTANCE_BOUND),
    ):
        if not np.isclose(values[field], expected, rtol=0.0, atol=0.0):
            raise RuntimeError(
                f"{metadata_path} {field}={values[field]} does not match the "
                f"frozen R2 value {expected}."
            )
    return values


def reset_array_scenario_sha256(arrays: Mapping[str, Any]) -> str:
    """Hash the five arrays consumed by reset in one canonical order."""
    if set(arrays) != set(RESET_ARRAY_FOLDERS):
        raise ValueError(
            "Scenario identity requires exactly "
            f"{RESET_ARRAY_FOLDERS}; got {tuple(arrays)}."
        )
    digest = hashlib.sha256()
    for name in RESET_ARRAY_FOLDERS:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Missing required dataset file: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError(
                f"Expected an object in {path} at line {line_number}."
            )
        rows.append(row)
    return rows


def _indexed_sidecars(directory: Path, prefix: str, suffix: str) -> list[int]:
    indices = []
    for path in directory.glob(f"{prefix}*{suffix}"):
        middle = path.name[len(prefix) : -len(suffix)]
        if not middle.isdigit():
            raise RuntimeError(f"Malformed indexed dataset sidecar: {path}")
        indices.append(int(middle))
    return sorted(indices)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_source_registry(
    dataset_directory: Path,
    metadata: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    identity_contract: str,
) -> None:
    relative_registry = metadata.get("source_registry")
    expected_sha256 = metadata.get("source_registry_sha256")
    if not isinstance(relative_registry, str) or not relative_registry:
        raise RuntimeError(
            f"{EXACT_DATASET_METADATA} must declare source_registry."
        )
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError(
            f"{EXACT_DATASET_METADATA} must declare source_registry_sha256."
        )
    registry_path = (dataset_directory / relative_registry).resolve()
    if not registry_path.is_file():
        raise RuntimeError(f"Missing required source registry: {registry_path}")
    actual_sha256 = _sha256_file(registry_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Source registry hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256} for {registry_path}."
        )

    registry_rows = _load_json_lines(registry_path)
    source_splits: dict[str, set[str]] = {}
    identities: dict[str, tuple[str, ...]] = {}
    for row in registry_rows:
        for field in ("map_id", "source_id", "split"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise RuntimeError(
                    f"Source registry row is missing nonempty {field}: {row}"
                )
        map_id = row["map_id"]
        identity = (row["source_id"], row["split"])
        if identity_contract == RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT:
            if not _is_sha256(row.get("scenario_id")):
                raise RuntimeError(
                    "Strict source registry row is missing a valid scenario_id: "
                    f"{row}"
                )
            identity = (*identity, row["scenario_id"])
        if map_id in identities and identities[map_id] != identity:
            raise RuntimeError(
                f"Source registry assigns conflicting provenance to {map_id}."
            )
        identities[map_id] = identity
        source_splits.setdefault(row["source_id"], set()).add(row["split"])

    overlaps = {
        source_id: sorted(splits)
        for source_id, splits in source_splits.items()
        if len(splits) > 1
    }
    if overlaps:
        example = next(iter(overlaps.items()))
        raise RuntimeError(
            "Source IDs are not split-disjoint; "
            f"example {example[0]} appears in {example[1]}."
        )

    for row in manifest_rows:
        identity = identities.get(row["map_id"])
        expected = (row["source_id"], row["split"])
        if identity_contract == RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT:
            expected = (*expected, row["scenario_id"])
        if identity != expected:
            raise RuntimeError(
                "Manifest provenance does not match source registry for "
                f"{row['map_id']}: expected {expected}, got {identity}."
            )


def validate_exact_dataset_contract(
    folder_path: str | Path,
    expected_count: int,
) -> tuple[list[dict[str, Any]], tuple[int, int], float | None]:
    """Validate one frozen bank before arrays enter JAX construction."""
    directory = Path(folder_path)
    metadata_path = directory / EXACT_DATASET_METADATA
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing required dataset file: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {metadata_path}: {exc}") from exc
    if metadata.get("schema") != EXACT_DATASET_SCHEMA:
        raise RuntimeError(
            f"{metadata_path} must use schema {EXACT_DATASET_SCHEMA!r}."
        )
    identity_contract = metadata.get("scenario_identity_contract")
    if identity_contract not in (
        RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
        LEGACY_SCENARIO_IDENTITY_CONTRACT,
    ):
        raise RuntimeError(
            f"{metadata_path} must explicitly declare scenario_identity_contract "
            f"as {RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT!r} for byte-verified "
            f"banks or {LEGACY_SCENARIO_IDENTITY_CONTRACT!r} for legacy banks."
        )
    if metadata.get("slot_count") != expected_count:
        raise RuntimeError(
            f"Dataset slot count mismatch: DATASET_SIZE={expected_count}, "
            f"dataset declares {metadata.get('slot_count')}."
        )
    shape = metadata.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise RuntimeError(f"{metadata_path} must declare a positive 2-D shape.")
    expected_shape = (shape[0], shape[1])
    for field in ("distance_metric", "distance_normalization"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise RuntimeError(f"{metadata_path} must declare nonempty {field}.")
    if metadata.get("accepted_dump_contract") != "exact_visible_dump_v1":
        raise RuntimeError(
            f"{metadata_path} must declare accepted_dump_contract "
            "'exact_visible_dump_v1'."
        )

    minimum_capacity_ratio = metadata.get("minimum_dump_capacity_ratio")
    if minimum_capacity_ratio is not None:
        if (
            not isinstance(minimum_capacity_ratio, (int, float))
            or not np.isfinite(minimum_capacity_ratio)
            or minimum_capacity_ratio <= 0
        ):
            raise RuntimeError(
                f"{metadata_path} has invalid minimum_dump_capacity_ratio."
            )
        minimum_capacity_ratio = float(minimum_capacity_ratio)

    manifest_path = directory / EXACT_DATASET_MANIFEST
    rows = _load_json_lines(manifest_path)
    if len(rows) != expected_count:
        raise RuntimeError(
            f"{manifest_path} has {len(rows)} slots, expected {expected_count}."
        )
    actual_slots = [row.get("slot_index") for row in rows]
    expected_slots = list(range(1, expected_count + 1))
    if actual_slots != expected_slots:
        raise RuntimeError(
            f"{manifest_path} must enumerate contiguous ordered slots "
            f"1..{expected_count}; got {actual_slots[:8]}."
        )
    for row in rows:
        missing = [
            field
            for field in EXACT_DATASET_REQUIRED_ROW_FIELDS
            if field not in row
        ]
        if missing:
            raise RuntimeError(
                f"Manifest slot {row.get('slot_index')} is missing fields {missing}."
            )
        for field in (
            "map_id",
            "source_id",
            "split",
            "family",
            "stratum",
            "primary_cell",
        ):
            if not isinstance(row[field], str) or not row[field]:
                raise RuntimeError(
                    f"Manifest slot {row['slot_index']} has invalid {field}."
                )
        if (
            identity_contract == RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT
            and not _is_sha256(row.get("scenario_id"))
        ):
            raise RuntimeError(
                f"Strict manifest slot {row['slot_index']} must declare a valid "
                "scenario_id."
            )
        if (
            not isinstance(row["slot_weight"], (int, float))
            or not np.isfinite(row["slot_weight"])
            or row["slot_weight"] <= 0
        ):
            raise RuntimeError(
                f"Manifest slot {row['slot_index']} has invalid slot_weight."
            )

    multiplicities: dict[str, int] = {}
    for row in rows:
        multiplicities[row["map_id"]] = multiplicities.get(row["map_id"], 0) + 1
    if metadata.get("unique_identity_count") != len(multiplicities):
        raise RuntimeError(
            "Unique identity count mismatch: dataset declares "
            f"{metadata.get('unique_identity_count')}, observed "
            f"{len(multiplicities)}."
        )
    for row in rows:
        observed = multiplicities[row["map_id"]]
        if row["identity_slot_multiplicity"] != observed:
            raise RuntimeError(
                f"Manifest slot {row['slot_index']} declares multiplicity "
                f"{row['identity_slot_multiplicity']} for {row['map_id']}, "
                f"observed {observed}."
            )

    required_indices = list(range(1, expected_count + 1))
    sidecars = (
        ("images", "img_", ".npy"),
        ("occupancy", "img_", ".npy"),
        ("dumpability", "img_", ".npy"),
        ("actions", "img_", ".npy"),
        ("distance", "img_", ".npy"),
        ("metadata", "trench_", ".json"),
    )
    for subdirectory, prefix, suffix in sidecars:
        path = directory / subdirectory
        observed_indices = _indexed_sidecars(path, prefix, suffix)
        if observed_indices != required_indices:
            raise RuntimeError(
                f"Dataset sidecars in {path} must enumerate exactly "
                f"1..{expected_count}; got {observed_indices[:8]}."
            )
    owners_directory = directory / "trench_axis_owners"
    if owners_directory.exists():
        observed_indices = _indexed_sidecars(owners_directory, "img_", ".npy")
        if observed_indices != required_indices:
            raise RuntimeError(
                f"Dataset sidecars in {owners_directory} must enumerate exactly "
                f"1..{expected_count}; got {observed_indices[:8]}."
            )

    _validate_source_registry(
        directory,
        metadata,
        rows,
        identity_contract,
    )
    if identity_contract == RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT:
        for row in rows:
            slot = int(row["slot_index"])
            arrays = {
                folder: np.load(
                    directory / folder / f"img_{slot}.npy",
                    allow_pickle=False,
                )
                for folder in RESET_ARRAY_FOLDERS
            }
            actual_scenario_id = reset_array_scenario_sha256(arrays)
            if actual_scenario_id != row["scenario_id"]:
                raise RuntimeError(
                    f"Scenario identity mismatch at slot {slot}: manifest "
                    f"declares {row['scenario_id']}, published arrays hash to "
                    f"{actual_scenario_id}."
                )
    return rows, expected_shape, minimum_capacity_ratio


def load_single_map(map_path: str) -> Array:
    """
    Load a single map and its associated files from the specified path.
    Supports two directory structures:
    
    1. Flat structure (original): Directory containing files directly:
       - image.npy: The map image
       - occupancy.npy: The occupancy map
       - dumpability.npy: The dumpability mask
       - distance.npy: The distance map
       - actions.npy: The actions map (optional)
       - metadata.json: Metadata file containing trench axes (optional)
    
    2. Subdirectory structure (like test_map2): Directory with subdirectories:
       - images/img_1.npy: The map image
       - occupancy/img_1.npy: The occupancy map
       - dumpability/img_1.npy: The dumpability mask
       - distance/img_1.npy: The distance map
       - actions/img_1.npy: The actions map (optional)
       - metadata/map.json: Metadata file containing trench axes (optional)

    Args: map_path: Path to the map files
    Returns: Tuple containing map data in the same format as load_maps_from_disk
    """
    map_path = Path(map_path)

    # Check if subdirectory structure exists
    images_dir = map_path / "images"
    has_subdir_structure = images_dir.exists() and (images_dir / "img_1.npy").exists()

    if has_subdir_structure:
        # Load from subdirectory structure
        image_file = map_path / "images" / "img_1.npy"
        occupancy_file = map_path / "occupancy" / "img_1.npy"
        dumpability_file = map_path / "dumpability" / "img_1.npy"
        distance_file = map_path / "distance" / "img_1.npy"
        actions_file = map_path / "actions" / "img_1.npy"
        trench_axis_owners_file = (
            map_path / "trench_axis_owners" / "img_1.npy"
        )
        metadata_file = map_path / "metadata" / "map.json"
    else:
        # Load from flat structure (original behavior)
        image_file = map_path / "image.npy"
        occupancy_file = map_path / "occupancy.npy"
        dumpability_file = map_path / "dumpability.npy"
        distance_file = map_path / "distance.npy"
        actions_file = map_path / "actions.npy"
        trench_axis_owners_file = map_path / "trench_axis_owners.npy"
        metadata_file = map_path / "metadata.json"

    # Load map
    image = _ensure_spatial_2d(np.load(image_file), str(image_file))
    map_sanity_check(image)

    # Load occupancy
    occupancy = _ensure_spatial_2d(np.load(occupancy_file), str(occupancy_file))
    occupancy_sanity_check(occupancy)

    # Load dumpability mask
    dumpability_mask_init = _ensure_spatial_2d(
        np.load(dumpability_file), str(dumpability_file)
    )
    dumpability_sanity_check(dumpability_mask_init)

    # Load distance map
    distance_map_init = _ensure_spatial_2d(np.load(distance_file), str(distance_file))

    # Check if actions map exists
    if actions_file.exists():
        actions_map = _ensure_spatial_2d(np.load(actions_file), str(actions_file))
        actions_sanity_check(actions_map)
    else:
        actions_map = np.zeros_like(image, dtype=IntMap)
    trench_axis_owners = (
        _ensure_spatial_2d(
            np.load(trench_axis_owners_file),
            str(trench_axis_owners_file),
        )
        if trench_axis_owners_file.exists()
        else np.zeros_like(image, dtype=np.uint8)
    )
    contained_dump_capacity_sanity_check(
        image,
        occupancy,
        dumpability_mask_init,
        actions_map,
    )

    # Try to load metadata
    max_trench_type = 4
    max_foundation_border_type = 64
    trench_axes = -97.0 * np.ones(
        (max_trench_type, TRENCH_AXIS_RECORD_SIZE)
    )
    trench_type = -1
    foundation_border_axes = -97.0 * np.ones((max_foundation_border_type, 3))
    foundation_border_type = -1
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        trench_ax, trench_type = _trench_records_from_metadata(
            metadata,
            max_trench_type,
        )
        trench_axes = np.array(trench_ax)
        trench_axis_contract_sanity_check(
            metadata,
            trench_axis_owners,
            trench_type,
        )
        foundation_ax = metadata.get("foundation_border_axes_ABC", [])
        if len(foundation_ax) > 0:
            metadata_sanity_check(foundation_ax[0])
            foundation_ax = [[el["A"], el["B"], el["C"]] for el in foundation_ax]
            foundation_border_type = len(foundation_ax)
            if len(foundation_ax) > max_foundation_border_type:
                foundation_ax = foundation_ax[:max_foundation_border_type]
                foundation_border_type = max_foundation_border_type
            while len(foundation_ax) < max_foundation_border_type:
                foundation_ax.append([-97, -97, -97])
            foundation_border_axes = np.array(foundation_ax)
    else:
        print(f"No metadata found for given map: {map_path}.")

    trench_axis_owners_sanity_check(
        image,
        trench_axis_owners,
        trench_type,
        max_trench_type,
    )

    # Convert to single-element arrays
    maps = jnp.array([image], dtype=IntMap)
    occupancies = jnp.array([occupancy], dtype=IntMap)
    trench_axes = jnp.array([trench_axes])
    trench_types = jnp.array([trench_type], dtype=jnp.int32)
    trench_axis_owners = jnp.array([trench_axis_owners], dtype=jnp.uint8)
    foundation_border_axes = jnp.array([foundation_border_axes])
    dumpability_masks_init = jnp.array([dumpability_mask_init], dtype=jnp.bool_)
    actions = jnp.array([actions_map], dtype=IntMap)
    distances = jnp.array([distance_map_init], dtype=jnp.float32)

    return (
        maps,
        occupancies,
        trench_axes,
        trench_types,
        trench_axis_owners,
        foundation_border_axes,
        foundation_border_type,
        dumpability_masks_init,
        actions,
        distances,
    )


def load_maps_from_disk(
    folder_path: str,
    require_trench_metadata: bool = False,
    require_exact_contract: bool = True,
    required_distance_protocol_id: str = LEGACY_DISTANCE_PROTOCOL_ID,
) -> Array:
    # Set the max number of branches the trench has. v5-main net4 trenches carry
    # four axes, so this pads to 4 and truncates anything longer (the foundation
    # border path below has always truncated; the trench path used to produce a
    # ragged list and crash in jnp.array).
    max_trench_type = 4
    max_foundation_border_type = 64

    dataset_size = int(os.getenv("DATASET_SIZE", -1))
    if dataset_size <= 0:
        raise RuntimeError("DATASET_SIZE must be > 0.")
    expected_shape = None
    minimum_dump_capacity_ratio = None
    reward_v2_distance = None
    if require_exact_contract:
        (
            _,
            expected_shape,
            minimum_dump_capacity_ratio,
        ) = validate_exact_dataset_contract(folder_path, dataset_size)
        reward_v2_distance = _reward_v2_distance_contract(
            Path(folder_path),
            expected_shape,
        )
    if required_distance_protocol_id == LEGACY_DISTANCE_PROTOCOL_ID:
        if reward_v2_distance is not None:
            raise RuntimeError(
                "Legacy reward requested an R2 physical-distance dataset."
            )
    elif required_distance_protocol_id == REWARD_V2_DISTANCE_PROTOCOL_ID:
        if reward_v2_distance is None:
            raise RuntimeError(
                "Reward-v2 requires a dataset declaring and validating "
                f"distance_protocol_id={REWARD_V2_DISTANCE_PROTOCOL_ID!r}."
            )
    else:
        raise ValueError(
            "Unsupported required_distance_protocol_id "
            f"{required_distance_protocol_id!r}."
        )
    maps = []
    occupancies = []
    dumpability_masks_init = []
    trench_axes = []
    trench_types = []
    trench_axis_owners = []
    foundation_border_axes = []
    foundation_border_types = []
    actions = []
    distances = []
    n_loaded_metadata = 0
    # Check if the actions folder exists (only for relocations)
    actions_folder = Path(folder_path) / "actions"
    has_actions = actions_folder.exists()
    # Track if we found any distance maps at all
    found_any_distance = False
    images_dir = Path(folder_path) / "images"
    image_files = sorted(images_dir.glob("img_*.npy"))
    if not image_files:
        raise RuntimeError(f"No image files found in {images_dir}; expected img_*.npy.")

    available_indices = []
    for f in image_files:
        stem = f.stem
        try:
            available_indices.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    available_indices = sorted(available_indices)
    if not available_indices:
        raise RuntimeError(f"Could not parse any numeric indices from {images_dir}/img_*.npy.")

    expected_indices = list(range(1, dataset_size + 1))
    if available_indices != expected_indices:
        raise RuntimeError(
            f"{images_dir} must enumerate exactly contiguous indices "
            f"1..{dataset_size}; got {available_indices[:8]}."
        )
    selected_indices = expected_indices

    for i in tqdm(selected_indices, desc="Data Loader"):
        image_path = Path(folder_path) / "images" / f"img_{i}.npy"
        map = _ensure_spatial_2d(np.load(image_path), str(image_path))
        map_sanity_check(map)
        if expected_shape is not None and map.shape != expected_shape:
            raise RuntimeError(
                f"Target map shape mismatch for {image_path}: "
                f"expected {expected_shape}, got {map.shape}."
            )
        occupancy_path = Path(folder_path) / "occupancy" / f"img_{i}.npy"
        occupancy = _ensure_spatial_2d(np.load(occupancy_path), str(occupancy_path))
        occupancy_sanity_check(occupancy)
        if occupancy.shape != map.shape:
            raise RuntimeError(
                f"Occupancy shape mismatch for {occupancy_path}: "
                f"expected {map.shape}, got {occupancy.shape}."
            )
        dumpability_path = Path(folder_path) / "dumpability" / f"img_{i}.npy"
        dumpability_mask_init = _ensure_spatial_2d(
            np.load(dumpability_path), str(dumpability_path)
        )
        dumpability_sanity_check(dumpability_mask_init)
        if dumpability_mask_init.shape != map.shape:
            raise RuntimeError(
                f"Dumpability shape mismatch for {dumpability_path}: "
                f"expected {map.shape}, got {dumpability_mask_init.shape}."
            )
        maps.append(map)
        occupancies.append(occupancy)
        dumpability_masks_init.append(dumpability_mask_init)
        # Generate an actions map if present, otherwise initialize an empty one
        if has_actions:
            actions_path = actions_folder / f"img_{i}.npy"
            actions_map = _ensure_spatial_2d(np.load(actions_path), str(actions_path))
            actions_sanity_check(actions_map)
            if actions_map.shape != map.shape:
                raise RuntimeError(
                    f"Actions shape mismatch for {actions_path}: "
                    f"expected {map.shape}, got {actions_map.shape}."
                )
            actions.append(actions_map)
        else:
            actions.append(np.zeros_like(map, dtype=IntMap))
        if (
            required_distance_protocol_id == REWARD_V2_DISTANCE_PROTOCOL_ID
            and np.any(actions[-1] != 0)
        ):
            raise RuntimeError(
                "Reward-v2 R2 supports full-reset maps only; initial action "
                f"map is nonzero for slot {i}."
            )
        contained_dump_capacity_sanity_check(
            map,
            occupancy,
            dumpability_mask_init,
            actions[-1],
            minimum_single_layer_ratio=minimum_dump_capacity_ratio,
        )

        # Dense-reward distance maps are part of the map contract. A missing,
        # malformed, or silently zero-filled field changes the reward while
        # pretending to preserve the map treatment, so fail at the loader
        # boundary.
        distance_file = Path(folder_path) / "distance" / f"img_{i}.npy"
        if not distance_file.exists():
            raise RuntimeError(f"Missing required distance map: {distance_file}")
        dist_map = _ensure_spatial_2d(
            np.load(distance_file), str(distance_file)
        )
        if dist_map.shape != map.shape:
            raise RuntimeError(
                f"Distance map shape mismatch for {distance_file}: "
                f"expected {map.shape}, got {dist_map.shape}."
            )
        if not np.issubdtype(dist_map.dtype, np.floating):
            raise RuntimeError(
                f"Distance map must use a floating dtype: {distance_file} "
                f"has {dist_map.dtype}."
            )
        if not np.all(np.isfinite(dist_map)):
            raise RuntimeError(
                f"Distance map contains non-finite values: {distance_file}"
            )
        minimum = float(np.min(dist_map))
        maximum = float(np.max(dist_map))
        if reward_v2_distance is None:
            if minimum < 0.0 or maximum > 1.0:
                raise RuntimeError(
                    f"Legacy distance map must be normalized to [0, 1]: "
                    f"{distance_file} has min={minimum}, max={maximum}."
                )
        else:
            expected_distance = compute_reward_v2_distance_map(
                map,
                occupancy,
                tile_size_m=reward_v2_distance["tile_size_m"],
                distance_ref_m=reward_v2_distance["distance_ref_m"],
                distance_bound=reward_v2_distance["distance_bound"],
            )
            observed_distance = dist_map.astype(np.float32)
            if not np.array_equal(observed_distance, expected_distance):
                max_error = float(
                    np.max(np.abs(observed_distance - expected_distance))
                )
                raise RuntimeError(
                    "R2 distance sidecar is not the canonical function of its "
                    f"visible target/obstacle maps: {distance_file}; "
                    f"max_abs_error={max_error:.9g}."
                )
        found_any_distance = True
        distances.append(dist_map.astype(np.float32))

        owners_path = (
            Path(folder_path)
            / "trench_axis_owners"
            / f"img_{i}.npy"
        )
        owners = (
            _ensure_spatial_2d(np.load(owners_path), str(owners_path))
            if owners_path.exists()
            else np.zeros_like(map, dtype=np.uint8)
        )

        # Metadata needs to be loaded only for trenches (A, B, C coefficients)
        metadata_path = Path(folder_path) / "metadata" / f"trench_{i}.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            trench_ax, trench_type = _trench_records_from_metadata(
                metadata,
                max_trench_type,
            )
            if trench_type > 0 and not owners_path.exists():
                raise RuntimeError(
                    f"Missing required trench owner sidecar: {owners_path}"
                )
            trench_axis_contract_sanity_check(metadata, owners, trench_type)

            trench_axes.append(trench_ax)
            trench_types.append(trench_type)
            trench_axis_owners.append(
                trench_axis_owners_sanity_check(
                    map,
                    owners,
                    trench_type,
                    max_trench_type,
                )
            )
            foundation_ax = metadata.get("foundation_border_axes_ABC", [])
            foundation_border_type = -1
            if len(foundation_ax) > 0:
                metadata_sanity_check(foundation_ax[0])
                foundation_ax = [[el["A"], el["B"], el["C"]] for el in foundation_ax]
                foundation_border_type = len(foundation_ax)
            if len(foundation_ax) > max_foundation_border_type:
                foundation_ax = foundation_ax[:max_foundation_border_type]
                foundation_border_type = max_foundation_border_type
            while len(foundation_ax) < max_foundation_border_type:
                foundation_ax.append([-97, -97, -97])
            foundation_border_axes.append(foundation_ax)
            foundation_border_types.append(foundation_border_type)
            n_loaded_metadata += 1
        else:
            # Missing metadata for this map.
            if require_trench_metadata:
                # If the curriculum/level requires trench metadata (e.g. trench rewards),
                # raise immediately so the user can provide the metadata.
                raise RuntimeError(
                    f"Missing trench metadata file {metadata_path} but trench metadata is required for this level."
                )
            # Otherwise use defaults and continue. If some metadata were loaded
            # earlier this is allowed (we treat missing as defaults).
            if n_loaded_metadata > 0:
                print(f"Warning: missing metadata file {metadata_path}; using defaults for img_{i}.")
            trench_axes.append(
                [
                    [-97] * TRENCH_AXIS_RECORD_SIZE
                    for _ in range(max_trench_type)
                ]
            )
            trench_types.append(-1)
            trench_axis_owners.append(
                trench_axis_owners_sanity_check(
                    map,
                    owners,
                    -1,
                    max_trench_type,
                )
            )
            foundation_border_axes.append(
                [[-97, -97, -97] for _ in range(max_foundation_border_type)]
            )
            foundation_border_types.append(-1)
            continue
    # If no distance maps were found at all, raise an error (strict behavior)
    if not found_any_distance:
        raise RuntimeError(f"No distance maps found in {Path(folder_path) / 'distance'}; please provide distance/img_*.npy files.")
    loaded_count = len(maps)
    print(f"Loaded {loaded_count} maps from {folder_path}.")
    if n_loaded_metadata > 0:
        print(f"Loaded {n_loaded_metadata} metadata files from {folder_path}.")
    else:
        trench_axes = -97.0 * jnp.ones(
            (
                loaded_count,
                max_trench_type,
                TRENCH_AXIS_RECORD_SIZE,
            )
        )
        trench_types = -1 * jnp.ones((loaded_count,), dtype=jnp.int32)
        foundation_border_axes = -97.0 * jnp.ones(
            (
                loaded_count,
                max_foundation_border_type,
                3,
            )
        )
        foundation_border_types = -1 * jnp.ones((loaded_count,), dtype=jnp.int32)
        print(f"Did NOT load any metadata file from {folder_path}.")
    return (
        jnp.array(maps, dtype=IntMap),
        jnp.array(occupancies, dtype=IntMap),
        jnp.array(trench_axes),
        jnp.array(trench_types, dtype=jnp.int32),
        jnp.array(trench_axis_owners, dtype=jnp.uint8),
        jnp.array(foundation_border_axes),
        jnp.array(foundation_border_types, dtype=jnp.int32),
        jnp.array(dumpability_masks_init, dtype=jnp.bool_),
        jnp.array(actions, dtype=IntMap),
        jnp.array(distances, dtype=jnp.float32),
    )


def _pad_map_array(m: Array, max_w: int, max_h: int) -> tuple[Array, Array]:
    """
    Pads the map array to dimensions (max_w, max_h) and returns the padded map and padding mask.

    Args:
        m (Array): The input map array.
        max_w (int): The maximum width for padding.
        max_h (int): The maximum height for padding.
    Returns:
        Tuple[Array, Array]: Padded map and padding mask.
    """

    z = np.zeros((m.shape[0], max_w, max_h), dtype=IntMap)
    z_mask = np.ones((m.shape[0], max_w, max_h), dtype=IntMap)  # 1 for obstacles
    z[:, : m.shape[1], : m.shape[2]] = m
    # Set mask to zero where original map is present
    z_mask[:, : m.shape[1], : m.shape[2]] = np.zeros_like(m)  # 0 for free
    return z, z_mask


def _pad_maps(
    maps: list[Array],
    occupancies: list[Array],
    dumpability_masks: list[Array],
    actions: list[Array],
    maps_width,
    maps_height,
):
    """
    Pads multiple maps along with their occupancies and dumpability masks.

    Args:
    maps (List[Array]): List of map arrays.
    occupancies (List[Array]): List of occupancy arrays.
    dumpability_masks (List[Array]): List of dumpability mask arrays.
    actions (List[Array]): List of action arrays.
    maps_width (int): Maximum width for padding.
    maps_height (int): Maximum height for padding.

    Returns:
    Tuple[Array, Array, Array]: Padded maps, padding masks, and padded dumpability masks.
    """
    max_w = maps_width
    max_h = maps_height
    padding_mask = []
    maps_padded = []
    dumpability_masks_padded = []
    actions_padded = []
    for m, o, d, a in zip(maps, occupancies, dumpability_masks, actions):
        z, z_mask = _pad_map_array(m, max_w, max_h)
        z_mask[:, : o.shape[1], : o.shape[2]] = o  # use occupancies from dataset
        d_padded = np.zeros(
            (d.shape[0], max_w, max_h),
            dtype=np.bool_,
        )
        d_padded[:, : d.shape[1], : d.shape[2]] = d

        a_padded = np.zeros(
            (a.shape[0], max_w, max_h),
            dtype=IntMap,
        )
        a_padded[:, : a.shape[1], : a.shape[2]] = a

        maps_padded.append(z)
        padding_mask.append(z_mask)
        dumpability_masks_padded.append(d_padded)
        actions_padded.append(a_padded)
    return (
        np.array(maps_padded, dtype=IntMap),
        np.array(padding_mask, dtype=IntMap),
        np.array(dumpability_masks_padded, dtype=jnp.bool_),
        np.array(actions_padded, dtype=IntMap),
    )


def _check_maps(maps: list[Array]) -> tuple[int, int]:
    """
    Checks if the maps have the same dimensions and returns them.

    Args:
    maps (List[Array]): List of map arrays.

    Returns:
    Tuple[int, int]: Width and height of the maps.
    """
    maps_width = maps[0].shape[1]
    maps_height = maps[0].shape[2]
    print(f"Maps width: {maps_width}")
    print(f"Maps height: {maps_height}")
    for m in maps:
        if m.shape[1] != maps_width or m.shape[2] != maps_height:
            raise ValueError("Maps have different dimensions.")
    return maps_width, maps_height


def init_maps_buffer(
    batch_cfg: BatchConfig,
    shuffle_maps: bool,
    single_map_path: str = None,
    required_distance_protocol_id: str = LEGACY_DISTANCE_PROTOCOL_ID,
    partial_reset_root: str | Path | None = None,
):
    manifest_rows_per_level: list[list[dict[str, Any]]] | None = None
    if single_map_path is not None:
        if partial_reset_root is not None:
            raise RuntimeError(
                "Partial-reset sidecars require the canonical dataset loader, "
                "not single_map_path."
            )
        if required_distance_protocol_id != LEGACY_DISTANCE_PROTOCOL_ID:
            raise RuntimeError(
                "Reward-v2 accepts only exact datasets with validated physical "
                "distance sidecars, not the legacy single-map loader."
            )
        print(f"Loading single map from {single_map_path}")
        maps_from_disk = []
        occupancies_from_disk = []
        dumpability_masks_init_from_disk = []
        trench_axes_list = []
        trench_types = []
        trench_axis_owners_list = []
        foundation_border_axes_list = []
        foundation_border_types = []
        actions_from_disk = []
        distances_from_disk = []

        # Load the single map (now also returns a distance map)
        (
            maps,
            occupancies,
            trench_axes,
            trench_type,
            trench_axis_owners,
            foundation_border_axes,
            foundation_border_type,
            dumpability_masks_init,
            actions,
            distances,
        ) = load_single_map(single_map_path)

        # Repeat the map for each curriculum level
        num_levels = len(batch_cfg.curriculum_global.levels)
        maps_from_disk = [maps] * num_levels
        occupancies_from_disk = [occupancies] * num_levels
        dumpability_masks_init_from_disk = [dumpability_masks_init] * num_levels
        trench_axes_list = [trench_axes] * num_levels
        trench_types = [trench_type] * num_levels
        trench_axis_owners_list = [trench_axis_owners] * num_levels
        foundation_border_axes_list = [foundation_border_axes] * num_levels
        foundation_border_types = [jnp.array([foundation_border_type], dtype=jnp.int32)] * num_levels
        actions_from_disk = [actions] * num_levels
        distances_from_disk = [distances] * num_levels
    else:
        if os.getenv("DATASET_PATH", "") == "":
            raise RuntimeError("DATASET_PATH not defined, can't load maps from disk.")
        maps_paths = [el["maps_path"] for el in batch_cfg.curriculum_global.levels]
        folder_paths = [str(Path(os.getenv("DATASET_PATH", "")) / el) for el in maps_paths]
        print(f"Loading maps from {folder_paths}.")
        maps_from_disk = []
        occupancies_from_disk = []
        dumpability_masks_init_from_disk = []
        trench_axes_list = []
        trench_types = []
        trench_axis_owners_list = []
        foundation_border_axes_list = []
        foundation_border_types = []
        actions_from_disk = []
        distances_from_disk = []
        manifest_rows_per_level = []
        for idx, folder_path in enumerate(folder_paths):
            (
                maps,
                occupancies,
                trench_axes,
                trench_types_for_level,
                trench_axis_owners,
                foundation_border_axes,
                foundation_border_type,
                dumpability_masks_init,
                actions,
                distances,
            ) = load_maps_from_disk(
                folder_path,
                require_trench_metadata=(
                    batch_cfg.curriculum_global.levels[idx].get(
                        "apply_trench_rewards", False
                    )
                ),
                required_distance_protocol_id=required_distance_protocol_id,
            )
            maps_from_disk.append(maps)
            occupancies_from_disk.append(occupancies)
            dumpability_masks_init_from_disk.append(dumpability_masks_init)
            trench_axes_list.append(trench_axes)
            trench_types.append(trench_types_for_level)
            trench_axis_owners_list.append(trench_axis_owners)
            foundation_border_axes_list.append(foundation_border_axes)
            foundation_border_types.append(foundation_border_type)
            actions_from_disk.append(actions)
            distances_from_disk.append(distances)
            manifest_rows_per_level.append(
                _load_json_lines(Path(folder_path) / EXACT_DATASET_MANIFEST)
            )

    partial_action_maps = None
    partial_action_available = None
    partial_reset_bank_digest = ""
    partial_reset_supported_levels = np.zeros(
        (
            len(PARTIAL_RESET_FRACTIONS) + 1,
            len(batch_cfg.curriculum_global.levels),
        ),
        dtype=np.bool_,
    )
    partial_reset_supported_levels[0] = True
    if partial_reset_root is not None:
        if manifest_rows_per_level is None:
            raise RuntimeError(
                "Partial resets require canonical manifest rows for source identity."
            )
        (
            partial_action_maps,
            partial_action_available,
            partial_reset_bank_digest,
            partial_reset_supported_levels,
        ) = load_partial_reset_action_sidecars(
            partial_reset_root,
            maps_paths,
            maps_from_disk,
            occupancies_from_disk,
            dumpability_masks_init_from_disk,
            manifest_rows_per_level,
        )

    if manifest_rows_per_level is None:
        family_names = ("unknown",)
        primary_cell_names = ("unknown",)
        slot_indices = np.broadcast_to(
            np.arange(maps_from_disk[0].shape[0], dtype=np.int32),
            (len(maps_from_disk), maps_from_disk[0].shape[0]),
        ).copy()
        family_ids = np.zeros_like(slot_indices)
        primary_cell_ids = np.zeros_like(slot_indices)
    else:
        family_names = (
            "unknown",
            *sorted(
                {
                    row["family"]
                    for rows in manifest_rows_per_level
                    for row in rows
                }
            ),
        )
        primary_cell_names = (
            "unknown",
            *sorted(
                {
                    row["primary_cell"]
                    for rows in manifest_rows_per_level
                    for row in rows
                }
            ),
        )
        family_lookup = {
            name: index for index, name in enumerate(family_names)
        }
        primary_cell_lookup = {
            name: index
            for index, name in enumerate(primary_cell_names)
        }
        slot_indices = np.asarray(
            [
                [int(row["slot_index"]) - 1 for row in rows]
                for rows in manifest_rows_per_level
            ],
            dtype=np.int32,
        )
        family_ids = np.asarray(
            [
                [family_lookup[row["family"]] for row in rows]
                for rows in manifest_rows_per_level
            ],
            dtype=np.int32,
        )
        primary_cell_ids = np.asarray(
            [
                [
                    primary_cell_lookup[row["primary_cell"]]
                    for row in rows
                ]
                for rows in manifest_rows_per_level
            ],
            dtype=np.int32,
        )

    # Apply padding to ALL maps (unified logic like single-agent)
    maps_width, maps_height = _check_maps(maps_from_disk)
    maps_from_disk_padded, padding_mask, dumpability_masks_init_from_disk_padded, actions_from_disk_padded = _pad_maps(
        maps_from_disk,
        occupancies_from_disk,
        dumpability_masks_init_from_disk,
        actions_from_disk,
        maps_width,
        maps_height,
    )
    # Distance maps don't require padding masks; pad to same shape
    distances_padded = []
    for d in distances_from_disk:
        z = np.zeros((d.shape[0], maps_width, maps_height), dtype=np.float32)
        z[:, : d.shape[1], : d.shape[2]] = d
        distances_padded.append(z)
    trench_axis_owners_padded = []
    for owners in trench_axis_owners_list:
        z = np.zeros(
            (owners.shape[0], maps_width, maps_height),
            dtype=np.uint8,
        )
        z[:, : owners.shape[1], : owners.shape[2]] = owners
        trench_axis_owners_padded.append(z)

    unique_shapes = set([trench_axes.shape for trench_axes in trench_axes_list])
    print(f"Unique shapes of trench_axes_list: {unique_shapes}")

    maps_from_disk_padded = jnp.array(maps_from_disk_padded)
    padding_mask = jnp.array(padding_mask)
    dumpability_masks_init_from_disk = jnp.array(dumpability_masks_init_from_disk_padded)
    trench_axes_list = jnp.array(trench_axes_list)
    trench_types = jnp.array(trench_types)
    trench_axis_owners = jnp.array(
        trench_axis_owners_padded,
        dtype=jnp.uint8,
    )
    foundation_border_axes_list = jnp.array(foundation_border_axes_list)
    foundation_border_types = jnp.array(foundation_border_types)
    actions_from_disk_padded = jnp.array(actions_from_disk_padded)
    if partial_action_maps is None:
        partial_action_maps = jnp.zeros(
            (0, *maps_from_disk_padded.shape),
            dtype=IntLowDim,
        )
        partial_action_available = jnp.zeros(
            (0, *maps_from_disk_padded.shape[:2]),
            dtype=jnp.bool_,
        )
    else:
        partial_action_maps = jnp.asarray(partial_action_maps, dtype=IntLowDim)
        partial_action_available = jnp.asarray(
            partial_action_available,
            dtype=jnp.bool_,
        )
    distances_padded = jnp.array(distances_padded)
    slot_indices = jnp.array(slot_indices, dtype=jnp.int32)
    family_ids = jnp.array(family_ids, dtype=jnp.int32)
    primary_cell_ids = jnp.array(
        primary_cell_ids,
        dtype=jnp.int32,
    )
    print(f"Maps shape: {maps_from_disk_padded.shape}.")
    print(f"Padding mask shape: {padding_mask.shape}.")
    print(f"Dumpability mask shape: {dumpability_masks_init_from_disk.shape}.")
    print(f"Trench axes shape: {trench_axes_list.shape}.")
    print(f"Trench types shape: {trench_types.shape}.")
    print(f"Trench owner maps shape: {trench_axis_owners.shape}.")
    print(f"Foundation border axes shape: {foundation_border_axes_list.shape}.")
    print(f"Foundation border types shape: {foundation_border_types.shape}.")
    print(f"Actions shape: {actions_from_disk_padded.shape}.")
    print(f"Distance maps shape: {distances_padded.shape}.")
    print(f"Map provenance shape: {slot_indices.shape}.")
    if shuffle_maps:
        # NOTE: this is only for visualization purposes (allows to visualize in a single gif every level of the curriculum)
        print("Shuffling maps between curriculum levels...")
        rng = jax.random.PRNGKey(3333)  # doesn't matter which key, it's used only once
        d0 = maps_from_disk_padded.shape[0]
        d1 = maps_from_disk_padded.shape[1]
        # Reshape
        maps_from_disk_padded = maps_from_disk_padded.reshape(
            (-1, *maps_from_disk_padded.shape[2:])
        )
        padding_mask = padding_mask.reshape((-1, *padding_mask.shape[2:]))
        dumpability_masks_init_from_disk = dumpability_masks_init_from_disk.reshape(
            (-1, *dumpability_masks_init_from_disk.shape[2:])
        )
        trench_axes_list = trench_axes_list.reshape((-1, *trench_axes_list.shape[2:]))
        trench_types = trench_types.reshape((-1,))
        trench_axis_owners = trench_axis_owners.reshape(
            (-1, *trench_axis_owners.shape[2:])
        )
        foundation_border_axes_list = foundation_border_axes_list.reshape(
            (-1, *foundation_border_axes_list.shape[2:])
        )
        foundation_border_types = foundation_border_types.reshape((-1,))
        actions_from_disk_padded = actions_from_disk_padded.reshape(
            (-1, *actions_from_disk_padded.shape[2:])
        )
        if partial_action_maps.shape[0] > 0:
            partial_action_maps = partial_action_maps.reshape(
                (partial_action_maps.shape[0], -1, *partial_action_maps.shape[3:])
            )
            partial_action_available = partial_action_available.reshape(
                (partial_action_available.shape[0], -1)
            )
        distances_padded = distances_padded.reshape((-1, *distances_padded.shape[2:]))
        slot_indices = slot_indices.reshape((-1,))
        family_ids = family_ids.reshape((-1,))
        primary_cell_ids = primary_cell_ids.reshape((-1,))
        # Shuffle
        maps_from_disk_padded = jax.random.permutation(
            rng, maps_from_disk_padded, axis=0
        )
        padding_mask = jax.random.permutation(rng, padding_mask, axis=0)
        dumpability_masks_init_from_disk = jax.random.permutation(
            rng, dumpability_masks_init_from_disk, axis=0
        )
        trench_axes_list = jax.random.permutation(rng, trench_axes_list, axis=0)
        trench_types = jax.random.permutation(rng, trench_types, axis=0)
        trench_axis_owners = jax.random.permutation(
            rng,
            trench_axis_owners,
            axis=0,
        )
        foundation_border_axes_list = jax.random.permutation(
            rng, foundation_border_axes_list, axis=0
        )
        foundation_border_types = jax.random.permutation(
            rng, foundation_border_types, axis=0
        )
        actions_from_disk_padded = jax.random.permutation(
            rng, actions_from_disk_padded, axis=0
        )
        if partial_action_maps.shape[0] > 0:
            partial_action_maps = jax.vmap(
                lambda values: jax.random.permutation(rng, values, axis=0)
            )(partial_action_maps)
            partial_action_available = jax.vmap(
                lambda values: jax.random.permutation(rng, values, axis=0)
            )(partial_action_available)
        distances_padded = jax.random.permutation(
            rng, distances_padded, axis=0
        )
        slot_indices = jax.random.permutation(rng, slot_indices, axis=0)
        family_ids = jax.random.permutation(rng, family_ids, axis=0)
        primary_cell_ids = jax.random.permutation(
            rng,
            primary_cell_ids,
            axis=0,
        )
        # Reshape back
        maps_from_disk_padded = maps_from_disk_padded.reshape(
            (d0, d1, *maps_from_disk_padded.shape[1:])
        )
        padding_mask = padding_mask.reshape((d0, d1, *padding_mask.shape[1:]))
        dumpability_masks_init_from_disk = dumpability_masks_init_from_disk.reshape(
            (d0, d1, *dumpability_masks_init_from_disk.shape[1:])
        )
        trench_axes_list = trench_axes_list.reshape(
            (d0, d1, *trench_axes_list.shape[1:])
        )
        trench_types = trench_types.reshape((d0, d1))
        trench_axis_owners = trench_axis_owners.reshape(
            (d0, d1, *trench_axis_owners.shape[1:])
        )
        foundation_border_axes_list = foundation_border_axes_list.reshape(
            (d0, d1, *foundation_border_axes_list.shape[1:])
        )
        foundation_border_types = foundation_border_types.reshape((d0, d1))
        actions_from_disk_padded = actions_from_disk_padded.reshape(
            (d0, d1, *actions_from_disk_padded.shape[1:])
        )
        if partial_action_maps.shape[0] > 0:
            partial_action_maps = partial_action_maps.reshape(
                (
                    partial_action_maps.shape[0],
                    d0,
                    d1,
                    *partial_action_maps.shape[2:],
                )
            )
            partial_action_available = partial_action_available.reshape(
                (partial_action_available.shape[0], d0, d1)
            )
            partial_reset_supported_levels = np.concatenate(
                (
                    np.ones((1, d0), dtype=np.bool_),
                    np.asarray(jnp.any(partial_action_available, axis=2)),
                ),
                axis=0,
            )
        distances_padded = distances_padded.reshape(
            (d0, d1, *distances_padded.shape[1:])
        )
        slot_indices = slot_indices.reshape((d0, d1))
        family_ids = family_ids.reshape((d0, d1))
        primary_cell_ids = primary_cell_ids.reshape((d0, d1))
        print("Maps shuffled.")
    maps_buffer = MapsBuffer.new(
        maps=maps_from_disk_padded,
        padding_mask=padding_mask,
        trench_axes=trench_axes_list,
        trench_types=trench_types,
        trench_axis_owners=trench_axis_owners,
        foundation_border_axes=foundation_border_axes_list,
        foundation_border_types=foundation_border_types,
        dumpability_masks_init=dumpability_masks_init_from_disk,
        action_maps=actions_from_disk_padded,
        distance_maps=distances_padded,
        slot_indices=slot_indices,
        family_ids=family_ids,
        primary_cell_ids=primary_cell_ids,
        family_names=family_names,
        primary_cell_names=primary_cell_names,
        partial_action_maps=partial_action_maps,
        partial_action_available=partial_action_available,
        partial_reset_supported_levels=partial_reset_supported_levels,
        partial_reset_bank_sha256=partial_reset_bank_digest,
    )
    # Update batch config with the actual map dimensions
    maps_width = maps_from_disk_padded.shape[2]
    maps_height = maps_from_disk_padded.shape[3]
    assert maps_width == maps_height, "Maps are not square."
    batch_cfg = batch_cfg._replace(
        maps_dims=batch_cfg.maps_dims._replace(maps_edge_length=maps_width)
    )
    return maps_buffer, batch_cfg
