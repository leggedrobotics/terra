"""Materialize sparse relay-reset action sidecars over an exact Terra bank."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from terra.config import PARTIAL_RESET_FRACTIONS
from terra.env_generation.partial_completion import PartialCompletionConfig
from terra.env_generation.partial_completion import PartialCompletionError
from terra.env_generation.partial_completion import _load_source_layers
from terra.env_generation.partial_completion import generate_partial_action_map
from terra.maps_buffer import LEGACY_SCENARIO_IDENTITY_CONTRACT
from terra.maps_buffer import PARTIAL_COMPLETION_CONFIG
from terra.maps_buffer import PARTIAL_COMPLETION_MANIFEST
from terra.maps_buffer import PARTIAL_COMPLETION_REJECTIONS
from terra.maps_buffer import PARTIAL_RESET_BANK_INDEX
from terra.maps_buffer import PARTIAL_RESET_BANK_SCHEMA
from terra.maps_buffer import PARTIAL_RESET_LEAF_SCHEMA
from terra.maps_buffer import PARTIAL_RESET_TRIPLET_CONTRACT
from terra.maps_buffer import RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT
from terra.maps_buffer import contained_dump_capacity_sanity_check
from terra.maps_buffer import partial_reset_action_sanity_check
from terra.maps_buffer import partial_reset_bank_sha256
from terra.maps_buffer import partial_reset_triplet_sanity_check
from terra.maps_buffer import validate_exact_dataset_contract


CURRICULUM_LOADER_BANK_SCHEMA = "terra_curriculum_loader_bank_v1"


def _json_line(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True))
            stream.write("\n")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declared_training_leaves(root: Path) -> list[tuple[str, Path]]:
    """Read the canonical ordered training registry; never infer leaves by scan."""
    index_path = root / "dataset.json"
    if not index_path.is_file():
        raise PartialCompletionError(
            f"Accepted loader bank is missing its root registry: {index_path}"
        )
    try:
        registry = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        raise PartialCompletionError(f"Invalid JSON in {index_path}: {exc}") from exc
    if registry.get("schema") != CURRICULUM_LOADER_BANK_SCHEMA:
        raise PartialCompletionError(
            f"{index_path} must use schema {CURRICULUM_LOADER_BANK_SCHEMA!r}."
        )
    declared = registry.get("train")
    if not isinstance(declared, list) or not declared:
        raise PartialCompletionError(f"{index_path} has no declared training levels.")

    root_resolved = root.resolve()
    leaves: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for expected_level, row in enumerate(declared):
        if not isinstance(row, dict):
            raise PartialCompletionError(
                f"{index_path} train[{expected_level}] must be an object."
            )
        maps_path = row.get("maps_path")
        if (
            not isinstance(maps_path, str)
            or not maps_path
            or Path(maps_path).is_absolute()
            or row.get("level_index") != expected_level
        ):
            raise PartialCompletionError(
                f"{index_path} train[{expected_level}] has invalid level_index/maps_path."
            )
        source_directory = (root / maps_path).resolve()
        if root_resolved not in source_directory.parents or maps_path in seen:
            raise PartialCompletionError(
                f"{index_path} declares unsafe or duplicate maps_path {maps_path!r}."
            )
        if not (source_directory / "dataset.json").is_file():
            raise PartialCompletionError(
                f"Declared training leaf is missing dataset.json: {source_directory}"
            )
        seen.add(maps_path)
        leaves.append((maps_path, source_directory))
    return leaves


def materialize_sparse_partial_reset_bank(
    input_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 0,
    max_attempts_per_variant: int = 100,
    min_spawn_centers: int = 16,
) -> dict[str, Any]:
    """Generate one sparse action-only relay bank over every exact condition."""
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if not input_root.is_dir():
        raise PartialCompletionError(f"Input root does not exist: {input_root}")
    if output_root.exists():
        raise PartialCompletionError(f"Output path already exists: {output_root}")
    leaves = _declared_training_leaves(input_root)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    supported_paths: list[str] = []
    root_rejections: list[dict[str, Any]] = []
    try:
        for maps_path, source_directory in leaves:
            dataset_metadata = json.loads(
                (source_directory / "dataset.json").read_text()
            )
            if (
                dataset_metadata.get("scenario_identity_contract")
                != RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT
            ):
                observed = dataset_metadata.get("scenario_identity_contract")
                if observed == LEGACY_SCENARIO_IDENTITY_CONTRACT:
                    raise PartialCompletionError(
                        f"{maps_path} uses legacy source identities; strict scenario "
                        "IDs are required for partial resets."
                    )
                raise PartialCompletionError(
                    f"{maps_path} has unsupported scenario identity {observed!r}."
                )
            slot_count = dataset_metadata.get("slot_count")
            if not isinstance(slot_count, int) or slot_count <= 0:
                raise PartialCompletionError(
                    f"{source_directory / 'dataset.json'} has invalid slot_count."
                )
            manifest_rows, _, _ = validate_exact_dataset_contract(
                source_directory,
                slot_count,
            )

            leaf_directory = temporary_root / maps_path
            actions_directory = leaf_directory / "actions"
            actions_directory.mkdir(parents=True, exist_ok=True)
            success_rows: list[dict[str, Any]] = []
            rejection_rows: list[dict[str, Any]] = []
            tier_counts = np.zeros((len(PARTIAL_RESET_FRACTIONS),), dtype=np.int32)
            rejected_source_count = 0
            sidecar_index = 0
            condition_seed = int.from_bytes(
                hashlib.sha256(maps_path.encode("utf-8")).digest()[:4],
                "little",
            )
            for source_index, source_row in enumerate(manifest_rows, start=1):
                target, occupancy, dumpability = _load_source_layers(
                    source_directory,
                    source_index,
                )
                variant_seed = int(
                    np.random.SeedSequence(
                        [seed, condition_seed, source_index]
                    ).generate_state(1, dtype=np.uint64)[0]
                )
                triplet_results: list[tuple[int, float, Any]] = []
                for tier_index, fraction in enumerate(
                    PARTIAL_RESET_FRACTIONS,
                    start=1,
                ):
                    config = PartialCompletionConfig(
                        completion_fractions=(fraction,),
                        variants_per_fraction=1,
                        mode_weights=(("relay_corridor", 1.0),),
                        min_piles=1,
                        max_piles=1,
                        min_spawn_centers=min_spawn_centers,
                        max_attempts_per_variant=max_attempts_per_variant,
                        seed=variant_seed,
                    )
                    try:
                        result = generate_partial_action_map(
                            target,
                            occupancy,
                            dumpability,
                            rng=np.random.default_rng(variant_seed),
                            config=config,
                        )
                        partial_reset_action_sanity_check(
                            target,
                            occupancy,
                            dumpability,
                            result.action_map,
                            expected_fraction=fraction,
                        )
                        contained_dump_capacity_sanity_check(
                            target,
                            occupancy,
                            dumpability,
                            result.action_map,
                        )
                    except (PartialCompletionError, RuntimeError) as exc:
                        rejection_rows.append(
                            {
                                "maps_path": maps_path,
                                "source_index": source_index,
                                "source_map_id": source_row["map_id"],
                                "source_scenario_id": source_row["scenario_id"],
                                "reset_tier": tier_index,
                                "requested_completion_fraction": fraction,
                                "variant_seed": variant_seed,
                                "error": str(exc),
                            }
                        )
                        continue
                    triplet_results.append((tier_index, fraction, result))

                if len(triplet_results) != len(PARTIAL_RESET_FRACTIONS):
                    rejected_source_count += 1
                    rejection_rows.append(
                        {
                            "maps_path": maps_path,
                            "source_index": source_index,
                            "source_map_id": source_row["map_id"],
                            "source_scenario_id": source_row["scenario_id"],
                            "variant_seed": variant_seed,
                            "successful_tiers_discarded": [
                                tier_index
                                for tier_index, _, _ in triplet_results
                            ],
                            "error": "discarded incomplete source triplet",
                        }
                    )
                    continue

                try:
                    partial_reset_triplet_sanity_check(
                        triplet_results[0][2].action_map,
                        triplet_results[1][2].action_map,
                        triplet_results[2][2].action_map,
                    )
                except RuntimeError as exc:
                    rejected_source_count += 1
                    rejection_rows.append(
                        {
                            "maps_path": maps_path,
                            "source_index": source_index,
                            "source_map_id": source_row["map_id"],
                            "source_scenario_id": source_row["scenario_id"],
                            "variant_seed": variant_seed,
                            "successful_tiers_discarded": [1, 2, 3],
                            "error": str(exc),
                        }
                    )
                    continue

                for tier_index, _, result in triplet_results:
                    sidecar_index += 1
                    action_path = actions_directory / f"img_{sidecar_index}.npy"
                    np.save(action_path, result.action_map.astype(np.int8))
                    success_rows.append(
                        {
                            "sidecar_index": sidecar_index,
                            "maps_path": maps_path,
                            "source_index": source_index,
                            "source_map_id": source_row["map_id"],
                            "source_scenario_id": source_row["scenario_id"],
                            "reset_tier": tier_index,
                            "variant_seed": variant_seed,
                            "action_sha256": _file_sha256(action_path),
                            **result.manifest,
                        }
                    )
                    tier_counts[tier_index - 1] += 1

            if not np.all(tier_counts > 0):
                root_rejections.extend(rejection_rows)
                root_rejections.append(
                    {
                        "maps_path": maps_path,
                        "error": "condition lacks a successful source in every tier",
                        "tier_success_counts": tier_counts.tolist(),
                    }
                )
                shutil.rmtree(leaf_directory)
                continue

            leaf_config = {
                "schema": PARTIAL_RESET_LEAF_SCHEMA,
                "maps_path": maps_path,
                "pile_mode": "relay_corridor",
                "source_triplet_contract": PARTIAL_RESET_TRIPLET_CONTRACT,
                "completion_fractions": list(PARTIAL_RESET_FRACTIONS),
                "canonical_slot_count": slot_count,
                "canonical_dataset_sha256": _file_sha256(
                    source_directory / "dataset.json"
                ),
                "canonical_manifest_sha256": _file_sha256(
                    source_directory / "manifest.jsonl"
                ),
                "successful_variant_count": len(success_rows),
                "tier_success_counts": tier_counts.tolist(),
                "rejected_variant_count": len(rejection_rows),
                "rejected_source_count": rejected_source_count,
                "seed": seed,
                "max_attempts_per_variant": max_attempts_per_variant,
                "min_spawn_centers": min_spawn_centers,
            }
            (leaf_directory / PARTIAL_COMPLETION_CONFIG).write_text(
                json.dumps(leaf_config, indent=2, sort_keys=True) + "\n"
            )
            _json_line(leaf_directory / PARTIAL_COMPLETION_MANIFEST, success_rows)
            _json_line(leaf_directory / PARTIAL_COMPLETION_REJECTIONS, rejection_rows)
            supported_paths.append(maps_path)

        if not supported_paths:
            raise PartialCompletionError(
                "No condition has a successful relay partial reset."
            )
        supported_tier_counts = np.zeros(
            (len(PARTIAL_RESET_FRACTIONS),),
            dtype=np.int32,
        )
        for maps_path in supported_paths:
            leaf_config = json.loads(
                (temporary_root / maps_path / PARTIAL_COMPLETION_CONFIG).read_text()
            )
            supported_tier_counts += np.asarray(
                leaf_config["tier_success_counts"],
                dtype=np.int32,
            )
        if np.any(supported_tier_counts == 0):
            raise PartialCompletionError(
                "Sparse bank has no supported condition for scheduled tiers "
                f"{(np.flatnonzero(supported_tier_counts == 0) + 1).tolist()}."
            )
        _json_line(
            temporary_root / PARTIAL_COMPLETION_REJECTIONS,
            root_rejections,
        )
        bank_sha256 = partial_reset_bank_sha256(temporary_root)
        bank_index = {
            "schema": PARTIAL_RESET_BANK_SCHEMA,
            "input_root": str(input_root),
            "canonical_loader_registry_sha256": _file_sha256(
                input_root / "dataset.json"
            ),
            "completion_fractions": list(PARTIAL_RESET_FRACTIONS),
            "pile_mode": "relay_corridor",
            "source_triplet_contract": PARTIAL_RESET_TRIPLET_CONTRACT,
            "supported_maps_paths": supported_paths,
            "rejected_condition_count": len(leaves) - len(supported_paths),
            "bank_sha256": bank_sha256,
        }
        (temporary_root / PARTIAL_RESET_BANK_INDEX).write_text(
            json.dumps(bank_index, indent=2, sort_keys=True) + "\n"
        )
        temporary_root.rename(output_root)
        return bank_index
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
