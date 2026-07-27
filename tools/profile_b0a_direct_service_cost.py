#!/usr/bin/env python3
"""Profile the exact direct-service validator without emitting admission results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np

import terra.benchmark_direct_service as direct_service
from terra.benchmark_protocol import BENCHMARK_MAP_SIZE as MAP_SIZE
from terra.benchmark_protocol import BENCHMARK_RELEASE_ID as RELEASE_ID
from terra.benchmark_protocol import frozen_benchmark_protocol
from terra.benchmark_state import agent_to_record
from terra.benchmark_state import agent_state_sha256
from terra.benchmark_state import sample_benchmark_initial_agent
from terra.benchmark_state import validate_benchmark_initial_agent
from terra.config import EnvConfig
from terra.env import TerraEnv
from terra.maps_buffer import load_maps_from_disk
from terra.maps_buffer import validate_exact_dataset_contract

SCHEMA = "terra_direct_service_validation_cost_probe_v1"
BENCHMARK_SPLIT = "public_train"
LEGACY_B0A_SPLIT = "train"
SELECTED_MAP_ID = "b0a-train-f_apron_d02-00"
WARM_SUBSET_ROWS = 16
WARM_REPEATS = 3
EXPECTED_B0A_IDENTITIES = 256
EXPECTED_B0A_SOURCE_GROUP_SIZES = {1: 112, 4: 16, 5: 16}
DIRECT_SERVICE_OUTCOME_KEYS = {
    "required_volume",
    "admissible_pose_count_initial",
    "base_pose_cabin_heading_candidates_initial",
    "successful_target_dig_replays",
    "movement_source_rows_logical",
    "movement_transition_attempts_logical",
    "movement_source_rows_padded_executed",
    "movement_transition_attempts_padded_executed",
    "dig_prefilter_candidate_rows_logical",
    "dig_prefilter_candidate_rows_padded_executed",
    "service_dig_candidate_attempts_logical",
    "service_candidate_rows_padded_executed",
    "service_dig_do_transitions_padded_executed",
    "dump_do_attempts_logical",
    "dump_do_transitions_padded_executed",
    "legal_complete_dump_attempts",
    "wrong_complete_dump_attempts",
    "rejected_dump_attempts",
    "workspace_serviceable_volume_initial",
    "direct_serviceable_volume_initial",
    "initial_workspace_coverage",
    "direct_service_coverage_initial",
    "any_direct_transfer_pose_exists_initial",
}


@dataclass(frozen=True)
class ScenarioInput:
    record: dict[str, Any]
    dataset_directory: Path
    slot_index: int
    target: np.ndarray
    padding_mask: np.ndarray
    trench_axes: np.ndarray
    trench_type: np.ndarray
    foundation_border_axes: np.ndarray
    foundation_border_type: np.ndarray
    dumpability_mask: np.ndarray
    action_map: np.ndarray
    distance_map: np.ndarray
    selected_files: tuple[Path, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return {key: _jsonable(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} into the receipt.")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid JSON in {path} at line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected an object in {path} at line {line_number}.")
        rows.append(row)
    return rows


def _select_group_records(
    identities_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    rows = _load_jsonl(identities_path)
    groups = _source_groups(rows)
    selected_rows = [row for row in rows if row.get("map_id") == SELECTED_MAP_ID]
    if len(selected_rows) != 1:
        raise RuntimeError(
            f"Expected exactly one {SELECTED_MAP_ID} identity, got "
            f"{len(selected_rows)}."
        )
    selected = selected_rows[0]
    if selected.get("split") != LEGACY_B0A_SPLIT or selected.get("stratum") != "B0a":
        raise RuntimeError(
            f"{SELECTED_MAP_ID} must be a train B0a identity, got "
            f"split={selected.get('split')!r}, stratum={selected.get('stratum')!r}."
        )

    source_group_id = selected.get("source_id")
    if not isinstance(source_group_id, str) or not source_group_id:
        raise RuntimeError(f"{SELECTED_MAP_ID} has no canonical source_id.")
    group = groups[source_group_id]

    group = sorted(group, key=lambda row: row["map_id"])
    if not group:
        raise RuntimeError(f"Source group {source_group_id!r} is empty.")
    for row in group:
        if row.get("split") != LEGACY_B0A_SPLIT or row.get("stratum") != "B0a":
            raise RuntimeError(
                f"Source group {source_group_id!r} crosses B0a splits/strata."
            )
    return selected, group, source_group_id


def _source_groups(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError(
                f"B0a identity has invalid source_id: {row.get('map_id')}"
            )
        groups.setdefault(source_id, []).append(row)
    for source_id, source_rows in groups.items():
        if len(source_rows) == 1:
            continue
        splits = {row.get("split") for row in source_rows}
        dig_hashes = {row.get("dig_identity_sha256") for row in source_rows}
        if len(splits) != 1 or None in dig_hashes or len(dig_hashes) != 1:
            raise RuntimeError(
                f"Repeated source_id {source_id!r} crosses split or dig identity."
            )
    return groups


def _frozen_source_group_receipt(identities_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(identities_path)
    groups = _source_groups(rows)
    size_counts: dict[int, int] = {}
    for source_rows in groups.values():
        size = len(source_rows)
        size_counts[size] = size_counts.get(size, 0) + 1
    if (
        len(rows) != EXPECTED_B0A_IDENTITIES
        or size_counts != EXPECTED_B0A_SOURCE_GROUP_SIZES
    ):
        raise RuntimeError(
            "Frozen B0a source grouping changed: "
            f"identities={len(rows)}, group_sizes={size_counts}."
        )
    return {
        "identity_count": len(rows),
        "source_group_count": len(groups),
        "source_group_size_counts": {
            str(size): count for size, count in sorted(size_counts.items())
        },
        "canonical_source_group_field": "source_id",
        "paired_source_group_field": "paired_source_group_id",
    }


@contextmanager
def _dataset_size(slot_count: int) -> Iterator[None]:
    previous = os.environ.get("DATASET_SIZE")
    os.environ["DATASET_SIZE"] = str(slot_count)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DATASET_SIZE", None)
        else:
            os.environ["DATASET_SIZE"] = previous


def _load_scenario(root: Path, record: dict[str, Any]) -> ScenarioInput:
    directory = root / "cells" / record["split"] / record["primary_cell"]
    dataset_metadata = json.loads((directory / "dataset.json").read_text())
    slot_count = dataset_metadata.get("slot_count")
    if not isinstance(slot_count, int) or slot_count <= 0:
        raise RuntimeError(f"{directory}/dataset.json has invalid slot_count.")

    manifest, shape, _ = validate_exact_dataset_contract(directory, slot_count)
    if shape != (MAP_SIZE, MAP_SIZE):
        raise RuntimeError(
            f"{directory} has shape {shape}, expected {(MAP_SIZE,) * 2}."
        )
    matching = [row for row in manifest if row["map_id"] == record["map_id"]]
    if len(matching) != 1:
        raise RuntimeError(
            f"{directory} contains {len(matching)} slots for {record['map_id']}."
        )
    manifest_row = matching[0]
    for field in ("source_id", "split", "family", "primary_cell"):
        if manifest_row[field] != record[field]:
            raise RuntimeError(
                f"{record['map_id']} disagrees between identity and manifest "
                f"for {field}: {record[field]!r} != {manifest_row[field]!r}."
            )

    with _dataset_size(slot_count):
        loaded = load_maps_from_disk(
            str(directory),
            require_trench_metadata=False,
            require_exact_contract=True,
        )
    slot = int(manifest_row["slot_index"]) - 1
    arrays = [np.asarray(jax.device_get(value)) for value in loaded]
    (
        targets,
        padding_masks,
        trench_axes,
        trench_types,
        foundation_border_axes,
        foundation_border_types,
        dumpability_masks,
        action_maps,
        distance_maps,
    ) = arrays
    target = targets[slot]
    file_index = slot + 1
    target_path = directory / "images" / f"img_{file_index}.npy"
    serialized_target = np.load(target_path, allow_pickle=False)
    if _sha256_array(serialized_target) != record["target_identity_sha256"]:
        raise RuntimeError(f"{record['map_id']} target identity hash changed.")
    if not np.array_equal(target, serialized_target):
        raise RuntimeError(f"{record['map_id']} changed while entering the loader.")

    selected_files = (
        directory / "dataset.json",
        directory / "manifest.jsonl",
        target_path,
        directory / "occupancy" / f"img_{file_index}.npy",
        directory / "dumpability" / f"img_{file_index}.npy",
        directory / "actions" / f"img_{file_index}.npy",
        directory / "distance" / f"img_{file_index}.npy",
        directory / "metadata" / f"trench_{file_index}.json",
    )
    return ScenarioInput(
        record=record,
        dataset_directory=directory,
        slot_index=file_index,
        target=target,
        padding_mask=padding_masks[slot],
        trench_axes=trench_axes[slot],
        trench_type=trench_types[slot],
        foundation_border_axes=foundation_border_axes[slot],
        foundation_border_type=foundation_border_types[slot],
        dumpability_mask=dumpability_masks[slot],
        action_map=action_maps[slot],
        distance_map=distance_maps[slot],
        selected_files=selected_files,
    )


def _intersect_spawn_contract(
    scenarios: list[ScenarioInput],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    shapes = {scenario.target.shape for scenario in scenarios}
    if shapes != {(MAP_SIZE, MAP_SIZE)}:
        raise RuntimeError(f"Expected only 64 x 64 source variants, got {shapes}.")

    padding = np.maximum.reduce(
        [np.asarray(scenario.padding_mask, dtype=np.int8) for scenario in scenarios]
    )
    occupied_soil = np.logical_or.reduce(
        [np.asarray(scenario.action_map) != 0 for scenario in scenarios]
    ).astype(np.int8)
    dumpability = np.logical_and.reduce(
        [
            np.asarray(scenario.dumpability_mask, dtype=np.bool_)
            for scenario in scenarios
        ]
    )
    return padding, occupied_soil, dumpability


def _frozen_env_config() -> tuple[EnvConfig, dict[str, Any]]:
    return frozen_benchmark_protocol()


def _synchronize(tree: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return tree


def _materialize_initial_state(
    selected: ScenarioInput,
    group: list[ScenarioInput],
    source_group_id: str,
    env_config: EnvConfig,
) -> tuple[Any, dict[str, Any]]:
    padding, actions, dumpability = _intersect_spawn_contract(group)
    sample_start = time.perf_counter()
    agent, seed_receipt = sample_benchmark_initial_agent(
        release_id=RELEASE_ID,
        split=BENCHMARK_SPLIT,
        source_group_id=source_group_id,
        state_index=0,
        env_cfg=env_config,
        padding_mask=padding,
        action_map=actions,
        dumpability_mask=dumpability,
    )
    sample_seconds = time.perf_counter() - sample_start
    for scenario in group:
        validate_benchmark_initial_agent(
            agent,
            env_cfg=env_config,
            padding_mask=scenario.padding_mask,
            action_map=scenario.action_map,
            dumpability_mask=scenario.dumpability_mask,
        )

    env = TerraEnv.new(maps_size_px=MAP_SIZE)
    reset_start = time.perf_counter()
    timestep = env.reset(
        jax.random.PRNGKey(seed_receipt["seed_uint32"]),
        jnp.asarray(selected.target),
        jnp.asarray(selected.padding_mask),
        jnp.asarray(selected.trench_axes),
        jnp.asarray(selected.trench_type),
        jnp.asarray(selected.foundation_border_axes),
        jnp.asarray(selected.foundation_border_type),
        jnp.asarray(selected.dumpability_mask),
        jnp.asarray(selected.action_map),
        jnp.asarray(selected.distance_map),
        env_config,
        agent,
    )
    _synchronize(timestep)
    reset_seconds = time.perf_counter() - reset_start
    if (
        agent_state_sha256(timestep.state.agent)
        != seed_receipt["initial_agent_state_sha256"]
    ):
        raise RuntimeError("Explicit reset changed the serialized initial agent.")
    if int(np.asarray(jax.device_get(timestep.state.env_steps))) != 0:
        raise RuntimeError("The cost probe must start from env_steps == 0.")

    return timestep.state, {
        **seed_receipt,
        "initial_agent_state": agent_to_record(agent),
        "group_variant_count": len(group),
        "initial_agent_sampling_seconds": sample_seconds,
        "explicit_reset_compile_execute_seconds": reset_seconds,
    }


def _device_memory_stats(device: Any) -> dict[str, Any] | None:
    try:
        stats = device.memory_stats()
    except (AttributeError, RuntimeError):
        return None
    if stats is None:
        return None
    return {
        str(key): _jsonable(value)
        for key, value in stats.items()
        if isinstance(value, (bool, int, float, str, np.generic))
    }


def _max_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _run_graph_prefilter(state: Any) -> dict[str, Any]:
    movement_start = time.perf_counter()
    poses, movement_stats = direct_service._reachable_base_poses(state)
    movement_seconds = time.perf_counter() - movement_start

    candidate_start = time.perf_counter()
    candidates = direct_service._candidate_rows(
        poses,
        int(state.env_cfg.agent.angles_cabin),
    )
    candidate_seconds = time.perf_counter() - candidate_start

    prefilter_start = time.perf_counter()
    accepted, prefilter_stats = direct_service._prefilter_candidates(
        state,
        candidates,
    )
    prefilter_seconds = time.perf_counter() - prefilter_start
    return {
        "poses": poses,
        "movement_stats": movement_stats,
        "candidates": candidates,
        "accepted": accepted,
        "prefilter_stats": prefilter_stats,
        "timings": {
            "movement_graph": movement_seconds,
            "candidate_build": candidate_seconds,
            "prefilter": prefilter_seconds,
            "total": movement_seconds + candidate_seconds + prefilter_seconds,
        },
    }


def _profile_exact_service_cost(state: Any) -> dict[str, Any]:
    rss_before = _max_rss_kib()
    device = jax.devices()[0]
    device_memory_before = _device_memory_stats(device)

    cold_graph = _run_graph_prefilter(state)
    warm_graph = _run_graph_prefilter(state)
    for key in ("poses", "candidates", "accepted"):
        if not np.array_equal(cold_graph[key], warm_graph[key]):
            raise RuntimeError(f"Cold/warm graph-prefilter {key} rows changed.")
    for key in ("movement_stats", "prefilter_stats"):
        if cold_graph[key] != warm_graph[key]:
            raise RuntimeError(f"Cold/warm graph-prefilter {key} changed.")

    poses = cold_graph["poses"]
    candidates = cold_graph["candidates"]
    replay_candidates = cold_graph["accepted"]
    movement_stats = cold_graph["movement_stats"]
    prefilter_stats = cold_graph["prefilter_stats"]
    if len(replay_candidates) == 0:
        raise RuntimeError(f"{SELECTED_MAP_ID} has no exact-service replay candidates.")

    service_batch_size = int(direct_service._SERVICE_BATCH_SIZE)
    full_service_batch_count = (
        len(replay_candidates) + service_batch_size - 1
    ) // service_batch_size
    cold_rows, cold_valid_count = direct_service._pad_rows(
        replay_candidates[:service_batch_size],
        service_batch_size,
    )
    cold_device_rows = jnp.asarray(cold_rows, dtype=jnp.int32)
    _synchronize(cold_device_rows)

    jax.clear_caches()
    lower_start = time.perf_counter()
    lowered = direct_service._service_batch.lower(state, cold_device_rows)
    lowering_seconds = time.perf_counter() - lower_start
    compile_start = time.perf_counter()
    compiled = lowered.compile()
    cold_compile_seconds = time.perf_counter() - compile_start
    execute_start = time.perf_counter()
    _synchronize(compiled(state, cold_device_rows))
    first_execute_seconds = time.perf_counter() - execute_start

    warm_unique_count = min(WARM_SUBSET_ROWS, len(replay_candidates))
    warm_rows = replay_candidates[:warm_unique_count]
    warm_samples: list[dict[str, Any]] = []
    for repeat in range(WARM_REPEATS):
        for batch_index, rows in enumerate(
            direct_service._iter_chunks(warm_rows, service_batch_size)
        ):
            padded, valid_count = direct_service._pad_rows(rows, service_batch_size)
            device_rows = jnp.asarray(padded, dtype=jnp.int32)
            _synchronize(device_rows)
            started = time.perf_counter()
            _synchronize(compiled(state, device_rows))
            elapsed = time.perf_counter() - started
            warm_samples.append(
                {
                    "repeat": repeat,
                    "batch_index": batch_index,
                    "logical_rows": valid_count,
                    "padded_rows": service_batch_size,
                    "batch_seconds": elapsed,
                }
            )

    per_batch = np.asarray(
        [sample["batch_seconds"] for sample in warm_samples],
        dtype=np.float64,
    )
    per_batch_p50 = float(np.percentile(per_batch, 50))
    per_batch_p95 = float(np.percentile(per_batch, 95))
    projections = _project_cost(
        cold_graph_prefilter_seconds=cold_graph["timings"]["total"],
        warm_graph_prefilter_seconds=warm_graph["timings"]["total"],
        lowering_seconds=lowering_seconds,
        cold_compile_seconds=cold_compile_seconds,
        first_execute_seconds=first_execute_seconds,
        service_padded_batch_count=full_service_batch_count,
        steady_seconds_per_padded_batch_p50=per_batch_p50,
        steady_seconds_per_padded_batch_p95=per_batch_p95,
    )

    prefilter_batch_size = int(direct_service._PREFILTER_BATCH_SIZE)
    movement_batch_size = int(direct_service._MOVEMENT_BATCH_SIZE)
    if (
        movement_stats["source_rows_padded_executed"] % movement_batch_size
        or prefilter_stats["candidate_rows_padded_executed"] % prefilter_batch_size
    ):
        raise RuntimeError("Validator padded-row counters are not whole batches.")
    movement_batches = (
        movement_stats["source_rows_padded_executed"] // movement_batch_size
    )
    prefilter_batches = (
        prefilter_stats["candidate_rows_padded_executed"] // prefilter_batch_size
    )
    cabin_headings = int(state.env_cfg.agent.angles_cabin)
    warm_logical_rows = int(sum(sample["logical_rows"] for sample in warm_samples))
    warm_padded_rows = int(sum(sample["padded_rows"] for sample in warm_samples))
    full_padded_rows = full_service_batch_count * service_batch_size
    result = {
        "logical_and_padded_counters": {
            "movement_graph_pose_rows_per_full_execution": int(len(poses)),
            "movement_source_rows_logical_per_full_execution": int(
                movement_stats["source_rows_logical"]
            ),
            "movement_transition_attempts_logical_per_full_execution": int(
                movement_stats["transition_attempts_logical"]
            ),
            "movement_kernel_batches_per_full_execution": int(movement_batches),
            "movement_source_rows_padded_per_full_execution": int(
                movement_stats["source_rows_padded_executed"]
            ),
            "movement_transition_attempts_padded_per_full_execution": int(
                movement_stats["transition_attempts_padded_executed"]
            ),
            "pose_cabin_rows_logical_per_full_execution": int(len(candidates)),
            "prefilter_rows_logical_per_full_execution": int(
                prefilter_stats["candidate_rows_logical"]
            ),
            "prefilter_kernel_batches_per_full_execution": int(prefilter_batches),
            "prefilter_rows_padded_per_full_execution": int(
                prefilter_stats["candidate_rows_padded_executed"]
            ),
            "movement_graph_pose_rows_executed_total": int(2 * len(poses)),
            "movement_source_rows_logical_executed_total": int(
                2 * movement_stats["source_rows_logical"]
            ),
            "movement_transition_attempts_logical_executed_total": int(
                2 * movement_stats["transition_attempts_logical"]
            ),
            "movement_kernel_batches_executed_total": int(2 * movement_batches),
            "movement_source_rows_padded_executed_total": int(
                2 * movement_stats["source_rows_padded_executed"]
            ),
            "movement_transition_attempts_padded_executed_total": int(
                2 * movement_stats["transition_attempts_padded_executed"]
            ),
            "pose_cabin_rows_logical_executed_total": int(2 * len(candidates)),
            "prefilter_rows_logical_executed_total": int(
                2 * prefilter_stats["candidate_rows_logical"]
            ),
            "prefilter_kernel_batches_executed_total": int(2 * prefilter_batches),
            "prefilter_rows_padded_executed_total": int(
                2 * prefilter_stats["candidate_rows_padded_executed"]
            ),
            "exact_service_logical_rows_available": int(len(replay_candidates)),
            "full_service_padded_batch_count_projected": int(full_service_batch_count),
            "full_service_dig_transitions_logical_projected": int(
                len(replay_candidates)
            ),
            "full_service_dig_transitions_padded_projected": int(full_padded_rows),
            "full_service_dump_transitions_logical_projected": int(
                len(replay_candidates) * cabin_headings
            ),
            "full_service_dump_transitions_padded_projected": int(
                full_padded_rows * cabin_headings
            ),
            "cold_service_dig_transitions_logical_executed": int(cold_valid_count),
            "cold_service_dig_transitions_padded_executed": service_batch_size,
            "cold_service_dump_transitions_logical_executed": int(
                cold_valid_count * cabin_headings
            ),
            "cold_service_dump_transitions_padded_executed": int(
                service_batch_size * cabin_headings
            ),
            "warm_subset_unique_logical_rows": int(warm_unique_count),
            "warm_subset_repeats": WARM_REPEATS,
            "warm_service_dig_transitions_logical_executed": warm_logical_rows,
            "warm_service_dig_transitions_padded_executed": warm_padded_rows,
            "warm_service_dump_transitions_logical_executed": int(
                warm_logical_rows * cabin_headings
            ),
            "warm_service_dump_transitions_padded_executed": int(
                warm_padded_rows * cabin_headings
            ),
            "warm_service_kernel_batch_executions": len(warm_samples),
            "dump_headings_per_service_dig": cabin_headings,
            "graph_prefilter_full_executions": 2,
        },
        "timings_seconds": {
            "graph_prefilter_cold": cold_graph["timings"],
            "graph_prefilter_warm": warm_graph["timings"],
            "service_lowering": lowering_seconds,
            "service_cold_compile": cold_compile_seconds,
            "service_first_synchronized_execute": first_execute_seconds,
            "warm_seconds_per_padded_batch_p50": per_batch_p50,
            "warm_seconds_per_padded_batch_p95": per_batch_p95,
            "warm_samples": warm_samples,
        },
        "memory": {
            "process_max_rss_kib_before": rss_before,
            "process_max_rss_kib_after": _max_rss_kib(),
            "device_before": device_memory_before,
            "device_after": _device_memory_stats(device),
        },
        "projections": projections,
    }
    _assert_cost_only(result)
    return result


def _project_cost(
    *,
    cold_graph_prefilter_seconds: float,
    warm_graph_prefilter_seconds: float,
    lowering_seconds: float,
    cold_compile_seconds: float,
    first_execute_seconds: float,
    service_padded_batch_count: int,
    steady_seconds_per_padded_batch_p50: float,
    steady_seconds_per_padded_batch_p95: float,
) -> dict[str, Any]:
    numeric_inputs = (
        cold_graph_prefilter_seconds,
        warm_graph_prefilter_seconds,
        lowering_seconds,
        cold_compile_seconds,
        first_execute_seconds,
        steady_seconds_per_padded_batch_p50,
        steady_seconds_per_padded_batch_p95,
    )
    if any(value < 0 for value in numeric_inputs) or service_padded_batch_count <= 0:
        raise ValueError("Cost projection inputs must be nonnegative and nonempty.")
    remaining_cold_batches = service_padded_batch_count - 1
    cold_p50 = (
        cold_graph_prefilter_seconds
        + lowering_seconds
        + cold_compile_seconds
        + first_execute_seconds
        + remaining_cold_batches * steady_seconds_per_padded_batch_p50
    )
    cold_p95 = (
        cold_graph_prefilter_seconds
        + lowering_seconds
        + cold_compile_seconds
        + first_execute_seconds
        + remaining_cold_batches * steady_seconds_per_padded_batch_p95
    )
    warm_p50 = (
        warm_graph_prefilter_seconds
        + service_padded_batch_count * steady_seconds_per_padded_batch_p50
    )
    warm_p95 = (
        warm_graph_prefilter_seconds
        + service_padded_batch_count * steady_seconds_per_padded_batch_p95
    )
    return {
        "method": (
            "first scenario uses cold graph/prefilter, lowering, one compile, "
            "first synchronized service batch, then warmed padded batches; "
            "later scenarios use warm graph/prefilter and fixed padded batches"
        ),
        "service_padded_batch_count": service_padded_batch_count,
        "first_scenario_cold_seconds_p50": cold_p50,
        "first_scenario_cold_seconds_p95": cold_p95,
        "later_scenario_warm_seconds_p50": warm_p50,
        "later_scenario_warm_seconds_p95": warm_p95,
        "cold_components_seconds": {
            "graph_prefilter": cold_graph_prefilter_seconds,
            "lowering": lowering_seconds,
            "compile": cold_compile_seconds,
            "first_synchronized_batch": first_execute_seconds,
        },
        "warm_components_seconds": {
            "graph_prefilter": warm_graph_prefilter_seconds,
            "padded_batch_p50": steady_seconds_per_padded_batch_p50,
            "padded_batch_p95": steady_seconds_per_padded_batch_p95,
        },
        "scenario_counts": {
            str(count): {
                "seconds_p50": cold_p50 + (count - 1) * warm_p50,
                "seconds_p95": cold_p95 + (count - 1) * warm_p95,
                "cold_compile_executions": 1,
            }
            for count in (1, 256, 448)
        },
        "requires_one_complete_scenario_confirmation": True,
    }


def _assert_cost_only(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = {
            key
            for key in value
            if key in DIRECT_SERVICE_OUTCOME_KEYS
            or "coverage" in key.lower()
            or "validity" in key.lower()
            or key in {"admission_decision", "feasibility_decision"}
        }
        if forbidden:
            raise RuntimeError(
                f"Cost-only probe attempted to emit result fields: {sorted(forbidden)}"
            )
        for item in value.values():
            _assert_cost_only(item)
    elif isinstance(value, list):
        for item in value:
            _assert_cost_only(item)


def _verify_selected_inputs(root: Path, paths: set[Path]) -> dict[str, str]:
    manifest_path = root / "files.sha256"
    entries: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        entries[relative] = digest

    verified = {}
    for path in sorted(paths):
        relative = str(path.resolve().relative_to(root.resolve()))
        expected = entries.get(relative)
        if expected is None:
            raise RuntimeError(f"{relative} is absent from {manifest_path}.")
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen B0a input hash changed for {relative}: "
                f"{expected} != {actual}."
            )
        verified[relative] = actual
    return verified


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_receipt(repository: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "head": head,
        "dirty": bool(porcelain),
        "porcelain_v1": porcelain,
        "code_file_sha256_is_authoritative_for_worktree_state": True,
    }


def _device_receipt() -> dict[str, Any]:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX exposed no devices.")
    device = devices[0]
    return {
        "device_count": len(devices),
        "selected_device": str(device),
        "platform": device.platform,
        "device_kind": getattr(device, "device_kind", None),
        "process_index": int(device.process_index),
        "local_hardware_id": int(device.local_hardware_id),
    }


def _code_receipt(repository: Path) -> dict[str, Any]:
    code_paths = (
        Path(__file__).resolve(),
        repository / "terra" / "benchmark_direct_service.py",
        repository / "terra" / "benchmark_state.py",
        repository / "terra" / "maps_buffer.py",
        repository / "terra" / "env.py",
        repository / "terra" / "state.py",
        repository / "terra" / "agent.py",
        repository / "terra" / "actions.py",
        repository / "terra" / "config.py",
        repository / "terra" / "map.py",
        repository / "terra" / "settings.py",
        repository / "terra" / "utils.py",
        repository / "terra" / "wrappers.py",
    )
    code_hashes = {
        str(path.relative_to(repository)): _sha256_file(path) for path in code_paths
    }
    return {
        "git": _git_receipt(repository),
        "code_file_sha256": code_hashes,
        "code_bundle_sha256": _canonical_json_sha256(code_hashes),
    }


def run_probe(b0a_root: Path, output_directory: Path) -> Path:
    b0a_root = b0a_root.resolve()
    output_directory = output_directory.resolve()
    output_path = output_directory / "validation_cost_probe.json"
    if output_path.exists():
        raise FileExistsError(output_path)

    identities_path = b0a_root / "identities.jsonl"
    provenance_path = b0a_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    identities_sha256 = _sha256_file(identities_path)
    if provenance.get("identity_manifest_sha256") != identities_sha256:
        raise RuntimeError("B0a provenance does not match identities.jsonl.")
    selected_record, group_records, source_group_id = _select_group_records(
        identities_path
    )
    source_group_receipt = _frozen_source_group_receipt(identities_path)
    group = [_load_scenario(b0a_root, record) for record in group_records]
    selected = next(
        scenario for scenario in group if scenario.record["map_id"] == SELECTED_MAP_ID
    )

    selected_paths = {
        identities_path,
        provenance_path,
        b0a_root / "source_registry.jsonl",
        *(path for scenario in group for path in scenario.selected_files),
    }
    verified_inputs = _verify_selected_inputs(b0a_root, selected_paths)

    env_config, protocol = _frozen_env_config()
    state, initial_state = _materialize_initial_state(
        selected,
        group,
        source_group_id,
        env_config,
    )

    repository = Path(__file__).resolve().parents[1]
    code_before = _code_receipt(repository)
    measured = _profile_exact_service_cost(state)
    code_after = _code_receipt(repository)
    if code_after != code_before:
        raise RuntimeError(
            "Repository or validator code changed while the cost probe was running."
        )

    receipt = {
        "schema": SCHEMA,
        "release_id": RELEASE_ID,
        "result_scope": "non_admission_cost_only",
        "admission_result_emitted": False,
        "subset_outputs_discarded_after_synchronization": True,
        "selected_identity": {
            "map_id": selected_record["map_id"],
            "legacy_dataset_split": selected_record["split"],
            "benchmark_split": BENCHMARK_SPLIT,
            "family": selected_record["family"],
            "primary_cell": selected_record["primary_cell"],
            "source_id": selected_record["source_id"],
            "source_group_id": source_group_id,
            "paired_source_group_ids": sorted(
                {
                    scenario.record["paired_source_group_id"]
                    for scenario in group
                    if scenario.record.get("paired_source_group_id") is not None
                }
            ),
            "group_map_ids": [scenario.record["map_id"] for scenario in group],
        },
        "input": {
            "b0a_root": str(b0a_root),
            "files_sha256_manifest_sha256": _sha256_file(b0a_root / "files.sha256"),
            "source_grouping": source_group_receipt,
            "verified_selected_files": verified_inputs,
        },
        "protocol": protocol,
        "initial_state": initial_state,
        "validator": {
            **code_before,
            "pre_and_post_measurement_receipts_identical": True,
            "service_kernel": "terra.benchmark_direct_service._service_batch",
            "service_kernel_unchanged_by_probe": True,
        },
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "packages": {
                name: _package_version(name)
                for name in ("jax", "jaxlib", "numpy", "scipy")
            },
            "jax": _device_receipt(),
            "compilation_cache_environment": {
                key: os.environ.get(key)
                for key in (
                    "JAX_COMPILATION_CACHE_DIR",
                    "JAX_ENABLE_COMPILATION_CACHE",
                    "JAX_PLATFORMS",
                )
            },
        },
        "measurement": measured,
    }
    _assert_cost_only(receipt)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a cost-only exact direct-service probe for the frozen "
            f"{SELECTED_MAP_ID} B0a identity."
        )
    )
    parser.add_argument("--b0a-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.b0a_root, args.output_directory)


if __name__ == "__main__":
    main()
