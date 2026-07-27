#!/usr/bin/env python3
"""Measure exact direct-service throughput for service batches 4, 8, and 16."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import terra.benchmark_direct_service as direct_service
import tools.confirm_b0a_direct_service_cost as confirmation
import tools.profile_b0a_direct_service_cost as probe_tool

SCHEMA = "terra_direct_service_batch_size_sweep_v1"
OUTPUT_NAME = "direct_service_batch_size_sweep.json"
BATCH_SIZES = (4, 8, 16)
TIMING_SUBSET_ROWS = 16
PARITY_SUBSET_ROWS = 18
TIMED_REPEATS = 12
MAX_BASELINE_P50_DRIFT_FRACTION = 0.05


def _service_batch_count(logical_rows: int, batch_size: int) -> int:
    if logical_rows <= 0 or batch_size <= 0:
        raise ValueError("Logical rows and batch size must be positive.")
    return (logical_rows + batch_size - 1) // batch_size


def _execute_subset(
    compiled: Any,
    state: Any,
    rows: np.ndarray,
    batch_size: int,
    *,
    capture_outputs: bool,
) -> tuple[dict[str, Any], tuple[np.ndarray, ...] | None]:
    started = time.perf_counter()
    outputs: list[tuple[np.ndarray, ...]] = []
    logical_rows = 0
    padded_rows = 0
    batch_count = 0
    for chunk in direct_service._iter_chunks(rows, batch_size):
        padded, valid_count = direct_service._pad_rows(chunk, batch_size)
        device_rows = jnp.asarray(padded, dtype=jnp.int32)
        probe_tool._synchronize(device_rows)
        result = compiled(state, device_rows)
        probe_tool._synchronize(result)
        logical_rows += valid_count
        padded_rows += len(padded)
        batch_count += 1
        if capture_outputs:
            outputs.append(
                tuple(np.asarray(jax.device_get(leaf))[:valid_count] for leaf in result)
            )
    elapsed_seconds = time.perf_counter() - started
    if logical_rows != len(rows):
        raise RuntimeError("Service subset did not execute every logical row.")

    concatenated = None
    if capture_outputs:
        concatenated = tuple(
            np.concatenate([batch[index] for batch in outputs], axis=0)
            for index in range(len(outputs[0]))
        )
    return (
        {
            "elapsed_seconds": elapsed_seconds,
            "logical_rows": logical_rows,
            "padded_rows": padded_rows,
            "batch_count": batch_count,
        },
        concatenated,
    )


def _output_sha256(outputs: tuple[np.ndarray, ...]) -> str:
    leaf_hashes = [probe_tool._sha256_array(leaf) for leaf in outputs]
    return probe_tool._canonical_json_sha256(leaf_hashes)


def _measure_batch_size(
    *,
    state: Any,
    replay_candidates: np.ndarray,
    batch_size: int,
    cold_graph_seconds: float,
    warm_graph_seconds: float,
    host_capacity_kib: int,
    cabin_headings: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
    if TIMING_SUBSET_ROWS % batch_size:
        raise RuntimeError("Every batch size must divide the timing subset exactly.")
    if len(replay_candidates) < PARITY_SUBSET_ROWS:
        raise RuntimeError(
            f"Need {PARITY_SUBSET_ROWS} candidates, got {len(replay_candidates)}."
        )

    rss_before = probe_tool._max_rss_kib()
    jax.clear_caches()
    cold_rows, cold_valid_count = direct_service._pad_rows(
        replay_candidates[:batch_size],
        batch_size,
    )
    cold_device_rows = jnp.asarray(cold_rows, dtype=jnp.int32)
    probe_tool._synchronize(cold_device_rows)

    lower_started = time.perf_counter()
    lowered = direct_service._service_batch.lower(state, cold_device_rows)
    lowering_seconds = time.perf_counter() - lower_started
    compile_started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    first_started = time.perf_counter()
    probe_tool._synchronize(compiled(state, cold_device_rows))
    first_execute_seconds = time.perf_counter() - first_started

    timing_rows = replay_candidates[:TIMING_SUBSET_ROWS]
    warmup_execution, _ = _execute_subset(
        compiled,
        state,
        timing_rows,
        batch_size,
        capture_outputs=False,
    )
    timing_samples = []
    for repeat in range(TIMED_REPEATS):
        sample, _ = _execute_subset(
            compiled,
            state,
            timing_rows,
            batch_size,
            capture_outputs=False,
        )
        timing_samples.append({"repeat": repeat, **sample})

    parity_rows = replay_candidates[:PARITY_SUBSET_ROWS]
    parity_execution, parity_outputs = _execute_subset(
        compiled,
        state,
        parity_rows,
        batch_size,
        capture_outputs=True,
    )
    if parity_outputs is None:
        raise RuntimeError("Parity execution did not return captured outputs.")

    launches_per_repeat = TIMING_SUBSET_ROWS // batch_size
    seconds_per_batch = np.asarray(
        [sample["elapsed_seconds"] / launches_per_repeat for sample in timing_samples],
        dtype=np.float64,
    )
    repeat_seconds = np.asarray(
        [sample["elapsed_seconds"] for sample in timing_samples],
        dtype=np.float64,
    )
    repeat_p50 = float(np.percentile(repeat_seconds, 50))
    repeat_p95 = float(np.percentile(repeat_seconds, 95))
    timing_values = np.concatenate(
        (
            np.asarray(
                (
                    lowering_seconds,
                    compile_seconds,
                    first_execute_seconds,
                    warmup_execution["elapsed_seconds"],
                ),
                dtype=np.float64,
            ),
            repeat_seconds,
        )
    )
    timing_finite_positive = bool(
        np.all(np.isfinite(timing_values)) and np.all(timing_values > 0)
    )

    full_batch_count = _service_batch_count(len(replay_candidates), batch_size)
    per_batch_p50 = float(np.percentile(seconds_per_batch, 50))
    per_batch_p95 = float(np.percentile(seconds_per_batch, 95))
    projections = probe_tool._project_cost(
        cold_graph_prefilter_seconds=cold_graph_seconds,
        warm_graph_prefilter_seconds=warm_graph_seconds,
        lowering_seconds=lowering_seconds,
        cold_compile_seconds=compile_seconds,
        first_execute_seconds=first_execute_seconds,
        service_padded_batch_count=full_batch_count,
        steady_seconds_per_padded_batch_p50=per_batch_p50,
        steady_seconds_per_padded_batch_p95=per_batch_p95,
    )

    rss_after = probe_tool._max_rss_kib()
    memory_headroom = confirmation._memory_headroom(
        rss_after,
        host_capacity_kib,
        "cpu",
    )
    timed_logical_rows = TIMED_REPEATS * TIMING_SUBSET_ROWS
    timed_padded_rows = sum(sample["padded_rows"] for sample in timing_samples)
    actual_logical_rows = (
        cold_valid_count
        + warmup_execution["logical_rows"]
        + timed_logical_rows
        + PARITY_SUBSET_ROWS
    )
    actual_padded_rows = (
        batch_size
        + warmup_execution["padded_rows"]
        + timed_padded_rows
        + parity_execution["padded_rows"]
    )
    return (
        {
            "batch_size": batch_size,
            "timing": {
                "lowering_seconds": lowering_seconds,
                "compile_seconds": compile_seconds,
                "first_synchronized_batch_seconds": first_execute_seconds,
                "untimed_complete_16_row_warmup_seconds": warmup_execution[
                    "elapsed_seconds"
                ],
                "complete_16_row_repeat_seconds": repeat_seconds.tolist(),
                "complete_16_row_repeat_p50": repeat_p50,
                "complete_16_row_repeat_p95": repeat_p95,
                "seconds_per_padded_batch_p50": per_batch_p50,
                "seconds_per_padded_batch_p95": per_batch_p95,
                "logical_rows_per_second_p50": TIMING_SUBSET_ROWS / repeat_p50,
                "logical_rows_per_second_p05": TIMING_SUBSET_ROWS / repeat_p95,
                "finite_positive": timing_finite_positive,
            },
            "execution_counters": {
                "full_service_logical_rows_projected": int(len(replay_candidates)),
                "full_service_batch_count_projected": full_batch_count,
                "full_service_padded_rows_projected": full_batch_count * batch_size,
                "cold_logical_rows_executed": cold_valid_count,
                "cold_padded_rows_executed": batch_size,
                "warmup_logical_rows_executed": warmup_execution["logical_rows"],
                "warmup_padded_rows_executed": warmup_execution["padded_rows"],
                "warmup_batch_count": warmup_execution["batch_count"],
                "timed_repeat_count": TIMED_REPEATS,
                "timed_logical_rows_executed": timed_logical_rows,
                "timed_padded_rows_executed": timed_padded_rows,
                "parity_logical_rows_executed": PARITY_SUBSET_ROWS,
                "parity_padded_rows_executed": parity_execution["padded_rows"],
                "actual_service_dig_transitions_logical": actual_logical_rows,
                "actual_service_dig_transitions_padded": actual_padded_rows,
                "actual_dump_do_transitions_logical": (
                    actual_logical_rows * cabin_headings
                ),
                "actual_dump_do_transitions_padded": (
                    actual_padded_rows * cabin_headings
                ),
            },
            "parity": {
                "subset_logical_rows": PARITY_SUBSET_ROWS,
                "padded_rows_executed": parity_execution["padded_rows"],
                "batch_count": parity_execution["batch_count"],
                "concatenated_output_sha256": _output_sha256(parity_outputs),
            },
            "memory": {
                "measurement_kind": (
                    "process_cumulative_peak; not isolated per batch-size arm"
                ),
                "process_max_rss_kib_before_arm": rss_before,
                "process_max_rss_kib_after_arm": rss_after,
                "process_max_rss_kib_increase": max(0, rss_after - rss_before),
                **memory_headroom,
            },
            "projections": projections,
        },
        parity_outputs,
    )


def _select_batch_size(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    opening = arms["4_open"]
    closing = arms["4_close"]
    if not opening["gates"]["eligible_for_reprobe"]:
        raise RuntimeError("Opening batch-size 4 control is not eligible.")
    if not closing["gates"]["eligible_for_reprobe"]:
        raise RuntimeError("Closing batch-size 4 control is not eligible.")

    opening_repeat_p50 = opening["timing"]["complete_16_row_repeat_p50"]
    closing_repeat_p50 = closing["timing"]["complete_16_row_repeat_p50"]
    faster_repeat_p50 = min(opening_repeat_p50, closing_repeat_p50)
    drift_fraction = abs(closing_repeat_p50 - opening_repeat_p50) / faster_repeat_p50
    drift_passes = drift_fraction <= MAX_BASELINE_P50_DRIFT_FRACTION
    opening_projected_p50 = opening["projections"]["scenario_counts"]["256"][
        "seconds_p50"
    ]
    closing_projected_p50 = closing["projections"]["scenario_counts"]["256"][
        "seconds_p50"
    ]
    baseline_p50 = min(opening_projected_p50, closing_projected_p50)

    qualifying = []
    for batch_size in BATCH_SIZES[1:]:
        arm = arms[str(batch_size)]
        candidate_p95 = arm["projections"]["scenario_counts"]["256"]["seconds_p95"]
        if (
            drift_passes
            and arm["gates"]["eligible_for_reprobe"]
            and candidate_p95 < baseline_p50
        ):
            qualifying.append((candidate_p95, batch_size))

    drift_receipt = {
        "opening_complete_16_row_p50_seconds": opening_repeat_p50,
        "closing_complete_16_row_p50_seconds": closing_repeat_p50,
        "relative_to_faster_control": drift_fraction,
        "maximum_allowed": MAX_BASELINE_P50_DRIFT_FRACTION,
        "passes": drift_passes,
    }
    if not qualifying:
        if drift_passes:
            reason = (
                "No eligible larger batch has projected 256-scenario p95 "
                "strictly below the faster batch-4 projected p50; retain batch 4."
            )
        else:
            reason = (
                "Opening/closing batch-4 p50 drift exceeds 5%; retain batch 4 "
                "and do not interpret arm timing."
            )
        return {
            "baseline_batch_size": 4,
            "selected_for_reprobe_batch_size": 4,
            "baseline_256_scenario_p50_seconds": baseline_p50,
            "selected_256_scenario_p95_seconds": opening["projections"][
                "scenario_counts"
            ]["256"]["seconds_p95"],
            "baseline_drift": drift_receipt,
            "strict_p95_below_baseline_p50": False,
            "selection_reason": reason,
            "authorizes_256_profile": False,
            "bank_admission_result_emitted": False,
            "requires_new_probe_and_full_confirmation": False,
        }

    selected_p95, selected = min(qualifying)
    return {
        "baseline_batch_size": 4,
        "selected_for_reprobe_batch_size": selected,
        "baseline_256_scenario_p50_seconds": baseline_p50,
        "selected_256_scenario_p95_seconds": selected_p95,
        "baseline_drift": drift_receipt,
        "strict_p95_below_baseline_p50": True,
        "selection_reason": (
            "Select the eligible batch with minimum projected 256-scenario p95; "
            "ties prefer the smaller batch."
        ),
        "authorizes_256_profile": False,
        "bank_admission_result_emitted": False,
        "requires_new_probe_and_full_confirmation": True,
    }


def _code_receipt(repository: Path) -> tuple[dict[str, Any], dict[str, str]]:
    confirmation_receipt, dependency_hashes = confirmation._code_receipt(repository)
    code_hashes = dict(confirmation_receipt["code_file_sha256"])
    script_path = Path(__file__).resolve()
    code_hashes[str(script_path.relative_to(repository))] = probe_tool._sha256_file(
        script_path
    )
    return (
        {
            "git": confirmation_receipt["git"],
            "code_file_sha256": code_hashes,
            "code_bundle_sha256": probe_tool._canonical_json_sha256(code_hashes),
        },
        dependency_hashes,
    )


def _machine_receipt() -> dict[str, Any]:
    device = confirmation._require_current_cpu_device()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "jax": {
            "device_count": 1,
            "selected_device": str(device),
            "platform": device.platform,
            "device_kind": getattr(device, "device_kind", None),
        },
        "packages": {
            name: probe_tool._package_version(name)
            for name in ("jax", "jaxlib", "numpy", "scipy")
        },
        "compilation_cache_environment": {
            key: os.environ.get(key)
            for key in (
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_ENABLE_COMPILATION_CACHE",
                "JAX_PLATFORMS",
            )
        },
    }


def _validate_pinned_runtime(
    probe: dict[str, Any],
    current: dict[str, Any],
) -> None:
    cache_environment = current["compilation_cache_environment"]
    cache_directory = cache_environment["JAX_COMPILATION_CACHE_DIR"]
    cache_enabled = cache_environment["JAX_ENABLE_COMPILATION_CACHE"]
    if cache_directory not in (None, "") or str(cache_enabled).lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise RuntimeError("Persistent JAX compilation caching must be disabled.")

    expected = probe["machine"]
    comparisons = {
        "hostname": (current["hostname"], expected["hostname"]),
        "machine": (current["machine"], expected["machine"]),
        "python": (current["python"], expected["python"]),
        "packages": (current["packages"], expected["packages"]),
        "JAX device count": (
            current["jax"]["device_count"],
            expected["jax"]["device_count"],
        ),
        "JAX platform": (
            current["jax"]["platform"],
            expected["jax"]["platform"],
        ),
        "JAX device kind": (
            current["jax"]["device_kind"],
            expected["jax"]["device_kind"],
        ),
        "compilation-cache environment": (
            current["compilation_cache_environment"],
            expected["compilation_cache_environment"],
        ),
    }
    mismatches = [
        name for name, (actual, pinned) in comparisons.items() if actual != pinned
    ]
    if mismatches:
        raise RuntimeError(
            "Batch sweep runtime no longer matches the pinned probe: "
            + ", ".join(mismatches)
            + "."
        )


def run_sweep(
    b0a_root: Path,
    probe_path: Path,
    output_directory: Path,
) -> Path:
    b0a_root = b0a_root.resolve()
    probe_path = probe_path.resolve()
    output_directory = output_directory.resolve()
    output_path = output_directory / OUTPUT_NAME
    if output_path.exists():
        raise FileExistsError(output_path)

    probe = confirmation._load_pinned_probe(probe_path)
    confirmation._require_current_cpu_device()
    machine = _machine_receipt()
    _validate_pinned_runtime(probe, machine)
    machine["persistent_compilation_cache_disabled"] = True
    identities_path = b0a_root / "identities.jsonl"
    selected_record, group_records, source_group_id = probe_tool._select_group_records(
        identities_path
    )
    source_grouping = probe_tool._frozen_source_group_receipt(identities_path)
    group = [probe_tool._load_scenario(b0a_root, record) for record in group_records]
    selected = next(
        scenario
        for scenario in group
        if scenario.record["map_id"] == probe_tool.SELECTED_MAP_ID
    )
    selected_identity = confirmation._selected_identity(
        selected_record,
        group,
        source_group_id,
    )
    selected_paths = {
        identities_path,
        b0a_root / "provenance.json",
        b0a_root / "source_registry.jsonl",
        *(path for scenario in group for path in scenario.selected_files),
    }
    verified_selected_files = probe_tool._verify_selected_inputs(
        b0a_root,
        selected_paths,
    )
    manifest_sha256 = probe_tool._sha256_file(b0a_root / "files.sha256")
    env_config, protocol = probe_tool._frozen_env_config()
    state, initial_state = probe_tool._materialize_initial_state(
        selected,
        group,
        source_group_id,
        env_config,
    )

    repository = Path(__file__).resolve().parents[1]
    code_before, dependency_hashes = _code_receipt(repository)
    confirmation._validate_rebuilt_contract(
        probe=probe,
        selected_identity=selected_identity,
        files_sha256_manifest_sha256=manifest_sha256,
        source_grouping=source_grouping,
        verified_selected_files=verified_selected_files,
        protocol=protocol,
        initial_state=initial_state,
        dependency_file_sha256=dependency_hashes,
    )

    sweep_rss_before = probe_tool._max_rss_kib()
    cold_graph = probe_tool._run_graph_prefilter(state)
    warm_graph = probe_tool._run_graph_prefilter(state)
    for key in ("poses", "candidates", "accepted"):
        if not np.array_equal(cold_graph[key], warm_graph[key]):
            raise RuntimeError(f"Cold/warm graph-prefilter {key} rows changed.")
    for key in ("movement_stats", "prefilter_stats"):
        if cold_graph[key] != warm_graph[key]:
            raise RuntimeError(f"Cold/warm graph-prefilter {key} changed.")

    replay_candidates = cold_graph["accepted"]
    if len(replay_candidates) < PARITY_SUBSET_ROWS:
        raise RuntimeError(
            f"Need {PARITY_SUBSET_ROWS} candidates, got {len(replay_candidates)}."
        )
    timing_rows = replay_candidates[:TIMING_SUBSET_ROWS]
    parity_rows = replay_candidates[:PARITY_SUBSET_ROWS]
    timing_subset_sha256 = probe_tool._sha256_array(timing_rows)
    parity_subset_sha256 = probe_tool._sha256_array(parity_rows)
    host_capacity_kib = confirmation._host_memory_capacity_kib()
    cabin_headings = int(state.env_cfg.agent.angles_cabin)

    arm_order = (
        ("4_open", 4),
        ("8", 8),
        ("16", 16),
        ("4_close", 4),
    )
    arms: dict[str, dict[str, Any]] = {}
    reference_hash = None
    for arm_name, batch_size in arm_order:
        arm, outputs = _measure_batch_size(
            state=state,
            replay_candidates=replay_candidates,
            batch_size=batch_size,
            cold_graph_seconds=cold_graph["timings"]["total"],
            warm_graph_seconds=warm_graph["timings"]["total"],
            host_capacity_kib=host_capacity_kib,
            cabin_headings=cabin_headings,
        )
        output_hash = _output_sha256(outputs)
        if reference_hash is None:
            reference_hash = output_hash
            parity_passes = True
        else:
            parity_passes = output_hash == reference_hash
        arm["arm_name"] = arm_name
        arm["parity"]["reference_batch_size"] = 4
        arm["parity"]["reference_arm_name"] = "4_open"
        arm["parity"]["exact_concatenated_output_parity"] = parity_passes
        arm["gates"] = {
            "exact_concatenated_output_parity": parity_passes,
            "timing_finite_positive": arm["timing"]["finite_positive"],
            "cumulative_memory_headroom_passes": arm["memory"]["all_memory_gates_pass"],
        }
        arm["gates"]["eligible_for_reprobe"] = all(arm["gates"].values())
        arms[arm_name] = arm

    decision = _select_batch_size(arms)
    code_after, dependency_hashes_after = _code_receipt(repository)
    if code_after != code_before or dependency_hashes_after != dependency_hashes:
        raise RuntimeError(
            "Repository, validator, confirmation, or sweep code changed during "
            "execution."
        )

    receipt = {
        "schema": SCHEMA,
        "release_id": probe_tool.RELEASE_ID,
        "result_scope": "non_admission_service_batch_throughput_sweep",
        "bank_admission_result_emitted": False,
        "full_exact_validator_called": False,
        "selected_identity": selected_identity,
        "input": {
            "b0a_root": str(b0a_root),
            "probe_receipt": str(probe_path),
            "probe_receipt_sha256": confirmation.EXPECTED_PROBE_SHA256,
            "files_sha256_manifest_sha256": manifest_sha256,
            "source_grouping": source_grouping,
            "verified_selected_files": verified_selected_files,
        },
        "protocol": protocol,
        "initial_state": initial_state,
        "experiment": {
            "arm_names_in_execution_order": [name for name, _ in arm_order],
            "batch_sizes_in_execution_order": [size for _, size in arm_order],
            "timing_subset_logical_rows": TIMING_SUBSET_ROWS,
            "timing_subset_sha256": timing_subset_sha256,
            "parity_subset_logical_rows": PARITY_SUBSET_ROWS,
            "parity_subset_sha256": parity_subset_sha256,
            "timed_complete_subset_repeats": TIMED_REPEATS,
            "equal_complete_subset_warmups_per_arm": 1,
            "maximum_open_close_batch4_p50_drift_fraction": (
                MAX_BASELINE_P50_DRIFT_FRACTION
            ),
            "service_kernel": "terra.benchmark_direct_service._service_batch",
            "full_validator_entrypoint_call_count": 0,
        },
        "graph_prefilter": {
            "cold_timings_seconds": cold_graph["timings"],
            "warm_timings_seconds": warm_graph["timings"],
            "logical_service_candidate_rows": int(len(replay_candidates)),
            "movement_counters": cold_graph["movement_stats"],
            "prefilter_counters": cold_graph["prefilter_stats"],
            "cold_warm_rows_and_counters_identical": True,
        },
        "arms": arms,
        "decision": decision,
        "memory": {
            "measurement_kind": (
                "one-process cumulative peak; batch-size arms are not isolated"
            ),
            "process_max_rss_kib_before_sweep": sweep_rss_before,
            "process_max_rss_kib_after_sweep": probe_tool._max_rss_kib(),
            "host_physical_memory_kib": host_capacity_kib,
            "maximum_allowed_fraction": confirmation.MAX_MEMORY_FRACTION,
        },
        "validator": {
            "code_before": code_before,
            "code_after": code_after,
            "pre_and_post_execution_receipts_identical": True,
            "probe_dependency_file_hashes_unchanged": True,
        },
        "machine": machine,
        "command": [str(argument) for argument in sys.argv],
    }
    probe_tool._assert_cost_only(receipt)
    output_directory.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep exact direct-service kernel batches 4/8/16 on the pinned "
            "B0a state without running the full validator."
        )
    )
    parser.add_argument("--b0a-root", type=Path, required=True)
    parser.add_argument("--probe-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run_sweep(args.b0a_root, args.probe_receipt, args.output_directory)


if __name__ == "__main__":
    main()
