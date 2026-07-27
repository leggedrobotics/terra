#!/usr/bin/env python3
"""Confirm the B0a direct-service cost probe on one complete scenario."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import jax

import terra.benchmark_direct_service as direct_service
import tools.profile_b0a_direct_service_cost as probe_tool

SCHEMA = "terra_direct_service_validation_cost_confirmation_v1"
EXPECTED_PROBE_SHA256 = (
    "d283351640e5eaf2cc2ac5734766175a9863dc749bd4067a6c88c94f3d72e110"
)
OUTPUT_NAME = "validation_cost_confirmation.json"
FACTOR_TWO_LOWER = 0.5
FACTOR_TWO_UPPER = 2.0
SECONDS_PER_HOUR = 60 * 60
LIMIT_256_SECONDS = 24 * SECONDS_PER_HOUR
LIMIT_448_SECONDS = 48 * SECONDS_PER_HOUR
MAX_MEMORY_FRACTION = 0.8
MAX_PROBE_SCENARIO_P95_SECONDS = 60 * 60

_STABLE_INITIAL_STATE_FIELDS = (
    "schema",
    "release_id",
    "split",
    "source_group_id",
    "state_index",
    "seed_uint32",
    "seed_byte_order",
    "seed_digest_sha256",
    "initial_agent_state_sha256",
    "initial_agent_state",
    "group_variant_count",
)


def _load_pinned_probe(path: Path) -> dict[str, Any]:
    actual_sha256 = probe_tool._sha256_file(path)
    if actual_sha256 != EXPECTED_PROBE_SHA256:
        raise RuntimeError(
            "Cost-probe receipt SHA-256 changed: "
            f"{EXPECTED_PROBE_SHA256} != {actual_sha256}."
        )
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != probe_tool.SCHEMA:
        raise RuntimeError(f"Unexpected cost-probe schema: {receipt.get('schema')!r}.")
    if receipt.get("admission_result_emitted") is not False:
        raise RuntimeError("The pinned probe must be non-admission cost evidence.")
    if receipt.get("selected_identity", {}).get("map_id") != probe_tool.SELECTED_MAP_ID:
        raise RuntimeError("The pinned probe selected a different map.")
    projections = receipt.get("measurement", {}).get("projections", {})
    if projections.get("requires_one_complete_scenario_confirmation") is not True:
        raise RuntimeError("The pinned probe does not authorize confirmation.")
    for scenario_count in ("256", "448"):
        p95 = (
            projections.get("scenario_counts", {})
            .get(scenario_count, {})
            .get("seconds_p95")
        )
        if not isinstance(p95, (int, float)) or not math.isfinite(p95) or p95 <= 0:
            raise RuntimeError(
                f"The pinned probe has no positive {scenario_count}-scenario p95."
            )
    p50 = projections.get("first_scenario_cold_seconds_p50")
    if not isinstance(p50, (int, float)) or not math.isfinite(p50) or p50 <= 0:
        raise RuntimeError("The pinned probe has no positive one-scenario p50.")
    p95 = projections.get("first_scenario_cold_seconds_p95")
    if (
        not isinstance(p95, (int, float))
        or not math.isfinite(p95)
        or p95 <= 0
        or p95 > MAX_PROBE_SCENARIO_P95_SECONDS
    ):
        raise RuntimeError("The pinned probe fails the 60-minute p95 gate.")

    probe_machine = receipt.get("machine", {})
    if probe_machine.get("hostname") != socket.gethostname():
        raise RuntimeError(
            "The pinned probe lacks a portable capacity receipt and may only "
            "be confirmed on its original host."
        )
    probe_platform = probe_machine.get("jax", {}).get("platform")
    if probe_platform != "cpu":
        raise RuntimeError(
            "The pinned probe has no normalized accelerator-memory capacity; "
            "confirmation fails closed."
        )
    probe_peak_rss = (
        receipt.get("measurement", {})
        .get("memory", {})
        .get("process_max_rss_kib_after")
    )
    probe_memory = _memory_headroom(
        probe_peak_rss,
        _host_memory_capacity_kib(),
        probe_platform,
    )
    if not probe_memory["all_memory_gates_pass"]:
        raise RuntimeError("The pinned probe fails the 20% memory-headroom gate.")
    return receipt


def _selected_identity(
    selected_record: dict[str, Any],
    group: list[probe_tool.ScenarioInput],
    source_group_id: str,
) -> dict[str, Any]:
    return {
        "map_id": selected_record["map_id"],
        "legacy_dataset_split": selected_record["split"],
        "benchmark_split": probe_tool.BENCHMARK_SPLIT,
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
    }


def _stable_initial_state(receipt: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in _STABLE_INITIAL_STATE_FIELDS if field not in receipt]
    if missing:
        raise RuntimeError(f"Initial-state receipt is missing fields: {missing}.")
    return {field: receipt[field] for field in _STABLE_INITIAL_STATE_FIELDS}


def _validate_rebuilt_contract(
    *,
    probe: dict[str, Any],
    selected_identity: dict[str, Any],
    files_sha256_manifest_sha256: str,
    source_grouping: dict[str, Any],
    verified_selected_files: dict[str, str],
    protocol: dict[str, Any],
    initial_state: dict[str, Any],
    dependency_file_sha256: dict[str, str],
) -> None:
    comparisons = {
        "selected identity": (
            selected_identity,
            probe["selected_identity"],
        ),
        "files.sha256 manifest": (
            files_sha256_manifest_sha256,
            probe["input"]["files_sha256_manifest_sha256"],
        ),
        "source grouping": (
            source_grouping,
            probe["input"]["source_grouping"],
        ),
        "selected input files": (
            verified_selected_files,
            probe["input"]["verified_selected_files"],
        ),
        "protocol": (
            protocol,
            probe["protocol"],
        ),
        "explicit initial state": (
            _stable_initial_state(initial_state),
            _stable_initial_state(probe["initial_state"]),
        ),
        "direct-service dependency files": (
            dependency_file_sha256,
            probe["validator"]["code_file_sha256"],
        ),
    }
    mismatches = [
        name for name, (current, expected) in comparisons.items() if current != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Confirmation no longer matches the pinned probe: "
            + ", ".join(mismatches)
            + "."
        )


def _calibrate_cost(
    observed_wall_seconds: float,
    projections: dict[str, Any],
) -> dict[str, Any]:
    projected_p50 = float(projections["first_scenario_cold_seconds_p50"])
    if (
        not math.isfinite(observed_wall_seconds)
        or observed_wall_seconds <= 0
        or not math.isfinite(projected_p50)
        or projected_p50 <= 0
    ):
        raise ValueError("Observed and projected one-scenario costs must be positive.")

    ratio = observed_wall_seconds / projected_p50
    calibrated = {}
    limits = {"256": LIMIT_256_SECONDS, "448": LIMIT_448_SECONDS}
    for scenario_count, limit_seconds in limits.items():
        probe_p95 = float(projections["scenario_counts"][scenario_count]["seconds_p95"])
        if not math.isfinite(probe_p95) or probe_p95 <= 0:
            raise ValueError(f"Probe {scenario_count}-scenario p95 must be positive.")
        calibrated_p95 = ratio * probe_p95
        calibrated[scenario_count] = {
            "probe_p95_seconds": probe_p95,
            "calibrated_p95_seconds": calibrated_p95,
            "limit_seconds": limit_seconds,
            "passes_limit": calibrated_p95 <= limit_seconds,
        }

    gates = {
        "observed_within_factor_two_of_probe_p50": (
            FACTOR_TWO_LOWER <= ratio <= FACTOR_TWO_UPPER
        ),
        "calibrated_256_p95_at_most_24h": calibrated["256"]["passes_limit"],
        "calibrated_448_p95_at_most_48h": calibrated["448"]["passes_limit"],
    }
    return {
        "method": (
            "multiply each probe bank p95 projection by the observed complete-"
            "scenario wall time divided by the probe one-scenario p50 projection"
        ),
        "observed_wall_seconds": observed_wall_seconds,
        "probe_first_scenario_p50_seconds": projected_p50,
        "observed_to_probe_p50_ratio": ratio,
        "factor_two_interval": [FACTOR_TWO_LOWER, FACTOR_TWO_UPPER],
        "scenario_counts": calibrated,
        "gates": {
            **gates,
            "all_runtime_gates_pass": all(gates.values()),
        },
    }


def _host_memory_capacity_kib() -> int:
    try:
        capacity = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 1024
    except (OSError, ValueError) as error:
        raise RuntimeError("Host physical-memory capacity is unavailable.") from error
    if capacity <= 0:
        raise RuntimeError("Host physical-memory capacity must be positive.")
    return int(capacity)


def _memory_headroom(
    peak_rss_kib: Any,
    host_capacity_kib: int,
    device_platform: str,
) -> dict[str, Any]:
    if (
        isinstance(peak_rss_kib, bool)
        or not isinstance(peak_rss_kib, (int, float))
        or not math.isfinite(peak_rss_kib)
        or peak_rss_kib < 0
    ):
        raise RuntimeError("Process peak RSS must be a finite nonnegative number.")
    if host_capacity_kib <= 0:
        raise RuntimeError("Host memory capacity must be positive.")
    if device_platform != "cpu":
        raise RuntimeError(
            "Accelerator memory has no normalized capacity receipt; fail closed."
        )
    peak = float(peak_rss_kib)
    ratio = peak / host_capacity_kib
    passes = peak <= MAX_MEMORY_FRACTION * host_capacity_kib
    return {
        "process_peak_rss_kib": peak_rss_kib,
        "host_physical_memory_kib": host_capacity_kib,
        "process_peak_fraction_of_host": ratio,
        "maximum_allowed_fraction": MAX_MEMORY_FRACTION,
        "host_memory_headroom_passes": passes,
        "device_platform": device_platform,
        "device_memory_gate": "shared_host_memory",
        "device_memory_headroom_passes": passes,
        "all_memory_gates_pass": passes,
    }


def _code_receipt(repository: Path) -> tuple[dict[str, Any], dict[str, str]]:
    dependency_receipt = probe_tool._code_receipt(repository)
    dependency_hashes = dependency_receipt["code_file_sha256"]
    code_hashes = {
        **dependency_hashes,
        str(Path(__file__).resolve().relative_to(repository)): probe_tool._sha256_file(
            Path(__file__).resolve()
        ),
    }
    return (
        {
            "git": dependency_receipt["git"],
            "code_file_sha256": code_hashes,
            "code_bundle_sha256": probe_tool._canonical_json_sha256(code_hashes),
        },
        dependency_hashes,
    )


def _machine_receipt() -> dict[str, Any]:
    device = jax.devices()[0]
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "jax": {
            "selected_device": str(device),
            "platform": device.platform,
            "device_kind": getattr(device, "device_kind", None),
        },
    }


def _require_current_cpu_device() -> Any:
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "cpu":
        raise RuntimeError(
            "Confirmation must use the pinned single CPU device; accelerator "
            "memory lacks a normalized capacity receipt."
        )
    return devices[0]


def _compute_exact_once(state: Any) -> tuple[dict[str, Any], float]:
    _require_current_cpu_device()
    started = time.perf_counter()
    exact_outcome = direct_service.compute_initial_direct_service(state)
    probe_tool._synchronize(exact_outcome)
    return exact_outcome, time.perf_counter() - started


def run_confirmation(
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

    probe = _load_pinned_probe(probe_path)
    device = _require_current_cpu_device()
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
    current_selected_identity = _selected_identity(
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
    _validate_rebuilt_contract(
        probe=probe,
        selected_identity=current_selected_identity,
        files_sha256_manifest_sha256=manifest_sha256,
        source_grouping=source_grouping,
        verified_selected_files=verified_selected_files,
        protocol=protocol,
        initial_state=initial_state,
        dependency_file_sha256=dependency_hashes,
    )

    rss_before = probe_tool._max_rss_kib()
    exact_outcome, observed_wall_seconds = _compute_exact_once(state)
    rss_after = probe_tool._max_rss_kib()

    if set(exact_outcome) != probe_tool.DIRECT_SERVICE_OUTCOME_KEYS:
        raise RuntimeError("The exact direct-service outcome schema changed.")
    code_after, dependency_hashes_after = _code_receipt(repository)
    if code_after != code_before or dependency_hashes_after != dependency_hashes:
        raise RuntimeError(
            "Repository, validator, or confirmation code changed during execution."
        )

    calibration = _calibrate_cost(
        observed_wall_seconds,
        probe["measurement"]["projections"],
    )
    memory_headroom = _memory_headroom(
        rss_after,
        _host_memory_capacity_kib(),
        device.platform,
    )
    calibration["gates"]["all_operational_gates_pass"] = (
        calibration["gates"]["all_runtime_gates_pass"]
        and memory_headroom["all_memory_gates_pass"]
    )
    receipt = {
        "schema": SCHEMA,
        "release_id": probe_tool.RELEASE_ID,
        "result_scope": "one_complete_exact_scenario_cost_confirmation",
        "bank_admission_result_emitted": False,
        "single_scenario_exact_outcome_emitted": True,
        "selected_identity": current_selected_identity,
        "input": {
            "b0a_root": str(b0a_root),
            "probe_receipt": str(probe_path),
            "probe_receipt_sha256": EXPECTED_PROBE_SHA256,
            "files_sha256_manifest_sha256": manifest_sha256,
            "source_grouping": source_grouping,
            "verified_selected_files": verified_selected_files,
        },
        "protocol": protocol,
        "initial_state": initial_state,
        "validator": {
            "code_before": code_before,
            "code_after": code_after,
            "pre_and_post_execution_receipts_identical": True,
            "probe_dependency_file_hashes_unchanged": True,
            "exact_entrypoint": (
                "terra.benchmark_direct_service.compute_initial_direct_service"
            ),
            "exact_entrypoint_call_count": 1,
        },
        "machine": _machine_receipt(),
        "measurement": {
            "wall_seconds": observed_wall_seconds,
            "process_max_rss_kib_before": rss_before,
            "process_max_rss_kib_after": rss_after,
            "process_max_rss_kib_increase": max(0, rss_after - rss_before),
            "synchronization": (
                "compute_initial_direct_service synchronizes each JAX batch with "
                "device_get; the returned tree is synchronized before timing stops"
            ),
            "exact_outcome": probe_tool._jsonable(exact_outcome),
        },
        "memory_headroom": memory_headroom,
        "calibration": calibration,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one complete exact direct-service confirmation for the pinned "
            f"{probe_tool.SELECTED_MAP_ID} cost probe."
        )
    )
    parser.add_argument("--b0a-root", type=Path, required=True)
    parser.add_argument("--probe-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run_confirmation(args.b0a_root, args.probe_receipt, args.output_directory)


if __name__ == "__main__":
    main()
